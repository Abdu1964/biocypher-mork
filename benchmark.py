# working_benchmark_mork.py
"""
Comprehensive MORK benchmark + storage analysis.

- Reconstructs namespaces from local `data/*.metta` files (same sanitization as loader)
- Benchmarks query performance per-namespace
- Analyzes storage: local .metta sizes & estimated facts vs MORK stored bytes & facts
- Produces JSON and CSV reports under `benchmarks/`
"""

import time
import json
import csv
import re
from pathlib import Path
from datetime import datetime
from client import MORK
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def sanitize_part(part: str) -> str:
    p = re.sub(r'\.metta$', '', part, flags=re.IGNORECASE)
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', p)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return sanitized.lower() if sanitized else ''


def create_namespace_from_path(file_path: Path, base_data_dir: Path) -> str:
    """Build namespace from a file path using the same rules as loader."""
    rel = file_path.relative_to(base_data_dir)
    parts = []
    for part in rel.parts:
        s = sanitize_part(part)
        if s:
            parts.append(s)
    return "_".join(parts) if parts else "data"


def estimate_facts_in_file(file_path: Path) -> int:
    """Estimate number of facts in a .metta file (non-empty, non-comment lines)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        valid = [ln for ln in (l.strip() for l in lines) if ln and not ln.startswith(';')]
        return len(valid)
    except Exception:
        return 0


def discover_local_namespace_file_map(base_data_dir: str = "data"):
    """
    Returns:
      all_files: list[Path] of .metta files found
      namespace_map: dict[str, list[Path]] mapping namespace -> list of file Paths
    """
    base = Path(base_data_dir)
    all_files = []
    ns_map = {}
    if not base.exists():
        logger.info(f"Local data directory {base} does not exist.")
        return all_files, ns_map

    for f in base.rglob("*.metta"):
        if f.is_file():
            all_files.append(f)
            ns = create_namespace_from_path(f, base)
            ns_map.setdefault(ns, []).append(f)

    return all_files, ns_map


class MORKBenchmark:
    def __init__(self, host="localhost", port=8027, base_data_dir="data"):
        self.host = host
        self.port = port
        self.server = None
        self.base_data_dir = Path(base_data_dir)
        self.benchmark_dir = Path("benchmarks")
        self.benchmark_dir.mkdir(exist_ok=True)
        self.connect_server()

    def connect_server(self) -> bool:
        try:
            self.server = MORK(f"http://{self.host}:{self.port}")
            # quick smoke test
            test_cmd = self.server.download_()
            test_cmd.block()
            logger.info(f"Connected to MORK server at http://{self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MORK server: {e}")
            self.server = None
            return False

    def get_all_namespaces_from_local(self):
        _, ns_map = discover_local_namespace_file_map(str(self.base_data_dir))
        namespaces = list(ns_map.keys())
        logger.info(f"Discovered {len(namespaces)} namespaces from local data folder")
        return namespaces, ns_map

    def get_loaded_namespaces(self):
        """Return list of tuples (namespace, fact_count) for namespaces that have data in MORK."""
        namespaces, ns_map = self.get_all_namespaces_from_local()
        active = []
        if not self.server:
            logger.warning("Server not connected — skipping MORK namespace checks")
            return active

        for ns in namespaces:
            try:
                with self.server.work_at(ns) as scope:
                    # quick check
                    check = scope.download_(max_results=1)
                    check.block()
                    if check.data and check.data.strip():
                        # fetch full (careful: could be large)
                        full = scope.download_()
                        full.block()
                        data_text = full.data or ""
                        fact_count = data_text.count('\n') if data_text else 0
                        active.append((ns, fact_count))
                        logger.debug(f"Namespace {ns}: {fact_count} facts")
                    else:
                        logger.debug(f"Namespace {ns} appears empty in MORK")
            except Exception as e:
                logger.debug(f"Error accessing namespace {ns}: {e}")

        logger.info(f"Found {len(active)} namespaces with data in MORK")
        return active

    def measure_query_performance(self, namespace: str, fact_count: int, iterations: int = 5):
        """Run several queries against a namespace and return timings/results."""
        results = {}
        # Download all
        dl_results = []
        for _ in range(iterations):
            start = time.time()
            try:
                with self.server.work_at(namespace) as scope:
                    r = scope.download_()
                    r.block()
                    duration_ms = (time.time() - start) * 1000
                    returned = r.data.count('\n') if r.data else 0
                    dl_results.append({"duration_ms": duration_ms, "facts_returned": returned, "success": True})
            except Exception as e:
                dl_results.append({"duration_ms": (time.time() - start) * 1000, "facts_returned": 0, "success": False, "error": str(e)})
        results["download_all"] = dl_results

        # Pattern queries (lightweight)
        patterns = [
            ("(Person_$x)", "$x", "person_pattern"),
            ("($type $x)", "$type", "generic_pattern"),
            ("($x $y $z)", "$x", "triple_pattern"),
            ("($predicate $subject $object)", "$predicate", "rdf_pattern"),
        ]
        for pat, outv, name in patterns:
            entries = []
            for _ in range(iterations):
                start = time.time()
                try:
                    with self.server.work_at(namespace) as scope:
                        r = scope.download(pat, outv)
                        r.block()
                        duration_ms = (time.time() - start) * 1000
                        returned = r.data.count('\n') if r.data else 0
                        entries.append({"duration_ms": duration_ms, "facts_returned": returned, "success": True})
                except Exception as e:
                    entries.append({"duration_ms": (time.time() - start) * 1000, "facts_returned": 0, "success": False, "error": str(e)})
            results[name] = entries

        return results

    def run_comprehensive_benchmark(self):
        """Run benchmark on all loaded namespaces and save a JSON report."""
        perf_report = {
            "timestamp": datetime.now().isoformat(),
            "total_namespaces": 0,
            "total_facts": 0,
            "namespace_results": {},
            "summary": {}
        }

        loaded = self.get_loaded_namespaces()
        if not loaded:
            logger.warning("No namespaces with data found — performance report will be empty")
            return perf_report

        perf_report["total_namespaces"] = len(loaded)
        perf_report["total_facts"] = sum(cnt for _, cnt in loaded)

        for ns_name, fact_count in loaded:
            logger.info(f"Benchmarking {ns_name} ({fact_count} facts)")
            perf = self.measure_query_performance(ns_name, fact_count)
            perf_report["namespace_results"][ns_name] = {"fact_count": fact_count, "performance": perf}

        # Summary statistics
        all_downloads = []
        all_pattern_times = []
        total_processed = 0
        for ns_val in perf_report["namespace_results"].values():
            for r in ns_val["performance"]["download_all"]:
                if r.get("success"):
                    all_downloads.append(r["duration_ms"])
                    total_processed += r["facts_returned"]
            # pick pattern times (aggregate all patterns)
            for k, v in ns_val["performance"].items():
                if k != "download_all":
                    for r in v:
                        if r.get("success"):
                            all_pattern_times.append(r["duration_ms"])

        if all_downloads:
            perf_report["summary"] = {
                "avg_download_time_ms": sum(all_downloads) / len(all_downloads),
                "min_download_time_ms": min(all_downloads),
                "max_download_time_ms": max(all_downloads),
                "avg_pattern_time_ms": (sum(all_pattern_times) / len(all_pattern_times)) if all_pattern_times else 0,
                "total_facts_processed": total_processed,
                "throughput_facts_per_second": (total_processed / (sum(all_downloads) / 1000)) if sum(all_downloads) > 0 else 0,
                "namespaces_benchmarked": len(loaded)
            }

        # Save performance report
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        perf_path = self.benchmark_dir / f"benchmark_{ts}.json"
        with open(perf_path, 'w') as fh:
            json.dump(perf_report, fh, indent=2)
        logger.info(f"Performance benchmark saved to: {perf_path}")

        return perf_report

    def analyze_storage_efficiency(self):
        """
        Build:
         - original file stats (total files, size, per-namespace breakdown)
         - mork storage stats (per-namespace facts, bytes as returned)
         - compression analysis + efficiency metrics
        """
        all_files, ns_map = discover_local_namespace_file_map(str(self.base_data_dir))

        original_stats = {
            "total_files": 0,
            "total_size_bytes": 0,
            "file_breakdown": {}  # file -> {size_bytes, estimated_facts, namespace}
        }
        ns_orig_breakdown = {}

        # analyze local files
        for f in all_files:
            try:
                size = f.stat().st_size
            except Exception:
                size = 0
            est_f = estimate_facts_in_file(f)
            ns = create_namespace_from_path(f, self.base_data_dir)
            original_stats["file_breakdown"][str(f)] = {"size_bytes": size, "estimated_facts": est_f, "namespace": ns}
            original_stats["total_files"] += 1
            original_stats["total_size_bytes"] += size
            ns_orig_breakdown.setdefault(ns, {"size_bytes": 0, "estimated_facts": 0, "files": []})
            ns_orig_breakdown[ns]["size_bytes"] += size
            ns_orig_breakdown[ns]["estimated_facts"] += est_f
            ns_orig_breakdown[ns]["files"].append(str(f))

        # analyze MORK storage
        mork_stats = {
            "total_namespaces": 0,
            "total_facts": 0,
            "total_mork_bytes": 0,
            "namespace_breakdown": {}  # ns -> {facts, size_bytes, bytes_per_fact}
        }

        if not self.server:
            logger.warning("Server not connected — MORK stats will be empty")
        else:
            namespaces = list(ns_map.keys())
            for ns in namespaces:
                try:
                    with self.server.work_at(ns) as scope:
                        r = scope.download_()
                        r.block()
                        data_text = r.data or ""
                        facts = data_text.count('\n') if data_text else 0
                        bytes_len = len(data_text.encode('utf-8')) if data_text else 0
                        if facts > 0 or bytes_len > 0:
                            mork_stats["namespace_breakdown"][ns] = {
                                "facts": facts,
                                "size_bytes": bytes_len,
                                "bytes_per_fact": (bytes_len / facts) if facts > 0 else 0
                            }
                            mork_stats["total_facts"] += facts
                            mork_stats["total_mork_bytes"] += bytes_len
                except Exception as e:
                    logger.debug(f"Error fetching data for namespace {ns}: {e}")

        mork_stats["total_namespaces"] = len(mork_stats["namespace_breakdown"])

        # Compression analysis
        comp = {}
        if original_stats["total_size_bytes"] > 0 and mork_stats["total_mork_bytes"] > 0:
            compression_ratio = mork_stats["total_mork_bytes"] / original_stats["total_size_bytes"]
            space_saved = original_stats["total_size_bytes"] - mork_stats["total_mork_bytes"]
            space_saved_percent = (space_saved / original_stats["total_size_bytes"]) * 100
            comp = {
                "original_size_mb": original_stats["total_size_bytes"] / 1024 / 1024,
                "mork_size_mb": mork_stats["total_mork_bytes"] / 1024 / 1024,
                "compression_ratio": compression_ratio,
                "space_saved_bytes": space_saved,
                "space_saved_mb": space_saved / 1024 / 1024,
                "space_saved_percent": space_saved_percent,
                "compression_effective": space_saved_percent > 0
            }
        else:
            comp = {"note": "Compression analysis skipped - no original files found or empty data"}

        avg_bytes_per_fact = (mork_stats["total_mork_bytes"] / mork_stats["total_facts"]) if mork_stats["total_facts"] > 0 else 0
        facts_per_mb = (mork_stats["total_facts"] / (mork_stats["total_mork_bytes"] / 1024 / 1024)) if mork_stats["total_mork_bytes"] > 0 else 0
        density = "high" if facts_per_mb > 10000 else "medium" if facts_per_mb > 1000 else "low"

        efficiency = {
            "avg_bytes_per_fact": avg_bytes_per_fact,
            "facts_per_mb": facts_per_mb,
            "storage_density": density,
            "total_facts_stored": mork_stats["total_facts"]
        }

        storage_analysis = {
            "original_file_stats": original_stats,
            "original_namespace_breakdown": ns_orig_breakdown,
            "mork_storage_stats": mork_stats,
            "compression_analysis": comp,
            "efficiency_metrics": efficiency,
        }

        return storage_analysis

    def print_storage_analysis(self, storage_data):
        print("\n" + "=" * 60)
        print("MORK STORAGE EFFICIENCY ANALYSIS")
        print("=" * 60)

        orig = storage_data["original_file_stats"]
        if orig["total_files"] > 0:
            print(f"Original .metta files:")
            print(f"  Files: {orig['total_files']}")
            print(f"  Total size: {orig['total_size_bytes'] / 1024 / 1024:.2f} MB")
        else:
            print("Original .metta files: Not found or empty")

        mork = storage_data["mork_storage_stats"]
        print(f"\nMORK storage:")
        print(f"  Namespaces: {mork['total_namespaces']}")
        print(f"  Facts: {mork['total_facts']:,}")
        print(f"  Storage size: {mork['total_mork_bytes'] / 1024 / 1024:.2f} MB")

        comp = storage_data["compression_analysis"]
        if "note" in comp:
            print(f"\n{comp['note']}")
        else:
            print(f"\nCompression analysis:")
            print(f"  Compression ratio: {comp['compression_ratio']:.3f}")
            if comp['compression_effective']:
                print(f"  Space saved: {comp['space_saved_mb']:.2f} MB ({comp['space_saved_percent']:.1f}%)")
                print(f"  Compression: {'Excellent' if comp['space_saved_percent'] > 50 else 'Good' if comp['space_saved_percent'] > 20 else 'Moderate'}")
            else:
                print(f"  Space overhead: {abs(comp['space_saved_mb']):.2f} MB")

        eff = storage_data["efficiency_metrics"]
        print(f"\nStorage efficiency:")
        print(f"  Bytes per fact: {eff['avg_bytes_per_fact']:.1f}")
        print(f"  Facts per MB: {eff['facts_per_mb']:.0f}")
        print(f"  Storage density: {eff['storage_density']}")

        # Top namespaces by MORK storage size
        ns_break = mork.get("namespace_breakdown", {})
        if ns_break:
            print(f"\nLargest namespaces by storage (top 5):")
            ns_by_storage = sorted(ns_break.items(), key=lambda x: x[1]["size_bytes"], reverse=True)
            for ns_name, ns_data in ns_by_storage[:5]:
                size_kb = ns_data["size_bytes"] / 1024
                print(f"  {ns_name}: {size_kb:.1f} KB ({ns_data['facts']} facts)")

    def save_combined_reports(self, perf, storage):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined = {"performance": perf, "storage": storage, "analysis_timestamp": datetime.now().isoformat()}
        json_path = self.benchmark_dir / f"complete_analysis_{ts}.json"
        with open(json_path, 'w') as fh:
            json.dump(combined, fh, indent=2)
        logger.info(f"Combined analysis JSON saved to: {json_path}")

        # CSV summary
        csv_path = self.benchmark_dir / f"complete_analysis_{ts}.csv"
        with open(csv_path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Timestamp", combined["analysis_timestamp"]])
            # original summary
            o = storage["original_file_stats"]
            writer.writerow(["Total .metta files", o["total_files"]])
            writer.writerow(["Total local size (MB)", f"{o['total_size_bytes'] / 1024 / 1024:.2f}"])
            # mork summary
            m = storage["mork_storage_stats"]
            writer.writerow(["Total MORK namespaces", m["total_namespaces"]])
            writer.writerow(["Total facts in MORK", m["total_facts"]])
            writer.writerow(["Total MORK size (MB)", f"{m['total_mork_bytes'] / 1024 / 1024:.2f}"])
            # compression
            c = storage["compression_analysis"]
            if "note" in c:
                writer.writerow(["Compression analysis", c["note"]])
            else:
                writer.writerow(["Compression ratio", f"{c['compression_ratio']:.3f}"])
                writer.writerow(["Space saved (MB)", f"{c['space_saved_mb']:.2f}"])
                writer.writerow(["Space saved (%)", f"{c['space_saved_percent']:.2f}"])
            # efficiency
            e = storage["efficiency_metrics"]
            writer.writerow(["Avg bytes per fact", f"{e['avg_bytes_per_fact']:.1f}"])
            writer.writerow(["Facts per MB", f"{e['facts_per_mb']:.0f}"])
        logger.info(f"Combined summary CSV saved to: {csv_path}")

    def run_complete_analysis(self):
        print("Running comprehensive MORK analysis...")

        perf = self.run_comprehensive_benchmark()
        storage = self.analyze_storage_efficiency()

        # pretty-print
        self.print_results(perf)
        self.print_storage_analysis(storage)

        # save combined JSON + CSV
        self.save_combined_reports(perf, storage)

        # concise summary
        print("\n" + "=" * 60)
        print("OVERALL MORK PERFORMANCE SUMMARY")
        print("=" * 60)
        total_facts = storage["efficiency_metrics"]["total_facts_stored"]
        print(f"Total facts processed: {total_facts:,}")
        print(f"Storage efficiency: {storage['efficiency_metrics']['storage_density']}")
        if perf.get("summary"):
            s = perf["summary"]
            if s.get("throughput_facts_per_second") is not None:
                print(f"Query throughput: {s['throughput_facts_per_second']:.0f} facts/sec")
            if s.get("avg_download_time_ms") is not None:
                print(f"Average query time: {s['avg_download_time_ms']:.2f} ms")

        return {"performance": perf, "storage": storage}


    def print_results(self, results):
        print("\n" + "=" * 60)
        print("MORK PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Timestamp: {results.get('timestamp')}")
        print(f"Total Namespaces: {results.get('total_namespaces')}")
        print(f"Total Facts: {results.get('total_facts', 0):,}")

        if "summary" in results and results["summary"]:
            s = results["summary"]
            print("\nPerformance Summary:")
            if s.get("avg_download_time_ms") is not None:
                print(f"  Average download time: {s['avg_download_time_ms']:.2f} ms")
            if s.get("min_download_time_ms") is not None and s.get("max_download_time_ms") is not None:
                print(f"  Download time range: {s['min_download_time_ms']:.2f} - {s['max_download_time_ms']:.2f} ms")
            if s.get("avg_pattern_time_ms") is not None:
                print(f"  Average pattern query time: {s['avg_pattern_time_ms']:.2f} ms")
            if s.get("throughput_facts_per_second") is not None:
                print(f"  Throughput: {s['throughput_facts_per_second']:.0f} facts/second")

        if results.get("namespace_results"):
            print("\nTop 5 Largest Namespaces:")
            ns_by_size = sorted(results["namespace_results"].items(), key=lambda x: x[1]["fact_count"], reverse=True)
            for ns_name, ns_data in ns_by_size[:5]:
                fact_count = ns_data["fact_count"]
                dl_times = [r["duration_ms"] for r in ns_data["performance"]["download_all"] if r.get("success")]
                avg_time = (sum(dl_times) / len(dl_times)) if dl_times else 0
                print(f"  {ns_name}: {fact_count:,} facts, {avg_time:.2f}ms avg")


def main():
    benchmark = MORKBenchmark(host="localhost", port=8027, base_data_dir="data")
    try:
        combined = benchmark.run_complete_analysis()
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
