# working_benchmark_mork.py
"""
Comprehensive MORK benchmark + storage analysis.

- Benchmarks query performance on the single namespace
- Analyzes storage: local .metta sizes vs MORK stored bytes & facts
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


def estimate_facts_in_file(file_path: Path) -> int:
    """Estimate number of facts in a .metta file (non-empty, non-comment lines)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        valid = [ln for ln in (l.strip() for l in lines) if ln and not ln.startswith(';')]
        return len(valid)
    except Exception:
        return 0


def discover_local_files(base_data_dir: str = "data"):
    """
    Returns:
      all_files: list[Path] of .metta files found
    """
    base = Path(base_data_dir)
    all_files = []
    if not base.exists():
        logger.info(f"Local data directory {base} does not exist.")
        return all_files

    for f in base.rglob("*.metta"):
        if f.is_file():
            all_files.append(f)

    return all_files


class MORKBenchmark:
    def __init__(self, host="localhost", port=8027, base_data_dir="data"):
        self.host = host
        self.port = port
        self.server = None
        self.base_data_dir = Path(base_data_dir)
        self.benchmark_dir = Path("benchmarks")
        self.benchmark_dir.mkdir(exist_ok=True)
        self.target_namespace = "bioatomspace"  
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

    def get_loaded_data_stats(self):
        """Return fact count for the bioatomspace namespace."""
        if not self.server:
            logger.warning("Server not connected — skipping MORK data check")
            return 0

        try:
            with self.server.work_at(self.target_namespace) as scope:
                # quick check
                check = scope.download_(max_results=1)
                check.block()
                if check.data and check.data.strip():
                    # fetch full
                    full = scope.download_()
                    full.block()
                    data_text = full.data or ""
                    fact_count = data_text.count('\n') if data_text else 0
                    logger.debug(f"Namespace {self.target_namespace}: {fact_count} facts")
                    return fact_count
                else:
                    logger.debug(f"Namespace {self.target_namespace} appears empty in MORK")
                    return 0
        except Exception as e:
            logger.debug(f"Error accessing namespace {self.target_namespace}: {e}")
            return 0

    def measure_query_performance(self, fact_count: int, iterations: int = 5):
        """Run several queries against the bioatomspace namespace and return timings/results."""
        results = {}
        # Download all
        dl_results = []
        for _ in range(iterations):
            start = time.time()
            try:
                with self.server.work_at(self.target_namespace) as scope:
                    r = scope.download_()
                    r.block()
                    duration_ms = (time.time() - start) * 1000
                    returned = r.data.count('\n') if r.data else 0
                    dl_results.append({"duration_ms": duration_ms, "facts_returned": returned, "success": True})
            except Exception as e:
                dl_results.append({"duration_ms": (time.time() - start) * 1000, "facts_returned": 0, "success": False, "error": str(e)})
        results["download_all"] = dl_results

        # Pattern queries 
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
                    with self.server.work_at(self.target_namespace) as scope:
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
        """Run benchmark on the bioatomspace namespace and save a JSON report."""
        perf_report = {
            "timestamp": datetime.now().isoformat(),
            "namespace": self.target_namespace,
            "total_facts": 0,
            "performance": {},
            "summary": {}
        }

        fact_count = self.get_loaded_data_stats()
        if fact_count == 0:
            logger.warning("No data found in bioatomspace namespace — performance report will be empty")
            return perf_report

        perf_report["total_facts"] = fact_count

        logger.info(f"Benchmarking {self.target_namespace} ({fact_count} facts)")
        perf = self.measure_query_performance(fact_count)
        perf_report["performance"] = perf

        # Summary statistics
        all_downloads = []
        all_pattern_times = []
        total_processed = 0
        
        for r in perf["download_all"]:
            if r.get("success"):
                all_downloads.append(r["duration_ms"])
                total_processed += r["facts_returned"]
        
        # pick pattern times (aggregate all patterns)
        for k, v in perf.items():
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
         - original file stats (total files, size)
         - mork storage stats (facts, bytes as returned)
         - compression analysis + efficiency metrics
        """
        all_files = discover_local_files(str(self.base_data_dir))

        original_stats = {
            "total_files": 0,
            "total_size_bytes": 0,
            "file_breakdown": {}  # file -> {size_bytes, estimated_facts}
        }

        # analyze local files
        for f in all_files:
            try:
                size = f.stat().st_size
            except Exception:
                size = 0
            est_f = estimate_facts_in_file(f)
            original_stats["file_breakdown"][str(f)] = {"size_bytes": size, "estimated_facts": est_f}
            original_stats["total_files"] += 1
            original_stats["total_size_bytes"] += size

        # analyze MORK storage
        mork_stats = {
            "namespace": self.target_namespace,
            "facts": 0,
            "size_bytes": 0,
            "bytes_per_fact": 0
        }

        if not self.server:
            logger.warning("Server not connected — MORK stats will be empty")
        else:
            try:
                with self.server.work_at(self.target_namespace) as scope:
                    r = scope.download_()
                    r.block()
                    data_text = r.data or ""
                    facts = data_text.count('\n') if data_text else 0
                    bytes_len = len(data_text.encode('utf-8')) if data_text else 0
                    mork_stats["facts"] = facts
                    mork_stats["size_bytes"] = bytes_len
                    mork_stats["bytes_per_fact"] = (bytes_len / facts) if facts > 0 else 0
            except Exception as e:
                logger.debug(f"Error fetching data for namespace {self.target_namespace}: {e}")

        # Compression analysis
        comp = {}
        if original_stats["total_size_bytes"] > 0 and mork_stats["size_bytes"] > 0:
            compression_ratio = mork_stats["size_bytes"] / original_stats["total_size_bytes"]
            space_saved = original_stats["total_size_bytes"] - mork_stats["size_bytes"]
            space_saved_percent = (space_saved / original_stats["total_size_bytes"]) * 100
            comp = {
                "original_size_mb": original_stats["total_size_bytes"] / 1024 / 1024,
                "mork_size_mb": mork_stats["size_bytes"] / 1024 / 1024,
                "compression_ratio": compression_ratio,
                "space_saved_bytes": space_saved,
                "space_saved_mb": space_saved / 1024 / 1024,
                "space_saved_percent": space_saved_percent,
                "compression_effective": space_saved_percent > 0
            }
        else:
            comp = {"note": "Compression analysis skipped - no original files found or empty data"}

        avg_bytes_per_fact = mork_stats["bytes_per_fact"]
        facts_per_mb = (mork_stats["facts"] / (mork_stats["size_bytes"] / 1024 / 1024)) if mork_stats["size_bytes"] > 0 else 0
        density = "high" if facts_per_mb > 10000 else "medium" if facts_per_mb > 1000 else "low"

        efficiency = {
            "avg_bytes_per_fact": avg_bytes_per_fact,
            "facts_per_mb": facts_per_mb,
            "storage_density": density,
            "total_facts_stored": mork_stats["facts"]
        }

        storage_analysis = {
            "original_file_stats": original_stats,
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
        print(f"  Namespace: {mork['namespace']}")
        print(f"  Facts: {mork['facts']:,}")
        print(f"  Storage size: {mork['size_bytes'] / 1024 / 1024:.2f} MB")

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
            writer.writerow(["Namespace", "bioatomspace"])
            # original summary
            o = storage["original_file_stats"]
            writer.writerow(["Total .metta files", o["total_files"]])
            writer.writerow(["Total local size (MB)", f"{o['total_size_bytes'] / 1024 / 1024:.2f}"])
            # mork summary
            m = storage["mork_storage_stats"]
            writer.writerow(["Total facts in MORK", m["facts"]])
            writer.writerow(["Total MORK size (MB)", f"{m['size_bytes'] / 1024 / 1024:.2f}"])
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
        print(f"Namespace: {results.get('namespace')}")
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