from pathlib import Path
import os
import time
import json
import logging
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import csv
from datetime import datetime
import re
import urllib.parse

from client import MORK

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_loading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LoadStats:
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_facts: int = 0
    total_local_size_bytes: int = 0
    total_mork_size_estimate: int = 0
    start_time: float = 0
    end_time: float = 0
    failed_files_list: List[dict] = None

    def __post_init__(self):
        if self.failed_files_list is None:
            self.failed_files_list = []

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def success_rate(self) -> float:
        return (self.successful_files / self.total_files * 100) if self.total_files > 0 else 0

class MeTTaDataLoader:
    def __init__(self, base_data_dir: str = "data", max_workers: int = 5, batch_size: int = 10):
        self.base_data_dir = Path(base_data_dir)
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.stats = LoadStats()
        self.server = None
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
    def connect_server(self, host: str = "localhost", port: int = 8027) -> MORK:
        """Connect to MORK server via HTTP"""
        try:
            # Connect to the MORK server running in Docker container
            self.server = MORK(f"http://{host}:{port}")
            
            # Test the connection
            test_cmd = self.server.download_()
            test_cmd.block()
            
            logger.info(f"- Successfully connected to MORK server at http://{host}:{port}")
            return self.server
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MORK server: {e}")
    
    def discover_metta_files(self) -> List[Path]:
        """Discover all MeTTa files recursively"""
        logger.info(f"Discovering MeTTa files in {self.base_data_dir}...")
        metta_files = list(self.base_data_dir.rglob("*.metta"))
        logger.info(f"Found {len(metta_files)} MeTTa files")
        return metta_files
    
    def create_mork_namespace(self, file_path: Path) -> str:
        """
        Create MORK namespace using underscore separators instead of slashes
        to avoid URL encoding issues with MORK import
        """
        relative_path = file_path.relative_to(self.base_data_dir)
        
        # Remove .metta extension and convert path to underscore-separated string
        namespace_parts = []
        for part in relative_path.parts:
            if part.endswith('.metta'):
                part = part[:-6]  # Remove .metta extension
            
            # Sanitize each part
            sanitized = re.sub(r'[^a-zA-Z0-9]', '_', part)
            sanitized = re.sub(r'_+', '_', sanitized).strip('_')
            
            if sanitized:  # Only add non-empty parts
                namespace_parts.append(sanitized.lower())
        
        # If namespace would be empty, use a default
        if not namespace_parts:
            return "data"
        
        return "_".join(namespace_parts)
    
    def validate_metta_file(self, file_path: Path) -> Tuple[bool, str]:
        """Validate MeTTa file content before loading"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                if not content:
                    return False, "Empty file"
                
                lines = [line.strip() for line in content.split('\n') 
                        if line.strip() and not line.strip().startswith(';')]
                
                if not lines:
                    return False, "No valid MeTTa expressions"
                
                return True, "Valid MeTTa content"
                
        except Exception as e:
            return False, f"Read error: {e}"
    
    def analyze_file(self, file_path: Path) -> Tuple[int, int, int]:
        """Analyze file content"""
        try:
            file_size = file_path.stat().st_size
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = [
                    line.strip() for line in content.split('\n') 
                    if line.strip() and not line.strip().startswith(';')
                ]
                return len(lines), file_size, len(content.split('\n'))
        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")
            return 0, 0, 0
    
    def load_single_file(self, file_path: Path) -> Tuple[bool, int, int, str]:
        """Load a single MeTTa file into MORK"""
        namespace = self.create_mork_namespace(file_path)
        
        # Convert host path to container path
        
        # docker-compose mounts ./data to /app/data
        relative_path = file_path.relative_to(self.base_data_dir)
        container_file_path = Path("/app/data") / relative_path
        file_uri = f"file://{container_file_path}"
        
        # validate the file 
        is_valid, validation_msg = self.validate_metta_file(file_path)
        if not is_valid:
            return False, 0, 0, f"Invalid file: {validation_msg}"
        
        fact_count, file_size, _ = self.analyze_file(file_path)
        
        try:
            logger.debug(f"Attempting to load {file_path} -> {namespace}")
            logger.debug(f"Container path: {container_file_path}")
            
            with self.server.work_at(namespace) as scope:
                # Import the file using container path
                import_cmd = scope.sexpr_import_(file_uri)
                import_cmd.block()
                
                if import_cmd.response and import_cmd.response.status_code == 200:
                    logger.info(f"- Successfully loaded {file_path} -> {namespace}")
                    
                    # Verify the data was actually loaded
                    verify_cmd = scope.download_(max_results=5)
                    verify_cmd.block()
                    if verify_cmd.data and verify_cmd.data.strip():
                        logger.debug(f"Verified load: {len(verify_cmd.data.splitlines())} facts")
                        return True, fact_count, file_size, ""
                    else:
                        return False, 0, 0, "No data found after import"
                else:
                    error_msg = f"HTTP {import_cmd.response.status_code}" if import_cmd.response else "No response"
                    if import_cmd.response and import_cmd.response.text:
                        error_msg += f" - {import_cmd.response.text[:200]}"
                    return False, 0, 0, error_msg
                    
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            return False, 0, 0, error_msg
    
    def load_files_sequential(self, files: List[Path]) -> Tuple[int, int, int]:
        """Load files sequentially for better debugging"""
        successful = 0
        total_facts = 0
        total_local_size = 0
        
        for i, file_path in enumerate(files, 1):
            logger.info(f"- Loading file {i}/{len(files)}: {file_path}")
            
            success, fact_count, file_size, error_msg = self.load_single_file(file_path)
            
            if success:
                successful += 1
                total_facts += fact_count
                total_local_size += file_size
                logger.info(f"   - Success - {fact_count} facts")
            else:
                self.stats.failed_files += 1
                self.stats.failed_files_list.append({
                    "file": str(file_path),
                    "error": error_msg,
                    "namespace": self.create_mork_namespace(file_path)
                })
                logger.error(f"   ... Failed: {error_msg}")
        
        return successful, total_facts, total_local_size
    
    def organize_data_with_transformations(self):
        """Organize the loaded data using transformations to create proper structure"""
        logger.info("... Organizing data with transformations...")
        
        try:
            # try to access data  from root namespace
            with self.server.work_at("") as root:
                # Try to list available namespaces first
                try:
                    list_cmd = root.list_namespaces_()
                    list_cmd.block()
                    if list_cmd.data:
                        logger.info(f"Available namespaces: {list_cmd.data}")
                except:
                    logger.warning("Could not list namespaces")
                
                # Count 
                count_cmd = root.download_(max_results=100)  # Limit to avoid huge responses
                count_cmd.block()
                if count_cmd.data:
                    total_facts = count_cmd.data.count('\n')
                    logger.info(f"Total facts in root namespace: {total_facts}")
                    if total_facts > 0:
                        logger.info(f"Sample facts:\n{count_cmd.data[:500]}")
                else:
                    logger.warning("No data found in root namespace")
                    
        except Exception as e:
            logger.error(f"Error organizing data: {e}")
    
    def mork_storage(self) -> int:
        """Estimate storage usage in MORK using actual stored data"""
        if not self.server:
            return 0

        total_bytes = 0
        for file_info in [{"namespace": self.create_mork_namespace(Path(f))} for f in self.discover_metta_files()]:
            namespace = file_info.get("namespace")
            if namespace:
                try:
                    with self.server.work_at(namespace) as scope:
                        download_cmd = scope.download_()
                        download_cmd.block()
                        if download_cmd.data:
                            total_bytes += len(download_cmd.data.encode('utf-8'))
                except Exception as e:
                    logger.debug(f"Could not access namespace {namespace}: {e}")

        return total_bytes
    
    def get_detailed_stats(self) -> Dict:
        """Get comprehensive statistics by checking all namespaces"""
        stats = {
            "total_mork_facts": 0,
            "total_mork_size_bytes": 0,
            "sample_data": ""
        }
        
        if not self.server:
            return stats
            
        try:
            total_facts = 0
            all_data = ""
            
            # Since data is loaded into specific namespaces, check each one
            for file_info in self.stats.failed_files_list + [{"namespace": self.create_mork_namespace(Path(f))} for f in self.discover_metta_files()]:
                namespace = file_info.get("namespace")
                if namespace:
                    try:
                        with self.server.work_at(namespace) as scope:
                            download_cmd = scope.download_()
                            download_cmd.block()
                            if download_cmd.data:
                                ns_facts = download_cmd.data.count('\n')
                                total_facts += ns_facts
                                if len(all_data) < 1000:  # Keep sample manageable
                                    all_data += f"\n--- {namespace} ---\n{download_cmd.data[:200]}"
                    except Exception as e:
                        logger.debug(f"Could not access namespace {namespace}: {e}")
            
            stats["total_mork_facts"] = total_facts
            stats["total_mork_size_bytes"] = len(all_data.encode('utf-8')) if all_data else 0
            stats["sample_data"] = all_data
            
        except Exception as e:
            logger.error(f"Error in detailed stats: {e}")
            # Fallback to loaded count
            stats["total_mork_facts"] = self.stats.total_facts
        
        return stats
    
    def generate_reports(self):
        """Generate comprehensive reports"""
        # Get final stats from MORK
        mork_stats = self.get_detailed_stats()
        
        manifest = {
            "load_operation": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": self.stats.duration,
                "base_data_dir": str(self.base_data_dir),
            },
            "file_statistics": {
                "total_files": self.stats.total_files,
                "successful_files": self.stats.successful_files,
                "failed_files": self.stats.failed_files,
                "success_rate_percent": self.stats.success_rate,
                "total_facts_loaded": self.stats.total_facts,
                "total_local_size_bytes": self.stats.total_local_size_bytes,
                "total_local_size_mb": self.stats.total_local_size_bytes / 1024 / 1024,
            },
            "mork_statistics": mork_stats,
            "mork_size_bytes": mork_stats["total_mork_size_bytes"],
            "mork_size_mb": mork_stats["total_mork_size_bytes"] / 1024 / 1024,
            "compression_ratio": (
                mork_stats["total_mork_size_bytes"] / self.stats.total_local_size_bytes 
                if self.stats.total_local_size_bytes > 0 else 0
            ),
            "failed_files": self.stats.failed_files_list,
        }
        
        # Save JSON report
        manifest_path = self.reports_dir / f"load_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # CSV Report
        csv_path = self.reports_dir / f"load_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Files', self.stats.total_files])
            writer.writerow(['Successful Files', self.stats.successful_files])
            writer.writerow(['Failed Files', self.stats.failed_files])
            writer.writerow(['Success Rate', f"{self.stats.success_rate:.2f}%"])
            writer.writerow(['Total Facts Loaded', self.stats.total_facts])
            writer.writerow(['Total Facts in MORK', mork_stats['total_mork_facts']])
            writer.writerow(['Local Size (MB)', f"{self.stats.total_local_size_bytes / 1024 / 1024:.2f}"])
            writer.writerow(['MORK Size (MB)', f"{mork_stats['total_mork_size_bytes'] / 1024 / 1024:.2f}"])
            writer.writerow(['Compression Ratio', f"{mork_stats['total_mork_size_bytes'] / self.stats.total_local_size_bytes if self.stats.total_local_size_bytes > 0 else 0:.2f}"])
        
        logger.info(f"* Reports saved to: {manifest_path}, {csv_path}")
    
    def load_all_data(self) -> LoadStats:
        """Main method to load all data"""
        self.stats.start_time = time.time()
        
        try:
            # Connect to server via HTTP (Docker container)
            self.connect_server(host="localhost", port=8027)
            
            # Discover all MeTTa files
            metta_files = self.discover_metta_files()
            self.stats.total_files = len(metta_files)
            
            if self.stats.total_files == 0:
                logger.warning("... No MeTTa files found!")
                return self.stats
            
            logger.info(f"- Found {self.stats.total_files} MeTTa files to load")
            
            # Load files sequentially
            successful, total_facts, total_local_size = self.load_files_sequential(metta_files)
            self.stats.successful_files = successful
            self.stats.total_facts = total_facts
            self.stats.total_local_size_bytes = total_local_size
            
            # Organize the data
            self.organize_data_with_transformations()
            
            self.stats.end_time = time.time()
            
            # Generate reports
            self.generate_reports()
            
            # Log summary
            mork_stats = self.get_detailed_stats()
            logger.info("... Loading Summary:")
            logger.info(f"   Duration: {self.stats.duration:.2f} seconds")
            logger.info(f"   Files: {self.stats.successful_files}/{self.stats.total_files} ({self.stats.success_rate:.1f}%)")
            logger.info(f"   Facts loaded: {self.stats.total_facts:,}")
            logger.info(f"   Facts in MORK: {mork_stats['total_mork_facts']:,}")
            logger.info(f"   Local size: {self.stats.total_local_size_bytes / 1024 / 1024:.2f} MB")
            
            # Show sample data
            if mork_stats['sample_data']:
                logger.info("Sample of loaded data:")
                for line in mork_stats['sample_data'].split('\n')[:3]:
                    if line.strip():
                        logger.info(f"   {line.strip()}")
            
            return self.stats
            
        except Exception as e:
            logger.error(f"... Fatal error during loading: {e}")
            self.stats.end_time = time.time()
            raise

def main():
    """Main execution function"""
    loader = MeTTaDataLoader(
        base_data_dir="data",
        max_workers=1,  # Sequential loading for better debugging
        batch_size=1
    )
    
    try:
        stats = loader.load_all_data()
        return 0
    except Exception as e:
        logger.error(f"... Data loading failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())