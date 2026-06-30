import logging
import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import OUTPUT_DIR, RESUME_STATE_PATH

# Configure logging
def setup_logging() -> logging.Logger:
    log_file = OUTPUT_DIR / "pipeline.log"
    logger = logging.getLogger("esg_xbrl_pipeline")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()

def scan_xbrl_files(directories: List[Path]) -> List[Path]:
    """Scan directories recursively for .xml, .XML, .xbrl, .XBRL files."""
    extensions = ["*.xml", "*.XML", "*.xbrl", "*.XBRL"]
    found_files = []
    
    for directory in directories:
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            continue
        
        dir_files = []
        for ext in extensions:
            dir_files.extend(list(directory.rglob(ext)))
            
        # Deduplicate files (by resolving absolute path)
        seen = set()
        for f in dir_files:
            abs_p = f.resolve()
            if abs_p not in seen:
                seen.add(abs_p)
                found_files.append(f)
                
    logger.info(f"Discovered {len(found_files)} XBRL files across {len(directories)} directories.")
    return found_files

def get_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of a file for resume logic."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in chunks of 64kb
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

class ResumeState:
    """Manages the state of successfully processed files to allow resumption."""
    def __init__(self, state_path: Path = RESUME_STATE_PATH):
        self.state_path = state_path
        self.state = self.load_state()

    def load_state(self) -> Dict[str, Dict[str, Any]]:
        if not self.state_path.exists():
            return {}
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading resume state: {e}. Starting fresh.")
            return {}

    def save_state(self):
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving resume state: {e}")

    def is_processed(self, file_path: Path) -> bool:
        file_key = str(file_path.resolve())
        if file_key not in self.state:
            return False
        
        # Verify hash and file existence
        current_hash = get_file_hash(file_path)
        saved_hash = self.state[file_key].get("hash")
        return current_hash == saved_hash

    def mark_processed(self, file_path: Path, metadata: Dict[str, Any] = None):
        file_key = str(file_path.resolve())
        self.state[file_key] = {
            "hash": get_file_hash(file_path),
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata or {}
        }
        self.save_state()

    def clear(self):
        self.state = {}
        if self.state_path.exists():
            try:
                self.state_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting state file: {e}")

def run_concurrently(task_func: Callable[[Path], Any], items: List[Path], max_workers: int = 4) -> List[Any]:
    """Execute task_func on items concurrently using ThreadPoolExecutor."""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(task_func, item): item for item in items}
        
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                data = future.result()
                if data is not None:
                    results.append(data)
            except Exception as e:
                logger.error(f"Exception raised while processing {item.name}: {e}", exc_info=True)
                
    return results

def time_it(func):
    """Decorator to measure execution time of functions."""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"Function '{func.__name__}' completed in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper
