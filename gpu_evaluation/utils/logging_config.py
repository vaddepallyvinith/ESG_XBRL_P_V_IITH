import logging
import sys
from pathlib import Path

def setup_logger(name: str = "gpu_evaluation", log_file: str = "logs/evaluation.log", level=logging.INFO):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        
        # Stream Handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
        # File Handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
