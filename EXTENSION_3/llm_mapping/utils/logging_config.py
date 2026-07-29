"""
Logging Configuration Utility for LLM Baseline Mapping Framework.
Provides centralized and reusable logging setup across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "llm_mapping",
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configures and returns a logger instance with console and optional file handlers.

    Args:
        name: Logger name.
        log_file: Optional Path to log file.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
