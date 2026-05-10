"""
Logging system - writes to console AND log file simultaneously.
"""

import logging
import os
import sys
from datetime import datetime


def setup_logger(log_dir="logs"):
    """
    Setup logging to both console and file.
    Returns configured logger.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Log filename with date
    log_filename = os.path.join(
        log_dir,
        "roulette_ai_" + datetime.now().strftime("%Y-%m-%d") + ".log"
    )

    # Create logger
    logger = logging.getLogger("RouletteAI")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # File handler (detailed)
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)

    # Console handler (info and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "[%(levelname)s] %(message)s"
    )
    console_handler.setFormatter(console_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 60)
    logger.info("Log file: " + os.path.abspath(log_filename))
    logger.info("=" * 60)

    return logger


# Global logger instance
log = setup_logger()