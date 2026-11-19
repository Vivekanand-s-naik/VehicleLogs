"""
Centralized Logging Configuration
=================================
Configures console and rotating file logging.
Provides utilities to retrieve recent log entries for the API and dashboard.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List
from app.core.config import settings


def setup_logging() -> logging.Logger:
    """Initialize application logging with rotating file handler and console output."""
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger = logging.getLogger("anpr")
    logger.setLevel(log_level)

    # Avoid duplicate handlers on re-initialization
    if logger.handlers:
        return logger

    # Log format
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Rotating File Handler
    file_handler = RotatingFileHandler(
        filename=str(settings.LOG_FILE_PATH),
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Level={settings.LOG_LEVEL}, LogFile={settings.LOG_FILE_PATH}")
    return logger


def get_recent_logs(max_lines: int = 100) -> List[str]:
    """Read recent log lines from the active log file safely."""
    log_file = settings.LOG_FILE_PATH
    if not log_file.exists():
        return ["Log file does not exist yet."]

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            return [line.strip() for line in lines[-max_lines:]]
    except Exception as e:
        return [f"Error reading log file: {e}"]
