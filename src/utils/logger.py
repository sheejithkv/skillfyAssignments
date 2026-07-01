"""
Logging utility for the MLOps Wine Quality Project.

Provides a reusable logger with console and file handlers.
"""

import logging
from pathlib import Path

from src.config import LOG_DIR, LOG_FILE

# Ensure log directory exists
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger
