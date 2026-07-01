"""
Metrics utilities for model evaluation.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2_score": float(r2_score(y_true, y_pred)),
    }


def save_metrics(metrics: dict, file_path: Path) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    logger.info("Metrics saved to %s", file_path)


def load_metrics(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def log_metrics(metrics: dict) -> None:
    logger.info("=" * 80)
    logger.info("MODEL METRICS")
    logger.info("=" * 80)

    for metric_name, metric_value in metrics.items():
        logger.info("%s: %.6f", metric_name, metric_value)

    logger.info("=" * 80)
