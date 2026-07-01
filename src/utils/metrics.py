"""
Metrics utility module for the MLOps Wine Quality Project.

Provides:
- Regression metrics calculation
- Metrics persistence
- Metrics loading
- Metrics logging
- Confusion matrix generation (optional)
- Classification report (optional)

Author: Sheejith
"""

from pathlib import Path
from typing import Dict

import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
)

from src.utils.logger import logger
from src.utils.exceptions import MLOpsException
from src.utils.common import create_directory


# ============================================================
# Regression Metrics
# ============================================================

def calculate_regression_metrics(
    y_true,
    y_pred,
) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    dict
    """

    try:

        mse = mean_squared_error(y_true, y_pred)

        metrics = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2_score": float(r2_score(y_true, y_pred)),
        }

        logger.info("Regression metrics calculated.")

        return metrics

    except Exception as exc:
        logger.exception("Failed to calculate regression metrics.")
        raise MLOpsException(str(exc))


# ============================================================
# Classification Metrics
# ============================================================

def calculate_classification_metrics(
    y_true,
    y_pred,
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    """

    try:

        metrics = {

            "accuracy": float(
                accuracy_score(y_true, y_pred)
            ),

            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),

            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),

            "f1_score": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }

        logger.info("Classification metrics calculated.")

        return metrics

    except Exception as exc:
        logger.exception("Failed to calculate classification metrics.")
        raise MLOpsException(str(exc))


# ============================================================
# Confusion Matrix
# ============================================================

def generate_confusion_matrix(
    y_true,
    y_pred,
):
    """
    Generate confusion matrix.
    """

    try:

        cm = confusion_matrix(
            y_true,
            y_pred,
        )

        logger.info("Confusion matrix generated.")

        return cm

    except Exception as exc:
        logger.exception("Unable to generate confusion matrix.")
        raise MLOpsException(str(exc))


# ============================================================
# Classification Report
# ============================================================

def generate_classification_report(
    y_true,
    y_pred,
) -> str:
    """
    Generate sklearn classification report.
    """

    try:

        report = classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )

        logger.info("Classification report generated.")

        return report

    except Exception as exc:
        logger.exception("Unable to generate classification report.")
        raise MLOpsException(str(exc))


# ============================================================
# Save Metrics
# ============================================================

def save_metrics(
    metrics: Dict,
    file_path: Path,
) -> None:
    """
    Save metrics to JSON.
    """

    try:

        create_directory(file_path.parent)

        with open(
            file_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4,
            )

        logger.info(f"Metrics saved to {file_path}")

    except Exception as exc:
        logger.exception("Unable to save metrics.")
        raise MLOpsException(str(exc))


# ============================================================
# Load Metrics
# ============================================================

def load_metrics(
    file_path: Path,
) -> Dict:
    """
    Load metrics JSON.
    """

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as file:

            metrics = json.load(file)

        logger.info(f"Metrics loaded from {file_path}")

        return metrics

    except Exception as exc:
        logger.exception("Unable to load metrics.")
        raise MLOpsException(str(exc))


# ============================================================
# Pretty Logging
# ============================================================

def log_metrics(
    metrics: Dict,
) -> None:
    """
    Print metrics nicely in logs.
    """

    logger.info("=" * 60)
    logger.info("Model Evaluation Metrics")
    logger.info("=" * 60)

    for key, value in metrics.items():
        logger.info(f"{key:<20}: {value:.6f}")

    logger.info("=" * 60)


# ============================================================
# Compare Models
# ============================================================

def compare_models(
    model_results: Dict[str, Dict],
):
    """
    Compare multiple models using R² score.

    Example
    -------
    {
        "RandomForest": {"r2_score":0.89},
        "XGBoost":{"r2_score":0.92}
    }
    """

    try:

        best_model = max(
            model_results.items(),
            key=lambda x: x[1]["r2_score"],
        )

        logger.info(
            f"Best model: {best_model[0]} "
            f"(R²={best_model[1]['r2_score']:.4f})"
        )

        return best_model

    except Exception as exc:
        logger.exception("Unable to compare models.")
        raise MLOpsException(str(exc))
