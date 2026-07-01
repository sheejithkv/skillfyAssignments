"""
Model Evaluation Module for Wine Quality MLOps Project.
"""

import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from src.config import (
    PROJECT_ROOT,
    TEST_DATA,
    MODEL_FILE,
    METRICS_FILE,
    METRICS_DIR,
    ARTIFACTS_DIR,
    TARGET_COLUMN,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

PARAMS_FILE = PROJECT_ROOT / "params.yaml"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
PREDICTIONS_FILE = REPORTS_DIR / "predictions.csv"


def load_params() -> dict:
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(f"params.yaml not found: {PARAMS_FILE}")

    with open(PARAMS_FILE, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_test_data() -> pd.DataFrame:
    if not TEST_DATA.exists():
        raise FileNotFoundError(
            f"Test data not found: {TEST_DATA}. Run preprocessing first."
        )

    df = pd.read_csv(TEST_DATA)

    if df.empty:
        raise ValueError("Test dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in test data.")

    return df


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}. Run training first."
        )

    model = joblib.load(MODEL_FILE)

    logger.info("Model loaded from %s", MODEL_FILE)

    return model


def calculate_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2_score": float(r2_score(y_true, y_pred)),
    }

    return metrics


def save_metrics(metrics: dict) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    logger.info("Metrics saved to %s", METRICS_FILE)


def save_predictions(test_df: pd.DataFrame, y_pred) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = test_df.copy()
    predictions_df["prediction"] = y_pred

    predictions_df.to_csv(PREDICTIONS_FILE, index=False)

    logger.info("Predictions saved to %s", PREDICTIONS_FILE)


def log_metrics(metrics: dict) -> None:
    logger.info("=" * 80)
    logger.info("MODEL EVALUATION METRICS")
    logger.info("=" * 80)

    for key, value in metrics.items():
        logger.info("%s: %.6f", key, value)

    logger.info("=" * 80)


def evaluate_model() -> dict:
    logger.info("=" * 80)
    logger.info("MODEL EVALUATION STARTED")
    logger.info("=" * 80)

    params = load_params()
    df = load_test_data()
    model = load_model()

    X_test = df.drop(columns=[TARGET_COLUMN])
    y_test = df[TARGET_COLUMN]

    logger.info("Test data shape: %s", df.shape)
    logger.info("Feature count: %d", X_test.shape[1])

    y_pred = model.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred)

    save_metrics(metrics)
    save_predictions(df, y_pred)
    log_metrics(metrics)

    mlflow_tracking_uri = params.get("mlflow", {}).get("tracking_uri", "mlruns")
    experiment_name = params.get("training", {}).get("experiment_name", "wine-quality")

    mlflow.set_tracking_uri(str(PROJECT_ROOT / mlflow_tracking_uri))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="random_forest_evaluation"):
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.log_artifact(str(METRICS_FILE))
        mlflow.log_artifact(str(PREDICTIONS_FILE))

    logger.info("=" * 80)
    logger.info("MODEL EVALUATION COMPLETED")
    logger.info("=" * 80)

    return metrics


if __name__ == "__main__":
    evaluate_model()
