"""
Model Training Module for Wine Quality MLOps Project.
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

from src.utils.model_registry import (
    register_model,
    wait_until_model_ready,
    set_model_alias,
)

from src.config import (
    PROJECT_ROOT,
    TRAIN_DATA,
    MODEL_FILE,
    MODELS_DIR,
    TARGET_COLUMN,
    RANDOM_STATE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

PARAMS_FILE = PROJECT_ROOT / "params.yaml"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.json"


def load_params() -> dict:
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(f"params.yaml not found: {PARAMS_FILE}")

    with open(PARAMS_FILE, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_training_data() -> pd.DataFrame:
    if not TRAIN_DATA.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_DATA}. Run preprocessing first."
        )

    df = pd.read_csv(TRAIN_DATA)

    if df.empty:
        raise ValueError("Training dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in training data.")

    return df


def build_model(params: dict) -> RandomForestRegressor:
    rf_params = params.get("model", {}).get("random_forest", {})

    model = RandomForestRegressor(
        n_estimators=rf_params.get("n_estimators", 200),
        max_depth=rf_params.get("max_depth", 15),
        min_samples_split=rf_params.get("min_samples_split", 2),
        min_samples_leaf=rf_params.get("min_samples_leaf", 1),
        random_state=rf_params.get("random_state", RANDOM_STATE),
        n_jobs=rf_params.get("n_jobs", -1),
    )

    return model


def save_training_artifacts(model, feature_names, params: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    with open(FEATURE_NAMES_FILE, "w", encoding="utf-8") as file:
        json.dump(feature_names, file, indent=4)

    metadata = {
        "model_type": model.__class__.__name__,
        "model_file": str(MODEL_FILE),
        "feature_names_file": str(FEATURE_NAMES_FILE),
        "target_column": TARGET_COLUMN,
        "feature_count": len(feature_names),
        "features": feature_names,
        "trained_at": datetime.utcnow().isoformat(),
        "params": params.get("model", {}).get("random_forest", {}),
    }

    with open(MODEL_METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    logger.info("Model saved to %s", MODEL_FILE)
    logger.info("Model metadata saved to %s", MODEL_METADATA_FILE)
    logger.info("Feature names saved to %s", FEATURE_NAMES_FILE)


def train_model():
    logger.info("=" * 80)
    logger.info("MODEL TRAINING STARTED")
    logger.info("=" * 80)

    params = load_params()
    df = load_training_data()

    X_train = df.drop(columns=[TARGET_COLUMN])
    y_train = df[TARGET_COLUMN]

    feature_names = list(X_train.columns)

    logger.info("Training data shape: %s", df.shape)
    logger.info("Feature count: %d", len(feature_names))
    logger.info("Target column: %s", TARGET_COLUMN)

    mlflow_params = params.get("training", {})
    experiment_name = mlflow_params.get("experiment_name", "wine-quality")

    mlflow_tracking_uri = params.get("mlflow", {}).get("tracking_uri", "mlruns")
    mlflow.set_tracking_uri(str(PROJECT_ROOT / mlflow_tracking_uri))
    mlflow.set_experiment(experiment_name)

    model = build_model(params)

    with mlflow.start_run(run_name="random_forest_training"):
        rf_params = params.get("model", {}).get("random_forest", {})

        mlflow.log_params(rf_params)
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("training_rows", X_train.shape[0])
        mlflow.log_param("feature_count", X_train.shape[1])

        logger.info("Training RandomForestRegressor")
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)

        logger.info("Training R2 score: %.6f", train_score)

        mlflow.log_metric("train_r2_score", train_score)

        save_training_artifacts(
            model=model,
            feature_names=feature_names,
            params=params,
        )


        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )
        
        run_id = mlflow.active_run().info.run_id
        
        model_name = params.get(
            "training",
            {},
        ).get(
            "registered_model_name",
            "wine-quality-model",
        )
        
        model_version = register_model(
            run_id=run_id,
            model_name=model_name,
            artifact_path="model",
        )
        
        wait_until_model_ready(
            model_name=model_name,
            version=model_version,
        )
        
       
        set_model_alias(
            model_name=model_name,
            version=model_version,
            alias="champion",
        )
        
        mlflow.log_artifact(str(MODEL_METADATA_FILE))
        mlflow.log_artifact(str(FEATURE_NAMES_FILE))

    logger.info("=" * 80)
    logger.info("MODEL TRAINING COMPLETED")
    logger.info("=" * 80)

    return model


if __name__ == "__main__":
    train_model()
