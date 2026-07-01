"""
Optuna Hyperparameter Tuning Module for Wine Quality MLOps Project.

Responsibilities
----------------
1. Load processed training data.
2. Run Optuna trials for RandomForestRegressor.
3. Optimize validation R2 score.
4. Save best parameters.
5. Save best model.
6. Save Optuna study report.
7. Log trial results and best model metadata to MLflow.

Author: Sheejith
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

from src.utils.model_registry import (
    register_model,
    wait_until_model_ready,
    set_model_alias,
)

from src.config import (
    PROJECT_ROOT,
    TRAIN_DATA,
    TARGET_COLUMN,
    RANDOM_STATE,
    MODEL_FILE,
    MODELS_DIR,
    ARTIFACTS_DIR,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

PARAMS_FILE = PROJECT_ROOT / "params.yaml"

OPTUNA_DIR = ARTIFACTS_DIR / "optuna"
OPTUNA_REPORT_FILE = OPTUNA_DIR / "best_trial.json"
OPTUNA_TRIALS_FILE = OPTUNA_DIR / "trials.csv"
OPTUNA_BEST_PARAMS_FILE = OPTUNA_DIR / "best_params.json"

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
        raise ValueError(f"Target column missing: {TARGET_COLUMN}")

    return df


def calculate_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2_score": float(r2_score(y_true, y_pred)),
    }


def build_model(trial: optuna.Trial) -> RandomForestRegressor:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", None],
        ),
        "bootstrap": trial.suggest_categorical(
            "bootstrap",
            [True, False],
        ),
    }

    return RandomForestRegressor(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def save_best_artifacts(
    model,
    feature_names,
    best_params: dict,
    best_metrics: dict,
    study: optuna.Study,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

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
        "source": "optuna_tuner",
        "best_params": best_params,
        "best_metrics": best_metrics,
    }

    with open(MODEL_METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    best_trial_report = {
        "best_trial_number": study.best_trial.number,
        "best_value": float(study.best_value),
        "best_params": best_params,
        "best_metrics": best_metrics,
        "completed_trials": len(study.trials),
        "generated_at": datetime.utcnow().isoformat(),
    }

    with open(OPTUNA_REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(best_trial_report, file, indent=4)

    with open(OPTUNA_BEST_PARAMS_FILE, "w", encoding="utf-8") as file:
        json.dump(best_params, file, indent=4)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(OPTUNA_TRIALS_FILE, index=False)

    logger.info("Best tuned model saved to %s", MODEL_FILE)
    logger.info("Best trial report saved to %s", OPTUNA_REPORT_FILE)
    logger.info("All trials saved to %s", OPTUNA_TRIALS_FILE)


def tune_model():
    logger.info("=" * 80)
    logger.info("OPTUNA HYPERPARAMETER TUNING STARTED")
    logger.info("=" * 80)

    params = load_params()
    df = load_training_data()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    feature_names = list(X.columns)

    validation_size = params.get("optuna", {}).get("validation_size", 0.2)
    n_trials = params.get("optuna", {}).get("n_trials", 20)
    direction = params.get("optuna", {}).get("direction", "maximize")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    mlflow_tracking_uri = params.get("mlflow", {}).get("tracking_uri", "mlruns")
    experiment_name = params.get("training", {}).get("experiment_name", "wine-quality")

    mlflow.set_tracking_uri(str(PROJECT_ROOT / mlflow_tracking_uri))
    mlflow.set_experiment(experiment_name)

    def objective(trial: optuna.Trial) -> float:
        model = build_model(trial)

        model.fit(X_train, y_train)

        predictions = model.predict(X_valid)

        metrics = calculate_metrics(y_valid, predictions)

        trial.set_user_attr("mae", metrics["mae"])
        trial.set_user_attr("mse", metrics["mse"])
        trial.set_user_attr("rmse", metrics["rmse"])
        trial.set_user_attr("r2_score", metrics["r2_score"])

        with mlflow.start_run(
            run_name=f"optuna_trial_{trial.number}",
            nested=True,
        ):
            mlflow.log_params(trial.params)
            mlflow.log_metrics(metrics)

        logger.info(
            "Trial %d completed | R2=%.6f | RMSE=%.6f",
            trial.number,
            metrics["r2_score"],
            metrics["rmse"],
        )

        return metrics["r2_score"]

    with mlflow.start_run(run_name="optuna_hyperparameter_tuning"):
        study = optuna.create_study(direction=direction)

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
        )

        logger.info("Best trial number: %s", study.best_trial.number)
        logger.info("Best validation R2: %.6f", study.best_value)
        logger.info("Best params: %s", study.best_params)

        best_model = RandomForestRegressor(
            **study.best_params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        best_model.fit(X, y)

        final_predictions = best_model.predict(X_valid)
        best_metrics = calculate_metrics(y_valid, final_predictions)

        save_best_artifacts(
            model=best_model,
            feature_names=feature_names,
            best_params=study.best_params,
            best_metrics=best_metrics,
            study=study,
        )

        mlflow.log_params(study.best_params)
        mlflow.log_metrics(
            {
                f"best_{key}": value
                for key, value in best_metrics.items()
            }
        )

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_model",
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
            artifact_path="best_model",
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
        
        mlflow.log_artifact(str(OPTUNA_REPORT_FILE))
        mlflow.log_artifact(str(OPTUNA_TRIALS_FILE))
        mlflow.log_artifact(str(OPTUNA_BEST_PARAMS_FILE))
        mlflow.log_artifact(str(MODEL_METADATA_FILE))
        mlflow.log_artifact(str(FEATURE_NAMES_FILE))

    logger.info("=" * 80)
    logger.info("OPTUNA HYPERPARAMETER TUNING COMPLETED")
    logger.info("=" * 80)

    return study.best_params


if __name__ == "__main__":
    tune_model()
