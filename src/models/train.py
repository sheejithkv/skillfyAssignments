"""
Model training module.

Responsibilities
----------------
1. Load processed training dataset
2. Train Random Forest classifier
3. Save trained model
4. Generate feature importance plot
5. Log everything to MLflow
"""

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from mlflow.models.signature import infer_signature

from src.config import (
    TRAIN_DATA,
    TARGET_COLUMN,
    MODEL_FILE,
    MODEL_DIR,
    ARTIFACT_DIR,
    FEATURE_IMPORTANCE_FILE,
    RANDOM_STATE,
    N_ESTIMATORS,
    MAX_DEPTH,
    MIN_SAMPLES_SPLIT,
    MIN_SAMPLES_LEAF,
    N_JOBS,
    MLFLOW_MODEL_NAME,
)

from src.utils.common import create_directory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Train and save Random Forest model.
    """

    def __init__(self):

        create_directory(MODEL_DIR)
        create_directory(ARTIFACT_DIR)

    def load_training_data(self):
        """
        Load processed training dataset.
        """

        logger.info("Loading training dataset...")

        df = pd.read_csv(TRAIN_DATA)

        logger.info("Training dataset shape: %s", df.shape)

        X = df.drop(columns=[TARGET_COLUMN])

        y = df[TARGET_COLUMN]

        return X, y

    def build_model(self):
        """
        Build Random Forest model.
        """

        logger.info("Building Random Forest model...")

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )

        return model

    def train(self):

        X_train, y_train = self.load_training_data()

        model = self.build_model()

        logger.info("Training started...")

        model.fit(X_train, y_train)

        logger.info("Training completed.")

        joblib.dump(model, MODEL_FILE)

        logger.info("Model saved -> %s", MODEL_FILE)

        self.plot_feature_importance(model, X_train)

        self.log_mlflow(model, X_train)

        return model

    def plot_feature_importance(self, model, X_train):
        """
        Save feature importance graph.
        """

        logger.info("Generating feature importance plot...")

        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns,
        )

        importance = importance.sort_values()

        plt.figure(figsize=(10, 6))

        importance.plot(kind="barh")

        plt.xlabel("Importance")

        plt.ylabel("Features")

        plt.title("Random Forest Feature Importance")

        plt.tight_layout()

        plt.savefig(FEATURE_IMPORTANCE_FILE)

        plt.close()

        logger.info(
            "Feature importance saved -> %s",
            FEATURE_IMPORTANCE_FILE,
        )

    def log_mlflow(self, model, X_train):
        """
        Log model and parameters to MLflow.
        """

        logger.info("Logging model to MLflow...")

        mlflow.log_params(
            {
                "algorithm": "RandomForestClassifier",
                "n_estimators": N_ESTIMATORS,
                "max_depth": MAX_DEPTH,
                "min_samples_split": MIN_SAMPLES_SPLIT,
                "min_samples_leaf": MIN_SAMPLES_LEAF,
                "random_state": RANDOM_STATE,
            }
        )

        signature = infer_signature(
            X_train,
            model.predict(X_train),
        )

        try:

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(5),
                registered_model_name=MLFLOW_MODEL_NAME,
            )

        except Exception as e:

            logger.warning(
                "Model Registry unavailable (%s). Logging model without registration.",
                e,
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(5),
            )

        mlflow.log_artifact(FEATURE_IMPORTANCE_FILE)

        logger.info("MLflow logging completed.")


def train_model():
    """
    Entry point used by pipeline.py
    """

    trainer = ModelTrainer()

    return trainer.train()


if __name__ == "__main__":

    train_model()
