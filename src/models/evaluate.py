"""
Model evaluation module.

Responsibilities
----------------
1. Load trained model
2. Load test dataset
3. Predict on test data
4. Calculate evaluation metrics
5. Generate confusion matrix
6. Save classification report
7. Log metrics and artifacts to MLflow
"""

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from src.config import (
    TEST_DATA,
    TARGET_COLUMN,
    MODEL_FILE,
    ARTIFACT_DIR,
    CONFUSION_MATRIX_FILE,
    CLASSIFICATION_REPORT_FILE,
)

from src.utils.common import create_directory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluate trained model.
    """

    def __init__(self):

        create_directory(ARTIFACT_DIR)

    def load_test_data(self):
        """
        Load processed test dataset.
        """

        logger.info("Loading test dataset...")

        df = pd.read_csv(TEST_DATA)

        logger.info("Test dataset shape: %s", df.shape)

        X = df.drop(columns=[TARGET_COLUMN])

        y = df[TARGET_COLUMN]

        return X, y

    def load_model(self):
        """
        Load trained Random Forest model.
        """

        logger.info("Loading trained model...")

        model = joblib.load(MODEL_FILE)

        return model

    def evaluate(self):

        X_test, y_test = self.load_test_data()

        model = self.load_model()

        logger.info("Running predictions...")

        predictions = model.predict(X_test)

        metrics = self.calculate_metrics(
            y_test,
            predictions,
        )

        self.save_classification_report(
            y_test,
            predictions,
        )

        self.save_confusion_matrix(
            y_test,
            predictions,
        )

        self.log_mlflow(metrics)

        logger.info("Evaluation completed.")

        return metrics

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate evaluation metrics.
        """

        metrics = {
            "accuracy": accuracy_score(
                y_true,
                y_pred,
            ),
            "precision": precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
        }

        logger.info("Evaluation Metrics")

        for key, value in metrics.items():
            logger.info("%s : %.4f", key, value)

        return metrics

    def save_classification_report(
        self,
        y_true,
        y_pred,
    ):
        """
        Save classification report.
        """

        report = classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )

        with open(
            CLASSIFICATION_REPORT_FILE,
            "w",
        ) as file:
            file.write(report)

        logger.info(
            "Classification report saved -> %s",
            CLASSIFICATION_REPORT_FILE,
        )

    def save_confusion_matrix(
        self,
        y_true,
        y_pred,
    ):
        """
        Save confusion matrix.
        """

        cm = confusion_matrix(
            y_true,
            y_pred,
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
        )

        plt.figure(figsize=(8, 6))

        disp.plot(cmap="Blues")

        plt.tight_layout()

        plt.savefig(
            CONFUSION_MATRIX_FILE,
            dpi=150,
        )

        plt.close()

        logger.info(
            "Confusion matrix saved -> %s",
            CONFUSION_MATRIX_FILE,
        )

    def log_mlflow(self, metrics):
        """
        Log metrics and artifacts.
        """

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(
            CLASSIFICATION_REPORT_FILE
        )

        mlflow.log_artifact(
            CONFUSION_MATRIX_FILE
        )

        logger.info("Metrics logged to MLflow.")


def evaluate_model():
    """
    Entry point for pipeline.
    """

    evaluator = ModelEvaluator()

    return evaluator.evaluate()


if __name__ == "__main__":

    metrics = evaluate_model()

    print("\nFinal Metrics\n")

    for key, value in metrics.items():
        print(f"{key:15} : {value:.4f}")
