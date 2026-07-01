import joblib
import pandas as pd
import matplotlib.pyplot as plt

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
    MODEL_DIR,
    TARGET_COLUMN,
    ARTIFACT_DIR,
)

from src.utils.common import create_directory
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model():

    logger.info("Loading test dataset")

    test_df = pd.read_csv(TEST_DATA)

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    model = joblib.load(MODEL_DIR / "random_forest.pkl")

    logger.info("Running predictions")

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }

    create_directory(ARTIFACT_DIR)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    with open(
        ARTIFACT_DIR / "classification_report.txt",
        "w",
    ) as f:
        f.write(report)

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.tight_layout()

    plt.savefig(
        ARTIFACT_DIR / "confusion_matrix.png"
    )

    plt.close()

    logger.info("Evaluation complete")

    for k, v in metrics.items():
        logger.info("%s : %.4f", k, v)

    return metrics


if __name__ == "__main__":
    evaluate_model()
