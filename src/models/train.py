import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from src.config import (
    TRAIN_DATA,
    MODEL_DIR,
    TARGET_COLUMN,
    RANDOM_STATE,
)
from src.utils.common import create_directory
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_model():

    logger.info("Loading training dataset")

    train_df = pd.read_csv(TRAIN_DATA)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    logger.info("Training samples: %d", len(train_df))

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logger.info("Training Random Forest model")

    model.fit(X_train, y_train)

    create_directory(MODEL_DIR)

    model_path = MODEL_DIR / "random_forest.pkl"

    joblib.dump(model, model_path)

    logger.info("Model saved to %s", model_path)

    return model


if __name__ == "__main__":
    train_model()
