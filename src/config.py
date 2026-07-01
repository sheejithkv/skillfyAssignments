from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data" / "raw" / "winequality-red.csv"

PROCESSED_DIR = BASE_DIR / "data" / "processed"

TRAIN_DATA = PROCESSED_DIR / "train.csv"

TEST_DATA = PROCESSED_DIR / "test.csv"

MODEL_DIR = BASE_DIR / "models"

ARTIFACT_DIR = BASE_DIR / "artifacts"

MLFLOW_EXPERIMENT = "wine-quality"

TARGET_COLUMN = "quality"

TEST_SIZE = 0.20

RANDOM_STATE = 42
