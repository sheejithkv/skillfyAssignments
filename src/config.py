from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

LOG_DIR = PROJECT_ROOT / "logs"

RAW_DATA = RAW_DIR / "winequality-red.csv"

TRAIN_DATA = PROCESSED_DIR / "train.csv"

TEST_DATA = PROCESSED_DIR / "test.csv"

MODEL_FILE = MODELS_DIR / "model.pkl"

METRICS_FILE = METRICS_DIR / "metrics.json"

LOG_FILE = LOG_DIR / "mlops.log"

TARGET_COLUMN = "quality"

RANDOM_STATE = 42

TEST_SIZE = 0.20

for directory in [
    RAW_DIR,
    PROCESSED_DIR,
    INTERIM_DIR,
    MODELS_DIR,
    METRICS_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
