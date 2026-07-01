"""
Application configuration.

This module centralizes all project paths and constants so the rest of the
application imports configuration from a single place.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Project Paths
# -----------------------------------------------------------------------------

# Project Root
# mlops-end2end/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA = RAW_DATA_DIR / "winequality-red.csv"
TRAIN_DATA = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA = PROCESSED_DATA_DIR / "test.csv"

# Models
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "random_forest.pkl"

# Artifacts
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

CONFUSION_MATRIX_FILE = ARTIFACT_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_FILE = ARTIFACT_DIR / "feature_importance.png"
CLASSIFICATION_REPORT_FILE = ARTIFACT_DIR / "classification_report.txt"

# Logs
LOG_DIR = PROJECT_ROOT / "logs"

# MLflow
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT = "wine-quality"
MLFLOW_MODEL_NAME = "wine-quality-model"

# -----------------------------------------------------------------------------
# Dataset Configuration
# -----------------------------------------------------------------------------

TARGET_COLUMN = "quality"

TEST_SIZE = 0.20

RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Random Forest Defaults
# -----------------------------------------------------------------------------

N_ESTIMATORS = 100

MAX_DEPTH = None

MIN_SAMPLES_SPLIT = 2

MIN_SAMPLES_LEAF = 1

N_JOBS = -1

# -----------------------------------------------------------------------------
# Plot Settings
# -----------------------------------------------------------------------------

FIGURE_WIDTH = 10

FIGURE_HEIGHT = 6

DPI = 120

# -----------------------------------------------------------------------------
# Directory List
# -----------------------------------------------------------------------------

REQUIRED_DIRECTORIES = [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    ARTIFACT_DIR,
    LOG_DIR,
]
