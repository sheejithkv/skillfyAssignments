"""
Data Validation Module
"""

import pandas as pd

from src.config import RAW_DATA, TARGET_COLUMN, ARTIFACTS_DIR
from src.utils.logger import get_logger
from src.utils.common import write_json

logger = get_logger(__name__)

VALIDATION_REPORT = ARTIFACTS_DIR / "validation_report.json"


def validate_dataset(df: pd.DataFrame) -> dict:
    logger.info("=" * 80)
    logger.info("DATA VALIDATION STARTED")
    logger.info("=" * 80)

    if df.empty:
        raise ValueError("Dataset is empty")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column missing: {TARGET_COLUMN}")

    missing_values = int(df.isnull().sum().sum())
    if missing_values > 0:
        raise ValueError(f"Dataset contains missing values: {missing_values}")

    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    non_numeric_columns = [
        col for col in feature_columns
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if non_numeric_columns:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric_columns}")

    duplicate_rows = int(df.duplicated().sum())

    report = {
        "status": "SUCCESS",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": TARGET_COLUMN,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "non_numeric_columns": non_numeric_columns,
    }

    write_json(VALIDATION_REPORT, report)

    logger.info("Validation report saved to %s", VALIDATION_REPORT)
    logger.info("DATA VALIDATION COMPLETED")

    return report


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA, sep=";")
    result = validate_dataset(df)
    print(result)
