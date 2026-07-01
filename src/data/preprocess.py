"""
Data Preprocessing Module

Responsibilities
----------------
1. Load raw data using ingestion module.
2. Validate dataset.
3. Remove duplicate rows if configured.
4. Split data into train/test sets.
5. Save processed datasets.

Author: Sheejith
"""

from src.config import TARGET_COLUMN
from src.data.ingest import ingest_data
from src.data.validate import validate_dataset
from src.data.split import split_dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


def remove_duplicates(df):
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        logger.warning("Removing %d duplicate rows", duplicate_count)
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        logger.info("No duplicate rows found")

    return df


def preprocess():
    logger.info("=" * 80)
    logger.info("PREPROCESSING STARTED")
    logger.info("=" * 80)

    df = ingest_data()

    validate_dataset(df)

    df = remove_duplicates(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column missing after preprocessing: {TARGET_COLUMN}")

    train_df, test_df = split_dataset(df)

    logger.info("Train shape: %s", train_df.shape)
    logger.info("Test shape: %s", test_df.shape)

    logger.info("=" * 80)
    logger.info("PREPROCESSING COMPLETED")
    logger.info("=" * 80)

    return train_df, test_df


if __name__ == "__main__":
    preprocess()
