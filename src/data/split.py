"""
Train/Test Split Module
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_DATA,
    TEST_DATA,
    PROCESSED_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.utils.logger import get_logger
from src.utils.common import create_directory

logger = get_logger(__name__)


def split_dataset(df: pd.DataFrame):
    logger.info("=" * 80)
    logger.info("TRAIN TEST SPLIT STARTED")
    logger.info("=" * 80)

    if df.empty:
        raise ValueError("Input dataframe is empty")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column missing: {TARGET_COLUMN}")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    create_directory(PROCESSED_DIR)

    train_df.to_csv(TRAIN_DATA, index=False)
    test_df.to_csv(TEST_DATA, index=False)

    logger.info("Train data saved to %s", TRAIN_DATA)
    logger.info("Test data saved to %s", TEST_DATA)
    logger.info("Train shape: %s", train_df.shape)
    logger.info("Test shape: %s", test_df.shape)

    logger.info("TRAIN TEST SPLIT COMPLETED")

    return train_df, test_df


if __name__ == "__main__":
    from src.config import RAW_DATA

    dataset = pd.read_csv(RAW_DATA, sep=";")
    split_dataset(dataset)
