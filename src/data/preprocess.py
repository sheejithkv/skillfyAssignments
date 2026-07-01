"""
Data Preprocessing Module

Responsibilities
----------------
1. Load raw dataset using DataIngestion
2. Validate dataset
3. Perform basic preprocessing
4. Split train/test dataset
5. Save processed datasets

Author: Sheejith
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Config
from src.data.ingest import DataIngestion

from src.utils.logger import logger
from src.utils.common import (
    create_directory,
    save_csv,
)

from src.utils.exceptions import (
    PreprocessingException,
    handle_exceptions,
)


class DataPreprocessor:
    """
    Data preprocessing pipeline.
    """

    def __init__(self):

        self.ingestion = DataIngestion()

        create_directory(Config.PROCESSED_DATA_DIR)

    @handle_exceptions
    def preprocess(self):
        """
        Complete preprocessing pipeline.
        """

        logger.info("=" * 70)
        logger.info("Starting Data Preprocessing")
        logger.info("=" * 70)

        # ---------------------------------------------------
        # Load dataset
        # ---------------------------------------------------

        df = self.ingestion.ingest()

        logger.info(
            "Dataset loaded successfully. Shape=%s",
            df.shape,
        )

        # ---------------------------------------------------
        # Remove duplicate rows
        # ---------------------------------------------------

        duplicate_rows = df.duplicated().sum()

        if duplicate_rows > 0:
            logger.warning(
                "Removing %d duplicate rows.",
                duplicate_rows,
            )

            df = df.drop_duplicates()

        # ---------------------------------------------------
        # Missing values
        # ---------------------------------------------------

        missing = df.isnull().sum().sum()

        if missing > 0:
            raise PreprocessingException(
                f"Dataset contains {missing} missing values."
            )

        logger.info("No missing values detected.")

        # ---------------------------------------------------
        # Train Test Split
        # ---------------------------------------------------

        train_df, test_df = train_test_split(
            df,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=df[Config.TARGET_COLUMN],
        )

        logger.info(
            "Train Shape : %s",
            train_df.shape,
        )

        logger.info(
            "Test Shape : %s",
            test_df.shape,
        )

        # ---------------------------------------------------
        # Save processed data
        # ---------------------------------------------------

        save_csv(
            train_df,
            Config.TRAIN_DATA_FILE,
        )

        save_csv(
            test_df,
            Config.TEST_DATA_FILE,
        )

        logger.info(
            "Train dataset saved to %s",
            Config.TRAIN_DATA_FILE,
        )

        logger.info(
            "Test dataset saved to %s",
            Config.TEST_DATA_FILE,
        )

        logger.info("=" * 70)
        logger.info("Preprocessing completed successfully")
        logger.info("=" * 70)

        return train_df, test_df


def preprocess():
    """
    Functional interface.
    """

    processor = DataPreprocessor()

    return processor.preprocess()


if __name__ == "__main__":

    preprocess()
