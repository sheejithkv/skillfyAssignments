"""
Data Ingestion Module

Responsibilities
----------------
1. Verify raw dataset exists.
2. Load the dataset.
3. Perform basic validation.
4. Return a pandas DataFrame.

Author: Sheejith
"""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA, TARGET_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """
    Handles loading of the raw dataset.
    """

    def __init__(self, data_path: str | Path = RAW_DATA):
        self.data_path = Path(data_path)

    def validate_source(self) -> None:
        """
        Validate that the dataset exists.
        """
        logger.info("Validating dataset path")

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}"
            )

        if self.data_path.stat().st_size == 0:
            raise ValueError(
                f"Dataset is empty: {self.data_path}"
            )

        logger.info("Dataset path validation successful")

    def load_data(self) -> pd.DataFrame:
        """
        Load the raw dataset.

        Returns
        -------
        pd.DataFrame
        """

        self.validate_source()

        logger.info("Loading dataset")

        df = pd.read_csv(
            self.data_path,
            sep=";",
        )

        logger.info(
            "Dataset loaded successfully. Shape=%s",
            df.shape,
        )

        return df

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Perform basic dataframe validation.
        """

        logger.info("Performing dataframe validation")

        if df.empty:
            raise ValueError("Dataset is empty.")

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' not found."
            )

        if df.duplicated().sum() > 0:
            logger.warning(
                "Dataset contains %d duplicate rows.",
                df.duplicated().sum(),
            )

        logger.info("Dataframe validation completed")

    def ingest(self) -> pd.DataFrame:
        """
        Complete ingestion pipeline.

        Returns
        -------
        pd.DataFrame
        """

        logger.info("=" * 60)
        logger.info("Starting Data Ingestion")
        logger.info("=" * 60)

        df = self.load_data()

        self.validate_dataframe(df)

        logger.info("Data ingestion completed successfully")

        return df


def ingest_data() -> pd.DataFrame:
    """
    Functional interface for pipeline usage.
    """

    ingestion = DataIngestion()

    return ingestion.ingest()


if __name__ == "__main__":

    dataframe = ingest_data()

    print(dataframe.head())
