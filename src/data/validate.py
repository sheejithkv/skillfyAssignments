"""
Data Validation Module

Responsibilities
----------------
1. Validate input dataset
2. Check schema
3. Check missing values
4. Check duplicate rows
5. Validate target column
6. Validate numeric columns
7. Produce validation report

Author: Sheejith
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import Config
from src.utils.logger import logger
from src.utils.exceptions import (
    DataValidationException,
    handle_exceptions,
)
from src.utils.common import write_json


class DataValidator:
    """
    Validates the Wine Quality dataset before model training.
    """

    def __init__(self):

        self.target_column = Config.TARGET_COLUMN

        self.validation_report = {}

    @handle_exceptions
    def validate(self, df: pd.DataFrame) -> Dict:
        """
        Execute complete validation pipeline.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict
        """

        logger.info("=" * 70)
        logger.info("Starting Data Validation")
        logger.info("=" * 70)

        self._check_empty(df)
        self._check_target_column(df)
        self._check_missing_values(df)
        self._check_duplicate_rows(df)
        self._check_numeric_columns(df)

        self.validation_report = {
            "status": "SUCCESS",
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_values": int(df.isnull().sum().sum()),
            "target_column": self.target_column,
        }

        report_path = (
            Config.ARTIFACTS_DIR
            / "validation_report.json"
        )

        write_json(
            report_path,
            self.validation_report,
        )

        logger.info(
            "Validation report saved to %s",
            report_path,
        )

        logger.info("Dataset validation completed.")

        logger.info("=" * 70)

        return self.validation_report

    def _check_empty(self, df: pd.DataFrame):

        if df.empty:
            raise DataValidationException(
                "Dataset is empty."
            )

        logger.info("Dataset is not empty.")

    def _check_target_column(self, df: pd.DataFrame):

        if self.target_column not in df.columns:
            raise DataValidationException(
                f"Target column '{self.target_column}' not found."
            )

        logger.info(
            "Target column '%s' found.",
            self.target_column,
        )

    def _check_missing_values(self, df: pd.DataFrame):

        missing = df.isnull().sum().sum()

        if missing > 0:
            raise DataValidationException(
                f"Dataset contains {missing} missing values."
            )

        logger.info("No missing values found.")

    def _check_duplicate_rows(self, df: pd.DataFrame):

        duplicates = df.duplicated().sum()

        if duplicates > 0:
            logger.warning(
                "Dataset contains %d duplicate rows.",
                duplicates,
            )
        else:
            logger.info("No duplicate rows found.")

    def _check_numeric_columns(self, df: pd.DataFrame):

        feature_columns = [
            column
            for column in df.columns
            if column != self.target_column
        ]

        non_numeric = []

        for column in feature_columns:

            if not pd.api.types.is_numeric_dtype(
                df[column]
            ):
                non_numeric.append(column)

        if non_numeric:
            raise DataValidationException(
                "Non-numeric feature columns detected: "
                + ", ".join(non_numeric)
            )

        logger.info(
            "All feature columns are numeric."
        )


def validate_dataset(df: pd.DataFrame) -> Dict:
    """
    Functional interface for validation.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
    """

    validator = DataValidator()

    return validator.validate(df)


if __name__ == "__main__":

    import pandas as pd

    dataset = pd.read_csv(
        Config.RAW_DATA_FILE,
        sep=";",
    )

    report = validate_dataset(dataset)

    print(report)
