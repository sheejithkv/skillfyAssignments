"""
Train/Test Split Module

Responsibilities
----------------
1. Split dataset into training and testing sets.
2. Preserve class distribution using stratified sampling.
3. Save train and test datasets.
4. Return train and test DataFrames.

Author: Sheejith
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Config
from src.utils.logger import logger
from src.utils.common import (
    create_directory,
    save_csv,
)
from src.utils.exceptions import (
    TrainTestSplitException,
    handle_exceptions,
)


class DataSplitter:
    """
    Handles train-test split operations.
    """

    def __init__(self):

        self.test_size = Config.TEST_SIZE
        self.random_state = Config.RANDOM_STATE
        self.target_column = Config.TARGET_COLUMN

        create_directory(Config.PROCESSED_DATA_DIR)

    @handle_exceptions
    def split(
        self,
        df: pd.DataFrame,
    ):
        """
        Split dataset into train and test sets.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        tuple(pd.DataFrame, pd.DataFrame)
        """

        logger.info("=" * 70)
        logger.info("Starting Train/Test Split")
        logger.info("=" * 70)

        if df.empty:
            raise TrainTestSplitException(
                "Input dataframe is empty."
            )

        if self.target_column not in df.columns:
            raise TrainTestSplitException(
                f"Target column '{self.target_column}' not found."
            )

        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.target_column],
        )

        logger.info(
            "Train Shape : %s",
            train_df.shape,
        )

        logger.info(
            "Test Shape : %s",
            test_df.shape,
        )

        save_csv(
            train_df,
            Config.TRAIN_DATA_FILE,
        )

        save_csv(
            test_df,
            Config.TEST_DATA_FILE,
        )

        logger.info(
            "Training dataset saved to %s",
            Config.TRAIN_DATA_FILE,
        )

        logger.info(
            "Testing dataset saved to %s",
            Config.TEST_DATA_FILE,
        )

        logger.info("=" * 70)
        logger.info("Train/Test Split Completed Successfully")
        logger.info("=" * 70)

        return train_df, test_df


def split_dataset(
    df: pd.DataFrame,
):
    """
    Functional interface.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame)
    """

    splitter = DataSplitter()

    return splitter.split(df)


if __name__ == "__main__":

    import pandas as pd

    dataset = pd.read_csv(
        Config.RAW_DATA_FILE,
        sep=";",
    )

    train, test = split_dataset(dataset)

    print(f"Train Shape : {train.shape}")
    print(f"Test Shape  : {test.shape}")
