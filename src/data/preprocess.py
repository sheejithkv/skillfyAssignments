import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA,
    TRAIN_DATA,
    TEST_DATA,
    PROCESSED_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)

from src.utils.common import create_directory
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_dataset():

    logger.info("Loading dataset")

    df = pd.read_csv(RAW_DATA, sep=";")

    logger.info("Dataset shape: %s", df.shape)

    return df


def validate_dataset(df):

    logger.info("Validating dataset")

    if df.empty:
        raise Exception("Dataset is empty")

    if TARGET_COLUMN not in df.columns:
        raise Exception("Target column missing")

    if df.isnull().sum().sum() > 0:
        raise Exception("Dataset contains null values")

    logger.info("Validation successful")


def split_dataset(df):

    logger.info("Splitting dataset")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    return train_df, test_df


def save_dataset(train_df, test_df):

    create_directory(PROCESSED_DIR)

    train_df.to_csv(TRAIN_DATA, index=False)

    test_df.to_csv(TEST_DATA, index=False)

    logger.info("Processed dataset saved")


def preprocess():

    df = load_dataset()

    validate_dataset(df)

    train_df, test_df = split_dataset(df)

    save_dataset(train_df, test_df)

    logger.info("Preprocessing completed")


if __name__ == "__main__":

    preprocess()
