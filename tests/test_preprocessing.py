"""
Unit tests for preprocessing pipeline.
"""

from pathlib import Path

import pandas as pd

from src.config import TRAIN_DATA, TEST_DATA
from src.data.preprocess import preprocess


def test_preprocess_creates_train_and_test_files():
    train_df, test_df = preprocess()

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    assert not train_df.empty
    assert not test_df.empty

    assert Path(TRAIN_DATA).exists()
    assert Path(TEST_DATA).exists()


def test_train_test_have_same_columns():
    train_df, test_df = preprocess()

    assert list(train_df.columns) == list(test_df.columns)


def test_target_column_exists_after_preprocessing():
    train_df, test_df = preprocess()

    assert "quality" in train_df.columns
    assert "quality" in test_df.columns


def test_no_missing_values_after_preprocessing():
    train_df, test_df = preprocess()

    assert train_df.isnull().sum().sum() == 0
    assert test_df.isnull().sum().sum() == 0
