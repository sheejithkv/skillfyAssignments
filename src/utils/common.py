"""
Common utility functions for the MLOps Wine Quality Project.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_directory(directory: Path) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info("Directory ensured: %s", directory)


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    logger.info("YAML loaded: %s", path)
    return data


def write_yaml(path: Path, data: dict) -> None:
    create_directory(path.parent)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, default_flow_style=False, sort_keys=False)
    logger.info("YAML written: %s", path)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    logger.info("JSON loaded: %s", path)
    return data


def write_json(path: Path, data: dict) -> None:
    create_directory(path.parent)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
    logger.info("JSON written: %s", path)


def read_csv(path: Path, sep: str = ",") -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    logger.info("CSV loaded: %s", path)
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    create_directory(path.parent)
    df.to_csv(path, index=False)
    logger.info("CSV saved: %s", path)


def save_object(obj, path: Path) -> None:
    create_directory(path.parent)
    joblib.dump(obj, path)
    logger.info("Object saved: %s", path)


def load_object(path: Path):
    obj = joblib.load(path)
    logger.info("Object loaded: %s", path)
    return obj


def ensure_file(path: Path) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def file_exists(path: Path) -> bool:
    return Path(path).exists()
