from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import DATA_PATH, DROP_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    working = df.copy()

    if TARGET_COLUMN not in working.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    missing_columns = [col for col in FEATURE_COLUMNS if col not in working.columns]
    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")

    columns_to_drop = [col for col in DROP_COLUMNS if col in working.columns]
    if columns_to_drop:
        working = working.drop(columns=columns_to_drop)

    X = working[FEATURE_COLUMNS].copy()
    y = working[TARGET_COLUMN].astype(int).copy()

    return X, y
