from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import FeatureMeta, ITEM_CATEGORICAL, ITEM_NUMERIC, USER_CATEGORICAL, USER_NUMERIC


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Convert specified columns to numeric dtype in-place.

    Non-parsable values are coerced to NaN (`errors="coerce"`), so downstream
    preprocessing can handle them uniformly (for example, fill with 0).
    """
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _sanitize_categorical(
    df: pd.DataFrame,
    cols: List[str],
    max_values: Dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Normalize categorical feature columns into valid embedding ids.

    For each column:
    - convert to numeric (`errors="coerce"`), fill invalid/missing with 0
    - cast to int64
    - clamp values to be non-negative
    - optionally clamp to per-column max id from `max_values`

    This keeps category ids safe for embedding lookup, especially when
    val/test may contain unseen or out-of-range ids.
    """
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)
        df[col] = df[col].clip(lower=0)
        if max_values is not None and col in max_values:
            df[col] = df[col].clip(upper=max_values[col])
    return df


def load_data(
    path: Path,
    label_col: str,
    meta: FeatureMeta | None = None,
) -> Tuple[pd.DataFrame, FeatureMeta]:
    """
    Load and sanitize one split CSV, and return data + categorical metadata.

    Steps:
    1. Read CSV and validate required columns exist.
    2. Coerce numeric feature columns to numeric (invalid values -> NaN).
    3. Sanitize categorical columns to non-negative int ids.
    4. Fill NaN in numeric features with 0.

    Behavior depends on `meta`:
    - If `meta is None`: infer category vocab sizes from this dataframe
      (`max_id + 1`) and return a newly built `FeatureMeta`.
    - If `meta` is provided: clip categorical ids to the known vocab range
      so val/test cannot exceed train-time embedding sizes.

    Returns:
    - sanitized dataframe
    - `FeatureMeta` used for categorical embedding dimensions
    """
    df = pd.read_csv(path)
    return preprocess_df(df=df, label_col=label_col, meta=meta, source_name=path.name)


def preprocess_df(
    df: pd.DataFrame,
    label_col: str,
    meta: FeatureMeta | None = None,
    source_name: str = "<dataframe>",
) -> Tuple[pd.DataFrame, FeatureMeta]:
    """Sanitize a preloaded split dataframe and return data + categorical metadata."""
    required_cols = set(USER_CATEGORICAL + USER_NUMERIC + ITEM_CATEGORICAL + ITEM_NUMERIC + [label_col])
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {source_name}: {missing}")

    df = df.copy()
    df = _coerce_numeric(df, USER_NUMERIC + ITEM_NUMERIC)

    if meta is None:
        df = _sanitize_categorical(df, USER_CATEGORICAL + ITEM_CATEGORICAL)
        df[USER_NUMERIC + ITEM_NUMERIC] = df[USER_NUMERIC + ITEM_NUMERIC].fillna(0)
        user_cat_sizes = {col: int(df[col].max() + 1) for col in USER_CATEGORICAL}
        item_cat_sizes = {col: int(df[col].max() + 1) for col in ITEM_CATEGORICAL}
        meta = FeatureMeta(user_cat_sizes=user_cat_sizes, item_cat_sizes=item_cat_sizes)
        return df, meta

    max_values = {
        **{k: v - 1 for k, v in meta.user_cat_sizes.items()},
        **{k: v - 1 for k, v in meta.item_cat_sizes.items()},
    }
    df = _sanitize_categorical(df, USER_CATEGORICAL + ITEM_CATEGORICAL, max_values=max_values)
    df[USER_NUMERIC + ITEM_NUMERIC] = df[USER_NUMERIC + ITEM_NUMERIC].fillna(0)
    return df, meta


def load_inference_data(
    path: Path,
    meta: FeatureMeta,
) -> pd.DataFrame:
    """Load and sanitize inference data that may not contain a label column."""
    df = pd.read_csv(path)

    required_cols = set(USER_CATEGORICAL + USER_NUMERIC + ITEM_CATEGORICAL + ITEM_NUMERIC)
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    df = _coerce_numeric(df, USER_NUMERIC + ITEM_NUMERIC)
    max_values = {
        **{k: v - 1 for k, v in meta.user_cat_sizes.items()},
        **{k: v - 1 for k, v in meta.item_cat_sizes.items()},
    }
    df = _sanitize_categorical(df, USER_CATEGORICAL + ITEM_CATEGORICAL, max_values=max_values)
    df[USER_NUMERIC + ITEM_NUMERIC] = df[USER_NUMERIC + ITEM_NUMERIC].fillna(0)
    return df


def build_meta_from_dfs(dfs: List[pd.DataFrame]) -> FeatureMeta:
    """
    Build categorical feature metadata from multiple data splits.

    For each user/item categorical column, this scans all provided DataFrames,
    finds the maximum observed category id, and sets embedding vocab size to
    `max_id + 1`.

    Why this is needed:
    - ensures one consistent embedding size across train/val/test
    - prevents out-of-range category ids during validation/test
    """
    user_cat_sizes = {}
    item_cat_sizes = {}
    for col in USER_CATEGORICAL:
        max_val = max(int(pd.to_numeric(df[col], errors="coerce").fillna(0).max()) for df in dfs)
        user_cat_sizes[col] = max_val + 1
    for col in ITEM_CATEGORICAL:
        max_val = max(int(pd.to_numeric(df[col], errors="coerce").fillna(0).max()) for df in dfs)
        item_cat_sizes[col] = max_val + 1
    return FeatureMeta(user_cat_sizes=user_cat_sizes, item_cat_sizes=item_cat_sizes)
