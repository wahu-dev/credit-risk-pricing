"""Turn the model table into model-ready matrices, and split it consistently.

Every stage-2 model is evaluated on the SAME holdout applicants as the stage-1
baseline, so comparisons are like for like.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import project_path

ID_COL = "sk_id_curr"
TARGET_COL = "target"
NON_FEATURES = {ID_COL, TARGET_COL, "is_train"}


def load_model_table(cfg: dict, sample: bool = False) -> pd.DataFrame:
    name = "model_table_sample.parquet" if sample else "model_table.parquet"
    path: Path = project_path(cfg["data"]["processed_dir"]) / name
    if not path.exists():
        flag = " --sample" if sample else " --skip-download"
        raise FileNotFoundError(f"{path} not found. Run `python -m src.pipeline{flag}` first.")
    return pd.read_parquet(path)


def training_rows(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Applicants with a known outcome, in stable order, plus their target."""
    rows = table[table["is_train"] == 1].reset_index(drop=True)
    return rows, rows[TARGET_COL].astype(int)


def holdout_split(y: pd.Series, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices for (train, holdout). Same seed, size and stratification as the
    stage-1 baseline, so it selects exactly the same holdout applicants."""
    idx = np.arange(len(y))
    train_idx, holdout_idx = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    return np.sort(train_idx), np.sort(holdout_idx)


def feature_columns(table: pd.DataFrame, categorical: list[str]) -> list[str]:
    """All numeric columns (except id, target, is_train) plus the categoricals."""
    numeric = [c for c in table.select_dtypes(include=[np.number]).columns if c not in NON_FEATURES]
    return numeric + [c for c in categorical if c in table.columns]


@dataclass
class CategoryEncoder:
    """Text categories -> integer codes learned from training data.

    Unseen or missing categories become -1, so a new value at scoring time
    never crashes the model. Tree models split on the codes directly.
    """
    columns: list[str]
    categories_: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "CategoryEncoder":
        for col in self.columns:
            self.categories_[col] = sorted(df[col].dropna().astype(str).unique().tolist())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            # Explicit lookup: unseen or missing values map to -1 directly
            # (pandas is deprecating silent handling of unseen categories).
            mapping = {value: code for code, value in enumerate(self.categories_[col])}
            out[col] = [mapping.get(str(v), -1) if pd.notna(v) else -1 for v in df[col]]
            out[col] = out[col].astype("int32")
        return out
