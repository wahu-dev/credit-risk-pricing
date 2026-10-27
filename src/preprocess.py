"""Preprocessing for the interpretable baseline (and reused later for scoring).

Every statistic (clip limits, medians, means, standard deviations) is learned
on the TRAINING split only and then applied to validation and new applicants.
Learning them on all the data would leak information from the validation set
and make results look better than they will be in production.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ScorecardPreprocessor:
    features: list[str]
    winsor_quantiles: tuple[float, float] = (0.01, 0.99)
    lower_: dict[str, float] = field(default_factory=dict)
    upper_: dict[str, float] = field(default_factory=dict)
    median_: dict[str, float] = field(default_factory=dict)
    mean_: dict[str, float] = field(default_factory=dict)
    std_: dict[str, float] = field(default_factory=dict)
    constant_features_: list[str] = field(default_factory=list)
    # Features with this many distinct values or fewer (flags, small counts,
    # arrears buckets) are never clipped: percentiles are meaningless for them.
    max_discrete_levels: int = 10

    def fit(self, X: pd.DataFrame) -> "ScorecardPreprocessor":
        lo_q, hi_q = self.winsor_quantiles
        self.constant_features_ = []
        for col in self.features:
            values = X[col].astype(float)
            if values.nunique(dropna=True) <= self.max_discrete_levels:
                # Clipping a rare flag at the 99th percentile would turn every
                # 1 into a 0 (e.g. thin_file at under 1% of applicants).
                self.lower_[col] = float(values.min()) if values.notna().any() else 0.0
                self.upper_[col] = float(values.max()) if values.notna().any() else 0.0
            else:
                # Clip extreme ratios (e.g. income typed as 1 shilling), so a
                # handful of outliers cannot dominate the coefficients.
                self.lower_[col] = float(values.quantile(lo_q))
                self.upper_[col] = float(values.quantile(hi_q))
            clipped = values.clip(self.lower_[col], self.upper_[col])
            # Median, not mean: robust to skewed amounts.
            median = clipped.median()
            self.median_[col] = float(median) if pd.notna(median) else 0.0
            filled = clipped.fillna(self.median_[col])
            self.mean_[col] = float(filled.mean())
            std = float(filled.std(ddof=0))
            if std > 0:
                self.std_[col] = std
            else:
                # No variation in training data: the model cannot learn from it,
                # and keeping it makes the regression unsolvable (singular matrix).
                self.std_[col] = 1.0
                self.constant_features_.append(col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.median_:
            raise RuntimeError("Call fit() before transform().")
        out = pd.DataFrame(index=X.index)
        for col in self.features:
            values = X[col].astype(float).clip(self.lower_[col], self.upper_[col])
            values = values.fillna(self.median_[col])
            # Standardise so each coefficient means "effect of a 1 SD change",
            # which makes coefficients directly comparable with each other.
            out[col] = (values - self.mean_[col]) / self.std_[col]
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["winsor_quantiles"] = list(self.winsor_quantiles)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ScorecardPreprocessor":
        d = dict(d)
        d["winsor_quantiles"] = tuple(d["winsor_quantiles"])
        return cls(**d)


def check_no_nans(Z: pd.DataFrame) -> None:
    if Z.isna().any().any() or np.isinf(Z.to_numpy()).any():
        raise ValueError("Preprocessed features still contain NaN or infinite values.")
