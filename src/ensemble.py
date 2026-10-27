"""Ensemble members, out-of-fold predictions, stacking and significance testing.

Stacking in one paragraph: each member model is trained K times, each time
leaving out one fold, and predicts the fold it did not see. That gives every
training applicant an honest "out-of-fold" (OOF) prediction from each member.
A small logistic regression then learns how much to trust each member, using
only those honest predictions, so it never learns from answers a member memorised.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.preprocess import ScorecardPreprocessor

EPS = 1e-6


class Member(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Member": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class ScorecardMember:
    """The stage-1 scorecard as an ensemble member (same features, same
    preprocessing). A very large C makes it effectively unpenalised, matching
    the Statsmodels baseline, so its holdout AUC should reproduce stage 1."""

    def __init__(self, features: list[str], winsor_quantiles=(0.01, 0.99)):
        self.features = list(features)
        self.winsor_quantiles = tuple(winsor_quantiles)

    def fit(self, X, y):
        self.prep_ = ScorecardPreprocessor(self.features, self.winsor_quantiles).fit(X[self.features])
        self.used_ = [f for f in self.features if f not in self.prep_.constant_features_]
        Z = self.prep_.transform(X[self.features])[self.used_]
        self.model_ = LogisticRegression(C=1e6, max_iter=2000).fit(Z, y)
        return self

    def predict(self, X):
        Z = self.prep_.transform(X[self.features])[self.used_]
        return self.model_.predict_proba(Z)[:, 1]


class BoostedMember:
    """A LightGBM or XGBoost model with tuned parameters and a fixed tree count
    (chosen during tuning), so CV folds need no early stopping on their own
    validation fold, which would leak into the out-of-fold predictions."""

    def __init__(self, kind: str, params: dict, n_estimators: int, seed: int, columns: list[str]):
        self.kind, self.params, self.n_estimators, self.seed = kind, dict(params), int(n_estimators), seed
        self.columns = list(columns)

    def fit(self, X, y):
        from src.boosting import make_model
        self.model_ = make_model(self.kind, self.params, self.n_estimators, self.seed)
        self.model_.fit(X[self.columns], y)
        return self

    def predict(self, X):
        return self.model_.predict_proba(X[self.columns])[:, 1]


def out_of_fold(factory: Callable[[], Member], X: pd.DataFrame, y: pd.Series, n_folds: int, seed: int) -> np.ndarray:
    """Every row gets a prediction from a model that never saw that row."""
    oof = np.full(len(X), np.nan)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, valid_idx in folds.split(X, y):
        member = factory().fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[valid_idx] = member.predict(X.iloc[valid_idx])
    if np.isnan(oof).any():
        raise RuntimeError("Some rows received no out-of-fold prediction.")
    return oof


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Stacker:
    """Logistic regression on the members' log-odds. Its coefficients are the
    learned trust in each member, which makes the ensemble explainable too."""

    def __init__(self, members: list[str]):
        self.members = list(members)

    def _design(self, P: pd.DataFrame) -> np.ndarray:
        return _logit(P[self.members].to_numpy(dtype=float))

    def fit(self, P: pd.DataFrame, y) -> "Stacker":
        self.model_ = LogisticRegression(C=1.0, max_iter=2000).fit(self._design(P), y)
        self.weights_ = {m: round(float(w), 4) for m, w in zip(self.members, self.model_.coef_[0])}
        return self

    def predict(self, P: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._design(P))[:, 1]

    def cross_validated_predictions(self, P: pd.DataFrame, y, n_folds: int, seed: int) -> np.ndarray:
        """Honest OOF predictions for the stacker itself, so its CV score is
        comparable with the members' (scoring it on the data it was fitted on
        would flatter it)."""
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), self._design(P), y,
                                 cv=folds, method="predict_proba")[:, 1]


def paired_bootstrap_auc(y, p_challenger, p_champion, n_samples: int = 300, seed: int = 0) -> dict:
    """Resample applicants with replacement; each time compute the AUC gap
    between two models on the SAME resample. The spread of gaps gives a 95%
    interval: if it excludes 0, the improvement is not luck."""
    y = np.asarray(y)
    a, b = np.asarray(p_challenger), np.asarray(p_champion)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_samples):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    diffs = np.array(diffs)
    return {
        "auc_difference": round(float(roc_auc_score(y, a) - roc_auc_score(y, b)), 4),
        "ci95_low": round(float(np.percentile(diffs, 2.5)), 4),
        "ci95_high": round(float(np.percentile(diffs, 97.5)), 4),
        "share_of_resamples_better": round(float((diffs > 0).mean()), 4),
        "n_resamples": int(len(diffs)),
    }
