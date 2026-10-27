"""Probability calibration: make "PD = 5%" mean 5 in 100 such applicants default.

Ranking metrics (AUC, Gini) ignore whether probabilities are right; pricing
does not. Expected loss = PD x LGD x exposure, so a PD that is 30% too low
underprices every loan by 30% of its expected loss.

Three options, chosen by cross-validation on a calibration set:
  none      keep the model's probabilities as they are
  platt     logistic regression on the model's log-odds (smooth, 2 parameters)
  isotonic  step function that only assumes "higher score = higher risk"
            (flexible, needs more data, can overfit small samples)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

EPS = 1e-6
METHODS = ("none", "platt", "isotonic")


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    def __init__(self, method: str = "platt"):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method = method

    def fit(self, p, y) -> "Calibrator":
        p, y = np.asarray(p, dtype=float), np.asarray(y)
        if self.method == "platt":
            self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.model_ = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(p, y)
        return self

    def transform(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "none":
            return np.clip(p, EPS, 1 - EPS)
        if self.method == "platt":
            return self.model_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return np.clip(self.model_.predict(p), EPS, 1 - EPS)


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Average gap between predicted and observed default rate across
    equal-sized bins, weighted by bin size. 0.005 = off by 0.5 points."""
    table = calibration_table(y, p, n_bins)
    return float((table["n"] * (table["mean_predicted"] - table["observed_rate"]).abs()).sum() / table["n"].sum())


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=n_bins, labels=False)
    return (df.groupby("bin")
              .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
              .reset_index())


def calibration_metrics(y, p) -> dict:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return {
        "brier": round(float(brier_score_loss(y, p)), 6),
        "log_loss": round(float(log_loss(y, p)), 6),
        "ece": round(expected_calibration_error(y, p), 5),
        "mean_predicted": round(float(p.mean()), 5),
        "observed_rate": round(float(np.mean(y)), 5),
    }


def choose_method(p, y, n_folds: int = 5, seed: int = 0) -> tuple[str, pd.DataFrame]:
    """Cross-validated log loss of each method on the calibration set.
    Returns the best method and the comparison table."""
    p, y = np.asarray(p, dtype=float), np.asarray(y)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for method in METHODS:
        losses, eces = [], []
        for tr, va in folds.split(p.reshape(-1, 1), y):
            q = Calibrator(method).fit(p[tr], y[tr]).transform(p[va])
            losses.append(log_loss(y[va], q))
            eces.append(expected_calibration_error(y[va], q))
        rows.append({"method": method, "cv_log_loss": round(float(np.mean(losses)), 6),
                     "cv_ece": round(float(np.mean(eces)), 5)})
    table = pd.DataFrame(rows).sort_values("cv_log_loss").reset_index(drop=True)
    return str(table.iloc[0]["method"]), table
