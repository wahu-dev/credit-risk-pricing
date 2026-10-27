"""Interpretable baseline: logistic regression with Statsmodels.

Why start here instead of jumping to XGBoost:
  * It is the industry-standard scorecard. Regulators and risk committees
    understand it, and every later model must beat it to justify its complexity.
  * Statsmodels gives standard errors and p-values, so we can check each
    feature's direction against business sense (does more debt raise risk?).
  * A wrong sign is a red flag for leakage or a data bug, caught early.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from src.metrics import credit_metrics
from src.preprocess import ScorecardPreprocessor, check_no_nans

log = logging.getLogger(__name__)


def train_baseline(
    model_table: pd.DataFrame,
    feature_signs: dict[str, str],
    test_size: float,
    winsor_quantiles: tuple[float, float],
    seed: int,
    reports_dir: Path,
) -> dict:
    """Fit the baseline on training applicants, evaluate on a held-out split,
    and write coefficients, metrics and the fitted model to reports/."""
    features = list(feature_signs)
    missing = [f for f in features if f not in model_table.columns]
    if missing:
        raise KeyError(f"Baseline features not in model table: {missing}")

    train_rows = model_table[model_table["is_train"] == 1]
    X = train_rows[features]
    y = train_rows["target"].astype(int)

    # Stratify so both splits keep the same default rate (about 8% here).
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    prep = ScorecardPreprocessor(features=features, winsor_quantiles=tuple(winsor_quantiles))
    Z_train = prep.fit_transform(X_train)
    Z_valid = prep.transform(X_valid)
    check_no_nans(Z_train)
    check_no_nans(Z_valid)

    # Drop features with no variation in training data, loudly. Keeping one
    # makes the regression unsolvable ("singular matrix").
    dropped = list(prep.constant_features_)
    if dropped:
        log.warning("Dropping features with no variation in training data: %s", dropped)
        features = [f for f in features if f not in dropped]
        feature_signs = {f: s for f, s in feature_signs.items() if f not in dropped}
        Z_train, Z_valid = Z_train[features], Z_valid[features]

    # Exact duplicates or linear combinations also make the matrix singular.
    design = sm.add_constant(Z_train, has_constant="add")
    rank = np.linalg.matrix_rank(design.to_numpy())
    if rank < design.shape[1]:
        raise ValueError(
            f"Baseline features are perfectly collinear (rank {rank} of {design.shape[1]}). "
            "Remove a feature that is a combination of others in config.yaml."
        )

    logit = sm.Logit(y_train, design)
    result = logit.fit(disp=False, maxiter=200)
    if not result.mle_retvals.get("converged", True):
        log.warning("Logistic regression did not fully converge; check for separation.")

    p_train = result.predict(sm.add_constant(Z_train, has_constant="add"))
    p_valid = result.predict(sm.add_constant(Z_valid, has_constant="add"))

    metrics = {
        "train": credit_metrics(y_train, p_train),
        "valid": credit_metrics(y_valid, p_valid),
    }
    # A big train/valid gap means overfitting; a logit rarely overfits, so a
    # gap here usually points to a data problem.
    metrics["auc_gap"] = round(metrics["train"]["roc_auc"] - metrics["valid"]["roc_auc"], 4)
    metrics["features_used"] = features
    metrics["features_dropped_constant"] = dropped

    coefficients = _coefficient_table(result, feature_signs)
    wrong_signs = coefficients.loc[~coefficients["sign_as_expected"], "feature"].tolist()
    if wrong_signs:
        log.warning("Features with unexpected direction (investigate): %s", wrong_signs)

    reports_dir.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(reports_dir / "baseline_coefficients.csv", index=False)
    (reports_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (reports_dir / "baseline_summary.txt").write_text(str(result.summary2()), encoding="utf-8")
    model_artifact = {
        "model_type": "statsmodels_logit",
        "preprocessor": prep.to_dict(),
        "intercept": float(result.params["const"]),
        "coefficients": {f: float(result.params[f]) for f in features},
    }
    (reports_dir / "baseline_model.json").write_text(json.dumps(model_artifact, indent=2), encoding="utf-8")

    log.info(
        "Baseline logit | valid AUC %.4f | Gini %.4f | KS %.4f | PR-AUC %.4f",
        metrics["valid"]["roc_auc"], metrics["valid"]["gini"],
        metrics["valid"]["ks"], metrics["valid"]["pr_auc"],
    )
    return {"metrics": metrics, "wrong_signs": wrong_signs, "coefficients": coefficients}


def _coefficient_table(result, feature_signs: dict[str, str]) -> pd.DataFrame:
    rows = []
    for feature, expected in feature_signs.items():
        coef = float(result.params[feature])
        rows.append({
            "feature": feature,
            "coef_per_sd": round(coef, 4),
            "std_err": round(float(result.bse[feature]), 4),
            "p_value": round(float(result.pvalues[feature]), 5),
            # Odds ratio: 1.30 means a 1 SD increase raises the odds of default by 30%.
            "odds_ratio_per_sd": round(float(np.exp(coef)), 4),
            "expected_sign": expected,
            "sign_as_expected": (coef > 0) == (expected == "positive"),
        })
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def score_with_artifact(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    """Score new applicants from the saved JSON artifact (no Statsmodels needed).
    Used later by the API, and a test proves it matches the fitted model."""
    prep = ScorecardPreprocessor.from_dict(artifact["preprocessor"])
    Z = prep.transform(X)
    linear = artifact["intercept"] + sum(Z[f] * w for f, w in artifact["coefficients"].items())
    return (1.0 / (1.0 + np.exp(-linear))).to_numpy()
