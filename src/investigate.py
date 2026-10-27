"""Investigate features whose model coefficient points the "wrong" way.

    python -m src.investigate
    python -m src.investigate --features credit_to_annuity_ratio years_employed

A coefficient with an unexpected sign has three usual explanations:
  1. The business assumption is wrong (the raw data really runs the other way).
  2. Overlap with other features: on its own the feature behaves as expected,
     but once correlated features are in the model its coefficient flips.
  3. Missing-value handling: filling blanks with the median mixes "unknown"
     with a real value and distorts the relationship.

For each feature this script produces the evidence to tell them apart:
  * default rate by band, with missing values kept as their own band  -> (1), (3)
  * direction on its own (univariate AUC) vs the baseline coefficient  -> (2)
  * variance inflation factors for all baseline features              -> (2)

Reads data/processed/model_table.parquet (written by the stage 1 pipeline),
so it runs in seconds without rebuilding anything.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import load_config, project_path
from src.preprocess import ScorecardPreprocessor

log = logging.getLogger("investigate")

DEFAULT_FEATURES = ["credit_to_annuity_ratio", "bureau_worst_dpd_12m"]
MISSING_BAND = "Missing"


def default_rate_by_band(df: pd.DataFrame, feature: str, target: str = "target", n_bands: int = 10) -> pd.DataFrame:
    """Default rate per band of a feature, with missing values as a separate band.

    Features with few distinct values (flags, arrears buckets) get one band per
    value; continuous features get equal-sized quantile bands.
    """
    values, y = df[feature], df[target]
    overall = y.mean()
    present = values.notna()

    if values[present].nunique() <= n_bands:
        bands = values[present].astype(float)
    else:
        bands = pd.qcut(values[present], q=n_bands, duplicates="drop")

    grouped = (
        pd.DataFrame({"band": bands, "value": values[present], "target": y[present]})
        .groupby("band", observed=True)
        .agg(min=("value", "min"), max=("value", "max"), n=("target", "size"), default_rate=("target", "mean"))
        .reset_index()
    )
    grouped["band"] = grouped["band"].astype(str)
    grouped[["min", "max"]] = grouped[["min", "max"]].astype(float)

    if (~present).any():
        # Append the Missing row in place (concat with an all-empty row warns in pandas 2.x)
        grouped.loc[len(grouped)] = [MISSING_BAND, np.nan, np.nan, int((~present).sum()), float(y[~present].mean())]

    grouped["share"] = grouped["n"] / len(df)
    grouped["lift_vs_average"] = grouped["default_rate"] / overall
    return grouped[["band", "min", "max", "n", "share", "default_rate", "lift_vs_average"]]


def trend_direction(bands: pd.DataFrame) -> str:
    """Does default rate rise or fall across the (non-missing) bands?
    Uses the rank correlation between band order and default rate."""
    ordered = bands[bands["band"] != MISSING_BAND].reset_index(drop=True)
    if len(ordered) < 3:
        return "too few bands"
    rho = pd.Series(range(len(ordered))).corr(ordered["default_rate"], method="spearman")
    if rho >= 0.6:
        return "increasing"
    if rho <= -0.6:
        return "decreasing"
    return "mixed / not monotonic"


def univariate_direction(df: pd.DataFrame, feature: str, target: str = "target") -> dict:
    """Direction of a feature on its own, among applicants where it is known.
    AUC above 0.5 means higher values go with more defaults."""
    known = df[feature].notna()
    if known.sum() == 0 or df.loc[known, target].nunique() < 2:
        return {"auc_alone": None, "direction_alone": "unknown"}
    auc = roc_auc_score(df.loc[known, target], df.loc[known, feature])
    return {
        "auc_alone": round(float(auc), 4),
        "direction_alone": "positive" if auc > 0.5 else "negative",
        "strength_alone": round(abs(auc - 0.5) * 2, 4),  # Gini of the feature alone
    }


def variance_inflation_factors(X: pd.DataFrame) -> pd.DataFrame:
    """VIF_j = 1 / (1 - R^2_j), where R^2_j comes from regressing feature j on
    all the others. Above about 5, a feature largely duplicates the others and
    its coefficient becomes unstable (it can even flip sign)."""
    A = X.to_numpy(dtype=float)
    rows = []
    for j, col in enumerate(X.columns):
        others = np.delete(A, j, axis=1)
        design = np.column_stack([np.ones(len(A)), others])
        coef, *_ = np.linalg.lstsq(design, A[:, j], rcond=None)
        resid = A[:, j] - design @ coef
        total = ((A[:, j] - A[:, j].mean()) ** 2).sum()
        r2 = 1 - (resid ** 2).sum() / total if total > 0 else 1.0
        rows.append({"feature": col, "vif": round(float(1 / max(1 - r2, 1e-12)), 2), "r2_vs_others": round(float(r2), 4)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def run(features: list[str]) -> dict:
    cfg = load_config()
    table_path = project_path(cfg["data"]["processed_dir"]) / "model_table.parquet"
    if not table_path.exists():
        raise FileNotFoundError(f"{table_path} not found. Run `python -m src.pipeline` first.")

    df = pd.read_parquet(table_path)
    df = df[df["is_train"] == 1].copy()
    df["target"] = df["target"].astype(int)

    out_dir = project_path(cfg["reports_dir"]) / "investigation"
    out_dir.mkdir(parents=True, exist_ok=True)

    coef_path = project_path(cfg["reports_dir"]) / "baseline_coefficients.csv"
    model_signs = {}
    if coef_path.exists():
        coefs = pd.read_csv(coef_path)
        model_signs = {r.feature: ("positive" if r.coef_per_sd > 0 else "negative") for r in coefs.itertuples()}
    expected = cfg["baseline"]["features"]

    summary = {}
    for feature in features:
        bands = default_rate_by_band(df, feature)
        bands.to_csv(out_dir / f"{feature}_bands.csv", index=False)
        alone = univariate_direction(df, feature)
        missing_row = bands[bands["band"] == MISSING_BAND]
        known_rate = df.loc[df[feature].notna(), "target"].mean()
        summary[feature] = {
            "expected_direction": expected.get(feature, "n/a"),
            "direction_in_model": model_signs.get(feature, "n/a"),
            **alone,
            "trend_across_bands": trend_direction(bands),
            "missing_share": round(float(df[feature].isna().mean()), 4),
            "default_rate_missing": round(float(missing_row["default_rate"].iloc[0]), 4) if len(missing_row) else None,
            "default_rate_known": round(float(known_rate), 4),
            "median_used_for_missing": float(df[feature].median()) if df[feature].notna().any() else None,
        }
        log.info("\n=== %s ===\n%s", feature, bands.round(4).to_string(index=False))

    # Overlap between the baseline features, on the same scale the model sees
    base_features = list(expected)
    prep = ScorecardPreprocessor(features=base_features, winsor_quantiles=tuple(cfg["baseline"]["winsor_quantiles"]))
    Z = prep.fit_transform(df[base_features]).drop(columns=prep.constant_features_)
    vif = variance_inflation_factors(Z)
    vif.to_csv(out_dir / "vif.csv", index=False)
    corr = df[base_features].corr(method="spearman").round(3)
    corr.to_csv(out_dir / "feature_correlations.csv")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("\n=== Variance inflation factors ===\n%s", vif.to_string(index=False))
    log.info("\n=== Summary ===\n%s", json.dumps(summary, indent=2))
    log.info("Investigation files written to %s", out_dir)
    return {"summary": summary, "vif": vif}


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate unexpected coefficient signs")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pd.set_option("display.width", 140)
    run(args.features)


if __name__ == "__main__":
    main()
