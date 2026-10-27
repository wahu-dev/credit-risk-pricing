"""Data validation for the model table.

A model trained on broken data fails silently: it still produces scores.
These checks stop the pipeline loudly instead, and write a data-quality
report you can show in an interview.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ID_COL = "sk_id_curr"
TARGET_COL = "target"


class DataValidationError(Exception):
    """Raised when the model table breaks a rule. Message lists every failure."""


def validate_model_table(
    df: pd.DataFrame,
    protected_attributes: list[str],
    max_missing_share: float = 0.95,
    min_rows: int = 100,
    expected_rows: int | None = None,
) -> dict:
    """Check the model table and return a quality report.

    Raises DataValidationError listing ALL failures (not just the first),
    so one run tells you everything that needs fixing.
    """
    failures: list[str] = []

    # 1. Size
    if len(df) < min_rows:
        failures.append(f"Only {len(df)} rows; expected at least {min_rows}.")
    if expected_rows is not None and len(df) != expected_rows:
        failures.append(
            f"Row count {len(df):,} != applications {expected_rows:,}. "
            "A join duplicated or dropped applicants."
        )

    # 2. Primary key: one row per applicant
    if df[ID_COL].isna().any():
        failures.append(f"{ID_COL} has missing values.")
    duplicates = int(df[ID_COL].duplicated().sum())
    if duplicates:
        failures.append(f"{duplicates:,} duplicate {ID_COL} values (a join fanned out).")

    # 3. Target: 0/1 for training rows, empty for the test set
    train = df[df["is_train"] == 1]
    test = df[df["is_train"] == 0]
    bad_targets = set(train[TARGET_COL].dropna().unique()) - {0, 1}
    if bad_targets:
        failures.append(f"Unexpected target values: {sorted(bad_targets)}")
    if train[TARGET_COL].isna().any():
        failures.append("Some training rows have no target.")
    if test[TARGET_COL].notna().any():
        failures.append("Test rows have a target value (possible leakage).")

    # 4. Fairness guardrail: protected attributes must never reach the model
    leaked = [c for c in protected_attributes if c in df.columns]
    if leaked:
        failures.append(f"Protected attributes found in model table: {leaked}")

    # 5. Numeric sanity
    numeric = df.select_dtypes(include=[np.number]).drop(columns=[TARGET_COL], errors="ignore")
    infinite_cols = [c for c in numeric.columns if np.isinf(numeric[c].to_numpy(dtype=float)).any()]
    if infinite_cols:
        failures.append(f"Infinite values in: {infinite_cols}")

    missing_share = df.drop(columns=[TARGET_COL]).isna().mean().sort_values(ascending=False)
    too_empty = missing_share[missing_share > max_missing_share]
    if not too_empty.empty:
        failures.append(
            f"Columns more than {max_missing_share:.0%} missing: {too_empty.round(3).to_dict()}"
        )

    report = {
        "rows": int(len(df)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "columns": int(df.shape[1]),
        "default_rate_train": round(float(train[TARGET_COL].mean()), 4) if len(train) else None,
        "thin_file_share": round(float(df["thin_file"].mean()), 4) if "thin_file" in df else None,
        "missing_share_top10": missing_share.head(10).round(4).to_dict(),
        "passed": not failures,
        "failures": failures,
    }

    if failures:
        raise DataValidationError("Data validation failed:\n- " + "\n- ".join(failures))

    log.info(
        "Validation passed: %s rows, default rate %.2f%%, thin-file share %.1f%%",
        f"{report['rows']:,}",
        100 * (report["default_rate_train"] or 0),
        100 * (report["thin_file_share"] or 0),
    )
    return report


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
