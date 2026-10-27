"""Each test checks one feature against a value worked out by hand."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.features import read_table


def _row(df: pd.DataFrame, sk_id: int) -> pd.Series:
    rows = df[df["sk_id_curr"] == sk_id]
    assert len(rows) == 1, f"expected exactly one row for applicant {sk_id}"
    return rows.iloc[0]


def _is_null(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value)


@pytest.fixture
def model_table(warehouse) -> pd.DataFrame:
    return read_table(warehouse, "model_table")


def test_one_row_per_applicant_including_test_set(model_table):
    assert sorted(model_table["sk_id_curr"].tolist()) == [1, 2, 3, 4]
    assert _is_null(_row(model_table, 4)["target"])
    assert _row(model_table, 4)["is_train"] == 0


def test_affordability_ratios(model_table):
    r = _row(model_table, 1)
    assert r["credit_to_income_ratio"] == pytest.approx(3.0)
    assert r["annuity_to_income_ratio"] == pytest.approx(0.15)
    assert r["credit_to_goods_ratio"] == pytest.approx(1.2)
    assert r["credit_to_annuity_ratio"] == pytest.approx(20.0)


def test_employment_placeholder_becomes_null_with_flag(model_table):
    r = _row(model_table, 2)
    assert _is_null(r["years_employed"])
    assert r["employment_placeholder_flag"] == 1
    assert _row(model_table, 1)["years_employed"] == pytest.approx(730 / 365.25)


def test_ext_source_mean_ignores_missing_scores(model_table):
    assert _row(model_table, 1)["ext_source_mean"] == pytest.approx(0.55)  # (0.5 + 0.6) / 2
    assert _is_null(_row(model_table, 2)["ext_source_mean"])               # all three missing
    assert _row(model_table, 2)["ext_source_count"] == 0


def test_bureau_aggregates(model_table):
    r = _row(model_table, 1)
    assert r["bureau_loan_count"] == 2
    assert r["bureau_active_count"] == 1
    assert r["bureau_debt_to_credit"] == pytest.approx(0.5)  # 50 debt / 100 active credit
    assert r["bureau_latest_loan_active"] == 1               # most recent loan is the active one
    assert r["bureau_prolongation_count"] == 1


def test_bureau_balance_uses_last_12_months_only(model_table):
    r = _row(model_table, 1)
    assert r["bureau_worst_dpd_12m"] == 2  # the '5' at month -13 is ignored
    # loan 11: 1 late of 2 months = 0.5; loan 12: 0 of 2 = 0.0; average = 0.25
    assert r["bureau_late_month_share_12m"] == pytest.approx(0.25)


def test_any_arrears_flag_separates_unknown_from_clean(model_table):
    r1 = _row(model_table, 1)
    assert r1["bureau_any_late_12m"] == 1        # a bucket-2 month inside 12m
    assert r1["bureau_balance_missing"] == 0
    r3 = _row(model_table, 3)                      # no bureau history at all
    assert _is_null(r3["bureau_any_late_12m"])     # unknown, NOT "never late"
    assert r3["bureau_balance_missing"] == 1


def test_previous_applications(model_table):
    r = _row(model_table, 1)
    assert r["prev_app_count"] == 2
    assert r["prev_refusal_rate"] == pytest.approx(0.5)
    assert r["prev_last_was_refused"] == 1
    assert r["prev_days_since_last_decision"] == 20
    # applicant 3 asked for 0, so the ratio is skipped rather than divided by zero
    assert _is_null(_row(model_table, 3)["prev_avg_granted_to_requested"])


def test_split_payment_is_not_underpaid(model_table):
    r = _row(model_table, 1)
    assert r["inst_count"] == 2               # 3 payment rows collapse to 2 instalments
    assert r["inst_underpaid_rate"] == pytest.approx(0.0)
    assert r["inst_late_rate"] == pytest.approx(0.5)
    assert r["inst_max_days_late"] == 5
    assert r["inst_avg_days_late"] == pytest.approx(2.5)


def test_history_flags(model_table):
    assert _row(model_table, 1)["thin_file"] == 0
    assert _row(model_table, 2)["thin_file"] == 1
    r3 = _row(model_table, 3)
    assert r3["no_bureau_history"] == 1 and r3["thin_file"] == 0
    assert r3["bureau_loan_count"] == 0       # counts default to 0 ...
    assert _is_null(r3["bureau_debt_to_credit"])  # ... but ratios stay unknown


def test_protected_attributes_are_separate(warehouse, model_table):
    assert "gender" not in model_table.columns
    assert "age_years" not in model_table.columns
    protected = read_table(warehouse, "protected_attributes")
    assert len(protected) == 4
    assert set(protected.columns) == {"sk_id_curr", "gender", "age_years"}
