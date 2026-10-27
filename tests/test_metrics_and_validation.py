from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import credit_metrics, ks_statistic
from src.preprocess import ScorecardPreprocessor
from src.validate import DataValidationError, validate_model_table


# ---------- metrics ----------

def test_perfect_model_scores_one():
    y = np.array([0, 0, 1, 1])
    m = credit_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == 1.0 and m["gini"] == 1.0 and m["ks"] == 1.0


def test_random_model_is_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20000)
    m = credit_metrics(y, rng.random(20000))
    assert abs(m["roc_auc"] - 0.5) < 0.02
    assert abs(m["gini"]) < 0.04


def test_ks_equals_max_tpr_minus_fpr():
    y = np.array([0, 1, 0, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.5, 0.9])
    assert ks_statistic(y, s) == pytest.approx(2 / 3)


# ---------- preprocessing ----------

def test_preprocessor_learns_only_from_training_data():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan]})
    new = pd.DataFrame({"x": [np.nan, 1000.0]})
    prep = ScorecardPreprocessor(features=["x"], winsor_quantiles=(0.0, 1.0)).fit(train)
    out = prep.transform(new)
    # Missing value -> training median (2.0); 1000 -> clipped to training max (3.0)
    raw_back = out["x"] * prep.std_["x"] + prep.mean_["x"]
    assert raw_back.tolist() == pytest.approx([2.0, 3.0])


def test_rare_flag_survives_clipping():
    # Regression test: thin_file is 1 for under 1% of real applicants. Clipping
    # at the 99th percentile used to turn every 1 into 0 -> singular matrix.
    flag = pd.DataFrame({"thin_file": [1.0] * 5 + [0.0] * 995})
    prep = ScorecardPreprocessor(features=["thin_file"], winsor_quantiles=(0.01, 0.99)).fit(flag)
    assert prep.transform(flag)["thin_file"].nunique() == 2
    assert prep.constant_features_ == []


def test_continuous_feature_is_still_clipped():
    x = pd.DataFrame({"ratio": np.arange(1000, dtype=float)})
    prep = ScorecardPreprocessor(features=["ratio"], winsor_quantiles=(0.01, 0.99)).fit(x)
    assert prep.upper_["ratio"] < 999


def test_constant_feature_is_reported():
    x = pd.DataFrame({"always_zero": [0.0] * 100})
    prep = ScorecardPreprocessor(features=["always_zero"]).fit(x)
    assert prep.constant_features_ == ["always_zero"]


def test_preprocessor_round_trips_through_json_dict():
    train = pd.DataFrame({"x": [1.0, 5.0, 9.0]})
    prep = ScorecardPreprocessor(features=["x"]).fit(train)
    clone = ScorecardPreprocessor.from_dict(prep.to_dict())
    pd.testing.assert_frame_equal(prep.transform(train), clone.transform(train))


# ---------- validation ----------

def _valid_table(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "sk_id_curr": np.arange(n),
        "is_train": [1] * (n - 20) + [0] * 20,
        "target": list(rng.integers(0, 2, n - 20).astype(float)) + [np.nan] * 20,
        "thin_file": rng.integers(0, 2, n),
        "ratio": rng.random(n),
    })
    return df


def test_valid_table_passes():
    report = validate_model_table(_valid_table(), protected_attributes=["gender"])
    assert report["passed"]


def test_duplicate_ids_fail():
    df = _valid_table()
    df.loc[1, "sk_id_curr"] = df.loc[0, "sk_id_curr"]
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_model_table(df, protected_attributes=["gender"])


def test_protected_attribute_in_model_table_fails():
    df = _valid_table().assign(gender="F")
    with pytest.raises(DataValidationError, match="Protected"):
        validate_model_table(df, protected_attributes=["gender"])


def test_infinite_values_fail():
    df = _valid_table()
    df.loc[3, "ratio"] = np.inf
    with pytest.raises(DataValidationError, match="Infinite"):
        validate_model_table(df, protected_attributes=[])


def test_row_count_mismatch_fails():
    with pytest.raises(DataValidationError, match="Row count"):
        validate_model_table(_valid_table(), protected_attributes=[], expected_rows=999)


def test_all_failures_reported_together():
    df = _valid_table().assign(gender="F")
    df.loc[1, "sk_id_curr"] = df.loc[0, "sk_id_curr"]
    with pytest.raises(DataValidationError) as err:
        validate_model_table(df, protected_attributes=["gender"])
    assert "duplicate" in str(err.value) and "Protected" in str(err.value)
