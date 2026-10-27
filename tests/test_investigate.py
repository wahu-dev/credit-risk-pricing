from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.investigate import (
    MISSING_BAND,
    default_rate_by_band,
    trend_direction,
    univariate_direction,
    variance_inflation_factors,
)


@pytest.fixture
def risk_table() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 5000
    x = rng.normal(size=n)
    target = (rng.random(n) < 1 / (1 + np.exp(-(-2.5 + 1.0 * x)))).astype(int)
    df = pd.DataFrame({"x": x, "target": target, "flag": (x > 1.5).astype(float)})
    df.loc[df.index[:500], "x"] = np.nan  # 10% missing
    return df


def test_missing_values_get_their_own_band(risk_table):
    bands = default_rate_by_band(risk_table, "x")
    assert bands["band"].iloc[-1] == MISSING_BAND
    assert bands.loc[bands["band"] == MISSING_BAND, "n"].iloc[0] == 500
    assert bands["n"].sum() == len(risk_table)
    assert bands["share"].sum() == pytest.approx(1.0)


def test_rising_risk_is_detected(risk_table):
    bands = default_rate_by_band(risk_table, "x")
    assert len(bands) == 11  # 10 deciles + Missing
    assert trend_direction(bands) == "increasing"
    assert univariate_direction(risk_table, "x")["direction_alone"] == "positive"


def test_discrete_feature_gets_one_band_per_value(risk_table):
    bands = default_rate_by_band(risk_table, "flag")
    assert len(bands) == 2


def test_vif_flags_near_duplicate_features():
    rng = np.random.default_rng(5)
    a = rng.normal(size=2000)
    X = pd.DataFrame({
        "a": a,
        "a_copy": a + rng.normal(0, 0.05, 2000),  # almost the same as a
        "independent": rng.normal(size=2000),
    })
    vif = variance_inflation_factors(X).set_index("feature")["vif"]
    assert vif["a"] > 50 and vif["a_copy"] > 50
    assert vif["independent"] < 1.5
