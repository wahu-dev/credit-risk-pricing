"""The baseline learns real signal and its saved artifact reproduces its scores."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.baseline import score_with_artifact, train_baseline


@pytest.fixture
def synthetic_table() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 4000
    risk = rng.normal(size=n)
    df = pd.DataFrame({
        "sk_id_curr": np.arange(n),
        "is_train": 1,
        "good_signal": -risk + rng.normal(0, 0.5, n),  # higher = safer
        "bad_signal": risk + rng.normal(0, 0.5, n),    # higher = riskier
        "noise": rng.normal(size=n),
    })
    df.loc[rng.random(n) < 0.1, "good_signal"] = np.nan  # some missing values
    df["target"] = (rng.random(n) < 1 / (1 + np.exp(-(-2.5 + 1.2 * risk)))).astype(int)
    return df


def test_baseline_finds_signal_and_correct_signs(synthetic_table, tmp_path):
    result = train_baseline(
        synthetic_table,
        feature_signs={"good_signal": "negative", "bad_signal": "positive", "noise": "positive"},
        test_size=0.25, winsor_quantiles=(0.01, 0.99), seed=1, reports_dir=tmp_path,
    )
    assert result["metrics"]["valid"]["roc_auc"] > 0.75
    coefs = result["coefficients"].set_index("feature")
    assert coefs.loc["good_signal", "sign_as_expected"]
    assert coefs.loc["bad_signal", "sign_as_expected"]
    for name in ["baseline_metrics.json", "baseline_coefficients.csv", "baseline_model.json"]:
        assert (tmp_path / name).exists()


def test_constant_feature_is_dropped_not_fatal(synthetic_table, tmp_path):
    table = synthetic_table.assign(always_zero=0.0, rare_flag=0.0)
    table.loc[table.index[:20], "rare_flag"] = 1.0  # 0.5% of rows, like thin_file
    result = train_baseline(
        table,
        feature_signs={"good_signal": "negative", "bad_signal": "positive",
                       "always_zero": "positive", "rare_flag": "positive"},
        test_size=0.25, winsor_quantiles=(0.01, 0.99), seed=1, reports_dir=tmp_path,
    )
    assert result["metrics"]["features_dropped_constant"] == ["always_zero"]
    assert "rare_flag" in result["metrics"]["features_used"]


def test_saved_artifact_reproduces_statsmodels_scores(synthetic_table, tmp_path):
    import statsmodels.api as sm
    from src.preprocess import ScorecardPreprocessor

    signs = {"good_signal": "negative", "bad_signal": "positive"}
    train_baseline(synthetic_table, signs, 0.25, (0.01, 0.99), 1, tmp_path)
    artifact = json.loads((tmp_path / "baseline_model.json").read_text())

    X = synthetic_table[list(signs)].head(50)
    from_artifact = score_with_artifact(artifact, X)

    prep = ScorecardPreprocessor.from_dict(artifact["preprocessor"])
    params = pd.Series({"const": artifact["intercept"], **artifact["coefficients"]})
    expected = 1 / (1 + np.exp(-(sm.add_constant(prep.transform(X), has_constant="add") @ params)))
    np.testing.assert_allclose(from_artifact, expected.to_numpy(), rtol=1e-10)
