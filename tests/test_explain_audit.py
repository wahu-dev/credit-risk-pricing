from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.boosting import KINDS
from src.ensemble import BoostedMember, ScorecardMember
from src.explain import BASE, champion_contributions, global_importance, member_contributions, reason_codes
from src.fairness import age_groups, fairness_flags, group_metrics
from src.model_card import md_table


@pytest.fixture
def data():
    rng = np.random.default_rng(8)
    n = 1500
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    X.loc[X.index[:50], "b"] = np.nan
    logit = 1.2 * X["a"].fillna(0) - 0.8 * X["b"].fillna(0) - 2.0
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


# ---------- SHAP contributions add up to the model's own score ----------

@pytest.mark.parametrize("kind", KINDS)
def test_tree_contributions_sum_to_the_model_score(kind, data):
    X, y = data
    member = BoostedMember(kind, {"learning_rate": 0.1}, 60, seed=0, columns=list(X.columns)).fit(X, y)
    contrib = member_contributions(member, X)
    assert np.allclose(contrib.sum(axis=1), _logit(member.predict(X)), atol=1e-4)


def test_scorecard_contributions_sum_to_the_model_score(data):
    X, y = data
    member = ScorecardMember(["a", "b"], (0.01, 0.99)).fit(X, y)
    contrib = member_contributions(member, X)
    assert np.allclose(contrib.sum(axis=1), _logit(member.predict(X)), atol=1e-6)
    assert (contrib["c"] == 0).all()   # features the scorecard does not use contribute nothing


def test_average_champion_contributions_decompose_the_mean_log_odds(data):
    X, y = data
    members = {k: BoostedMember(k, {"learning_rate": 0.1}, 40, 0, list(X.columns)).fit(X, y) for k in KINDS}
    bundle = {"champion": "average_boosted", "members": members}
    contrib = champion_contributions(bundle, X)
    mean_logit = np.mean([_logit(m.predict(X)) for m in members.values()], axis=0)
    assert np.allclose(contrib.sum(axis=1), mean_logit, atol=1e-4)


# ---------- reason codes ----------

def test_reason_codes_pick_risk_increasing_factors_in_order():
    contrib = pd.DataFrame({"score": [0.9, -0.5], "debt": [0.3, -0.2], "late": [0.6, -0.1],
                            "income_type": [0.1, -0.3], BASE: [-2.0, -2.0]})
    X = pd.DataFrame({"score": [0.2, 0.8], "debt": [0.9, 0.1], "late": [np.nan, 0.0], "income_type": [1, 0]})
    typical = pd.Series({"score": 0.5, "debt": 0.4, "late": 0.05, "income_type": 0})
    labels = {"score": "External credit score", "debt": "Debt ratio"}
    out = reason_codes(contrib, X, typical, labels, top_k=3, categories={"income_type": ["Working", "Pensioner"]})
    first = out.iloc[0]
    assert first["reason_1"] == "External credit score is lower than typical (0.20 vs 0.50)"
    assert first["reason_2"] == "Late is missing"
    assert first["reason_3"] == "Debt ratio is higher than typical (0.90 vs 0.40)"
    # an applicant with no risk-increasing factors gets no reasons
    assert (out.iloc[1][["reason_1", "reason_2", "reason_3"]] == "").all()


def test_categorical_reasons_use_category_names():
    contrib = pd.DataFrame({"income_type": [0.5], BASE: [0.0]})
    X = pd.DataFrame({"income_type": [1]})
    out = reason_codes(contrib, X, pd.Series(dtype=float), {}, top_k=1,
                       categories={"income_type": ["Working", "Pensioner"]})
    assert out.iloc[0]["reason_1"] == "Income type is 'Pensioner'"


def test_global_importance_shares_sum_to_one():
    contrib = pd.DataFrame({"a": [1.0, -1.0], "b": [0.1, 0.1], BASE: [0, 0]})
    imp = global_importance(contrib, {})
    assert imp.iloc[0]["feature"] == "a"
    assert imp["share"].sum() == pytest.approx(1.0)


# ---------- fairness ----------

def test_age_groups_boundaries():
    groups = age_groups(pd.Series([24.9, 25.0, 34.99, 55.0, np.nan]), [0, 25, 35, 45, 55, 120],
                        ["under 25", "25-34", "35-44", "45-54", "55+"])
    assert groups.tolist() == ["under 25", "25-34", "25-34", "55+", "unknown"]


def test_group_metrics_and_flags():
    n = 1000
    group = np.array(["A"] * n + ["B"] * n + ["C"] * 50)
    y = np.array([0] * 900 + [1] * 100 + [0] * 900 + [1] * 100 + [0] * 50)
    pd_ = np.concatenate([np.full(n, 0.10), np.full(n, 0.13), np.full(50, 0.1)])  # B over-predicted by 3 pts
    approved = np.concatenate([np.ones(n), np.r_[np.ones(n // 2), np.zeros(n // 2)], np.ones(50)]).astype(bool)
    rate = np.where(approved, 0.2, np.nan)
    table = group_metrics(group, y, pd_, approved, rate, min_group_size=300).set_index("group")
    assert table.loc["A", "adverse_impact_ratio"] == pytest.approx(1.0)
    assert table.loc["B", "adverse_impact_ratio"] == pytest.approx(0.5)
    assert not table.loc["C", "reportable"] and np.isnan(table.loc["C", "adverse_impact_ratio"])
    flags = fairness_flags(table.reset_index(), "gender",
                           {"adverse_impact_ratio": 0.8, "calibration_gap": 0.01, "auc_gap": 0.03})
    assert any("gender=B" in f and "four-fifths" in f for f in flags)
    assert any("gender=B" in f and "over-predicted" in f for f in flags)
    assert not any("gender=A" in f for f in flags)
    assert not any("gender=C" in f for f in flags)   # too small to judge


# ---------- model card ----------

def test_markdown_table_format():
    md = md_table([{"m": "x", "v": 0.5}], [("Model", "m", str), ("Value", "v", lambda v: f"{v:.1f}")])
    assert md.splitlines() == ["| Model | Value |", "| --- | --- |", "| x | 0.5 |"]
