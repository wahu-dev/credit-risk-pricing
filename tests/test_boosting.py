"""Smoke tests for the real LightGBM / XGBoost / Optuna integration (small data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.boosting import KINDS, feature_importance, fit_with_early_stopping, tune
from src.ensemble import BoostedMember


@pytest.fixture
def split():
    rng = np.random.default_rng(4)
    n = 2000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(X["a"] - 0.5 * X["b"] - 1.5)))).astype(int))
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


@pytest.mark.parametrize("kind", KINDS)
def test_early_stopping_returns_a_sensible_tree_count(kind, split):
    X_tr, X_va, y_tr, y_va = split
    params = {"learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.9}
    model, best_n = fit_with_early_stopping(kind, params, X_tr, y_tr, X_va, y_va, seed=0,
                                            max_estimators=200, patience=20)
    assert 1 <= best_n <= 200
    assert model.predict_proba(X_va).shape == (len(X_va), 2)


@pytest.mark.parametrize("kind", KINDS)
def test_tuning_returns_params_and_beats_chance(kind, split):
    X_tr, X_va, y_tr, y_va = split
    result = tune(kind, X_tr, y_tr, X_va, y_va, n_trials=2, timeout=120, seed=0,
                  max_estimators=100, patience=10)
    assert result["n_trials"] == 2
    assert "learning_rate" in result["params"]
    assert result["inner_auc"] > 0.6


@pytest.mark.parametrize("kind", KINDS)
def test_boosted_member_and_importance(kind, split):
    X_tr, X_va, y_tr, _ = split
    member = BoostedMember(kind, {"learning_rate": 0.1}, 50, seed=0, columns=list(X_tr.columns)).fit(X_tr, y_tr)
    p = member.predict(X_va)
    assert ((p > 0) & (p < 1)).all()
    imp = feature_importance(member.model_, kind, list(X_tr.columns))
    assert imp.iloc[0]["feature"] == "a"   # the strongest true driver
    assert imp["share"].sum() == pytest.approx(1.0)
