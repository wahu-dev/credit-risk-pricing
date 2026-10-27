"""Stage 2 building blocks, tested with plain scikit-learn models so the tests
are fast and do not depend on LightGBM or XGBoost."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.ensemble import Stacker, out_of_fold, paired_bootstrap_auc
from src.model_data import CategoryEncoder, feature_columns, holdout_split
from src.tracking import Tracker


@pytest.fixture
def data():
    rng = np.random.default_rng(11)
    n = 3000
    risk = rng.normal(size=n)
    X = pd.DataFrame({
        "strong": risk + rng.normal(0, 0.5, n),
        "weak": risk + rng.normal(0, 3.0, n),
        "noise": rng.normal(size=n),
    })
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(-2.0 + 1.2 * risk)))).astype(int))
    return X, y


class SpyMember:
    """Logistic regression on one column that records which rows it trained on."""
    trained_on: list = []

    def __init__(self, column):
        self.column = column

    def fit(self, X, y):
        SpyMember.trained_on.append(set(X.index))
        self.model = LogisticRegression().fit(X[[self.column]], y)
        return self

    def predict(self, X):
        SpyMember.trained_on_last_predict = set(X.index)
        return self.model.predict_proba(X[[self.column]])[:, 1]


def test_out_of_fold_never_predicts_a_row_it_trained_on(data):
    X, y = data
    SpyMember.trained_on = []
    oof = out_of_fold(lambda: SpyMember("strong"), X, y, n_folds=4, seed=0)
    assert not np.isnan(oof).any()
    assert len(SpyMember.trained_on) == 4
    # each row is missing from exactly one training set: the fold it was predicted in
    missing_counts = pd.Series(0, index=X.index)
    for rows in SpyMember.trained_on:
        missing_counts[~X.index.isin(list(rows))] += 1
    assert (missing_counts == 1).all()


def test_stacker_trusts_the_stronger_member(data):
    X, y = data
    P = pd.DataFrame({
        "good": out_of_fold(lambda: SpyMember("strong"), X, y, 4, 0),
        "poor": out_of_fold(lambda: SpyMember("weak"), X, y, 4, 0),
    })
    stacker = Stacker(["good", "poor"]).fit(P, y)
    assert stacker.weights_["good"] > stacker.weights_["poor"]
    cv = stacker.cross_validated_predictions(P, y, n_folds=4, seed=0)
    assert cv.shape == (len(y),) and ((cv > 0) & (cv < 1)).all()


def test_bootstrap_detects_a_real_improvement(data):
    X, y = data
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
    good = LogisticRegression().fit(X_tr[["strong"]], y_tr).predict_proba(X_te[["strong"]])[:, 1]
    poor = LogisticRegression().fit(X_tr[["weak"]], y_tr).predict_proba(X_te[["weak"]])[:, 1]
    result = paired_bootstrap_auc(y_te, good, poor, n_samples=200, seed=0)
    assert result["auc_difference"] > 0
    assert result["ci95_low"] > 0              # improvement is statistically clear
    assert result["share_of_resamples_better"] > 0.95


def test_bootstrap_of_identical_models_is_centred_on_zero(data):
    X, y = data
    p = LogisticRegression().fit(X[["strong"]], y).predict_proba(X[["strong"]])[:, 1]
    result = paired_bootstrap_auc(y, p, p, n_samples=50, seed=0)
    assert result["auc_difference"] == 0
    assert result["ci95_low"] == 0 and result["ci95_high"] == 0


def test_holdout_split_matches_the_stage1_baseline_split():
    rng = np.random.default_rng(2)
    y = pd.Series(rng.integers(0, 2, 1000))
    X = pd.DataFrame({"a": rng.normal(size=1000)})
    _, X_valid, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    _, holdout_idx = holdout_split(y, test_size=0.2, seed=42)
    assert sorted(X_valid.index) == list(holdout_idx)


def test_category_encoder_handles_unseen_and_missing():
    train = pd.DataFrame({"income_type": ["Working", "Pensioner", "Working"]})
    new = pd.DataFrame({"income_type": ["Pensioner", "Student", None]})
    enc = CategoryEncoder(["income_type"]).fit(train)
    codes = enc.transform(new)["income_type"].tolist()
    assert codes[0] == enc.categories_["income_type"].index("Pensioner")
    assert codes[1] == -1 and codes[2] == -1


def test_feature_columns_exclude_id_target_and_flags():
    table = pd.DataFrame({"sk_id_curr": [1], "target": [0.0], "is_train": [1],
                          "ratio": [0.5], "income_type": ["Working"]})
    assert feature_columns(table, ["income_type"]) == ["ratio", "income_type"]


def test_disabled_tracker_is_a_silent_no_op(tmp_path):
    tracker = Tracker(enabled=False, db_path=tmp_path / "x.db", experiment="test")
    with tracker.run("anything"):
        tracker.params({"a": 1})
        tracker.metrics({"auc": 0.7})
    assert not tracker.active
    assert not (tmp_path / "x.db").exists()
