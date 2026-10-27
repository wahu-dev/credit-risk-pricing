"""Gradient-boosted tree models (LightGBM, XGBoost) and Optuna tuning.

Design choices worth defending:
  * No class re-weighting (scale_pos_weight). It would inflate predicted
    probabilities; stage 3 uses PD as a real probability to price loans, so we
    keep the model's probabilities honest and handle imbalance in the metrics.
  * Tuning uses an inner validation split carved out of the training data with
    early stopping to choose the number of trees. The holdout is never seen.
  * Search ranges are deliberately conservative (min_child_samples >= 50,
    learning rate 0.02-0.1): a credit model must be stable, not just accurate.
"""
from __future__ import annotations

import logging
from typing import Callable

import lightgbm as lgb
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)

KINDS = ("lightgbm", "xgboost")


def lightgbm_space(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 500, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


def xgboost_space(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


SEARCH_SPACES: dict[str, Callable] = {"lightgbm": lightgbm_space, "xgboost": xgboost_space}


def make_model(kind: str, params: dict, n_estimators: int, seed: int, early_stopping_rounds: int | None = None):
    if kind == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=n_estimators, subsample_freq=1, random_state=seed,
            n_jobs=-1, verbose=-1, **params,
        )
    if kind == "xgboost":
        extra = {"early_stopping_rounds": early_stopping_rounds} if early_stopping_rounds else {}
        return xgb.XGBClassifier(
            n_estimators=n_estimators, tree_method="hist", eval_metric="auc",
            random_state=seed, n_jobs=-1, **extra, **params,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def fit_with_early_stopping(kind, params, X_tr, y_tr, X_va, y_va, seed, max_estimators, patience):
    """Fit with early stopping on the validation set.
    Returns (model, number_of_trees_at_best_validation_score)."""
    if kind == "lightgbm":
        model = make_model(kind, params, max_estimators, seed)
        model.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
            callbacks=[lgb.early_stopping(patience, verbose=False)],
        )
        best = model.best_iteration_ or max_estimators
    else:
        model = make_model(kind, params, max_estimators, seed, early_stopping_rounds=patience)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        best = model.best_iteration + 1  # XGBoost counts from 0
    return model, int(best)


def tune(kind: str, X_tr: pd.DataFrame, y_tr, X_va: pd.DataFrame, y_va, *, n_trials: int,
         timeout: int | None, seed: int, max_estimators: int, patience: int,
         on_trial: Callable[[int, dict, float, int], None] | None = None) -> dict:
    """Bayesian search (Optuna TPE) for the best hyperparameters by validation AUC."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = SEARCH_SPACES[kind]

    def objective(trial):
        params = space(trial)
        model, best_n = fit_with_early_stopping(kind, params, X_tr, y_tr, X_va, y_va, seed, max_estimators, patience)
        auc = float(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]))
        trial.set_user_attr("n_estimators", best_n)
        log.info("%s trial %d: AUC %.4f with %d trees", kind, trial.number, auc, best_n)
        if on_trial:
            on_trial(trial.number, params, auc, best_n)
        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = study.best_trial
    return {
        "params": dict(best.params),
        "n_estimators": int(best.user_attrs["n_estimators"]),
        "inner_auc": round(float(best.value), 4),
        "n_trials": len(study.trials),
    }


def feature_importance(model, kind: str, columns: list[str]) -> pd.DataFrame:
    """Gain importance: total loss reduction from splits on each feature."""
    if kind == "lightgbm":
        gain = model.booster_.feature_importance(importance_type="gain")
        scores = dict(zip(columns, gain))
    else:
        scores = model.get_booster().get_score(importance_type="total_gain")
        scores = {c: scores.get(c, 0.0) for c in columns}
    imp = pd.DataFrame({"feature": list(scores), "gain": [float(v) for v in scores.values()]})
    total = imp["gain"].sum() or 1.0
    imp["share"] = imp["gain"] / total
    return imp.sort_values("gain", ascending=False).reset_index(drop=True)
