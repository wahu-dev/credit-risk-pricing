"""Stage 2: challenger models vs the scorecard.

    python -m src.train_models --quick      # 5-minute check that everything runs
    python -m src.train_models              # full run (about 30-60 minutes)
    python -m src.train_models --sample --quick --no-tracking   # CI smoke test

Flow:
  1. Same holdout as the stage-1 baseline (never used for tuning or selection).
  2. Tune LightGBM and XGBoost with Optuna on an inner validation split.
  3. Out-of-fold predictions for the scorecard, LightGBM and XGBoost.
  4. Stacked ensemble + simple average.
  5. Pick the champion by cross-validated AUC, then report it on the holdout,
     with a bootstrap confidence interval against the scorecard.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.boosting import KINDS, feature_importance, tune
from src.config import load_config, project_path
from src.ensemble import BoostedMember, ScorecardMember, Stacker, out_of_fold, paired_bootstrap_auc
from src.metrics import credit_metrics
from src.model_data import CategoryEncoder, feature_columns, holdout_split, load_model_table, training_rows
from src.tracking import Tracker

log = logging.getLogger("train_models")

SCORECARD = "scorecard"


def run(sample: bool = False, quick: bool = False, tracking: bool = True) -> dict:
    started = time.perf_counter()
    cfg = load_config()
    mcfg = cfg["models"]
    seed = cfg["project"]["random_seed"]
    settings = {**mcfg["tuning"], "n_folds": mcfg["n_folds"]}
    if quick:
        settings.update(mcfg["quick"])

    # ---------- data ----------
    table = load_model_table(cfg, sample)
    rows, y = training_rows(table)
    train_idx, holdout_idx = holdout_split(y, cfg["baseline"]["test_size"], seed)

    categorical = cfg["features"]["categorical"]
    encoder = CategoryEncoder(categorical).fit(rows.iloc[train_idx])
    columns = feature_columns(rows, categorical)
    X = encoder.transform(rows)[columns]
    X_tr, y_tr = X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
    X_ho, y_ho = X.iloc[holdout_idx].reset_index(drop=True), y.iloc[holdout_idx].reset_index(drop=True)
    log.info("Training %s applicants, holdout %s, %d features", f"{len(X_tr):,}", f"{len(X_ho):,}", len(columns))

    suffix = "_sample" if sample else ""
    out_dir = project_path(cfg["reports_dir"]) / f"models{suffix}"
    model_dir = project_path(mcfg["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    tcfg = cfg["tracking"]
    tracker = Tracker(tracking and tcfg["enabled"], project_path(tcfg["db_path"]), tcfg["experiment"] + suffix)

    with tracker.run(f"stage2{'-quick' if quick else ''}{suffix}"):
        tracker.params({"n_features": len(columns), "n_train": len(X_tr), "n_holdout": len(X_ho), **settings})

        # ---------- 1. tune the challengers ----------
        X_in, X_iv, y_in, y_iv = train_test_split(
            X_tr, y_tr, test_size=settings["inner_valid_size"], stratify=y_tr, random_state=seed)
        tuned = {}
        for kind in KINDS:
            with tracker.run(f"tune-{kind}", nested=True):
                result = tune(
                    kind, X_in, y_in, X_iv, y_iv,
                    n_trials=settings["n_trials"], timeout=settings["timeout_seconds"], seed=seed,
                    max_estimators=settings["max_estimators"], patience=settings["early_stopping_rounds"],
                    on_trial=lambda n, p, auc, trees: tracker.metrics({"trial_auc": auc, "trial_trees": trees}, step=n),
                )
                tracker.params(result["params"], prefix="best_")
                tracker.metrics({"inner_auc": result["inner_auc"], "n_estimators": result["n_estimators"]})
            tuned[kind] = result
            log.info("Best %s: inner AUC %.4f, %d trees, %s", kind, result["inner_auc"], result["n_estimators"], result["params"])

        # ---------- 2. members ----------
        scorecard_features = list(cfg["baseline"]["features"])
        winsor = tuple(cfg["baseline"]["winsor_quantiles"])
        factories = {
            SCORECARD: lambda: ScorecardMember(scorecard_features, winsor),
            **{kind: (lambda k=kind: BoostedMember(k, tuned[k]["params"], tuned[k]["n_estimators"], seed, columns))
               for kind in KINDS},
        }
        members = list(factories)

        oof = pd.DataFrame(index=range(len(X_tr)))
        holdout = pd.DataFrame(index=range(len(X_ho)))
        fitted = {}
        for name, factory in factories.items():
            log.info("Out-of-fold predictions for %s (%d folds)...", name, settings["n_folds"])
            oof[name] = out_of_fold(factory, X_tr, y_tr, settings["n_folds"], seed)
            fitted[name] = factory().fit(X_tr, y_tr)
            holdout[name] = fitted[name].predict(X_ho)

        # ---------- 3. ensembles ----------
        stacker = Stacker(members).fit(oof, y_tr)
        oof["stacked"] = stacker.cross_validated_predictions(oof[members], y_tr, settings["n_folds"], seed)
        holdout["stacked"] = stacker.predict(holdout[members])
        oof["average_boosted"] = oof[list(KINDS)].mean(axis=1)
        holdout["average_boosted"] = holdout[list(KINDS)].mean(axis=1)
        log.info("Stacker weights (trust in each member): %s", stacker.weights_)

        # ---------- 4. compare ----------
        rows_out = []
        for name in oof.columns:
            m = credit_metrics(y_ho, holdout[name])
            rows_out.append({"model": name, "cv_auc": round(float(roc_auc_score(y_tr, oof[name])), 4),
                             **{f"holdout_{k}": v for k, v in m.items() if k not in ("n", "default_rate")}})
        comparison = pd.DataFrame(rows_out).sort_values("cv_auc", ascending=False).reset_index(drop=True)

        # Select on cross-validation, NOT the holdout: choosing the best holdout
        # score and then reporting that same score would overstate it.
        champion = comparison.iloc[0]["model"]
        # Always test the best challenger against the scorecard (even when the
        # scorecard wins), so the report answers "is complexity worth it?".
        challenger = comparison.loc[comparison["model"] != SCORECARD, "model"].iloc[0]
        significance = paired_bootstrap_auc(y_ho, holdout[challenger], holdout[SCORECARD],
                                            n_samples=mcfg["bootstrap_samples"], seed=seed)

        importance = feature_importance(fitted["lightgbm"].model_, "lightgbm", columns)

        # ---------- 5. save ----------
        comparison.to_csv(out_dir / "comparison.csv", index=False)
        importance.to_csv(out_dir / "feature_importance_lightgbm.csv", index=False)
        summary = {
            "champion": champion,
            "best_challenger": challenger,
            "best_challenger_vs_scorecard_holdout": significance,
            "stacker_weights": stacker.weights_,
            "tuned": tuned,
            "settings": settings,
            "runtime_minutes": round((time.perf_counter() - started) / 60, 1),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        joblib.dump({"encoder": encoder, "columns": columns, "members": fitted, "stacker": stacker,
                     "champion": champion}, model_dir / f"stage2_models{suffix}.joblib")

        tracker.params({"champion": champion, "best_challenger": challenger})
        tracker.metrics({f"{r['model']}_cv_auc": r["cv_auc"] for r in rows_out})
        tracker.metrics({f"{r['model']}_holdout_auc": r["holdout_roc_auc"] for r in rows_out})
        tracker.metrics({k: v for k, v in significance.items() if k != "n_resamples"}, prefix="challenger_vs_scorecard_")
        for f in ("comparison.csv", "feature_importance_lightgbm.csv", "summary.json"):
            tracker.artifact(out_dir / f)

    pd.set_option("display.width", 160)
    log.info("\n=== Model comparison (sorted by cross-validated AUC) ===\n%s", comparison.to_string(index=False))
    log.info("Champion (by CV AUC): %s", champion)
    log.info("Best challenger %s vs scorecard on holdout: AUC %+.4f (95%% CI %+.4f to %+.4f), better in %.0f%% of resamples",
             challenger, significance["auc_difference"], significance["ci95_low"], significance["ci95_high"],
             100 * significance["share_of_resamples_better"])
    log.info("\nTop 10 LightGBM features by gain:\n%s", importance.head(10).round(4).to_string(index=False))
    log.info("Finished in %.1f minutes. Results in %s", summary["runtime_minutes"], out_dir)
    return {"comparison": comparison, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: challenger models and ensemble")
    parser.add_argument("--sample", action="store_true", help="use the synthetic sample model table")
    parser.add_argument("--quick", action="store_true", help="few trials and folds, to check the plumbing")
    parser.add_argument("--no-tracking", action="store_true", help="skip MLflow logging")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for noisy in ("lightgbm", "optuna", "alembic", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run(sample=args.sample, quick=args.quick, tracking=not args.no_tracking)


if __name__ == "__main__":
    main()
