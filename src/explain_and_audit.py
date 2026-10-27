"""Stage 4: reason codes, fairness report and model card.

    python -m src.explain_and_audit
    python -m src.explain_and_audit --sample --no-tracking   # CI smoke test

Works on the stage-3 test half (applicants never used for training, model
selection, calibration or pricing), with the stage-3 price grid applied.
Writes reports/audit/ (charts committed, tables not) and docs/model_card.md.
"""
from __future__ import annotations

import argparse
import json
import logging

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import load_config, project_path  # noqa: E402
from src.db import connect, query_df  # noqa: E402
from src.explain import BASE, champion_contributions, global_importance, readable, reason_codes  # noqa: E402
from src.fairness import age_groups, fairness_flags, group_metrics  # noqa: E402
from src.model_card import render  # noqa: E402
from src.model_data import ID_COL, holdout_halves, holdout_split, load_model_table, training_rows  # noqa: E402
from src.pricing import apply_grid  # noqa: E402
from src.scoring import champion_probability, load_bundle  # noqa: E402
from src.tracking import Tracker  # noqa: E402

log = logging.getLogger("explain_and_audit")


def run(sample: bool = False, tracking: bool = True) -> dict:
    cfg = load_config()
    ecfg, fcfg = cfg["explain"], cfg["fairness"]
    seed = cfg["project"]["random_seed"]
    suffix = "_sample" if sample else ""
    labels = ecfg["labels"]
    rng = np.random.default_rng(seed)

    # ---------- the stage-3 test half ----------
    table = load_model_table(cfg, sample)
    rows, y = training_rows(table)
    train_idx, holdout_idx = holdout_split(y, cfg["baseline"]["test_size"], seed)
    models_dir = project_path(cfg["models"]["models_dir"])
    bundle = load_bundle(models_dir / f"stage2_models{suffix}.joblib")
    pricing = load_bundle(models_dir / f"pricing{suffix}.joblib")

    encoder, columns = bundle["encoder"], bundle["columns"]
    X_all = encoder.transform(rows)[columns]
    y_ho = y.iloc[holdout_idx].to_numpy()
    _, test_pos = holdout_halves(y_ho, cfg["pricing"]["calibration_share"], seed)
    test_rows = holdout_idx[test_pos]
    X_test = X_all.iloc[test_rows].reset_index(drop=True)
    y_test = y.iloc[test_rows].to_numpy()
    ids = rows.iloc[test_rows][ID_COL].to_numpy()

    raw = champion_probability(bundle, X_test)
    pd_test = pricing["calibrators"]["champion"].transform(raw)
    bands, rates = apply_grid(pd_test, pricing["grid"])
    approved = ~np.isnan(rates)
    log.info("Test half: %s applicants, %.1f%% approved by the price grid", f"{len(y_test):,}", 100 * approved.mean())

    # ---------- 1. explanations ----------
    log.info("Computing TreeSHAP contributions...")
    contrib = champion_contributions(bundle, X_test)
    log_odds = contrib.sum(axis=1).to_numpy()
    p_from_contrib = 1 / (1 + np.exp(-log_odds))
    additivity = {
        "mean_abs_probability_diff": float(np.mean(np.abs(p_from_contrib - raw))),
        "max_abs_probability_diff": float(np.max(np.abs(p_from_contrib - raw))),
        "correlation": float(np.corrcoef(p_from_contrib, raw)[0, 1]),
    }
    log.info("Contributions reproduce champion probabilities: mean gap %.4f, correlation %.5f",
             additivity["mean_abs_probability_diff"], additivity["correlation"])

    typical = X_all.iloc[train_idx].median(numeric_only=True)
    categories = {c: encoder.categories_[c] for c in encoder.columns}
    declined = np.flatnonzero(~approved)
    approved_pos = np.flatnonzero(approved)
    n_extra = max(0, min(ecfg["sample_size"] - len(declined), len(approved_pos)))
    chosen = np.sort(np.concatenate([declined, rng.choice(approved_pos, n_extra, replace=False)]))
    reasons = reason_codes(contrib.iloc[chosen], X_test.iloc[chosen], typical, labels,
                           top_k=ecfg["top_reasons"], categories=categories)
    reasons.insert(0, ID_COL, ids[chosen])
    reasons.insert(1, "pd", np.round(pd_test[chosen], 4))
    reasons.insert(2, "band", bands[chosen])
    reasons.insert(3, "decision", np.where(approved[chosen], "approve", "decline"))
    reasons.insert(4, "rate", np.round(rates[chosen], 4))
    importance = global_importance(contrib, labels)

    example = None
    if len(declined):
        example = declined[np.argsort(pd_test[declined])[len(declined) // 2]]  # median-risk decline
    example_reasons = (reason_codes(contrib.iloc[[example]], X_test.iloc[[example]], typical, labels,
                                    top_k=ecfg["top_reasons"], categories=categories).iloc[0].tolist()
                       if example is not None else [])

    # ---------- 2. fairness ----------
    warehouse = project_path(cfg["data"]["warehouse_path"])
    if sample:
        warehouse = warehouse.with_name("sample_warehouse.duckdb")
    con = connect(warehouse, read_only=True)
    try:
        protected = query_df(con, "SELECT sk_id_curr, gender, age_years FROM protected_attributes")
    finally:
        con.close()
    protected = protected.set_index(ID_COL).reindex(ids)
    groups = {
        "gender": protected["gender"].fillna("unknown").astype(str).to_numpy(),
        "age_group": age_groups(protected["age_years"], fcfg["age_bins"], fcfg["age_labels"]).to_numpy(),
    }
    # The synthetic sample is tiny, so lower the minimum group size there; this
    # keeps CI exercising the fairness code. Real runs use the configured minimum.
    min_size = 30 if sample else fcfg["min_group_size"]
    fairness, flags = {}, []
    for attribute, values in groups.items():
        order = fcfg["age_labels"] + ["unknown"] if attribute == "age_group" else None
        fairness[attribute] = group_metrics(values, y_test, pd_test, approved, rates, min_size, order=order)
        flags += fairness_flags(fairness[attribute], attribute, fcfg["thresholds"])

    # ---------- 3. save ----------
    out_dir = project_path(cfg["reports_dir"]) / f"audit{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    reasons.to_csv(out_dir / "reason_codes_sample.csv", index=False)
    importance.to_csv(out_dir / "feature_influence.csv", index=False)
    for attribute, t in fairness.items():
        t.to_csv(out_dir / f"fairness_by_{attribute}.csv", index=False)
    _plot_importance(importance, out_dir / "feature_influence.png")
    if example is not None:
        _plot_example(contrib.iloc[example], X_test.iloc[example], pd_test[example], bands[example], labels,
                      out_dir / "example_decline.png")
    _plot_fairness(fairness, fcfg["thresholds"]["adverse_impact_ratio"], out_dir / "fairness.png")

    stage2 = json.loads((project_path(cfg["reports_dir"]) / f"models{suffix}" / "summary.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(project_path(cfg["reports_dir"]) / f"models{suffix}" / "comparison.csv")
    stage3 = json.loads((project_path(cfg["reports_dir"]) / f"pricing{suffix}" / "summary.json").read_text(encoding="utf-8"))
    card = render(stage2=stage2, comparison=comparison, stage3=stage3, fairness=fairness, flags=flags,
                  importance=importance, example_reasons=example_reasons, n_features=len(columns),
                  n_train=len(train_idx), n_holdout=len(holdout_idx), n_test=len(test_rows), additivity=additivity)
    card_path = (out_dir / "model_card.md") if sample else (project_path("docs") / "model_card.md")
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card, encoding="utf-8")

    summary = {"additivity": additivity, "flags": flags, "example_decline_reasons": example_reasons,
               "top_features": importance.head(10)[["label", "share"]].to_dict(orient="records")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    tcfg = cfg["tracking"]
    tracker = Tracker(tracking and tcfg["enabled"], project_path(tcfg["db_path"]), "credit-risk-stage4" + suffix)
    with tracker.run("explain-and-audit"):
        tracker.metrics({"shap_mean_abs_probability_diff": additivity["mean_abs_probability_diff"],
                         "fairness_flags": len(flags)})
        for attribute, t in fairness.items():
            for r in t[t["reportable"]].itertuples():
                key = f"{attribute}_{str(r.group).replace('<', 'under').replace('+', 'plus').replace('-', 'to')}"
                tracker.metrics({f"{key}_air": r.adverse_impact_ratio, f"{key}_auc": r.auc,
                                 f"{key}_calibration_gap": r.calibration_gap})
        for f in list(out_dir.iterdir()) + [card_path]:
            tracker.artifact(f)

    pd.set_option("display.width", 200)
    log.info("\n=== Example: reasons for a declined applicant (PD %.1f%%) ===\n%s",
             100 * pd_test[example] if example is not None else float("nan"),
             "\n".join(f"  {i + 1}. {r}" for i, r in enumerate(example_reasons) if r))
    log.info("\n=== Most influential features ===\n%s",
             importance.head(10)[["label", "share"]].round(3).to_string(index=False))
    for attribute, t in fairness.items():
        cols = ["group", "applicants", "default_rate", "mean_predicted_pd", "calibration_gap", "auc",
                "approval_rate", "adverse_impact_ratio", "good_customer_approval_rate"]
        log.info("\n=== Fairness by %s ===\n%s", attribute, t[t["reportable"]][cols].round(4).to_string(index=False))
    log.info("\n=== Fairness warnings ===\n%s", "\n".join(f"  - {f}" for f in flags) or "  none")
    log.info("Model card written to %s; charts and tables in %s", card_path, out_dir)
    return summary


def _plot_importance(importance: pd.DataFrame, path) -> None:
    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["label"], 100 * top["share"], color="#4C78A8")
    ax.set_xlabel("Share of total influence on scores (%, mean |SHAP| on unseen applicants)")
    ax.set_title("What drives the risk scores", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_example(contrib_row: pd.Series, x_row: pd.Series, pd_value: float, band: str, labels: dict, path) -> None:
    c = contrib_row.drop(BASE)
    top = c.reindex(c.abs().sort_values(ascending=False).index[:8]).iloc[::-1]
    names = [f"{readable(f, labels)} = {x_row[f]:.3g}" if pd.notna(x_row[f]) else f"{readable(f, labels)} = missing"
             for f in top.index]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(names, top.values, color=["#E45756" if v > 0 else "#54A24B" for v in top.values])
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel("Contribution to risk (log-odds): red raises risk, green lowers it")
    ax.set_title(f"Why this applicant was declined: PD {100 * pd_value:.1f}%, band {band}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_fairness(fairness: dict, air_threshold: float, path) -> None:
    fig, axes = plt.subplots(1, len(fairness), figsize=(4.5 * len(fairness), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (attribute, t) in zip(axes, fairness.items()):
        t = t[t["reportable"]]
        x = np.arange(len(t))
        ax.bar(x - 0.2, 100 * t["approval_rate"], width=0.4, color="#4C78A8", label="Approval rate")
        ax.bar(x + 0.2, 100 * t["good_customer_approval_rate"], width=0.4, color="#9ECAE9",
               label="Approval rate, good customers")
        for xi, air in zip(x, t["adverse_impact_ratio"]):
            ax.annotate(f"AIR {air:.2f}", (xi, 2), ha="center", fontsize=8,
                        color="#E45756" if air < air_threshold else "white")
        ax.set_xticks(x, t["group"].astype(str))
        ax.set_title(f"By {attribute.replace('_', ' ')}", fontsize=10)
    axes[0].set_ylabel("Approved (%)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle(f"Approval by group on unseen applicants (AIR below {air_threshold} shown in red)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: explainability, fairness and model card")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for noisy in ("alembic", "mlflow", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run(sample=args.sample, tracking=not args.no_tracking)


if __name__ == "__main__":
    main()
