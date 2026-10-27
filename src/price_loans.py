"""Stage 3: calibrate PDs, build the risk-based price list, value the models in money.

    python -m src.price_loans
    python -m src.price_loans --sample --no-tracking     # CI smoke test

Uses the stage-2 holdout only (never seen in training or model selection):
  * calibration half: fit the calibrator, build the pricing grid
  * test half:        check calibration, check each band's realised margin,
                      and compare champion vs scorecard profit
"""
from __future__ import annotations

import argparse
import json
import logging

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402

from src.calibration import Calibrator, calibration_metrics, calibration_table, choose_method  # noqa: E402
from src.config import load_config, project_path  # noqa: E402
from src.model_data import holdout_halves, holdout_split, load_model_table, training_rows  # noqa: E402
from src.pricing import (PricingAssumptions, apply_grid, band_performance, build_grid,  # noqa: E402
                         profit_curve, required_rate)
from src.scoring import champion_probability, load_bundle, member_predictions  # noqa: E402
from src.tracking import Tracker  # noqa: E402

log = logging.getLogger("price_loans")
EXPOSURE_COL = "credit_amount"


def run(sample: bool = False, tracking: bool = True) -> dict:
    cfg = load_config()
    pcfg = cfg["pricing"]
    seed = cfg["project"]["random_seed"]
    a = PricingAssumptions(**pcfg["assumptions"])
    labels, bounds = pcfg["bands"]["labels"], pcfg["bands"]["upper_pd"]
    suffix = "_sample" if sample else ""

    # ---------- holdout, exactly as in stages 1 and 2 ----------
    table = load_model_table(cfg, sample)
    rows, y = training_rows(table)
    _, holdout_idx = holdout_split(y, cfg["baseline"]["test_size"], seed)
    bundle = load_bundle(project_path(cfg["models"]["models_dir"]) / f"stage2_models{suffix}.joblib")
    X_ho = bundle["encoder"].transform(rows.iloc[holdout_idx])[bundle["columns"]].reset_index(drop=True)
    y_ho = y.iloc[holdout_idx].to_numpy()
    exposure = rows.iloc[holdout_idx][EXPOSURE_COL].to_numpy(dtype=float)

    raw = {
        "champion": champion_probability(bundle, X_ho),
        "scorecard": member_predictions(bundle, X_ho)["scorecard"].to_numpy(),
    }
    cal_idx, test_idx = holdout_halves(y_ho, pcfg["calibration_share"], seed)
    log.info("Holdout %s applicants: %s to calibrate and price, %s to test",
             f"{len(y_ho):,}", f"{len(cal_idx):,}", f"{len(test_idx):,}")

    # ---------- 1. calibration ----------
    calibrated, calibrators, calib_report = {}, {}, {}
    for name, p in raw.items():
        method, cv_table = choose_method(p[cal_idx], y_ho[cal_idx], seed=seed)
        calibrators[name] = Calibrator(method).fit(p[cal_idx], y_ho[cal_idx])
        calibrated[name] = calibrators[name].transform(p)
        calib_report[name] = {
            "method": method,
            "cv_comparison": cv_table.to_dict(orient="records"),
            "test_before": calibration_metrics(y_ho[test_idx], p[test_idx]),
            "test_after": calibration_metrics(y_ho[test_idx], calibrated[name][test_idx]),
        }
        log.info("%s calibration: %s | test ECE %.4f -> %.4f", name, method,
                 calib_report[name]["test_before"]["ece"], calib_report[name]["test_after"]["ece"])

    pd_champ = calibrated["champion"]

    # ---------- 2. pricing grid (built on calibration half, checked on test half) ----------
    grid = build_grid(pd_champ[cal_idx], y_ho[cal_idx], bounds, labels, a)
    bands_test, rates_test = apply_grid(pd_champ[test_idx], grid)
    performance = band_performance(grid, bands_test, exposure[test_idx], y_ho[test_idx], rates_test, a)

    # ---------- 3. model value: same flat price, same approval rate ----------
    flat_rate = float(required_rate(pd_champ[cal_idx].mean(), a))
    steps = pcfg["approval_rates"]
    approval_rates = np.round(np.arange(steps["start"], steps["stop"] + 1e-9, steps["step"]), 4)
    curves = {}
    for name in ("champion", "scorecard"):
        curves[f"{name}_flat"] = profit_curve(calibrated[name][test_idx], exposure[test_idx], y_ho[test_idx],
                                              np.full(len(test_idx), flat_rate), approval_rates, a)
    curves["champion_risk_based"] = profit_curve(pd_champ[test_idx], exposure[test_idx], y_ho[test_idx],
                                                 rates_test, approval_rates, a)

    n_test = len(test_idx)
    comparison = []
    for share in pcfg["compare_at"]:
        c = curves["champion_flat"].set_index("approval_rate_target").loc[round(share, 4)]
        s = curves["scorecard_flat"].set_index("approval_rate_target").loc[round(share, 4)]
        comparison.append({
            "approval_rate": share,
            "champion_default_rate": c["default_rate_approved"],
            "scorecard_default_rate": s["default_rate_approved"],
            "defaults_avoided_per_10k_applicants": round((s["default_rate_approved"] - c["default_rate_approved"])
                                                         * c["approved"] / n_test * 10_000, 1),
            "extra_profit": round(c["profit"] - s["profit"], 0),
            "extra_profit_per_10k_applicants": round((c["profit"] - s["profit"]) / n_test * 10_000, 0),
        })
    comparison = pd.DataFrame(comparison)

    # ---------- 4. save ----------
    out_dir = project_path(cfg["reports_dir"]) / f"pricing{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out_dir / "pricing_grid.csv", index=False)
    performance.to_csv(out_dir / "band_performance_test.csv", index=False)
    comparison.to_csv(out_dir / "champion_vs_scorecard_value.csv", index=False)
    pd.concat([c.assign(curve=k) for k, c in curves.items()]).to_csv(out_dir / "profit_curves.csv", index=False)
    calibration_table(y_ho[test_idx], pd_champ[test_idx]).to_csv(out_dir / "calibration_table_test.csv", index=False)
    _plot_calibration(y_ho[test_idx], raw["champion"][test_idx], pd_champ[test_idx],
                      calib_report["champion"]["method"], out_dir / "calibration.png")
    _plot_profit(curves, flat_rate, out_dir / "profit_vs_approval.png")
    _plot_band_margins(performance, a.target_margin, out_dir / "band_margins.png")

    summary = {
        "assumptions": a.to_dict(),
        "flat_rate_for_comparison": round(flat_rate, 4),
        "calibration": calib_report,
        "pricing_grid": grid.to_dict(orient="records"),
        "band_performance_test": performance.to_dict(orient="records"),
        "champion_vs_scorecard": comparison.to_dict(orient="records"),
        "caveats": [
            "TARGET means payment difficulties, used as a proxy for default.",
            "One-period pricing with a single LGD assumption; amounts in the dataset's currency units.",
            "Customers are assumed to accept the offered price (no competitor response or adverse selection).",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    joblib.dump({"calibrators": calibrators, "grid": grid, "assumptions": a.to_dict()},
                project_path(cfg["models"]["models_dir"]) / f"pricing{suffix}.joblib")

    tcfg = cfg["tracking"]
    tracker = Tracker(tracking and tcfg["enabled"], project_path(tcfg["db_path"]), "credit-risk-stage3" + suffix)
    with tracker.run("pricing"):
        tracker.params(a.to_dict())
        tracker.params({f"{k}_calibration": v["method"] for k, v in calib_report.items()})
        tracker.metrics({f"{k}_test_ece": v["test_after"]["ece"] for k, v in calib_report.items()})
        for row in comparison.itertuples():
            tracker.metrics({f"extra_profit_per_10k_at_{int(row.approval_rate * 100)}pct": row.extra_profit_per_10k_applicants})
        for f in out_dir.iterdir():
            tracker.artifact(f)

    pd.set_option("display.width", 170)
    log.info("\n=== Pricing grid (built on calibration half) ===\n%s", grid.to_string(index=False))
    log.info("\n=== Pricing check on unseen applicants: realised vs target margin ===\n%s", performance.to_string(index=False))
    log.info("\n=== What the better model is worth (same flat rate %.1f%%) ===\n%s", 100 * flat_rate, comparison.to_string(index=False))
    log.info("Results and charts in %s", out_dir)
    return summary


def _plot_calibration(y, p_before, p_after, method, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for p, label, style in ((p_before, "before", "o--"), (p_after, f"after ({method})", "o-")):
        t = calibration_table(y, p)
        ax.plot(100 * t["mean_predicted"], 100 * t["observed_rate"], style, label=label)
    top = 100 * max(np.max(calibration_table(y, p_after)["observed_rate"]), np.max(calibration_table(y, p_after)["mean_predicted"])) * 1.1
    ax.plot([0, top], [0, top], color="grey", lw=1, label="perfect calibration")
    ax.set_xlabel("Predicted default rate (%)")
    ax.set_ylabel("Observed default rate (%)")
    ax.set_title("Champion calibration on unseen applicants (deciles)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_profit(curves: dict, flat_rate: float, path) -> None:
    """Champion vs scorecard at the SAME price: the gap is the money value of
    better ranking. (Flat vs risk-based is not plotted: in a static simulation a
    flat price looks profitable by overcharging low-risk customers, who in
    reality would leave; the grid is judged on band margins instead.)"""
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, label, style in (("champion_flat", "Champion", "-"), ("scorecard_flat", "Scorecard", "--")):
        c = curves[key]
        ax.plot(100 * c["approval_rate_actual"], c["profit"] / 1e6, style, lw=2, label=label)
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel("Applicants approved, lowest predicted risk first (%)")
    ax.set_ylabel("Realised profit (millions, dataset currency)")
    ax.set_title(f"Same price ({100 * flat_rate:.1f}%), better ranking: profit on unseen applicants")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_band_margins(performance: pd.DataFrame, target: float, path) -> None:
    """Pricing check: realised margin per band on unseen applicants vs target."""
    approved = performance[performance["decision"] == "approve"].dropna(subset=["realised_margin"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(approved))
    ax.bar(x, 100 * approved["realised_margin"], color="#4C78A8", label="Realised margin")
    ax.axhline(100 * target, color="#E45756", lw=2, ls="--", label=f"Target margin ({100 * target:.0f}%)")
    ax.axhline(0, color="grey", lw=1)
    ax.set_xticks(x, [f"{b}\nrate {100 * r:.1f}%\n{n:,} loans"
                      for b, r, n in zip(approved["band"], approved["rate"], approved["loans"])], fontsize=8)
    ax.set_ylabel("Margin on amount lent (%)")
    ax.set_title("Risk-based price grid: realised vs target margin by band (unseen applicants)", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: calibration and risk-based pricing")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for noisy in ("alembic", "mlflow", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run(sample=args.sample, tracking=not args.no_tracking)


if __name__ == "__main__":
    main()
