"""Drift monitoring with the Population Stability Index.

    python -m src.monitor --simulate          # stable vs economic-shock scenario
    python -m src.monitor --from-log          # applicants actually scored by the API

Reference = the population the model was validated on (stored in the serving
bundle). Current = new applicants. A PSI above 0.25 on the score or a key
feature means the model is seeing a population it was not built for.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import load_config, project_path  # noqa: E402
from src.model_data import holdout_halves, holdout_split, load_model_table, training_rows  # noqa: E402
from src.monitoring import ALERT, STABLE, psi_report  # noqa: E402
from src.serving import ScoringService  # noqa: E402

log = logging.getLogger("monitor")

# An illustrative downturn: for most new applicants, external credit scores
# fall, repayment burdens rise and late payments become more common.
SHOCK_SHARE = 0.7
SHOCK = {
    "ext_source_mean": 0.75, "ext_source_1": 0.75, "ext_source_2": 0.75, "ext_source_3": 0.75,
    "annuity_to_income_ratio": 1.3, "inst_late_rate": 2.0, "inst_late_rate_365d": 2.0,
}


def _current(service: ScoringService, records: list[dict]) -> dict[str, np.ndarray]:
    results = service.score(records, with_reasons=False)
    current = {"pd": np.array([r["pd"] for r in results])}
    for f in service.serving["monitored_features"]:
        current[f] = np.array([np.nan if r.get(f) is None else float(r[f]) for r in records])
    current["_approval_rate"] = np.mean([r["decision"] == "approve" for r in results])
    return current


def simulate(service: ScoringService, sample: bool) -> pd.DataFrame:
    cfg = load_config()
    seed = cfg["project"]["random_seed"]
    rows, y = training_rows(load_model_table(cfg, sample))
    _, holdout_idx = holdout_split(y, cfg["baseline"]["test_size"], seed)
    _, test_pos = holdout_halves(y.iloc[holdout_idx].to_numpy(), cfg["pricing"]["calibration_share"], seed)
    test = rows.iloc[holdout_idx[test_pos]][service.columns]
    records = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in test.to_dict(orient="records")]

    rng = np.random.default_rng(seed)
    shocked = []
    for r in records:
        r = dict(r)
        if rng.random() < SHOCK_SHARE:
            for f, factor in SHOCK.items():
                if f in r and r[f] is not None:
                    r[f] = float(r[f]) * factor
        shocked.append(r)

    reference = service.serving["reference"]
    scenarios = {"stable_month": _current(service, records), "economic_shock": _current(service, shocked)}
    report = None
    for name, current in scenarios.items():
        part = psi_report(reference, current).rename(columns={"psi": f"psi_{name}", "status": f"status_{name}"})
        report = part if report is None else report.merge(part, on="variable")
        log.info("%s: %s applicants, mean PD %.2f%%, approval rate %.1f%%", name, f"{len(current['pd']):,}",
                 100 * current["pd"].mean(), 100 * current["_approval_rate"])
    return report


def from_log(service: ScoringService, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Score some applicants through the API first.")
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(entries) < 100:
        log.warning("Only %d scored applicants in the log; PSI is unreliable below a few hundred.", len(entries))
    current = {"pd": np.array([e["pd"] for e in entries], dtype=float)}
    for f in service.serving["monitored_features"]:
        current[f] = np.array([np.nan if e["features"].get(f) is None else float(e["features"][f])
                               for e in entries], dtype=float)
    return psi_report(service.serving["reference"], current)


def _plot(report: pd.DataFrame, labels: dict, path: Path) -> None:
    psi_cols = [c for c in report.columns if c.startswith("psi_")]
    names = ["Score (PD)" if v == "pd" else labels.get(v, v) for v in report["variable"]]
    y = np.arange(len(report))
    h = 0.8 / len(psi_cols)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(report) + 1.8))
    colors = ["#4C78A8", "#E45756", "#54A24B"]
    for k, col in enumerate(psi_cols):
        ax.barh(y + (k - (len(psi_cols) - 1) / 2) * h, report[col], height=h, color=colors[k % 3],
                label=col.removeprefix("psi_").replace("_", " "))
    ax.axvline(STABLE, color="grey", ls=":", lw=1.5, label=f"Investigate ({STABLE})")
    ax.axvline(ALERT, color="black", ls="--", lw=1.5, label=f"Alert ({ALERT})")
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Drift monitoring: new applicants vs the validation population", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift monitoring (PSI)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--simulate", action="store_true", help="stable month vs economic-shock scenario")
    mode.add_argument("--from-log", nargs="?", const="logs/scores.jsonl", help="score log written by the API")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    cfg = load_config()
    suffix = "_sample" if args.sample else ""
    bundle_path = project_path(cfg["models"]["models_dir"]) / f"serving_bundle{suffix}.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"{bundle_path} not found. Run `python -m src.build_serving` first.")
    service = ScoringService(joblib.load(bundle_path))

    report = simulate(service, args.sample) if args.simulate else from_log(service, project_path(args.from_log))
    out_dir = project_path(cfg["reports_dir"]) / f"monitoring{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "psi_simulation" if args.simulate else "psi_from_log"
    report.to_csv(out_dir / f"{name}.csv", index=False)
    _plot(report, service.serving["labels"], out_dir / f"{name}.png")
    pd.set_option("display.width", 160)
    log.info("\n=== PSI report (stable < %.2f <= investigate < %.2f <= ALERT) ===\n%s",
             STABLE, ALERT, report.to_string(index=False))
    log.info("Report and chart in %s", out_dir)


if __name__ == "__main__":
    main()
