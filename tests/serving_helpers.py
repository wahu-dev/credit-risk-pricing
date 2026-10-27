"""A small, fully fitted serving bundle for API and service tests (no files needed)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.boosting import KINDS
from src.calibration import Calibrator
from src.ensemble import BoostedMember, ScorecardMember
from src.model_data import CategoryEncoder
from src.pricing import PricingAssumptions, build_grid
from src.scoring import champion_probability

COLUMNS = ["ext_source_mean", "inst_late_rate", "thin_file", "income_type"]


def make_serving(seed: int = 3) -> tuple[dict, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n = 2000
    raw = pd.DataFrame({
        "ext_source_mean": rng.uniform(0.05, 0.9, n),
        "inst_late_rate": rng.uniform(0, 0.6, n),
        "thin_file": (rng.random(n) < 0.1).astype("int64"),
        "income_type": rng.choice(["Working", "Pensioner", "Commercial associate"], n),
    })
    logit = -1.0 - 6.0 * raw["ext_source_mean"] + 4.0 * raw["inst_late_rate"] + 0.5 * raw["thin_file"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))

    encoder = CategoryEncoder(["income_type"]).fit(raw)
    X = encoder.transform(raw)[COLUMNS]
    members = {k: BoostedMember(k, {"learning_rate": 0.1}, 60, 0, COLUMNS).fit(X, y) for k in KINDS}
    members["scorecard"] = ScorecardMember(["ext_source_mean", "inst_late_rate"]).fit(X, y)
    bundle = {"champion": "average_boosted", "members": members, "encoder": encoder,
              "columns": COLUMNS, "stacker": None}

    p = champion_probability(bundle, X)
    calibrator = Calibrator("none").fit(p, y)
    a = PricingAssumptions(lgd=0.45, cost_of_funds=0.08, operating_cost=0.04, target_margin=0.03, max_rate=0.30)
    grid = build_grid(calibrator.transform(p), y.to_numpy(), [0.05, 0.15, 1.0], ["A", "B", "C"], a)

    serving = {
        "model_version": "test-1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bundle": bundle,
        "calibrator": calibrator,
        "grid": grid,
        "assumptions": a.to_dict(),
        "labels": {"ext_source_mean": "External credit score (average)",
                   "inst_late_rate": "Share of instalments paid late"},
        "top_reasons": 3,
        "categories": {"income_type": encoder.categories_["income_type"]},
        "typical": {k: float(v) for k, v in raw.median(numeric_only=True).items()},
        "dtypes": {c: str(raw[c].dtype) for c in COLUMNS},
        "example": {"ext_source_mean": 0.5, "inst_late_rate": 0.1, "thin_file": 0, "income_type": "Working"},
        "monitored_features": ["ext_source_mean", "inst_late_rate"],
        "reference": {"pd": calibrator.transform(p), "ext_source_mean": raw["ext_source_mean"].to_numpy(),
                      "inst_late_rate": raw["inst_late_rate"].to_numpy()},
        "versions": {},
    }
    return serving, raw, y


SAFE = {"ext_source_mean": 0.85, "inst_late_rate": 0.0, "thin_file": 0, "income_type": "Working"}
RISKY = {"ext_source_mean": 0.06, "inst_late_rate": 0.6, "thin_file": 1, "income_type": "Working"}
