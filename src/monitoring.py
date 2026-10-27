"""Population Stability Index (PSI): has the population drifted?

PSI compares how a variable is distributed now vs at training time:
    PSI = sum over bins of (now% - then%) * ln(now% / then%)
Bins are deciles of the reference data; missing values get their own bin.

Industry rule of thumb (credit risk):
    below 0.10   stable
    0.10 - 0.25  moderate shift: investigate
    above 0.25   significant shift: review, likely retrain
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4
STABLE, ALERT = 0.10, 0.25


def psi(reference, current, n_bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref_ok, cur_ok = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref_ok) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref_ok, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    def shares(values, valid):
        counts = np.histogram(valid, bins=edges)[0] if len(edges) > 1 else np.array([len(valid)])
        counts = np.append(counts, np.isnan(values).sum())  # missing-value bin
        return np.clip(counts / max(len(values), 1), EPS, None)

    e, a = shares(ref, ref_ok), shares(cur, cur_ok)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    return "stable" if value < STABLE else ("investigate" if value < ALERT else "ALERT")


def psi_report(reference: dict[str, np.ndarray], current: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, ref in reference.items():
        if name not in current:
            continue
        value = psi(ref, current[name])
        rows.append({"variable": name, "psi": round(value, 4), "status": status(value)})
    return pd.DataFrame(rows)
