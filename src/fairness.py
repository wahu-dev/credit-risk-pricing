"""Fairness checks by protected attribute (gender, age group).

The model never sees these attributes (stage 1 keeps them in a separate table).
That alone does not guarantee fair outcomes, because other features can act as
proxies, so we measure outcomes per group on unseen applicants:

  approval rate           share approved
  adverse impact ratio    group approval rate / highest group's rate.
                          Below 0.8 is the common "four-fifths" warning level.
  good-customer approval  approval rate among people who did NOT default:
                          are equally creditworthy people treated alike?
  AUC within group        does the model rank risk equally well for everyone?
  calibration gap         mean predicted PD minus actual default rate. Positive
                          means the group is charged for more risk than it has.

Different approval rates can reflect real differences in risk; a model that is
less accurate, or systematically over-predicts risk, for one group is the
stronger warning sign. Both are reported; neither is hidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def age_groups(age_years: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(age_years, bins=bins, labels=labels, right=False).astype("object").fillna("unknown")


def group_metrics(group: pd.Series, y, pd_, approved, rate, min_group_size: int,
                  order: list[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame({"group": np.asarray(group, dtype=object), "y": np.asarray(y),
                       "pd": np.asarray(pd_, dtype=float), "approved": np.asarray(approved, dtype=bool),
                       "rate": np.asarray(rate, dtype=float)})
    rows = []
    for name, g in df.groupby("group", sort=True):
        good = g[g["y"] == 0]
        auc = roc_auc_score(g["y"], g["pd"]) if g["y"].nunique() == 2 else np.nan
        rows.append({
            "group": name,
            "applicants": int(len(g)),
            "share": len(g) / len(df),
            "default_rate": g["y"].mean(),
            "mean_predicted_pd": g["pd"].mean(),
            "calibration_gap": g["pd"].mean() - g["y"].mean(),
            "auc": auc,
            "approval_rate": g["approved"].mean(),
            "good_customer_approval_rate": good["approved"].mean() if len(good) else np.nan,
            "default_rate_if_approved": g.loc[g["approved"], "y"].mean() if g["approved"].any() else np.nan,
            "mean_rate_offered": g.loc[g["approved"], "rate"].mean() if g["approved"].any() else np.nan,
            "reportable": len(g) >= min_group_size,
        })
    out = pd.DataFrame(rows)
    if order:  # keep a natural order (e.g. youngest to oldest) instead of alphabetical
        rank = {g: i for i, g in enumerate(order)}
        out = out.sort_values("group", key=lambda s: s.map(lambda g: rank.get(g, len(rank)))).reset_index(drop=True)
    reportable = out[out["reportable"]]
    top = reportable["approval_rate"].max() if len(reportable) else np.nan
    out["adverse_impact_ratio"] = np.where(out["reportable"], out["approval_rate"] / top, np.nan)
    return out


def fairness_flags(table: pd.DataFrame, attribute: str, thresholds: dict) -> list[str]:
    """Plain-language warnings for reportable groups that cross a threshold."""
    flags = []
    t = table[table["reportable"]]
    if t.empty:
        return flags
    best_auc = t["auc"].max()
    for row in t.itertuples():
        if row.adverse_impact_ratio < thresholds["adverse_impact_ratio"]:
            flags.append(f"{attribute}={row.group}: approval rate {row.approval_rate:.1%} is "
                         f"{row.adverse_impact_ratio:.2f}x the highest group (below the "
                         f"{thresholds['adverse_impact_ratio']} four-fifths level)")
        if abs(row.calibration_gap) > thresholds["calibration_gap"]:
            direction = "over" if row.calibration_gap > 0 else "under"
            flags.append(f"{attribute}={row.group}: risk {direction}-predicted by "
                         f"{abs(row.calibration_gap):.1%} points ({row.mean_predicted_pd:.1%} predicted "
                         f"vs {row.default_rate:.1%} actual)")
        if pd.notna(row.auc) and best_auc - row.auc > thresholds["auc_gap"]:
            flags.append(f"{attribute}={row.group}: model ranks risk less well (AUC {row.auc:.3f} "
                         f"vs best {best_auc:.3f})")
    return flags
