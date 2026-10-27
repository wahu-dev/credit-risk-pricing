"""Risk-based loan pricing and profit analysis.

One-period pricing equation. For each unit lent, the lender
  earns the interest rate r on borrowers who repay      (probability 1 - PD)
  loses LGD on borrowers who default                    (probability PD)
  pays cost of funds and operating cost on everyone.

Expected profit per unit = (1 - PD) * r - PD * LGD - funding - opex.
Setting that equal to the target margin and solving for r:

    r = (margin + funding + opex + PD * LGD) / (1 - PD)

Simplifications (stated in the README, defended in interviews):
  * one period: the rate is earned on the full amount for the term;
  * LGD is a single assumption, not modelled per applicant;
  * customers accept the offered price: no competitor response or adverse
    selection, which is why flat vs risk-based is compared on margin
    accuracy rather than raw profit;
  * TARGET in this dataset means "payment difficulties", a proxy for default.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PricingAssumptions:
    lgd: float              # share of exposure lost when a borrower defaults
    cost_of_funds: float    # annual funding cost per unit lent
    operating_cost: float   # annual servicing cost per unit lent
    target_margin: float    # required profit per unit lent
    max_rate: float         # highest rate the lender will offer (policy / market cap)

    def to_dict(self) -> dict:
        return asdict(self)


def required_rate(pd_: np.ndarray | float, a: PricingAssumptions):
    pd_ = np.asarray(pd_, dtype=float)
    return (a.target_margin + a.cost_of_funds + a.operating_cost + pd_ * a.lgd) / (1 - pd_)


def assign_bands(pd_: np.ndarray, upper_bounds: list[float], labels: list[str]) -> np.ndarray:
    """Band by PD: labels[i] covers PDs up to upper_bounds[i] (last bound = 1.0)."""
    if len(upper_bounds) != len(labels):
        raise ValueError("Need one upper bound per band label.")
    idx = np.searchsorted(np.asarray(upper_bounds), np.asarray(pd_, dtype=float), side="left")
    return np.asarray(labels, dtype=object)[np.minimum(idx, len(labels) - 1)]


def build_grid(pd_: np.ndarray, y: np.ndarray, upper_bounds: list[float], labels: list[str],
               a: PricingAssumptions) -> pd.DataFrame:
    """The price list: one row per band, priced at the band's average PD.
    Built on the calibration set, then applied unchanged to new applicants."""
    bands = assign_bands(pd_, upper_bounds, labels)
    df = pd.DataFrame({"band": bands, "pd": pd_, "y": y})
    lower = [0.0] + list(upper_bounds[:-1])
    rows = []
    for label, lo, hi in zip(labels, lower, upper_bounds):
        part = df[df["band"] == label]
        mean_pd = float(part["pd"].mean()) if len(part) else (lo + hi) / 2
        rate = float(required_rate(mean_pd, a))
        rows.append({
            "band": label, "pd_from": lo, "pd_to": hi, "applicants": int(len(part)),
            "share": round(len(part) / len(df), 4), "mean_pd": round(mean_pd, 4),
            "observed_default_rate": round(float(part["y"].mean()), 4) if len(part) else None,
            "expected_loss_rate": round(mean_pd * a.lgd, 4),
            "rate": round(rate, 4),
            "decision": "approve" if rate <= a.max_rate else "decline",
        })
    return pd.DataFrame(rows)


def apply_grid(pd_: np.ndarray, grid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Band and offered rate for each applicant (NaN rate = declined)."""
    bands = assign_bands(pd_, grid["pd_to"].tolist(), grid["band"].tolist())
    lookup = grid.set_index("band")
    rates = np.array([lookup.at[b, "rate"] if lookup.at[b, "decision"] == "approve" else np.nan for b in bands])
    return bands, rates


def realised_profit(exposure, y, rate, a: PricingAssumptions) -> np.ndarray:
    """Profit per loan on actual outcomes: interest if repaid, LGD loss if
    defaulted, minus funding and operating cost either way."""
    exposure, y, rate = (np.asarray(v, dtype=float) for v in (exposure, y, rate))
    return exposure * ((1 - y) * rate - y * a.lgd - a.cost_of_funds - a.operating_cost)


def band_performance(grid: pd.DataFrame, bands, exposure, y, rate, a: PricingAssumptions) -> pd.DataFrame:
    """Pricing check on unseen applicants: did each band earn its target margin?"""
    profit = realised_profit(exposure, y, np.nan_to_num(rate), a)
    df = pd.DataFrame({"band": bands, "exposure": exposure, "y": y, "profit": profit, "approved": ~np.isnan(rate)})
    out = []
    for row in grid.itertuples():
        part = df[(df["band"] == row.band) & df["approved"]]
        out.append({
            "band": row.band, "decision": row.decision, "rate": row.rate, "loans": int(len(part)),
            "predicted_pd": row.mean_pd,
            "observed_default_rate": round(float(part["y"].mean()), 4) if len(part) else None,
            "realised_margin": round(float(part["profit"].sum() / part["exposure"].sum()), 4) if len(part) else None,
            "target_margin": a.target_margin,
        })
    return pd.DataFrame(out)


def profit_curve(pd_, exposure, y, rate, approval_rates, a: PricingAssumptions) -> pd.DataFrame:
    """Approve the lowest-PD share of applicants at each approval rate and add
    up realised profit. `rate` is the price per applicant (NaN = never approve).
    Ranking quality drives this curve: a better model approves fewer future
    defaulters at the same volume."""
    pd_, exposure, y, rate = (np.asarray(v, dtype=float) for v in (pd_, exposure, y, rate))
    profit = realised_profit(exposure, y, np.nan_to_num(rate), a)
    order = np.argsort(pd_, kind="stable")
    rows = []
    for share in approval_rates:
        chosen = order[: int(round(share * len(pd_)))]
        chosen = chosen[~np.isnan(rate[chosen])]
        rows.append({
            "approval_rate_target": round(float(share), 4),
            "approval_rate_actual": round(len(chosen) / len(pd_), 4),
            "approved": int(len(chosen)),
            "default_rate_approved": round(float(y[chosen].mean()), 5) if len(chosen) else None,
            "profit": float(profit[chosen].sum()),
            "profit_per_loan": float(profit[chosen].mean()) if len(chosen) else 0.0,
        })
    return pd.DataFrame(rows)
