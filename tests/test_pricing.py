from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.calibration import Calibrator, calibration_table, choose_method, expected_calibration_error
from src.pricing import (PricingAssumptions, apply_grid, assign_bands, build_grid, profit_curve,
                         realised_profit, required_rate)

A = PricingAssumptions(lgd=0.45, cost_of_funds=0.08, operating_cost=0.04, target_margin=0.03, max_rate=0.36)
LABELS = ["A", "B", "C"]
BOUNDS = [0.05, 0.20, 1.0]


# ---------- pricing equation ----------

def test_rate_for_zero_risk_is_costs_plus_margin():
    assert float(required_rate(0.0, A)) == pytest.approx(0.15)


def test_rate_hand_calculation():
    # (0.03 + 0.08 + 0.04 + 0.10 * 0.45) / 0.90 = 0.195 / 0.90
    assert float(required_rate(0.10, A)) == pytest.approx(0.195 / 0.90)


@pytest.mark.parametrize("pd_", [0.01, 0.08, 0.25])
def test_priced_loan_earns_exactly_the_target_margin_in_expectation(pd_):
    r = float(required_rate(pd_, A))
    expected_profit = (1 - pd_) * r - pd_ * A.lgd - A.cost_of_funds - A.operating_cost
    assert expected_profit == pytest.approx(A.target_margin)


def test_realised_profit_hand_calculation():
    # repaid at 20%: 1000 * (0.20 - 0.12) = 80 ; defaulted: 1000 * (-0.45 - 0.12) = -570
    p = realised_profit([1000, 1000], [0, 1], [0.20, 0.20], A)
    assert p.tolist() == pytest.approx([80.0, -570.0])


# ---------- bands and grid ----------

def test_band_boundaries_are_inclusive_upper():
    bands = assign_bands(np.array([0.0, 0.05, 0.0500001, 0.2, 0.9]), BOUNDS, LABELS)
    assert bands.tolist() == ["A", "A", "B", "B", "C"]


def test_grid_declines_bands_priced_above_the_cap():
    pd_ = np.array([0.02] * 50 + [0.10] * 30 + [0.40] * 20)
    y = np.zeros(100)
    grid = build_grid(pd_, y, BOUNDS, LABELS, A).set_index("band")
    assert grid.loc["A", "decision"] == "approve"
    assert grid.loc["B", "decision"] == "approve"
    assert grid.loc["C", "decision"] == "decline"     # (0.15 + 0.18) / 0.60 = 0.55 > 0.36
    assert grid.loc["C", "rate"] == pytest.approx((0.15 + 0.40 * 0.45) / 0.60, abs=1e-4)
    assert grid["share"].sum() == pytest.approx(1.0)


def test_apply_grid_gives_no_rate_to_declined_applicants():
    pd_ = np.array([0.02] * 50 + [0.10] * 30 + [0.40] * 20)
    grid = build_grid(pd_, np.zeros(100), BOUNDS, LABELS, A)
    bands, rates = apply_grid(np.array([0.01, 0.15, 0.5]), grid)
    assert bands.tolist() == ["A", "B", "C"]
    assert not np.isnan(rates[0]) and not np.isnan(rates[1]) and np.isnan(rates[2])


# ---------- profit curve ----------

def test_better_ranking_approves_fewer_defaulters():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.1).astype(int)
    good_model = y + rng.normal(0, 0.3, 5000)      # ranks defaulters high
    bad_model = rng.normal(size=5000)               # random ranking
    ones = np.ones(5000)
    rate = np.full(5000, 0.25)
    good = profit_curve(good_model, ones, y, rate, [0.8], A).iloc[0]
    bad = profit_curve(bad_model, ones, y, rate, [0.8], A).iloc[0]
    assert good["approved"] == bad["approved"] == 4000
    assert good["default_rate_approved"] < bad["default_rate_approved"]
    assert good["profit"] > bad["profit"]


# ---------- calibration ----------

@pytest.fixture
def overconfident():
    rng = np.random.default_rng(1)
    true_logit = rng.normal(-2.5, 1.0, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-true_logit))).astype(int)
    raw = 1 / (1 + np.exp(-(2.0 * true_logit + 2.0)))   # right ranking, wrong probabilities
    return raw, y


def test_calibration_fixes_overconfident_probabilities(overconfident):
    raw, y = overconfident
    before = expected_calibration_error(y, raw)
    after = expected_calibration_error(y, Calibrator("platt").fit(raw, y).transform(raw))
    assert after < before / 3


def test_cross_validation_rejects_no_calibration_when_it_is_needed(overconfident):
    raw, y = overconfident
    method, table = choose_method(raw, y, n_folds=3, seed=0)
    assert method in ("platt", "isotonic")
    assert set(table["method"]) == {"none", "platt", "isotonic"}


def test_calibration_preserves_ranking(overconfident):
    raw, y = overconfident
    q = Calibrator("platt").fit(raw, y).transform(raw)
    assert pd.Series(raw).corr(pd.Series(q), method="spearman") > 0.999


def test_calibration_table_covers_every_applicant(overconfident):
    raw, y = overconfident
    table = calibration_table(y, raw, n_bins=10)
    assert len(table) == 10 and table["n"].sum() == len(y)
