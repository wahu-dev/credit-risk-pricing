# Decision log

Modelling decisions, the evidence behind them, and their effect.
Newest first. Evidence files are produced by `python -m src.investigate`.

---

## 2026-10-27 · Two baseline features had the opposite sign to business expectation

**Trigger.** The stage 1 baseline (validation AUC 0.742) flagged two coefficients
pointing the "wrong" way: `credit_to_annuity_ratio` and `bureau_worst_dpd_12m`.

**Method.** For each feature: default rate by decile with missing values as their
own band, direction on its own (univariate AUC), and variance inflation factors
for all baseline features (to test whether overlap with other features flipped the sign).

### `credit_to_annuity_ratio` — relationship is not monotonic

| Band (ratio) | Default rate |
| --- | --- |
| 8.0 – 10.5 | 4.8% |
| 10.5 – 14.6 | 11.3% |
| 19.5 – 20.0 | 6.0% |
| 20.0 – 22.0 | 14.3% |
| 22.0 – 25.2 | 5.0% |
| 34.1 – 45.3 | 5.8% |

- Default rate zigzags across the range; no single linear coefficient can describe it.
- 34,068 loans cluster just below 20.0. Credit ÷ annuity is roughly "number of
  payments adjusted for the interest rate", so the spikes most likely reflect the
  lender's standard products and price points: the feature partly encodes the
  lender's own earlier risk pricing.
- VIF 1.05: not an overlap problem.

**Decision.** Remove from the linear baseline. Keep in the model table for
gradient boosting, which handles non-monotonic shapes.

### `bureau_worst_dpd_12m` — missing values filled with "never late"

| Group | Applicants | Default rate |
| --- | --- | --- |
| No arrears (0) | 24.1% | 7.4% |
| Any arrears (1–5) | 5.2% | 11–12% |
| No monthly bureau records | 70.7% | 8.0% |

- On its own the feature points the expected way (AUC 0.54).
- 71% of applicants have no monthly records; median imputation set them to 0,
  merging an average-risk "unknown" group into the low-risk "never late" group.
- Buckets 2–5 cover under 1% of applicants and are no riskier than bucket 1:
  the signal is any arrears vs none, not severity.
- VIF 1.03: not an overlap problem.

**Decision.** Replace with `bureau_any_late_12m` (1 / 0 / unknown) plus a
`bureau_balance_missing` flag, so unknown is never treated as clean.

### Noted, no action

`ext_source_mean` has VIF 5.8 because it is built from the individual external
scores. All three external-score coefficients point the expected way; revisit if
they become unstable.

**Effect.** All 13 baseline coefficients now have the expected sign. Validation AUC 0.7416 -> 0.7395, KS 0.3594 -> 0.3604: a negligible change in exchange for a fully explainable scorecard.


---

## 2026-10-27 · Stage 2: gradient boosting vs the scorecard

**Setup.** Same 61,503-applicant holdout as the scorecard, untouched until the end. LightGBM and XGBoost tuned with Optuna (20 trials each, inner validation split, early stopping). Five-fold out-of-fold predictions for a stacked ensemble. Champion chosen by cross-validated AUC, then reported on the holdout.

| Model | CV AUC | Holdout AUC | Gini | KS | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| Average of LightGBM + XGBoost | 0.7773 | 0.7757 | 0.551 | 0.419 | 0.262 |
| Stacked ensemble | 0.7773 | 0.7757 | 0.551 | 0.419 | 0.262 |
| XGBoost | 0.7768 | 0.7753 | 0.551 | 0.417 | 0.262 |
| LightGBM | 0.7766 | 0.7749 | 0.550 | 0.417 | 0.260 |
| Scorecard | 0.7388 | 0.7395 | 0.479 | 0.360 | 0.221 |

**Findings.**
- Champion beats the scorecard by +0.036 AUC (95% bootstrap CI +0.032 to +0.040, better in 100% of 300 resamples).
- The scorecard reproduced its stage-1 holdout AUC exactly (0.7395): the comparison is like for like.
- Stacking and simple averaging tied; the stacker gave the scorecard about zero weight (-0.03). Chose the simpler average.
- `credit_to_annuity_ratio`, removed from the scorecard as non-linear, is the trees' 4th most important feature, confirming the stage-1 decision.

**Decision.** Champion = equal-weight average of LightGBM and XGBoost. The scorecard stays as the explainable benchmark.

---

## 2026-10-27 · Stage 3: calibration and risk-based pricing

**Setup.** The 61,503-applicant holdout split in half: 30,751 to choose calibration and build the price grid, 30,752 to test both. Pricing: rate = (margin + funding + opex + PD × LGD) / (1 − PD), with LGD 45%, funding 8%, opex 4%, target margin 3%, rate cap 36%.

**Calibration.** Cross-validation chose no calibration: test ECE 0.0026 for the champion (0.0031 scorecard). This confirms the stage-2 decision not to re-weight defaults during training.

**Grid check on unseen applicants.** Bands A, B, C, E, F realised margins of 2.95–3.13% against the 3% target. Band D realised 2.23% (defaults 10.4% vs 9.8% predicted, about 1.3 standard errors): monitor, do not reprice on one sample. Band G (PD 39%) needs 53.7% and is declined under the 36% cap (3.5% of applicants).

**Model value.** Same flat price (20.1%) and approval rate, champion vs scorecard:

| Approval rate | Champion default rate | Scorecard default rate | Defaults avoided per 10k applications | Extra profit per 10k applications |
| --- | --- | --- | --- | --- |
| 70% | 3.72% | 4.44% | 50 | 19.6 million |
| 80% | 4.58% | 5.26% | 54 | 17.0 million |
| 90% | 5.82% | 6.16% | 31 | 10.3 million |

**Caveats.** TARGET means payment difficulties (a default proxy); one-period pricing with a single LGD; customers assumed to accept the price; amounts in the dataset's currency units.

---

## 2026-10-27 · Stage 4: reason codes, fairness audit and model card

**Setup.** Stage-3 test half (30,752 applicants never used for training, selection, calibration or pricing), with the stage-3 price grid applied. Reason codes from native TreeSHAP (LightGBM and XGBoost), averaged in log-odds for the ensemble. Fairness measured by gender and age group from the separate protected-attributes table; the model never sees these attributes.

**Approval rates.** Every group passes the four-fifths rule (adverse impact ratio at least 0.8). No approval-rate warning.

**Calibration gaps above 1 point:**

| Group | Predicted PD | Actual default rate | Effect on price |
| --- | --- | --- | --- |
| Men | 9.0% | 10.3% | slightly underpriced |
| Under 25 | 13.9% | 12.1% | slightly overpriced |
| 25–34 | 10.7% | 9.6% | slightly overpriced |
| 35–44 | 7.7% | 8.8% | slightly underpriced |

**Ranking quality.** Lower AUC for under-25s (0.717) and over-55s (0.728) than the best group (0.791): thin credit files for the young, pension and employment placeholders for the old.

**Interpretation.** Excluding protected attributes ("fairness through unawareness") does not equalise calibration across groups: men default more often in this data, and a model that cannot see gender averages them with women.

**Decision.**
- Do **not** adjust scores or prices using gender or age: that would make protected attributes pricing inputs.
- Document the gaps in the model card and monitor them quarterly.
- Next improvements: find which features act as proxies for the gaps, and add alternative data (e.g. mobile-money history) for thin-file young applicants.
