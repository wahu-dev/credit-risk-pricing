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
