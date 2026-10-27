# Credit Risk Scoring & Risk-Based Loan Pricing

> **Status: stage 1 of 5 complete** — SQL feature pipeline, data validation and an interpretable baseline scorecard.
> Headline results and the demo video will be added as later stages land.

![CI](https://github.com/<your-username>/credit-risk-pricing/actions/workflows/ci.yml/badge.svg)

## Business problem

A consumer lender wants to approve more good borrowers, price each loan to its risk, and keep
expected losses within target, including **thin-file** applicants with little or no credit history.
This project builds the full decision engine: probability of default (PD) → risk band →
approve / decline → interest rate, with a reason for every decision.

## Data

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk): about 307,000
real loan applications with credit-bureau history, previous applications and repayment records
(7 linked tables). The data is not stored in this repo; the pipeline downloads it.

## Approach (stage 1)

```
Kaggle CSVs ──► DuckDB staging tables ──► SQL feature models ──► validation ──► baseline logit
                                           (sql/features/*.sql)    (fails loudly)  (Statsmodels)
```

- **SQL feature engineering** in DuckDB: affordability ratios, bureau debt and arrears in the last
  12 months, refusal history, repayment behaviour. Window functions find each applicant's most
  recent loan and decision.
- **Data traps handled:** split instalment payments are collapsed before measuring underpayment;
  the `365243` employment placeholder becomes missing plus a flag; "no history" (counts = 0) is kept
  separate from "unknown" (ratios = NULL).
- **Fairness by design:** gender and age are kept in a separate `protected_attributes` table and can
  never enter the model. They are used only for fairness checks (credit scoring is high-risk under
  the EU AI Act).
- **Validation gate:** one row per applicant, valid targets, no protected attributes, no infinite
  values, no near-empty columns. The pipeline stops if any check fails.
- **Interpretable baseline:** logistic regression with standardised inputs, odds ratios per standard
  deviation, and an automatic check that every coefficient points in the business-expected direction.

## Results

Stage 1 results (baseline on validation split) are written to `reports/baseline_metrics.json`.
_To be summarised here after the full run._

| Model | ROC-AUC | Gini | KS | PR-AUC |
| --- | --- | --- | --- | --- |
| **Champion: LightGBM + XGBoost average** | **0.776** | **0.551** | **0.419** | **0.262** |
| Baseline scorecard (13 features) | 0.740 | 0.479 | 0.360 | 0.221 |

Champion vs scorecard on 61,503 held-out applicants: **+0.036 AUC** (95% CI +0.032 to +0.040).

## Pricing results (stage 3)

Risk-based price grid, checked on 30,752 applicants it never saw: every approved band lands close to its 3% target margin.

![Band margins](reports/pricing/band_margins.png)

At the same price and 80% approval, the champion approves a portfolio with a 4.58% default rate vs 5.26% for the scorecard: 54 fewer defaults and about 17 million more profit per 10,000 applications.

![Profit vs approval](reports/pricing/profit_vs_approval.png)

## Roadmap

1. ✅ SQL features, validation, interpretable baseline
2. ✅ Gradient boosting + stacking, Optuna tuning, MLflow tracking, calibration
3. ✅ Calibration, risk-based pricing grid and profit-vs-approval curve
4. Champion vs challenger Bayesian A/B test
5. SHAP reasons, fairness report, model card
6. FastAPI scoring service, Docker, monitoring (PSI)

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python -m src.pipeline --sample  # 10-second run on synthetic data
python -m src.pipeline           # full run on Kaggle data (needs a Kaggle API key)
pytest                           # tests
```

## Project structure

```
config/config.yaml        all paths and parameters
sql/features/             feature logic, one SELECT per file, run in order
src/                      pipeline code (download, load, features, validate, baseline)
tests/                    unit tests with hand-checked expected values
reports/                  generated metrics and coefficients
```
