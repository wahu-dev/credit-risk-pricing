# Credit Risk Scoring & Risk-Based Loan Pricing

An end-to-end lending decision engine: **probability of default → risk band → approve or decline → interest rate**, with the top reasons behind every decision. Built on 356,255 real loan applications, from SQL features to a tested, containerised API with drift monitoring.

![CI](https://github.com/wahu-dev/credit-risk-pricing/actions/workflows/ci.yml/badge.svg)

**Portfolio:** [wahu-dev.github.io](https://wahu-dev.github.io) · **Model card:** [docs/model_card.md](docs/model_card.md) · **Decision log:** [docs/decision_log.md](docs/decision_log.md)

## Headline results

| | |
| --- | --- |
| Champion model | LightGBM + XGBoost average, holdout **AUC 0.776** (Gini 0.551, KS 0.419) |
| Gain over the scorecard | **+0.036 AUC**, 95% CI +0.032 to +0.040 (paired bootstrap, 61,503 held-out applicants) |
| Business value | At 80% approval: default rate **4.58% vs 5.26%**, 54 fewer defaults per 10,000 applications |
| Pricing | Seven risk bands, each landing near its 3% target margin on unseen applicants |
| Responsible AI | Reason codes for every decision; all gender and age groups pass the four-fifths rule |
| Engineering | 93 tests; CI runs every stage and calls the Docker API on each push |

## Business problem

A consumer lender wants to approve more good borrowers, price each loan to its risk, and keep
expected losses within target, including **thin-file** applicants with little or no credit history.
This project builds the full decision engine and puts a money value on the model.

## Data

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk): 356,255 loan
applications (about 307,000 with a known outcome) with credit-bureau history, previous applications and
repayment records across 7 linked tables. The data is not stored in this repo; the pipeline downloads it.

## How it works

```
Kaggle CSVs ─► DuckDB ─► SQL features ─► validation ─► scorecard baseline
                                                         │
             LightGBM + XGBoost (Optuna, MLflow) ◄────────┘
                         │
             calibration check ─► risk bands and price grid ─► reason codes, fairness audit
                                                                 │
                         FastAPI service ◄── serving bundle ◄────┘
                         │
                         Docker image, PSI drift monitoring, GitHub Actions CI
```

| Stage | What it does | Run |
| --- | --- | --- |
| 1. Features and baseline | SQL feature models, validation gate, logistic scorecard | `python -m src.pipeline` |
| 2. Challenger models | LightGBM and XGBoost tuned with Optuna, stacking and averaging, bootstrap comparison | `python -m src.train_models` |
| 3. Pricing | Calibration choice by cross-validation, risk bands, price grid, profit vs approval | `python -m src.price_loans` |
| 4. Explain and audit | TreeSHAP reason codes, fairness audit, generated model card | `python -m src.explain_and_audit` |
| 5. Serve and monitor | Serving bundle, FastAPI, Docker, PSI drift monitoring | `python -m src.build_serving` |

## Stage 1: data and features

- **SQL feature engineering** in DuckDB: affordability ratios, bureau debt and arrears in the last
  12 months, refusal history, repayment behaviour. Window functions find each applicant's most
  recent loan and decision. One `SELECT` per file in `sql/features/`, run in order.
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
  Two features failed that check; after investigation they were replaced or removed, at a cost of
  0.002 AUC, so the scorecard is one a risk team can sign off.

## Stage 2: model results

| Model | ROC-AUC | Gini | KS | PR-AUC |
| --- | --- | --- | --- | --- |
| **Champion: LightGBM + XGBoost average** | **0.776** | **0.551** | **0.419** | **0.262** |
| Baseline scorecard (13 features) | 0.740 | 0.479 | 0.360 | 0.221 |

Holdout of 61,503 applicants, never used for training or tuning. Champion vs scorecard: **+0.036 AUC**
(95% CI +0.032 to +0.040). A stacked model was also tested; it gave the scorecard a slightly negative
weight and did not beat the simple average, so the average was kept.

**On the benchmark.** The top Kaggle scores for this competition were around 0.80 AUC. Those came from
hundreds of engineered features and large blends tuned to the leaderboard. This project trades a few
points of AUC for a compact feature set built in SQL, strict leakage tests, calibrated probabilities and
reasons a lender can give to a customer. The aim is a decision system that could go through model
risk review, not a leaderboard score.

## Stage 3: risk-based pricing

The champion's probabilities were already well calibrated (expected calibration error 0.26%), so
cross-validation chose no extra calibration layer. Each band is priced so that, after expected losses,
it earns the target margin:

```
rate = (target margin + cost of funds + operating cost + PD × LGD) / (1 − PD)
```

**These inputs are assumptions, not values from the data.** The dataset contains no interest rates or
funding costs, so they are set in `config/config.yaml` and can be changed:

| Assumption | Value | Note |
| --- | --- | --- |
| Loss given default (LGD) | 45% | Common benchmark for unsecured consumer credit |
| Cost of funds | 8% | |
| Operating cost | 4% | |
| Target margin | 3% | |
| Rate cap | 36% | Bands that would need more are declined |

The formula is a simplified one-period view: it ignores loan term, amortisation and recoveries timing.

| Band | Upper PD | Rate | Loans (unseen) | Realised margin |
| --- | --- | --- | --- | --- |
| A | 2.5% | 16.0% | 7,636 | ≈ 3.0% |
| B | 5% | 17.3% | 7,982 | ≈ 3.0% |
| C | 8% | 19.1% | 5,282 | ≈ 3.0% |
| D | 12% | 21.5% | 3,751 | 2.2% (on the watch list) |
| E | 20% | 25.9% | 3,426 | ≈ 3.0% |
| F | 30% | 34.1% | 1,640 | ≈ 3.1% |
| G | 100% | declined | | Average PD 39% would need a 53.7% rate |

![Band margins](reports/pricing/band_margins.png)

At the same price and 80% approval, the champion approves a portfolio with a 4.58% default rate vs
5.26% for the scorecard: 54 fewer defaults and about 17 million more profit per 10,000 applications.
Profit is in the dataset's own currency units, which Home Credit does not name, and depends on the
assumptions above.

![Profit vs approval](reports/pricing/profit_vs_approval.png)

## Stage 4: explainability and fairness

Every score comes with its top three reasons, computed with TreeSHAP and written in plain language.
Factors close to a typical applicant's value are skipped, so a typical applicant gets no reasons.
Example: why one applicant was declined.

![Example decline](reports/audit/example_decline.png)

Fairness audit on 30,752 unseen applicants: every gender and age group passes the four-fifths rule on
approval rates. Small calibration gaps (1 to 1.7 points) and lower ranking accuracy for under-25s
(AUC 0.717) and over-55s (0.728) are documented and monitored, never corrected by using protected
attributes. Full details in the **[model card](docs/model_card.md)**.

![Fairness](reports/audit/fairness.png)

## Stage 5: scoring API and monitoring

A FastAPI service turns the models into a decision engine: send an applicant's features, get back their
**probability of default, risk band, approve/decline, interest rate and top three reasons**. Invalid
input returns a clear 422 error, and every decision is logged.

```bash
python -m src.build_serving          # package models, calibrator, price grid
uvicorn src.api:app --port 8000      # then open http://127.0.0.1:8000/docs
```

Example response for a declined applicant:

```json
{
  "pd": 0.50,
  "risk_band": "G",
  "decision": "decline",
  "interest_rate": null,
  "reasons": [
    "External credit score (average) is lower than typical (0.05 vs 0.52)",
    "External credit score 2 is lower than typical (0.05 vs 0.57)",
    "Share of instalments paid late is higher than typical (0.60 vs 0.00)"
  ]
}
```

**Drift monitoring.** `python -m src.monitor --simulate` compares new applicants with the validation
population using the Population Stability Index (PSI). A normal month stays stable (PSI about 0.001); a
simulated downturn triggers alerts on the external credit scores (PSI up to 0.56) before the score
itself crosses the alert line.

![Drift monitoring](reports/monitoring/psi_simulation.png)

**Docker.** `docker build -t credit-risk-api .` then `docker run -p 8000:8000 credit-risk-api`. The image
pins the exact library versions that trained the models, runs as a non-root user and has a health check.
CI builds it and calls the running container on every push.

## Limitations

- **Pricing inputs are assumed** (see stage 3), and the rate formula is a one-period simplification.
- **Through-the-door bias.** The data only contains outcomes for loans that were granted, so the model
  has not seen how rejected applicants would have performed. No reject inference is applied.
- **One point in time.** The holdout is a random split, not a later time period, so it measures
  ranking power but not stability over time. The drift monitor is the guard for that in production.
- **Market fit.** Home Credit is consumer lending with bureau data. Applying the approach to Kenyan
  mobile lending would mean replacing bureau features with mobile money and airtime behaviour.

## Roadmap

- ✅ SQL features, validation, interpretable baseline
- ✅ Gradient boosting, stacking, Optuna tuning, MLflow tracking
- ✅ Calibration, risk-based price grid, profit vs approval
- ✅ TreeSHAP reason codes, fairness audit, model card
- ✅ FastAPI scoring service, Docker, PSI monitoring, CI
- ⬜ Champion vs challenger Bayesian A/B test
- ⬜ Out-of-time validation and reject inference

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pytest                           # 93 tests

# Full pipeline on the real data (needs a Kaggle API key)
python -m src.pipeline           # 1. SQL features, validation, baseline scorecard
python -m src.train_models       # 2. tuned LightGBM + XGBoost ensemble (about 40 min)
python -m src.price_loans        # 3. calibration and risk-based pricing
python -m src.explain_and_audit  # 4. reason codes, fairness audit, model card
python -m src.build_serving      # 5. package for the API
python -m src.monitor --simulate #    drift monitoring demo
uvicorn src.api:app --port 8000  #    scoring API

# Every step also runs on synthetic data in seconds: add --sample
```

## Project structure

```
config/config.yaml        all paths, features, model and pricing settings
sql/features/             feature logic, one SELECT per file, run in order
src/                      pipeline code for all five stages, the API and monitoring
tests/                    93 unit tests with hand-checked expected values
reports/                  charts and results written by the pipeline
docs/                     decision log and generated model card
Dockerfile                serving image for the API
.github/workflows/ci.yml  tests and every stage on synthetic data, plus the Docker API
```

## Author

**Ann Wahu**, data scientist in Nairobi.
[Portfolio](https://wahu-dev.github.io) · [LinkedIn](https://www.linkedin.com/in/ann-wahu/) · [GitHub](https://github.com/wahu-dev)
