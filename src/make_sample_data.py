"""Generate small synthetic CSVs with the same schema as the Home Credit data.

Why: the real data cannot be committed to GitHub (size and licence), so
GitHub Actions and anyone cloning the repo would have nothing to run on.
This generator creates a few thousand fake applicants whose default risk
depends on their features, so the full pipeline runs end to end in seconds.

    python -m src.make_sample_data

Never report results from this data: it exists to test the plumbing.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, project_path

log = logging.getLogger(__name__)


def _applications(rng: np.random.Generator, ids: np.ndarray, risk: np.ndarray) -> pd.DataFrame:
    n = len(ids)
    income = rng.lognormal(mean=11.9, sigma=0.5, size=n).round(-2)
    credit = (income * rng.uniform(1.0, 6.0, n)).round(-2)
    goods = (credit / rng.uniform(1.0, 1.3, n)).round(-2)
    annuity = (credit / rng.uniform(10, 40, n)).round(2)
    days_employed = -rng.integers(30, 12000, n)
    days_employed[rng.random(n) < 0.15] = 365243  # pensioner placeholder, as in the real data

    def ext(noise):
        v = np.clip(0.55 - 0.25 * risk + rng.normal(0, noise, n), 0.01, 0.99)
        v[rng.random(n) < 0.3] = np.nan
        return v

    return pd.DataFrame({
        "SK_ID_CURR": ids,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
        "CODE_GENDER": rng.choice(["F", "M"], n, p=[0.65, 0.35]),
        "FLAG_OWN_CAR": rng.choice(["Y", "N"], n, p=[0.35, 0.65]),
        "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n, p=[0.7, 0.3]),
        "CNT_CHILDREN": rng.poisson(0.4, n),
        "CNT_FAM_MEMBERS": rng.integers(1, 6, n).astype(float),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods,
        "NAME_INCOME_TYPE": rng.choice(["Working", "Commercial associate", "Pensioner", "State servant"], n),
        "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education", "Incomplete higher"], n),
        "DAYS_BIRTH": -rng.integers(21 * 365, 69 * 365, n),
        "DAYS_EMPLOYED": days_employed,
        "DAYS_ID_PUBLISH": -rng.integers(0, 7000, n),
        "REGION_RATING_CLIENT": rng.choice([1, 2, 3], n, p=[0.1, 0.75, 0.15]),
        "EXT_SOURCE_1": ext(0.15),
        "EXT_SOURCE_2": ext(0.12),
        "EXT_SOURCE_3": ext(0.12),
    })


def generate(out_dir: Path, n_train: int = 3000, n_test: int = 600, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = np.arange(100001, 100001 + n_train + n_test)
    risk = rng.normal(0, 1, len(ids))  # hidden "true" riskiness per applicant
    apps = _applications(rng, ids, risk)

    # Default probability rises with risk; about 8% overall like the real data.
    p_default = 1 / (1 + np.exp(-(-2.6 + 1.1 * risk)))
    train = apps.iloc[:n_train].copy()
    train.insert(1, "TARGET", (rng.random(n_train) < p_default[:n_train]).astype(int))
    train.to_csv(out_dir / "application_train.csv", index=False)
    apps.iloc[n_train:].to_csv(out_dir / "application_test.csv", index=False)

    risk_by_id = dict(zip(ids, risk))

    # Bureau: 80% of applicants have history (the rest are thin-file)
    bureau_rows, balance_rows = [], []
    bureau_id = 5000000
    for cid in ids[rng.random(len(ids)) < 0.8]:
        r = risk_by_id[cid]
        for _ in range(rng.integers(1, 6)):
            bureau_id += 1
            credit_sum = float(rng.uniform(20000, 600000))
            active = rng.random() < 0.4
            bureau_rows.append({
                "SK_ID_CURR": cid, "SK_ID_BUREAU": bureau_id,
                "CREDIT_ACTIVE": "Active" if active else "Closed",
                "DAYS_CREDIT": -int(rng.integers(30, 2900)),
                "CREDIT_DAY_OVERDUE": int(rng.random() < 0.02 + 0.03 * max(r, 0)) * int(rng.integers(1, 90)),
                "AMT_CREDIT_MAX_OVERDUE": float(rng.exponential(2000)) if rng.random() < 0.3 else np.nan,
                "CNT_CREDIT_PROLONG": int(rng.random() < 0.05),
                "AMT_CREDIT_SUM": credit_sum,
                "AMT_CREDIT_SUM_DEBT": credit_sum * float(rng.uniform(0, 1)) * (1 + 0.3 * r) if active else 0.0,
                "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Car loan"]),
            })
            for m in range(int(rng.integers(0, 16))):
                late = rng.random() < 0.04 + 0.06 * max(r, 0)
                balance_rows.append({
                    "SK_ID_BUREAU": bureau_id, "MONTHS_BALANCE": -m,
                    "STATUS": str(int(rng.integers(1, 4))) if late else rng.choice(["0", "C", "X"]),
                })
    pd.DataFrame(bureau_rows).to_csv(out_dir / "bureau.csv", index=False)
    pd.DataFrame(balance_rows).to_csv(out_dir / "bureau_balance.csv", index=False)

    # Previous applications and their instalment payments
    prev_rows, inst_rows = [], []
    prev_id = 2000000
    for cid in ids[rng.random(len(ids)) < 0.85]:
        r = risk_by_id[cid]
        for _ in range(rng.integers(1, 5)):
            prev_id += 1
            refused = rng.random() < 0.15 + 0.1 * max(r, 0)
            requested = float(rng.uniform(20000, 400000))
            prev_rows.append({
                "SK_ID_PREV": prev_id, "SK_ID_CURR": cid,
                "NAME_CONTRACT_STATUS": "Refused" if refused else rng.choice(["Approved", "Canceled"], p=[0.85, 0.15]),
                "AMT_APPLICATION": requested,
                "AMT_CREDIT": 0.0 if refused else requested * float(rng.uniform(0.8, 1.1)),
                "DAYS_DECISION": -int(rng.integers(10, 2900)),
            })
            if refused:
                continue
            start = -int(rng.integers(100, 2500))
            for k in range(1, int(rng.integers(3, 13))):
                due = start + 30 * k
                amount = round(requested / 12, 2)
                delay = int(rng.exponential(3 + 6 * max(r, 0))) if rng.random() < 0.15 + 0.15 * max(r, 0) else -int(rng.integers(0, 10))
                if rng.random() < 0.1:  # split payment into two rows
                    inst_rows.append({"SK_ID_PREV": prev_id, "SK_ID_CURR": cid, "NUM_INSTALMENT_VERSION": 1,
                                      "NUM_INSTALMENT_NUMBER": k, "DAYS_INSTALMENT": due,
                                      "DAYS_ENTRY_PAYMENT": due + delay - 5, "AMT_INSTALMENT": amount,
                                      "AMT_PAYMENT": round(amount / 2, 2)})
                    inst_rows.append({"SK_ID_PREV": prev_id, "SK_ID_CURR": cid, "NUM_INSTALMENT_VERSION": 1,
                                      "NUM_INSTALMENT_NUMBER": k, "DAYS_INSTALMENT": due,
                                      "DAYS_ENTRY_PAYMENT": due + delay, "AMT_INSTALMENT": amount,
                                      "AMT_PAYMENT": round(amount / 2, 2)})
                else:
                    inst_rows.append({"SK_ID_PREV": prev_id, "SK_ID_CURR": cid, "NUM_INSTALMENT_VERSION": 1,
                                      "NUM_INSTALMENT_NUMBER": k, "DAYS_INSTALMENT": due,
                                      "DAYS_ENTRY_PAYMENT": due + delay, "AMT_INSTALMENT": amount,
                                      "AMT_PAYMENT": amount})
    pd.DataFrame(prev_rows).to_csv(out_dir / "previous_application.csv", index=False)
    pd.DataFrame(inst_rows).to_csv(out_dir / "installments_payments.csv", index=False)

    log.info("Synthetic sample data written to %s (%s applicants)", out_dir, f"{len(ids):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-test", type=int, default=600)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    generate(project_path(cfg["data"]["sample_dir"]), args.n_train, args.n_test, cfg["project"]["random_seed"])


if __name__ == "__main__":
    main()
