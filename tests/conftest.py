"""Shared test fixtures: a tiny warehouse where every answer is known.

Four applicants, each designed to test one behaviour:
  1  rich history: bureau loans, a refusal, a split payment and a late payment
  2  thin file (no bureau, no previous loans) with the 365243 employment placeholder
  3  previous applications only, no bureau history
  4  test-set applicant (no target)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import PROJECT_ROOT
from src.db import connect
from src.features import build_features
from src.load import load_raw_tables

SQL_DIR = PROJECT_ROOT / "sql" / "features"

TABLES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "installments_payments": "installments_payments.csv",
}


def _application(sk_id: int, **overrides) -> dict:
    row = {
        "SK_ID_CURR": sk_id, "NAME_CONTRACT_TYPE": "Cash loans", "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "N", "FLAG_OWN_REALTY": "Y", "CNT_CHILDREN": 0, "CNT_FAM_MEMBERS": 2.0,
        "AMT_INCOME_TOTAL": 100000.0, "AMT_CREDIT": 300000.0, "AMT_ANNUITY": 15000.0,
        "AMT_GOODS_PRICE": 250000.0, "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education", "DAYS_BIRTH": -12000, "DAYS_EMPLOYED": -730,
        "DAYS_ID_PUBLISH": -1000, "REGION_RATING_CLIENT": 2,
        "EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.6, "EXT_SOURCE_3": None,
    }
    row.update(overrides)
    return row


def write_fixture_csvs(folder: Path) -> None:
    train = pd.DataFrame([
        {"TARGET": 0, **_application(1)},
        {"TARGET": 1, **_application(2, DAYS_EMPLOYED=365243, EXT_SOURCE_1=None, EXT_SOURCE_2=None)},
        {"TARGET": 0, **_application(3)},
    ])
    test = pd.DataFrame([_application(4)])

    bureau = pd.DataFrame([
        {"SK_ID_CURR": 1, "SK_ID_BUREAU": 11, "CREDIT_ACTIVE": "Active", "DAYS_CREDIT": -10,
         "CREDIT_DAY_OVERDUE": 0, "AMT_CREDIT_MAX_OVERDUE": None, "CNT_CREDIT_PROLONG": 0,
         "AMT_CREDIT_SUM": 100.0, "AMT_CREDIT_SUM_DEBT": 50.0, "CREDIT_TYPE": "Consumer credit"},
        {"SK_ID_CURR": 1, "SK_ID_BUREAU": 12, "CREDIT_ACTIVE": "Closed", "DAYS_CREDIT": -100,
         "CREDIT_DAY_OVERDUE": 0, "AMT_CREDIT_MAX_OVERDUE": 30.0, "CNT_CREDIT_PROLONG": 1,
         "AMT_CREDIT_SUM": 200.0, "AMT_CREDIT_SUM_DEBT": 0.0, "CREDIT_TYPE": "Credit card"},
    ])
    bureau_balance = pd.DataFrame([
        {"SK_ID_BUREAU": 11, "MONTHS_BALANCE": 0, "STATUS": "0"},
        {"SK_ID_BUREAU": 11, "MONTHS_BALANCE": -1, "STATUS": "2"},   # late, inside 12m
        {"SK_ID_BUREAU": 11, "MONTHS_BALANCE": -13, "STATUS": "5"},  # outside window: ignored
        {"SK_ID_BUREAU": 12, "MONTHS_BALANCE": -2, "STATUS": "C"},
        {"SK_ID_BUREAU": 12, "MONTHS_BALANCE": -3, "STATUS": "X"},
    ])
    previous = pd.DataFrame([
        {"SK_ID_PREV": 101, "SK_ID_CURR": 1, "NAME_CONTRACT_STATUS": "Approved",
         "AMT_APPLICATION": 1000.0, "AMT_CREDIT": 900.0, "DAYS_DECISION": -50},
        {"SK_ID_PREV": 102, "SK_ID_CURR": 1, "NAME_CONTRACT_STATUS": "Refused",
         "AMT_APPLICATION": 500.0, "AMT_CREDIT": 0.0, "DAYS_DECISION": -20},   # latest decision
        {"SK_ID_PREV": 301, "SK_ID_CURR": 3, "NAME_CONTRACT_STATUS": "Approved",
         "AMT_APPLICATION": 0.0, "AMT_CREDIT": 0.0, "DAYS_DECISION": -400},
    ])
    installments = pd.DataFrame([
        # Instalment 1 paid on time in two halves: must NOT count as underpaid or late
        {"SK_ID_PREV": 101, "SK_ID_CURR": 1, "NUM_INSTALMENT_VERSION": 1, "NUM_INSTALMENT_NUMBER": 1,
         "DAYS_INSTALMENT": -40, "DAYS_ENTRY_PAYMENT": -45, "AMT_INSTALMENT": 100.0, "AMT_PAYMENT": 50.0},
        {"SK_ID_PREV": 101, "SK_ID_CURR": 1, "NUM_INSTALMENT_VERSION": 1, "NUM_INSTALMENT_NUMBER": 1,
         "DAYS_INSTALMENT": -40, "DAYS_ENTRY_PAYMENT": -42, "AMT_INSTALMENT": 100.0, "AMT_PAYMENT": 50.0},
        # Instalment 2 paid in full but 5 days late
        {"SK_ID_PREV": 101, "SK_ID_CURR": 1, "NUM_INSTALMENT_VERSION": 1, "NUM_INSTALMENT_NUMBER": 2,
         "DAYS_INSTALMENT": -10, "DAYS_ENTRY_PAYMENT": -5, "AMT_INSTALMENT": 100.0, "AMT_PAYMENT": 100.0},
    ])

    folder.mkdir(parents=True, exist_ok=True)
    for name, df in {
        "application_train.csv": train, "application_test.csv": test, "bureau.csv": bureau,
        "bureau_balance.csv": bureau_balance, "previous_application.csv": previous,
        "installments_payments.csv": installments,
    }.items():
        df.to_csv(folder / name, index=False)


@pytest.fixture
def warehouse(tmp_path):
    """An in-memory warehouse with the fixture data loaded and features built."""
    write_fixture_csvs(tmp_path)
    con = connect(None)
    load_raw_tables(con, tmp_path, TABLES)
    build_features(con, SQL_DIR)
    yield con
    con.close()
