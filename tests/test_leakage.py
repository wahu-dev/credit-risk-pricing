"""Guardrails against the two most expensive modelling mistakes:
target leakage and protected attributes reaching the model."""
from __future__ import annotations

import re

from src.config import PROJECT_ROOT, load_config

SQL_DIR = PROJECT_ROOT / "sql" / "features"
# Only these files may mention the target: one passes it through, one joins it on.
ALLOWED_TARGET_FILES = {"01_application_features.sql", "10_model_table.sql"}


def _sql_without_comments(text: str) -> str:
    return re.sub(r"--[^\n]*", "", text)


def test_target_only_referenced_where_allowed():
    for sql_file in SQL_DIR.glob("*.sql"):
        body = _sql_without_comments(sql_file.read_text(encoding="utf-8"))
        mentions_target = re.search(r"\btarget\b", body, flags=re.IGNORECASE)
        if sql_file.name not in ALLOWED_TARGET_FILES:
            assert not mentions_target, f"{sql_file.name} references TARGET: possible leakage"


def test_model_table_sql_does_not_join_protected_attributes():
    body = _sql_without_comments((SQL_DIR / "10_model_table.sql").read_text(encoding="utf-8"))
    assert "protected_attributes" not in body
    assert "code_gender" not in body.lower()
    assert "days_birth" not in body.lower()


def test_baseline_features_exclude_protected_attributes():
    cfg = load_config()
    protected = set(cfg["features"]["protected_attributes"])
    assert not protected & set(cfg["baseline"]["features"])
