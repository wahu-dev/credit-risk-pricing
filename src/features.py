"""Build feature tables by running the SQL files in sql/features in order.

Each .sql file is a single SELECT statement. This runner wraps it in
CREATE TABLE ... AS and names the table after the file:
    03_bureau_features.sql  ->  table bureau_features

The numeric prefix sets the run order, so later files can read earlier tables.
This is the same idea dbt uses: SQL holds the logic, Python only orchestrates,
and every transformation is version-controlled and reviewable.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.db import query_df, safe_identifier, table_row_count

log = logging.getLogger(__name__)

_PREFIX = re.compile(r"^\d+_")


def table_name_for(sql_file: Path) -> str:
    return safe_identifier(_PREFIX.sub("", sql_file.stem))


def ordered_sql_files(sql_dir: Path) -> list[Path]:
    files = sorted(Path(sql_dir).glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"No .sql files found in {sql_dir}")
    return files


def materialise(con, table_name: str, select_sql: str) -> None:
    """Create (or replace) a table from a SELECT statement."""
    select_sql = select_sql.strip().rstrip(";")
    con.execute(f"DROP TABLE IF EXISTS {safe_identifier(table_name)}")
    con.execute(f"CREATE TABLE {table_name} AS\n{select_sql}")


def build_features(con, sql_dir: Path) -> dict[str, int]:
    """Run every SQL file in order. Returns {table_name: row_count}."""
    counts: dict[str, int] = {}
    for sql_file in ordered_sql_files(sql_dir):
        name = table_name_for(sql_file)
        materialise(con, name, sql_file.read_text(encoding="utf-8"))
        counts[name] = table_row_count(con, name)
        log.info("Built %-32s %12s rows", name, f"{counts[name]:,}")
    return counts


def read_table(con, table_name: str) -> pd.DataFrame:
    return query_df(con, f"SELECT * FROM {safe_identifier(table_name)}")
