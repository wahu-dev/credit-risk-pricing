"""Small helpers around the DuckDB warehouse.

DuckDB is an in-process analytical database: no server to install, it reads
CSV and Parquet directly, and it runs SQL over millions of rows on a laptop.
That makes it a good stand-in for a cloud warehouse (Snowflake, BigQuery)
in a portfolio project, and the SQL transfers almost unchanged.
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_identifier(name: str) -> str:
    """Reject table names that are not plain identifiers (blocks SQL injection
    through file names or config values)."""
    if not _IDENTIFIER.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


def connect(path: str | Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open (or create) the warehouse file. None gives an in-memory database,
    which the tests use. read_only=True for stages that only read from it."""
    if path is None:
        return duckdb.connect()
    path = Path(path)
    if read_only:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python -m src.pipeline` first.")
        return duckdb.connect(str(path), read_only=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def query_df(con, sql: str) -> pd.DataFrame:
    """Run a query and return a pandas DataFrame."""
    cursor = con.execute(sql)
    if hasattr(cursor, "df"):  # DuckDB fast path (columnar, no Python loop)
        return cursor.df()
    columns = [d[0] for d in cursor.description]
    return pd.DataFrame(cursor.fetchall(), columns=columns)


def df_to_table(con, name: str, df: pd.DataFrame) -> None:
    """Write a DataFrame into the warehouse as a table (replacing it)."""
    name = safe_identifier(name)
    con.register("_incoming_df", df)
    try:
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _incoming_df")
    finally:
        con.unregister("_incoming_df")


def table_row_count(con, name: str) -> int:
    name = safe_identifier(name)
    return int(con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
