"""Load raw CSV files into the DuckDB warehouse as staging tables."""
from __future__ import annotations

import logging
from pathlib import Path

from src.db import safe_identifier, table_row_count

log = logging.getLogger(__name__)


def load_raw_tables(con, raw_dir: Path, tables: dict[str, str]) -> dict[str, int]:
    """Create one table per CSV. Returns {table_name: row_count}.

    read_csv_auto infers column types from the file. Paths are passed with
    forward slashes (as_posix) because a Windows backslash inside a SQL string
    can be read as an escape character.
    """
    counts: dict[str, int] = {}
    for table_name, file_name in tables.items():
        table_name = safe_identifier(table_name)
        csv_path = Path(raw_dir) / file_name
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. Run the download step first, "
                "or use --sample to run on synthetic data."
            )
        con.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS "
            f"SELECT * FROM read_csv_auto('{csv_path.as_posix()}', header = true)"
        )
        counts[table_name] = table_row_count(con, table_name)
        log.info("Loaded %-24s %12s rows", table_name, f"{counts[table_name]:,}")
    return counts
