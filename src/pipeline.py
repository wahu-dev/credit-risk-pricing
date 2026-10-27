"""End-to-end pipeline for stage 1.

    python -m src.pipeline            # real Kaggle data (downloads if needed)
    python -m src.pipeline --sample   # synthetic data, runs in seconds

Steps: download -> load to DuckDB -> SQL features -> validate -> baseline model.
"""
from __future__ import annotations

import argparse
import logging
import time

from src.baseline import train_baseline
from src.config import load_config, project_path
from src.db import connect, table_row_count
from src.download import download_competition
from src.features import build_features, read_table
from src.load import load_raw_tables
from src.make_sample_data import generate as generate_sample
from src.validate import validate_model_table, write_report

log = logging.getLogger("pipeline")


def run(sample: bool = False, skip_download: bool = False) -> dict:
    cfg = load_config()
    data_cfg = cfg["data"]
    reports_dir = project_path(cfg["reports_dir"]) / ("sample" if sample else "")
    started = time.perf_counter()

    # 1. Get the raw files
    if sample:
        raw_dir = project_path(data_cfg["sample_dir"])
        if not (raw_dir / "application_train.csv").exists():
            generate_sample(raw_dir, seed=cfg["project"]["random_seed"])
        warehouse = project_path(data_cfg["warehouse_path"]).with_name("sample_warehouse.duckdb")
    else:
        raw_dir = project_path(data_cfg["raw_dir"])
        if not skip_download:
            download_competition(data_cfg["kaggle_competition"], raw_dir, list(data_cfg["tables"].values()))
        warehouse = project_path(data_cfg["warehouse_path"])

    con = connect(warehouse)
    try:
        # 2. Staging tables
        load_raw_tables(con, raw_dir, data_cfg["tables"])
        expected_rows = table_row_count(con, "application_train") + table_row_count(con, "application_test")

        # 3. Feature tables (SQL)
        build_features(con, project_path(cfg["features"]["sql_dir"]))
        model_table = read_table(con, cfg["features"]["model_table"])
    finally:
        con.close()

    # 4. Validate before anything trains on it
    report = validate_model_table(
        model_table,
        protected_attributes=cfg["features"]["protected_attributes"],
        max_missing_share=cfg["validation"]["max_missing_share"],
        min_rows=cfg["validation"]["min_rows"],
        expected_rows=expected_rows,
    )
    write_report(report, reports_dir / "data_quality.json")

    processed_dir = project_path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_table.to_parquet(processed_dir / ("model_table_sample.parquet" if sample else "model_table.parquet"), index=False)

    # 5. Interpretable baseline
    base_cfg = cfg["baseline"]
    baseline = train_baseline(
        model_table,
        feature_signs=base_cfg["features"],
        test_size=base_cfg["test_size"],
        winsor_quantiles=tuple(base_cfg["winsor_quantiles"]),
        seed=cfg["project"]["random_seed"],
        reports_dir=reports_dir,
    )

    log.info("Pipeline finished in %.1f s. Reports in %s", time.perf_counter() - started, reports_dir)
    return {"data_quality": report, "baseline": baseline["metrics"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit risk pipeline, stage 1")
    parser.add_argument("--sample", action="store_true", help="run on synthetic sample data")
    parser.add_argument("--skip-download", action="store_true", help="use CSVs already in data/raw")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    run(sample=args.sample, skip_download=args.skip_download)


if __name__ == "__main__":
    main()
