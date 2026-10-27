"""Thin wrapper around MLflow experiment tracking.

Every run logs its parameters, metrics and output files to a local SQLite
database (mlflow.db). Browse them with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

If MLflow is not installed or tracking is switched off (--no-tracking), every
call becomes a no-op, so the training code never has to check.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)


class Tracker:
    def __init__(self, enabled: bool, db_path: Path, experiment: str):
        self.mlflow = None
        if not enabled:
            return
        try:
            import mlflow
        except ImportError:
            log.warning("MLflow not installed; continuing without experiment tracking.")
            return
        mlflow.set_tracking_uri(f"sqlite:///{Path(db_path).resolve().as_posix()}")
        mlflow.set_experiment(experiment)
        self.mlflow = mlflow

    @property
    def active(self) -> bool:
        return self.mlflow is not None

    @contextmanager
    def run(self, name: str, nested: bool = False):
        if not self.active:
            yield
            return
        with self.mlflow.start_run(run_name=name, nested=nested):
            yield

    def params(self, values: dict, prefix: str = "") -> None:
        if self.active:
            self.mlflow.log_params({f"{prefix}{k}": str(v)[:250] for k, v in values.items()})

    def metrics(self, values: dict, prefix: str = "", step: int | None = None) -> None:
        if self.active:
            numeric = {f"{prefix}{k}": float(v) for k, v in values.items() if isinstance(v, (int, float))}
            self.mlflow.log_metrics(numeric, step=step)

    def artifact(self, path: Path) -> None:
        if self.active and Path(path).exists():
            self.mlflow.log_artifact(str(path))
