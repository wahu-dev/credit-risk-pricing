"""Load settings from config/config.yaml and resolve project paths.

Keeping every path and parameter in one YAML file means the same code runs on
your laptop, in GitHub Actions and on a server without edits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config. Defaults to config/config.yaml in the project."""
    config_path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str | Path) -> Path:
    """Turn a path from the config into an absolute path inside the project.

    Absolute paths are returned unchanged, so users can point at data
    stored elsewhere (for example an external drive).
    """
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
