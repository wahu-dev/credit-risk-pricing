"""Download the Home Credit data from Kaggle.

Before this works you must:
  1. Have kaggle.json in C:\\Users\\<you>\\.kaggle\\ (Kaggle > Settings > API > Create New Token)
  2. Open https://www.kaggle.com/competitions/home-credit-default-risk,
     click "Join Competition" / "I Understand and Accept". Without this
     Kaggle returns 403 Forbidden even with a valid key.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)


def all_files_present(raw_dir: Path, file_names: list[str]) -> bool:
    return all((raw_dir / name).exists() for name in file_names)


def download_competition(competition: str, raw_dir: Path, file_names: list[str]) -> None:
    """Download and unzip the competition files, skipping work already done."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    if all_files_present(raw_dir, file_names):
        log.info("All raw files already present in %s, skipping download.", raw_dir)
        return

    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "The 'kaggle' command was not found. Activate your virtual environment "
            "(.venv\\Scripts\\activate) and run: pip install -r requirements.txt"
        )

    log.info("Downloading %s from Kaggle (about 700 MB zipped)...", competition)
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition, "-p", str(raw_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        hint = ""
        if "403" in result.stderr or "403" in result.stdout:
            hint = (
                "\nHint: 403 usually means you have not accepted the competition rules. "
                f"Open https://www.kaggle.com/competitions/{competition} and click Join."
            )
        if "401" in result.stderr or "401" in result.stdout:
            hint = "\nHint: 401 means kaggle.json is missing or wrong."
        raise RuntimeError(f"Kaggle download failed:\n{result.stderr or result.stdout}{hint}")

    for zip_path in raw_dir.glob("*.zip"):
        log.info("Extracting %s", zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        zip_path.unlink()  # free disk space; the CSVs are what we need

    missing = [n for n in file_names if not (raw_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"Download finished but these files are missing: {missing}")
    log.info("Download complete.")
