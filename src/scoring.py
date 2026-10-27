"""Load the stage-2 model bundle and turn features into probabilities.
Shared by the pricing stage and (later) the scoring API."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.boosting import KINDS


def load_bundle(path: Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.train_models` first.")
    return joblib.load(path)


def member_predictions(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: m.predict(X) for name, m in bundle["members"].items()}, index=X.index)


def champion_probability(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Raw (uncalibrated) probability from whichever model won stage 2."""
    preds = member_predictions(bundle, X)
    champion = bundle["champion"]
    if champion == "average_boosted":
        return preds[list(KINDS)].mean(axis=1).to_numpy()
    if champion == "stacked":
        return bundle["stacker"].predict(preds)
    return preds[champion].to_numpy()
