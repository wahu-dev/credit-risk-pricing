"""The scoring service: features in, decision out.

Reuses exactly the same code as the analysis stages (encoder, champion models,
calibrator, price grid, TreeSHAP reason codes), so what the API returns can
never drift from what the reports describe.

Input contract: the API receives the *model features* for an applicant (the
columns of the model table), as an upstream feature store would provide them.
Unknown names are rejected; missing values are allowed (the models handle
them), except for count/flag features, which are filled with the typical value
and reported back so nothing is silently guessed.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.explain import champion_contributions, reason_codes
from src.pricing import apply_grid
from src.scoring import champion_probability


class InputError(ValueError):
    """The request does not match the model's input contract."""


class ScoringService:
    def __init__(self, serving: dict):
        self.serving = serving
        self.bundle = serving["bundle"]
        self.columns: list[str] = list(self.bundle["columns"])
        self.categorical: list[str] = list(self.bundle["encoder"].columns)
        self.dtypes: dict[str, str] = serving["dtypes"]
        self.typical = pd.Series(serving["typical"], dtype=float)
        self.model_version: str = serving["model_version"]

    # ---------- input handling ----------
    def to_frame(self, records: list[dict]) -> tuple[pd.DataFrame, list[list[str]]]:
        """Validate and convert requests into a model-ready frame.
        Returns the frame and, per record, the features that were imputed."""
        unknown = sorted({k for r in records for k in r} - set(self.columns))
        if unknown:
            raise InputError(f"Unknown feature names: {unknown}. GET /model-info lists the accepted names.")
        frame = pd.DataFrame([{c: r.get(c) for c in self.columns} for r in records], columns=self.columns)
        imputed: list[list[str]] = [[] for _ in records]
        for col in self.columns:
            if col in self.categorical:
                frame[col] = frame[col].map(lambda v: None if v is None else str(v))
                continue
            try:
                frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
            except (ValueError, TypeError):
                raise InputError(f"Feature '{col}' must be a number.") from None
            if frame[col].map(lambda v: isinstance(v, float) and math.isinf(v)).any():
                raise InputError(f"Feature '{col}' must be finite.")
            if self.dtypes.get(col, "float").startswith(("int", "uint")):
                missing = frame[col].isna()
                for i in np.flatnonzero(missing.to_numpy()):
                    imputed[i].append(col)
                frame[col] = frame[col].fillna(self.typical.get(col, 0.0)).round().astype(self.dtypes[col])
        return frame, imputed

    # ---------- scoring ----------
    def score(self, records: list[dict], with_reasons: bool = True) -> list[dict]:
        if not records:
            return []
        raw, imputed = self.to_frame(records)
        X = self.bundle["encoder"].transform(raw)[self.columns]
        pd_ = self.serving["calibrator"].transform(champion_probability(self.bundle, X))
        bands, rates = apply_grid(pd_, self.serving["grid"])
        reasons = None
        if with_reasons:
            contrib = champion_contributions(self.bundle, X)
            reasons = reason_codes(contrib, X, self.typical, self.serving["labels"],
                                   top_k=self.serving["top_reasons"], categories=self.serving["categories"],
                                   binary=set(self.serving.get("binary_features", [])))
        out = []
        for i in range(len(records)):
            declined = bool(np.isnan(rates[i]))
            out.append({
                "pd": round(float(pd_[i]), 5),
                "risk_band": str(bands[i]),
                "decision": "decline" if declined else "approve",
                "interest_rate": None if declined else round(float(rates[i]), 4),
                "reasons": [r for r in reasons.iloc[i].tolist() if r] if reasons is not None else [],
                "imputed_features": imputed[i],
                "model_version": self.model_version,
            })
        return out

    # ---------- info ----------
    def info(self) -> dict:
        grid = self.serving["grid"]
        return {
            "model_version": self.model_version,
            "champion": self.bundle["champion"],
            "created_at": self.serving["created_at"],
            "features": [{"name": c, "type": "category" if c in self.categorical else "number",
                          "label": self.serving["labels"].get(c, c)} for c in self.columns],
            "categories": self.serving["categories"],
            "pricing_assumptions": self.serving["assumptions"],
            "pricing_grid": grid[["band", "pd_from", "pd_to", "rate", "decision"]].to_dict(orient="records"),
        }
