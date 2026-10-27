"""Credit risk scoring and loan pricing API.

    uvicorn src.api:app --port 8000
    then open http://127.0.0.1:8000/docs for interactive documentation.

Endpoints
  GET  /health        is the model loaded, and which version
  GET  /model-info    accepted features, price grid, pricing assumptions
  GET  /example       a typical applicant to start from (edit and send to /score)
  POST /score         one applicant -> PD, band, decision, rate, top reasons
  POST /score/batch   up to 1,000 applicants

Every scored applicant is appended to a JSON-lines log (SCORE_LOG) with the PD
and the monitored features, which `python -m src.monitor --from-log` uses to
check for drift. No identifying data beyond the caller's applicant_id is stored.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Union

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.serving import InputError, ScoringService

log = logging.getLogger("api")
MAX_BATCH = 1000
_log_lock = threading.Lock()

Value = Optional[Union[float, int, str]]


class ScoreRequest(BaseModel):
    applicant_id: Optional[str] = Field(None, description="Your reference for this applicant (optional)")
    features: dict[str, Value] = Field(..., description="Model features by name; see GET /model-info")

    model_config = {"json_schema_extra": {"examples": [{
        "applicant_id": "demo-001",
        "features": {"ext_source_mean": 0.31, "annuity_to_income_ratio": 0.22, "years_employed": 1.5,
                     "inst_late_rate": 0.25, "income_type": "Working"},
    }]}}


class ScoreResponse(BaseModel):
    applicant_id: Optional[str]
    pd: float = Field(..., description="Calibrated probability of payment difficulties")
    risk_band: str
    decision: Literal["approve", "decline"]
    interest_rate: Optional[float] = Field(None, description="Annual rate offered; null when declined")
    reasons: list[str] = Field(..., description="Top factors that raised this applicant's risk")
    imputed_features: list[str] = Field(..., description="Count/flag features missing from the request, "
                                                          "filled with the typical value")
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "service", None) is None:
        path = Path(os.getenv("MODEL_PATH", "models/serving_bundle.joblib"))
        if path.exists():
            app.state.service = ScoringService(joblib.load(path))
            log.info("Loaded model %s from %s", app.state.service.model_version, path)
        else:
            app.state.service = None
            log.error("No serving bundle at %s. Run `python -m src.build_serving`.", path)
    if getattr(app.state, "score_log", None) is None:
        app.state.score_log = Path(os.getenv("SCORE_LOG", "logs/scores.jsonl"))
    yield


app = FastAPI(
    title="Credit risk scoring and loan pricing API",
    description="Probability of default, risk band, approve/decline, risk-based interest rate and "
                "plain-language reasons for each applicant. Built on a tuned LightGBM + XGBoost ensemble "
                "with calibrated probabilities and TreeSHAP reason codes.",
    version="1.0.0",
    lifespan=lifespan,
)


def _service(request: Request) -> ScoringService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `python -m src.build_serving`.")
    return service


def _log_scores(request: Request, service: ScoringService, payloads: list[ScoreRequest], results: list[dict]) -> None:
    path: Path = request.app.state.score_log
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    monitored = service.serving["monitored_features"]
    with _log_lock, open(path, "a", encoding="utf-8") as f:
        for p, r in zip(payloads, results):
            f.write(json.dumps({
                "scored_at": now, "applicant_id": p.applicant_id, "model_version": r["model_version"],
                "pd": r["pd"], "risk_band": r["risk_band"], "decision": r["decision"],
                "features": {k: p.features.get(k) for k in monitored},
            }) + "\n")


def _score(request: Request, payloads: list[ScoreRequest]) -> list[ScoreResponse]:
    service = _service(request)
    try:
        results = service.score([p.features for p in payloads])
    except InputError as err:
        raise HTTPException(status_code=422, detail=str(err)) from None
    _log_scores(request, service, payloads, results)
    return [ScoreResponse(applicant_id=p.applicant_id, **r) for p, r in zip(payloads, results)]


@app.get("/health")
def health(request: Request) -> dict:
    service = getattr(request.app.state, "service", None)
    return {"status": "ok" if service else "model not loaded",
            "model_version": service.model_version if service else None}


@app.get("/model-info")
def model_info(request: Request) -> dict:
    return _service(request).info()


@app.get("/example", response_model=ScoreRequest)
def example(request: Request) -> ScoreRequest:
    service = _service(request)
    return ScoreRequest(applicant_id="typical-applicant", features=service.serving["example"])


@app.post("/score", response_model=ScoreResponse)
def score(request: Request, payload: ScoreRequest) -> ScoreResponse:
    return _score(request, [payload])[0]


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(request: Request, payloads: list[ScoreRequest]) -> list[ScoreResponse]:
    if len(payloads) > MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"At most {MAX_BATCH} applicants per batch.")
    return _score(request, payloads)
