"""The scoring service and API: correct, consistent with the analysis, and strict about input."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.monitoring import psi, status
from src.scoring import champion_probability
from src.serving import InputError, ScoringService
from tests.serving_helpers import RISKY, SAFE, make_serving


@pytest.fixture(scope="module")
def built():
    return make_serving()


@pytest.fixture
def service(built):
    return ScoringService(built[0])


# ---------- service ----------

def test_service_matches_the_analysis_code_exactly(built, service):
    serving, raw, _ = built
    records = raw.head(50).to_dict(orient="records")
    from_service = np.array([r["pd"] for r in service.score(records, with_reasons=False)])
    X = serving["bundle"]["encoder"].transform(raw.head(50))[serving["bundle"]["columns"]]
    direct = serving["calibrator"].transform(champion_probability(serving["bundle"], X))
    assert np.allclose(from_service, np.round(direct, 5))


def test_safe_applicant_is_approved_with_a_rate(service):
    r = service.score([SAFE])[0]
    assert r["decision"] == "approve" and r["risk_band"] == "A"
    assert 0 < r["pd"] < 0.05 and r["interest_rate"] is not None


def test_risky_applicant_is_declined_with_reasons(service):
    r = service.score([RISKY])[0]
    assert r["decision"] == "decline" and r["interest_rate"] is None
    assert 1 <= len(r["reasons"]) <= 3
    assert any("External credit score" in reason for reason in r["reasons"])


def test_unknown_feature_is_rejected(service):
    with pytest.raises(InputError, match="Unknown feature"):
        service.score([{**SAFE, "favourite_colour": 3}])


def test_text_in_a_numeric_feature_is_rejected(service):
    with pytest.raises(InputError, match="must be a number"):
        service.score([{**SAFE, "ext_source_mean": "high"}])


def test_missing_values_are_allowed_and_count_flags_are_reported(service):
    r = service.score([{"ext_source_mean": None, "income_type": "Working"}])[0]
    assert r["imputed_features"] == ["thin_file"]   # integer flag filled with typical value
    assert 0 < r["pd"] < 1                          # float features left missing for the model


def test_unseen_category_does_not_crash(service):
    r = service.score([{**SAFE, "income_type": "Astronaut"}])[0]
    assert 0 < r["pd"] < 1


def test_batch_keeps_order(service):
    results = service.score([SAFE, RISKY, SAFE])
    assert [r["decision"] for r in results] == ["approve", "decline", "approve"]


# ---------- API ----------

@pytest.fixture
def client(built, tmp_path):
    app.state.service = ScoringService(built[0])
    app.state.score_log = tmp_path / "scores.jsonl"
    with TestClient(app) as c:
        yield c
    app.state.service = None
    app.state.score_log = None


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "test-1"}
    info = client.get("/model-info").json()
    assert [f["name"] for f in info["features"]] == ["ext_source_mean", "inst_late_rate", "thin_file", "income_type"]
    assert {b["band"] for b in info["pricing_grid"]} == {"A", "B", "C"}


def test_example_round_trip(client):
    example = client.get("/example").json()
    response = client.post("/score", json=example)
    assert response.status_code == 200
    body = response.json()
    assert body["applicant_id"] == "typical-applicant"
    assert set(body) >= {"pd", "risk_band", "decision", "interest_rate", "reasons", "model_version"}


def test_bad_input_returns_422_with_a_clear_message(client):
    response = client.post("/score", json={"features": {**SAFE, "favourite_colour": 3}})
    assert response.status_code == 422
    assert "favourite_colour" in response.json()["detail"]
    assert client.post("/score", json={"applicant_id": "x"}).status_code == 422   # features missing


def test_batch_and_score_log(client):
    response = client.post("/score/batch", json=[{"applicant_id": "a", "features": SAFE},
                                                 {"applicant_id": "b", "features": RISKY}])
    assert [r["decision"] for r in response.json()] == ["approve", "decline"]
    entries = [json.loads(line) for line in app.state.score_log.read_text().splitlines()]
    assert [e["applicant_id"] for e in entries] == ["a", "b"]
    assert set(entries[0]["features"]) == {"ext_source_mean", "inst_late_rate"}   # monitored features only


def test_batch_size_limit(client):
    response = client.post("/score/batch", json=[{"features": SAFE}] * 1001)
    assert response.status_code == 422


def test_no_model_gives_503(tmp_path):
    os.environ["MODEL_PATH"] = str(tmp_path / "missing.joblib")
    try:
        app.state.service, app.state.score_log = None, tmp_path / "s.jsonl"
        with TestClient(app) as c:
            assert c.get("/health").json()["status"] == "model not loaded"
            assert c.post("/score", json={"features": SAFE}).status_code == 503
    finally:
        del os.environ["MODEL_PATH"]
        app.state.service = None
        app.state.score_log = None


# ---------- monitoring ----------

def test_psi_is_near_zero_for_the_same_population():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=20000), rng.normal(size=20000)) < 0.01


def test_psi_flags_a_shifted_population():
    rng = np.random.default_rng(0)
    value = psi(rng.normal(size=20000), rng.normal(0.8, 1, size=20000))
    assert value > 0.25 and status(value) == "ALERT"


def test_psi_notices_more_missing_values():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=10000)
    cur = rng.normal(size=10000)
    cur[:3000] = np.nan
    assert psi(ref, cur) > 0.25
