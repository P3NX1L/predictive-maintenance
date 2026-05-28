"""FastAPI service exposing the predictive-maintenance model.

Endpoints:
  GET  /health            -> liveness + which model is loaded
  POST /predict           -> failure-within-horizon risk for one machine's history

Run locally:
  uvicorn predmaint.api:app --reload --port 8000
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .data import SENSOR_COLUMNS
from .predict import Predictor

app = FastAPI(
    title="Predictive Maintenance API",
    description="Failure-risk scoring for industrial machines from sensor time-series.",
    version="1.0.0",
)


class CycleReading(BaseModel):
    """One cycle of sensor readings for a machine."""

    cycle: int = Field(..., ge=0, description="Monotonic cycle index")
    temperature: float
    vibration: float
    spindle_load: float
    tool_wear: float = Field(..., ge=0)
    acoustic_emission: float


class PredictRequest(BaseModel):
    machine_id: int = Field(..., ge=0)
    history: list[CycleReading] = Field(
        ..., min_length=1, description="Recent cycles for ONE machine, any order"
    )

    @field_validator("history")
    @classmethod
    def _non_empty(cls, v: list[CycleReading]) -> list[CycleReading]:
        if not v:
            raise ValueError("history must contain at least one cycle")
        return v


class PredictResponse(BaseModel):
    machine_id: int
    failure_risk: float = Field(..., ge=0.0, le=1.0)
    model_name: str
    n_cycles_seen: int


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    model_path = Path("artifacts/model.pkl")
    if not model_path.exists():
        raise RuntimeError(
            "artifacts/model.pkl not found. Run `python -m predmaint.train` first."
        )
    return Predictor(model_path)


@app.get("/health")
def health() -> dict:
    try:
        predictor = get_predictor()
        return {"status": "ok", "model": predictor.name}
    except Exception as exc:  # model not trained yet
        return {"status": "degraded", "detail": str(exc)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        predictor = get_predictor()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    rows = []
    for r in req.history:
        row = {"machine_id": req.machine_id, "cycle": r.cycle}
        for s in SENSOR_COLUMNS:
            row[s] = getattr(r, s)
        rows.append(row)
    history_df = pd.DataFrame(rows)

    risk = predictor.predict_latest(history_df)
    return PredictResponse(
        machine_id=req.machine_id,
        failure_risk=risk,
        model_name=predictor.name,
        n_cycles_seen=len(req.history),
    )
