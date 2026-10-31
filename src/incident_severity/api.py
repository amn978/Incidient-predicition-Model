"""FastAPI service.  Run:  uvicorn incident_severity.api:app --port 8000

Endpoints: GET /health, GET /model-info, POST /predict, POST /predict/batch, POST /feedback, GET /feedback/summary
Optional auth: set env ISP_API_KEY and send header  X-API-Key.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .config import load_config
from .predict import SeverityPredictor
from .storage import feedback_summary, log_feedback, log_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("isp.api")


class Ticket(BaseModel):
    ticket_id: Optional[str] = None
    short_description: str = Field(..., min_length=3, examples=["Primary database is down for checkout-service"])
    description: str = ""
    category: Optional[str] = None
    subcategory: Optional[str] = None
    affected_service: Optional[str] = Field(None, examples=["checkout-service"])
    environment: str = Field("prod", pattern="^(prod|staging|dev)$")
    channel: str = "portal"
    users_affected: Optional[int] = Field(None, ge=0)
    created_at: Optional[datetime] = None


class Feedback(BaseModel):
    ticket_id: str
    predicted: str
    final_severity: str
    reviewer: str = ""
    reason: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.cfg = cfg
    app.state.predictor = SeverityPredictor(cfg)
    log.info("Loaded model %s v%s", app.state.predictor.meta["model_name"], app.state.predictor.meta["version"])
    yield


app = FastAPI(title="Incident Severity API", version="0.1.0", lifespan=lifespan)


def check_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("ISP_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.get("/health")
def health():
    return {"status": "ok", "model": app.state.predictor.meta["model_name"]}


@app.get("/model-info", dependencies=[Depends(check_key)])
def model_info():
    return app.state.predictor.meta


@app.post("/predict", dependencies=[Depends(check_key)])
def predict(ticket: Ticket):
    rec = app.state.predictor.predict([ticket.model_dump()])[0]
    try:
        log_prediction(app.state.cfg["paths"]["feedback_db"], rec, ticket.short_description)
    except Exception:                      # logging must never break triage
        log.exception("prediction logging failed")
    return rec


@app.post("/predict/batch", dependencies=[Depends(check_key)])
def predict_batch(tickets: list[Ticket]):
    if not tickets or len(tickets) > 500:
        raise HTTPException(status_code=422, detail="send between 1 and 500 tickets")
    return app.state.predictor.predict([t.model_dump() for t in tickets])


@app.post("/feedback", dependencies=[Depends(check_key)])
def feedback(fb: Feedback):
    classes = app.state.predictor.classes
    if fb.final_severity not in classes or fb.predicted not in classes:
        raise HTTPException(status_code=422, detail=f"severity must be one of {classes}")
    log_feedback(app.state.cfg["paths"]["feedback_db"], **fb.model_dump())
    return {"status": "stored"}


@app.get("/feedback/summary", dependencies=[Depends(check_key)])
def summary():
    return feedback_summary(app.state.cfg["paths"]["feedback_db"], app.state.predictor.classes)
