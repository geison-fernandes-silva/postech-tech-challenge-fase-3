"""FastAPI inference service with Prometheus instrumentation."""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from medical_triage.model import OnnxPredictor, OriginalPredictor

REQUESTS = Counter(
    "triage_requests_total", "Total triage inference requests", ["status", "urgency"]
)
ERRORS = Counter("triage_errors_total", "Total triage inference errors")
LATENCY = Histogram(
    "triage_request_duration_seconds",
    "Triage inference latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

app = FastAPI(title="Medical Triage NLP API", version="1.0.0")


class TriageRequest(BaseModel):
    report_text: str = Field(
        min_length=10, max_length=5_000, examples=["Dor toracica intensa com alteracao no ECG."]
    )


class TriageResponse(BaseModel):
    urgency: str
    confidence: float
    model_backend: str
    latency_ms: float


def get_predictor():
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    backend = os.getenv("MODEL_BACKEND", "onnx").lower()
    if backend == "original":
        return OriginalPredictor(model_dir / "triage_original.joblib"), backend
    return OnnxPredictor(model_dir / "triage_optimized.onnx"), "onnx"


@app.get("/health")
def health() -> dict:
    try:
        _, backend = get_predictor()
        return {"status": "ok", "backend": backend}
    except Exception as exc:  # pragma: no cover - service diagnostic
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc


@app.post("/triage", response_model=TriageResponse)
def triage(payload: TriageRequest) -> TriageResponse:
    started = time.perf_counter()
    try:
        predictor, backend = get_predictor()
        urgency, confidence = predictor.predict(payload.report_text)
        latency = time.perf_counter() - started
        LATENCY.observe(latency)
        REQUESTS.labels(status="success", urgency=urgency).inc()
        return TriageResponse(
            urgency=urgency,
            confidence=round(confidence, 4),
            model_backend=backend,
            latency_ms=round(latency * 1000, 3),
        )
    except Exception as exc:
        ERRORS.inc()
        REQUESTS.labels(status="error", urgency="unknown").inc()
        raise HTTPException(status_code=500, detail="Inference failed") from exc


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
