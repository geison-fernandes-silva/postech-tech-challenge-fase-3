"""Compare serialized Scikit-Learn and ONNX Runtime inference latency."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from medical_triage.model import OnnxPredictor, OriginalPredictor

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = "Dor toracica intensa com alteracao isquemica no ECG; acionar cardiologia imediatamente."


def measure(predictor, iterations: int = 500, warmup: int = 25) -> dict:
    for _ in range(warmup):
        predictor.predict(SAMPLE)
    latencies = []
    for _ in range(iterations):
        started = time.perf_counter()
        predictor.predict(SAMPLE)
        latencies.append((time.perf_counter() - started) * 1000)
    return {
        "p50_ms": round(statistics.median(latencies), 4),
        "p95_ms": round(sorted(latencies)[int(iterations * 0.95) - 1], 4),
        "mean_ms": round(statistics.mean(latencies), 4),
    }


def main() -> None:
    models = ROOT / "models"
    results = {
        "original_scikit_learn": measure(OriginalPredictor(models / "triage_original.joblib")),
        "optimized_onnx_runtime": measure(OnnxPredictor(models / "triage_optimized.onnx")),
    }
    original = results["original_scikit_learn"]["mean_ms"]
    optimized = results["optimized_onnx_runtime"]["mean_ms"]
    results["mean_latency_reduction_pct"] = (
        round((1 - optimized / original) * 100, 2) if original else 0
    )
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "latency_benchmark.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
