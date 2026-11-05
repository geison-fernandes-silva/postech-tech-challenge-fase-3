from pathlib import Path

from fastapi.testclient import TestClient

from medical_triage import api


class FakePredictor:
    def predict(self, report_text: str):
        return "urgente", 0.97


def test_triage_returns_structured_prediction(monkeypatch):
    monkeypatch.setattr(api, "get_predictor", lambda: (FakePredictor(), "onnx"))
    client = TestClient(api.app)
    response = client.post(
        "/triage", json={"report_text": "Dor toracica intensa com alteracao isquemica no ECG."}
    )
    assert response.status_code == 200
    assert response.json()["urgency"] == "urgente"
    assert response.json()["model_backend"] == "onnx"


def test_compose_contains_observability_services():
    compose = (Path(__file__).resolve().parents[1] / "compose.yml").read_text(encoding="utf-8")
    assert "prometheus:" in compose
    assert "grafana:" in compose
