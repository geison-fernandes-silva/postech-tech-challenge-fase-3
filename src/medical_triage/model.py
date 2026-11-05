"""Training, export and inference helpers for the triage classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

LABELS = ["atencao", "normal", "urgente"]


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=1_500, ngram_range=(1, 2), sublinear_tf=True)),
            (
                "classifier",
                LogisticRegression(max_iter=400, class_weight="balanced", random_state=42),
            ),
        ]
    )


def train_and_export(train_texts, train_labels, test_texts, test_labels, model_dir: Path) -> dict:
    model_dir.mkdir(parents=True, exist_ok=True)
    pipeline = create_pipeline()
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts)
    joblib.dump(pipeline, model_dir / "triage_original.joblib")
    onnx_model = to_onnx(
        pipeline,
        initial_types=[("report_text", StringTensorType([None, 1]))],
        target_opset=15,
    )
    (model_dir / "triage_optimized.onnx").write_bytes(onnx_model.SerializeToString())
    return {
        "accuracy": round(float(accuracy_score(test_labels, predictions)), 4),
        "classification_report": classification_report(test_labels, predictions, output_dict=True),
        "classes": list(pipeline.classes_),
    }


class OriginalPredictor:
    def __init__(self, path: Path):
        self.model = joblib.load(path)

    def predict(self, report_text: str) -> tuple[str, float]:
        probabilities = self.model.predict_proba([report_text])[0]
        class_index = int(np.argmax(probabilities))
        return str(self.model.classes_[class_index]), float(probabilities[class_index])


class OnnxPredictor:
    def __init__(self, path: Path):
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        labels = self.session.run(None, {self.input_name: np.array([[""]], dtype=object)})[0]
        self.labels = [str(label) for label in labels] if len(labels) else LABELS

    def predict(self, report_text: str) -> tuple[str, float]:
        outputs = self.session.run(None, {self.input_name: np.array([[report_text]], dtype=object)})
        label = str(outputs[0][0])
        probability_map = outputs[1][0]
        return label, float(probability_map[label])
