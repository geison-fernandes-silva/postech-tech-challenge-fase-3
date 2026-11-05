"""Generate synthetic data, train the baseline and export ONNX."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from medical_triage.data import make_dataset
from medical_triage.model import train_and_export

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dataset = make_dataset()
    (ROOT / "data").mkdir(exist_ok=True)
    dataset.to_csv(ROOT / "data" / "synthetic_medical_reports.csv", index=False)
    train, test = train_test_split(
        dataset, test_size=0.2, stratify=dataset["urgency"], random_state=42
    )
    result = train_and_export(
        train["report_text"],
        train["urgency"],
        test["report_text"],
        test["urgency"],
        ROOT / "models",
    )
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "training_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"Training complete. Accuracy: {result['accuracy']}")


if __name__ == "__main__":
    main()
