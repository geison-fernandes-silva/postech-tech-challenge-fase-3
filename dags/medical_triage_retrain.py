"""Airflow DAG: generate data, train, export optimized model and benchmark."""

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = Path(__file__).resolve().parents[1]

with DAG(
    dag_id="medical_triage_retrain",
    description="Academic retraining pipeline for the medical triage NLP model",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["tech-challenge", "nlp", "medical-triage"],
) as dag:
    train = BashOperator(
        task_id="generate_data_train_and_export_onnx",
        bash_command=f"cd {PROJECT_DIR} && python scripts/train_model.py",
    )
    benchmark = BashOperator(
        task_id="compare_original_and_onnx_latency",
        bash_command=f"cd {PROJECT_DIR} && python scripts/benchmark_latency.py",
    )
    train >> benchmark
