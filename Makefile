.PHONY: setup train test lint up down benchmark

setup:
	uv sync --all-groups

train:
	uv run python scripts/train_model.py

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts dags

benchmark:
	uv run python scripts/benchmark_latency.py

up:
	podman compose up --build -d

down:
	podman compose down
