FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY models ./models
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "medical_triage.api:app", "--host", "0.0.0.0", "--port", "8000"]
