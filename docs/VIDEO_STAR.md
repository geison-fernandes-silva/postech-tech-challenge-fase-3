# Roteiro de vídeo - método STAR

## Situation (40 s)

Explique que a triagem de laudos textuais precisa indicar rapidamente quando um exame demanda atenção. Ressalte que a demonstração usa textos sintéticos e não substitui avaliação clínica.

## Task (45 s)

Apresente os requisitos: inferência real-time em FastAPI, container, CI/CD, DAG de retreino, monitoramento e otimização de latência.

## Action (2 min 30 s)

1. Mostre `POST /triage` no Swagger ou via `curl`.
2. Mostre o modelo original e o arquivo ONNX em `models/`.
3. Rode `python scripts/benchmark_latency.py` e abra o relatório em `reports/`.
4. Mostre o workflow `.github/workflows/ci.yml`.
5. Abra a DAG `medical_triage_retrain` no Airflow.
6. Abra o dashboard do Grafana e gere algumas requisições para atualizar os três painéis.

## Result (45 s)

Mostre a classificação, a métrica de latência do endpoint e os três indicadores no Grafana. Conclua que o projeto demonstra um ciclo reproduzível de treino, otimização, entrega e operação do modelo.
