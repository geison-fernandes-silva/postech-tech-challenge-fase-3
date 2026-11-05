# Checklist de entrega - Tech Challenge Fase 3

## Repositório

- [x] API FastAPI para classificação de laudo em `normal`, `atencao` ou `urgente`.
- [x] Dockerfile funcional.
- [x] Compose com API, Prometheus e Grafana.
- [x] Workflow GitHub Actions com lint, testes e build.
- [x] DAG Airflow de treino, exportação ONNX e benchmark.
- [x] Dashboard Grafana provisionado com três painéis.
- [x] Modelo Scikit-Learn e modelo ONNX versionados em `models/`.
- [x] Script de benchmark de latência.
- [x] README com decisão de arquitetura real-time e instruções de execução.
- [x] Roteiro para vídeo STAR.

## Antes de enviar

1. Execute `make test`, `make lint`, `make train` e `make benchmark`.
2. Execute `podman compose up --build -d` e abra Grafana em `http://localhost:13000`.
3. Faça ao menos 20 chamadas ao endpoint e capture o dashboard com os três painéis preenchidos.
4. Execute ou apresente a DAG `medical_triage_retrain` no Airflow.
5. Grave o vídeo de no máximo cinco minutos seguindo `docs/VIDEO_STAR.md`.
6. Publique o diretório em um repositório GitHub e confirme que os checks do workflow estão verdes.
