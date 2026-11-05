# Medical Report Triage - Tech Challenge Fase 3

Sistema acadêmico de triagem automática de laudos médicos em texto. A API recebe um laudo, classifica sua urgência como `normal`, `atencao` ou `urgente` e expõe indicadores operacionais para observabilidade.

> **Aviso:** esta é uma demonstração educacional com textos sintéticos, sem pacientes ou informações de saúde reais. Não deve ser usada para decisões clínicas.

## Objetivo e requisitos atendidos

| Requisito da Fase 3 | Implementação |
|---|---|
| NLP leve | TF-IDF + Logistic Regression com Scikit-Learn |
| API REST | FastAPI com `POST /triage` |
| Container | `Dockerfile` funcional e compatível com Docker/Podman |
| CI/CD | GitHub Actions: lint, testes e build da imagem |
| Retreino | DAG Airflow com treino, exportação ONNX e benchmark |
| Monitoramento | Prometheus + Grafana via Compose |
| Dashboard | 3 painéis: total de requisições, latência P95 e taxa de erro |
| Otimização | Exportação do classificador para ONNX Runtime |
| Latência | Script reproduzível comparando Scikit-Learn e ONNX |

## Decisão arquitetural de cloud

O caso exige resposta durante a triagem, portanto foi escolhida uma arquitetura **real-time**, e não batch. Em uma nuvem, a API seria implantada atrás de um balanceador com autoscaling e conectada a um registro de modelos. O Prometheus/Grafana atenderia a camada de observabilidade. O Airflow permanece separado da rota de inferência: ele executa retreinos controlados, valida o resultado e só então publica um novo artefato do modelo.

```text
Laudo de texto -> FastAPI -> ONNX Runtime -> classificação de urgência
                    |              |
                    v              v
              /metrics       Prometheus -> Grafana

Airflow -> dados sintéticos -> treino Scikit-Learn -> ONNX + benchmark
```

## Dados

O projeto gera uma base determinística de **2.400 laudos sintéticos** (800 por classe). Isto torna a entrega reprodutível sem exigir dados clínicos ou credenciais externas. O arquivo gerado fica em `data/synthetic_medical_reports.csv` e não é versionado no Git.

## Estrutura

```text
src/medical_triage/        API, dados e modelo
scripts/                   treinamento e benchmark de latência
dags/                      DAG do Airflow
monitoring/                Prometheus e dashboard provisionado do Grafana
tests/                     testes automatizados
.github/workflows/ci.yml   lint + teste + build automático
```

## Execução local

Pré-requisitos: Python 3.11+, [uv](https://docs.astral.sh/uv/) e Podman ou Docker Compose.

```bash
uv sync --all-groups
uv run python scripts/train_model.py
uv run python scripts/benchmark_latency.py
uv run pytest -q
uv run ruff check src tests scripts dags
```

O treinamento cria `models/triage_original.joblib`, `models/triage_optimized.onnx`, `reports/training_metrics.json` e `reports/latency_benchmark.json`.

### API local sem container

```bash
MODEL_BACKEND=onnx uv run uvicorn medical_triage.api:app --app-dir src --reload
```

Exemplo de chamada:

```bash
curl -X POST http://127.0.0.1:8000/triage \
  -H 'Content-Type: application/json' \
  -d '{"report_text":"Dor toracica intensa com alteracao isquemica no ECG; acionar cardiologia imediatamente."}'
```

### Stack de observabilidade com Podman

Após treinar os modelos:

```bash
podman compose up --build -d
```

Alternativamente, use `docker compose up --build -d`.

| Serviço | Endereço |
|---|---|
| API e OpenAPI | http://localhost:18003/docs |
| Métricas | http://localhost:18003/metrics |
| Prometheus | http://localhost:19090 |
| Grafana | http://localhost:13000 (admin/admin) |

Gere chamadas para popular o dashboard:

```bash
for i in {1..20}; do
  curl -s -X POST http://127.0.0.1:18003/triage -H 'Content-Type: application/json' \
    -d '{"report_text":"Dor toracica intensa com alteracao isquemica no ECG; acionar cardiologia imediatamente."}' > /dev/null
done
```

## Airflow

Copie `dags/medical_triage_retrain.py` para a pasta de DAGs de uma instalação Airflow com as dependências do projeto disponíveis. A DAG possui duas etapas:

1. gera os dados sintéticos, treina o classificador e exporta o artefato ONNX;
2. executa o benchmark comparativo e grava o relatório.

## Latência e otimização

O baseline usa diretamente o pipeline serializado do Scikit-Learn. A versão otimizada exporta o mesmo pipeline para **ONNX Runtime**, reduzindo a sobrecarga de inferência e usando um runtime voltado a execução. Os resultados dependem do computador e ficam registrados em `reports/latency_benchmark.json`, com média, P50, P95 e redução percentual. A comparação só deve ser apresentada após executar o script localmente.

## CI/CD

Todo `push` ou `pull request` executa automaticamente:

1. `ruff check`;
2. `pytest`;
3. build da imagem Docker.

O histórico esperado para o repositório usa commits semânticos, por exemplo: `feat: add onnx triage inference`, `test: cover urgency endpoint` e `docs: add observability runbook`.

## Roteiro STAR para o vídeo

Um roteiro de até cinco minutos está em [docs/VIDEO_STAR.md](docs/VIDEO_STAR.md). Ele orienta a demonstração da API, CI/CD, Airflow, Grafana e benchmark de latência.
