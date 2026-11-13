# GenAI-Powered Intelligent IT Helpdesk & Automated Resolution System

MLOps course project (Milestone: reproducible core pipeline — no Docker/cloud
deployment yet). Dual-tier architecture:

- **Tier 1 — Triage:** fine-tuned DistilBERT classifies ticket `category` and
  `priority`. A TF-IDF + Logistic Regression model is kept as the tracked
  baseline so DistilBERT's gain is measurable, not assumed.
- **Tier 2 — Resolution co-pilot:** a RAG pipeline (sentence-transformer
  embeddings -> ChromaDB -> an instruction-tuned generator) drafts a
  structured troubleshooting playbook for each new ticket, grounded in
  similar past resolutions.
- **Serving:** a local FastAPI app (`/predict`, `/suggest-resolution`,
  `/feedback`) plus a one-page dashboard for manual testing.

## Team

Syed Abdullah Ijaz Shah (23L-1000) · Muhammad Abdullah Kamal Batta (23L-0898)

## Dataset

[Customer IT Support - Ticket Dataset](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)
(Tobias Bueck, Kaggle, CC BY 4.0, synthetic-but-realistic). 28,587 tickets with
subject, body, agent answer, queue (-> `category`), priority, type, tags.
Note: this dataset only has low/medium/high priority (no "critical" tier).

Raw CSVs are **not** committed to git — see `data/raw.dvc` / DVC below.

## Project layout

```
project/
  data/
    raw/            # original Kaggle CSVs (DVC-tracked)
    interim/        # ingested intermediate parquet
    processed/      # cleaned + train/val/test splits (DVC-tracked via dvc.yaml)
    vector_store/   # ChromaDB persistent index for RAG (DVC-tracked)
  src/
    data/           # ingest, clean, validate, eda, split  (Phase 1)
    models/         # baseline + DistilBERT training        (Phase 2-3)
    rag/            # chunk/embed/index + generate + eval    (Phase 4)
    api/            # FastAPI app, DB, inference, dashboard  (Phase 5)
  models/           # trained model artifacts (joblib + HF checkpoints) — gitignored
  reports/
    figures/        # EDA plots, confusion matrices
    metrics/        # DVC-tracked metrics JSON per stage
  configs/           # rag.yaml (retrieval + prompt config, DVC-versioned)
  tests/             # pytest unit tests
  dvc.yaml           # pipeline stages (dvc repro runs everything)
  mlruns/            # local MLflow tracking store (sqlite)
```

## Setup

```bash
cd project
python -m venv .venv
.venv/Scripts/activate        # Windows; `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

## Running the pipeline

Run everything, fully reproducible, with DVC:

```bash
dvc repro
```

This executes, in dependency order:

| Stage | What it does | Key outputs |
|---|---|---|
| `data_pipeline` | ingest -> sanitize/dedupe -> validate (schema/nulls/ranges) -> EDA -> stratified 70/15/15 split | `data/processed/tickets_{train,val,test}.parquet` |
| `baseline` | TF-IDF + Logistic Regression for `category` and `priority`, logged to MLflow | `models/baseline_*.joblib`, `reports/metrics/baseline_metrics.json` |
| `distilbert` | fine-tunes `distilbert-base-uncased` for `category` and `priority`, logged + registered in MLflow Model Registry (stage: Staging) | `models/distilbert_*/final`, `reports/metrics/distilbert_metrics.json` |
| `rag_index` | embeds resolved tickets (`sentence-transformers/all-MiniLM-L6-v2`) and indexes them in ChromaDB | `data/vector_store/` |
| `rag_eval` | generates resolutions for a test sample, scores ROUGE-L + embedding cosine similarity vs. ground truth | `reports/metrics/rag_metrics.json` |

Or run any stage individually, e.g.:

```bash
python -m src.data.run_phase1
python -m src.models.baseline
python -m src.models.distilbert_train            # CPU fine-tune, subsampled for speed — see --help
python -m src.rag.build_index
python -m src.rag.evaluate
```

Check metrics at any point:

```bash
dvc metrics show
```

### MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Open http://127.0.0.1:5000 to see every run's params/metrics/artifacts and the
registered `it-helpdesk-distilbert-category` model (Staging).

### Serving locally (Phase 5)

```bash
uvicorn src.api.main:app --reload --port 8000
```

Open http://127.0.0.1:8000 for the test dashboard, or call the API directly:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"subject":"VPN disconnects","description":"My VPN client keeps dropping every few minutes."}'
```

Data persists to a local SQLite file (`data/helpdesk.db`). Set `DATABASE_URL`
to a `postgresql://...` URL to switch to PostgreSQL without code changes
(this submission uses SQLite to avoid requiring a local Postgres server).

## Data & experiment versioning

- **DVC** tracks `data/raw`, `data/processed`, and `data/vector_store` by
  content hash (`*.dvc` files / `dvc.lock`), and `configs/rag.yaml` is a
  tracked pipeline dependency so prompt/retrieval changes are versioned too.
  Fully reproducible locally via `dvc repro` with no remote configured.
- **MLflow** logs every baseline/DistilBERT/RAG run (params, metrics,
  confusion matrices, model artifacts). By default it logs to a local
  SQLite-backed store (`mlruns/mlflow.db`); set `MLFLOW_TRACKING_URI` (and
  `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD`) in `project/.env` to
  point at a hosted server instead — no code changes needed.

### Connecting a DagsHub remote (DVC storage + hosted MLflow)

1. On [dagshub.com](https://dagshub.com), **Create → Connect a repo** and
   pick this project's GitHub repo. DagsHub mirrors it and gives it a DVC
   remote and an MLflow tracking server at the same URL.
2. Generate a token: avatar → **Your Settings → Tokens → Generate New Token**.
3. Point DVC at it (credentials go to `.dvc/config.local`, already
   git-ignored — never committed):
   ```bash
   dvc remote add origin https://dagshub.com/<user>/<repo>.dvc
   dvc remote modify origin --local auth basic
   dvc remote modify origin --local user <user>
   dvc remote modify origin --local password <token>
   dvc push
   ```
4. Add to `project/.env` (already git-ignored):
   ```
   MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
   MLFLOW_TRACKING_USERNAME=<user>
   MLFLOW_TRACKING_PASSWORD=<token>
   ```
   Future `dvc repro` / training runs will log to DagsHub automatically.

## Tests

```bash
pytest tests/ -v
```
