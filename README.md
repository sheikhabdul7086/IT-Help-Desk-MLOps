# GenAI-Powered Intelligent IT Helpdesk & Automated Resolution System

MLOps course project. Dual-tier architecture: fine-tuned DistilBERT for ticket
classification/priority scoring, plus an instruction-tuned LLM + RAG for
auto-generated resolution playbooks. TF-IDF + Logistic Regression is kept as
a tracked baseline in MLflow.

## Project layout

```
project/
  data/
    raw/        # original Kaggle CSVs (DVC-tracked, not in git)
    interim/     # ingested/intermediate data
    processed/   # cleaned + train/val/test splits
  src/
    data/        # ingestion, cleaning, EDA, splitting (Phase 1)
    models/      # baseline + DistilBERT training (Phase 2-3)
    api/         # FastAPI serving layer (Phase 5)
    monitoring/  # Evidently AI drift checks (Phase 7/10)
  notebooks/     # exploratory notebooks
  reports/figures/  # EDA plots, confusion matrices
  configs/       # yaml/json config files
  tests/         # pytest unit tests
```

## Dataset

[Customer IT Support - Ticket Dataset](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)
(Tobias Bueck, Kaggle, CC BY 4.0). 28,587 real-world-style support tickets with
subject, body, agent answer, queue (category), priority, type, and tags.
Primary file: `aa_dataset-tickets-multi-lang-5-2-50-version.csv`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
```

## Phase 1 — Data pipeline

```bash
python -m src.data.run_phase1
```

Runs: ingest raw CSV -> sanitize/dedup -> EDA (`reports/figures/eda_overview.png`)
-> stratified 70/15/15 split -> `data/processed/tickets_{train,val,test}.parquet`.

## Phase 2 — Baseline model

```bash
python -m src.models.baseline
```

Trains TF-IDF + Logistic Regression for category classification, logs
params/metrics/confusion matrix to MLflow (`mlruns/`).
