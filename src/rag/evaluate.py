"""Phase 4 step 3: evaluate generated resolutions against ground-truth resolution_notes.

Scores a sample of the test split with ROUGE-L (and a cheap embedding-cosine
similarity as a lightweight BERTScore stand-in, since full BERTScore needs a
large model that is not practical on this CPU-only box) and logs results to
MLflow for comparison against future prompt/model changes.
"""
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sentence_transformers import util

from ..common.mlflow_utils import setup_mlflow
from .config import load_config
from .generate import _get_embedder, suggest_resolution

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"


def evaluate(n_samples: int = 30, seed: int = 42):
    cfg = load_config()
    test_df = pd.read_parquet(PROCESSED_DIR / "tickets_test.parquet")
    test_df["text"] = (test_df["ticket_subject"].fillna("") + " " + test_df["description"].fillna("")).str.strip()
    sample = test_df.sample(n=min(n_samples, len(test_df)), random_state=seed)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    embedder = _get_embedder(cfg["embedding_model"])

    rouge_l_scores, cos_sims = [], []
    rows = []
    for row in sample.itertuples():
        result = suggest_resolution(row.text, row.category, row.priority, cfg=cfg)
        generated = result["suggestion"]
        reference = row.resolution_notes

        rl = scorer.score(reference, generated)["rougeL"].fmeasure
        rouge_l_scores.append(rl)

        emb = embedder.encode([generated, reference])
        cos_sim = util.cos_sim(emb[0], emb[1]).item()
        cos_sims.append(cos_sim)

        rows.append({"ticket_id": row.ticket_id, "rouge_l": rl, "cosine_sim": cos_sim})

    metrics = {
        "rouge_l_mean": float(np.mean(rouge_l_scores)),
        "cosine_sim_mean": float(np.mean(cos_sims)),
        "n_samples": len(sample),
        "llm_backend": cfg["llm_backend"],
    }

    setup_mlflow("it-helpdesk-rag")
    with mlflow.start_run(run_name=f"rag_eval_{cfg['llm_backend']}"):
        mlflow.log_params({"llm_backend": cfg["llm_backend"], "top_k": cfg["top_k"], "n_samples": len(sample)})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})

        detail_path = PROJECT_ROOT / "reports" / "figures" / "rag_eval_detail.csv"
        pd.DataFrame(rows).to_csv(detail_path, index=False)
        mlflow.log_artifact(str(detail_path))

    metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "rag_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("RAG evaluation:", metrics)
    return metrics


if __name__ == "__main__":
    evaluate()
