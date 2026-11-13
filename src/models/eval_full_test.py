"""Evaluate a trained DistilBERT checkpoint on the FULL test split (not the
training-time subsample) so its macro-F1 is directly comparable to the
baseline, which was evaluated on the full test set. Logs a corrected metric
to MLflow as a separate run and overwrites the confusion matrix/report.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..common.mlflow_utils import setup_mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"


def evaluate(target: str, batch_size: int = 32, max_length: int = 96) -> dict:
    final_dir = MODEL_DIR / f"distilbert_{target}" / "final"
    classes = np.load(MODEL_DIR / f"distilbert_{target}" / "label_classes.npy", allow_pickle=True)

    tokenizer = AutoTokenizer.from_pretrained(str(final_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(final_dir))
    model.eval()

    df = pd.read_parquet(PROCESSED_DIR / "tickets_test.parquet")
    df["text"] = (df["ticket_subject"].fillna("") + " " + df["description"].fillna("")).str.strip()

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch = df["text"].iloc[i : i + batch_size].tolist()
            inputs = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            logits = model(**inputs).logits
            preds = logits.argmax(-1).numpy()
            all_preds.extend(classes[preds])

    y_true = df[target].tolist()
    macro_f1 = f1_score(y_true, all_preds, average="macro")
    report = classification_report(y_true, all_preds)

    print(f"[{target}] FULL test set (n={len(df)}) macro_f1={macro_f1:.4f}")
    print(report)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    report_path = FIG_DIR / f"distilbert_{target}_classification_report.txt"
    report_path.write_text(report)

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, all_preds, ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"DistilBERT — {target} (full test set)")
    fig.tight_layout()
    cm_path = FIG_DIR / f"distilbert_{target}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    setup_mlflow("it-helpdesk-distilbert")
    with mlflow.start_run(run_name=f"distilbert_{target}_fulltest_eval"):
        mlflow.log_param("eval_set", "full_test")
        mlflow.log_param("n_test", len(df))
        mlflow.log_metric("test_macro_f1_full", macro_f1)
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(cm_path))

    return {"target": target, "test_macro_f1_full": macro_f1, "n_test": len(df)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["category", "priority"])
    args = parser.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    metrics_path = METRICS_DIR / "distilbert_metrics.json"
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text())

    for target in args.targets:
        final_dir = MODEL_DIR / f"distilbert_{target}" / "final"
        if not final_dir.exists():
            print(f"Skipping '{target}': no trained checkpoint at {final_dir}")
            continue
        result = evaluate(target)
        existing[target] = {"test_macro_f1_full": result["test_macro_f1_full"], "n_test": result["n_test"]}

    metrics_path.write_text(json.dumps(existing, indent=2))
    print(f"\nWrote {metrics_path}")
