"""Phase 3: fine-tune DistilBERT for ticket category and priority classification.

Logs every run to MLflow (params, per-epoch metrics, confusion matrix, model
artifact) and registers the category model in the MLflow Model Registry at
stage "Staging" so it can be promoted to "Production" after review, same as
the proposal's MLOps lifecycle calls for.

CPU-only fine-tuning is used (matches the proposal's cost-efficiency goal of
running on e2-micro/e2-small-class hardware), so training uses a subsample of
the full split to keep wall-clock time reasonable; the full train split is
still available for a GPU run via --full.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.transformers
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ..common.mlflow_utils import setup_mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
CHECKPOINT_NAME = "distilbert-base-uncased"

RANDOM_STATE = 42


class TicketDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / f"tickets_{name}.parquet")
    df["text"] = (df["ticket_subject"].fillna("") + " " + df["description"].fillna("")).str.strip()
    return df


def subsample(df: pd.DataFrame, target: str, n: int) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df
    frac = n / len(df)
    parts = [group.sample(frac=frac, random_state=RANDOM_STATE) for _, group in df.groupby(target)]
    return pd.concat(parts, ignore_index=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}


class WeightedTrainer(Trainer):
    """Trainer that applies class weights to the loss so minority classes
    aren't drowned out by majority ones (mirrors the baseline's
    class_weight="balanced"), since the subsampled training set is still
    class-imbalanced relative to the real distribution."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train_target(
    target: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int,
    max_length: int,
    batch_size: int,
):
    encoder = LabelEncoder()
    encoder.fit(train_df[target])
    num_labels = len(encoder.classes_)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT_NAME, num_labels=num_labels
    )

    def tokenize(df):
        enc = tokenizer(
            df["text"].tolist(), truncation=True, padding="max_length", max_length=max_length,
            return_tensors="pt",
        )
        labels = encoder.transform(df[target])
        return TicketDataset(enc, labels)

    train_ds = tokenize(train_df)
    val_ds = tokenize(val_df)
    test_ds = tokenize(test_df)

    train_labels_encoded = encoder.transform(train_df[target])
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(num_labels), y=train_labels_encoded
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    out_dir = MODEL_DIR / f"distilbert_{target}"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=20,
        report_to=[],
        warmup_steps=max(1, int(0.1 * (len(train_df) / batch_size) * epochs)),
        weight_decay=0.01,
        learning_rate=5e-5,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    with mlflow.start_run(run_name=f"distilbert_{target}"):
        mlflow.log_params(
            {
                "target": target,
                "model": CHECKPOINT_NAME,
                "num_labels": num_labels,
                "epochs": epochs,
                "max_length": max_length,
                "batch_size": batch_size,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "learning_rate": 5e-5,
            }
        )

        train_result = trainer.train()
        mlflow.log_metrics({f"train_{k}": v for k, v in train_result.metrics.items()})

        for log in trainer.state.log_history:
            if "eval_macro_f1" in log:
                mlflow.log_metric("val_macro_f1", log["eval_macro_f1"], step=int(log.get("epoch", 0)))
            if "loss" in log:
                mlflow.log_metric("train_loss", log["loss"], step=int(log.get("step", 0)))

        test_pred = trainer.predict(test_ds)
        test_preds = test_pred.predictions.argmax(-1)
        test_labels = test_pred.label_ids
        test_macro_f1 = f1_score(test_labels, test_preds, average="macro")
        mlflow.log_metric("test_macro_f1", test_macro_f1)

        report = classification_report(
            encoder.inverse_transform(test_labels), encoder.inverse_transform(test_preds)
        )
        report_path = FIG_DIR / f"distilbert_{target}_classification_report.txt"
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path))

        fig, ax = plt.subplots(figsize=(8, 8))
        ConfusionMatrixDisplay.from_predictions(
            encoder.inverse_transform(test_labels),
            encoder.inverse_transform(test_preds),
            ax=ax,
            xticks_rotation=45,
            colorbar=False,
        )
        ax.set_title(f"DistilBERT — {target} (test)")
        fig.tight_layout()
        cm_path = FIG_DIR / f"distilbert_{target}_confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))

        final_dir = out_dir / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        np.save(out_dir / "label_classes.npy", encoder.classes_)

        try:
            model_info = mlflow.transformers.log_model(
                transformers_model={"model": trainer.model, "tokenizer": tokenizer},
                name=f"distilbert_{target}",
                task="text-classification",
            )

            if target == "category":
                registered = mlflow.register_model(model_info.model_uri, "it-helpdesk-distilbert-category")
                client = mlflow.MlflowClient()
                client.transition_model_version_stage(
                    name="it-helpdesk-distilbert-category",
                    version=registered.version,
                    stage="Staging",
                )
                print(f"Registered it-helpdesk-distilbert-category v{registered.version} -> Staging")
        except Exception as e:  # noqa: BLE001 - packaging/logging issues must not lose a completed training run
            print(f"WARNING: mlflow model logging/registration failed for '{target}': {e}")
            print(f"Training + metrics are still saved; local checkpoint is at {final_dir}")

        print(f"[{target}] test_macro_f1={test_macro_f1:.4f}")
        print(report)

        return {"target": target, "test_macro_f1": test_macro_f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["category", "priority"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-n", type=int, default=6000, help="subsample size for train split")
    parser.add_argument("--val-n", type=int, default=1000)
    parser.add_argument("--test-n", type=int, default=1000)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    setup_mlflow("it-helpdesk-distilbert")

    results = []
    for target in args.targets:
        train_df = subsample(load_split("train"), target, args.train_n)
        val_df = subsample(load_split("val"), target, args.val_n)
        test_df = subsample(load_split("test"), target, args.test_n)

        print(f"\n=== Training DistilBERT for '{target}' "
              f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)}) ===")
        results.append(
            train_target(target, train_df, val_df, test_df, args.epochs, args.max_length, args.batch_size)
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_json = {r["target"]: {"test_macro_f1": r["test_macro_f1"]} for r in results}
    (METRICS_DIR / "distilbert_metrics.json").write_text(json.dumps(metrics_json, indent=2))

    print("\nDistilBERT summary:")
    for r in results:
        print(f"  {r['target']}: test_macro_f1={r['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
