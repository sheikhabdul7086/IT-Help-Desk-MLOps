"""Phase 2: TF-IDF + Logistic Regression baseline, tracked in MLflow.

Trains two classifiers (category, priority) on the same TF-IDF features and
logs params/metrics/confusion-matrix artifacts to MLflow so the later
fine-tuned DistilBERT run can be benchmarked against this baseline.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

TARGETS = ["category", "priority"]


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / f"tickets_{name}.parquet")
    df["text"] = (df["ticket_subject"].fillna("") + " " + df["description"].fillna("")).str.strip()
    return df


def train_and_log(target: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    with mlflow.start_run(run_name=f"tfidf_logreg_{target}"):
        max_features = 20000
        ngram_range = (1, 2)
        C = 2.0

        mlflow.log_params(
            {
                "target": target,
                "model": "TF-IDF + LogisticRegression",
                "max_features": max_features,
                "ngram_range": str(ngram_range),
                "C": C,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
            }
        )

        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=2)),
                ("clf", LogisticRegression(max_iter=1000, C=C, class_weight="balanced")),
            ]
        )
        pipeline.fit(train_df["text"], train_df[target])

        val_preds = pipeline.predict(val_df["text"])
        test_preds = pipeline.predict(test_df["text"])

        val_macro_f1 = f1_score(val_df[target], val_preds, average="macro")
        test_macro_f1 = f1_score(test_df[target], test_preds, average="macro")

        mlflow.log_metrics(
            {
                "val_macro_f1": val_macro_f1,
                "test_macro_f1": test_macro_f1,
            }
        )

        report = classification_report(test_df[target], test_preds)
        report_path = FIG_DIR / f"baseline_{target}_classification_report.txt"
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path))

        fig, ax = plt.subplots(figsize=(8, 8))
        ConfusionMatrixDisplay.from_predictions(
            test_df[target], test_preds, ax=ax, xticks_rotation=45, colorbar=False
        )
        ax.set_title(f"Baseline TF-IDF+LogReg — {target} (test)")
        fig.tight_layout()
        cm_path = FIG_DIR / f"baseline_{target}_confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))

        mlflow.sklearn.log_model(pipeline, artifact_path=f"model_{target}")

        print(f"[{target}] val_macro_f1={val_macro_f1:.4f} test_macro_f1={test_macro_f1:.4f}")
        print(report)

        return {"target": target, "val_macro_f1": val_macro_f1, "test_macro_f1": test_macro_f1}


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("it-helpdesk-baseline")

    train_df, val_df, test_df = load_split("train"), load_split("val"), load_split("test")

    results = [train_and_log(target, train_df, val_df, test_df) for target in TARGETS]

    print("\nBaseline summary:")
    for r in results:
        print(f"  {r['target']}: test_macro_f1={r['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
