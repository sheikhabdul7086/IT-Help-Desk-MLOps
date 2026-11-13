"""Model loading + prediction for the API layer.

Prefers the fine-tuned DistilBERT model for each target (category, priority)
if it has been trained (models/distilbert_<target>/final exists); otherwise
falls back to the TF-IDF+LogReg baseline (models/baseline_<target>.joblib).
This mirrors the proposal's two-tier design while keeping the API usable
even before DistilBERT has been trained.
"""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

TARGETS = ["category", "priority"]


@lru_cache(maxsize=4)
def _load_distilbert(target: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    final_dir = MODEL_DIR / f"distilbert_{target}" / "final"
    classes_path = MODEL_DIR / f"distilbert_{target}" / "label_classes.npy"
    if not final_dir.exists() or not classes_path.exists():
        return None

    tokenizer = AutoTokenizer.from_pretrained(str(final_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(final_dir))
    model.eval()
    classes = np.load(classes_path, allow_pickle=True)
    return {"tokenizer": tokenizer, "model": model, "classes": classes, "torch": torch}


@lru_cache(maxsize=4)
def _load_baseline(target: str):
    path = MODEL_DIR / f"baseline_{target}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def predict_one(text: str, target: str) -> dict:
    distil = _load_distilbert(target)
    if distil is not None:
        torch = distil["torch"]
        inputs = distil["tokenizer"](text, truncation=True, padding=True, max_length=96, return_tensors="pt")
        with torch.no_grad():
            logits = distil["model"](**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].numpy()
        label = distil["classes"][int(np.argmax(probs))]
        confidence = float(np.max(probs))
        return {"label": str(label), "confidence": confidence, "model": "distilbert"}

    baseline = _load_baseline(target)
    if baseline is not None:
        label = baseline.predict([text])[0]
        proba = baseline.predict_proba([text])[0]
        confidence = float(np.max(proba))
        return {"label": str(label), "confidence": confidence, "model": "tfidf_logreg"}

    raise RuntimeError(
        f"No trained model found for target '{target}'. Run src.models.baseline "
        f"and/or src.models.distilbert_train first."
    )


def predict_ticket(text: str) -> dict:
    return {target: predict_one(text, target) for target in TARGETS}
