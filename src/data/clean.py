"""Phase 1 step 2: sanitize and deduplicate tickets (PII/HTML removal, null handling)."""
import re
from pathlib import Path

import pandas as pd

from .ingest import INTERIM_DIR, ingest

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

HTML_TAG_RE = re.compile(r"<[^>]+>")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4})")
URL_RE = re.compile(r"https?://\S+")
WHITESPACE_RE = re.compile(r"\s+")
PLACEHOLDER_NAME_RE = re.compile(r"<name>", re.IGNORECASE)


def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub("[URL]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PLACEHOLDER_NAME_RE.sub("[NAME]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["ticket_subject", "description", "resolution_notes"]:
        df[col] = df[col].apply(sanitize_text)

    # drop tickets with no body text or no resolution notes (needed for the RAG/LLM corpus)
    df = df[df["description"].str.len() > 0]
    df = df[df["resolution_notes"].str.len() > 0]

    # fill missing subjects from the first words of the body instead of dropping the row
    empty_subject = df["ticket_subject"].str.len() == 0
    df.loc[empty_subject, "ticket_subject"] = df.loc[empty_subject, "description"].str.slice(0, 60)

    df = df.drop_duplicates(subset=["description", "resolution_notes"])

    valid_priorities = {"low", "medium", "high", "critical"}
    df = df[df["priority"].str.lower().isin(valid_priorities)]
    df["priority"] = df["priority"].str.lower()

    df = df.reset_index(drop=True)
    df["ticket_id"] = range(1, len(df) + 1)
    return df


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = INTERIM_DIR / "tickets_ingested.parquet"
    df = pd.read_parquet(raw_path) if raw_path.exists() else ingest()
    cleaned = clean(df)
    out_path = PROCESSED_DIR / "tickets_clean.parquet"
    cleaned.to_parquet(out_path, index=False)
    print(f"Cleaned {len(df)} -> {len(cleaned)} rows -> {out_path}")
