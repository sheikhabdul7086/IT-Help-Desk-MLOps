"""Phase 1 step 1: ingest the raw Kaggle ticket dataset into a single working frame."""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"
PRIMARY_FILE = RAW_DIR / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"

SCHEMA_RENAME = {
    "subject": "ticket_subject",
    "body": "description",
    "answer": "resolution_notes",
    "queue": "category",
    "priority": "priority",
    "type": "ticket_type",
    "language": "language",
}


def ingest() -> pd.DataFrame:
    df = pd.read_csv(PRIMARY_FILE)
    df = df.rename(columns=SCHEMA_RENAME)
    tag_cols = [c for c in df.columns if c.startswith("tag_")]
    df["tags"] = df[tag_cols].apply(
        lambda row: [t for t in row if isinstance(t, str) and t.strip()], axis=1
    )
    df = df.drop(columns=tag_cols + ["version"], errors="ignore")
    df["ticket_id"] = range(1, len(df) + 1)
    return df


if __name__ == "__main__":
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df = ingest()
    out_path = INTERIM_DIR / "tickets_ingested.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Ingested {len(df)} rows -> {out_path}")
    print(df.dtypes)
