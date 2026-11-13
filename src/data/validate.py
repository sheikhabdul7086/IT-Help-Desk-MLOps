"""Phase 1 data validation: schema, null, and range checks run before training.

Raises AssertionError (which fails `dvc repro`) on any violation so bad data
never silently reaches the training stages.
"""
from pathlib import Path

import pandas as pd

from .clean import PROCESSED_DIR

EXPECTED_COLUMNS = {
    "ticket_id",
    "ticket_subject",
    "description",
    "resolution_notes",
    "ticket_type",
    "category",
    "priority",
    "language",
    "tags",
}
VALID_PRIORITIES = {"low", "medium", "high", "critical"}
VALID_TYPES = {"Incident", "Request", "Problem", "Change"}
REQUIRED_NON_NULL = ["description", "resolution_notes", "category", "priority"]


def validate(df: pd.DataFrame, name: str = "dataset") -> None:
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    assert not missing_cols, f"[{name}] missing expected columns: {missing_cols}"

    assert len(df) > 0, f"[{name}] is empty"

    for col in REQUIRED_NON_NULL:
        n_null = df[col].isna().sum() + (df[col].astype(str).str.len() == 0).sum()
        assert n_null == 0, f"[{name}] column '{col}' has {n_null} null/empty values"

    bad_priority = set(df["priority"].unique()) - VALID_PRIORITIES
    assert not bad_priority, f"[{name}] unexpected priority values: {bad_priority}"

    bad_type = set(df["ticket_type"].unique()) - VALID_TYPES
    assert not bad_type, f"[{name}] unexpected ticket_type values: {bad_type}"

    assert df["ticket_id"].is_unique, f"[{name}] ticket_id is not unique"

    desc_len = df["description"].str.split().apply(len)
    assert (desc_len >= 1).all(), f"[{name}] found descriptions with zero words"
    assert (desc_len <= 2000).all(), f"[{name}] found descriptions longer than 2000 words (outlier)"

    print(f"[{name}] OK — {len(df)} rows, {df['category'].nunique()} categories, "
          f"{df['priority'].nunique()} priority levels")


if __name__ == "__main__":
    clean_path = PROCESSED_DIR / "tickets_clean.parquet"
    df = pd.read_parquet(clean_path)
    validate(df, name="tickets_clean")

    for split in ["train", "val", "test"]:
        split_df = pd.read_parquet(PROCESSED_DIR / f"tickets_{split}.parquet")
        validate(split_df, name=f"tickets_{split}")

    print("All validation checks passed.")
