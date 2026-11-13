"""Phase 1 step 3: stratified 70/15/15 train/val/test split, versioned as separate files."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .clean import PROCESSED_DIR, clean
from .ingest import INTERIM_DIR

RANDOM_STATE = 42


def stratified_split(df: pd.DataFrame):
    strat_col = df["category"] + "_" + df["priority"]

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_STATE, stratify=strat_col
    )
    temp_strat = temp_df["category"] + "_" + temp_df["priority"]
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_strat
    )
    return train_df, val_df, test_df


if __name__ == "__main__":
    clean_path = PROCESSED_DIR / "tickets_clean.parquet"
    if clean_path.exists():
        df = pd.read_parquet(clean_path)
    else:
        ingested_path = INTERIM_DIR / "tickets_ingested.parquet"
        df = clean(pd.read_parquet(ingested_path))

    train_df, val_df, test_df = stratified_split(df)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = PROCESSED_DIR / f"tickets_{name}.parquet"
        split_df.to_parquet(out_path, index=False)
        print(f"{name}: {len(split_df)} rows -> {out_path}")

    print("\nTotal:", len(df), "| train/val/test:", len(train_df), len(val_df), len(test_df))
