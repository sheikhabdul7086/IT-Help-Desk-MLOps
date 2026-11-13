"""Run Phase 1 end to end: ingest -> clean -> EDA -> split."""
from .ingest import INTERIM_DIR, ingest
from .clean import PROCESSED_DIR, clean
from .eda import run_eda
from .split import stratified_split
from .validate import validate


def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("1/4 Ingesting raw data...")
    raw_df = ingest()
    raw_df.to_parquet(INTERIM_DIR / "tickets_ingested.parquet", index=False)
    print(f"    {len(raw_df)} rows ingested")

    print("2/5 Cleaning & sanitizing...")
    clean_df = clean(raw_df)
    clean_df.to_parquet(PROCESSED_DIR / "tickets_clean.parquet", index=False)
    print(f"    {len(raw_df)} -> {len(clean_df)} rows after cleaning/dedup")

    print("3/5 Validating (schema/nulls/ranges)...")
    validate(clean_df, name="tickets_clean")

    print("4/5 Running EDA...")
    summary = run_eda(clean_df)
    for k, v in summary.items():
        print(f"    {k}: {v}")

    print("5/5 Stratified 70/15/15 split...")
    train_df, val_df, test_df = stratified_split(clean_df)
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df.to_parquet(PROCESSED_DIR / f"tickets_{name}.parquet", index=False)
        validate(split_df, name=f"tickets_{name}")
        print(f"    {name}: {len(split_df)} rows")

    print("\nPhase 1 complete. Processed files in data/processed/")


if __name__ == "__main__":
    main()
