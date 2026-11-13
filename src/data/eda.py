"""Phase 1 step 4: exploratory data analysis — class balance, text length, language mix."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .clean import PROCESSED_DIR

FIG_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


def run_eda(df: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df["description_len"] = df["description"].str.split().apply(len)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    df["category"].value_counts().plot(kind="barh", ax=axes[0, 0], title="Tickets per category")
    df["priority"].value_counts().plot(kind="bar", ax=axes[0, 1], title="Tickets per priority")
    df["description_len"].plot(kind="hist", bins=40, ax=axes[1, 0], title="Description length (words)")
    df["language"].value_counts().plot(kind="bar", ax=axes[1, 1], title="Language mix")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_overview.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_rows": len(df),
        "n_categories": df["category"].nunique(),
        "n_priorities": df["priority"].nunique(),
        "avg_description_words": round(df["description_len"].mean(), 1),
        "duplicate_descriptions": int(df["description"].duplicated().sum()),
        "null_counts": df.isnull().sum().to_dict(),
    }
    return summary


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_DIR / "tickets_clean.parquet")
    summary = run_eda(df)
    print("EDA summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFigure saved -> {FIG_DIR / 'eda_overview.png'}")
