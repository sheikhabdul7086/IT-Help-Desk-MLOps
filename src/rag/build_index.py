"""Phase 4 step 1: chunk resolved tickets, embed them, and index into ChromaDB.

Each indexed document is one resolved ticket (subject+body as the query side,
resolution notes as the payload) so retrieval at inference time returns real
past fixes the LLM can condition its generation on.
"""
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build_index():
    cfg = load_config()
    df = pd.read_parquet(PROCESSED_DIR / f"tickets_{cfg['index_source_split']}.parquet")
    df = df.head(cfg["index_max_docs"]).copy()
    df["text"] = (df["ticket_subject"].fillna("") + " " + df["description"].fillna("")).str.strip()

    print(f"Embedding {len(df)} resolved tickets with {cfg['embedding_model']}...")
    embedder = SentenceTransformer(cfg["embedding_model"])
    embeddings = embedder.encode(df["text"].tolist(), show_progress_bar=True, batch_size=64)

    store_dir = Path(cfg["vector_store_dir"])
    store_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(store_dir))

    try:
        client.delete_collection(cfg["collection_name"])
    except Exception:
        pass
    collection = client.create_collection(cfg["collection_name"])

    collection.add(
        ids=[str(i) for i in df["ticket_id"]],
        embeddings=embeddings.tolist(),
        documents=df["text"].tolist(),
        metadatas=[
            {
                "category": row.category,
                "priority": row.priority,
                "resolution_notes": row.resolution_notes,
                "ticket_type": row.ticket_type,
            }
            for row in df.itertuples()
        ],
    )

    print(f"Indexed {collection.count()} documents into '{cfg['collection_name']}' at {store_dir}")


if __name__ == "__main__":
    build_index()
