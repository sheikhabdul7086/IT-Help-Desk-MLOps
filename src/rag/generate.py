"""Phase 4 step 2: retrieve similar resolved tickets and generate a resolution playbook.

Backend is configurable in configs/rag.yaml:
  - "xai":        calls x.ai's hosted Grok API, rotating across multiple keys in
                  XAI_API_KEYS (project/.env) so a single key's rate limit doesn't
                  fail the request — the default when keys are configured.
  - "groq":       calls Groq's hosted Llama-3/Mistral API (needs GROQ_API_KEY env var)
  - "local_hf":   runs a small local instruction-tuned seq2seq model (flan-t5-base),
                  fully offline after the first download. In practice these small
                  models tend to just echo the retrieved context on this task rather
                  than synthesize new steps — kept as an offline-capable option.
  - "extractive": no model at all, just returns the best-matching retrieved resolution,
                  adapted for the new ticket's category/priority (deterministic fallback).
"""
import os
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from .config import load_config
from .key_rotation import _load_dotenv

_load_dotenv()  # populate os.environ from project/.env before any backend checks below

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _get_embedder(model_name: str):
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def _get_collection(store_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=store_dir)
    return client.get_collection(collection_name)


@lru_cache(maxsize=1)
def _get_local_llm(model_name: str):
    # transformers 5.x removed the "text2text-generation" pipeline task, so we
    # drive the seq2seq model directly instead of going through `pipeline()`.
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return {"tokenizer": tokenizer, "model": model}


def retrieve(ticket_text: str, cfg: dict, top_k: int | None = None):
    embedder = _get_embedder(cfg["embedding_model"])
    collection = _get_collection(cfg["vector_store_dir"], cfg["collection_name"])
    query_emb = embedder.encode([ticket_text]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=top_k or cfg["top_k"])

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"ticket_text": doc, "distance": dist, **meta})
    return hits


def _format_context(hits: list[dict]) -> str:
    """Full multi-doc context, used for the extractive fallback."""
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] Ticket: {h['ticket_text'][:200]}\n    Resolution: {h['resolution_notes'][:300]}")
    return "\n".join(lines) if lines else "(no similar past tickets found)"


def _format_best_resolution(hits: list[dict]) -> str:
    """Single best match, short — small instruction models follow this far better
    than a long multi-document context block."""
    if not hits:
        return "(no similar past ticket found)"
    return hits[0]["resolution_notes"][:300]


def _generate_extractive(hits: list[dict]) -> str:
    if not hits:
        return "No similar past resolutions found — route to a human agent for manual triage."
    best = hits[0]
    return (
        "1) Likely root cause: similar to a previously resolved ticket in the same category.\n"
        "2) Suggested steps (adapted from closest past resolution):\n"
        f"   {best['resolution_notes']}\n"
        "3) Escalation note: escalate if this does not resolve the issue within the SLA window."
    )


def _generate_local_hf(prompt: str, cfg: dict) -> str:
    import torch

    llm = _get_local_llm(cfg["local_hf_model"])
    tokenizer, model = llm["tokenizer"], llm["model"]
    inputs = tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg["max_new_tokens"],
            min_new_tokens=20,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def _generate_groq(prompt: str, cfg: dict) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=cfg["groq_model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cfg["max_new_tokens"],
    )
    return resp.choices[0].message.content.strip()


def _generate_xai(prompt: str, cfg: dict) -> str:
    from openai import OpenAI

    from .key_rotation import get_rotator

    model = os.environ.get("XAI_MODEL", cfg.get("xai_model", "grok-3-mini"))

    def _call(api_key: str) -> str:
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise, practical IT helpdesk resolution assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=cfg["max_new_tokens"],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    return get_rotator().call_with_rotation(_call)


def suggest_resolution(
    ticket_text: str, category: str, priority: str, cfg: dict | None = None
) -> dict:
    cfg = cfg or load_config()
    hits = retrieve(ticket_text, cfg)
    backend = cfg["llm_backend"]
    context = _format_best_resolution(hits) if backend == "local_hf" else _format_context(hits)
    prompt = cfg["prompt_template"].format(
        category=category, priority=priority, ticket_text=ticket_text[:400], retrieved_context=context
    )

    try:
        if backend == "xai" and os.environ.get("XAI_API_KEYS"):
            suggestion = _generate_xai(prompt, cfg)
        elif backend == "groq" and os.environ.get("GROQ_API_KEY"):
            suggestion = _generate_groq(prompt, cfg)
        elif backend == "local_hf":
            suggestion = _generate_local_hf(prompt, cfg)
        else:
            suggestion = _generate_extractive(hits)
            backend = "extractive"
    except Exception as e:  # noqa: BLE001 - never let a provider outage (billing, rate limit, network) 500 the API
        print(f"WARNING: '{backend}' generation failed ({e}); falling back to extractive")
        suggestion = _generate_extractive(hits)
        backend = "extractive (fallback)"

    return {"suggestion": suggestion, "retrieved": hits, "backend": backend}


if __name__ == "__main__":
    cfg = load_config()
    example = "My VPN client keeps disconnecting every few minutes when I work from home."
    result = suggest_resolution(example, category="IT Support", priority="high", cfg=cfg)
    print("Backend:", result["backend"])
    print("\nRetrieved:")
    for h in result["retrieved"]:
        print(" -", h["ticket_text"][:80], "...")
    print("\nSuggestion:\n", result["suggestion"])
