"""Cross-encoder reranker using BGE-reranker-v2-m3.

Where the embedder produces vectors for separate sentences and we then
compare via cosine similarity, the reranker reads a (query, passage) pair
*together* and outputs a single relevance score. That joint encoding is
what makes it more accurate than bi-encoder similarity — and what makes
it expensive enough to only run on a small number of candidates.

Pipeline:
    semantic search → top 20 candidates → reranker → top k

Usage:
    from rag.retrieve.rerank import rerank
    candidates = [(doc_id, text), (doc_id, text), ...]
    ranked = rerank(query, candidates, k=5)
    for doc_id, text, score in ranked:
        ...
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import torch
from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    """Lazy-load the reranker. First call downloads ~2.3GB."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[reranker] Loading {MODEL_NAME} on {device}...")
    # Sigmoid activation maps the raw cross-encoder logit into [0, 1] so
    # scores are comparable across queries and easier to set thresholds on.
    model = CrossEncoder(
        MODEL_NAME,
        device=device,
        activation_fn=torch.nn.Sigmoid(),
    )
    print(f"[reranker] Loaded.")
    return model


def rerank(
    query: str,
    candidates: Sequence[tuple],
    k: int = 5,
) -> list[tuple]:
    """Score each (id, text) candidate against the query and return top k.

    Input:  list of (anything_id, passage_text) — the id is opaque to the
            reranker and just passed through so callers can correlate scores
            with their own metadata.
    Output: list of (id, text, score) sorted by score descending.

    The reranker computes scores in a single batched forward pass, so
    passing 20 candidates is barely slower than passing 5 — the GPU
    utilisation rather than the candidate count dominates.
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [[query, text] for _, text in candidates]
    scores = model.predict(pairs, show_progress_bar=False)

    scored = [
        (cid, text, float(score))
        for (cid, text), score in zip(candidates, scores)
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:k]
