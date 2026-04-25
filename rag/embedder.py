"""Embedding wrapper for multilingual-e5-large.

Loads the model once on first use, then provides fast embed() calls.
e5 expects a prefix on the input ('query: ' for questions, 'passage: ' for documents)
to maximise retrieval quality — this is documented in the model card and matters.
"""
from functools import lru_cache
from typing import Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-large"
EMBEDDING_DIM = 1024


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Lazy-load the model. First call downloads ~2GB and warms up the GPU.
    Subsequent calls in the same process reuse the cached instance.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embedder] Loading {MODEL_NAME} on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"[embedder] Loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")
    return model


def embed_passages(texts: Iterable[str]) -> np.ndarray:
    """Embed document chunks for storage in the knowledge base.
    Returns a (N, 1024) numpy array of L2-normalised vectors.
    """
    model = _get_model()
    prefixed = [f"passage: {t}" for t in texts]
    return model.encode(
        prefixed,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


def embed_query(text: str) -> np.ndarray:
    """Embed a user question for retrieval.
    Returns a single (1024,) numpy array, L2-normalised.
    """
    model = _get_model()
    return model.encode(
        f"query: {text}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )