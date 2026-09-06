"""Croatian stemming / lemmatisation for the lexical retrieval branch.

Route B of the FTS experiment: normalise Croatian morphology in Python at
ingest and query time, store the result in its own column, and index that
with the 'simple' config. The database never needs a dictionary, so this
works identically on self-hosted and managed Postgres.

Two backends:

  classla — proper lemmatisation for Croatian (CLASSLA-Stanza). Needs a model
            download (~250MB) and torch, which this project already has.
            Slow on CPU, fast on the GPU box.
  crude   — a conservative suffix stripper written for this experiment.
            No dependencies, instant, and deliberately timid.

The crude backend is a FLOOR, not a result. If even it moves top-1 retrieval,
proper lemmatisation will move it further. If it moves nothing, that is weak
evidence rather than a verdict — rerun with classla before concluding
Croatian morphology doesn't matter here.

Selection: automatic (classla if importable), overridable with
STEM_BACKEND=crude|classla. Force crude on the laptop for quick iteration,
classla on the Omen for the run that counts. Whatever you index with, you
must query with — mismatched backends collapse lexical recall silently.

Usage:
    from rag.stem_hr import stem_text, stem_query, stem_many, backend_name
"""
from __future__ import annotations

import functools
import os
import re
from typing import Iterable

# ── Backend selection ─────────────────────────────────────────────────────────

_REQUESTED = os.getenv("STEM_BACKEND", "auto").lower()

_CLASSLA_AVAILABLE = False
if _REQUESTED in ("auto", "classla"):
    try:  # pragma: no cover - depends on local install
        import classla  # noqa: F401
        _CLASSLA_AVAILABLE = True
    except ImportError:
        if _REQUESTED == "classla":
            raise RuntimeError(
                "STEM_BACKEND=classla but classla is not installed. "
                "pip install classla, or unset STEM_BACKEND to use the crude backend."
            )


def backend_name() -> str:
    return "classla" if _CLASSLA_AVAILABLE else "crude"


@functools.lru_cache(maxsize=1)
def _classla_pipeline():
    """Lazy-load the Croatian lemmatisation pipeline.

    GPU use is detected rather than assumed — this project runs on a laptop
    without one and a desktop with one, and hardcoding use_gpu=True makes the
    laptop path fail or fall back with a warning buried in the output.
    """
    import classla

    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except ImportError:
        use_gpu = False

    print(f"[stem_hr] loading classla (use_gpu={use_gpu})...")
    classla.download("hr", processors="tokenize,pos,lemma")
    return classla.Pipeline("hr", processors="tokenize,pos,lemma", use_gpu=use_gpu)


# ── Crude backend ─────────────────────────────────────────────────────────────

# Ordered longest-first: the first match wins, so "ovima" is tried before "ima".
# Limited to common nominal and adjectival inflection. Verb morphology is left
# alone — stripping it badly costs more than leaving it, and legal text is
# noun-heavy anyway.
_SUFFIXES = (
    "ovima", "evima", "ijima", "ijama",
    "ima", "ama", "oga", "ome", "omu", "emu", "ega",
    "og", "om", "im", "ih", "em", "oj", "ju", "ov", "ev",
    "a", "e", "i", "o", "u",
)

_MIN_STEM = 4
_TOKEN_RE = re.compile(r"\w+|\W+", re.UNICODE)


def _crude_stem_token(token: str) -> str:
    low = token.lower()
    if len(low) <= _MIN_STEM or any(ch.isdigit() for ch in low):
        return low
    for suf in _SUFFIXES:
        if low.endswith(suf) and len(low) - len(suf) >= _MIN_STEM:
            return low[: -len(suf)]
    return low


def _crude_stem_text(text: str) -> str:
    """Stem word tokens, pass everything else through unchanged.

    Non-word runs are preserved so article references like '38/3' and
    'čl. 85a' survive intact — losing those would hurt more than the
    morphology gains.
    """
    return "".join(
        _crude_stem_token(p) if p.isalnum() else p
        for p in _TOKEN_RE.findall(text)
    )


def _lemmas(doc) -> str:
    return " ".join(
        w.lemma or w.text for s in doc.sentences for w in s.words
    )


# ── Public API ────────────────────────────────────────────────────────────────

def stem_text(text: str) -> str:
    """Normalise a single passage."""
    if not text:
        return ""
    if _CLASSLA_AVAILABLE:
        return _lemmas(_classla_pipeline()(text))
    return _crude_stem_text(text)


def stem_query(text: str) -> str:
    """Normalise a user question.

    MUST use the same backend as the indexed column, or the two vocabularies
    won't line up and lexical recall collapses without raising anything.
    """
    return stem_text(text)


def stem_text_crude(text: str) -> str:
    """Always the crude backend, whatever STEM_BACKEND says.

    The eval grader uses this to normalise keywords and chunk text before
    comparing them. It must not depend on whether classla happens to be
    installed on the machine running the eval — a metric that changes with
    the local environment is not a metric.
    """
    return _crude_stem_text(text or "")


def stem_many(texts: Iterable[str]) -> list[str]:
    """Batch variant.

    Stanza-family pipelines are dramatically faster processing a batch of
    documents in one call than looping — the difference is roughly an order of
    magnitude, which at ~127k chunks is the difference between an afternoon
    and a coffee break. bulk_process isn't in every classla release, so fall
    back to a loop if it's missing.
    """
    texts = list(texts)
    if not texts:
        return []
    if not _CLASSLA_AVAILABLE:
        return [_crude_stem_text(t) for t in texts]

    nlp = _classla_pipeline()

    try:
        from classla import Document
        docs = [Document([], text=t) for t in texts]
        return [_lemmas(d) for d in nlp.bulk_process(docs)]
    except (ImportError, AttributeError):
        return [_lemmas(nlp(t)) for t in texts]


if __name__ == "__main__":
    samples = [
        "Kolika je opća stopa PDV-a?",
        "poreznim obveznicima priznaje se pretporez",
        "Porezna osnovica kod uvoza dobara iz članka 35/3",
    ]
    print(f"backend: {backend_name()}\n")
    for original, stemmed in zip(samples, stem_many(samples)):
        print(f"  {original!r}\n→ {stemmed!r}\n")
