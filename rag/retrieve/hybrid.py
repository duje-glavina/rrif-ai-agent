"""Hybrid retrieval: semantic (vector) + keyword (FTS) fused with RRF.

Two ranked lists, combined by Reciprocal Rank Fusion. RRF doesn't need
normalised scores — it just looks at where each document ranks in each
list. A document ranking high in both gets boosted; ranking high in only
one is still useful but less so. Standard k=60.

Important: ts_rank_cd does not penalise common terms the way BM25 does.
We compensate by stripping query stopwords AND corpus-specific noise terms
("pdv", "članak", etc.) which appear in nearly every chunk and would
otherwise drown out the actually-discriminating keywords.

Usage:
    from rag.retrieve.hybrid import hybrid_search
    hits = hybrid_search("Kolika je opća stopa PDV-a?", k=5)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from rag.embedder import embed_query

load_dotenv()


CANDIDATES_PER_RANKER = 50
RRF_K = 60


# Croatian function-word stopwords. These are the words a question is
# *built around* — they don't tell you what the question is about.
_CROATIAN_STOPWORDS = {
    # interrogatives & pronouns
    "kolika", "kolik", "koliko", "koji", "koja", "koje", "kakav", "kakva",
    "što", "sto", "tko", "kada", "kad", "gdje", "kako", "zašto",
    # auxiliaries / copulas
    "je", "su", "bio", "bila", "biti", "bude", "ima", "imati",
    # prepositions / conjunctions
    "na", "se", "iz", "od", "do", "za", "pri", "pod", "nad", "kroz",
    "kod", "uz", "po", "prema", "nakon", "prije", "između", "tijekom",
    "ili", "ali", "te", "tu", "to", "ta", "taj", "ova", "ovo", "ovaj",
    "ovi", "one", "oni", "u", "i", "a", "ne", "li",
}

# Corpus-specific noise: words that appear in nearly every chunk of a
# tax-law corpus and are therefore non-discriminating. Adding more laws
# later may require revisiting this list.
_CORPUS_NOISE = {
    "pdv", "članak", "clanak", "zakon", "zakona", "zakonom",
    "ovoga", "ovog", "stavak", "stavka", "točka", "tocka",
    "porez", "porezni", "porezno", "porezna", "porezne", "poreza",
}


def _build_tsquery(query: str) -> str:
    """Tokenise the user query, drop noise, OR the survivors.

    The OR (`|`) is intentional — we want recall, then RRF lets the
    semantic ranker confirm relevance. AND would be too restrictive
    on questions where wording differs from the law's own phrasing.
    """
    cleaned = re.sub(
        r"[^\w\sčćšžđČĆŠŽĐ]", " ", query, flags=re.UNICODE,
    ).lower()
    tokens = []
    for tok in cleaned.split():
        if len(tok) < 3:
            continue
        if tok in _CROATIAN_STOPWORDS:
            continue
        if tok in _CORPUS_NOISE:
            continue
        tokens.append(tok)

    return " | ".join(tokens) if tokens else ""


@dataclass
class Hit:
    chunk_id: str
    article_number: str | None
    chunk_text: str
    semantic_rank: int | None
    keyword_rank: int | None
    rrf_score: float


def hybrid_search(query: str, k: int = 5) -> list[Hit]:
    """Run semantic + keyword search, fuse with RRF, return top k."""
    qvec = embed_query(query)
    tsquery = _build_tsquery(query)

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT id, article_number, chunk_text
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (qvec, CANDIDATES_PER_RANKER),
            )
            semantic_rows = cur.fetchall()

            keyword_rows = []
            if tsquery:
                cur.execute(
                    """
                    SELECT
                        id, article_number, chunk_text,
                        ts_rank_cd(to_tsvector('simple', chunk_text),
                                   to_tsquery('simple', %s)) AS rank
                    FROM chunks
                    WHERE to_tsvector('simple', chunk_text) @@ to_tsquery('simple', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (tsquery, tsquery, CANDIDATES_PER_RANKER),
                )
                keyword_rows = cur.fetchall()

    semantic_rank: dict = {row[0]: i + 1 for i, row in enumerate(semantic_rows)}
    keyword_rank: dict = {row[0]: i + 1 for i, row in enumerate(keyword_rows)}

    metadata: dict = {}
    for row in semantic_rows:
        metadata[row[0]] = (row[1], row[2])
    for row in keyword_rows:
        if row[0] not in metadata:
            metadata[row[0]] = (row[1], row[2])

    candidate_ids = set(semantic_rank) | set(keyword_rank)

    hits: list[Hit] = []
    for cid in candidate_ids:
        s_rank = semantic_rank.get(cid)
        k_rank = keyword_rank.get(cid)
        score = 0.0
        if s_rank is not None:
            score += 1.0 / (RRF_K + s_rank)
        if k_rank is not None:
            score += 1.0 / (RRF_K + k_rank)
        article_number, chunk_text = metadata[cid]
        hits.append(Hit(
            chunk_id=str(cid),
            article_number=article_number,
            chunk_text=chunk_text,
            semantic_rank=s_rank,
            keyword_rank=k_rank,
            rrf_score=score,
        ))

    hits.sort(key=lambda h: h.rrf_score, reverse=True)
    return hits[:k]


# CLI helper for ad-hoc tsquery debugging
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(f"Query: {q!r}")
        print(f"tsquery: {_build_tsquery(q)!r}")