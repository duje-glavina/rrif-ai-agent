"""Ingestion pipeline: chunks + metadata → embeddings → chunks table.

The "shape" of the data this function expects matches the chunks table from
migration 001. Anything that produces a list of (text, article_number) pairs
plus a single SourceMetadata bundle can feed it — PDF loader, nn.hr fetcher,
priručnik importer, transcript ingester, all the same downstream pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from rag.embedder import embed_passages
from rag.ingest.pdf_loader import ArticleChunk

load_dotenv()


@dataclass
class SourceMetadata:
    """Everything about the source document that's the same for all its chunks."""
    category: str                # 'PDV', 'dohodak', 'doprinosi', 'GDPR', 'racunovodstvo', 'ostalo'
    source_type: str             # 'zakon', 'clanak', 'FAQ', 'prirucnik', 'transkript'
    source: str                  # citation string, e.g. "Zakon o PDV-u, NN 73/13"
    law_name: str | None         # "Zakon o porezu na dodanu vrijednost"
    nn_reference: str | None     # "NN 73/13"
    valid_from: date             # when this version came into force
    valid_to: date | None        # NULL means currently valid
    status: str                  # 'vazeci' / 'nevazeci'
    citable: bool = True         # F-12 — appears in user-facing citations
    extra_metadata: dict = field(default_factory=dict)


_INSERT_SQL = """
INSERT INTO chunks (
    chunk_text, embedding,
    category, source_type, source,
    law_name, article_number, nn_reference,
    valid_from, valid_to, status, citable,
    chunk_index, total_chunks, extra_metadata
)
VALUES (
    %s, %s,
    %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s
)
"""


def ingest(chunks: Iterable[ArticleChunk], meta: SourceMetadata) -> int:
    """Embed all chunks and insert them into the chunks table.

    Returns the number of rows inserted. Wraps everything in a single
    transaction so a failure mid-way leaves the table in a clean state.
    """
    chunks = list(chunks)
    if not chunks:
        return 0

    print(f"[ingest] Embedding {len(chunks)} chunks...")
    texts = [c.text for c in chunks]
    embeddings = embed_passages(texts)

    database_url = os.environ["DATABASE_URL"]
    print(f"[ingest] Writing to database...")

    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for c, vec in zip(chunks, embeddings):
                cur.execute(_INSERT_SQL, (
                    c.text, vec,
                    meta.category, meta.source_type, meta.source,
                    meta.law_name, c.article_number, meta.nn_reference,
                    meta.valid_from, meta.valid_to, meta.status, meta.citable,
                    c.chunk_index, len(chunks),
                    psycopg.types.json.Jsonb(meta.extra_metadata),
                ))
        conn.commit()

    print(f"[ingest] Inserted {len(chunks)} rows.")
    return len(chunks)
