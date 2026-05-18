"""Ingestion pipeline: chunks + metadata → embeddings → chunks table.

The "shape" of the data this function expects matches the chunks table schema.
Anything that produces a list of (text, article_number) pairs plus a single
SourceMetadata bundle can feed it — PDF loader, nn.hr fetcher, article
importer, transcript ingester, all the same downstream pipeline.

Classification happens at ingest time: each chunk is classified via the
Haiku classifier (batched, 20 chunks per API call) and domain/subdomain are
written directly to the DB. This removes the need to run reclassify_all.py
after every ingest.

To skip classification (e.g. for law chunks where category is known):
    ingest(chunks, meta, classify=False)
In that case domain/subdomain are left NULL and must be filled by
reclassify_all.py or a subsequent migration.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

import anthropic
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from rag.embedder import embed_passages
from rag.ingest.pdf_loader import ArticleChunk
from rag.taxonomy import SUBDOMAIN_TO_DOMAIN, VALID_SUBDOMAINS, prompt_taxonomy_block

load_dotenv()
log = logging.getLogger(__name__)

CLASSIFIER_MODEL  = "claude-haiku-4-5"
CLASSIFY_BATCH    = 20    # chunks per Haiku API call
CLASSIFY_WORKERS  = 5     # parallel API calls


# ── Source metadata ───────────────────────────────────────────────────────────

@dataclass
class SourceMetadata:
    """Everything about the source document that's the same for all its chunks."""
    category: str                # legacy flat category, e.g. 'PDV', 'plaće', 'ostalo'
    source_type: str             # 'zakon', 'članak', 'FAQ', 'priručnik', 'transkript'
    source: str                  # citation string, e.g. "Zakon o PDV-u, NN 73/13"
    law_name: str | None         # "Zakon o porezu na dodanu vrijednost"
    nn_reference: str | None     # "NN 73/13"
    valid_from: date             # when this version came into force
    valid_to: date | None        # NULL means currently valid
    status: str                  # 'vazeci' / 'nevazeci'
    citable: bool = True
    extra_metadata: dict = field(default_factory=dict)


# ── Inline classifier ─────────────────────────────────────────────────────────

_CLF_SYSTEM = f"""Ti si klasifikator tekstualnih isječaka iz hrvatskog računovodstvenog i poreznog znanja.

Za svaki isječak odredi domain i subdomain prema ovoj taksonomiji:
{prompt_taxonomy_block()}

Vrati ISKLJUČIVO JSON array bez ikakvog teksta prije ili nakon:
[
  {{"id": "<id>", "domain": "<domain>", "subdomain": "<subdomain>"}},
  ...
]"""


def _classify_batch(client: anthropic.Anthropic, batch: list[tuple[int, str]]) -> list[dict]:
    """Classify a batch of (idx, text) pairs. Returns list of {id, domain, subdomain}."""
    payload = [{"id": str(idx), "text": text[:600]} for idx, text in batch]
    try:
        r = client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=1024,
            system=_CLF_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        )
        raw = r.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        results = json.loads(raw)
        validated = []
        for res in results:
            subdomain = res.get("subdomain", "ostalo")
            domain    = res.get("domain", "ostalo")
            if subdomain in SUBDOMAIN_TO_DOMAIN:
                domain = SUBDOMAIN_TO_DOMAIN[subdomain]
            if subdomain not in VALID_SUBDOMAINS:
                subdomain, domain = "ostalo", "ostalo"
            validated.append({"id": res["id"], "domain": domain, "subdomain": subdomain})
        return validated
    except Exception as e:
        log.warning("Classification batch failed: %s", e)
        return [{"id": str(idx), "domain": "ostalo", "subdomain": "ostalo"} for idx, _ in batch]


def _classify_chunks(texts: list[str]) -> list[tuple[str, str]]:
    """Classify all chunk texts. Returns list of (domain, subdomain) in same order."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=30.0)

    batches = [
        [(i, texts[i]) for i in range(start, min(start + CLASSIFY_BATCH, len(texts)))]
        for start in range(0, len(texts), CLASSIFY_BATCH)
    ]

    results: dict[int, tuple[str, str]] = {}

    with ThreadPoolExecutor(max_workers=CLASSIFY_WORKERS) as executor:
        futures = {executor.submit(_classify_batch, client, batch): batch for batch in batches}
        for future in as_completed(futures):
            for res in future.result():
                results[int(res["id"])] = (res["domain"], res["subdomain"])

    return [results.get(i, ("ostalo", "ostalo")) for i in range(len(texts))]


# ── SQL ───────────────────────────────────────────────────────────────────────

_INSERT_SQL = """
INSERT INTO chunks (
    chunk_text, embedding,
    category_legacy, source_type, source,
    law_name, article_number, nn_reference,
    valid_from, valid_to, status, citable,
    chunk_index, total_chunks, extra_metadata,
    domain, subdomain
)
VALUES (
    %s, %s,
    %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s,
    %s, %s
)
"""


# ── Public API ────────────────────────────────────────────────────────────────

def ingest(
    chunks: Iterable[ArticleChunk],
    meta: SourceMetadata,
    classify: bool = True,
) -> int:
    """Embed all chunks, optionally classify them, and insert into the chunks table.

    Args:
        chunks:   Iterable of chunk objects with .text, .article_number, .chunk_index
        meta:     Source-level metadata shared across all chunks
        classify: If True (default), classify each chunk via Haiku and write
                  domain/subdomain directly. If False, domain/subdomain are NULL.

    Returns:
        Number of rows inserted.
    """
    chunks = list(chunks)
    if not chunks:
        return 0

    texts = [c.text for c in chunks]

    # Step 1: embed
    print(f"[ingest] Embedding {len(chunks)} chunks...")
    embeddings = embed_passages(texts)

    # Step 2: classify (optional)
    if classify:
        n_batches = (len(chunks) + CLASSIFY_BATCH - 1) // CLASSIFY_BATCH
        print(f"[ingest] Classifying {len(chunks)} chunks ({n_batches} batches)...")
        classifications = _classify_chunks(texts)
    else:
        classifications = [("ostalo", "ostalo")] * len(chunks)

    # Step 3: insert
    print(f"[ingest] Writing to database...")
    database_url = os.environ["DATABASE_URL"]

    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for c, vec, (domain, subdomain) in zip(chunks, embeddings, classifications):
                cur.execute(_INSERT_SQL, (
                    c.text, vec,
                    meta.category, meta.source_type, meta.source,
                    meta.law_name, c.article_number, meta.nn_reference,
                    meta.valid_from, meta.valid_to, meta.status, meta.citable,
                    c.chunk_index, len(chunks),
                    psycopg.types.json.Jsonb(meta.extra_metadata),
                    domain, subdomain,
                ))
        conn.commit()

    print(f"[ingest] Inserted {len(chunks)} rows.")
    return len(chunks)
