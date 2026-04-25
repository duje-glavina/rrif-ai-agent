# 001 — RAG Architecture

**Status:** Accepted
**Date:** 2026-04-25
**Decided by:** S (project lead), implemented by Duje
**Source:** S's design email (16.04.2026) + TechSpecs F-12

---

## Context

The RRiF AI Agent must answer questions about Croatian accounting, tax, and finance regulations using a curated knowledge base. The tender (RRiF_AI_agent.pdf) requires:

- Citation of sources for every answer
- Distinction between currently valid and historical regulations, going back at least 5 years
- A configurable list of trusted external sources
- Refusal to answer when the knowledge base does not contain a confident answer, with redirection to the RRiF advisory line

This document captures the agreed architecture for retrieval and answer generation.

---

## Decision

### Pipeline overview

```
User question
   │
   ▼
[1] Classifier (LLM call)
        outputs: { category, time_period, source_type }
   │
   ▼
[2] Hybrid retrieval on filtered subset
        semantic (vector) + keyword (BM25)
        returns top 20 candidates
   │
   ▼
[3] Cross-encoder reranker
        reorders by true relevance
        returns top 5
   │
   ▼
[4] Answer generation (LLM call)
        produces answer + citations + temporal disclosure
```

### Component choices

| Stage | Choice | Rationale |
|---|---|---|
| Embedding | `multilingual-e5-large` | Open weights, EU-hostable on the office 5090, no per-token cost, strong Croatian coverage. One model for all content types — never mix embedders, similarity scores would not be comparable. |
| Vector store | PostgreSQL + pgvector | ACID transactions matter for versioned regulation updates (F-11). Single source of truth alongside metadata. |
| Keyword search | BM25 (rank-bm25 or Postgres FTS) | Complements semantic search. Catches exact terms, statute numbers, article references. |
| Classifier | Claude (Haiku, for cost) | Pre-retrieval filter on category, time period, source type. |
| Reranker | BGE-reranker-v2-m3 (local on 5090) or Cohere Rerank | Cross-encoder reranking is consistently the difference between good and excellent RAG. |
| Generator | Claude | Already validated in benchmark (8/8 including refusal traps). |

### Schema requirements

Each chunk in the knowledge base carries explicit, queryable metadata. Metadata is filtered on every retrieval, so it lives in indexed columns rather than JSONB.

| Column | Type | Purpose |
|---|---|---|
| `id` | UUID PK | Chunk identifier |
| `chunk_text` | TEXT | The actual content |
| `embedding` | vector(1024) | e5-large embedding |
| `category` | TEXT | PDV, dohodak, doprinosi, GDPR, računovodstvo, … |
| `source_type` | TEXT | zakon, članak, FAQ, priručnik, transkript, … |
| `source` | TEXT | Citation string (e.g., "Zakon o PDV-u, NN 73/13, čl. 38") |
| `law_name` | TEXT | "Zakon o PDV-u" |
| `article_number` | TEXT | "85", "38", "66" |
| `nn_reference` | TEXT | "NN 73/13", "NN 114/23" |
| `valid_from` | DATE | When this version became effective |
| `valid_to` | DATE NULL | When superseded; NULL = currently valid |
| `status` | TEXT | "važeći" / "nevažeći" — derived from valid_to, indexed for filtering |
| `citable` | BOOLEAN | TRUE = may appear in user-facing citations; FALSE = silent context only |
| `extra_metadata` | JSONB | Flexibility for fields that emerge later |
| `created_at`, `updated_at` | TIMESTAMP | Audit trail |

Indexes on `category`, `source_type`, `status`, `valid_from`, `valid_to`, `citable`, plus an HNSW index on `embedding`.

### Temporal versioning rules

The 5-year-history requirement (and the tender's emphasis on distinguishing valid from invalid regulations) drives these rules:

1. **Never delete superseded versions.** When a law changes, the old chunks have their `valid_to` set to the day before the new version takes effect. The new version is inserted with its own `valid_from`.
2. **Time-aware retrieval.** When the classifier identifies a temporal reference ("u 2022.", "od 2024."), the retrieval filter selects versions where `valid_from ≤ target_date ≤ COALESCE(valid_to, '9999-12-31')`.
3. **Default to currently valid.** Questions without a temporal reference filter on `status = 'važeći'`.
4. **Always disclose the temporal basis.** Every answer states the date the cited regulation was valid on. Example footer: *"Odgovor se temelji na propisu važećem na datum [X]."*

### Citable vs. non-citable sources

F-12 in the tech specs distinguishes two tiers of knowledge-base content:

**Citable (`citable = TRUE`)**
- Laws and Narodne novine entries
- RRiF priručnici, knjige, articles published in the magazine
- Curated content from whitelisted government sources (porezna-uprava.gov.hr, mfin.gov.hr, hanfa.hr, hnb.hr, carina.gov.hr, nn.hr)

These appear in user-facing citations.

**Non-citable / silent context (`citable = FALSE`)**
- Internal written advisor responses
- Call-centre transcripts, webinar transcripts (Phase 2)

These improve retrieval and answer quality but are **never** named as sources to the user. The system prompt must instruct the generator to only cite sources where `citable = TRUE`. The retrieval layer may use both tiers as context.

This is a trust and liability decision: RRiF's users (accountants, tax advisors) need to verify sources independently. A citation to a NN article is verifiable; a citation to "internal advisor call from March 2023" is neither verifiable nor defensible.

### Pre-retrieval classification details

The classifier runs first and outputs structured JSON:

```json
{
  "category": "PDV" | "dohodak" | "doprinosi" | "GDPR" | "računovodstvo" | "ostalo",
  "time_period": { "type": "current" | "specific_date" | "range" | "historical",
                   "from": "2024-01-01" | null,
                   "to":   null | "2024-12-31" },
  "source_type_preference": ["zakon", "članak", "FAQ"] | null
}
```

S's worked example: question *"kolika je stopa PDV-a na hranu od 2024?"* should classify as `category=PDV, time_period.from=2024-01-01, status=važeći` and the retrieval filter is built directly from that.

### What the answer must contain

Every generated answer must include:
1. The substantive response in Croatian.
2. Source citations limited to chunks where `citable = TRUE`.
3. An explicit temporal-basis statement when regulations are involved.
4. A redirect to the RRiF advisory line if confidence is low or the question falls outside the knowledge base.

---

## Consequences

**Positive**
- Aligned with S's stated direction; no rework expected on architecture.
- Schema supports all current TechSpecs requirements (F-10, F-11, F-12) and the temporal-comparison ask in the tender.
- Each component is independently testable: classifier, retrieval, rerank, generation.

**Negative / open**
- Classifier adds ~1 LLM call per question. Latency budget needs measuring; Haiku should keep this cheap and fast.
- BM25 implementation choice (rank-bm25 in Python vs. Postgres FTS) is not yet decided. To be resolved in 002.
- Rerank model choice (BGE local vs. Cohere API) not finalised. Local BGE on the 5090 preserves EU residency; Cohere is simpler. To be resolved in 003.
- Initial corpus source is nn.hr (permissive robots.txt, structured ELI metadata). Ingestion of RRiF's own priručnici and articles depends on access to the source files.

---

## References

- S's email, 16.04.2026 — the source of items 1–5 in the pipeline.
- TechSpecs.pdf — F-10 (audio in Phase 2), F-11 (corpus update SLA), F-12 (citable vs. non-citable).
- RRiF_AI_agent.pdf (tender) — sections on temporal comparison, source citation, refusal behaviour.
