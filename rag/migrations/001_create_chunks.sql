-- ============================================================================
-- RRiF AI Agent — Knowledge base schema
-- Decision record: docs/decisions/001-rag-architecture.md
-- Run against: rrif_rag database (must have pgvector extension enabled)
-- ============================================================================

-- Verify pgvector is available before creating the table.
-- This is a no-op if already enabled.
CREATE EXTENSION IF NOT EXISTS vector;

-- ----------------------------------------------------------------------------
-- chunks: one row per embedded chunk of a document
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Content
    chunk_text      TEXT        NOT NULL,
    embedding       vector(1024)        ,    -- multilingual-e5-large = 1024 dims

    -- Classification (filtered on every retrieval)
    category        TEXT        NOT NULL,    -- 'PDV', 'dohodak', 'doprinosi', 'GDPR', 'racunovodstvo', 'ostalo'
    source_type     TEXT        NOT NULL,    -- 'zakon', 'clanak', 'FAQ', 'prirucnik', 'transkript', ...

    -- Source identification (for citations and de-duplication)
    source          TEXT        NOT NULL,    -- Full citation string
    law_name        TEXT,                    -- 'Zakon o PDV-u'
    article_number  TEXT,                    -- '85', '85a', '38/3' — TEXT to handle suffixes
    nn_reference    TEXT,                    -- 'NN 73/13', 'NN 114/23'

    -- Temporal versioning (the 5-year history requirement)
    valid_from      DATE        NOT NULL,
    valid_to        DATE,                    -- NULL = currently in force
    status          TEXT        NOT NULL,    -- 'vazeci' / 'nevazeci' — derived from valid_to but indexed

    -- Citability (F-12)
    citable         BOOLEAN     NOT NULL DEFAULT TRUE,

    -- Chunk position within source document
    chunk_index     INTEGER,
    total_chunks    INTEGER,

    -- Flexibility
    extra_metadata  JSONB       NOT NULL DEFAULT '{}'::jsonb,

    -- Audit
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- Indexes
-- ----------------------------------------------------------------------------

-- Metadata filters used by the classifier-driven retrieval
CREATE INDEX IF NOT EXISTS idx_chunks_category     ON chunks (category);
CREATE INDEX IF NOT EXISTS idx_chunks_source_type  ON chunks (source_type);
CREATE INDEX IF NOT EXISTS idx_chunks_status       ON chunks (status);
CREATE INDEX IF NOT EXISTS idx_chunks_citable      ON chunks (citable);

-- Temporal filters
CREATE INDEX IF NOT EXISTS idx_chunks_valid_from   ON chunks (valid_from);
CREATE INDEX IF NOT EXISTS idx_chunks_valid_to     ON chunks (valid_to);

-- Composite index for the most common filter combination:
--   "current PDV laws" → category + status + valid_to IS NULL
CREATE INDEX IF NOT EXISTS idx_chunks_cat_status   ON chunks (category, status);

-- Source lookup (used when rebuilding/replacing a document's chunks)
CREATE INDEX IF NOT EXISTS idx_chunks_law_article  ON chunks (law_name, article_number);

-- HNSW index on the embedding for fast approximate nearest-neighbour.
-- Cosine distance because e5 embeddings are normalised.
-- This index is only effective once there's enough data (~hundreds of rows minimum);
-- creating it now is fine but it'll really earn its keep after ingestion.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- BM25 / full-text search column (Croatian).
-- Postgres doesn't have a Croatian text-search config out of the box;
-- 'simple' avoids language-specific stemming (which we don't want for legal text anyway).
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_text_fts
    ON chunks
    USING gin (to_tsvector('simple', chunk_text));

-- ----------------------------------------------------------------------------
-- Trigger: keep updated_at in sync
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_chunks_updated_at ON chunks;
CREATE TRIGGER trg_chunks_updated_at
    BEFORE UPDATE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();

-- ----------------------------------------------------------------------------
-- Sanity check
-- ----------------------------------------------------------------------------
-- Run after creation:
--   SELECT count(*) FROM chunks;                        -- should be 0
--   \d chunks                                            -- inspect structure
--   SELECT indexname FROM pg_indexes WHERE tablename = 'chunks';
