-- ============================================================================
-- Migration 002: add domain + subdomain columns
-- Replaces the flat single-axis `category` column with a two-level taxonomy.
--
-- domain   → top-level: porezi | računovodstvo | pravo | ostalo
-- subdomain → leaf:     PDV | dohodak | dobit | paušal | porezi |
--                       knjiženje | plaće | revizija | fin_izv | računovodstvo |
--                       radno_pravo | trg_pravo | inozemstvo | EU_propisi | pravo |
--                       proračun | NPO | tržište | upravljanje | stručne | ostalo
--
-- The old `category` column is kept for now as a fallback and renamed to
-- category_legacy. Drop it after verifying reclassification is complete.
-- ============================================================================

BEGIN;

ALTER TABLE chunks RENAME COLUMN category TO category_legacy;

ALTER TABLE chunks
    ADD COLUMN domain     TEXT,
    ADD COLUMN subdomain  TEXT;

-- Indexes for the new columns
CREATE INDEX IF NOT EXISTS idx_chunks_domain
    ON chunks (domain);

CREATE INDEX IF NOT EXISTS idx_chunks_subdomain
    ON chunks (subdomain);

CREATE INDEX IF NOT EXISTS idx_chunks_domain_subdomain
    ON chunks (domain, subdomain);

-- Composite with status (the most common query pattern)
CREATE INDEX IF NOT EXISTS idx_chunks_domain_subdomain_status
    ON chunks (domain, subdomain, status);

COMMIT;

-- ============================================================================
-- After running reclassify_all.py, verify with:
--
--   SELECT domain, subdomain, count(*)
--   FROM chunks
--   GROUP BY domain, subdomain
--   ORDER BY domain, subdomain;
--
--   SELECT count(*) FROM chunks WHERE domain IS NULL;  -- should be 0
--
-- Then drop the legacy column when happy:
--   ALTER TABLE chunks DROP COLUMN category_legacy;
-- ============================================================================
