r"""Ingest RRiF magazine articles from the archive into the chunks table.

Walks one or more archive folders (e.g. RRIF2404, PIP2401, RRIF2401...),
loads each article with article_loader.py, and ingests into the RAG pipeline.

Usage (PowerShell from project root):
    # Single folder
    python scripts\ingest_rrif_articles.py "C:\path\to\Arhiva\RRIF2404"

    # Multiple folders
    python scripts\ingest_rrif_articles.py "C:\path\to\Arhiva\RRIF2404" "C:\path\to\Arhiva\RRIF2403"

    # Entire archive (all subfolders)
    python scripts\ingest_rrif_articles.py "C:\path\to\Arhiva"

    # Dry run — show what would be ingested without touching the DB
    python scripts\ingest_rrif_articles.py "C:\path\to\Arhiva\RRIF2404" --dry-run

Options:
    --dry-run       Print chunk counts per file without writing to DB
    --skip-existing Skip folders already ingested (checks DB for existing source citations)
"""
import argparse
import sys
from datetime import date
from pathlib import Path

# Make 'rag' importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.ingest.article_loader import load_article, PUB_TYPE
from rag.ingest.pipeline import SourceMetadata, ingest
from rag.classifier import STATUS_VALID


# ── VALIDITY RULES ────────────────────────────────────────────────────────────
#
# Revised 7 Sep 2026. Previously: valid_to = 31 December of the publication
# year, and anything published before HISTORICAL_CUTOFF = 2019 was stamped
# `nevazeci` on ingest. Both were wrong, and both were measured to be wrong.
#
# `status` is a legal property. A statute article is in force or superseded; a
# magazine article is neither. A 2014 piece on how to book depreciation was
# correct then and is correct now. Using publication year as a proxy for legal
# currency hid 5,174 chunks of still-correct method from every current-state
# question.
#
# `valid_to` was worse. On "Koji je novi prag za ulazak u sustav PDV-a od
# 2025.?" the classifier asks for 2025-01-01, and the December 2024 issue —
# the issue where RRiF publishes next year's changes — failed
# `valid_to >= 2025-01-01`. That question returned nothing at all; without the
# constraint it returns five correct chunks at 0.999.
#
# So an article now never expires: valid_to = NULL, status = vazeci. Age is a
# ranking concern, not an eligibility test, and belongs in retrieval rather
# than in the row. See rag/query.py, TEMPORAL_MODE.


def _source_metadata(chunk) -> SourceMetadata:
    """Build SourceMetadata from an ArticleChunk."""
    year = chunk.year
    month = chunk.month

    # valid_from = first day of publication month. This one is meaningful: it
    # is when the article appeared, and it is what a recency ranking will read.
    valid_from = date(year, month, 1)

    # An article does not expire. See the note above.
    valid_to = None
    status = STATUS_VALID

    return SourceMetadata(
        category=chunk.default_category,
        source_type="članak",
        source=chunk.source_citation,
        law_name=None,
        nn_reference=None,
        valid_from=valid_from,
        valid_to=valid_to,
        status=status,
        citable=True,
        extra_metadata={
            "pub_type": chunk.pub_type,
            "pub_label": chunk.pub_label,
            "article_num": chunk.article_num,
            "author": chunk.author or "",
            "title": chunk.title,
        },
    )


def collect_pdfs(roots: list[Path]) -> list[Path]:
    """Collect all PDFs from given paths (files or folders, recursively one level)."""
    pdfs = []
    for root in roots:
        if root.is_file() and root.suffix.upper() == ".PDF":
            pdfs.append(root)
        elif root.is_dir():
            # Check if this is a leaf folder (e.g. RRIF2404) or a parent (e.g. Arhiva)
            sub_pdfs = list(root.glob("*.PDF")) + list(root.glob("*.pdf"))
            if sub_pdfs:
                # Leaf folder — collect directly
                pdfs.extend(sub_pdfs)
            else:
                # Parent folder — recurse one level into subfolders
                for sub in sorted(root.iterdir()):
                    if sub.is_dir():
                        pdfs.extend(sorted(sub.glob("*.PDF")))
                        pdfs.extend(sorted(sub.glob("*.pdf")))
    return sorted(set(pdfs))


def run(roots: list[Path], dry_run: bool = False):
    pdfs = collect_pdfs(roots)
    if not pdfs:
        print("No PDF files found.")
        return

    print(f"\n{'='*65}")
    print(f"  RRiF Article Ingestion")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE — writing to database'}")
    print(f"  PDFs found: {len(pdfs)}")
    print(f"{'='*65}\n")

    total_files = 0
    total_skipped = 0
    total_chunks = 0
    total_inserted = 0
    errors = []

    for pdf in pdfs:
        folder_name = pdf.parent.name

        result = load_article(pdf, verbose=False)

        if result is None:
            total_skipped += 1
            continue

        if not result:
            total_skipped += 1
            continue

        total_files += 1
        total_chunks += len(result)

        first = result[0]
        label = f"{folder_name}/{pdf.name}"
        toks = sorted(c.token_count for c in result)
        over = sum(1 for x in toks if x > 508)
        print(f"  {label:35s}  {len(result):3d} chunks  "
              f"tok {toks[0]}/{toks[len(toks)//2]}/{toks[-1]}"
              f"{'  ⚠ ' + str(over) + ' over 508' if over else ''}  "
              f"{first.title[:60]}")

        if dry_run:
            continue

        # All chunks in one article share the same SourceMetadata
        # (same pub date, category, citation prefix)
        meta = _source_metadata(first)

        # The pipeline's ingest() expects ArticleChunk-like objects with .text and .article_number
        # Our ArticleChunk has .text but not .article_number — wrap for compatibility
        class _WrappedChunk:
            def __init__(self, c):
                self.text = c.text
                self.article_number = c.section_num  # use section number as article_number
                self.chunk_index = c.chunk_index

        wrapped = [_WrappedChunk(c) for c in result]

        try:
            n = ingest(wrapped, meta)
            total_inserted += n
        except Exception as e:
            errors.append((label, str(e)))
            print(f"    ⚠️  ERROR: {e}")

    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"  Files processed : {total_files + total_skipped}")
    print(f"  Skipped         : {total_skipped}  (ads, editorials, short files)")
    print(f"  Articles loaded : {total_files}")
    print(f"  Chunks produced : {total_chunks}")
    if not dry_run:
        print(f"  Rows inserted   : {total_inserted}")
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for label, err in errors:
            print(f"    {label}: {err}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest RRiF magazine articles into the RAG knowledge base."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Folder(s) to ingest — can be individual issue folders (RRIF2404) or a parent archive folder",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without writing to the database",
    )
    args = parser.parse_args()

    run([Path(p) for p in args.paths], dry_run=args.dry_run)
