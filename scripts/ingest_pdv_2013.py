"""Ingest the 2013 Zakon o PDV-u from the local PDF.

Validates the full pipeline end-to-end on one law, before we wire up
nn.hr fetching. Run from project root:

    python scripts\\ingest_pdv_2013.py
"""
from datetime import date
from pathlib import Path
import sys

# Make 'rag' importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.ingest.pdf_loader import extract_text, split_by_article
from rag.ingest.pipeline import SourceMetadata, ingest


PDF_PATH = Path("data/raw/laws/zakon_o_pdv_2013.pdf")

META = SourceMetadata(
    category="PDV",
    source_type="zakon",
    source="Zakon o porezu na dodanu vrijednost, NN 73/2013, čl. ...",
    law_name="Zakon o porezu na dodanu vrijednost",
    nn_reference="NN 73/2013",
    # NN 73/2013 is the base Zakon o PDV-u and is still in force, amended.
    # Marking it `nevazeci` (as this script did until 7 Sep 2026) makes every
    # law chunk invisible to `status = 'vazeci'`, which is what TEMPORAL_MODE
    # applies to statute — so all 188 chunks dropped out of every current-state
    # question and ZAKONI article top-5 read 0.0%.
    #
    # scripts/fix_pdv_status.py existed to undo this after the fact. A two-step
    # ingest whose second step is optional is a step that gets forgotten: it
    # was, on the v2 corpus, and cost a full evaluation round to find. The
    # status is now correct at write time and that script is redundant.
    #
    # When consolidated texts and amendments arrive in F2, this becomes a real
    # versioning problem and valid_from/valid_to earn their keep. Until then a
    # single base law in force is what the row should say.
    valid_from=date(2013, 7, 1),
    valid_to=None,
    status="vazeci",
    citable=True,
    extra_metadata={"ingestion_source": "manual_pdf", "pdf_filename": PDF_PATH.name},
)


def main() -> int:
    if not PDF_PATH.exists():
        print(f"PDF not found at {PDF_PATH.resolve()}")
        return 1

    print(f"Reading {PDF_PATH}...")
    text = extract_text(PDF_PATH)
    print(f"Extracted {len(text):,} characters.")

    chunks = split_by_article(text)
    print(f"Split into {len(chunks)} chunks.")

    by_article = sum(1 for c in chunks if c.article_number)
    print(f"  {by_article} have an article number, {len(chunks) - by_article} are preamble/other.")

    if not chunks:
        print("No chunks produced — aborting.")
        return 1

    n_inserted = ingest(chunks, META)
    print(f"\nDone. {n_inserted} rows in `chunks` table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
