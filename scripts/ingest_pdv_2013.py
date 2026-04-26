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
    # Original 2013 text. As of today the version in force has been amended
    # many times — we'll handle consolidated text + amendments in a later ADR.
    # For this initial ingestion we mark this version as historical so it
    # doesn't surface for "current law" queries.
    valid_from=date(2013, 7, 1),
    valid_to=date(2013, 12, 31),
    status="nevazeci",
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
