"""Smoke test for article_loader.py — no database, no embeddings.

Run this on a folder from your archive to verify the loader works
before ingesting anything into the database.

Usage (PowerShell from project root):
    python scripts\smoke_test_articles.py "C:\path\to\RRIF2404"
    python scripts\smoke_test_articles.py "C:\path\to\RRIF2404" "C:\path\to\RRIF2402"

Prints a structured summary: what was skipped, what was loaded,
how many chunks each article produced, with short previews.
"""
import sys
from pathlib import Path

# Make 'rag' importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.ingest.article_loader import load_article


def smoke_test(folders: list[Path]):
    pdfs = []
    for folder in folders:
        if folder.is_dir():
            seen = {p.name.upper(): p for p in folder.iterdir()
                    if p.suffix.upper() == ".PDF"}
            pdfs.extend(sorted(seen.values()))
        elif folder.suffix.upper() == ".PDF":
            pdfs.append(folder)

    if not pdfs:
        print("No PDF files found.")
        return

    print(f"\n{'='*70}")
    print(f"  RRiF Article Loader — Smoke Test")
    print(f"  Files found: {len(pdfs)}")
    print(f"{'='*70}\n")

    skipped = 0
    loaded  = 0
    total_chunks = 0

    for pdf in pdfs:
        print(f"{'─'*70}")
        print(f"FILE: {pdf.parent.name}/{pdf.name}")

        result = load_article(pdf, verbose=True)

        if result is None:
            skipped += 1
            continue

        if not result:
            print(f"  → LOADED but 0 chunks after filtering (all ads/noise)\n")
            skipped += 1
            continue

        loaded += 1
        total_chunks += len(result)
        first = result[0]

        print(f"  date     : {first.pub_date}  article #{first.article_num}")
        print(f"  title    : {first.title[:75]}")
        print(f"  author   : {first.author}")
        print(f"  chunks   : {len(result)}")
        print()

        for c in result:
            sec = f"§{c.section_num}. {c.section_title[:30]}" if c.section_num else "[intro]"
            preview = c.text[:100].replace('\n', ' ')
            print(f"  [{c.chunk_index:02d}] {sec:40s} {c.char_count:5d}ch")
            print(f"       {preview!r}")
        print()

    print(f"{'='*70}")
    print(f"  SUMMARY")
    print(f"  Files processed : {len(pdfs)}")
    print(f"  Skipped         : {skipped}  (ads, editorials, short files)")
    print(f"  Loaded          : {loaded}")
    print(f"  Total chunks    : {total_chunks}")
    print(f"  Avg chunks/file : {total_chunks/loaded:.1f}" if loaded else "")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python scripts\\smoke_test_articles.py <folder_or_pdf> [...]")
        print("Example: python scripts\\smoke_test_articles.py C:\\path\\to\\RRIF2404")
        sys.exit(1)

    smoke_test([Path(a) for a in args])
