"""PDF loader for Croatian legal documents.

Extracts plain text from a PDF and splits it into chunks at člának boundaries.
Designed for Croatian laws (Zakon o ...) where 'Članak N.' is the natural unit
of meaning — each article is a self-contained legal provision and lines up
nicely with how questions get asked ("što kaže članak 38. Zakona o PDV-u?").
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # pymupdf


# Token budget per chunk. e5-large has a 512-token input limit; we leave
# headroom because Croatian text encodes slightly more tokens per char than
# English. ~1500 chars ≈ 400-450 tokens, comfortably within the limit.
MAX_CHARS_PER_CHUNK = 1500

# Match "Članak 38." or "Članak 38a." at start of a line. The capturing group
# keeps the article number so we can attach it to the chunk metadata.
ARTICLE_PATTERN = re.compile(
    r"^\s*Članak\s+(\d+[a-z]?)\s*\.",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class ArticleChunk:
    """One chunk's content + structural metadata extracted from the PDF."""
    article_number: str | None  # "38", "38a", or None for preamble/closing
    chunk_index: int            # position within the source document
    text: str


def extract_text(pdf_path: Path | str) -> str:
    """Read all pages of a PDF and return concatenated plain text."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    try:
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return "\n".join(pages)


def split_by_article(text: str) -> list[ArticleChunk]:
    """Split a law's text into per-article chunks.

    Articles longer than MAX_CHARS_PER_CHUNK are sub-split on paragraph
    boundaries with the article number preserved on each piece.
    """
    matches = list(ARTICLE_PATTERN.finditer(text))

    if not matches:
        # No "Članak N." headings detected — fall back to whole-document chunks.
        return _fallback_chunks(text)

    chunks: list[ArticleChunk] = []

    # Anything before the first article (preamble, table of contents, etc.)
    preamble = text[: matches[0].start()].strip()
    if preamble:
        for piece in _sub_split(preamble):
            chunks.append(ArticleChunk(
                article_number=None,
                chunk_index=len(chunks),
                text=piece,
            ))

    for i, m in enumerate(matches):
        article_no = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        article_text = text[start:end].strip()
        if not article_text:
            continue

        for piece in _sub_split(article_text):
            chunks.append(ArticleChunk(
                article_number=article_no,
                chunk_index=len(chunks),
                text=piece,
            ))

    return chunks


def _sub_split(block: str) -> list[str]:
    """Split a block on paragraph boundaries if it's too long. Otherwise pass through."""
    if len(block) <= MAX_CHARS_PER_CHUNK:
        return [block]

    pieces: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", block) if p.strip()]
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        # If a single paragraph exceeds the budget on its own, force-emit
        # what we have and let the long paragraph through whole. We'd rather
        # over-shoot the budget by a bit than slice mid-sentence.
        if current_len + len(para) + 2 > MAX_CHARS_PER_CHUNK and current:
            pieces.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 2

    if current:
        pieces.append("\n\n".join(current))
    return pieces


def _fallback_chunks(text: str) -> list[ArticleChunk]:
    """No article headings found — chunk by paragraph blocks."""
    pieces = _sub_split(text)
    return [
        ArticleChunk(article_number=None, chunk_index=i, text=piece)
        for i, piece in enumerate(pieces)
    ]


if __name__ == "__main__":
    # Quick CLI: python -m rag.ingest.pdf_loader <path-to-pdf>
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m rag.ingest.pdf_loader <pdf_path>")
        sys.exit(1)

    text = extract_text(sys.argv[1])
    chunks = split_by_article(text)
    print(f"Extracted {len(text):,} chars, {len(chunks)} chunks.\n")
    for c in chunks[:3]:
        label = f"Članak {c.article_number}" if c.article_number else "[preamble]"
        preview = c.text[:200].replace("\n", " ")
        print(f"#{c.chunk_index:03d}  {label:15s}  {len(c.text):4d} chars  {preview!r}\n")
