"""Article loader for RRiF magazine PDFs — v2.

Changes from v1:
- Category auto-detected from repeating ALL-CAPS page banner
- Title drop-cap artefact fixed (single leading letter joined to next line)
- Ads chunks filtered out (contact-info density check)
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import fitz  # pymupdf

MIN_CHARS        = 4_000
MAX_CHARS_PER_CHUNK = 1_500

SECTION_PATTERN = re.compile(
    r'^\s*(\d{1,2})\.\t([A-ZČĆŠŽĐ][^\n]{3,})',
    re.MULTILINE,
)
SECTION_PATTERN_FALLBACK = re.compile(
    r'^\s*(\d{1,2})\.\s{2,}([A-ZČĆŠŽĐ][A-ZČĆŠŽĐ\s\-/,\.]{4,})$',
    re.MULTILINE,
)

PUB_TYPE = {
    "RRIF": "RRiF",
    "PIP":  "Porezno i pravno (PiP)",
    "PROR": "Proračun",
    "OBAV": "Obavijesti",
    "NEPR": "Neprofitne organizacije",
}

# Map page-banner labels → category values used in the chunks table
BANNER_TO_CATEGORY = {
    "RAČUNOVODSTVO":                        "računovodstvo",
    "POREZI":                               "porezi",
    "PLAĆE I NADOKNADE":                    "plaće",
    "RADNO I SOCIJALNO PRAVO":              "radno pravo",
    "TRGOVAČKO PRAVO":                      "trgovačko pravo",
    "PRAVO TRGOVAČKIH DRUŠTAVA":            "trgovačko pravo",
    "REVIZIJA":                             "revizija",
    "POSLOVANJE S INOZEMSTVOM":             "poslovanje s inozemstvom",
    "POSLOVANJE PRORAČUNA I PRORAČUNSKIH KORISNIKA": "proračun",
    "PRORAČUNSKO RAČUNOVODSTVO":            "proračun",
    "RAČUNOVODSTVO NEPROFITNIH ORGANIZACIJA": "neprofitne organizacije",
    "TRŽIŠTE I PROPISI":                    "tržište i propisi",
    "UPRAVLJANJE I USTROJ":                 "upravljanje",
    "VIJESTI IZ INSTITUCIJA EU":            "EU propisi",
    "NOVI PROPISI":                         "novi propisi",
    "STRUČNE INFORMACIJE":                  "stručne informacije",
}

DEFAULT_CATEGORY_BY_PUB = {
    "RRIF": "računovodstvo",
    "PIP":  "porezi",
    "PROR": "proračun",
    "OBAV": "ostalo",
    "NEPR": "neprofitne organizacije",
}

# Chunk is likely an ads page if it has this many phone/email/web patterns
ADS_PATTERN = re.compile(
    r'(tel\.|mob\.|fax\.|@|\bwww\b|d\.o\.o\.|obrt\b)',
    re.IGNORECASE,
)
ADS_THRESHOLD = 4  # more than this many hits → skip chunk


@dataclass
class ArticleChunk:
    pub_type:         str
    pub_label:        str
    year:             int
    month:            int
    article_num:      str
    pub_date:         date
    title:            str
    author:           str | None
    section_num:      str | None
    section_title:    str | None
    chunk_index:      int
    text:             str
    char_count:       int
    source_citation:  str
    default_category: str


def _parse_path(pdf_path: Path) -> tuple[str, int, int, str] | None:
    folder = pdf_path.parent.name.upper()
    stem   = pdf_path.stem.upper()
    fm = re.match(r'^([A-Z]+)(\d{2})(\d{2})$', folder)
    if not fm:
        return None
    pub_type = fm.group(1)
    year     = 2000 + int(fm.group(2))
    month    = int(fm.group(3))
    sm = re.match(r'^[A-Z](\d{2})(\d{2})(\d{2,3})$', stem)
    if not sm:
        return None
    return pub_type, year, month, sm.group(3)


def _knjizenje_to_text(table) -> str:
    """Convert a journal entry (Knjiženje) table to structured readable text."""
    try:
        cells = table.extract()
    except Exception:
        return ""
    if not cells or len(cells) < 2:
        return ""

    # Find header row containing Duguje/Potražuje
    data_start = 0
    for i, row in enumerate(cells[:3]):
        row_text = " ".join(str(c or "") for c in row)
        if "Duguje" in row_text or "Potražuje" in row_text:
            data_start = i + 1
            break
    else:
        return ""  # Not a journal entry table

    lines = ["Knjiženje:"]
    lines.append(f"{'Br.':5} {'OPIS':40} {'Račun':8} {'Duguje':>13} {'Potražuje':>13}")
    lines.append("-" * 82)

    for row in cells[data_start:]:
        if not row or not any(row):
            continue

        br_cell    = str(row[0] or "").strip()
        opis_cell  = str(row[1] or "").strip() if len(row) > 1 else ""
        racun_cell = str(row[2] or "").strip() if len(row) > 2 else ""
        dug_cell   = str(row[3] or "").strip() if len(row) > 3 else ""
        pot_cell   = str(row[4] or "").strip() if len(row) > 4 else ""

        # Section header rows
        if not br_cell and not racun_cell and not dug_cell and not pot_cell and opis_cell:
            lines.append(f"\n  [{opis_cell}]")
            continue

        opis_lines  = [l.strip() for l in opis_cell.split("\n")  if l.strip()]
        racun_lines = [l.strip() for l in racun_cell.split("\n") if l.strip()]
        dug_lines   = [l.strip() for l in dug_cell.split("\n")   if l.strip()]
        pot_lines   = [l.strip() for l in pot_cell.split("\n")   if l.strip()]

        max_r = max(len(opis_lines), len(racun_lines),
                    len(dug_lines), len(pot_lines), 1)

        for i in range(max_r):
            o = opis_lines[i]  if i < len(opis_lines)  else ""
            r = racun_lines[i] if i < len(racun_lines) else ""
            d = dug_lines[i]   if i < len(dug_lines)   else ""
            p = pot_lines[i]   if i < len(pot_lines)   else ""
            b = br_cell        if i == 0               else ""
            lines.append(f"{b:5} {o:40} {r:8} {d:>13} {p:>13}")

    lines.append("")
    return "\n".join(lines)


def _is_knjizenje(table) -> bool:
    try:
        cells = table.extract()
        for row in cells[:3]:
            row_text = " ".join(str(c or "") for c in row)
            if "Duguje" in row_text or "Potražuje" in row_text:
                return True
    except Exception:
        pass
    return False


def _extract_text(pdf_path: Path) -> str:
    """Extract text from PDF, replacing journal entry tables with structured text."""
    doc = fitz.open(pdf_path)
    page_texts = []

    for page in doc:
        # Find all table bounding boxes on this page
        tables = page.find_tables()
        table_bboxes = []
        table_replacements = {}  # bbox -> replacement text

        for table in tables.tables:
            if table.row_count < 2:
                continue
            if _is_knjizenje(table):
                replacement = _knjizenje_to_text(table)
                if replacement.strip():
                    bbox = table.bbox  # (x0, y0, x1, y1)
                    table_bboxes.append(bbox)
                    table_replacements[bbox] = replacement

        if not table_bboxes:
            # No tables — plain text extraction
            page_texts.append(page.get_text("text"))
            continue

        # Mix: extract text blocks, skip blocks inside table bboxes,
        # insert table replacements at the right position
        blocks = page.get_text("blocks")  # returns (x0,y0,x1,y1,text,block_no,type)
        blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))  # sort top-to-bottom

        used_tables = set()
        page_parts = []

        for block in blocks_sorted:
            bx0, by0, bx1, by1 = block[:4]
            block_text = block[4]

            # Check if this block overlaps with any table bbox
            in_table = False
            for tbbox in table_bboxes:
                tx0, ty0, tx1, ty1 = tbbox
                # Overlap check
                if bx0 < tx1 and bx1 > tx0 and by0 < ty1 and by1 > ty0:
                    in_table = True
                    # Insert table replacement before first overlapping block
                    if tbbox not in used_tables:
                        page_parts.append(table_replacements[tbbox])
                        used_tables.add(tbbox)
                    break

            if not in_table:
                page_parts.append(block_text)

        page_texts.append("\n".join(page_parts))

    doc.close()
    return "\n".join(page_texts)


def _detect_category(text: str, pub_type: str) -> str:
    """Find the most frequent ALL-CAPS banner line — that's the section label."""
    banners = re.findall(r'\n([A-ZČĆŠŽĐ][A-ZČĆŠŽĐ\s]{3,50})\n', text)
    counts = Counter(b.strip() for b in banners if len(b.strip()) > 3)
    # Ignore generic noise
    ignore = {"O P I S", "RRIF", "RRiF"}
    for label, _ in counts.most_common(10):
        if label in ignore:
            continue
        if label in BANNER_TO_CATEGORY:
            return BANNER_TO_CATEGORY[label]
    return DEFAULT_CATEGORY_BY_PUB.get(pub_type, "ostalo")


def _extract_title_and_author(text: str) -> tuple[str, str | None]:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    skip = re.compile(
        r'^(\d+|UDK\s|RRIF|RRiF|veljača|siječanj|ožujak|travanj|svibanj|'
        r'lipanj|srpanj|kolovoz|rujan|listopad|studeni|prosinac|Priredila)\b',
        re.IGNORECASE,
    )
    author_pat = re.compile(r'(Dr\.\s*sc\.|dipl\.|prof\.|ovl\.|mr\.\s*sc\.|mag\.|izv\.)', re.I)

    author = None
    title_candidates: list[str] = []

    i = 0
    while i < min(len(lines), 40):
        line = lines[i]
        if skip.match(line):
            i += 1
            continue
        if author_pat.search(line):
            author = line
            i += 1
            continue
        # Skip pure ALL-CAPS section banners (short, no lowercase)
        if re.match(r'^[A-ZČĆŠŽĐ\s\-/\.]{3,}$', line) and len(line) < 40:
            i += 1
            continue
        # Fix drop-cap: single letter on its own line followed by rest of word
        if re.match(r'^[A-ZČĆŠŽĐ]$', line) and i + 1 < len(lines):
            merged = line + lines[i + 1].lstrip()
            title_candidates.append(merged)
            i += 2
            continue
        if len(line) > 8:
            title_candidates.append(line)
        i += 1

    # Deduplicate (two-column layout duplicates lines)
    seen: list[str] = []
    for l in title_candidates[:8]:
        if l not in seen:
            seen.append(l)

    title = re.sub(r'\s+', ' ', ' '.join(seen[:4])).strip()
    return title or "Nepoznat naslov", author


def _is_ads_chunk(text: str) -> bool:
    """Return True if the chunk looks like an ads/contact-info page."""
    hits = len(ADS_PATTERN.findall(text))
    return hits >= ADS_THRESHOLD


def _find_sections(text: str) -> list[re.Match]:
    matches = list(SECTION_PATTERN.finditer(text))
    if len(matches) >= 2:
        return matches
    fb = list(SECTION_PATTERN_FALLBACK.finditer(text))
    return fb if len(fb) >= 2 else matches


def _split_into_sections(text: str) -> list[tuple[str | None, str | None, str]]:
    matches = _find_sections(text)
    sections = []
    if matches:
        intro = text[:matches[0].start()].strip()
        if len(intro) > 100:
            sections.append((None, None, intro))
        for i, m in enumerate(matches):
            sec_num   = m.group(1)
            sec_title = re.sub(r'\s+', ' ', m.group(2).strip())
            start = m.start()
            end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body  = text[start:end].strip()
            if body:
                sections.append((sec_num, sec_title, body))
    else:
        sections.append((None, None, text.strip()))
    return sections


def _sub_split(text: str) -> list[str]:
    if len(text) <= MAX_CHARS_PER_CHUNK:
        return [text]
    pieces: list[str] = []
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) + 2 > MAX_CHARS_PER_CHUNK and current:
            pieces.append('\n\n'.join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 2
    if current:
        pieces.append('\n\n'.join(current))
    return pieces or [text]


def load_article(pdf_path: Path | str, verbose: bool = True) -> list[ArticleChunk] | None:
    pdf_path = Path(pdf_path)
    parsed = _parse_path(pdf_path)
    if not parsed:
        if verbose:
            print(f"  [skip] {pdf_path.name}: path doesn't match naming convention")
        return None

    pub_type, year, month, article_num = parsed
    pub_label = PUB_TYPE.get(pub_type, pub_type)
    pub_date  = date(year, month, 1)

    text = _extract_text(pdf_path)
    if len(text) < MIN_CHARS:
        if verbose:
            print(f"  [skip] {pdf_path.name}: {len(text)} chars < {MIN_CHARS} (ads/short)")
        return None

    category  = _detect_category(text, pub_type)
    title, author = _extract_title_and_author(text)
    sections  = _split_into_sections(text)
    base_citation = f"{pub_label} br. {month}/{year} — {title}"

    chunks: list[ArticleChunk] = []
    idx = 0
    ads_skipped = 0

    for sec_num, sec_title, body in sections:
        for piece in _sub_split(body):
            if len(piece.strip()) < 100:
                continue
            if _is_ads_chunk(piece):
                ads_skipped += 1
                continue
            sec_label = f" § {sec_num}. {sec_title}" if sec_num else ""
            chunks.append(ArticleChunk(
                pub_type=pub_type, pub_label=pub_label,
                year=year, month=month, article_num=article_num,
                pub_date=pub_date, title=title, author=author,
                section_num=sec_num, section_title=sec_title,
                chunk_index=idx, text=piece, char_count=len(piece),
                source_citation=f"{base_citation}{sec_label}",
                default_category=category,
            ))
            idx += 1

    if verbose and ads_skipped:
        print(f"  [info] {pdf_path.name}: {ads_skipped} ads chunk(s) filtered")

    return chunks
