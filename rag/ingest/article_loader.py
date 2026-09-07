"""Article loader for RRiF magazine PDFs — v3.

WHY THIS WAS REWRITTEN
──────────────────────

A corpus audit on 7 Sep 2026 (`scripts/audit_corpus.py`) over the 12,749 chunks
v2 produced found:

    source label is body text     96.6%
    soft hyphens (U+00AD)         54.2%
    hyphen-newline splits         42.8%
    chunks over 4,000 chars       31.3%
    page furniture: URLs           5.0%

and `scripts/check_embed_truncation.py` found that **55.6% of magazine chunks
exceed the embedding model's 512-token window**, with 48.7% of all corpus
tokens never reaching the embedder at all. The median chunk was 649 tokens
against a budget of 508.

Four defects in v2, each fixed here.

1. THE SIZE CAP WAS UNENFORCEABLE

   `MAX_CHARS_PER_CHUNK = 1_500` existed, but `_sub_split` only broke text on
   `\\n{2,}` — blank lines — while `_extract_text` joined blocks and pages with
   single newlines. On any page that took the table-aware `blocks` path there
   was not one blank line in the output, so the whole section came back as a
   single chunk. Hence a nominal cap of 1,500 producing a 9,728-character
   maximum.

   v3 splits sections → paragraphs → sentences → hard token slice, so there is
   always a smaller unit to fall back to.

2. THE CAP WAS IN THE WRONG UNIT

   Characters are the wrong currency. The corpus averages 3.67 characters per
   token in prose, but a Knjiženje table of account numbers and amounts runs
   closer to 2 — so a character cap leaves exactly the accounting tables
   truncated, which is the content advisors most need. v3 measures with the
   embedding model's own tokenizer.

3. `_extract_title_and_author` DID NOT EXTRACT A TITLE

   It walked the first 40 lines, dropped ones matching a skip pattern, then
   joined `seen[:4]` with spaces. On a two-column magazine page the first
   surviving lines are body text, so the "title" was four lines of prose — and
   it went into `source_citation`, which is what an advisor sees as the
   citation under an answer. That is why 96.6% of `source` values are over 120
   characters.

4. THE ADS FILTER ATE REAL ARTICLES

   `ADS_THRESHOLD = 4` over a pattern including `d\\.o\\.o\\.` and `@`, counted
   absolutely rather than by density. Accounting articles say "d.o.o."
   constantly. v3 counts per 1,000 characters and requires several distinct
   marker types, so a long article mentioning companies is no longer mistaken
   for a page of adverts.

Also: soft hyphens and hyphen-newline pairs are repaired at extraction, and
page furniture (source URLs, "9 of 71" pagination, PDF export timestamps) is
dropped before chunking rather than being embedded and reranked as content.
"""
from __future__ import annotations

import functools
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import fitz  # pymupdf

# ── Sizing ────────────────────────────────────────────────────────────────────

EMBED_MODEL = "intfloat/multilingual-e5-large"

# The model window is 512. Two tokens go to the special tokens and two to the
# "passage: " prefix e5 requires, leaving 508. Targeting 400 leaves headroom
# for content that tokenises worse than prose — tables, long numbers, mixed
# Croatian and abbreviations — so a bad estimate costs a short chunk rather
# than a silently truncated one.
MAX_TOKENS = 400
HARD_CAP_TOKENS = 508
OVERLAP_TOKENS = 60

# A chunk this small carries no usable context; it is merged into its neighbour
# rather than stored.
MIN_CHUNK_TOKENS = 40

MIN_CHARS = 4_000          # whole-document floor: below this it is an advert or filler


@functools.lru_cache(maxsize=1)
def _tokenizer():
    """The embedding model's tokenizer, without loading the model itself.

    AutoTokenizer pulls a few hundred KB; SentenceTransformer would pull ~2 GB
    of weights the loader never uses.
    """
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    # "Token indices sequence length is longer than the specified maximum" is
    # emitted for every over-length string we *count*. We never run the model
    # here, so it is noise — and across a full ingest it prints thousands of
    # times and buries anything worth reading.
    hf_logging.set_verbosity_error()
    return AutoTokenizer.from_pretrained(EMBED_MODEL)


def n_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


# ── Text cleaning ─────────────────────────────────────────────────────────────

SOFT_HYPHEN = "­"
NBSP = " "

# Line-broken words: "djelat-\nnostima" → "djelatnostima". Only when both sides
# are lowercase letters, so "PDV-\nobveznik" and "2013-\n2014" survive.
_HYPHEN_BREAK = re.compile(r"([a-zà-ž])-\n([a-zà-ž])")

# Furniture that PDF export leaves in the text layer.
_FURNITURE = [
    re.compile(r"https?://\S+"),                                  # source URLs
    re.compile(r"^\s*\d{1,4}\s+of\s+\d{1,4}\s*$", re.MULTILINE),  # "9 of 71"
    re.compile(r"\d{1,2}/\d{1,2}/\d{4},\s*\d{1,2}:\d{2}\s*[AP]M"),  # export stamp
]


def clean_text(text: str) -> str:
    """Repair PDF extraction damage. Applied once, before any splitting.

    Order matters: soft hyphens are removed before the hyphen-newline repair,
    because a word can carry both.
    """
    text = text.replace(SOFT_HYPHEN, "").replace(NBSP, " ")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    for pat in _FURNITURE:
        text = pat.sub(" ", text)
    # Collapse runs of blank lines to exactly one blank line, so paragraph
    # boundaries stay detectable but do not multiply.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Sentence splitting ────────────────────────────────────────────────────────

# Croatian legal and accounting prose is dense with abbreviations that end in a
# full stop and must not end a sentence. Ordinals ("2024.", "čl. 38.") are the
# same problem. This is a lookbehind-guarded split rather than a full sentence
# tokeniser: cheap, and wrong splits cost a slightly odd chunk boundary rather
# than lost text.
_ABBREV = {
    "br", "čl", "st", "t", "toč", "dr", "mr", "sc", "prof", "npr", "tzv",
    "odn", "tj", "god", "kn", "sl", "ur", "izv", "dipl", "oec", "iur",
    "d", "o", "j", "ing", "spec", "usp", "str", "op", "cit",
}

# Every place a sentence *could* end. Whether it actually does is decided by
# looking at the word before the stop, because Python's re has no
# variable-width lookbehind and the abbreviation list is variable-width.
_SENT_CANDIDATE = re.compile(r"[.!?]\s+(?=[A-ZČĆŠŽĐ])")
_TRAILING_WORD = re.compile(r"(\w+)\s*[.!?]\s*$")


def split_sentences(text: str) -> list[str]:
    """Sentence split for Croatian legal and accounting prose.

    The hazard is that this register is dense with abbreviations ending in a
    full stop — "čl. 38. st. 1.", "d.o.o.", "dipl. oec." — and with ordinals,
    where "2024." is a year rather than the end of a thought. Splitting naively
    on ". " shatters exactly the sentences that carry the citations.

    Wrong splits cost an odd chunk boundary, never lost text, so a heuristic is
    the right tool here rather than a full tokeniser.
    """
    out: list[str] = []
    start = 0
    for m in _SENT_CANDIDATE.finditer(text):
        piece = text[start:m.start() + 1]
        w = _TRAILING_WORD.search(piece)
        last = w.group(1).lower() if w else ""
        if last in _ABBREV or last.isdigit():
            continue                     # abbreviation or ordinal, not a stop
        if piece.strip():
            out.append(piece.strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out or [text.strip()]


# ── Token-budget packing ──────────────────────────────────────────────────────

def _hard_slice(text: str, budget: int) -> list[str]:
    """Last resort: a single unit that exceeds the budget on its own.

    Slices on token boundaries and decodes back, so the result is never cut
    mid-token. Reached by very long unbroken tables; rare, but it is the case
    that silently produced 9,728-character chunks in v2.
    """
    tok = _tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    return [
        tok.decode(ids[i:i + budget], skip_special_tokens=True).strip()
        for i in range(0, len(ids), budget)
    ]


def _overlap_tail(units: list[str], budget: int) -> tuple[list[str], int]:
    """Trailing text from the finished chunk to seed the next one.

    Whole units first. If not even one fits — which is the common case, since
    a paragraph is usually larger than the overlap budget — fall back to the
    trailing sentences of the last unit. Without that fallback the overlap
    silently never happens, which is exactly the kind of feature that looks
    implemented and isn't.
    """
    tail: list[str] = []
    used = 0
    for prev in reversed(units):
        pt = n_tokens(prev) + 1
        if used + pt > budget:
            break
        tail.insert(0, prev)
        used += pt
    if tail:
        return tail, used

    sentences = split_sentences(units[-1])
    for sent in reversed(sentences):
        st = n_tokens(sent) + 1
        if used + st > budget:
            break
        tail.insert(0, sent)
        used += st
    return tail, used


def pack_units(units: list[str], max_tokens: int = MAX_TOKENS,
               overlap_tokens: int = OVERLAP_TOKENS) -> list[str]:
    """Greedily pack text units into chunks under the token budget.

    Each new chunk is seeded with the tail of the previous one, so a fact split
    across a boundary still appears whole in at least one chunk. Overlap is
    measured in tokens rather than units, so a chunk ending in one long
    paragraph does not drag half of it into the next.

    The +1 per unit is the newline the join inserts. Ignoring it looks harmless
    on a handful of paragraphs and quietly costs ~100 tokens on a chunk built
    from a hundred short table rows.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        ut = n_tokens(unit)

        if ut > max_tokens:
            if current:
                chunks.append("\n".join(current))
                current, current_tokens = [], 0
            chunks.extend(_hard_slice(unit, HARD_CAP_TOKENS))
            continue

        if current and current_tokens + ut + 1 > max_tokens:
            chunks.append("\n".join(current))
            current, current_tokens = _overlap_tail(current, overlap_tokens)

        current.append(unit)
        current_tokens += ut + 1

    if current:
        chunks.append("\n".join(current))

    if len(chunks) > 1 and n_tokens(chunks[-1]) < MIN_CHUNK_TOKENS:
        chunks[-2] = chunks[-2] + "\n" + chunks[-1]
        chunks.pop()

    return chunks


def split_body(text: str) -> list[str]:
    """Section body → chunks, descending through the available split points.

    Paragraphs first, because a paragraph boundary is a real semantic break.
    Sentences where a paragraph is too big — which is most of this corpus,
    since the block-based extraction path produces no blank lines at all.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return []

    units: list[str] = []
    for para in paragraphs:
        if n_tokens(para) <= MAX_TOKENS:
            units.append(para)
        else:
            units.extend(split_sentences(para))

    return pack_units(units)


# ── Publication metadata ──────────────────────────────────────────────────────

PUB_TYPE = {
    "RRIF": "RRiF",
    "PIP":  "Porezno i pravno (PiP)",
    "PROR": "Proračun",
    "OBAV": "Obavijesti",
    "NEPR": "Neprofitne organizacije",
}

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

SECTION_PATTERN = re.compile(
    r'^\s*(\d{1,2})\.\t([A-ZČĆŠŽĐ][^\n]{3,})',
    re.MULTILINE,
)
SECTION_PATTERN_FALLBACK = re.compile(
    r'^\s*(\d{1,2})\.\s{2,}([A-ZČĆŠŽĐ][A-ZČĆŠŽĐ\s\-/,\.]{4,})$',
    re.MULTILINE,
)


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
    token_count:      int
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


# ── Journal-entry tables ──────────────────────────────────────────────────────
# Unchanged from v2: this part worked. A Knjiženje table rendered as aligned
# text is far more useful to both the embedder and the generator than the
# column-major soup pymupdf's plain text extraction produces.

def _knjizenje_to_text(table) -> str:
    try:
        cells = table.extract()
    except Exception:
        return ""
    if not cells or len(cells) < 2:
        return ""

    data_start = 0
    for i, row in enumerate(cells[:3]):
        row_text = " ".join(str(c or "") for c in row)
        if "Duguje" in row_text or "Potražuje" in row_text:
            data_start = i + 1
            break
    else:
        return ""

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
    doc = fitz.open(pdf_path)
    page_texts = []

    for page in doc:
        tables = page.find_tables()
        table_bboxes = []
        table_replacements = {}

        for table in tables.tables:
            if table.row_count < 2:
                continue
            if _is_knjizenje(table):
                replacement = _knjizenje_to_text(table)
                if replacement.strip():
                    bbox = table.bbox
                    table_bboxes.append(bbox)
                    table_replacements[bbox] = replacement

        if not table_bboxes:
            page_texts.append(page.get_text("text"))
            continue

        blocks = page.get_text("blocks")
        blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))

        used_tables = set()
        page_parts = []

        for block in blocks_sorted:
            bx0, by0, bx1, by1 = block[:4]
            block_text = block[4]

            in_table = False
            for tbbox in table_bboxes:
                tx0, ty0, tx1, ty1 = tbbox
                if bx0 < tx1 and bx1 > tx0 and by0 < ty1 and by1 > ty0:
                    in_table = True
                    if tbbox not in used_tables:
                        page_parts.append(table_replacements[tbbox])
                        used_tables.add(tbbox)
                    break

            if not in_table:
                page_parts.append(block_text)

        # Blank line between blocks. v2 joined with a single newline, which
        # left the paragraph splitter with nothing to split on — the direct
        # cause of the oversized chunks.
        page_texts.append("\n\n".join(p.strip() for p in page_parts if p.strip()))

    doc.close()
    return "\n\n".join(page_texts)


def _detect_category(text: str, pub_type: str) -> str:
    banners = re.findall(r'\n([A-ZČĆŠŽĐ][A-ZČĆŠŽĐ\s]{3,50})\n', text)
    counts = Counter(b.strip() for b in banners if len(b.strip()) > 3)
    ignore = {"O P I S", "RRIF", "RRiF"}
    for label, _ in counts.most_common(10):
        if label in ignore:
            continue
        if label in BANNER_TO_CATEGORY:
            return BANNER_TO_CATEGORY[label]
    return DEFAULT_CATEGORY_BY_PUB.get(pub_type, "ostalo")


# ── Title and author ──────────────────────────────────────────────────────────

MAX_TITLE_CHARS = 140

_SKIP_LINE = re.compile(
    r'^(\d+|UDK\b|RRIF|RRiF|veljača|siječanj|ožujak|travanj|svibanj|'
    r'lipanj|srpanj|kolovoz|rujan|listopad|studeni|prosinac|Priredila|'
    r'Računovodstvo, revizija i financije)',
    re.IGNORECASE,
)
_AUTHOR = re.compile(
    r'(Dr\.\s*sc\.|dipl\.|prof\.|ovl\.|mr\.\s*sc\.|mag\.|izv\.|univ\.\s*spec\.)',
    re.I,
)
_SECTION_START = re.compile(r'^\d{1,2}\.\s')
_LEADING_DASH = re.compile(r'^(?:[–—-]\s*)+')
_DASH_RUN = re.compile(r'[–—-]\s*[–—-]')

TITLE_SEARCH_LINES = 60
MIN_DUP_TITLE_CHARS = 4


def _title_key(line: str) -> str:
    """Comparison form for spotting a duplicated headline line.

    Leading dashes are stripped because the two copies do not always agree on
    them — the outline pass can carry "– –" where the fill pass has nothing.
    """
    return re.sub(r"\s+", " ", _LEADING_DASH.sub("", line)).strip().lower()


_ADS_LINE = re.compile(
    r'@|\bwww\.|\+\d{3}\s*\d|\btel\.|\bmob\.|\bfax\.', re.I)


def _title_from_layout(pdf_path: Path) -> str | None:
    """The title is the largest type on the first page.

    Position-based rules kept breaking because the templates disagree. In the
    2024 issues the title follows the byline and each line is drawn twice; in
    2014 it precedes the byline, is not duplicated, and may sit under a
    full-page advert whose lines look exactly like title fragments. Trying to
    encode both — and then whatever PiP and Proračun do — is a losing game.

    Type size is the one thing every magazine layout agrees on. pymupdf exposes
    it per span, so we take the largest size on page one, collect every span at
    that size in reading order, and join them. Duplicate spans collapse (the
    2024 outline-plus-fill artefact), and a run carrying advert markers is
    rejected in favour of the next size down.

    Returns None when nothing plausible is found, so the caller can fall back
    to the text heuristics.
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        data = page.get_text("dict")
        doc.close()
    except Exception:
        return None

    spans: list[tuple[float, float, float, str]] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:          # 0 = text
            continue
        for line in block.get("lines", []):
            for s in line.get("spans", []):
                txt = re.sub(r"\s+", " ", s.get("text", "")).strip()
                if len(txt) < 2:
                    continue
                spans.append((round(float(s.get("size", 0)), 1),
                              float(s["bbox"][1]), float(s["bbox"][0]), txt))
    if not spans:
        return None

    # Descending distinct sizes, ignoring sizes that only ever carry fragments.
    sizes = sorted({sz for sz, _, _, tx in spans if len(tx) >= 4}, reverse=True)

    for size in sizes[:4]:
        run = [s for s in spans if abs(s[0] - size) < 0.6]
        run.sort(key=lambda s: (s[1], s[2]))     # reading order: top, then left

        parts: list[str] = []
        prev = None
        for _, _, _, txt in run:
            key = _title_key(txt)
            if key and key == prev:              # outline + fill pass
                continue
            parts.append(txt)
            prev = key

        title = _DASH_RUN.sub("–", " ".join(parts))
        title = re.sub(r"\s+", " ", title).strip()

        if len(title) < 8:
            continue
        if _ADS_LINE.search(title):              # advert block set in big type
            continue
        if _AUTHOR.search(title):                # byline set at title size
            continue
        if len(title) > MAX_TITLE_CHARS:
            title = title[:MAX_TITLE_CHARS].rsplit(" ", 1)[0]
        return title

    return None


def _extract_title_and_author(text: str) -> tuple[str, str | None]:
    """Find the article title and byline.

    RRiF's layout, confirmed against the December 2024 issue:

        masthead / page number
        SECTION BANNER (all caps)
        byline, with a qualification abbreviation
        [optional abstract, optional UDK line]
        title line          ← each line appears TWICE
        title line
        1.  SECTION HEADING
        body...

    The duplication is a rendering artefact — the headline is drawn as outline
    plus fill and the text layer captures both passes — and it is the most
    reliable marker on the page, because nothing else repeats itself
    line-for-line. Taking the run of duplicated lines also recovers multi-line
    titles intact, which a single-line heuristic cannot.

    v2 instead joined the first four lines that survived a skip filter, and on
    a two-column page those are body text. That is why 96.6% of `source` values
    in the v2 corpus were paragraphs rather than citations.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()][:TITLE_SEARCH_LINES]

    author: str | None = None
    author_idx: int | None = None
    for i, line in enumerate(lines):
        if _AUTHOR.search(line) and len(line) < 100:
            author, author_idx = line, i
            break

    # The title follows the byline. Where there is no byline, scan the lot.
    start = author_idx + 1 if author_idx is not None else 0

    parts: list[str] = []
    i = start
    while i < len(lines) - 1:
        key = _title_key(lines[i])
        if (key and len(key) >= MIN_DUP_TITLE_CHARS
                and key == _title_key(lines[i + 1])
                and not _SECTION_START.match(lines[i])):
            j = i
            while (j < len(lines) - 1
                   and _title_key(lines[j])
                   and _title_key(lines[j]) == _title_key(lines[j + 1])
                   and not _SECTION_START.match(lines[j])):
                parts.append(lines[j])
                j += 2
            break
        i += 1

    if parts:
        title = _DASH_RUN.sub("–", " ".join(parts))
        title = re.sub(r"\s+", " ", title).strip()
    else:
        # No duplicated run — an older issue or a different template. Fall back
        # to the first line after the byline that is not furniture, capped
        # hard: a short wrong label is visibly wrong, a paragraph is not.
        tail = lines[start:] if author_idx is not None else lines
        cand = [
            l for l in tail
            if 8 < len(l) <= MAX_TITLE_CHARS
            and not _SKIP_LINE.match(l)
            and not _SECTION_START.match(l)
            and any(c.islower() for c in l)
            and not l.endswith(('.', ','))
        ]
        title = cand[0] if cand else "Nepoznat naslov"

    if len(title) > MAX_TITLE_CHARS:
        title = title[:MAX_TITLE_CHARS].rsplit(" ", 1)[0]
    if len(title) < 4:
        title = "Nepoznat naslov"

    return title, author


# ── Repeated headline lines in the body ───────────────────────────────────────

MIN_DEDUP_CHARS = 8
# A line must contain a real word before it can be dropped as a duplicate.
# Numeric table rows repeat legitimately — two "0,00" cells under each other
# are data, not a rendering artefact.
_HAS_WORD = re.compile(r'[A-Za-zČĆŠŽĐČĆŠŽĐčćšžđ]{3,}')


def collapse_repeated_lines(text: str) -> str:
    """Drop a line that merely repeats the one before it.

    Same rendering artefact as above: without this the headline appears twice
    inside the first chunk of every article, and section headings do too. Only
    applied to lines over MIN_DEDUP_CHARS, so a table with two identical short
    rows — "0,00" under "0,00" — keeps both.

    Must run AFTER title extraction, which depends on the duplication.
    """
    out: list[str] = []
    prev_key = None
    for line in text.split("\n"):
        key = _title_key(line)
        if (key and len(key) >= MIN_DEDUP_CHARS and key == prev_key
                and _HAS_WORD.search(line)):
            continue
        out.append(line)
        prev_key = key
    return "\n".join(out)


# ── Adverts ───────────────────────────────────────────────────────────────────

_ADS_MARKERS = {
    "phone":   re.compile(r'\b(tel|mob|fax)\.', re.I),
    "email":   re.compile(r'\S+@\S+\.\w+'),
    "web":     re.compile(r'\bwww\.\S+', re.I),
    "company": re.compile(r'\b(d\.o\.o\.|d\.d\.|obrt)\b', re.I),
    "price":   re.compile(r'\bcijena\b|\bnaruč', re.I),
}
ADS_MARKERS_REQUIRED = 3          # distinct kinds, not raw hits
ADS_DENSITY_PER_1K = 3.0          # hits per 1,000 characters


def _is_ads_chunk(text: str) -> bool:
    """True only when the chunk looks like a page of adverts.

    v2 counted raw hits over a pattern that included `d.o.o.` and `@` and
    tripped at four. Accounting articles say "d.o.o." constantly, so real
    content was being discarded silently and in unknown quantity. Requiring
    several *distinct* marker kinds and a density threshold means a long
    article about companies no longer qualifies, while a contact-details page
    still does.
    """
    if not text:
        return False
    kinds = 0
    hits = 0
    for pat in _ADS_MARKERS.values():
        found = len(pat.findall(text))
        if found:
            kinds += 1
            hits += found
    density = hits / (len(text) / 1000)
    return kinds >= ADS_MARKERS_REQUIRED and density >= ADS_DENSITY_PER_1K


# ── Sections ──────────────────────────────────────────────────────────────────

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


# ── Public API ────────────────────────────────────────────────────────────────

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

    text = clean_text(_extract_text(pdf_path))
    if len(text) < MIN_CHARS:
        if verbose:
            print(f"  [skip] {pdf_path.name}: {len(text)} chars < {MIN_CHARS} (ads/short)")
        return None

    category      = _detect_category(text, pub_type)
    title, author = _extract_title_and_author(text)
    layout_title  = _title_from_layout(pdf_path)
    if layout_title:
        title = layout_title
    # Only now — the title detector reads the duplication this removes.
    sections      = _split_into_sections(collapse_repeated_lines(text))
    base_citation = f"{pub_label} br. {month}/{year} — {title}"

    chunks: list[ArticleChunk] = []
    idx = 0
    ads_skipped = 0

    for sec_num, sec_title, body in sections:
        for piece in split_body(body):
            if _is_ads_chunk(piece):
                ads_skipped += 1
                continue
            tok = n_tokens(piece)
            if tok < MIN_CHUNK_TOKENS:
                continue
            sec_label = f" § {sec_num}. {sec_title}" if sec_num else ""
            chunks.append(ArticleChunk(
                pub_type=pub_type, pub_label=pub_label,
                year=year, month=month, article_num=article_num,
                pub_date=pub_date, title=title, author=author,
                section_num=sec_num, section_title=sec_title,
                chunk_index=idx, text=piece,
                char_count=len(piece), token_count=tok,
                source_citation=f"{base_citation}{sec_label}",
                default_category=category,
            ))
            idx += 1

    if verbose and ads_skipped:
        print(f"  [info] {pdf_path.name}: {ads_skipped} ads chunk(s) filtered")

    return chunks


if __name__ == "__main__":
    # Eyeball one article before committing to a three-hour run:
    #   python -m rag.ingest.article_loader data/raw/Arhiva/RRIF2412/R241201.PDF
    import sys
    for arg in sys.argv[1:]:
        result = load_article(arg)
        if not result:
            continue
        first = result[0]
        toks = [c.token_count for c in result]
        print(f"\n{arg}")
        print(f"  title    : {first.title!r}")
        print(f"  author   : {first.author!r}")
        print(f"  category : {first.default_category}")
        print(f"  citation : {first.source_citation[:100]!r}")
        print(f"  chunks   : {len(result)}  tokens min/median/max: "
              f"{min(toks)}/{sorted(toks)[len(toks)//2]}/{max(toks)}")
        print(f"  over 508 : {sum(1 for t in toks if t > 508)}")
