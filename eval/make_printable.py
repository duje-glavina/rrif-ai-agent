"""Render the generated answers as a print-ready page.

WHY HTML AND NOT A PDF
──────────────────────
Reading 31 answers on screen and reading them on paper are different acts —
on paper you notice register, repetition and hedging that slide past on a
monitor. This produces one self-contained HTML file with print CSS: open it,
Ctrl+P, "Save as PDF" or straight to the printer. No reportlab, no weasyprint,
nothing to install on a machine that already has enough moving parts.

Set to serif at 10.5pt with generous margins, because this is for reading, not
for grading — the grading sheet is `make_review_sheet.py` and it is a
spreadsheet on purpose. Here there is a ruled space after each answer for
whatever you scribble.

USAGE
─────
    python -m eval.make_printable eval/results/<run>.json
    python -m eval.make_printable eval/results/<run>.json --include-traps
    python -m eval.make_printable eval/results/<run>.json -o odgovori.html

Then open the file and print it.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import date
from pathlib import Path


# ── Markdown → HTML ──────────────────────────────────────────────────────────
# Small on purpose. The generator emits bold, headings, and both kinds of list;
# anything more exotic is not worth the parser. Everything is HTML-escaped
# first, so a stray '<' in an answer cannot become markup.

_BOLD = re.compile(r"(?<!\w)\*\*(?!\s)(.+?)(?<!\s)\*\*(?![*\w])", re.S)
_ITALIC = re.compile(r"(?<![*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![*\w])")
_CODE = re.compile(r"`([^`\n]+?)`")
_HEAD = re.compile(r"^#{1,6}\s*(.+)$")
_ULI = re.compile(r"^\s*[-–•]\s+(.+)$")
_OLI = re.compile(r"^\s*(\d+)[.)]\s+(.+)$")
_RULE = re.compile(r"^\s*([-*_])\s*\1\s*\1[\s\-*_]*$")


def _inline(s: str) -> str:
    s = html.escape(s)
    s = _BOLD.sub(r"<strong>\1</strong>", s)
    s = _ITALIC.sub(r"<em>\1</em>", s)
    s = _CODE.sub(r"<code>\1</code>", s)
    return s


def md(text: str) -> str:
    """Block-level render. Returns HTML."""
    out: list[str] = []
    mode: str | None = None          # 'ul' | 'ol' | None

    def close():
        nonlocal mode
        if mode:
            out.append(f"</{mode}>")
            mode = None

    for raw in (text or "").split("\n"):
        line = raw.rstrip()
        if not line.strip():
            close()
            continue
        if _RULE.match(line):
            close()
            out.append("<hr>")
            continue
        m = _HEAD.match(line)
        if m:
            close()
            out.append(f"<h3>{_inline(m.group(1))}</h3>")
            continue
        m = _ULI.match(line)
        if m:
            if mode != "ul":
                close(); out.append("<ul>"); mode = "ul"
            out.append(f"<li>{_inline(m.group(1))}</li>")
            continue
        m = _OLI.match(line)
        if m:
            if mode != "ol":
                close(); out.append("<ol>"); mode = "ol"
            out.append(f"<li>{_inline(m.group(2))}</li>")
            continue
        close()
        out.append(f"<p>{_inline(line)}</p>")
    close()
    return "\n".join(out)


def sources_for(row: dict) -> list[str]:
    cites = [c.get("source", "").strip() for c in (row.get("citations_raw") or [])]
    cites = [c for c in cites if c]
    if cites:
        return list(dict.fromkeys(cites))
    seen: list[str] = []
    for m in (row.get("retrieved_meta") or [])[:3]:
        s = (m.get("source") or "").strip()
        if s and s not in seen:
            seen.append(s)
    return seen


CSS = """
@page { size: A4; margin: 18mm 17mm 16mm 17mm; }
* { box-sizing: border-box; }
body {
  font: 10.5pt/1.55 Georgia, "Times New Roman", serif;
  color: #111; margin: 0; padding: 0 0 8mm 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
header.doc { border-bottom: 1.5pt solid #333; padding-bottom: 3mm; margin-bottom: 7mm; }
header.doc h1 { font-size: 15pt; margin: 0 0 1.5mm; letter-spacing: -0.01em; }
header.doc .sub { font-size: 8.5pt; color: #555; font-family: Arial, sans-serif; }
header.doc .note {
  margin-top: 4mm; font-size: 9pt; color: #333;
  background: #f4f4f0; border-left: 2.5pt solid #999; padding: 2.5mm 3.5mm;
}
.q { margin-bottom: 9mm; }
.q h2 {
  font-size: 11.5pt; margin: 0 0 2.5mm; line-height: 1.35;
  break-after: avoid; page-break-after: avoid;
}
.q h2 .id {
  font-family: Arial, sans-serif; font-size: 8pt; font-weight: normal;
  color: #666; display: block; margin-bottom: 0.8mm; letter-spacing: 0.04em;
}
.answer p { margin: 0 0 2.2mm; }
.answer h3 {
  font-family: Arial, sans-serif; font-size: 9.5pt; margin: 3.5mm 0 1.5mm;
  break-after: avoid; page-break-after: avoid;
}
.answer ul, .answer ol { margin: 0 0 2.2mm; padding-left: 6mm; }
.answer li { margin-bottom: 1mm; }
.answer code { font-family: "Courier New", monospace; font-size: 9.5pt; }
.answer hr { border: 0; border-top: 0.5pt solid #ccc; margin: 3mm 0; }
.refused { font-family: Arial, sans-serif; font-size: 8.5pt; color: #8c3a3a; margin-bottom: 1.5mm; }
.src {
  font-family: Arial, sans-serif; font-size: 8pt; color: #555;
  margin-top: 2.5mm; padding-top: 1.5mm; border-top: 0.5pt solid #ddd;
}
.src span { display: block; }
.notes { margin-top: 3mm; }
.notes .lbl { font-family: Arial, sans-serif; font-size: 7.5pt; color: #999; letter-spacing: 0.06em; }
.notes .line { border-bottom: 0.4pt solid #d8d8d8; height: 6.5mm; }
.sep { border: 0; border-top: 0.5pt solid #bbb; margin: 0 0 7mm; }
@media screen {
  body { max-width: 185mm; margin: 12mm auto; padding: 0 6mm; }
}
"""


def build(rows: list[dict], run_name: str, notes: bool) -> str:
    parts = [
        "<!doctype html><html lang=hr><head><meta charset=utf-8>",
        f"<title>RRiF AI — odgovori ({run_name})</title>",
        f"<style>{CSS}</style></head><body>",
        "<header class=doc><h1>RRiF AI Agent — generirani odgovori</h1>",
        f"<div class=sub>{len(rows)} pitanja · {run_name} · {date.today().strftime('%d.%m.%Y.')}</div>",
        "<div class=note>Čitaj kao urednik, ne kao ocjenjivač. Traži: tvrdnju koja zvuči "
        "preuvjerljivo, svotu bez godine, rečenicu koju ne bi potpisao, i ton koji ne "
        "zvuči kao RRiF. Formalno ocjenjivanje ide u tablicu za savjetnike.</div>",
        "</header>",
    ]

    for row in rows:
        answer = row.get("answer") or row.get("answer_preview") or ""
        parts.append("<section class=q>")
        parts.append(
            f"<h2><span class=id>{html.escape(str(row.get('id','')))}</span>"
            f"{html.escape(row.get('query',''))}</h2>"
        )
        if row.get("refused"):
            parts.append("<div class=refused>SUSTAV JE ODBIO ODGOVORITI</div>")
        parts.append(f"<div class=answer>{md(answer) or '<p>(prazno)</p>'}</div>")

        src = sources_for(row)
        if src:
            label = "Izvori:" if (row.get("citations_raw") or []) and any(
                c.get("source") for c in row["citations_raw"]) else "Dohvaćeno iz:"
            parts.append("<div class=src><span><strong>" + label + "</strong></span>"
                         + "".join(f"<span>{html.escape(s)}</span>" for s in src)
                         + "</div>")
        if notes:
            parts.append("<div class=notes><div class=lbl>BILJEŠKE</div>"
                         + "<div class=line></div><div class=line></div></div>")
        parts.append("</section><hr class=sep>")

    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("-o", "--out", default="")
    ap.add_argument("--include-zakoni", action="store_true",
                    help="Include the PDV-law questions (out of F1 scope by default)")
    ap.add_argument("--include-traps", action="store_true",
                    help="Include out-of-corpus traps, to read how the refusals sound")
    ap.add_argument("--no-notes", action="store_true",
                    help="Drop the ruled note lines and fit more per page")
    args = ap.parse_args()

    src = Path(args.results)
    if not src.exists():
        print(f"Not found: {src}")
        return 1

    data = json.loads(src.read_text(encoding="utf-8"))
    if data.get("skip_generation"):
        print("This run was --skip-generation: there are no answers in it.")
        return 1

    rows = data.get("per_question", [])
    if not args.include_traps:
        rows = [r for r in rows if r.get("in_corpus")]
    if not args.include_zakoni:
        rows = [r for r in rows if r.get("scope") != "zakon"]
    if not rows:
        print("No questions selected.")
        return 1
    if not any((r.get("answer") or "").strip() for r in rows):
        print("No answer text in this results file — re-run the eval with generation on.")
        return 1

    out = Path(args.out) if args.out else src.with_suffix(".za_citanje.html")
    out.write_text(build(rows, src.stem, not args.no_notes), encoding="utf-8")
    print(f"  → {out}   ({len(rows)} pitanja)")
    print("\n  Otvori u pregledniku pa Ctrl+P → Spremi kao PDF ili ispiši.")
    print("  U dijalogu za ispis isključi zaglavlja i podnožja preglednika.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
