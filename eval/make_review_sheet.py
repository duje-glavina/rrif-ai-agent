"""Build an .xlsx of generated answers for a human expert to grade.

WHY
───
`grade_answers.py` measures whether an answer is faithful to the chunks it was
built from. It cannot measure whether the answer is CORRECT about Croatian tax
law, because that needs someone who knows Croatian tax law. That is the
acceptance criterion, and it is the one thing we cannot produce ourselves.

RRiF can. And grading 31 answers is a far smaller ask than writing 200 new
questions — it takes an advisor perhaps an hour, it needs no format spec, and
it produces the acceptance number directly instead of an input to it. It also
surfaces disagreements about what "correct" means while there is still time to
act on them, rather than at handover.

So: hand them this sheet, ask for the column marked `OCJENA`, and treat the
result as the F1 accuracy figure.

USAGE
─────
    python -m eval.make_review_sheet eval/results/<run>.json
    python -m eval.make_review_sheet eval/results/<run>.json -o za_RRiF.xlsx
    python -m eval.make_review_sheet eval/results/<run>.json --include-zakoni

Needs openpyxl:  pip install openpyxl
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

try:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation
except ImportError:
    print("openpyxl nije instaliran.  pip install openpyxl")
    raise SystemExit(1)


FONT = "Arial"
EXCEL_CELL_LIMIT = 32_000        # real limit is 32,767; leave room for the ellipsis

GRADES = [
    "1 - točno",
    "2 - djelomično",
    "3 - netočno",
    "4 - ne mogu ocijeniti",
]

HEADERS = [
    ("id", 12),
    ("Pitanje", 42),
    ("Odgovor sustava", 78),
    ("Izvori koje sustav navodi", 30),
    ("OCJENA", 18),
    ("Što nedostaje ili je krivo", 40),
    ("Ispravan izvor (ako znate)", 28),
    ("Ocjenjivač", 14),
]

HDR_FILL = PatternFill("solid", fgColor="1F3864")
ASK_FILL = PatternFill("solid", fgColor="FFF2CC")     # columns we want filled in
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def _clip(s: str, n: int = EXCEL_CELL_LIMIT) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + " […skraćeno…]"


def _sources_for(row: dict) -> str:
    """What the answer points at — with a fallback that matters.

    `Citation.source` is populated from the answerer's `law_name`, which is
    None for magazine chunks. So for most of the corpus the citation objects
    carry an empty source string even when the answer names the article in
    prose. Falling back to the retrieved chunks' own `source` keeps the
    grader able to check provenance; the label says which one they are
    looking at, so nobody mistakes the fallback for a real citation.
    """
    cites = [c.get("source", "").strip()
             for c in (row.get("citations_raw") or [])]
    cites = [c for c in cites if c]
    if cites:
        return "\n".join(dict.fromkeys(cites))

    seen: list[str] = []
    for m in (row.get("retrieved_meta") or [])[:3]:
        s = (m.get("source") or "").strip()
        if s and s not in seen:
            seen.append(s)
    if not seen:
        return "(sustav nije naveo izvor)"
    return "(iz dohvata, ne iz citata):\n" + "\n".join(seen)


def _style_header(ws, headers) -> None:
    for col, (label, width) in enumerate(headers, 1):
        c = ws.cell(row=1, column=col, value=label)
        c.font = Font(name=FONT, bold=True, color="FFFFFF", size=11)
        c.fill = HDR_FILL
        c.alignment = Alignment(vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.row_dimensions[1].height = 30


def build_instructions(wb: Workbook, run_name: str, n: int) -> None:
    ws = wb.create_sheet("Upute", 0)
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 96

    lines = [
        ("RRiF AI — provjera odgovora", ""),
        ("", ""),
        ("Što tražimo",
         "Za svaki odgovor u listu «Ocjena» popunite žuto obojene stupce. "
         "Najvažniji je stupac OCJENA."),
        ("Koliko traje", f"{n} pitanja, otprilike sat vremena."),
        ("", ""),
        ("Ljestvica", ""),
        ("1 - točno",
         "Odgovor je stručno ispravan i potpun. Savjetnik bi ga mogao poslati pretplatniku "
         "uz manje jezične dorade."),
        ("2 - djelomično",
         "Ništa u odgovoru nije pogrešno, ali nešto bitno nedostaje, ili je preopćenito "
         "da bi bilo korisno."),
        ("3 - netočno",
         "Odgovor sadrži tvrdnju koja nije točna, ili se poziva na propis koji ne kaže to. "
         "Ovo je jedina ocjena koja nas stvarno zabrinjava — molimo upišite što je krivo."),
        ("4 - ne mogu ocijeniti",
         "Pitanje je loše postavljeno, dvosmisleno, ili izvan Vašeg područja. "
         "Nije greška sustava."),
        ("", ""),
        ("Izvori",
         "Stupac «Izvori» pokazuje na što se sustav pozvao. Ako je odgovor točan ali je "
         "izvor pogrešan, to je ocjena 2, a ne 1 — i upišite ispravan izvor."),
        ("Izvatci",
         "List «Izvatci» sadrži tekst koji je sustav pročitao prije nego je odgovorio. "
         "Koristan je kad želite vidjeti odakle mu nešto — ali nemojte po njemu ocjenjivati. "
         "Ocjenjujete odgovor, ne izvadak."),
        ("", ""),
        ("Napomena",
         "Sustav je u ovoj fazi ograničen na članke iz RRiF-a i PiP-a. Zakonski tekstovi "
         "dolaze u sljedećoj fazi, pa odgovore koji bi trebali citirati zakon ocijenite "
         "prema onome što članak kaže."),
        ("", ""),
        ("Izvor podataka", run_name),
        ("Datum", date.today().isoformat()),
    ]
    for i, (a, b) in enumerate(lines, 1):
        ca = ws.cell(row=i, column=1, value=a)
        cb = ws.cell(row=i, column=2, value=b)
        ca.font = Font(name=FONT, bold=True, size=11)
        cb.font = Font(name=FONT, size=11)
        cb.alignment = Alignment(wrap_text=True, vertical="top")
    ws["A1"].font = Font(name=FONT, bold=True, size=14)

    # Worked example, so the expected format is unambiguous.
    r = len(lines) + 2
    ws.cell(row=r, column=1, value="Primjer ispunjenog retka").font = Font(
        name=FONT, bold=True, size=11)
    example = [
        ("OCJENA", "2 - djelomično"),
        ("Što nedostaje ili je krivo",
         "Stopa je točna, ali ne spominje da se od 1.1.2024. primjenjuje i na "
         "isporuku i ugradnju."),
        ("Ispravan izvor", "RRiF br. 1/2024, str. 41"),
        ("Ocjenjivač", "M. K."),
    ]
    for j, (a, b) in enumerate(example, r + 1):
        ws.cell(row=j, column=1, value=a).font = Font(name=FONT, italic=True, size=10)
        c = ws.cell(row=j, column=2, value=b)
        c.font = Font(name=FONT, italic=True, size=10)
        c.fill = ASK_FILL
        c.alignment = Alignment(wrap_text=True, vertical="top")


def build_grading(wb: Workbook, rows: list[dict]) -> None:
    ws = wb.create_sheet("Ocjena", 1)
    _style_header(ws, HEADERS)

    for i, row in enumerate(rows, start=2):
        answer = row.get("answer") or row.get("answer_preview") or ""
        if row.get("refused"):
            answer = "[SUSTAV JE ODBIO ODGOVORITI]\n\n" + answer
        values = [
            row.get("id", ""),
            row.get("query", ""),
            _clip(answer) or "(prazno)",
            _clip(_sources_for(row), 2000),
            "", "", "", "",
        ]
        for col, v in enumerate(values, 1):
            c = ws.cell(row=i, column=col, value=v)
            c.font = Font(name=FONT, size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.border = BORDER
            if col >= 5:
                c.fill = ASK_FILL
        ws.row_dimensions[i].height = 96

    last = len(rows) + 1
    dv = DataValidation(
        type="list",
        formula1='"' + ",".join(GRADES) + '"',
        allow_blank=True,
        showDropDown=False,          # False = show the dropdown arrow (openpyxl inverts this)
    )
    dv.error = "Odaberite jednu od ponuđenih ocjena."
    dv.errorTitle = "Neispravna ocjena"
    ws.add_data_validation(dv)
    dv.add(f"E2:E{last}")

    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:H{last}"


def build_excerpts(wb: Workbook, rows: list[dict]) -> None:
    ws = wb.create_sheet("Izvatci", 2)
    headers = [("id", 12), ("Pitanje", 34), ("#", 5),
               ("Izvor", 46), ("Tekst koji je sustav pročitao", 100)]
    _style_header(ws, headers)

    r = 2
    for row in rows:
        meta = row.get("retrieved_meta") or []
        for n, m in enumerate(meta, 1):
            vals = [
                row.get("id", "") if n == 1 else "",
                row.get("query", "") if n == 1 else "",
                n,
                (m.get("source") or "").strip() or "(bez oznake)",
                _clip((m.get("chunk_text") or "").strip(), 8000),
            ]
            for col, v in enumerate(vals, 1):
                c = ws.cell(row=r, column=col, value=v)
                c.font = Font(name=FONT, size=9)
                c.alignment = Alignment(wrap_text=True, vertical="top")
                c.border = BORDER
            ws.row_dimensions[r].height = 60
            r += 1

    ws.freeze_panes = "C2"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="eval/results/<run>.json from a generation run")
    ap.add_argument("-o", "--out", default="", help="Output .xlsx (default: alongside input)")
    ap.add_argument("--include-zakoni", action="store_true",
                    help="Include the PDV-law questions (out of F1 scope by default)")
    ap.add_argument("--include-traps", action="store_true",
                    help="Include out-of-corpus traps, to check that refusals read well")
    args = ap.parse_args()

    src = Path(args.results)
    if not src.exists():
        print(f"Not found: {src}")
        return 1

    data = json.loads(src.read_text(encoding="utf-8"))
    if data.get("skip_generation"):
        print("This run was --skip-generation: no answers to review.")
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
        print("No answer text in this results file. It predates the `answer` field —\n"
              "re-run the eval with generation on, then build the sheet from that run.")
        return 1

    wb = Workbook()
    wb.remove(wb.active)                      # drop the default empty sheet
    build_instructions(wb, src.name, len(rows))
    build_grading(wb, rows)
    build_excerpts(wb, rows)

    out = Path(args.out) if args.out else src.with_suffix(".pregled.xlsx")
    wb.save(out)
    print(f"  → {out}   ({len(rows)} pitanja)")
    print("\n  Prije slanja: otvorite ga i pročitajte 3 odgovora. Ako Vas neki\n"
          "  posrami, popravite to prije nego ga vide, ne poslije.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
