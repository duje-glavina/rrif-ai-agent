"""Build an .xlsx of generated answers for a RRiF advisor to grade.

WHY
───
`grade_answers.py` measures whether an answer is faithful to the chunks it was
built from. It cannot measure whether the answer is CORRECT about Croatian tax
law. Neither can the keyword metric in `run_eval.py`: we wrote the questions
AND the marking scheme, so a high score there means "the system said the words
we decided counted", not "an advisor would send this to a client".

Only RRiF can produce the second number. `kriteriji_prihvacanja_faza1.docx`
says exactly how: a four-point scale, two independent advisors per question, a
third senior advisor resolving disagreements, and a separate "would you send
this to a client?" field. This sheet is built to that spec, deliberately, so
grading it is a DRY RUN OF THE ACCEPTANCE MEASUREMENT ITSELF — on our golden
set, before the pilot, while there is still time to discover that two advisors
disagree about what "točno" means. Section 4 of that document tracks the
disagreement rate precisely because it expects this to be a problem.

So the ask at the meeting is not "please grade some answers". It is: let us run
your acceptance procedure once, now, on 31 questions, so that when it runs for
real on 100 pilot questions nothing about it is a surprise.

INDEPENDENCE
────────────
The document requires two advisors to grade each question INDEPENDENTLY. Build
one sheet per advisor with --grader and send them separately. One shared file
that both people type into is not the same measurement.

USAGE
─────
    python -m eval.make_review_sheet eval/results/<run>.json --grader "Savjetnik A"
    python -m eval.make_review_sheet eval/results/<run>.json --grader "Savjetnik B"
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

# Verbatim from kriteriji_prihvacanja_faza1.docx, section 2. Do not "improve"
# the wording: the whole point is that this sheet and the acceptance
# measurement use the same four labels, so the dry run transfers.
GRADES = ["Točno", "Djelomično točno", "Netočno", "Nije moguće ocijeniti"]

# Section 2, the parallel business-usability field.
SEND = ["Da", "Uz doradu", "Ne"]

HEADERS = [
    ("id", 12),
    ("Pitanje", 40),
    ("Odgovor sustava", 72),
    ("Izvori koje sustav navodi", 28),
    ("OCJENA", 20),
    ("Poslali biste klijentu?", 16),
    ("Što nedostaje ili je krivo", 38),
    ("Ispravan izvor (ako znate)", 26),
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
    None for magazine chunks, so the citation objects can carry an empty
    source string even when the answer names the article in prose. Falling
    back to the retrieved chunks' own `source` keeps the grader able to check
    provenance; the label says which one they are looking at, so nobody
    mistakes the fallback for a real citation.
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


def build_instructions(wb: Workbook, run_name: str, n: int, grader: str) -> None:
    ws = wb.create_sheet("Upute", 0)
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 96

    lines = [
        ("RRiF AI — probno ocjenjivanje", ""),
        ("", ""),
        ("Ocjenjivač", grader or "(upišite ime)"),
        ("Broj pitanja", str(n)),
        ("Procjena vremena", "oko sat vremena"),
        ("", ""),
        ("Zašto ovo radimo",
         "Skala, postupak i polje «poslali biste klijentu» preuzeti su doslovno iz "
         "dokumenta «Prijedlog kriterija prihvaćanja Faze 1». Ovo je probna provedba "
         "tog istog postupka — na 31 pitanje, prije pilota — da se eventualna "
         "neslaganja oko toga što znači «točno» pojave sada, a ne na kraju Faze 1."),
        ("Važno",
         "Ocjenjujte neovisno. Ovaj primjerak je Vaš; drugi savjetnik ocjenjuje isti "
         "skup u zasebnoj datoteci. Neslaganja su korisna informacija, ne problem."),
        ("", ""),
        ("Skala (iz kriterija)", ""),
        ("Točno",
         "Odgovor je činjenično ispravan, citira ispravan izvor, i mogli biste ga bez "
         "izmjene koristiti u radu s klijentom."),
        ("Djelomično točno",
         "Suština je ispravna, ali nedostaje članak ili dio citata, formulacija je "
         "nejasna, ili nedostaje nijansa koju biste dodali. Upotrebljivo uz manju doradu."),
        ("Netočno",
         "Odgovor je činjenično pogrešan, citira pogrešan izvor, izmišlja sadržaj koji "
         "ne postoji, ili biste ga morali odbaciti i napisati ispočetka."),
        ("Nije moguće ocijeniti",
         "Pitanje izlazi izvan opsega Faze 1 — tema nije pokrivena bazom znanja ili "
         "traženi propis nije u korpusu. Ne ulazi u izračun točnosti."),
        ("", ""),
        ("Poslali biste klijentu?",
         "Da / Uz doradu / Ne. Prati se odvojeno od ocjene, kao pokazatelj stvarne "
         "poslovne upotrebljivosti."),
        ("", ""),
        ("Izvori",
         "Stupac «Izvori» pokazuje na što se sustav pozvao. Ako je odgovor točan ali je "
         "izvor pogrešan, to je «Djelomično točno» — i upišite ispravan izvor."),
        ("Izvatci",
         "List «Izvatci» sadrži tekst koji je sustav pročitao prije odgovora. Koristan je "
         "kad želite vidjeti odakle mu je nešto došlo, ali nemojte po njemu ocjenjivati: "
         "ocjenjujete odgovor, ne izvadak."),
        ("", ""),
        ("Opseg",
         "Sustav je u Fazi 1 ograničen na članke iz RRiF-a i PiP-a; zakonski tekstovi "
         "dolaze u Fazi 2. Ako odgovor rješava pitanje iz članka umjesto iz zakona, to "
         "nije greška — ali nam je vrlo korisno znati je li Vam takav odgovor dovoljan."),
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
        ("OCJENA", "Djelomično točno"),
        ("Poslali biste klijentu?", "Uz doradu"),
        ("Što nedostaje ili je krivo",
         "Stopa je točna, ali ne spominje da se od 1.1.2024. primjenjuje i na "
         "isporuku i ugradnju."),
        ("Ispravan izvor", "RRiF br. 1/2024, str. 41"),
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

    # showDropDown=False shows the arrow. openpyxl passes this attribute
    # straight through to the XML, where it means "suppress the dropdown" —
    # so the sensible-looking True is the one that hides it.
    dv_grade = DataValidation(
        type="list", formula1='"' + ",".join(GRADES) + '"',
        allow_blank=True, showDropDown=False)
    dv_grade.errorTitle = "Neispravna ocjena"
    dv_grade.error = "Odaberite jednu od četiri ocjene iz kriterija prihvaćanja."
    ws.add_data_validation(dv_grade)
    dv_grade.add(f"E2:E{last}")

    dv_send = DataValidation(
        type="list", formula1='"' + ",".join(SEND) + '"',
        allow_blank=True, showDropDown=False)
    dv_send.errorTitle = "Neispravna vrijednost"
    dv_send.error = "Da, Uz doradu, ili Ne."
    ws.add_data_validation(dv_send)
    dv_send.add(f"F2:F{last}")

    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:H{last}"


def build_excerpts(wb: Workbook, rows: list[dict]) -> None:
    ws = wb.create_sheet("Izvatci", 2)
    headers = [("id", 12), ("Pitanje", 34), ("#", 5),
               ("Izvor", 46), ("Tekst koji je sustav pročitao", 100)]
    _style_header(ws, headers)

    r = 2
    for row in rows:
        for n, m in enumerate(row.get("retrieved_meta") or [], 1):
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
    ap.add_argument("--grader", default="",
                    help="Name of the advisor this copy is for — build one per advisor")
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
    build_instructions(wb, src.name, len(rows), args.grader)
    build_grading(wb, rows)
    build_excerpts(wb, rows)

    if args.out:
        out = Path(args.out)
    else:
        suffix = f".pregled_{args.grader.replace(' ', '_')}.xlsx" if args.grader \
            else ".pregled.xlsx"
        out = src.with_suffix(suffix)
    wb.save(out)
    print(f"  → {out}   ({len(rows)} pitanja"
          f"{', ocjenjivač: ' + args.grader if args.grader else ''})")
    if not args.grader:
        print("\n  Bez --grader gradite jedan zajednički primjerak. Kriteriji traže\n"
              "  DVA neovisna ocjenjivača — napravite dvije datoteke i pošaljite ih\n"
              "  odvojeno, inače mjerite nešto drugo.")
    print("\n  Prije slanja: otvorite ga i pročitajte 3 odgovora. Ako Vas neki\n"
          "  posrami, popravite to prije nego ga vide, ne poslije.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
