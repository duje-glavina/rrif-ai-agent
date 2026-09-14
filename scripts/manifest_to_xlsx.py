"""Turn manifest.csv into a workbook you can read across a table.

WHY
───
`build_manifest.py` prints a report. A report tells you 2016-08 is missing; it
does not let you run your eye down a column and see that every January from
2020 is thin, or hand the thing to RRiF and have them write answers next to the
questions.

So: same data, five sheets, formulas rather than baked-in numbers — repoint the
`Datoteke` sheet at a newer manifest and every count follows.

    python scripts/manifest_to_xlsx.py data/raw/RRIF_arhiva/manifest.csv
    python scripts/manifest_to_xlsx.py data\\manifest_laptop.csv -o pokrivenost.xlsx

Run it against the manifest built on a machine that has PyMuPDF if you want the
page counts filled in; without it that column is simply empty.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

FONT = "Arial"
NAVY = "1F3864"
H_FILL = PatternFill("solid", fgColor=NAVY)
SUB_FILL = PatternFill("solid", fgColor="D9E2F3")
MISS_FILL = PatternFill("solid", fgColor="F8CBAD")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

PUB_ORDER = ["RRIF", "PROR", "OBAV", "PRGO", "NEPR", "OBRT", "PRPI"]
MONTHLY = {"RRIF", "PROR"}          # deserve a 12-column grid; the rest are annual


def style_header(ws, row: int, ncols: int, start: int = 1) -> None:
    for c in range(start, start + ncols):
        cell = ws.cell(row=row, column=c)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF", size=10)
        cell.fill = H_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BOX


def sheet_datoteke(wb: Workbook, rows: list[dict], headers: list[str]):
    """The manifest itself. Every other sheet counts against this one."""
    ws = wb.create_sheet("Datoteke")
    ws.append(headers)
    style_header(ws, 1, len(headers))
    bools = {"folder_ok", "file_ok", "issue_mismatch", "nonstandard_case"}
    for r in rows:
        out = []
        for h in headers:
            v = r.get(h, "")
            if h in bools:
                v = "DA" if str(v).lower() == "true" else "NE"
            elif h in ("year", "month", "bytes", "pages") and str(v).strip():
                try:
                    v = int(v)
                except ValueError:
                    pass
            out.append(v)
        ws.append(out)
    for i, h in enumerate(headers, 1):
        ws.column_dimensions[get_column_letter(i)].width = {
            "rel_path": 52, "file": 22, "unit_ref": 14, "pub_label": 24,
            "supplement": 24, "md5": 10, "mtime": 11,
        }.get(h, 11)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name=FONT, size=9)
    return ws


def _countifs(n: int, pub: str, year: int, month: int | None) -> str:
    """Articles of one publication in one issue — parsed rows only.

    Columns are fixed by the manifest's own order; see COLS below.
    """
    last = n + 1
    base = (f'COUNTIFS(Datoteke!$B$2:$B${last},"{pub}",'
            f'Datoteke!$D$2:$D${last},{year},')
    if month is not None:
        base += f'Datoteke!$E$2:$E${last},{month},'
    return ("=" + base +
            f'Datoteke!$P$2:$P${last},"DA",'
            f'Datoteke!$Q$2:$Q${last},"DA")')


def sheet_pokrivenost(wb: Workbook, rows: list[dict], n: int):
    ws = wb.create_sheet("Pokrivenost", 1)
    years = sorted({int(r["year"]) for r in rows if r.get("year")})
    labels = {r["pub_code"]: r["pub_label"] for r in rows if r.get("pub_code")}
    present = [p for p in PUB_ORDER if p in labels]

    row = 1
    ws.cell(row=row, column=1, value="Pokrivenost arhiva — broj članaka po broju").font = \
        Font(name=FONT, bold=True, size=13, color=NAVY)
    row += 1
    ws.cell(row=row, column=1,
            value="Narančasto = nema nijednog članka. Brojke su formule nad listom "
                  "Datoteke, pa se osvježe ako zamijeniš manifest.").font = \
        Font(name=FONT, size=9, italic=True, color="666666")
    row += 2

    for pub in present:
        ws.cell(row=row, column=1, value=f"{pub} — {labels[pub]}").font = \
            Font(name=FONT, bold=True, size=11, color=NAVY)
        row += 1
        head = row
        ws.cell(row=head, column=1, value="Godina")
        months = list(range(1, 13)) if pub in MONTHLY else [None]
        if pub in MONTHLY:
            for m in months:
                ws.cell(row=head, column=1 + m, value=f"{m:02d}")
            ws.cell(row=head, column=14, value="Ukupno")
            style_header(ws, head, 14)
        else:
            ws.cell(row=head, column=2, value="Članaka")
            ws.cell(row=head, column=3, value="Mjeseci")
            style_header(ws, head, 3)
        row += 1

        by_year_month = defaultdict(set)
        for r in rows:
            if r.get("pub_code") == pub and r.get("folder_ok") == "True" \
                    and r.get("file_ok") == "True":
                by_year_month[int(r["year"])].add(int(r["month"]))

        first = row
        for y in years:
            ws.cell(row=row, column=1, value=y).font = Font(name=FONT, bold=True, size=10)
            ws.cell(row=row, column=1).alignment = Alignment(horizontal="center")
            ws.cell(row=row, column=1).border = BOX
            if pub in MONTHLY:
                for m in range(1, 13):
                    c = ws.cell(row=row, column=1 + m, value=_countifs(n, pub, y, m))
                    c.font = Font(name=FONT, size=10)
                    c.alignment = Alignment(horizontal="center")
                    c.border = BOX
                t = ws.cell(row=row, column=14,
                            value=f"=SUM({get_column_letter(2)}{row}:{get_column_letter(13)}{row})")
                t.font = Font(name=FONT, size=10, bold=True)
                t.alignment = Alignment(horizontal="center")
                t.border = BOX
            else:
                c = ws.cell(row=row, column=2, value=_countifs(n, pub, y, None))
                c.font = Font(name=FONT, size=10)
                c.alignment = Alignment(horizontal="center")
                c.border = BOX
                ms = sorted(by_year_month.get(y, []))
                d = ws.cell(row=row, column=3,
                            value=", ".join(f"{m:02d}" for m in ms) or "—")
                d.font = Font(name=FONT, size=9, color="666666")
                d.alignment = Alignment(horizontal="center")
                d.border = BOX
            row += 1

        rng = (f"B{first}:M{row - 1}" if pub in MONTHLY else f"B{first}:B{row - 1}")
        ws.conditional_formatting.add(
            rng, CellIsRule(operator="equal", formula=["0"], fill=MISS_FILL))
        row += 2

    ws.column_dimensions["A"].width = 9
    for i in range(2, 15):
        ws.column_dimensions[get_column_letter(i)].width = 6.5
    ws.column_dimensions["C"].width = 34      # months list for the annual blocks
    ws.freeze_panes = "B1"
    return ws


def sheet_brojevi(wb: Workbook, rows: list[dict], n: int):
    ws = wb.create_sheet("Brojevi", 2)
    head = ["Izdanje", "Kod", "Godina", "Mjesec", "Direktorij", "Članaka",
            "Stranica", "Prvi rb.", "Zadnji rb.", "Rupe u numeraciji", "Prilog"]
    ws.append(head)
    style_header(ws, 1, len(head))

    issues: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("folder_ok") == "True" and r.get("file_ok") == "True":
            issues[(r["pub_code"], int(r["year"]), int(r["month"]))].append(r)

    last = n + 1
    for (pub, y, m), rs in sorted(issues.items(),
                                  key=lambda kv: (PUB_ORDER.index(kv[0][0])
                                                  if kv[0][0] in PUB_ORDER else 99,
                                                  kv[0][1], kv[0][2])):
        nums = sorted(int(r["article_num"]) for r in rs)
        gaps = [x for x in range(1, max(nums) + 1) if x not in set(nums)]
        pages = [int(r["pages"]) for r in rs if str(r.get("pages", "")).strip().isdigit()]
        ws.append([
            rs[0]["pub_label"], pub, y, m, rs[0]["issue_dir"],
            f'={_countifs(n, pub, y, m)[1:]}',
            (f'=SUMIFS(Datoteke!$M$2:$M${last},Datoteke!$B$2:$B${last},"{pub}",'
             f'Datoteke!$D$2:$D${last},{y},Datoteke!$E$2:$E${last},{m},'
             f'Datoteke!$P$2:$P${last},"DA",Datoteke!$Q$2:$Q${last},"DA")')
            if pages else "",
            min(nums), max(nums),
            ", ".join(str(g) for g in gaps) if gaps else "",
            rs[0]["supplement"],
        ])
    for w, col in zip((26, 7, 8, 8, 14, 9, 9, 8, 10, 26, 24), "ABCDEFGHIJK"):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name=FONT, size=10)
            c.border = BOX
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    return ws


def sheet_odstupanja(wb: Workbook, rows: list[dict]):
    ws = wb.create_sheet("Odstupanja", 3)
    head = ["Razlog", "Putanja", "Direktorij", "Datoteka", "Bajtova", "Napomena"]
    ws.append(head)
    style_header(ws, 1, len(head))

    # Group by md5 so duplicates can be named as such.
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("md5"):
            by_hash[r["md5"]].append(r)
    dup = {r["rel_path"] for rs in by_hash.values() if len(rs) > 1 for r in rs}

    out = []
    for r in rows:
        why = []
        if r.get("folder_ok") != "True":
            why.append("direktorij izvan konvencije")
        if r.get("file_ok") != "True":
            why.append("naziv datoteke izvan konvencije")
        if r.get("issue_mismatch") == "True":
            why.append("broj u imenu ≠ broj u direktoriju")
        if r["rel_path"] in dup:
            why.append("identičan sadržaj kao druga datoteka")
        if not why:
            continue
        note = ""
        p = r["rel_path"]
        if "/Backup/" in p:
            note = "backup kopija broja — izuzeti iz ingesta"
        elif "ispravak" in p.lower():
            note = "ispravak ranijeg broja — kandidat za vezu „zamijenjeno s”"
        elif r["file"].lower() in ("a.pdf",) or r["file"].lower().startswith(("rrif", "prilog", "strucne")):
            note = "vjerojatno cijeli broj u jednom PDF-u, ne članak"
        out.append([" + ".join(why), p, r["issue_dir"], r["file"],
                    int(r["bytes"]) if str(r["bytes"]).isdigit() else r["bytes"], note])

    for o in sorted(out, key=lambda x: (x[0], x[1])):
        ws.append(o)
    for w, col in zip((34, 56, 16, 30, 11, 46), "ABCDEF"):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name=FONT, size=9)
            c.border = BOX
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    return ws


PITANJA = [
    ("RRiF", "Nedostaje broj 08/2016.", "Je li taj broj izašao?"),
    ("Godišnji obračun", "Postoje 2019.–2023., 2025., 2026. — nema 2024.",
     "Je li prilog za 2024. izašao?"),
    ("Obrtnici", "Nema 2019. (2018. postoji, ali pod drugačijim imenima datoteka).",
     "Je li prilog za 2019. izašao?"),
    ("Godišnji popis imovine", "Samo 12/2025.",
     "Je li serija nova ili nedostaju ranije godine?"),
    ("Stručne informacije 02/2020", "Nedostaje članak rb. 6.",
     "Postoji li taj članak?"),
    ("Siječanjski brojevi", "Od 2020. imaju 18–27 članaka, prije 33–47.",
     "Je li se promijenio opseg broja ili je dio sadržaja premješten u priloge?"),
    ("Ispravci", "Nađen direktorij ispravak_05_2023 uz Stručne informacije 02/2023.",
     "Vodi li se evidencija ispravaka sustavno? Ako da, u kojem obliku?"),
    ("RRIF2005/R200510", "Isporučen kao HTML sa 61 slikom umjesto PDF-a.",
     "Postoji li taj članak i kao PDF?"),
]


def sheet_sazetak(wb: Workbook, rows: list[dict], n: int, src: Path):
    ws = wb.create_sheet("Sažetak", 0)
    last = n + 1
    ws["A1"] = "RRiF — inventura arhiva"
    ws["A1"].font = Font(name=FONT, bold=True, size=15, color=NAVY)
    ws["A2"] = f"Izvor: {src.name}"
    ws["A2"].font = Font(name=FONT, size=9, italic=True, color="666666")

    has_pages = any(str(r_.get("pages", "")).strip().isdigit() for r_ in rows)
    stats = [
        ("PDF datoteka ukupno", f"=COUNTA(Datoteke!$K$2:$K${last})", ""),
        ("Članaka po konvenciji",
         f'=COUNTIFS(Datoteke!$P$2:$P${last},"DA",Datoteke!$Q$2:$Q${last},"DA")', ""),
        ("Odstupanja", "=B4-B5", ""),
        ("Brojeva (svih izdanja)", "=COUNTA(Brojevi!$E$2:$E$5000)", ""),
        ("Stranica ukupno",
         f"=SUM(Datoteke!$M$2:$M${last})" if has_pages else "—",
         "" if has_pages else "manifest je napravljen bez PyMuPDF-a, pa nema broja stranica"),
    ]
    r = 4
    for label, formula, note in stats:
        ws.cell(row=r, column=1, value=label).font = Font(name=FONT, size=10, bold=True)
        c = ws.cell(row=r, column=2, value=formula)
        c.font = Font(name=FONT, size=10)
        c.number_format = "#,##0"
        c.alignment = Alignment(horizontal="left")
        if note:
            nc = ws.cell(row=r, column=3, value=note)
            nc.font = Font(name=FONT, size=9, italic=True, color="666666")
        r += 1

    r += 1
    ws.cell(row=r, column=1, value="Pitanja za RRiF").font = \
        Font(name=FONT, bold=True, size=12, color=NAVY)
    r += 1
    head = ["Što", "Nalaz", "Pitanje", "Odgovor RRiF-a"]
    for i, h in enumerate(head, 1):
        ws.cell(row=r, column=i, value=h)
    style_header(ws, r, len(head))
    r += 1
    for what, finding, question in PITANJA:
        ws.cell(row=r, column=1, value=what)
        ws.cell(row=r, column=2, value=finding)
        ws.cell(row=r, column=3, value=question)
        ws.cell(row=r, column=4, value="")
        for i in range(1, 5):
            c = ws.cell(row=r, column=i)
            c.font = Font(name=FONT, size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.border = BOX
        # The one column meant to be written in.
        ws.cell(row=r, column=4).fill = PatternFill("solid", fgColor="FFFF99")
        ws.row_dimensions[r].height = 30
        r += 1

    r += 1
    ws.cell(row=r, column=1,
            value="Žuti stupac je za upisivanje. Brojke iznad su formule nad listom "
                  "Datoteke — zamijeniš li manifest novijim, sve se preračuna.").font = \
        Font(name=FONT, size=9, italic=True, color="666666")

    for w, col in zip((30, 52, 52, 40), "ABCD"):
        ws.column_dimensions[col].width = w
    return ws


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args()

    src = Path(args.manifest)
    if not src.exists():
        print(f"Nema: {src}")
        return 1
    with src.open(encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(fh)
        headers = list(rdr.fieldnames or [])
        rows = list(rdr)
    if not rows:
        print("Manifest je prazan.")
        return 1
    n = len(rows)

    wb = Workbook()
    wb.remove(wb.active)
    sheet_datoteke(wb, rows, headers)
    sheet_pokrivenost(wb, rows, n)
    sheet_brojevi(wb, rows, n)
    sheet_odstupanja(wb, rows)
    sheet_sazetak(wb, rows, n, src)
    wb._sheets = [wb["Sažetak"], wb["Pokrivenost"], wb["Brojevi"],
                  wb["Odstupanja"], wb["Datoteke"]]

    out = Path(args.out) if args.out else src.with_name("RRiF_inventura_arhiva.xlsx")
    wb.save(out)
    print(f"  → {out}   ({n} datoteka)")
    print("  Listovi: Sažetak · Pokrivenost · Brojevi · Odstupanja · Datoteke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
