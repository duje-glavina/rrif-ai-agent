r"""Inventory the RRiF PDF archive before anything is ingested.

WHY THIS EXISTS
───────────────
The archive arrived as 5,300 PDFs in 129 folders spanning 2016-01 to 2026-09.
Nothing about it is going to announce its own gaps: a mis-cased folder, a file
that lost its place in the numbering, a supplement buried one directory deeper
than the walker looks — all of those produce a *smaller* corpus, silently, and
you find out months later when an advisor asks about something that is sitting
right there on disk.

So: walk the whole tree, parse every path, hash every file, and write one CSV.
That CSV is then two things at once —

  1. the list of what actually exists, article by article, and
  2. the column the metadata XLS from RRiF joins onto.

When that XLS lands, `unit_ref` here should match theirs row for row. Where it
doesn't, one of you is wrong, and it is much cheaper to find that out against a
CSV than against an index.

This script reads PDFs. It does not write to the database, does not embed, and
does not care whether the validity metadata has arrived yet.

USAGE
─────
    python scripts/build_manifest.py data/raw/RRIF_arhiva/RRIF
    python scripts/build_manifest.py data/raw/RRIF_arhiva/RRIF --no-hash
    python scripts/build_manifest.py data/raw/... -o data/manifest_rrif.csv

Needs only the standard library. PyMuPDF, if importable, adds a page count.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:                                    # pragma: no cover
    try:
        import fitz                                    # older PyMuPDF
    except ImportError:
        fitz = None


# ── Naming conventions ────────────────────────────────────────────────────────
#
# Issue folder:  RRIF2609, PROR1706, NEPR2312, OBAV2602, PRGO2601, OBRT1601
#                <code><yy><mm>
# Article file:  R260931.PDF
#                <letter><yy><mm><nn>
#
# Both are matched case-insensitively on purpose. The archive contains RRiF1604
# and RRiF2111 with a lowercase i, and 24 files ending .pdf rather than .PDF.
# Neither is worth renaming on disk — the archive is a copy, not a source of
# truth — but every path must be built from this manifest rather than from a
# format string, because a format string will miss exactly those 26 things.

ISSUE_DIR = re.compile(r"^([A-Za-z]{3,6})(\d{2})(\d{2})$")
ARTICLE = re.compile(r"^([A-Za-z])(\d{2})(\d{2})(\d{1,3})$")

PUB_LABEL = {
    "RRIF": "RRiF",
    "PIP": "Porezno i pravno (PiP)",
    "PROR": "Proračun",
    "OBAV": "Obavijesti",
    "NEPR": "Neprofitne organizacije",
    # Supplements. Absent from article_loader.PUB_TYPE as of this writing,
    # which is why PRGO2601 would cite itself as "PRGO br. 1/2026".
    "PRGO": "RRiF Godišnji obračun",
    "PRPI": "RRiF Godišnji popis imovine",
    "OBRT": "RRiF Obrtnici",
}

# Supplements live at RRIF/_prilozi/<Naziv priloga>/<ISSUE>/, i.e. one level
# deeper than the magazine issues. That extra level is what the current
# collect_pdfs() walks past.
SUPPLEMENT_ROOT = "_prilozi"


def md5(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while chunk := fh.read(buf):
            h.update(chunk)
    return h.hexdigest()


def page_count(path: Path) -> int | None:
    if fitz is None:
        return None
    try:
        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:
        return None


def parse(pdf: Path, root: Path) -> dict:
    """One row. Unparseable paths still get a row — with flags set."""
    rel = pdf.relative_to(root)
    parts = rel.parts

    supplement = ""
    if parts and parts[0] == SUPPLEMENT_ROOT:
        # _prilozi / <Naziv> / <ISSUE> / file.PDF
        supplement = parts[1] if len(parts) > 2 else "?"

    folder = pdf.parent.name
    fm = ISSUE_DIR.match(folder)
    sm = ARTICLE.match(pdf.stem)

    pub_code = fm.group(1).upper() if fm else ""
    year = 2000 + int(fm.group(2)) if fm else None
    month = int(fm.group(3)) if fm else None
    article_num = sm.group(4) if sm else ""

    # Cross-check: the file name repeats the issue the folder already states.
    # When they disagree, the file was almost certainly filed in the wrong
    # folder — rare, but it silently mis-dates an article, which for a tax
    # magazine is the one error that matters most.
    mismatch = bool(
        fm and sm and (int(sm.group(2)) != int(fm.group(2))
                       or int(sm.group(3)) != int(fm.group(3)))
    )

    unit_ref = (f"{pub_code}{year % 100:02d}{month:02d}-{article_num}"
                if (fm and sm) else "")

    st = pdf.stat()
    return {
        "unit_ref": unit_ref,
        "pub_code": pub_code,
        "pub_label": PUB_LABEL.get(pub_code, pub_code),
        "year": year or "",
        "month": month or "",
        "article_num": article_num,
        "supplement": supplement,
        "issue_dir": folder,
        "file": pdf.name,
        "ext": pdf.suffix,
        "rel_path": str(rel).replace("\\", "/"),
        "bytes": st.st_size,
        "pages": "",
        "md5": "",
        "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d"),
        "folder_ok": bool(fm),
        "file_ok": bool(sm),
        "issue_mismatch": mismatch,
        "nonstandard_case": folder != folder.upper() or pdf.suffix != pdf.suffix.upper(),
    }


def report(rows: list[dict], others: list[Path], root: Path) -> None:
    ok = [r for r in rows if r["folder_ok"] and r["file_ok"]]
    print()
    print("=" * 72)
    print("  ARHIVA — INVENTURA")
    print("=" * 72)
    print(f"  Korijen:                {root}")
    print(f"  PDF datoteka:           {len(rows)}")
    print(f"  Parsirano po konvenciji:{len(ok)}")
    print(f"  Ostale datoteke:        {len(others)}")

    # ── Per publication ──────────────────────────────────────────────────────
    by_pub: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        by_pub[r["pub_code"]].append(r)
    print("\n  Po izdanju")
    print("  " + "-" * 68)
    for code in sorted(by_pub):
        rs = by_pub[code]
        issues = sorted({(r["year"], r["month"]) for r in rs})
        span = f"{issues[0][0]}-{issues[0][1]:02d} → {issues[-1][0]}-{issues[-1][1]:02d}"
        print(f"  {code:<6} {PUB_LABEL.get(code, code):<32} "
              f"{len(issues):>4} br.  {len(rs):>5} čl.   {span}")

    # ── Missing issues ───────────────────────────────────────────────────────
    # Only for publications that clearly run monthly; a supplement that comes
    # out every January is not "missing" eleven issues a year.
    print("\n  Rupe u nizu brojeva")
    print("  " + "-" * 68)
    found_any_gap = False
    for code in sorted(by_pub):
        issues = sorted({(r["year"], r["month"]) for r in by_pub[code]})
        months = {m for _, m in issues}
        if len(months) < 6:          # seasonal / annual supplement
            print(f"  {code:<6} izlazi u mjesecima {sorted(months)} — "
                  f"godine: {sorted({y for y, _ in issues})}")
            continue
        seq = [(y, m) for y in range(issues[0][0], issues[-1][0] + 1)
               for m in range(1, 13)]
        lo, hi = issues[0], issues[-1]
        seq = [x for x in seq if lo <= x <= hi]
        missing = [f"{y}-{m:02d}" for y, m in seq if (y, m) not in set(issues)]
        if missing:
            found_any_gap = True
            print(f"  {code:<6} nedostaje {len(missing)}: {', '.join(missing)}")
    if not found_any_gap:
        print("  (nema rupa u mjesečnim izdanjima)")

    # ── Gaps inside an issue ─────────────────────────────────────────────────
    # Article numbers run 01..NN. A hole means an article that exists in the
    # printed issue and not on disk — which is exactly what the XLS from RRiF
    # will let you confirm, and exactly what it will not tell you on its own.
    # Misfiled articles are excluded: one file named R260299 sitting in
    # RRIF2601 would otherwise invent a 95-article hole.
    numbered = [r for r in ok if not r["issue_mismatch"]]
    holes: list[tuple[str, list[int]]] = []
    for (code, y, m), rs in sorted(
            {(r["pub_code"], r["year"], r["month"]): None for r in numbered}.items()):
        nums = sorted(int(r["article_num"]) for r in numbered
                      if (r["pub_code"], r["year"], r["month"]) == (code, y, m))
        gap = [n for n in range(1, max(nums) + 1) if n not in set(nums)]
        if gap:
            holes.append((f"{code}{y % 100:02d}{m:02d}", gap))
    print(f"\n  Brojevi s prekidom u numeraciji članaka: {len(holes)}")
    for name, gap in holes[:15]:
        print(f"    {name}  nedostaju rb. {gap if len(gap) <= 12 else gap[:12] + ['…']}")
    if len(holes) > 15:
        print(f"    … i još {len(holes) - 15}")

    # ── Anomalies ────────────────────────────────────────────────────────────
    bad_folder = [r for r in rows if not r["folder_ok"]]
    bad_file = [r for r in rows if r["folder_ok"] and not r["file_ok"]]
    case = [r for r in rows if r["nonstandard_case"]]
    mism = [r for r in rows if r["issue_mismatch"]]

    print("\n  Odstupanja")
    print("  " + "-" * 68)
    print(f"  Direktorij ne odgovara konvenciji: {len(bad_folder)}")
    for r in bad_folder[:8]:
        print(f"    {r['rel_path']}")
    print(f"  Naziv datoteke ne odgovara:        {len(bad_file)}")
    for r in bad_file[:8]:
        print(f"    {r['rel_path']}")
    print(f"  Nestandardna velika/mala slova:    {len(case)}")
    for d in sorted({r['issue_dir'] for r in case if r['issue_dir'] != r['issue_dir'].upper()}):
        print(f"    direktorij: {d}")
    low_ext = sorted({r['ext'] for r in case if r['ext'] != r['ext'].upper()})
    if low_ext:
        print(f"    ekstenzije: {', '.join(low_ext)} "
              f"({sum(1 for r in case if r['ext'] in low_ext)} datoteka)")
    print(f"  Broj u imenu ≠ broj u direktoriju:  {len(mism)}")
    for r in mism[:8]:
        print(f"    {r['rel_path']}  (direktorij kaže {r['issue_dir']})")

    if others:
        print(f"\n  Ne-PDF datoteke: {len(others)}")
        by_dir = Counter(str(p.parent.relative_to(root)) for p in others)
        for d, n in by_dir.most_common(10):
            print(f"    {n:>4}  {d}")

    # ── Duplicates ───────────────────────────────────────────────────────────
    if any(r["md5"] for r in rows):
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r["md5"]:
                groups[r["md5"]].append(r)
        dups = {h: rs for h, rs in groups.items() if len(rs) > 1}
        n_extra = sum(len(rs) - 1 for rs in dups.values())
        print(f"\n  Identičan sadržaj (md5): {len(dups)} skupina, "
              f"{n_extra} višak datoteka")
        # The question this is here to answer: is a supplement also filed
        # inside its magazine issue? If so, ingesting both double-indexes it.
        cross = [rs for rs in dups.values()
                 if len({bool(r["supplement"]) for r in rs}) > 1]
        if cross:
            print(f"  ⚠ {len(cross)} skupina spaja prilog i redovni broj — "
                  f"isti članak na dva mjesta:")
            for rs in cross[:6]:
                for r in rs:
                    print(f"      {r['rel_path']}")
                print()
        for rs in list(dups.values())[:5]:
            if rs not in cross:
                print(f"    {rs[0]['bytes']:>9} B  " +
                      "  ·  ".join(r["rel_path"] for r in rs[:3]))

    pages = [r["pages"] for r in rows if isinstance(r["pages"], int)]
    if pages:
        print(f"\n  Stranica ukupno: {sum(pages)}   "
              f"prosječno po članku: {sum(pages) / len(pages):.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="Archive root, e.g. data/raw/RRIF_arhiva/RRIF")
    ap.add_argument("-o", "--out", default="", help="CSV path (default: <root>/../manifest.csv)")
    ap.add_argument("--no-hash", action="store_true",
                    help="Skip md5 — faster, but no duplicate detection")
    ap.add_argument("--no-pages", action="store_true",
                    help="Skip the page count even if PyMuPDF is available")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1

    all_files = [p for p in root.rglob("*") if p.is_file()]
    pdfs = sorted(p for p in all_files if p.suffix.lower() == ".pdf")
    others = sorted(p for p in all_files if p.suffix.lower() != ".pdf")
    if not pdfs:
        print(f"No PDFs under {root}")
        return 1

    print(f"  {len(pdfs)} PDF-ova pod {root}")
    if fitz is None and not args.no_pages:
        print("  (PyMuPDF nije dostupan — bez broja stranica)")

    rows: list[dict] = []
    for i, pdf in enumerate(pdfs, 1):
        r = parse(pdf, root)
        if not args.no_hash:
            r["md5"] = md5(pdf)
        if not args.no_pages:
            n = page_count(pdf)
            if n is not None:
                r["pages"] = n
        rows.append(r)
        if i % 250 == 0 or i == len(pdfs):
            print(f"    {i}/{len(pdfs)}", end="\r", flush=True)
    print(" " * 30, end="\r")

    out = Path(args.out) if args.out else root.parent / "manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    report(rows, others, root)
    print(f"\n  → {out}")
    print("\n  Kad stigne XLS od RRiF-a, spoji po stupcu unit_ref.")
    print("  Redak koji postoji samo s jedne strane je ili članak koji nemamo,")
    print("  ili članak koji oni ne znaju da imaju. Oboje želiš vidjeti sada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
