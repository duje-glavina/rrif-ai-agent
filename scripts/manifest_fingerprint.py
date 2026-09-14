"""Reduce a manifest to one hash, so two machines can be compared by eye.

The archive was built on the laptop and copied to the Omen. `build_manifest.py`
run on both produces identical *reports*, which is reassuring but not proof:
matching totals and matching anomaly lists would survive a handful of corrupted
bytes in the middle of a PDF.

This collapses every (path, md5) pair into a single fingerprint. Run it on both
machines; if the two lines match, all 5,302 files crossed intact. If they don't,
run the comparison in the docstring below to find which.

    python scripts/manifest_fingerprint.py data/manifest_laptop.csv
    python scripts/manifest_fingerprint.py data/raw/RRIF_arhiva/manifest.csv

To find the difference once you have both CSVs on one machine:

    python scripts/manifest_fingerprint.py A.csv B.csv --diff
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


def load(path: Path) -> dict[str, str]:
    # Paths are stored with forward slashes by build_manifest.py precisely so
    # that a Windows manifest and a Linux one compare as equal.
    with path.open(encoding="utf-8-sig") as fh:
        return {r["rel_path"]: r["md5"] for r in csv.DictReader(fh)}


def fingerprint(rows: dict[str, str]) -> str:
    joined = "\n".join(f"{k}|{v}" for k, v in sorted(rows.items()))
    return hashlib.md5(joined.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", nargs="+")
    ap.add_argument("--diff", action="store_true",
                    help="With two manifests: list what differs")
    args = ap.parse_args()

    loaded = []
    for m in args.manifest:
        p = Path(m)
        if not p.exists():
            print(f"Nema: {p}")
            return 1
        rows = load(p)
        if not any(rows.values()):
            print(f"  {p}: nema md5 vrijednosti — manifest je pokrenut s --no-hash")
            return 1
        print(f"  {fingerprint(rows)}   {len(rows):>5} datoteka   {p}")
        loaded.append((p, rows))

    if args.diff:
        if len(loaded) != 2:
            print("\n  --diff traži točno dva manifesta.")
            return 1
        (pa, a), (pb, b) = loaded
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
        print(f"\n  samo u {pa.name}: {len(only_a)}"
              f"   samo u {pb.name}: {len(only_b)}"
              f"   isti put, drugi sadržaj: {len(changed)}")
        for label, items in (("samo A", only_a), ("samo B", only_b),
                             ("različito", changed)):
            for k in items[:20]:
                print(f"    [{label}] {k}")
            if len(items) > 20:
                print(f"    … i još {len(items) - 20}")
        return 0 if not (only_a or only_b or changed) else 2

    if len(loaded) > 1:
        same = len({fingerprint(r) for _, r in loaded}) == 1
        print("\n  " + ("Identično." if same else
                        "RAZLIKA — pokreni ponovno s --diff."))
        return 0 if same else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
