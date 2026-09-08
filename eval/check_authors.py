"""Check that the system never attributes an answer to the wrong author.

WHY THIS IS ITS OWN SCRIPT
──────────────────────────
RRiF's authority rests on named experts. Every article carries a byline, and
advisors know those names. An answer that says "(RRiF br. 5/2024, autor: Dr.
sc. Nada DREMEL)" when Nada Dremel did not write that article is worse than a
wrong figure: it is a real professional's name attached to advice they did not
give. It is also the failure a keyword metric cannot see and the one a reader
is least likely to verify, because a plausible name reads as provenance.

The generator is handed `author` from each chunk's `extra_metadata`, so it has
a legitimate source for the name. Two things can still go wrong:

  1. the model names an author no retrieved chunk carries — invention;
  2. the byline extraction put the wrong name on the chunk in the first place —
     in which case the answer is faithfully repeating our own bad metadata.

This script separates them. It compares the names an answer states against the
authors actually stored on the chunks that answer was built from. Case (1) it
can detect on its own. Case (2) it cannot — that needs a human opening the
magazine — but it tells you which chunk to check.

`grade_answers.py` cannot do this: `retrieved_meta` carries `source` and
`chunk_text` but not `author`, so the judge sees an author attribution it has
no way to verify and correctly reports it as unsupported. That is a false
alarm from a blind grader, not evidence of invention — which is exactly why
this check reads the authors from the database instead.

USAGE
─────
    python -m eval.check_authors eval/results/<run>.json
    python -m eval.check_authors eval/results/<run>.json --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

# How the generator writes an attribution, e.g.
#   "(RRiF br. 1/2024, autor: Mr. Anja Božina, ovl. rač.)"
#   "(Porezno i pravno (PiP) br. 10/2024, autorica: Darko MAREČIĆ, dipl. iur.)"
_ATTRIB = re.compile(r"autor(?:ica)?\s*:\s*([^)\n;]+)", re.IGNORECASE)

# Academic titles, degrees and professional suffixes. Stripped from both sides
# so "Mr. Anja Božina, ovl. rač." and "Anja Božina" compare equal.
_TITLES = {
    "dr", "sc", "mr", "prof", "mag", "dipl", "iur", "oec", "ing", "univ",
    "spec", "ovl", "rac", "rač", "bacc", "phd", "msc", "struc", "struč",
    "doc", "emer", "d", "o",
}


def _fold(s: str) -> str:
    """Lowercase, strip diacritics and punctuation. Comparison form only."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s]", " ", s.lower())


def _name_tokens(s: str) -> set[str]:
    """The words of a name, with titles and initials removed."""
    return {
        t for t in _fold(s).split()
        if len(t) > 1 and t not in _TITLES
    }


def _chunk_ids(row: dict) -> list[str]:
    """The chunk ids behind one answer.

    `QueryResponse` has `retrieved_chunk_ids`, but `evaluate_one` never copies
    it into the per-question record — the ids only survive inside
    `retrieved_meta[i]["chunk_id"]`. Reading the wrong key returned an empty
    list for every question, which this script then reported as 74 unmatched
    author attributions: a checker whose own input was missing, announcing a
    catastrophe. Hence the fallback here and the hard guard in main().
    """
    ids = [m.get("chunk_id") for m in (row.get("retrieved_meta") or [])]
    ids = [str(i) for i in ids if i]
    if ids:
        return ids
    return [str(i) for i in (row.get("retrieved_chunk_ids") or []) if i]


def _matches(claimed: str, actual: str) -> bool:
    """Do these name the same person?

    Subset rather than equality, in both directions: the answer may give
    "Anja Božina" where the metadata says "Mr. Anja Božina, ovl. rač.", or
    only a surname. Requires at least one real token so an empty name on
    either side never counts as a match.
    """
    a, b = _name_tokens(claimed), _name_tokens(actual)
    if not a or not b:
        return False
    return a <= b or b <= a


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--verbose", action="store_true",
                    help="Print every attribution, not only the unmatched ones")
    args = ap.parse_args()

    src = Path(args.results)
    if not src.exists():
        print(f"Not found: {src}")
        return 1

    data = json.loads(src.read_text(encoding="utf-8"))
    rows = [r for r in data.get("per_question", [])
            if r.get("in_corpus") and (r.get("answer") or "").strip()]
    if not rows:
        print("No answers in this results file — was it run with --skip-generation?")
        return 1

    # One query for every chunk in the run, rather than one per question.
    all_ids = sorted({cid for r in rows for cid in _chunk_ids(r)})

    # Guard, not politeness. With no ids this script would compare every
    # author against an empty set and report every single one as unmatched —
    # a false alarm indistinguishable from the real thing, on the one check
    # where a false alarm is most expensive.
    if not all_ids:
        print("No chunk ids in this results file, so nothing can be verified.\n"
              "Expected `chunk_id` inside `retrieved_meta`. Re-run the eval with a\n"
              "current run_eval.py rather than trusting an empty result here.")
        return 1

    authors: dict[str, tuple[str, str]] = {}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        cur = conn.execute(
            """SELECT id::text,
                      coalesce(extra_metadata->>'author', ''),
                      coalesce(source, '')
               FROM chunks
               WHERE id::text = ANY(%s)""",
            (all_ids,),
        )
        for cid, author, source in cur.fetchall():
            authors[cid] = (author, source)

    if not authors:
        print(f"None of the {len(all_ids)} chunk ids in this run exist in "
              f"{os.environ['DATABASE_URL'].rsplit('/', 1)[-1]}.\n"
              "The results file and DATABASE_URL are pointing at different corpora —\n"
              "verify against the database the run actually used.")
        return 1

    missing_meta = sum(1 for cid in all_ids if not authors.get(cid, ("", ""))[0])

    n_attrib = 0
    unmatched: list[tuple[str, str, list[str]]] = []

    for r in rows:
        claimed = [c.strip().rstrip(".,") for c in _ATTRIB.findall(r.get("answer", ""))]
        claimed = list(dict.fromkeys(claimed))
        if not claimed:
            continue
        available = [authors.get(cid, ("", ""))[0] for cid in _chunk_ids(r)]
        available = [a for a in available if a]

        for name in claimed:
            n_attrib += 1
            if not any(_matches(name, a) for a in available):
                unmatched.append((r["id"], name, sorted(set(available))))
            elif args.verbose:
                print(f"  ok  {r['id']:<28} {name}")

    print()
    print("=" * 70)
    print("Pripisivanje autorstva")
    print("=" * 70)
    print(f"  Odgovora s navedenim autorom:  {sum(1 for r in rows if _ATTRIB.search(r.get('answer','')))}"
          f" / {len(rows)}")
    print(f"  Ukupno navedenih autora:       {n_attrib}")
    print(f"  Bez pokrića u dohvaćenim chunkovima: {len(unmatched)}")
    if missing_meta:
        print(f"  ⚠ {missing_meta} od {len(all_ids)} dohvaćenih chunkova nema autora u "
              f"metapodatcima —")
        print(f"    za njih ova provjera ne može ništa potvrditi.")

    if unmatched:
        print("\n  Za ručnu provjeru:")
        for qid, name, available in unmatched:
            print(f"\n    {qid}")
            print(f"      odgovor navodi : {name}")
            print(f"      u chunkovima   : {', '.join(available) or '(nijedan chunk nema autora)'}")
        print("\n  Ovo je ili izmišljeno ime ili kriva ekstrakcija potpisa pri ingestu.")
        print("  Razliku vidi samo čovjek koji otvori taj broj časopisa — ali prije")
        print("  nego išta ode RRiF-u, ovo treba biti prazno.")
    else:
        print("\n  Svako navedeno ime nalazi se na nekom dohvaćenom chunku.")
        print("  (To ne znači da je potpis točan — samo da nije izmišljen ovdje.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
