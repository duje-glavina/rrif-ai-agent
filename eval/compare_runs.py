"""Compare eval runs question by question.

The summary block in run_eval.py reports percentages over n=37 (content),
n=13 (source) and n=6 (law articles). One question is 2.7, 7.7 and 16.7
points respectively, so a headline move of "+5.4pp" means two questions
changed — and until you know WHICH two, and whether they change on a rerun
of the same config, you don't know whether you measured anything.

Two modes:

    # A vs B — what actually moved
    python -m eval.compare_runs eval/results/<base>.json eval/results/<new>.json

    # Same config, several runs — how much the number moves on its own
    python -m eval.compare_runs --spread eval/results/<run1>.json <run2>.json ...

Read --spread first. A metric whose spread across identical runs is wider
than the A/B delta has told you nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# metric key in per_question  →  (label, denominator key, gradeable flag)
METRICS = [
    ("content_top_1",     "ODGOVOR top-1",     "gradeable_content"),
    ("content_top_k",     "ODGOVOR top-5",     "gradeable_content"),
    ("content_any_top_k", "bar jedan pojam",   "gradeable_content"),
    ("source_top_1",      "ČLANCI src top-1",  "gradeable_source"),
    ("source_top_k",      "ČLANCI src top-5",  "gradeable_source"),
    ("retrieval_top_1",   "ZAKONI čl. top-1",  "gradeable_retrieval"),
    ("retrieval_top_k",   "ZAKONI čl. top-5",  "gradeable_retrieval"),
]


def load(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["_name"] = Path(path).stem
    data["_by_id"] = {q["id"]: q for q in data["per_question"]}
    return data


def _count(run: dict, metric: str, flag: str) -> tuple[int, int]:
    """(hits, denominator) for one metric in one run."""
    rows = [q for q in run["per_question"] if q.get(flag)]
    return sum(1 for q in rows if q.get(metric)), len(rows)


def _pct(hits: int, n: int) -> float:
    return 100.0 * hits / n if n else 0.0


def spread(paths: list[str]) -> None:
    runs = [load(p) for p in paths]
    print(f"\nSpread across {len(runs)} runs of (nominally) the same config")
    print("=" * 72)
    for name in (r["_name"] for r in runs):
        print(f"  {name}")
    print()
    print(f"  {'metric':<20} {'min':>7} {'max':>7} {'range':>8}   n")
    print("  " + "-" * 52)
    for metric, label, flag in METRICS:
        vals, n = [], 0
        for r in runs:
            hits, n = _count(r, metric, flag)
            vals.append(_pct(hits, n))
        lo, hi = min(vals), max(vals)
        flag_noisy = " ←" if hi - lo > 0 else ""
        print(f"  {label:<20} {lo:6.1f}% {hi:6.1f}% {hi-lo:7.1f}pp   {n}{flag_noisy}")
    print()
    print("  Any metric with a non-zero range moves without a code change.")
    print("  Treat a delta smaller than that range as unmeasured.")

    # Which questions are unstable, regardless of metric.
    ids = sorted(set(runs[0]["_by_id"]))
    unstable = []
    for qid in ids:
        for metric, label, flag in METRICS:
            rows = [r["_by_id"].get(qid, {}) for r in runs]
            if not all(row.get(flag) for row in rows):
                continue          # not graded on this metric — nothing to compare
            if len({bool(row.get(metric)) for row in rows}) > 1:
                unstable.append((qid, label))
    if unstable:
        print(f"\n  Unstable questions ({len({u[0] for u in unstable})}):")
        for qid, label in unstable:
            print(f"    {qid:<34} {label}")


def compare(base_path: str, new_path: str) -> None:
    base, new = load(base_path), load(new_path)
    print(f"\nbase: {base['_name']}")
    print(f"new : {new['_name']}")
    print("=" * 72)
    print(f"  {'metric':<20} {'base':>7} {'new':>7} {'delta':>8}   moved   n")
    print("  " + "-" * 60)

    changes: dict[str, list[str]] = {}

    for metric, label, flag in METRICS:
        b_hits, n = _count(base, metric, flag)
        a_hits, _ = _count(new, metric, flag)
        b_pct, a_pct = _pct(b_hits, n), _pct(a_hits, n)

        gained = [q["id"] for q in new["per_question"]
                  if q.get(flag) and q.get(metric)
                  and not base["_by_id"].get(q["id"], {}).get(metric)]
        lost = [q["id"] for q in new["per_question"]
                if q.get(flag) and not q.get(metric)
                and base["_by_id"].get(q["id"], {}).get(metric)]

        moved = f"+{len(gained)}/-{len(lost)}" if (gained or lost) else "·"
        print(f"  {label:<20} {b_pct:6.1f}% {a_pct:6.1f}% {a_pct-b_pct:+7.1f}pp {moved:>7}   {n}")

        for qid in gained:
            changes.setdefault(qid, []).append(f"+{label}")
        for qid in lost:
            changes.setdefault(qid, []).append(f"-{label}")

    if not changes:
        print("\n  No question changed on any metric.")
        return

    print(f"\n  {len(changes)} questions changed:")
    print("  " + "-" * 60)
    for qid, marks in sorted(changes.items()):
        b = base["_by_id"].get(qid, {})
        a = new["_by_id"].get(qid, {})
        print(f"\n  {qid}   {'  '.join(marks)}")
        print(f"    {a.get('query', '')[:80]}")
        if a.get("gradeable_retrieval"):
            print(f"    want čl. {a.get('expected_articles')}")
            print(f"    base got {b.get('retrieved_articles')}")
            print(f"    new  got {a.get('retrieved_articles')}")
        elif a.get("gradeable_source"):
            print(f"    want {a.get('expected_sources')}")
            print(f"    base got {[s for s in b.get('retrieved_sources', []) if s]}")
            print(f"    new  got {[s for s in a.get('retrieved_sources', []) if s]}")
        bk, ak = b.get("keyword_hits"), a.get("keyword_hits")
        if bk is not None or ak is not None:
            print(f"    keyword hits {bk}/{b.get('keyword_total')} → "
                  f"{ak}/{a.get('keyword_total')}")
        print(f"    rerank {b.get('top_rerank_score')} → {a.get('top_rerank_score')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="Result JSON files")
    ap.add_argument("--spread", action="store_true",
                    help="Treat all inputs as repeats of one config and report the range.")
    args = ap.parse_args()

    if args.spread:
        if len(args.runs) < 2:
            ap.error("--spread needs at least two runs")
        spread(args.runs)
    else:
        if len(args.runs) != 2:
            ap.error("give exactly two runs, or use --spread")
        compare(args.runs[0], args.runs[1])


if __name__ == "__main__":
    main()
