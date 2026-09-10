"""Replay RRiF's own questions from the PoC log against the current system.

WHY THIS BEATS THE GOLDEN SET
─────────────────────────────
The 41-question golden set was written by us and marked against keywords we
chose. The `queries` table on the PoC holds 92 questions three RRiF advisors
actually asked, and `feedback` holds their verdicts on 19 of the answers — in
their own words, on the four-point scale from the acceptance criteria.

Every one of those was answered by v1: the corpus where 48.7% of tokens never
reached the embedder, where `HISTORICAL_CUTOFF` hid 5,174 chunks of valid
method, and where the median chunk was 649 tokens. Half of those 92 questions
were refused. So this replay is a before/after on the client's own data, using
the client's own judgements as the yardstick — which is a far stronger claim
than any number we can compute about ourselves.

WHOSE QUESTIONS ARE WHOSE
─────────────────────────
Only `rriftest3` is RRiF — 27 questions on 25 and 29 May, 9 of them graded.
`rriftest1` and `Savjetnik 1` are our own testing and include things like
"Kako je komad sira?", which is why any statistic over the whole 92-row log is
meaningless. Use --advisor rriftest3 for the number that means anything.

Of RRiF's 9 refusals, only three are retrieval failures where the advisor told
us the article was in the corpus:

  - "Koji je opći zastarni rok…"  → RRiF 11/24, Računovodstvo otpisa obveza
  - the solar-panel VAT question  → RRiF 11/24, Porezno motrište … solarnih ploča
  - the same question with "solarne elektrane" instead of "solarne ploče",
    which the advisor noted matches the article's own title

FOUR of the nine are something a new corpus cannot fix: the advisor was asking
follow-up questions in one conversation — "Treba li ga primijeniti…", "Daj mi
primjer te detaljnije analize F. Plišića", "Iz prethodnog pitanja koje sam
postavio" — and `ask()` takes a single question with no history. Expect those
four to refuse again. That is not a regression; it is the missing feature, and
it is the largest single cause of failure in real advisor use.

WHAT IT DOES NOT MEASURE
────────────────────────
Whether the new answers are CORRECT. Only an advisor can say that, and the old
verdicts do not transfer to new answers. What it measures is: does the system
now answer where it used to refuse, and how do the two answers read side by
side.

USAGE
─────
    # on the Omen — reads the PoC log straight out of Postgres
    python -m eval.replay_real --from-db rrif_rag --advisor rriftest3
    python -m eval.replay_real --from-db rrif_rag --advisor rriftest3 --only-graded
    python -m eval.replay_real --from-db rrif_rag --dedupe

    # or from CSV exports, if the log is not on this machine
    python -m eval.replay_real queries.csv feedback.csv --advisor rriftest3

Note the two databases: --from-db names where the LOG lives (v1, because the
PoC process started before the switch), while the replay itself retrieves from
whatever DATABASE_URL currently points at (v2). That is the whole point.

Writes eval/results/<ts>_replay_real.json in the same shape run_eval produces,
so make_printable.py, check_authors.py and grade_answers.py all read it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.query import ask                                    # noqa: E402

RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_TRUE = {"t", "true", "True", "1", "y", "yes"}


def _b(v) -> bool:
    if isinstance(v, bool):
        return v
    return (v or "").strip() in _TRUE if isinstance(v, str) else bool(v)


# ── Reading the log ──────────────────────────────────────────────────────────
# The PoC writes to whatever DATABASE_URL pointed at when its process started,
# which is the v1 database — the switch to v2 only affects processes started
# after it. So the log and the corpus being replayed against are two different
# databases, deliberately, and --from-db names the one holding the log.

_Q_COLS = ("query_id", "created_at", "advisor_id", "question_text",
           "classified_category", "referred_to_advisor", "answer_text",
           "tokens_in", "tokens_out", "latency_ms")
_F_COLS = ("feedback_id", "created_at", "query_id", "advisor_id", "rating",
           "accuracy_verdict", "would_send_to_client", "failure_mode",
           "comment", "suggested_answer")


def _log_url(spec: str) -> str:
    """A bare name is resolved against DATABASE_URL; a full URL is used as-is."""
    if "://" in spec:
        return spec
    base = os.environ.get("DATABASE_URL")
    if not base:
        raise SystemExit("DATABASE_URL is not set, so a bare database name "
                         "cannot be resolved. Pass a full connection URL.")
    return base.rsplit("/", 1)[0] + "/" + spec


def load_from_db(spec: str) -> tuple[list[dict], dict[str, dict]]:
    import psycopg                                        # local: only this path needs it
    url = _log_url(spec)
    with psycopg.connect(url) as conn:
        qs = conn.execute(
            f"SELECT {', '.join(_Q_COLS)} FROM queries "
            "WHERE question_text IS NOT NULL AND btrim(question_text) <> '' "
            "ORDER BY created_at"
        ).fetchall()
        fs = conn.execute(
            f"SELECT {', '.join(_F_COLS)} FROM feedback ORDER BY created_at"
        ).fetchall()

    def _safe(v):
        """Anything psycopg hands back must survive json.dumps.

        Datetimes were handled; UUIDs were not, and the whole run died at the
        write step after every API call had already been paid for. Coerce at
        the boundary rather than at each use site — the next column type that
        is not JSON-native will otherwise do the same thing again.
        """
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        if hasattr(v, "isoformat"):
            return v.isoformat(sep=" ")
        return str(v)

    def row(cols, r):
        return {c: _safe(v) for c, v in zip(cols, r)}

    queries = [row(_Q_COLS, r) for r in qs]
    feedback = {}
    for r in fs:                                # later rows win, as in the CSV path
        d = row(_F_COLS, r)
        feedback[str(d["query_id"])] = d
    for q in queries:
        q["query_id"] = str(q["query_id"])
    print(f"  log: {len(queries)} pitanja, {len(feedback)} ocjena "
          f"iz {url.rsplit('/', 1)[-1]}")
    return queries, feedback


def _i(v: str | None) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def load_queries(path: Path) -> list[dict]:
    csv.field_size_limit(10 ** 9)
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if (r.get("question_text") or "").strip()]


def load_feedback(path: Path | None) -> dict[str, dict]:
    if not path:
        return {}
    csv.field_size_limit(10 ** 9)
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # One query can have several feedback rows; keep the most recent.
    out: dict[str, dict] = {}
    for r in sorted(rows, key=lambda x: x.get("created_at", "")):
        out[r["query_id"]] = r
    return out


def replay_one(src: dict, fb: dict | None) -> dict:
    """Ask the current system the same question, and keep the old answer beside it."""
    question = src["question_text"].strip()
    t0 = time.perf_counter()
    try:
        res = ask(question)
        err = None
    except Exception as exc:                       # noqa: BLE001 — one bad row must not stop 92
        return {
            "id": src["query_id"][:8],
            "query": question,
            "error": str(exc),
            "in_corpus": True,
            "scope": "članak",
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }

    cost = round(res.tokens_in / 1e6 * 3.0 + res.tokens_out / 1e6 * 15.0, 6)

    row = {
        # Shaped like run_eval's per_question so the rest of the tooling reads it.
        "id": src["query_id"][:8],
        "query": question,
        "in_corpus": True,
        "scope": "članak",
        "gradeable_content": False,
        "gradeable_source": False,
        "gradeable_retrieval": False,
        "answer": res.answer,
        "answer_preview": res.answer[:300],
        "refused": res.referred_to_advisor,
        "citations_raw": [
            {"source": c.source, "article_number": c.article_number, "excerpt": c.excerpt}
            for c in res.citations
        ],
        "n_citations": len(res.citations),
        "retrieved_meta": res.retrieved_meta,
        "latency_ms": res.latency_ms,
        "tokens_in": res.tokens_in,
        "tokens_out": res.tokens_out,
        "cost_usd": cost,
        "actual_category": res.classifier.category,
        "expected_category": None,
        "category_correct": False,
        "passed": None,
        "top_rerank_score": (res.retrieved_meta[0].get("rerank_score")
                             if res.retrieved_meta else None),
        # ── what the PoC did with the same question, in May ──────────────
        "v1": {
            "asked_at": src.get("created_at"),
            "advisor_id": src.get("advisor_id"),
            "refused": _b(src.get("referred_to_advisor")),
            "answer": src.get("answer_text") or "",
            "category": src.get("classified_category"),
            "tokens_in": _i(src.get("tokens_in")),
            "tokens_out": _i(src.get("tokens_out")),
            "latency_ms": _i(src.get("latency_ms")),
        },
    }
    if fb:
        row["feedback"] = {
            "accuracy_verdict": fb.get("accuracy_verdict") or None,
            "would_send_to_client": fb.get("would_send_to_client") or None,
            "failure_mode": fb.get("failure_mode") or None,
            "comment": fb.get("comment") or None,
            "advisor_id": fb.get("advisor_id"),
        }
    return row


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f}%" if d else "    -"


def report(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    if not n:
        return {}

    new_ref = sum(1 for r in ok if r.get("refused"))
    old_ref = sum(1 for r in ok if r["v1"]["refused"])
    recovered = [r for r in ok if r["v1"]["refused"] and not r.get("refused")]
    lost = [r for r in ok if not r["v1"]["refused"] and r.get("refused")]

    def avg(f, rows_=ok):
        v = [f(r) for r in rows_]
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None

    m = {
        "n": n,
        "refused_v1": old_ref, "refused_now": new_ref,
        "recovered": len(recovered), "newly_refused": len(lost),
        "latency_v1_ms": avg(lambda r: r["v1"]["latency_ms"]),
        "latency_now_ms": avg(lambda r: r["latency_ms"]),
        "tokens_in_v1": avg(lambda r: r["v1"]["tokens_in"]),
        "tokens_in_now": avg(lambda r: r["tokens_in"]),
        "tokens_out_v1": avg(lambda r: r["v1"]["tokens_out"]),
        "tokens_out_now": avg(lambda r: r["tokens_out"]),
        "cost_now": sum(r.get("cost_usd") or 0 for r in ok),
        "errors": len(rows) - n,
    }

    print()
    print("=" * 72)
    print("Ponovno pokretanje stvarnih pitanja iz PoC-a")
    print("=" * 72)
    print(f"  Pitanja:                       {n}")
    print(f"  Odbijeno tada (v1):            {old_ref:>3}   {_pct(old_ref, n)}")
    print(f"  Odbijeno sada:                 {new_ref:>3}   {_pct(new_ref, n)}   ← glavna brojka")
    print(f"    sada odgovara, prije nije:   {len(recovered):>3}")
    print(f"    sada odbija, prije nije:     {len(lost):>3}   ← provjeriti svako")
    print()
    for lbl, a, b in (("Prosječno trajanje", m["latency_v1_ms"], m["latency_now_ms"]),
                      ("Ulaznih tokena",     m["tokens_in_v1"], m["tokens_in_now"]),
                      ("Izlaznih tokena",    m["tokens_out_v1"], m["tokens_out_now"])):
        if a and b:
            print(f"  {lbl:<30} {a:>9,.0f} → {b:>9,.0f}   ({(b-a)/a:+.0%})")
    print(f"  {'Trošak ovog pokretanja':<30} ${m['cost_now']:.2f}")

    graded = [r for r in ok if r.get("feedback", {}).get("accuracy_verdict")]
    if graded:
        print(f"\n  Pitanja s ocjenom savjetnika ({len(graded)}):")
        print("  " + "-" * 68)
        for r in graded:
            f = r["feedback"]
            was = "odbio" if r["v1"]["refused"] else "odgovorio"
            now = "odbija" if r.get("refused") else "odgovara"
            flag = "  ←" if r["v1"]["refused"] and not r.get("refused") else ""
            print(f"    {f['accuracy_verdict']:<17} {was:>9} → {now:<9}{flag}  {r['query'][:44]}")
            if f.get("comment"):
                print(f"        ✎ {f['comment'][:88]}")

    if recovered:
        print(f"\n  Sada odgovara na pitanja koja je prije odbio ({len(recovered)}):")
        for r in recovered:
            print(f"    · {r['query'][:88]}")
    if lost:
        print(f"\n  REGRESIJA — sada odbija, prije je odgovarao ({len(lost)}):")
        for r in lost:
            print(f"    · {r['query'][:88]}")

    print("\n  Ovo ne mjeri točnost novih odgovora — stare ocjene se ne prenose.")
    print("  Mjeri odgovara li sustav ondje gdje je prije šutio.")
    return m


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queries", nargs="?", help="CSV export of the `queries` table")
    ap.add_argument("feedback", nargs="?", help="CSV export of the `feedback` table")
    ap.add_argument("--from-db", default="",
                    help="Read the log straight from Postgres instead of CSVs. "
                         "Give a database name (resolved against DATABASE_URL) or a "
                         "full URL. The PoC logs to the database its process started "
                         "with — v1 — so this is usually `rrif_rag`.")
    ap.add_argument("--advisor", default="",
                    help="Only this advisor_id. RRiF's own testing is `rriftest3` "
                         "(27 questions, 25 and 29 May); the rest of the log is ours.")
    ap.add_argument("--only-graded", action="store_true",
                    help="Only questions an advisor actually graded (cheap)")
    ap.add_argument("--dedupe", action="store_true",
                    help="Collapse repeated identical questions")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--note", default="replay_real")
    args = ap.parse_args()

    if args.from_db:
        rows, fb = load_from_db(args.from_db)
        source = args.from_db
    else:
        if not args.queries:
            print("Give a CSV export, or --from-db rrif_rag to read the log directly.")
            return 1
        qpath = Path(args.queries)
        if not qpath.exists():
            print(f"Not found: {qpath}\n"
                  "The CSVs were exported to the laptop; on this machine use "
                  "--from-db rrif_rag instead.")
            return 1
        fb = load_feedback(Path(args.feedback) if args.feedback else None)
        rows = load_queries(qpath)
        source = qpath.name

    if args.advisor:
        rows = [r for r in rows if r.get("advisor_id") == args.advisor]
    if args.only_graded:
        rows = [r for r in rows
                if fb.get(r["query_id"], {}).get("accuracy_verdict")]
    if args.dedupe:
        seen, keep = set(), []
        for r in rows:
            k = re.sub(r"\s+", " ", r["question_text"].strip().lower())
            if k in seen:
                continue
            seen.add(k)
            keep.append(r)
        print(f"  dedupe: {len(rows)} → {len(keep)}")
        rows = keep
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("Nothing to replay.")
        return 1

    print(f"Replaying {len(rows)} real questions "
          f"(~${0.03 * len(rows):.2f}, corpus="
          f"{os.environ.get('DATABASE_URL','?').rsplit('/', 1)[-1]}, "
          f"TOP_K={os.getenv('TOP_K','5')})…")

    out: list[dict] = [None] * len(rows)          # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(replay_one, r, fb.get(r["query_id"])): i
                for i, r in enumerate(rows)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            out[i] = fut.result()
            done += 1
            r = out[i]
            mark = "ERR" if "error" in r else ("odbio" if r.get("refused") else "ok   ")
            print(f"  [{done:>3}/{len(rows)}] {mark:<6} {r['query'][:66]}")

    metrics = report(out)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{ts}_{args.note}.json"
    path.write_text(json.dumps({
        "timestamp": ts, "note": args.note, "skip_generation": False,
        "source_queries": source,
        "metrics": metrics, "per_question": out,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → {path}")
    print("  Za čitanje na papiru:  python -m eval.make_printable " + str(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
