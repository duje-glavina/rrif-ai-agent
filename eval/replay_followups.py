"""Prototype: answer the follow-up questions the PoC dropped.

THE PROBLEM, IN RRIF'S OWN WORDS
────────────────────────────────
    "pitanje je postavljeno u istom razgovoru... trebao bi znati kontekst na
     temelju prethodnog pitanja... izgleda da ta funkcionalnost ovdje nije
     omogućena"

Four of RRiF's nine refusals on 29 May were follow-ups: "Treba li ga
primijeniti…", "Daj mi primjer te detaljnije analize F. Plišića", "Iz
prethodnog pitanja koje sam postavio", "Iz članka vezanog za reprezentaciju u
ugostiteljstvu iz 2024. godine". Look at the timestamps — 13:03, 13:04, 13:04,
13:05. That is an advisor drilling into a topic, which is how advisors work,
and the product dropped every one.

Rebuilding the corpus did not help and could not: `ask()` takes one string, and
"ga" carries the entire meaning of the question while appearing nowhere in the
index. The fix is not to pass history to the generator — by the time the
generator runs, retrieval has already happened on the wrong text. The follow-up
has to become standalone BEFORE classification.

WHAT THIS IS AND IS NOT
───────────────────────
This is a prototype, not a feature. It reconstructs conversations from the
`queries` log by time gaps (there is no conversation_id column yet — that, plus
API and frontend threading, is the actual work), condenses each follow-up
against its predecessors, and asks the condensed question.

It exists to turn "we should add conversations" into "here are the four
questions you asked that we dropped, and here they are answered" — which is a
different conversation to have about scope.

USAGE
─────
    python -m eval.replay_followups --from-db rrif_rag --advisor rriftest3
    python -m eval.replay_followups --from-db rrif_rag --advisor rriftest3 --all
    python -m eval.replay_followups --from-db rrif_rag --gap 30

By default only follow-ups the PoC refused are replayed — those are the ones
that demonstrate anything, and it keeps the run under a euro.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.replay_real import load_from_db, load_queries, load_feedback, _b   # noqa: E402
from rag.query import ask                                                    # noqa: E402
from rag.rewrite.rewriter import condense                                    # noqa: E402

RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _ts(v) -> datetime | None:
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except (TypeError, ValueError):
        return None


def build_conversations(rows: list[dict], gap_minutes: int) -> list[list[dict]]:
    """Reconstruct conversations from timestamps.

    There is no conversation_id in `queries`, so a conversation is defined as
    consecutive questions from one advisor with no gap longer than
    `gap_minutes`. Crude, and it will occasionally join two unrelated threads
    or split one — but the four cases we care about are minutes apart, and
    adding the column properly is part of the work this prototype argues for.
    """
    by_advisor: dict[str, list[dict]] = {}
    for r in rows:
        by_advisor.setdefault(r.get("advisor_id") or "?", []).append(r)

    convs: list[list[dict]] = []
    for advisor, rs in by_advisor.items():
        rs = [r for r in rs if _ts(r.get("created_at"))]
        rs.sort(key=lambda r: _ts(r["created_at"]))
        cur: list[dict] = []
        prev: datetime | None = None
        for r in rs:
            t = _ts(r["created_at"])
            if prev and (t - prev).total_seconds() > gap_minutes * 60:
                if cur:
                    convs.append(cur)
                cur = []
            cur.append(r)
            prev = t
        if cur:
            convs.append(cur)
    return convs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-db", default="",
                    help="Database holding the PoC log (usually rrif_rag)")
    ap.add_argument("queries", nargs="?", help="or a CSV export of `queries`")
    ap.add_argument("feedback", nargs="?", help="and of `feedback`")
    ap.add_argument("--advisor", default="", help="Restrict to one advisor_id")
    ap.add_argument("--gap", type=int, default=20,
                    help="Minutes of silence that end a conversation (default 20)")
    ap.add_argument("--all", action="store_true",
                    help="Replay every follow-up, not only the ones that were refused")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.from_db:
        rows, fb = load_from_db(args.from_db)
    elif args.queries:
        rows = load_queries(Path(args.queries))
        fb = load_feedback(Path(args.feedback) if args.feedback else None)
    else:
        print("Give --from-db rrif_rag, or a CSV export.")
        return 1

    if args.advisor:
        rows = [r for r in rows if r.get("advisor_id") == args.advisor]

    convs = build_conversations(rows, args.gap)
    convs = [c for c in convs if len(c) > 1]
    print(f"  {len(convs)} razgovora s više od jednog pitanja "
          f"(prekid > {args.gap} min)")

    # A follow-up is any question with at least one predecessor in its
    # conversation. Whether it actually NEEDS the context is decided by the
    # condenser — if it rewrites the question, it needed it.
    targets: list[tuple[list[dict], dict]] = []
    for conv in convs:
        for i, r in enumerate(conv):
            if i == 0:
                continue
            if not args.all and not _b(r.get("referred_to_advisor")):
                continue
            targets.append((conv[:i], r))
    if args.limit:
        targets = targets[:args.limit]

    if not targets:
        print("Nema pitanja koja zadovoljavaju uvjet. Probaj --all.")
        return 1

    print(f"  {len(targets)} pitanja za ponavljanje "
          f"({'sva nastavna' if args.all else 'samo ona koja je PoC odbio'})\n")

    out: list[dict] = []
    for n, (history, r) in enumerate(targets, 1):
        q = (r.get("question_text") or "").strip()
        hist = [((h.get("question_text") or ""), (h.get("answer_text") or ""))
                for h in history]

        c = condense(hist, q)
        print(f"[{n}/{len(targets)}] {q[:72]}")
        print(f"          ↳ {c.standalone[:100]}")

        res = ask(c.standalone)
        mark = "ODBIJA" if res.referred_to_advisor else "odgovara"
        was = "odbio" if _b(r.get("referred_to_advisor")) else "odgovorio"
        print(f"          PoC {was}  →  sada {mark}"
              f"   ({res.latency_ms} ms)\n")

        out.append({
            "id": str(r.get("query_id", ""))[:8],
            "query": q,
            "standalone": c.standalone,
            "condense_changed": c.changed,
            "n_history_turns": c.n_turns,
            "history": [h[0] for h in hist],
            "in_corpus": True,
            "scope": "članak",
            "answer": res.answer,
            "refused": res.referred_to_advisor,
            "citations_raw": [{"source": x.source, "article_number": x.article_number,
                               "excerpt": x.excerpt} for x in res.citations],
            "n_citations": len(res.citations),
            "retrieved_meta": res.retrieved_meta,
            "latency_ms": res.latency_ms,
            "timings_ms": getattr(res, "timings", {}) or {},
            "v1": {"refused": _b(r.get("referred_to_advisor")),
                   "answer": r.get("answer_text") or "",
                   "asked_at": str(r.get("created_at"))},
            "feedback": fb.get(str(r.get("query_id")), {}) or None,
        })

    recovered = [o for o in out if o["v1"]["refused"] and not o["refused"]]
    rewritten = [o for o in out if o["condense_changed"]]

    print("=" * 72)
    print("Nastavna pitanja u razgovoru — prototip")
    print("=" * 72)
    print(f"  Ponovljeno:                     {len(out)}")
    print(f"  Condenser preoblikovao pitanje: {len(rewritten)}")
    print(f"  PoC odbio → sada odgovara:      {len(recovered)}")
    if recovered:
        print("\n  Oporavljeno:")
        for o in recovered:
            print(f"    · {o['query'][:66]}")
            print(f"      → {o['standalone'][:88]}")
    still = [o for o in out if o["refused"]]
    if still:
        print(f"\n  I dalje odbija ({len(still)}):")
        for o in still:
            print(f"    · {o['query'][:66]}")
            print(f"      → {o['standalone'][:88]}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{ts}_followups.json"
    path.write_text(json.dumps(
        {"timestamp": ts, "note": "followups", "skip_generation": False,
         "gap_minutes": args.gap, "per_question": out},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → {path}")
    print("  Za čitanje na papiru:  python -m eval.make_printable " + str(path))
    print("\n  Prototip: razgovori su rekonstruirani po vremenskim razmacima, jer")
    print("  queries tablica još nema conversation_id. To je dio posla, ne dokaz")
    print("  da posao nije potreban.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
