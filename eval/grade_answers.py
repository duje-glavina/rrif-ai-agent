"""Grade generated answers without a gold answer key.

WHY THIS EXISTS
───────────────
Every number produced so far grades RETRIEVAL — did the chunk containing the
answer reach the context window. The acceptance criterion is about ANSWERS.
Those are not the same thing: retrieval can hand the generator the right
paragraph and the generator can still hedge, omit the figure, or attach it to
the wrong source.

The obvious way to measure answers is to compare them against RRiF's own
answers, which we do not have. But a gold answer is only needed for one of the
four things worth knowing, and the other three can be measured today:

  1. GROUNDEDNESS   — is every claim in the answer traceable to a retrieved
                      chunk? Needs the chunks and the answer. No key.
  2. CONTRADICTION  — does the answer state something a retrieved chunk
                      denies? Needs the chunks and the answer. No key.
  3. CITATION       — does the source the answer points at actually contain
                      what is attributed to it? Needs the chunks. No key.
  4. CORRECTNESS    — is the answer right about Croatian tax law? Needs a
                      domain expert. This is what RRiF is for, and what
                      `eval/make_review_sheet.py` prepares.

(1)–(3) are where an unusable RAG system fails loudly, and they are the ones a
non-accountant can act on. (4) is the acceptance criterion and cannot be faked.

WHAT "GROUNDED" DOES NOT MEAN
─────────────────────────────
The judge sees only the retrieved chunks. A fully grounded answer built on a
chunk that was itself wrong, outdated or off-topic scores as grounded. This
measures faithfulness to the corpus, not truth. Do not quote it to RRiF as an
accuracy figure.

USAGE
─────
    python -m eval.grade_answers eval/results/<run>.json
    python -m eval.grade_answers eval/results/<run>.json --workers 8
    python -m eval.grade_answers eval/results/<run>.json --limit 5   # smoke test

Writes <run>.graded.json next to the input and prints a summary.
The input must come from a run WITH generation (not --skip-generation).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Deliberately configurable. If the judge is the same model as the generator
# you are asking it to mark its own homework, which inflates groundedness.
# The script says so at startup rather than quietly letting it happen.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")
GENERATOR_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# Sonnet pricing, USD per million tokens. Update if the judge model changes.
PRICE_IN, PRICE_OUT = 3.0, 15.0

MAX_CHUNK_CHARS = 3000     # per chunk, in the judge prompt


SYSTEM = """Ti si strogi ocjenjivač RAG sustava za hrvatsko računovodstvo i porezno pravo.

Dobit ćeš: pitanje, dohvaćene izvatke iz baze znanja (numerirane), i odgovor koji je sustav generirao.

Tvoj JEDINI zadatak je provjeriti je li odgovor UTEMELJEN na priloženim izvatcima.
NE ocjenjuješ je li odgovor točan po hrvatskom pravu. Ne koristi vlastito znanje o propisima.
Ako izvadak tvrdi nešto pogrešno, a odgovor to vjerno prenosi — to je UTEMELJENO.

Za svaku samostalnu činjeničnu tvrdnju u odgovoru (brojke, stope, rokovi, uvjeti, definicije) odredi:
  - "supported": tvrdnja doslovno ili očito proizlazi iz nekog izvatka
  - "partial":   izvadak govori o tome, ali odgovor dodaje ili zaoštrava ono što izvadak ne kaže
  - "unsupported": ni jedan izvadak to ne pokriva
  - "contradicted": neki izvadak tvrdi suprotno

Opće fraze bez činjeničnog sadržaja ("obratite se savjetniku", "ovisi o okolnostima") preskoči.

NE ocjenjuj imena autora. Izvatci koje dobivaš ne sadrže podatak o autoru, pa ga ni ne možeš
provjeriti — ime autora u odgovoru nikad ne označavaj kao unsupported. Autorstvo se provjerava
zasebno (eval/check_authors.py).

Vrati ISKLJUČIVO JSON, bez ikakvog teksta oko njega:
{
  "claims": [
    {"claim": "<kratko, do 15 riječi>", "status": "supported|partial|unsupported|contradicted", "chunk": <broj izvatka ili null>}
  ],
  "verdict": "grounded|mostly_grounded|ungrounded",
  "has_contradiction": true|false,
  "citations_ok": true|false,
  "note": "<jedna rečenica, hrvatski>"
}

"verdict":
  grounded         — sve činjenične tvrdnje supported
  mostly_grounded  — sve supported ili partial, ništa unsupported/contradicted
  ungrounded       — bar jedna unsupported ili contradicted

"citations_ok": jesu li izvori koje odgovor navodi doista oni izvatci iz kojih tvrdnje dolaze.
Ako odgovor ne navodi nijedan izvor, stavi false."""


def _fmt_chunks(meta: list[dict]) -> str:
    out = []
    for i, m in enumerate(meta, 1):
        src = (m.get("source") or "").strip() or "(bez oznake izvora)"
        txt = (m.get("chunk_text") or "").strip()
        if len(txt) > MAX_CHUNK_CHARS:
            txt = txt[:MAX_CHUNK_CHARS] + " […skraćeno…]"
        out.append(f"--- IZVADAK {i} ---\nizvor: {src}\n{txt}")
    return "\n\n".join(out) if out else "(nema dohvaćenih izvadaka)"


def _parse_json(text: str) -> dict | None:
    """Models occasionally wrap JSON in prose or a fence. Dig it out."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def judge_one(row: dict, client: anthropic.Anthropic, model: str) -> dict:
    """Judge one question. Never raises — a failure is recorded, not fatal."""
    qid = row.get("id", "?")
    answer = row.get("answer") or row.get("answer_preview") or ""
    meta = row.get("retrieved_meta") or []

    base = {
        "id": qid,
        "query": row.get("query", ""),
        "scope": row.get("scope"),
        "refused": row.get("refused"),
        "n_chunks": len(meta),
    }

    if not answer.strip():
        return {**base, "judged": False, "reason": "no answer text in results file"}
    if row.get("refused"):
        # A refusal has nothing to ground. Whether refusing was RIGHT is
        # already graded by refusal_correct in run_eval.
        return {**base, "judged": False, "reason": "refused — nothing to ground"}

    user = (
        f"PITANJE:\n{row.get('query','')}\n\n"
        f"DOHVAĆENI IZVATCI:\n{_fmt_chunks(meta)}\n\n"
        f"ODGOVOR SUSTAVA:\n{answer}"
    )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1500,
            temperature=0,          # a grader that moves between runs is not a grader
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as exc:                       # noqa: BLE001 — report, don't crash the run
        return {**base, "judged": False, "reason": f"API error: {exc}"}

    raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    parsed = _parse_json(raw)
    if not parsed:
        return {**base, "judged": False, "reason": "judge returned unparseable JSON",
                "raw": raw[:400]}

    claims = parsed.get("claims") or []
    counts = {k: sum(1 for c in claims if c.get("status") == k)
              for k in ("supported", "partial", "unsupported", "contradicted")}

    cost = round(
        resp.usage.input_tokens / 1e6 * PRICE_IN
        + resp.usage.output_tokens / 1e6 * PRICE_OUT, 6)

    return {
        **base,
        "judged": True,
        "verdict": parsed.get("verdict"),
        "has_contradiction": bool(parsed.get("has_contradiction")),
        "citations_ok": bool(parsed.get("citations_ok")),
        "note": parsed.get("note", ""),
        "n_claims": len(claims),
        **{f"n_{k}": v for k, v in counts.items()},
        "claims": claims,
        "judge_cost_usd": cost,
    }


def summarise(graded: list[dict]) -> dict:
    ok = [g for g in graded if g.get("judged")]
    n = len(ok)
    if not n:
        return {"n_judged": 0}

    def rate(pred) -> float:
        return sum(1 for g in ok if pred(g)) / n

    total_claims = sum(g.get("n_claims", 0) for g in ok)
    return {
        "n_judged": n,
        "n_skipped": len(graded) - n,
        "grounded_rate": rate(lambda g: g.get("verdict") == "grounded"),
        "grounded_or_mostly_rate": rate(
            lambda g: g.get("verdict") in ("grounded", "mostly_grounded")),
        "ungrounded_rate": rate(lambda g: g.get("verdict") == "ungrounded"),
        "contradiction_rate": rate(lambda g: g.get("has_contradiction")),
        "citations_ok_rate": rate(lambda g: g.get("citations_ok")),
        "n_claims_total": total_claims,
        "claim_support_rate": (
            sum(g.get("n_supported", 0) for g in ok) / total_claims
            if total_claims else None),
        "judge_cost_usd": round(sum(g.get("judge_cost_usd", 0) or 0 for g in ok), 4),
        "judge_model": JUDGE_MODEL,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="eval/results/<run>.json from a generation run")
    ap.add_argument("--model", default=JUDGE_MODEL, help=f"Judge model (default {JUDGE_MODEL})")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="Judge only the first N (smoke test)")
    ap.add_argument("--include-zakoni", action="store_true",
                    help="Also judge the PDV-law questions (out of F1 scope by default)")
    args = ap.parse_args()

    src = Path(args.results)
    if not src.exists():
        print(f"Not found: {src}")
        return 1

    data = json.loads(src.read_text(encoding="utf-8"))
    rows = data.get("per_question", [])

    if data.get("skip_generation"):
        print("This results file was produced with --skip-generation. There are no\n"
              "answers in it to grade. Re-run without that flag first.")
        return 1

    rows = [r for r in rows if r.get("in_corpus")]
    if not args.include_zakoni:
        rows = [r for r in rows if r.get("scope") != "zakon"]
    if args.limit:
        rows = rows[:args.limit]

    if not rows:
        print("Nothing to grade.")
        return 1

    if args.model == GENERATOR_MODEL:
        print(f"  ! Judge and generator are both {args.model}. A model marking its own\n"
              f"    homework grades itself generously. Treat groundedness as a floor,\n"
              f"    and set JUDGE_MODEL to something else for a number you can quote.\n")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    graded: list[dict] = [None] * len(rows)          # type: ignore[list-item]

    print(f"Judging {len(rows)} answers with {args.model}…")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge_one, r, client, args.model): i
                   for i, r in enumerate(rows)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            graded[i] = fut.result()
            done += 1
            v = graded[i].get("verdict") or graded[i].get("reason", "?")
            print(f"  [{done:>2}/{len(rows)}] {graded[i]['id']:<12} {v}")

    metrics = summarise(graded)

    print()
    print("=" * 70)
    print("Utemeljenost odgovora (nije ocjena točnosti — vidi docstring)")
    print("=" * 70)
    if not metrics["n_judged"]:
        print("  Nijedan odgovor nije ocijenjen.")
    else:
        n = metrics["n_judged"]
        print(f"  Potpuno utemeljeno:            {metrics['grounded_rate']:.1%}  (n={n})")
        print(f"  Utemeljeno ili uglavnom:       {metrics['grounded_or_mostly_rate']:.1%}")
        print(f"  NEUTEMELJENO:                  {metrics['ungrounded_rate']:.1%}   ← ovo je izmišljanje")
        print(f"  Proturječi izvatku:            {metrics['contradiction_rate']:.1%}   ← ovo je najgore")
        print(f"  Izvori ispravno pripisani:     {metrics['citations_ok_rate']:.1%}")
        if metrics["claim_support_rate"] is not None:
            print(f"  Udio potkrijepljenih tvrdnji:  {metrics['claim_support_rate']:.1%}"
                  f"  ({metrics['n_claims_total']} tvrdnji ukupno)")
        print(f"  Preskočeno:                    {metrics['n_skipped']}")
        print(f"  Trošak ocjenjivanja:           ${metrics['judge_cost_usd']:.4f}")

    bad = [g for g in graded
           if g.get("judged") and (g.get("verdict") == "ungrounded"
                                   or g.get("has_contradiction"))]
    if bad:
        print(f"\n  Za ručnu provjeru ({len(bad)}):")
        for g in bad:
            print(f"    {g['id']:<12} {g.get('note','')[:80]}")
            for c in g.get("claims", []):
                if c.get("status") in ("unsupported", "contradicted"):
                    print(f"        [{c['status']}] {c.get('claim','')[:70]}")

    out = src.with_suffix(".graded.json")
    out.write_text(json.dumps(
        {"source_run": src.name, "judge_model": args.model,
         "metrics": metrics, "per_question": graded},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → {out}")

    # Judging is cheap and fast; verifying the judge is neither, so make the
    # reminder unavoidable rather than a line in a README.
    print("\n  Prije nego ovu brojku ikome citiraš: pročitaj 5 ocjena rukom.\n"
          "  Ako se ne slažeš s ocjenjivačem, brojka ne vrijedi ništa.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
