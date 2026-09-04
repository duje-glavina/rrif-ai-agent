"""Run the golden evaluation set against the pipeline.

Grades each question on:
  - retrieval: did the expected article(s) appear in the reranked chunks
  - refusal:   for traps, did Claude refuse correctly (needs generation)
  - keywords:  did the answer mention expected key terms (needs generation)

Usage from project root:
    python -m eval.run_eval
    python -m eval.run_eval --note "after-prompt-tweak"
    python -m eval.run_eval --skip-generation      # retrieval only, no Sonnet
    RETRIEVAL_MODE=wide python -m eval.run_eval --skip-generation --note ablation


WHAT CHANGED, AND WHY IT MATTERS
────────────────────────────────
The previous version reported "Top-1 retrieval accuracy" with a hard ceiling
of 16.2%, and nobody noticed because the number looked plausibly bad.

Three separate defects:

1. The denominator was every in_corpus question (37 of 41). But only SIX of
   those have a non-empty `expected_articles` — the PDV law questions. For
   the other 31, `expected_articles` is [], and `top_article in []` is always
   False. So 31 questions were scored as retrieval failures by construction
   while still counting in the denominator. 6/37 = 16.2% was the maximum
   achievable score.

2. The articles being graded came from `result.citations` — what the LLM
   chose to cite — not from what retrieval returned. That conflates
   retrieval, reranking, the generator's citation choices, citation string
   formatting, and the regex parser into one number labelled "retrieval".

3. `--skip-generation` did not skip generation. `ask()` was called without
   any such argument, so Sonnet ran on every question regardless. The flag
   only suppressed *grading* of the output. Retrieval experiments were
   neither free nor fast.

This version:
  - grades retrieval on `result.retrieved_meta` (the reranked chunks)
  - restricts the retrieval denominator to questions that can actually be
    graded, and prints n alongside the percentage
  - keeps the old citation-based number as a clearly-labelled secondary
    metric, so the two can be compared
  - passes skip_generation through, and marks ungradeable questions as
    None rather than silently passing them
  - reports the real cross-encoder score instead of `1.0 if citations`

Read any percentage below together with its n. On six questions, one
question is 16.7 points.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.query import ask


GOLDEN_SET_PATH = Path("eval/golden_set.yaml")
RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Article number normalizer
# ---------------------------------------------------------------------------
def _extract_article_number(raw: str | None) -> str | None:
    """Extract the base article number from a citation string.

        'čl. 35. st. 1. Zakona o PDV-u'   -> '35'
        'članak 38'                       -> '38'
        '38a'                             -> '38a'
        None / ''                         -> None

    CAVEAT: this takes the first digit run it finds, so a magazine citation
    like 'RRiF 12/2024' yields '12', and a bare year field yields '2024'.
    That is why `article_number` shows years for magazine chunks in the
    output. It is harmless for the six law questions we actually grade, but
    do not build anything new on this function without fixing it first.
    """
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    s = re.sub(
        r'^(čl(an(ak)?)?\.?|cl(an(ak)?)?\.?|art(icle)?\.?)\s*',
        '',
        s,
        flags=re.IGNORECASE,
    )
    s = s.strip()
    m = re.search(r'(\d+[a-z]?)', s)
    return m.group(1) if m else None


def _looks_like_year(a: str | None) -> bool:
    return bool(a) and bool(re.fullmatch(r'(19|20)\d{2}', a))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_golden_set() -> list[dict]:
    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _keyword_hits(answer_text: str, keywords: list[str]) -> tuple[int, int]:
    if not keywords:
        return 0, 0
    text = answer_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return hits, len(keywords)


def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_one(item: dict, *, skip_generation: bool, enable_rewrite: bool) -> dict:
    """Run one question through the pipeline and grade the result."""
    result = ask(
        item["query"],
        enable_rewrite=enable_rewrite,
        skip_generation=skip_generation,
    )

    expected_cat = item.get("expected_category")
    actual_cat = result.classifier.category
    category_correct = expected_cat is not None and actual_cat == expected_cat

    expected = item["expected_articles"]
    in_corpus = item["in_corpus"]
    # A question can only be graded on article-level retrieval if we know
    # which article is right. 31 of 41 do not, and pretending otherwise is
    # what produced the 16.2% ceiling.
    gradeable_retrieval = bool(in_corpus and expected)

    # ── Retrieval, graded on what retrieval returned ────────────────────────
    retrieved_articles = [
        _extract_article_number(m.get("article_number"))
        for m in result.retrieved_meta
    ]
    retrieved_articles = [a for a in retrieved_articles if a]

    if gradeable_retrieval:
        retrieval_top_1 = bool(retrieved_articles) and retrieved_articles[0] in expected
        retrieval_top_k = any(a in expected for a in retrieved_articles)
    else:
        retrieval_top_1 = None
        retrieval_top_k = None

    # ── Same thing graded on citations, kept for comparison ─────────────────
    citation_articles = [
        _extract_article_number(c.article_number)
        for c in result.citations
        if c.article_number
    ]
    citation_articles = [a for a in citation_articles if a]
    if gradeable_retrieval and not skip_generation:
        citation_top_1 = bool(citation_articles) and citation_articles[0] in expected
        citation_top_k = any(a in expected for a in citation_articles)
    else:
        citation_top_1 = None
        citation_top_k = None

    top_score = (
        result.retrieved_meta[0].get("rerank_score")
        if result.retrieved_meta else None
    )

    # ── Generation grading ──────────────────────────────────────────────────
    refusal_correct = None
    keyword_hits = keyword_total = None
    answer_text = ""
    citations_raw: list[dict] = []
    cost = None

    if not skip_generation:
        answer_text = result.answer
        citations_raw = [
            {"source": c.source, "article_number": c.article_number, "excerpt": c.excerpt}
            for c in result.citations
        ]
        cost = round(
            (result.tokens_in / 1_000_000) * 3.0
            + (result.tokens_out / 1_000_000) * 15.0,
            6,
        ) if (result.tokens_in or result.tokens_out) else None

        if in_corpus:
            refusal_correct = not result.referred_to_advisor
            keyword_hits, keyword_total = _keyword_hits(
                answer_text, item.get("expected_keywords", [])
            )
        else:
            refusal_correct = result.referred_to_advisor

    # ── Pass / fail ─────────────────────────────────────────────────────────
    # None means "this run cannot judge this question", which is different
    # from failing it. Ungradeable questions are excluded from the rate
    # rather than silently counted as passes.
    if skip_generation:
        passed = retrieval_top_1 if gradeable_retrieval else None
    else:
        if not in_corpus:
            passed = refusal_correct
        elif expected:
            passed = bool(retrieval_top_1) and bool(refusal_correct)
        elif item.get("expected_keywords"):
            has_kw = (
                keyword_total is not None
                and keyword_total > 0
                and (keyword_hits / keyword_total) >= 0.5
            )
            passed = bool(refusal_correct) and has_kw
        else:
            passed = refusal_correct

    return {
        "id": item["id"],
        "query": item["query"],
        "in_corpus": in_corpus,
        "gradeable_retrieval": gradeable_retrieval,
        "expected_articles": expected,
        "retrieved_articles": retrieved_articles,
        "retrieved_meta": result.retrieved_meta,
        "retrieval_top_1": retrieval_top_1,
        "retrieval_top_k": retrieval_top_k,
        "citation_articles": citation_articles,
        "citation_top_1": citation_top_1,
        "citation_top_k": citation_top_k,
        "top_rerank_score": top_score,
        "n_year_like_articles": sum(1 for a in retrieved_articles if _looks_like_year(a)),
        "refused": result.referred_to_advisor if not skip_generation else None,
        "expected_category": expected_cat,
        "actual_category": actual_cat,
        "category_correct": category_correct,
        "refusal_correct": refusal_correct,
        "answer_preview": answer_text[:300] if answer_text else "",
        "n_citations": len(citations_raw),
        "keyword_hits": keyword_hits,
        "keyword_total": keyword_total,
        "passed": passed,
        "original_query": result.original_query,
        "rewritten_query": result.rewritten_query,
        "rewrite_changed": result.rewrite_changed,
        "latency_ms": result.latency_ms,
        "cost_usd": cost,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(per_question: list[dict]) -> dict:
    n = len(per_question)
    in_corpus = [r for r in per_question if r["in_corpus"]]
    traps = [r for r in per_question if not r["in_corpus"]]
    gradeable = [r for r in per_question if r["gradeable_retrieval"]]
    judged = [r for r in per_question if r["passed"] is not None]
    has_generation = any(r["refusal_correct"] is not None for r in per_question)

    metrics = {
        "n_questions": n,
        "n_in_corpus": len(in_corpus),
        "n_traps": len(traps),
        # The honest denominator: questions where we know which article is
        # correct. Everything else cannot be graded this way at all.
        "n_gradeable_retrieval": len(gradeable),
        "retrieval_top_1": _mean([r["retrieval_top_1"] for r in gradeable]),
        "retrieval_top_k": _mean([r["retrieval_top_k"] for r in gradeable]),
        # Old-style number, for comparison with historical runs.
        "citation_top_1": _mean([r["citation_top_1"] for r in gradeable]),
        "citation_top_k": _mean([r["citation_top_k"] for r in gradeable]),
        "n_judged": len(judged),
        "overall_pass_rate": _mean([r["passed"] for r in judged]),
        "avg_latency_ms": _mean([r["latency_ms"] for r in per_question]),
        "avg_top_rerank_score": _mean([r["top_rerank_score"] for r in per_question]),
        "n_year_like_articles": sum(r["n_year_like_articles"] for r in per_question),
        "retrieval_mode": os.getenv("RETRIEVAL_MODE", "tight"),
    }

    in_corpus_with_cat = [
        r for r in per_question
        if r["in_corpus"] and r["expected_category"] is not None
    ]
    if in_corpus_with_cat:
        metrics["classifier_accuracy"] = _mean(
            [r["category_correct"] for r in in_corpus_with_cat]
        )

    if has_generation:
        metrics["in_corpus_no_false_refusal_rate"] = _mean(
            [r["refusal_correct"] for r in in_corpus]
        )
        metrics["trap_refusal_rate"] = _mean([r["refusal_correct"] for r in traps])
        costs = [r["cost_usd"] for r in per_question if r["cost_usd"] is not None]
        metrics["avg_cost_usd"] = sum(costs) / len(costs) if costs else None
        metrics["total_cost_usd"] = sum(costs) if costs else None

    rewrites_changed = [r for r in per_question if r.get("rewrite_changed")]
    metrics["n_rewrites_changed"] = len(rewrites_changed)
    metrics["rewrite_change_rate"] = len(rewrites_changed) / n if n else None

    return metrics


def _pct(v, n=None) -> str:
    if v is None:
        return "n/a"
    s = f"{v:.1%}"
    return f"{s}  (n={n})" if n is not None else s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="", help="Free-form label for the results filename")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Classify + retrieve + rerank only. Really skips Sonnet.")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers")
    parser.add_argument("--rewrite", action="store_true",
                        help="Enable Haiku query rewriter before classification")
    args = parser.parse_args()

    golden = load_golden_set()
    mode = os.getenv("RETRIEVAL_MODE", "tight")
    print(f"Loaded {len(golden)} questions from {GOLDEN_SET_PATH}")
    print(f"Retrieval mode: {mode}"
          f"{'  |  generation skipped' if args.skip_generation else ''}\n")
    print("Warming up models...")
    from rag.embedder import _get_model as _warm_embedder
    from rag.retrieve.rerank import _get_model as _warm_reranker
    _warm_embedder()
    _warm_reranker()
    print("Models ready.\n")

    per_question = [None] * len(golden)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_one,
                item,
                skip_generation=args.skip_generation,
                enable_rewrite=args.rewrite,
            ): i
            for i, item in enumerate(golden)
        }
        for future in as_completed(futures):
            i = futures[future]
            item = golden[i]
            result = future.result()
            per_question[i] = result

            if result["passed"] is None:
                verdict = "—  n/a "
            elif result["passed"]:
                verdict = "✅ pass"
            else:
                verdict = "❌ fail"

            bits = [f"cat: {result.get('actual_category')}"]
            if result["gradeable_retrieval"]:
                bits.append(f"want {result['expected_articles']}")
                bits.append(f"got {result['retrieved_articles'][:3]}")
            else:
                bits.append(f"got {result['retrieved_articles'][:3]} (not graded)")
            if result["top_rerank_score"] is not None:
                bits.append(f"score={result['top_rerank_score']:.3f}")
            if result.get("expected_category") and not result.get("category_correct"):
                bits.append(f"⚠ expected_cat={result['expected_category']}")
            if result["refused"] is not None:
                bits.append(f"refused={result['refused']}")
            if result["cost_usd"] is not None:
                bits.append(f"${result['cost_usd']:.5f}")

            print(f"[{i+1}/{len(golden)}] {item['id']}: {item['query']}")
            if result.get("rewrite_changed"):
                print(f"    ✏  rewritten: {result['rewritten_query']}")
            print(f"    {verdict}  {'  '.join(bits)}")
            if result["answer_preview"]:
                print(f"    ↳ {result['answer_preview'][:200]}")
            print()

    metrics = aggregate(per_question)
    ng = metrics["n_gradeable_retrieval"]

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Retrieval top-1 (reranked):    {_pct(metrics['retrieval_top_1'], ng)}")
    print(f"  Retrieval top-{5} (reranked):    {_pct(metrics['retrieval_top_k'], ng)}")
    if metrics["citation_top_1"] is not None:
        print(f"  — via citations (old metric):  {_pct(metrics['citation_top_1'], ng)}")
    if "classifier_accuracy" in metrics:
        print(f"  Classifier accuracy:           {_pct(metrics['classifier_accuracy'])}")
    if metrics.get("trap_refusal_rate") is not None:
        print(f"  Trap refusal rate:             {_pct(metrics['trap_refusal_rate'], metrics['n_traps'])}")
        print(f"  No-false-refusal (in corpus):  {_pct(metrics['in_corpus_no_false_refusal_rate'], metrics['n_in_corpus'])}")
    print(f"  Overall pass rate:             {_pct(metrics['overall_pass_rate'], metrics['n_judged'])}")
    print(f"  Avg latency:                   {metrics['avg_latency_ms']:.0f} ms")
    if metrics["avg_top_rerank_score"] is not None:
        print(f"  Avg top rerank score:          {metrics['avg_top_rerank_score']:.4f}")
    if metrics.get("total_cost_usd") is not None:
        print(f"  Total cost (this run):         ${metrics['total_cost_usd']:.5f}")

    print()
    print(f"  Note: {ng} of {metrics['n_questions']} questions have expected_articles")
    print(f"        and can be graded on retrieval. One question = "
          f"{100/ng:.1f} points." if ng else "")
    if metrics["n_year_like_articles"]:
        print(f"  Note: {metrics['n_year_like_articles']} retrieved article_number values "
              f"look like years — magazine chunks storing a publication year "
              f"in the article field.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    label_parts = [args.note] if args.note else []
    if args.skip_generation:
        label_parts.append("retrieval_only")
    if args.rewrite:
        label_parts.append("rewrite_on")
    if mode != "tight":
        label_parts.append(f"mode_{mode}")
    label = ("_" + "_".join(label_parts)) if label_parts else ""
    out_path = RESULTS_DIR / f"{timestamp}{label}.json"
    out_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "note": args.note,
                "skip_generation": args.skip_generation,
                "rewrite_enabled": args.rewrite,
                "retrieval_mode": mode,
                "metrics": metrics,
                "per_question": per_question,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
