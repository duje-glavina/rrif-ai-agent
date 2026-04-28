"""Run the golden evaluation set against the full pipeline (retrieval + Claude).

Grades each question on:
  - retrieval: did the expected article(s) appear in top-k after rerank
  - refusal: for traps, did Claude refuse correctly (and not refuse for real questions)
  - keyword presence: did the answer mention expected key terms (best-effort signal)

Usage from project root:
    python eval/run_eval.py
    python eval/run_eval.py --note "after-prompt-tweak"
    python eval/run_eval.py --skip-generation     # retrieval-only, no API spend
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

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

    Examples:
        'čl. 35. st. 1. Zakona o PDV-u'   -> '35'
        '38. st. 3. t. a) Zakona o PDV-u'  -> '38'
        'čl. 2'                             -> '2'
        '38'                                -> '38'
        None                                -> None
    """
    if not raw:
        return None
    s = raw.strip()
    s = re.sub(r'^č(l|lan)\.?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^art\.?\s*', '', s, flags=re.IGNORECASE)
    s = s.strip()
    m = re.match(r'^(\d+)', s)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_golden_set() -> list[dict]:
    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _keyword_hits(answer_text: str, keywords: list[str]) -> tuple[int, int]:
    """Count how many expected keywords appear in the answer (case-insensitive)."""
    if not keywords:
        return 0, 0
    text = answer_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return hits, len(keywords)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_one(item: dict, *, skip_generation: bool) -> dict:
    """Run one question through the full pipeline and grade the result."""
    result = ask(item["query"])

    # Normalize article numbers from citations for comparison
    top_articles = [
        _extract_article_number(c.article_number)
        for c in result.citations
        if c.article_number
    ]
    top_articles = [a for a in top_articles if a]  # drop unparseable

    top_1_article = top_articles[0] if top_articles else None
    expected = item["expected_articles"]
    max_score = 1.0 if result.citations else 0.0

    # Retrieval grading
    if item["in_corpus"]:
        retrieval_top_1 = top_1_article in expected
        retrieval_top_5 = any(art in expected for art in top_articles)
    else:
        retrieval_top_1 = False
        retrieval_top_5 = False

    # Generation grading
    refusal_correct = None
    keyword_hits = None
    keyword_total = None
    answer_text = ""
    citations_raw: list[dict] = []
    cost = None

    if not skip_generation:
        answer_text = result.answer
        citations_raw = [
            {
                "source": c.source,
                "article_number": c.article_number,
                "excerpt": c.excerpt,
            }
            for c in result.citations
        ]
        cost = round(
            (result.tokens_in / 1_000_000) * 3.0
            + (result.tokens_out / 1_000_000) * 15.0,
            6,
        ) if (result.tokens_in or result.tokens_out) else None

        if item["in_corpus"]:
            refusal_correct = not result.referred_to_advisor
            hits, total = _keyword_hits(answer_text, item.get("expected_keywords", []))
            keyword_hits, keyword_total = hits, total
        else:
            refusal_correct = result.referred_to_advisor

    if skip_generation:
        passed = retrieval_top_1 if item["in_corpus"] else (not result.citations)
    else:
        if item["in_corpus"]:
            passed = retrieval_top_1 and refusal_correct
        else:
            passed = refusal_correct

    return {
        "id": item["id"],
        "query": item["query"],
        "in_corpus": item["in_corpus"],
        "expected_articles": expected,
        "top_articles": top_articles,
        "retrieval_top_1": retrieval_top_1,
        "retrieval_top_5": retrieval_top_5,
        "max_rerank_score": max_score,
        "refused": result.referred_to_advisor if not skip_generation else None,
        "refusal_correct": refusal_correct,
        "answer_preview": answer_text[:300] if answer_text else "",
        "n_citations": len(citations_raw),
        "keyword_hits": keyword_hits,
        "keyword_total": keyword_total,
        "passed": passed,
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
    has_generation = any(r["refusal_correct"] is not None for r in per_question)

    metrics = {
        "n_questions": n,
        "n_in_corpus": len(in_corpus),
        "n_traps": len(traps),
        "in_corpus_top_1_accuracy": (
            sum(r["retrieval_top_1"] for r in in_corpus) / len(in_corpus)
            if in_corpus else None
        ),
        "in_corpus_top_5_hit_rate": (
            sum(r["retrieval_top_5"] for r in in_corpus) / len(in_corpus)
            if in_corpus else None
        ),
        "overall_pass_rate": sum(r["passed"] for r in per_question) / n,
        "avg_latency_ms": sum(r["latency_ms"] for r in per_question) / n,
        "avg_max_rerank_score_in_corpus": (
            sum(r["max_rerank_score"] for r in in_corpus) / len(in_corpus)
            if in_corpus else None
        ),
    }

    if has_generation:
        metrics["in_corpus_no_false_refusal_rate"] = (
            sum(1 for r in in_corpus if r["refusal_correct"]) / len(in_corpus)
            if in_corpus else None
        )
        metrics["trap_refusal_rate"] = (
            sum(1 for r in traps if r["refusal_correct"]) / len(traps)
            if traps else None
        )
        costs = [r["cost_usd"] for r in per_question if r["cost_usd"] is not None]
        metrics["avg_cost_usd"] = sum(costs) / len(costs) if costs else None
        metrics["total_cost_usd"] = sum(costs) if costs else None

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="", help="Free-form label for the results filename")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Classify + retrieve only, skip answer generation (saves API cost)")
    args = parser.parse_args()

    golden = load_golden_set()
    print(f"Loaded {len(golden)} questions from {GOLDEN_SET_PATH}\n")

    per_question = []
    for i, item in enumerate(golden, start=1):
        print(f"[{i}/{len(golden)}] {item['id']}: {item['query']}")
        result = evaluate_one(item, skip_generation=args.skip_generation)
        per_question.append(result)
        verdict = "✅ pass" if result["passed"] else "❌ fail"
        bits = [
            f"top: {result['top_articles'][:3]}",
            f"score={result['max_rerank_score']:.3f}",
        ]
        if result["refused"] is not None:
            bits.append(f"refused={result['refused']}")
        if result["cost_usd"] is not None:
            bits.append(f"${result['cost_usd']:.5f}")
        print(f"    {verdict}  {'  '.join(bits)}")
        if result["answer_preview"]:
            print(f"    ↳ {result['answer_preview'][:200]}")
        print()

    metrics = aggregate(per_question)

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Top-1 retrieval accuracy:      {metrics['in_corpus_top_1_accuracy']:.1%}")
    print(f"  Top-5 retrieval hit rate:      {metrics['in_corpus_top_5_hit_rate']:.1%}")
    if "trap_refusal_rate" in metrics:
        print(f"  Trap refusal rate:             {metrics['trap_refusal_rate']:.1%}")
        print(f"  No-false-refusal rate (real):  {metrics['in_corpus_no_false_refusal_rate']:.1%}")
    print(f"  Overall pass rate:             {metrics['overall_pass_rate']:.1%}")
    print(f"  Avg latency:                   {metrics['avg_latency_ms']:.0f} ms")
    print(f"  Avg max rerank score:          {metrics['avg_max_rerank_score_in_corpus']:.4f}")
    if metrics.get("total_cost_usd") is not None:
        print(f"  Total cost (this run):         ${metrics['total_cost_usd']:.5f}")
        print(f"  Avg cost per question:         ${metrics['avg_cost_usd']:.5f}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    label_parts = [args.note] if args.note else []
    if args.skip_generation:
        label_parts.append("retrieval_only")
    label = ("_" + "_".join(label_parts)) if label_parts else ""
    out_path = RESULTS_DIR / f"{timestamp}{label}.json"
    out_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "note": args.note,
                "skip_generation": args.skip_generation,
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
