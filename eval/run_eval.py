"""Run the golden evaluation set against the current retrieval pipeline.

For each question:
  - Run query through retrieval (using rag.retrieve.runner)
  - Check whether expected article appears in top_k (top-1 and top-5 hit)
  - Capture rerank score, latency

For the run as a whole:
  - Aggregate metrics: top-1 accuracy, top-5 hit rate, avg latency, etc.
  - Save full results JSON to eval/results/

Usage from project root:
    python eval/run_eval.py
    python eval/run_eval.py --note "after-chunking-fix"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.retrieve.runner import run_query


GOLDEN_SET_PATH = Path("eval/golden_set.yaml")
RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# A trap is "passed" if the highest rerank score is below this threshold —
# meaning the system has no confident match and would refuse / route to the
# advisory line. Calibrated from the smoke test: real answers score 0.4-0.99,
# noise scores below 0.1. Pick a conservative midpoint.
TRAP_REFUSAL_THRESHOLD = 0.30


def load_golden_set() -> list[dict]:
    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_one(item: dict) -> dict:
    """Run one question through the pipeline and grade the result."""
    result = run_query(item["query"], persist=True)

    top_articles = [c["article_number"] for c in result.top_chunks]
    top_1_article = top_articles[0] if top_articles else None
    expected = item["expected_articles"]
    max_score = result.top_chunks[0]["rerank_score"] if result.top_chunks else 0.0

    if item["in_corpus"]:
        # Correctness: did the expected article(s) show up?
        top_1_hit = top_1_article in expected
        top_5_hit = any(art in expected for art in top_articles)
        passed = top_1_hit
        verdict = "✅ pass" if passed else ("🟡 in-top-5" if top_5_hit else "❌ miss")
    else:
        # Trap: pass if confidence is low enough that the agent would refuse.
        # Once we add the classifier + Claude generation, this gets more
        # sophisticated (the agent's actual refusal becomes the test).
        top_1_hit = False
        top_5_hit = False
        passed = max_score < TRAP_REFUSAL_THRESHOLD
        verdict = "✅ would-refuse" if passed else "❌ would-answer"

    return {
        "id": item["id"],
        "query": item["query"],
        "expected_articles": expected,
        "in_corpus": item["in_corpus"],
        "top_articles": top_articles,
        "top_1_hit": top_1_hit,
        "top_5_hit": top_5_hit,
        "max_rerank_score": max_score,
        "passed": passed,
        "verdict": verdict,
        "latency_ms": result.total_latency_ms,
        "trace_path": str(result.trace_path),
    }


def aggregate(per_question: list[dict]) -> dict:
    """Compute summary metrics across the run."""
    n = len(per_question)
    in_corpus = [r for r in per_question if r["in_corpus"]]
    traps = [r for r in per_question if not r["in_corpus"]]

    return {
        "n_questions": n,
        "n_in_corpus": len(in_corpus),
        "n_traps": len(traps),
        "in_corpus_top_1_accuracy": (
            sum(r["top_1_hit"] for r in in_corpus) / len(in_corpus)
            if in_corpus else None
        ),
        "in_corpus_top_5_hit_rate": (
            sum(r["top_5_hit"] for r in in_corpus) / len(in_corpus)
            if in_corpus else None
        ),
        "trap_refusal_rate": (
            sum(r["passed"] for r in traps) / len(traps)
            if traps else None
        ),
        "overall_pass_rate": sum(r["passed"] for r in per_question) / n,
        "avg_latency_ms": sum(r["latency_ms"] for r in per_question) / n,
        "avg_max_rerank_score_in_corpus": (
            sum(r["max_rerank_score"] for r in in_corpus) / len(in_corpus)
            if in_corpus else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="", help="Free-form label appended to results filename")
    args = parser.parse_args()

    golden = load_golden_set()
    print(f"Loaded {len(golden)} questions from {GOLDEN_SET_PATH}\n")

    per_question = []
    for i, item in enumerate(golden, start=1):
        print(f"[{i}/{len(golden)}] {item['id']}: {item['query']}")
        result = evaluate_one(item)
        per_question.append(result)
        print(f"    {result['verdict']}  max_score={result['max_rerank_score']:.4f}  "
              f"top: {result['top_articles'][:3]}\n")

    metrics = aggregate(per_question)

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Top-1 accuracy (in-corpus):  {metrics['in_corpus_top_1_accuracy']:.1%}")
    print(f"  Top-5 hit rate (in-corpus):  {metrics['in_corpus_top_5_hit_rate']:.1%}")
    print(f"  Trap refusal rate:           {metrics['trap_refusal_rate']:.1%}")
    print(f"  Overall pass rate:           {metrics['overall_pass_rate']:.1%}")
    print(f"  Avg latency:                 {metrics['avg_latency_ms']:.0f} ms")
    print(f"  Avg max rerank score:        {metrics['avg_max_rerank_score_in_corpus']:.4f}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    label = f"_{args.note}" if args.note else ""
    out_path = RESULTS_DIR / f"{timestamp}{label}.json"
    out_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "note": args.note,
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
