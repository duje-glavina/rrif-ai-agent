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
from rag.stem_hr import stem_text_crude


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
# Publication key — how magazine questions get graded
# ---------------------------------------------------------------------------
_ISSUE_RE = re.compile(r"br\.\s*(\d+)\s*/\s*(\d{4})")


def _parse_source_key(text: str | None) -> tuple[str, int, int] | None:
    """Normalise a publication reference to (pub, issue, year).

    Handles both the DB's `source` column —

        'RRiF br. 11/2024 — Na isporuku i ugradnju solarnih ploča…'
        'Porezno i pravno (PiP) br. 11/2024 — PRAVO I POREZI, br. 11/24…'

    — and the shorthand used in the golden set's expected_sources:

        'RRiF 11/2024'   'PiP 10/2024'

    Only the part before the em dash is considered, because the article body
    frequently repeats an issue reference of its own ('PRAVO I POREZI, br.
    11/24') and matching on that would produce false hits.
    """
    if not text:
        return None
    head = text.split("—")[0]

    m = _ISSUE_RE.search(head)
    if m:
        issue, year = int(m.group(1)), int(m.group(2))
    else:
        # Bare 'N/YYYY' (the golden set shorthand). Bounded to 1–12 because a
        # magazine has at most twelve issues a year — without that, an NN law
        # reference in the source string parses as an issue and you get
        # nonsense like 'RRiF 73/2013' (NN 73/13 is the VAT Act).
        m = re.search(r"\b([1-9]|1[0-2])\s*/\s*(\d{4})\b", head)
        if not m:
            return None
        issue, year = int(m.group(1)), int(m.group(2))

    if not 1 <= issue <= 12:
        return None

    low = head.lower()
    pub = "PiP" if ("pip" in low or "pravo i porezi" in low
                    or "porezno i pravno" in low) else "RRiF"
    return (pub, issue, year)


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
    expected_sources = item.get("expected_sources") or []
    in_corpus = item["in_corpus"]

    # Two independent ways to grade retrieval, because the corpus has two
    # kinds of content and they are identified differently:
    #
    #   law questions      → article number, on `zakon` chunks only
    #   magazine questions → publication issue, from `source`
    #
    # Most of the corpus (12,561 of 12,749) is magazine, so the second one is
    # what actually measures the product.
    gradeable_retrieval = bool(in_corpus and expected)
    gradeable_source = bool(in_corpus and expected_sources)

    # ── Article grading — statute only ──────────────────────────────────────
    # `article_number` on a magazine chunk is a chunk position (2, 3, 4…),
    # not an article of a law, so including them here produces false hits
    # against expected values like ["38"]. The real magazine article number
    # lives in extra_metadata.article_num.
    # Rank-ordered, one slot per retrieved chunk, None where the chunk is not
    # statute. Keeping the positions is what makes "top-1" mean rank 1 OVERALL
    # rather than "the first law chunk anywhere in the top-5" — collapsing the
    # list first turned top-1 into a much weaker claim and inflated it.
    law_articles_ranked = [
        _extract_article_number(m.get("article_number"))
        if m.get("source_type") != "članak" else None
        for m in result.retrieved_meta
    ]
    retrieved_articles = [a for a in law_articles_ranked if a]

    # Kept for diagnostics — what the old, collision-prone grading saw.
    retrieved_articles_any = [
        a for a in (
            _extract_article_number(m.get("article_number"))
            for m in result.retrieved_meta
        ) if a
    ]

    if gradeable_retrieval:
        retrieval_top_1 = bool(law_articles_ranked) and law_articles_ranked[0] in expected
        retrieval_top_k = any(a in expected for a in retrieved_articles)
    else:
        retrieval_top_1 = None
        retrieval_top_k = None

    # ── Content grading — does a retrieved chunk contain the answer? ────────
    # The primary magazine metric. Issue-level matching proved too coarse:
    # the 2024 minimum wage figure appears in eleven of that year's twelve
    # issues, so "the right issue" is not well defined. What is well defined
    # is whether retrieval put the answer in front of the generator.
    #
    # `_all` is strict (every expected keyword present in one chunk); `_any`
    # is the loose floor. The gap between them is usually a keyword list that
    # is too demanding rather than a retrieval failure, so both are reported.
    #
    # MORPHOLOGY. Exact substring matching compares a nominative-singular
    # keyword against running Croatian, which inflects everything. The golden
    # set says "reprezentacija", the corpus says "reprezentacije"; the golden
    # set says "glavna knjiga", the corpus says "glavnu knjigu". Five of the
    # seven questions in the top-5 gap were this and nothing else -- chunks
    # that plainly answered the question, scored as misses.
    #
    # So both sides are stemmed before comparison, and both numbers are kept:
    # `content_*` is the morphology-aware metric and the one to read;
    # `content_*_exact` preserves the old definition so runs from before this
    # change stay comparable. Deliberately the crude backend, pinned, so the
    # metric cannot shift with whatever happens to be installed locally.
    keywords = [k for k in (item.get("expected_keywords") or []) if k]
    gradeable_content = bool(in_corpus and keywords)

    def _norm(s: str) -> str:
        """Collapse whitespace and close up '20 %' → '20%'.

        Croatian typography puts a space before the percent sign, and the
        golden set writes '20%', so an exact substring test misses a chunk
        that plainly contains the answer. Same for the non-breaking spaces
        and soft hyphens the PDF extraction leaves behind.
        """
        s = s.lower().replace(" ", " ").replace("­", "")
        s = re.sub(r"\s+", " ", s)
        return re.sub(r"\s+%", "%", s)

    def _fold(s: str) -> str:
        """_norm, then strip Croatian inflection from both sides equally."""
        return stem_text_crude(_norm(s))

    _kw_exact = [_norm(k) for k in keywords]
    _kw_fold = [_fold(k) for k in keywords]

    def _all_hits(texts: list[str], kws: list[str]) -> list[bool]:
        return [all(k in t for k in kws) for t in texts]

    def _any_hits(texts: list[str], kws: list[str]) -> list[bool]:
        return [any(k in t for k in kws) for t in texts]

    if gradeable_content and result.retrieved_meta:
        _exact_texts = [_norm(m.get("chunk_text") or "") for m in result.retrieved_meta]
        _fold_texts = [stem_text_crude(x) for x in _exact_texts]

        _ex_all = _all_hits(_exact_texts, _kw_exact)
        _fo_all = _all_hits(_fold_texts, _kw_fold)

        content_top_1 = _fo_all[0]
        content_top_k = any(_fo_all)
        content_any_top_k = any(_any_hits(_fold_texts, _kw_fold))

        content_top_1_exact = _ex_all[0]
        content_top_k_exact = any(_ex_all)
        content_any_top_k_exact = any(_any_hits(_exact_texts, _kw_exact))
    elif gradeable_content:
        content_top_1 = content_top_k = content_any_top_k = False
        content_top_1_exact = content_top_k_exact = content_any_top_k_exact = False
    else:
        content_top_1 = content_top_k = content_any_top_k = None
        content_top_1_exact = content_top_k_exact = content_any_top_k_exact = None

    # ── Source grading — magazine content, secondary ────────────────────────
    expected_keys = {k for k in (_parse_source_key(s) for s in expected_sources) if k}
    retrieved_keys = [_parse_source_key(m.get("source")) for m in result.retrieved_meta]

    if gradeable_source and expected_keys:
        source_top_1 = bool(retrieved_keys) and retrieved_keys[0] in expected_keys
        source_top_k = any(k in expected_keys for k in retrieved_keys)
    else:
        source_top_1 = None
        source_top_k = None

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
        if gradeable_retrieval:
            passed = retrieval_top_1
        elif gradeable_content:
            passed = content_top_k
        else:
            passed = None
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
        "gradeable_source": gradeable_source,
        "gradeable_content": gradeable_content,
        "content_top_1": content_top_1,
        "content_top_k": content_top_k,
        "content_any_top_k": content_any_top_k,
        "content_top_1_exact": content_top_1_exact,
        "content_top_k_exact": content_top_k_exact,
        "content_any_top_k_exact": content_any_top_k_exact,
        # F1 covers the magazine (članci). The six PDV-law questions are the
        # only ones with a non-empty expected_articles, so that field is also
        # the scope marker — and the F1 number must not be dragged down by
        # questions that are out of F1 scope by agreement.
        "scope": "zakon" if expected else "članak",
        "expected_articles": expected,
        "expected_sources": expected_sources,
        "retrieved_articles": retrieved_articles,
        "retrieved_articles_any": retrieved_articles_any,
        "retrieved_sources": [
            f"{k[0]} {k[1]}/{k[2]}" if k else None for k in retrieved_keys
        ],
        "retrieved_meta": result.retrieved_meta,
        "retrieval_top_1": retrieval_top_1,
        "retrieval_top_k": retrieval_top_k,
        "source_top_1": source_top_1,
        "source_top_k": source_top_k,
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
    src_gradeable = [r for r in per_question if r["gradeable_source"]]
    con_gradeable = [r for r in per_question if r["gradeable_content"]]
    con_clanci = [r for r in con_gradeable if r.get("scope") != "zakon"]
    con_zakoni = [r for r in con_gradeable if r.get("scope") == "zakon"]
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
        # Primary magazine metric: did retrieval surface the answer at all.
        "n_gradeable_content": len(con_gradeable),
        "content_top_1": _mean([r["content_top_1"] for r in con_gradeable]),
        "content_top_k": _mean([r["content_top_k"] for r in con_gradeable]),
        "content_any_top_k": _mean([r["content_any_top_k"] for r in con_gradeable]),
        # Split by scope. ČLANCI is the F1 number; ZAKONI is tracked but out
        # of F1 scope until the law question in the TehSpec is settled.
        "n_content_clanci": len(con_clanci),
        "content_top_1_clanci": _mean([r["content_top_1"] for r in con_clanci]),
        "content_top_k_clanci": _mean([r["content_top_k"] for r in con_clanci]),
        "content_any_top_k_clanci": _mean([r["content_any_top_k"] for r in con_clanci]),
        "n_content_zakoni": len(con_zakoni),
        "content_top_1_zakoni": _mean([r["content_top_1"] for r in con_zakoni]),
        "content_top_k_zakoni": _mean([r["content_top_k"] for r in con_zakoni]),
        # Old exact-substring definition, kept so pre-6-Sep runs stay readable.
        "content_top_1_exact": _mean([r.get("content_top_1_exact") for r in con_gradeable]),
        "content_top_k_exact": _mean([r.get("content_top_k_exact") for r in con_gradeable]),
        "content_any_top_k_exact": _mean([r.get("content_any_top_k_exact") for r in con_gradeable]),
        # Secondary, and coarse — see the comment in evaluate_one.
        "n_gradeable_source": len(src_gradeable),
        "source_top_1": _mean([r["source_top_1"] for r in src_gradeable]),
        "source_top_k": _mean([r["source_top_k"] for r in src_gradeable]),
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
                bits.append(f"want čl. {result['expected_articles']}")
                bits.append(f"got {result['retrieved_articles'][:3]}")
            elif result["gradeable_source"]:
                bits.append(f"want {result['expected_sources']}")
                got = [s for s in result["retrieved_sources"][:3] if s]
                bits.append(f"got {got}")
            else:
                bits.append(f"got {result['retrieved_articles_any'][:3]} (not graded)")
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
    ns = metrics["n_gradeable_source"]
    nc = metrics["n_gradeable_content"]
    ncl = metrics["n_content_clanci"]
    nzk = metrics["n_content_zakoni"]
    print(f"  ČLANCI  odgovor u top-1:       {_pct(metrics['content_top_1_clanci'], ncl)}   ← F1")
    print(f"  ČLANCI  odgovor u top-5:       {_pct(metrics['content_top_k_clanci'], ncl)}   ← F1")
    print(f"    (bar jedan pojam, top-5):    {_pct(metrics['content_any_top_k_clanci'], ncl)}")
    print()
    print(f"  ZAKONI  odgovor u top-1:       {_pct(metrics['content_top_1_zakoni'], nzk)}   (izvan F1)")
    print(f"  ZAKONI  odgovor u top-5:       {_pct(metrics['content_top_k_zakoni'], nzk)}   (izvan F1)")
    print()
    print(f"  SVE     odgovor u top-1:       {_pct(metrics['content_top_1'], nc)}")
    print(f"  SVE     odgovor u top-5:       {_pct(metrics['content_top_k'], nc)}")
    print(f"    (bar jedan pojam, top-5):    {_pct(metrics['content_any_top_k'], nc)}")
    print(f"    [staro, doslovno]:           "
          f"{_pct(metrics['content_top_1_exact'])} / "
          f"{_pct(metrics['content_top_k_exact'])} / "
          f"{_pct(metrics['content_any_top_k_exact'])}")
    print()
    print(f"  ČLANCI  source top-1:          {_pct(metrics['source_top_1'], ns)}")
    print(f"  ČLANCI  source top-5:          {_pct(metrics['source_top_k'], ns)}")
    print(f"  ZAKONI  article top-1:         {_pct(metrics['retrieval_top_1'], ng)}")
    print(f"  ZAKONI  article top-5:         {_pct(metrics['retrieval_top_k'], ng)}")
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
    if nc:
        print(f"  Note: F1 covers članci. {ncl} of the {nc} content-graded "
              f"questions are članci (one = {100/ncl:.1f} points); the other "
              f"{nzk} are PDV-law questions, tracked separately.")
        print(f"        {nc} questions graded on answer content "
              f"(one = {100/nc:.1f} points) — this is the primary number.")
    if ns:
        print(f"        {ns} of those also graded on publication issue "
              f"(coarse: recurring figures span many issues).")
    if ng:
        print(f"        {ng} law questions graded on article number "
              f"(one = {100/ng:.1f} points).")
    print(f"        {metrics['n_questions'] - nc} questions have no ground "
          f"truth (traps and out-of-corpus) and are not graded.")
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
