"""How much of each chunk actually reaches the embedding model?

multilingual-e5-large is XLM-R based and has a fixed input window.
sentence-transformers truncates anything longer *silently* — no warning, no
error, just a vector that represents the beginning of the chunk and nothing
else. The audit shows a median članak chunk of 2,323 characters and a maximum
of 9,728, so this is worth measuring rather than assuming in either direction.

It matters because it would explain the shape of the current results: the
answer is in the top 5 for every in-corpus question (100% "bar jedan pojam"),
but the chunk that carries it does not always rank first. If a majority of
vectors only describe the first page of a three-page chunk, semantic retrieval
is matching on chunk openings, and the cross-encoder — which has a much larger
window — is left to rescue the ranking afterwards.

    python scripts/check_embed_truncation.py
    python scripts/check_embed_truncation.py --sample 3000

Read-only. Costs one model load.
"""
from __future__ import annotations

import argparse
import os

import psycopg
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2000,
                    help="Chunks to tokenise (random sample).")
    args = ap.parse_args()

    from rag.embedder import _get_model
    model = _get_model()
    limit = model.max_seq_length
    tok = model.tokenizer

    # e5 requires the "passage: " prefix, and it costs tokens like anything else.
    prefix_cost = len(tok.encode("passage: ", add_special_tokens=False))

    print(f"\nmodel max_seq_length : {limit} tokens")
    print(f"'passage: ' prefix   : {prefix_cost} tokens")
    print(f"budget for chunk text: {limit - prefix_cost - 2} tokens "
          f"(2 reserved for the special tokens)\n")

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT source_type, chunk_text FROM chunks ORDER BY random() LIMIT %s",
            (args.sample,),
        )
        rows = cur.fetchall()

    budget = limit - prefix_cost - 2
    by_type: dict[str, list[tuple[int, int]]] = {}
    for source_type, text in rows:
        n_tok = len(tok.encode(text, add_special_tokens=False))
        by_type.setdefault(source_type or "?", []).append((n_tok, len(text)))

    print(f"  {'source_type':<12} {'n':>6} {'trunc':>7} {'share':>8} "
          f"{'p50 tok':>8} {'p95 tok':>8} {'chars/tok':>10}")
    print("  " + "-" * 64)

    for st, vals in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        toks = sorted(v[0] for v in vals)
        chars = sum(v[1] for v in vals)
        total_tok = sum(v[0] for v in vals)
        over = sum(1 for t in toks if t > budget)
        n = len(toks)
        p50 = toks[n // 2]
        p95 = toks[min(n - 1, int(n * 0.95))]
        print(f"  {st:<12} {n:>6,} {over:>7,} {100*over/n:>7.1f}% "
              f"{p50:>8,} {p95:>8,} {chars/total_tok:>10.2f}")

    all_toks = sorted(t for vals in by_type.values() for t, _ in vals)
    over = sum(1 for t in all_toks if t > budget)
    lost = sum(t - budget for t in all_toks if t > budget)
    total = sum(all_toks)

    print(f"\n  {over:,} of {len(all_toks):,} sampled chunks exceed the window "
          f"({100*over/len(all_toks):.1f}%).")
    print(f"  {lost:,} of {total:,} tokens are never seen by the embedder "
          f"({100*lost/total:.1f}%).")
    print(f"\n  Chunk length that fits: roughly "
          f"{int(budget * (sum(c for vals in by_type.values() for _, c in vals) / total)):,} "
          f"characters at this corpus's characters-per-token ratio.\n")


if __name__ == "__main__":
    main()
