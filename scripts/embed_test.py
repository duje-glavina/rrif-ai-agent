"""Sanity check the embedder. First run downloads ~2GB and is slow."""
import sys
from pathlib import Path

# Make 'rag' importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.embedder import embed_passages, embed_query, EMBEDDING_DIM

print("\n--- Embedding a query ---")
q = embed_query("Kolika je stopa PDV-a na hranu od 2024.?")
print(f"Query shape: {q.shape}")
print(f"Query dtype: {q.dtype}")
print(f"Query first 5 values: {q[:5]}")
print(f"Query L2 norm: {(q ** 2).sum() ** 0.5:.4f}  (should be ~1.0)")

print("\n--- Embedding 3 passages ---")
passages = [
    "Opća stopa PDV-a u Hrvatskoj iznosi 25%.",
    "Snižena stopa od 13% primjenjuje se na usluge smještaja.",
    "Mirovinsko osiguranje I. stupa je 15% iz plaće.",
]
P = embed_passages(passages)
print(f"Passage matrix shape: {P.shape}  (should be (3, {EMBEDDING_DIM}))")

print("\n--- Cosine similarity to query ---")
sims = P @ q
for text, sim in zip(passages, sims):
    print(f"  {sim:+.4f}  {text}")

print("\nExpected: PDV passages should score higher than the pension one.")