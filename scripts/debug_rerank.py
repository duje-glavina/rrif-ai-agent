import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from rag.classifier import classify
from rag.query import _retrieve
from rag.retrieve.rerank import rerank

tests = [
    'Koliki su doprinosi iz plaće na teret radnika?',
    'Kako se osniva društvo s ograničenom odgovornošću?',
    'Što donosi nova EU uredba o digitalnim tržištima?',
    'Koja prava imaju potrošači prema novom zakonu?',
]

for question in tests:
    print('=' * 70)
    print(f'Q: {question}')
    clf = classify(question)
    print(f'   category: {clf.category}')
    candidates = _retrieve(question, clf)
    rerank_input = [(row[0], row[1]) for row in candidates]
    ranked = rerank(question, rerank_input, k=5)
    meta = {row[0]: row for row in candidates}
    print(f'   Top 5 reranked chunks:')
    for cid, text, score in ranked:
        row = meta[cid]
        source = (row[2] or '?')[:55]
        preview = text[:150].replace('\n', ' ')
        print(f'   [{score:.4f}] {source}')
        print(f'            {preview}')
    print()
