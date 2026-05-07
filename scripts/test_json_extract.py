import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.generate.answerer import _extract_json

# Case 1: clean JSON
assert _extract_json('{"answer": "x"}')["answer"] == "x"

# Case 2: ```json ... ``` wrapped
assert _extract_json('```json\n{"answer": "x"}\n```')["answer"] == "x"

# Case 3: ```json with NO closing fence (the failing case)
assert _extract_json('```json\n{"answer": "x"}')["answer"] == "x"

# Case 4: nested objects in fences
nested = '```json\n{"answer": "x", "citations": [{"a": 1, "b": 2}]}\n```'
assert _extract_json(nested)["citations"][0]["a"] == 1

# Case 5: nested objects, no closing fence
nested_open = '```json\n{"answer": "x", "citations": [{"a": 1, "b": 2}]}'
assert _extract_json(nested_open)["citations"][0]["a"] == 1

# Case 6: Croatian text with escaped quotes
croatian = r'{"answer": "Stopa je \"25%\" za sve.", "citations": []}'
assert _extract_json(croatian)["answer"] == 'Stopa je "25%" za sve.'

# Case 7: ```json wrap + Croatian + nested + no closing
worst_case = '```json\n{"answer": "Trošak \\"amortizacije\\" se priznaje.\\n\\n1. Početak obračuna.", "citations": [{"law_name": "Zakon o porezu", "article_number": "12"}]}'
result = _extract_json(worst_case)
assert "amortizacije" in result["answer"]
assert result["citations"][0]["article_number"] == "12"

print("All 7 cases pass ✓")