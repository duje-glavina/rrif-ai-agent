"""Verify the article-number extractor handles all citation patterns we see."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the function directly from the eval script
import importlib.util
spec = importlib.util.spec_from_file_location("run_eval", "eval/run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)
extract = run_eval._extract_article_number

cases = [
    # (input, expected_output)
    ('38', '38'),
    ('38a', '38a'),
    ('čl. 38', '38'),
    ('čl 38', '38'),
    ('članak 38', '38'),
    ('članak 38.', '38'),
    ('čl. 12. st. 8. Zakona o porezu na dobit', '12'),
    ('38. st. 3. t. a) Zakona o PDV-u', '38'),
    ('čl. 2', '2'),
    ('art. 99', '99'),
    ('article 6', '6'),
    ('Article 99', '99'),
    (None, None),
    ('', None),
    ('Zakon o radu', None),  # no number at all
]

failures = []
for raw, expected in cases:
    actual = extract(raw)
    if actual != expected:
        failures.append((raw, expected, actual))

if failures:
    print(f"❌ {len(failures)} case(s) failed:")
    for raw, expected, actual in failures:
        print(f"  Input: {raw!r}")
        print(f"  Expected: {expected!r}")
        print(f"  Got:      {actual!r}")
        print()
    sys.exit(1)
else:
    print(f"✅ All {len(cases)} cases pass")