import anthropic, os
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Test 1: basic connectivity
r = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=256,
    messages=[{"role": "user", "content": "Reci mi samo rijec: zdravo"}],
)
print("Test 1 - basic:", repr(r.content[0].text))

# Test 2: JSON output with system prompt
SYSTEM = """Vraćaj ISKLJUČIVO JSON bez ikakvog teksta prije ili nakon.
Primjer: {"category": "PDV", "time_period": {"type": "current"}}"""

r2 = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=256,
    system=SYSTEM,
    messages=[{"role": "user", "content": "Kolika je stopa PDV-a na hranu od 2024.?"}],
)
print("Test 2 - JSON:", repr(r2.content[0].text))
