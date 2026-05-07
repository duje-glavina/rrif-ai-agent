"""Claude generation: produce a grounded, cited answer from retrieved chunks.

Uses Anthropic's tool-use feature for structured output. The model is
constrained to produce JSON matching the `submit_answer` tool's input schema,
so output is always valid and the parsing layer can't fail.

Migrated from prompt-based JSON output (with regex-based parsing) on 2026-05-07.
The old `_extract_json` and `_parse_json_text` helpers have been removed since
they're no longer needed.

The system prompt enforces:
  - Answer only from provided context
  - Cite sources (tool schema requires citations field)
  - Refuse and redirect to RRiF advisor line when uncertain
  - Disclose temporal basis when laws are involved
  - Croatian language only
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

load_dotenv()


DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 2500


# ---------------------------------------------------------------------------
# Tool schema — defines the structured output format the model must produce
# ---------------------------------------------------------------------------

ANSWER_TOOL_SCHEMA = {
    "name": "submit_answer",
    "description": (
        "Submit the structured answer to the user's question about Croatian "
        "accounting, tax, or finance regulations. Use this tool to deliver "
        "your response — do not respond conversationally."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "refused": {
                "type": "boolean",
                "description": (
                    "TRUE if the dostavljeni kontekst does not contain a "
                    "reliable answer to the question. In that case, set "
                    "answer to a brief explanation that the user should "
                    "contact the RRiF advisory line, leave citations empty, "
                    "and set temporal_note to null."
                ),
            },
            "answer": {
                "type": "string",
                "description": (
                    "The answer text in Croatian. Plain text only — do not "
                    "wrap in markdown code fences or include any JSON "
                    "structures. Cite sources inline using parentheses, "
                    "e.g. '(Zakon o PDV-u, NN 73/2013, čl. 38)'."
                ),
            },
            "citations": {
                "type": "array",
                "description": (
                    "List of source citations referenced in the answer. "
                    "Empty array if refused=true."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "law_name": {
                            "type": ["string", "null"],
                            "description": "e.g. 'Zakon o PDV-u'",
                        },
                        "nn_reference": {
                            "type": ["string", "null"],
                            "description": "e.g. 'NN 73/2013'",
                        },
                        "article_number": {
                            "type": ["string", "null"],
                            "description": (
                                "ONLY the bare article number with no prefix. "
                                "Correct: '38', '38a', '12', '85'. "
                                "Wrong: 'čl. 38', 'članak 38', '38. st. 1.', "
                                "'čl. 12. st. 8. Zakona o porezu na dobit'. "
                                "If the article has sub-paragraphs (st.) or points (t.), "
                                "use ONLY the main article number. "
                                "Use null if the citation has no article number."
                            ),
                        },
                    },
                    "required": ["law_name", "nn_reference", "article_number"],
                },
            },
            "temporal_note": {
                "type": ["string", "null"],
                "description": (
                    "If the answer relies on regulations, a brief note "
                    "stating which time period the answer applies to. "
                    "e.g. 'Odgovor se temelji na propisu važećem od 1.1.2024.' "
                    "Null if not applicable or if refused=true."
                ),
            },
        },
        "required": ["refused", "answer", "citations", "temporal_note"],
    },
}


# ---------------------------------------------------------------------------
# System prompt — narrower than before because the schema does the heavy lifting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Ti si AI asistent za RRiF-plus d.o.o. — pomažeš savjetnicima i pretplatnicima s pitanjima iz hrvatskog računovodstva, poreza i financija.

PRAVILA KOJA STROGO POŠTUJEŠ:

1. Odgovaraj ISKLJUČIVO na temelju dostavljenih izvora u odjeljku "KONTEKST". Ne koristi vlastito predznanje.

2. Svaki činjenični navod u tekstu odgovora popraćen je citatom izvora unutar zagrada, npr.: "(Zakon o PDV-u, NN 73/2013, čl. 38)". Citiraj samo izvore koji su ti dostavljeni.

3. Ako dostavljeni kontekst ne sadrži pouzdan odgovor:
   - Postavi refused=true
   - U answer polje napiši kratko objašnjenje da nemaš dovoljno informacija i uputu na RRiF savjetničku liniju
   - Ostavi citations prazno (citations=[])
   - Ostavi temporal_note prazno (temporal_note=null)

4. NEMOJ izmišljati odgovore, zakone, članke ili brojeve. NEMOJ koristiti svoje opće znanje o hrvatskim propisima.

5. Razlikuj važeće i nevažeće propise. Ako odgovor potječe iz starije verzije zakona, jasno to navedi u temporal_note.

6. Odgovaraj sažeto, jasno i u tonu primjerenom stručnoj publici (računovođe, savjetnici, porezni stručnjaci).

7. Odgovaraj ISKLJUČIVO na hrvatskom jeziku.

Koristi ALAT submit_answer za strukturirani odgovor — nemoj odgovarati slobodnim tekstom."""


def system_prompt_hash() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Context formatting — unchanged from previous version
# ---------------------------------------------------------------------------

def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(Nema rezultata pretrage. Baza znanja ne sadrži relevantne dokumente za ovo pitanje.)"

    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        meta_lines = [f"[Izvor {i}]"]
        if c.get("law_name"):
            meta_lines.append(f"Zakon: {c['law_name']}")
        if c.get("nn_reference"):
            meta_lines.append(f"Narodne novine: {c['nn_reference']}")
        if c.get("article_number"):
            meta_lines.append(f"Članak: {c['article_number']}")
        validity = []
        if c.get("valid_from"):
            validity.append(f"od {c['valid_from']}")
        if c.get("valid_to"):
            validity.append(f"do {c['valid_to']}")
        else:
            validity.append("trenutno na snazi")
        meta_lines.append(f"Razdoblje važenja: {', '.join(validity)}")
        meta_lines.append(f"Status: {c.get('status', 'nepoznato')}")
        meta_lines.append("Tekst:")
        meta_lines.append(c["chunk_text"])
        blocks.append("\n".join(meta_lines))

    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Response data class — kept compatible with the previous version
# ---------------------------------------------------------------------------

@dataclass
class GeneratedAnswer:
    refused: bool
    answer: str
    citations: list[dict]
    temporal_note: str | None
    raw_response: str
    input_tokens: int
    output_tokens: int
    model: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer(
    query: str,
    chunks: list[dict],
    *,
    model: str = DEFAULT_MODEL,
) -> GeneratedAnswer:
    """Generate a grounded answer from query + retrieved chunks.

    Uses the submit_answer tool for structured output. The model is forced
    to use the tool via tool_choice, so the response is guaranteed to match
    the schema and no JSON parsing is required on our side.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment.")

    client = anthropic.Anthropic(api_key=api_key, timeout=60.0)
    context = format_context(chunks)
    user_message = f"KONTEKST:\n\n{context}\n\n---\n\nPITANJE: {query}"

    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[ANSWER_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_answer"},
        messages=[{"role": "user", "content": user_message}],
    )

    # With tool_choice forcing the tool, response.content[0] should be a
    # ToolUseBlock with .input matching the schema. If the model somehow
    # returns text instead (rare, usually due to refusals), fall back gracefully.
    tool_use_block = None
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            tool_use_block = block
            break

    if tool_use_block is None:
        # Defensive fallback: model didn't use the tool. Surface as refusal.
        text_block = next(
            (b for b in response.content if getattr(b, "type", None) == "text"),
            None,
        )
        raw_text = text_block.text if text_block else ""
        return GeneratedAnswer(
            refused=True,
            answer=(
                "Tehnička pogreška u obradi odgovora. "
                "Molim Vas obratite se RRiF savjetničkoj liniji."
            ),
            citations=[],
            temporal_note=None,
            raw_response=raw_text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
        )

    data = tool_use_block.input
    return GeneratedAnswer(
        refused=bool(data.get("refused", False)),
        answer=data.get("answer", ""),
        citations=data.get("citations", []) or [],
        temporal_note=data.get("temporal_note"),
        raw_response=str(data),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model=model,
    )
