"""Query rewriter — turns colloquial/ambiguous user input into clean,
retrieval-friendly Croatian for the RAG pipeline.

Runs BEFORE the classifier, so its output becomes the canonical query
that flows through classify → retrieve → rerank → generate.

Uses Claude Haiku — cheap and fast (~$0.001/call). The rewriter is
deliberately conservative: it preserves the user's intent, expands
abbreviations, normalises phrasing, and never invents facts or details
that weren't in the original question.

Usage:
    from rag.rewrite.rewriter import rewrite
    r = rewrite("kak se računa pdv kad prodajemo van eu?")
    # r.rewritten == "Kako se obračunava PDV pri prodaji izvan EU?"
    # r.changed == True
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

REWRITER_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 200


SYSTEM_PROMPT = """Ti si pomoćnik koji prepisuje pitanja korisnika u jasniji, formalniji oblik prikladan za pretraživanje hrvatske baze propisa, računovodstvenih i poreznih dokumenata.

Tvoj zadatak: prepiši pitanje tako da:
1. Koristi formalni hrvatski jezik (nema kolokvijalnih oblika kao "kak", "kaj", "ča").
2. Proširuje skraćenice koje su nejednoznačne (npr. "d.o.o." → "društvo s ograničenom odgovornošću", "PDV" ostaje "PDV").
3. Koristi standardnu pravnu/računovodstvenu terminologiju.
4. Zadržava IZVORNI SMISAO i SVA SPECIFIČNA OGRANIČENJA pitanja (godine, iznose, kategorije).
5. Ne dodaje nove činjenice, brojeve ni godine kojih nije bilo u izvornom pitanju.
6. Ne mijenja pitanje koje je već formalno i jasno — tada vrati gotovo identičan tekst.

KRITIČNO:
- Ako pitanje već koristi formalni jezik, samo ga blago doradi (interpunkcija, dijakritika).
- Nikad ne postavljaj sub-pitanje ili objašnjenje. Vrati ISKLJUČIVO prepisano pitanje, jednu rečenicu (ili dvije ako je izvorno bilo više).
- Bez prefiksa "Prepisano:", bez navodnika, bez objašnjenja — samo čisti tekst pitanja.

Primjeri:

Pitanje: "kak se računa pdv kad prodajemo van eu?"
Odgovor: Kako se obračunava PDV pri prodaji izvan EU?

Pitanje: "kaj treba za mikro doo?"
Odgovor: Koji su kriteriji za razvrstavanje mikro društva s ograničenom odgovornošću?

Pitanje: "Kolika je opća stopa PDV-a?"
Odgovor: Kolika je opća stopa PDV-a?

Pitanje: "doprinosi za radnika koliko"
Odgovor: Koliki su doprinosi iz plaće na teret radnika?

Pitanje: "godišnji odmor minimalno"
Odgovor: Koliko najmanje dana godišnjeg odmora pripada radniku?

Pitanje: "Kako se oporezivao dohodak od kapitala 2021?"
Odgovor: Kako se oporezivao dohodak od kapitala u 2021. godini?
"""


@dataclass
class RewriteResult:
    original: str
    rewritten: str
    changed: bool
    input_tokens: int
    output_tokens: int
    model: str
    error: str | None = None


def rewrite(question: str) -> RewriteResult:
    """Rewrite a user question into clean, retrieval-friendly Croatian.

    Returns a RewriteResult with both original and rewritten text, plus
    a `changed` flag indicating whether meaningful rewriting occurred.

    On any API error, returns the original question unchanged with the
    error captured in `result.error`. The pipeline can keep running with
    the original query — rewriting is enhancement, not gatekeeping.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return RewriteResult(
            original=question,
            rewritten=question,
            changed=False,
            input_tokens=0,
            output_tokens=0,
            model=REWRITER_MODEL,
            error="ANTHROPIC_API_KEY not set",
        )

    client = anthropic.Anthropic(api_key=api_key, timeout=10.0)

    try:
        response = client.messages.create(
            model=REWRITER_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
        )
        rewritten = response.content[0].text.strip()

        # Strip any accidental quotes or prefixes the model might add
        rewritten = rewritten.strip('"').strip("'").strip()
        for prefix in ("Prepisano:", "Odgovor:", "Pitanje:"):
            if rewritten.startswith(prefix):
                rewritten = rewritten[len(prefix):].strip()

        # Defensive fallback — if the model returned empty or something
        # implausibly short, keep the original
        if len(rewritten) < 5:
            log.warning("Rewriter returned suspiciously short output: %r", rewritten)
            rewritten = question

        changed = _meaningfully_different(question, rewritten)

        return RewriteResult(
            original=question,
            rewritten=rewritten,
            changed=changed,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=REWRITER_MODEL,
        )

    except Exception as exc:
        log.warning("Rewriter API call failed: %s", exc)
        return RewriteResult(
            original=question,
            rewritten=question,  # safe fallback — pipeline continues with original
            changed=False,
            input_tokens=0,
            output_tokens=0,
            model=REWRITER_MODEL,
            error=str(exc),
        )


def _meaningfully_different(original: str, rewritten: str) -> bool:
    """Return True if the rewrite changed more than whitespace/punctuation.

    Used to flag whether the rewriter actually contributed work. Cheap to
    compute, useful for eval analysis ("did rewriting actually fire on this
    question?").
    """
    def _normalise(s: str) -> str:
        return "".join(c.lower() for c in s if c.isalnum())

    return _normalise(original) != _normalise(rewritten)


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    questions = sys.argv[1:] or [
        # Colloquial / messy
        "kak se računa pdv kad prodajemo van eu?",
        "kaj treba za mikro doo?",
        "doprinosi za radnika koliko",
        "godišnji odmor minimalno",
        # Already formal — should barely change
        "Kolika je opća stopa PDV-a?",
        "Koji su kriteriji za razvrstavanje mikro poduzetnika?",
        # Temporal
        "Kako se oporezivao dohodak od kapitala 2021?",
    ]

    for q in questions:
        r = rewrite(q)
        marker = "✏️ " if r.changed else "✓ "
        print(f"\n{marker}IN : {r.original}")
        print(f"  OUT: {r.rewritten}")
        if r.error:
            print(f"  ERR: {r.error}")
        print(f"  tokens: {r.input_tokens}in / {r.output_tokens}out")