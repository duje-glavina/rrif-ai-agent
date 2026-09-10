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

TVOJ ZADATAK JE ISKLJUČIVO PREPISIVANJE PITANJA. NIKADA NE ODGOVARAJ NA PITANJE.

Pravila prepisivanja:
1. Koristi formalni hrvatski jezik (nema kolokvijalnih oblika kao "kak", "kaj", "ča").
2. Proširuj samo NEDVOSMISLENE skraćenice (npr. "d.o.o." → "društvo s ograničenom odgovornošću").
3. Koristi standardnu pravnu/računovodstvenu terminologiju.
4. Zadrži IZVORNI SMISAO i SVA SPECIFIČNA OGRANIČENJA pitanja (godine, iznose, kategorije, nazive zakona).

KRITIČNA OGRANIČENJA — slijedi STROGO:

A) NIKADA ne odgovaraj na pitanje. Ne objašnjavaj, ne ispravljaj, ne komentiraj. Ako ti se čini da pitanje sadrži netočnu pretpostavku (nepostojeći zakon, izmišljeni pojam), JEDNOSTAVNO ga prepiši u formalniji oblik bez ikakve napomene.

B) NIKADA ne izmišljaj proširenja skraćenica ili pojmova koje ne razumiješ pouzdano. Ako nisi 100% siguran što neka skraćenica znači, OSTAVI JE U IZVORNOM OBLIKU. Bolje da kratica ostane neproširena nego pogrešno proširena.

C) NIKADA ne dodaj novi opseg ili specifičnost koje nije bilo u izvornom pitanju. Ako korisnik kaže "novi zakon", NEMOJ pisati "novi zakon o zaštiti potrošača" — koji zakon, znaš li? Ostavi neutralno.

D) NIKADA ne dodaj brojeve, godine ili imena zakona kojih nije bilo u izvornom pitanju.

E) Ako je pitanje već formalno i jasno, vrati ga GOTOVO IDENTIČNO. U nedoumici — manje promjena je bolje.

F) Vrati ISKLJUČIVO prepisano pitanje, jednu rečenicu (ili dvije ako je izvorno bilo više).
   Bez prefiksa "Prepisano:", bez navodnika, bez objašnjenja, bez nabrajanja opcija.

Primjeri ispravnog ponašanja:

Pitanje: "kak se računa pdv kad prodajemo van eu?"
Odgovor: Kako se obračunava PDV pri prodaji izvan EU?

Pitanje: "kaj treba za mikro doo?"
Odgovor: Koji su kriteriji za razvrstavanje mikro društva s ograničenom odgovornošću?

Pitanje: "Kolika je opća stopa PDV-a?"
Odgovor: Kolika je opća stopa PDV-a?

Pitanje: "doprinosi za radnika koliko"
Odgovor: Koliki su doprinosi iz plaće na teret radnika?

Pitanje: "Kako se oporezivao dohodak od kapitala 2021?"
Odgovor: Kako se oporezivao dohodak od kapitala u 2021. godini?

PRIMJERI POGREŠNOG PONAŠANJA — NIKADA TAKO:

Pitanje: "Što kaže članak 99. Zakona o fiktivnom porezu na digitalne usluge?"
POGREŠNO: "Nije moguće odgovoriti jer taj zakon ne postoji..."
POGREŠNO: "Možda mislite na Zakon o digitalnim uslugama?"
ISPRAVNO: Što propisuje članak 99. Zakona o fiktivnom porezu na digitalne usluge?

Pitanje: "Što je MRevS i tko ga primjenjuje?"
POGREŠNO: "Što su Mali Revizijski Standardi (MRevS)..."  (izmišljeno proširenje)
POGREŠNO: "Što su Modernizirani revizijski standardi..."  (izmišljeno proširenje)
ISPRAVNO: Što je MRevS i tko ga primjenjuje?

Pitanje: "Koja prava imaju potrošači prema novom zakonu?"
POGREŠNO: "Koja prava imaju potrošači prema novom zakonu o zaštiti potrošača?"  (dodaje opseg)
ISPRAVNO: Koja prava imaju potrošači prema novom zakonu?

Pitanje: "Kako podnijeti zahtjev za pristup osobnim podacima prema GDPR-u?"
POGREŠNO: "...prema Uredbi (EU) 2016/679 o zaštiti fizičkih osoba..."  (dodaje neimenovane detalje)
ISPRAVNO: Kako podnijeti zahtjev za pristup osobnim podacima prema GDPR-u?
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


# ═══════════════════════════════════════════════════════════════════════════
# Conversation condensation
# ═══════════════════════════════════════════════════════════════════════════
# RRiF's own advisor asked four follow-up questions inside one conversation on
# 29 May — "Treba li ga primijeniti…", "Daj mi primjer te detaljnije analize
# F. Plišića", "Iz prethodnog pitanja koje sam postavio" — and every one was
# refused. Not because retrieval failed but because there is nothing to
# retrieve on: "ga" and "te" and "prethodnog" carry the entire meaning and none
# of it is in the text.
#
# Passing history to the GENERATOR does not fix this. The embedding and the FTS
# query are built from the question alone, so the search happens before any
# history could help. The follow-up has to become a standalone question BEFORE
# classification — which is the rewriter's job, with more context.
#
# Deliberately separate from rewrite(): that one must never add specificity the
# user did not supply (rule C). This one exists precisely to add specificity —
# but only what the conversation already contains.

CONDENSE_MAX_TOKENS = 300
_HISTORY_TURNS = 4          # further back adds cost and rarely adds meaning
_ANSWER_CLIP = 700          # enough to identify what was discussed


CONDENSE_SYSTEM = """Ti pretvaraš pitanje iz razgovora u samostalno pitanje koje se može razumjeti bez razgovora.

NIKADA NE ODGOVARAJ NA PITANJE. Vraćaš samo preoblikovano pitanje.

Dobivaš prethodne izmjene u razgovoru i posljednje korisnikovo pitanje. Zadatak:

1. Razriješi sve što upućuje na prethodni tijek razgovora — zamjenice ("ga", "to", "te"), pokazne izraze ("iz prethodnog pitanja", "ona analiza"), i izostavljeni predmet. Umjesto njih upiši ono na što se odnose.
2. Koristi ISKLJUČIVO ono što se u razgovoru već spominje. Ne dodaji propise, godine, iznose ni nazive kojih u razgovoru nema.
3. Ako je posljednje pitanje već samostalno i razumljivo bez razgovora, vrati ga NEPROMIJENJENO.
4. Zadrži formalni hrvatski jezik i stručnu terminologiju iz razgovora.
5. Vrati JEDNU rečenicu — samo pitanje, bez navodnika, bez objašnjenja, bez prefiksa.

Primjeri:

Razgovor: "Tko mora primijeniti zahtjeve HSFI-ja 18?" → (odgovor o HSFI 18)
Pitanje: "Treba li ga primijeniti društvo koje prestaje po skraćenom postupku brisanja bez likvidacije?"
ISPRAVNO: Treba li društvo koje prestaje po skraćenom postupku brisanja bez likvidacije primijeniti zahtjeve HSFI-ja 18?

Razgovor: "Mora li se obračunati PDV na reprezentaciju u ugostiteljstvu?" → (odgovor koji spominje analizu F. Plišića)
Pitanje: "Daj mi primjer te detaljnije analize F. Plišića"
ISPRAVNO: Kakva je detaljnija analiza F. Plišića o obračunu PDV-a na reprezentaciju u ugostiteljstvu?

Razgovor: "Kolika je minimalna plaća u Hrvatskoj?" → (odgovor)
Pitanje: "Kolika je stopa PDV-a na knjige?"
ISPRAVNO: Kolika je stopa PDV-a na knjige?"""


@dataclass
class CondenseResult:
    original: str
    standalone: str
    changed: bool
    n_turns: int
    input_tokens: int
    output_tokens: int
    model: str
    error: str | None = None


def condense(history: list[tuple[str, str]], question: str) -> CondenseResult:
    """Turn a follow-up into a question that stands on its own.

    `history` is [(question, answer), …] oldest first. Only the last few turns
    are used, and answers are clipped — the point is to identify the subject
    under discussion, not to re-read it.

    Falls back to the original question on any failure. A condenser that
    breaks must not take the pipeline with it.
    """
    def _fail(err: str) -> CondenseResult:
        return CondenseResult(question, question, False, len(history or []),
                              0, 0, REWRITER_MODEL, err)

    if not history:
        return CondenseResult(question, question, False, 0, 0, 0, REWRITER_MODEL)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _fail("ANTHROPIC_API_KEY not set")

    turns = history[-_HISTORY_TURNS:]
    lines = []
    for i, (q, a) in enumerate(turns, 1):
        lines.append(f"[{i}] Korisnik: {(q or '').strip()}")
        ans = (a or "").strip().replace("\n", " ")
        if ans:
            clipped = ans[:_ANSWER_CLIP] + ("…" if len(ans) > _ANSWER_CLIP else "")
            lines.append(f"    Sustav: {clipped}")
    user = ("RAZGOVOR DO SADA:\n" + "\n".join(lines)
            + f"\n\nPOSLJEDNJE PITANJE: {question.strip()}")

    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=15.0)
        resp = client.messages.create(
            model=REWRITER_MODEL,
            max_tokens=CONDENSE_MAX_TOKENS,
            temperature=0,
            system=CONDENSE_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        out = resp.content[0].text.strip().strip('"').strip("'").strip()
        for prefix in ("ISPRAVNO:", "Pitanje:", "Samostalno pitanje:"):
            if out.startswith(prefix):
                out = out[len(prefix):].strip()
        if len(out) < 5:
            log.warning("Condenser returned suspiciously short output: %r", out)
            out = question
        return CondenseResult(
            original=question, standalone=out,
            changed=_meaningfully_different(question, out),
            n_turns=len(turns),
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=REWRITER_MODEL,
        )
    except Exception as exc:                                     # noqa: BLE001
        log.warning("Condenser API call failed: %s", exc)
        return _fail(str(exc))


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