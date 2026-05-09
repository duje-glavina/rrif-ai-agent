"""Pre-retrieval query classifier for the RRiF AI Agent.

Runs before hybrid search to extract:
  - category     → which subset of the KB to search
  - time_period  → current law, specific year, or historical range
  - source_type  → preference for zakon, članak, FAQ, etc.

Uses Claude Haiku — cheap and fast, runs on every query.
Output is a validated dataclass, never raw JSON, so callers
get type safety and sensible defaults if the model misbehaves.

Usage:
    from rag.classifier import classify
    q = classify("Kolika je stopa PDV-a na hranu od 2024.?")
    # q.category == "PDV"
    # q.time_period.type == "specific_date"
    # q.time_period.date_from == "2024-01-01"
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import anthropic
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

CLASSIFIER_MODEL = "claude-haiku-4-5"

# NOTE: DB stores status as 'vazeci' / 'nevazeci' (no diacritics).
# Always use these constants — never hardcode the strings elsewhere.
STATUS_VALID   = "vazeci"
STATUS_INVALID = "nevazeci"

# Categories must match exact values written to the DB by article_loader.py
# (BANNER_TO_CATEGORY + DEFAULT_CATEGORY_BY_PUB). Do not rename without
# updating that loader and re-running ingestion.
CATEGORIES = Literal[
    "PDV",
    "porezi",
    "računovodstvo",
    "plaće",
    "radno pravo",
    "trgovačko pravo",
    "revizija",
    "proračun",
    "neprofitne organizacije",
    "EU propisi",
    "poslovanje s inozemstvom",
    "tržište i propisi",
    "novi propisi",
    "stručne informacije",
    "upravljanje",
    "ostalo",
]

SOURCE_TYPES = Literal["zakon", "članak", "FAQ", "priručnik", "transkript"]

SYSTEM_PROMPT = """\
Ti si klasifikator upita za sustav pretraživanja hrvatskog računovodstvenog i poreznog znanja.

Tvoj zadatak je analizirati korisnikovo pitanje i vratiti ISKLJUČIVO JSON objekt bez ikakvog teksta prije ili nakon.

JSON mora imati točno ovaj oblik:
{
  "category": "<jedna od 16 kategorija navedenih u nastavku>",
  "time_period": {
    "type": "<jedna od: current, specific_date, range, historical>",
    "date_from": "<YYYY-MM-DD ili null>",
    "date_to": "<YYYY-MM-DD ili null>"
  },
  "source_type_preference": ["<zakon|članak|FAQ|priručnik|transkript>"] ili null
}

Pravila za category — odaberi TOČNO jednu vrijednost iz ove liste (vrijednosti moraju biti napisane točno ovako, sa svim razmacima i dijakritikama):

- "PDV" → pitanja o porezu na dodanu vrijednost: stope, obrasci (PDV, PDV-S, PDV-K), pretporez, oslobođenja, prag za upis u registar PDV-a
- "porezi" → ostali porezi: porez na dohodak, porez na dobit, doprinosi, dividende, osobni odbitak, porez po odbitku, paušalno oporezivanje
- "računovodstvo" → knjiženje, bilanca, RDG, financijski izvještaji, amortizacija, zalihe, MRS/MSFI, HSFI, kontni plan
- "plaće" → obračun plaće, bruto/neto, naknade, putni nalozi, dnevnice, bolovanje, jubilarna nagrada
- "radno pravo" → ugovori o radu, otkaz, godišnji odmor, radnička prava, kolektivni ugovori, zaštita na radu
- "trgovačko pravo" → osnivanje društva, statut, organi, pripajanja, likvidacija, stečaj, prava trgovačkih društava
- "revizija" → revizija financijskih izvještaja, revizorska izvješća, MRevS, neovisnost revizora
- "proračun" → proračunski korisnici, proračunsko računovodstvo, javna nabava, financiranje iz proračuna
- "neprofitne organizacije" → udruge, zaklade, vjerske zajednice, sindikati, neprofitno računovodstvo
- "EU propisi" → direktive i uredbe EU, vijesti iz institucija EU, harmonizacija s EU pravom
- "poslovanje s inozemstvom" → uvoz/izvoz, devizno poslovanje, transferne cijene, dvostruko oporezivanje, INTRASTAT
- "tržište i propisi" → tržišno natjecanje, zaštita potrošača, opći propisi koji utječu na poslovanje
- "novi propisi" → najave i analize novih zakona koji tek stupaju na snagu
- "stručne informacije" → kratke stručne obavijesti, pojašnjenja, tumačenja iz prakse
- "upravljanje" → korporativno upravljanje, organizacijska struktura, interni akti
- "ostalo" → koristi SAMO ako pitanje ne odgovara nijednoj gornjoj kategoriji (npr. opće tehničko pitanje, GDPR, IT teme, drugo)

Ako pitanje pokriva više područja, odaberi PRIMARNU kategoriju (onu koja najbolje opisuje glavno pitanje).

Pravila za time_period:
- current → pitanje je o trenutno važećim propisima (nema vremenskog navoda)
- specific_date → pitanje navodi konkretnu godinu ili datum (npr. "od 2024.", "u 2021.")
  → date_from = "YYYY-01-01", date_to = null ako je samo godina
- range → raspon godina (npr. "između 2020. i 2023.")
  → date_from i date_to postavljeni
- historical → nejasno historijsko pitanje bez konkretnog datuma (npr. "prije uvođenja eura")
  → date_from = null, date_to = null

Pravila za source_type_preference:
- ["zakon"] ako pita za konkretni zakon, članak, ili propis
- ["članak"] ako pita za stručno mišljenje ili komentar
- null ako nema jasnih naznaka

Primjeri:
Pitanje: "Kolika je stopa PDV-a na hranu od 2024.?"
{"category": "PDV", "time_period": {"type": "specific_date", "date_from": "2024-01-01", "date_to": null}, "source_type_preference": null}

Pitanje: "Kako knjižiti amortizaciju građevinskog objekta?"
{"category": "računovodstvo", "time_period": {"type": "current", "date_from": null, "date_to": null}, "source_type_preference": null}

Pitanje: "Što su transferne cijene?"
{"category": "poslovanje s inozemstvom", "time_period": {"type": "current", "date_from": null, "date_to": null}, "source_type_preference": null}

Pitanje: "Kako udruga vodi poslovne knjige?"
{"category": "neprofitne organizacije", "time_period": {"type": "current", "date_from": null, "date_to": null}, "source_type_preference": null}

Pitanje: "Što piše u članku 99. Zakona o radu?"
{"category": "radno pravo", "time_period": {"type": "current", "date_from": null, "date_to": null}, "source_type_preference": ["zakon"]}
"""


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class TimePeriod:
    type: str = "current"          # current | specific_date | range | historical
    date_from: str | None = None   # ISO date string or None
    date_to: str | None = None     # ISO date string or None

    def to_sql_filter(self) -> str | None:
        """Return a SQL WHERE fragment for temporal filtering, or None for current."""
        if self.type == "current":
            # Use no-diacritic value — matches what ingest scripts write to DB
            return f"status = '{STATUS_VALID}'"
        if self.type in ("specific_date", "range"):
            df = self.date_from or "1900-01-01"
            dt = self.date_to or self.date_from or "9999-12-31"
            return (
                f"valid_from <= '{df}'::date "
                f"AND COALESCE(valid_to, '9999-12-31'::date) >= '{dt}'::date"
            )
        # historical — no tight filter, search everything
        return None


# Maps classifier output categories to actual DB category values.
# Classifier output schema is now synced 1:1 with DB values written by
# article_loader.py, so this is mostly identity. Kept as an explicit map
# (rather than removed) so future taxonomy changes have one place to edit.
#
# NOTE: 'ostalo' currently maps to None (no filter, search all chunks).
# This conflates "confidently classified as ostalo" with "couldn't classify".
# TODO: split these once classifier returns a confidence score.
_CATEGORY_MAP: dict = {
    "PDV":                      "PDV",
    "porezi":                   "porezi",
    "računovodstvo":            "računovodstvo",
    "plaće":                    "plaće",
    "radno pravo":              "radno pravo",
    "trgovačko pravo":          "trgovačko pravo",
    "revizija":                 "revizija",
    "proračun":                 "proračun",
    "neprofitne organizacije":  "neprofitne organizacije",
    "EU propisi":               "EU propisi",
    "poslovanje s inozemstvom": "poslovanje s inozemstvom",
    "tržište i propisi":        "tržište i propisi",
    "novi propisi":             "novi propisi",
    "stručne informacije":      "stručne informacije",
    "upravljanje":              "upravljanje",
    "ostalo":                   None,  # no filter — see TODO above
}


@dataclass
class ClassifierResult:
    category: str = "ostalo"
    time_period: TimePeriod = field(default_factory=TimePeriod)
    source_type_preference: list[str] | None = None
    raw: dict = field(default_factory=dict, repr=False)

    def to_retrieval_filter(self) -> dict:
        """Return kwargs suitable for hybrid_search filter argument."""
        db_category = _CATEGORY_MAP.get(self.category)
        return {
            "category": db_category,
            "sql_time_filter": self.time_period.to_sql_filter(),
            "source_types": self.source_type_preference,
        }


# ── Classifier ───────────────────────────────────────────────────────────────

def classify(question: str) -> ClassifierResult:
    """Classify a user question into category, time period, and source preference.

    Returns a ClassifierResult with sensible defaults if the model
    returns malformed JSON or if the API call fails.
    """
    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        timeout=15.0,  # 15 second timeout — classifier should never take longer
    )

    try:
        response = client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
        )
        raw_text = response.content[0].text.strip()
        # Strip markdown code fences if present (```json...``` or ```...```)
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]
            raw_text = raw_text.rsplit("```", 1)[0].strip()
        log.debug("Classifier raw response: %s", raw_text)

        data = json.loads(raw_text)
        return _parse(data)

    except json.JSONDecodeError as e:
        log.warning("Classifier returned invalid JSON: %s", e)
        return ClassifierResult()
    except Exception as e:
        log.warning("Classifier API call failed: %s", e)
        return ClassifierResult()


def _parse(data: dict) -> ClassifierResult:
    """Parse raw classifier JSON into a ClassifierResult, with validation."""
    valid_categories = {
        "PDV", "porezi", "računovodstvo", "plaće", "radno pravo",
        "trgovačko pravo", "revizija", "proračun", "neprofitne organizacije",
        "EU propisi", "poslovanje s inozemstvom", "tržište i propisi",
        "novi propisi", "stručne informacije", "upravljanje", "ostalo",
    }
    valid_source_types = {"zakon", "članak", "FAQ", "priručnik", "transkript"}

    category = data.get("category", "ostalo")
    if category not in valid_categories:
        log.warning("Unknown category %r, falling back to 'ostalo'", category)
        category = "ostalo"

    tp_raw = data.get("time_period", {})
    time_period = TimePeriod(
        type=tp_raw.get("type", "current"),
        date_from=tp_raw.get("date_from"),
        date_to=tp_raw.get("date_to"),
    )
    if time_period.type not in ("current", "specific_date", "range", "historical"):
        time_period.type = "current"

    stp = data.get("source_type_preference")
    if isinstance(stp, list):
        stp = [s for s in stp if s in valid_source_types] or None
    else:
        stp = None

    return ClassifierResult(
        category=category,
        time_period=time_period,
        source_type_preference=stp,
        raw=data,
    )


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    questions = sys.argv[1:] or [
        "Kolika je stopa PDV-a na hranu od 2024.?",
        "Kako se oporezivao dohodak od kapitala u 2021. godini?",
        "Koji su kriteriji za razvrstavanje mikro poduzetnika?",
        "Što piše u članku 99. Zakona o radu?",
        "Kako udruga vodi poslovne knjige?",
        "Što su transferne cijene?",
        "Kako se obračunava bolovanje za radnika?",
        "Koje su nove obveze fiskalizacije?",
    ]
    for q in questions:
        result = classify(q)
        print(f"\nQ: {q}")
        print(f"   category  : {result.category}")
        print(f"   time      : {result.time_period}")
        print(f"   src_pref  : {result.source_type_preference}")
        print(f"   sql_filter: {result.time_period.to_sql_filter()}")