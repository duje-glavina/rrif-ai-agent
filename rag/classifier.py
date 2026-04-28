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

CATEGORIES = Literal[
    "PDV",
    "dohodak",
    "doprinosi",
    "računovodstvo",
    "GDPR",
    "radno_pravo",
    "proračun",
    "ostalo",
]

SOURCE_TYPES = Literal["zakon", "članak", "FAQ", "priručnik", "transkript"]

SYSTEM_PROMPT = """\
Ti si klasifikator upita za sustav pretraživanja hrvatskog računovodstvenog i poreznog znanja.

Tvoj zadatak je analizirati korisnikovo pitanje i vratiti ISKLJUČIVO JSON objekt bez ikakvog teksta prije ili nakon.

JSON mora imati točno ovaj oblik:
{
  "category": "<jedna od: PDV, dohodak, doprinosi, računovodstvo, GDPR, radno_pravo, proračun, ostalo>",
  "time_period": {
    "type": "<jedna od: current, specific_date, range, historical>",
    "date_from": "<YYYY-MM-DD ili null>",
    "date_to": "<YYYY-MM-DD ili null>"
  },
  "source_type_preference": ["<zakon|članak|FAQ|priručnik|transkript>"] ili null
}

Pravila za category:
- PDV → pitanja o porezu na dodanu vrijednost, stopama, obrascima, rokovima PDV-a
- dohodak → porez na dohodak, osobni odbitak, dohodak od kapitala, dividende
- doprinosi → doprinosi za mirovinsko, zdravstveno, plaće, bruto/neto obračun
- računovodstvo → knjiženje, bilanca, financijski izvještaji, amortizacija, zalihe
- GDPR → zaštita osobnih podataka, obrada podataka, prava ispitanika
- radno_pravo → ugovori o radu, otkaz, godišnji odmor, radnička prava
- proračun → proračunski korisnici, neprofitne organizacije, javna nabava
- ostalo → sve ostalo što ne spada u gornje kategorije

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
            return "status = 'važeći'"
        if self.type in ("specific_date", "range"):
            df = self.date_from or "1900-01-01"
            # For specific_date, use date_from as the point-in-time reference.
            # A chunk is valid on date D if: valid_from <= D <= valid_to (or valid_to IS NULL)
            dt = self.date_to or self.date_from or "9999-12-31"
            return (
                f"valid_from <= '{df}'::date "
                f"AND COALESCE(valid_to, '9999-12-31'::date) >= '{dt}'::date"
            )
        # historical — no tight filter, search everything
        return None


# Maps classifier output categories to actual DB category values.
# DB categories come from article_loader.py banner detection.
_CATEGORY_MAP: dict = {
    "PDV":          "porezi",
    "dohodak":      "porezi",
    "doprinosi":    "plaće",
    "računovodstvo":"računovodstvo",
    "GDPR":         "ostalo",
    "radno_pravo":  "radno pravo",
    "proračun":     "proračun",
    "ostalo":       None,  # no filter — search everything
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
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

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
        "PDV", "dohodak", "doprinosi", "računovodstvo",
        "GDPR", "radno_pravo", "proračun", "ostalo",
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
        "Što piše u članku 99. Zakona o fiktivnom porezu na digitalne usluge?",
    ]
    for q in questions:
        result = classify(q)
        print(f"\nQ: {q}")
        print(f"   category  : {result.category}")
        print(f"   time      : {result.time_period}")
        print(f"   src_pref  : {result.source_type_preference}")
        print(f"   sql_filter: {result.time_period.to_sql_filter()}")
