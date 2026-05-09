"""Pre-retrieval query classifier — v2.

Returns domain + subdomain (possibly multiple subdomains for broad queries)
using the two-level taxonomy defined in rag/taxonomy.py.

Changes from v1:
  - Returns ClassifierResult with domain + subdomains list (not single category)
  - to_retrieval_filter() emits WHERE domain=X AND subdomain=ANY(...)
  - Fallback: if subdomain query returns nothing, widen to full domain
  - novi_propisi retired — freshness expressed via recency_tag signal
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import anthropic
from dotenv import load_dotenv

from rag.taxonomy import (
    SUBDOMAIN_TO_DOMAIN,
    VALID_SUBDOMAINS,
    prompt_taxonomy_block,
    RECENCY_TAGS,
)

load_dotenv()
log = logging.getLogger(__name__)

CLASSIFIER_MODEL = "claude-haiku-4-5"
STATUS_VALID = "vazeci"

SYSTEM_PROMPT = f"""Ti si klasifikator upita za sustav pretraživanja hrvatskog računovodstvenog i poreznog znanja.

Tvoj zadatak: analizirati korisnikovo pitanje i vratiti ISKLJUČIVO JSON objekt.

Taksonomija (domain → subdomain):
{prompt_taxonomy_block()}

JSON oblik:
{{
  "subdomains": ["<subdomain1>", "<subdomain2>"],
  "time_period": {{
    "type": "<current|specific_date|range|historical>",
    "date_from": "<YYYY-MM-DD ili null>",
    "date_to": "<YYYY-MM-DD ili null>"
  }},
  "recency_boost": <true|false>,
  "source_type_preference": ["zakon"] ili null
}}

Pravila za subdomains:
- Navedi 1–3 subdomaina poredana po relevantnosti (najprecizniji prvi).
- Ako pitanje jasno spada pod jedan subdomain, navedi samo jedan.
- Ako pitanje pokriva više subdomena (npr. "PDV i porez na dohodak"), navedi oba.
- Za pitanja van tematskog opsega (strani porezi, GDPR, IT) navedi ["ostalo"].
- Koristi ISKLJUČIVO vrijednosti iz taksonomije — nikad ne izmišljaj nove.

Primjeri:
Pitanje: "Kolika je stopa PDV-a na hranu?"
{{"subdomains": ["PDV"], "time_period": {{"type": "current", "date_from": null, "date_to": null}}, "recency_boost": false, "source_type_preference": null}}

Pitanje: "Koji su doprinosi iz plaće na teret radnika?"
{{"subdomains": ["plaće"], "time_period": {{"type": "current", "date_from": null, "date_to": null}}, "recency_boost": false, "source_type_preference": null}}

Pitanje: "Što je osobni odbitak u porezu na dohodak?"
{{"subdomains": ["dohodak"], "time_period": {{"type": "current", "date_from": null, "date_to": null}}, "recency_boost": false, "source_type_preference": null}}

Pitanje: "Kako se osniva d.o.o.?"
{{"subdomains": ["trg_pravo"], "time_period": {{"type": "current", "date_from": null, "date_to": null}}, "recency_boost": false, "source_type_preference": null}}

Pitanje: "Koji su kriteriji za razvrstavanje mikro poduzetnika?"
{{"subdomains": ["fin_izv", "knjiženje"], "time_period": {{"type": "current", "date_from": null, "date_to": null}}, "recency_boost": false, "source_type_preference": null}}

Pitanje: "Koje su nove izmjene PDV zakona za 2025.?"
{{"subdomains": ["PDV"], "time_period": {{"type": "specific_date", "date_from": "2025-01-01", "date_to": null}}, "recency_boost": true, "source_type_preference": ["zakon"]}}

Pitanje: "Kako se oporezuju dividende u Hrvatskoj?"
{{"subdomains": ["dohodak", "dobit"], "time_period": {{"type": "current", "date_from": null, "date_to": null}}, "recency_boost": false, "source_type_preference": null}}

Pitanje: "Kolika je stopa poreza na nasljedstvo u Sloveniji?"
{{"subdomains": ["ostalo"], "time_period": {{"type": "current", "date_from": null, "date_to": null}}, "recency_boost": false, "source_type_preference": null}}
"""


@dataclass
class TimePeriod:
    type: str = "current"
    date_from: str | None = None
    date_to: str | None = None

    def to_sql_filter(self) -> str | None:
        if self.type == "current":
            return f"status = '{STATUS_VALID}'"
        if self.type in ("specific_date", "range"):
            df = self.date_from or "1900-01-01"
            dt = self.date_to or self.date_from or "9999-12-31"
            return (
                f"valid_from <= '{df}'::date "
                f"AND COALESCE(valid_to, '9999-12-31'::date) >= '{dt}'::date"
            )
        return None


@dataclass
class ClassifierResult:
    subdomains: list[str] = field(default_factory=lambda: ["ostalo"])
    domain: str = "ostalo"          # derived from primary subdomain
    time_period: TimePeriod = field(default_factory=TimePeriod)
    recency_boost: bool = False
    source_type_preference: list[str] | None = None
    raw: dict = field(default_factory=dict, repr=False)

    # Legacy compatibility — eval grader still reads .category
    @property
    def category(self) -> str:
        return self.subdomains[0] if self.subdomains else "ostalo"

    def to_retrieval_filter(self) -> dict:
        """Return kwargs for the hybrid_search filter.

        Primary filter:  domain + subdomain list (tight)
        Fallback filter: domain only (wider — used when tight returns nothing)
        """
        return {
            "domain": self.domain,
            "subdomains": self.subdomains,
            "sql_time_filter": self.time_period.to_sql_filter(),
            "source_types": self.source_type_preference,
            "recency_boost": self.recency_boost,
        }

    def fallback_filter(self) -> dict:
        """Wider filter — domain only, no subdomain constraint."""
        return {
            "domain": self.domain,
            "subdomains": None,
            "sql_time_filter": self.time_period.to_sql_filter(),
            "source_types": self.source_type_preference,
            "recency_boost": self.recency_boost,
        }


def classify(question: str) -> ClassifierResult:
    """Classify a user question. Returns ClassifierResult with domain + subdomains."""
    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        timeout=15.0,
    )
    try:
        response = client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
        )
        raw_text = response.content[0].text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(raw_text)
        return _parse(data)
    except json.JSONDecodeError as e:
        log.warning("Classifier returned invalid JSON: %s", e)
        return ClassifierResult()
    except Exception as e:
        log.warning("Classifier API call failed: %s", e)
        return ClassifierResult()


def _parse(data: dict) -> ClassifierResult:
    raw_subdomains = data.get("subdomains", ["ostalo"])
    if isinstance(raw_subdomains, str):
        raw_subdomains = [raw_subdomains]

    # Validate and filter to known values
    subdomains = [s for s in raw_subdomains if s in VALID_SUBDOMAINS]
    if not subdomains:
        subdomains = ["ostalo"]

    # Derive domain from primary subdomain
    domain = SUBDOMAIN_TO_DOMAIN.get(subdomains[0], "ostalo")

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
        valid_src = {"zakon", "članak", "FAQ", "priručnik", "transkript"}
        stp = [s for s in stp if s in valid_src] or None
    else:
        stp = None

    return ClassifierResult(
        subdomains=subdomains,
        domain=domain,
        time_period=time_period,
        recency_boost=bool(data.get("recency_boost", False)),
        source_type_preference=stp,
        raw=data,
    )


if __name__ == "__main__":
    import sys
    questions = sys.argv[1:] or [
        "Kolika je stopa PDV-a na hranu od 2024.?",
        "Koji su doprinosi iz plaće na teret radnika?",
        "Što je osobni odbitak u porezu na dohodak?",
        "Kako se osniva d.o.o.?",
        "Koji su kriteriji za razvrstavanje mikro poduzetnika?",
        "Kolika je stopa poreza na nasljedstvo u Sloveniji?",
        "Kako se oporezuju dividende?",
        "Koji je rok za predaju PDV obrasca?",
    ]
    for q in questions:
        r = classify(q)
        print(f"\nQ: {q}")
        print(f"   domain     : {r.domain}")
        print(f"   subdomains : {r.subdomains}")
        print(f"   recency    : {r.recency_boost}")
        print(f"   time       : {r.time_period.type}")
