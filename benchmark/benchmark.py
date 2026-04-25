#!/usr/bin/env python3
"""
RRiF AI Agent — LLM Benchmark Tool
====================================
Tests Claude, GPT-4o, and Llama on identical RAG conditions.
Outputs a CSV + JSON for use in your proposal.

Usage:
    python benchmark.py                   # run all models
    python benchmark.py --models claude   # run only Claude
    python benchmark.py --models claude gpt4o
"""

import time
import json
import csv
import os
import argparse
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
import anthropic
from openai import OpenAI
import requests

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "llama3.1")   # change to llama3.2 etc.

TOP_K = 3  # number of chunks retrieved per question

# ── SAMPLE KNOWLEDGE BASE ──────────────────────────────────────────────────────
# In a real setup, replace these with actual RRiF articles / Narodne novine texts.
# This sample is sufficient to demonstrate the benchmark methodology.

DOCUMENTS = [
    {
        "id": "pdv-rokovi",
        "source": "Zakon o PDV-u, NN 73/13, čl. 85",
        "year": 2023,
        "active": True,
        "text": (
            "Porezni obveznik mora predati PDV obrazac (Obrazac PDV) do 20. dana u mjesecu koji "
            "slijedi nakon isteka obračunskog razdoblja. Obračunsko razdoblje je kalendarski "
            "mjesec. Porezni obveznici čija je vrijednost isporuka u prethodnoj kalendarskoj "
            "godini bila niža od 800.000 kuna (106.080 EUR) mogu birati tromjesečno "
            "obračunsko razdoblje uz prethodno odobrenje Porezne uprave."
        )
    },
    {
        "id": "pdv-stope",
        "source": "Zakon o PDV-u, NN 73/13 s izmjenama, čl. 38",
        "year": 2024,
        "active": True,
        "text": (
            "Opća stopa PDV-a iznosi 25%. Snižena stopa PDV-a od 13% primjenjuje se na usluge "
            "smještaja ili smještaja s doručkom u hotelima ili objektima slične namjene, te na "
            "novine i časopise. Snižena stopa od 5% primjenjuje se na knjige stručnog, "
            "znanstvenog, umjetničkog, kulturnog i obrazovnog sadržaja te udžbenike."
        )
    },
    {
        "id": "dohodak-kapital-2021",
        "source": "Zakon o porezu na dohodak, NN 115/16, čl. 66 — verzija važeća 2021.",
        "year": 2021,
        "active": False,
        "text": (
            "NEVAŽEĆE OD 1.1.2024. — U 2021. godini porez na dohodak od kapitala "
            "(dividende i udjeli u dobiti) oporezivao se stopom od 12% uvećanom za prirez. "
            "Osnovni osobni odbitak iznosio je 4.000 kuna mjesečno (48.000 kuna godišnje). "
            "Porezne stope bile su 24% za dohodak do 360.000 kuna i 36% za iznos iznad toga."
        )
    },
    {
        "id": "dohodak-kapital-2024",
        "source": "Zakon o porezu na dohodak, NN 114/23, čl. 66 — važeće od 1.1.2024.",
        "year": 2024,
        "active": True,
        "text": (
            "VAŽEĆE OD 1.1.2024. — Porez na dohodak od kapitala (dividende i udjeli u dobiti) "
            "oporezuje se stopom od 10%. Osnovni osobni odbitak iznosi 560 EUR mjesečno "
            "(6.720 EUR godišnje). Porezne stope su 20% za dohodak do 50.400 EUR i 30% za "
            "iznos iznad toga. Prirez porezu na dohodak ukinut je od 1. siječnja 2024."
        )
    },
    {
        "id": "place-doprinosi",
        "source": "Zakon o doprinosima, NN 84/08 s izmjenama, čl. 10 i 12",
        "year": 2024,
        "active": True,
        "text": (
            "Doprinosi iz plaće na teret radnika: doprinos za mirovinsko osiguranje I. stup "
            "iznosi 15%, doprinos za mirovinsko osiguranje II. stup iznosi 5% — ukupno 20%. "
            "Doprinosi na plaću na teret poslodavca: doprinos za zdravstveno osiguranje iznosi "
            "16,5%. Minimalna plaća u 2024. godini iznosi 840 EUR bruto."
        )
    },
    {
        "id": "racunovodstvo-razvrstavanja",
        "source": "Zakon o računovodstvu, NN 78/15 s izmjenama, čl. 5",
        "year": 2023,
        "active": True,
        "text": (
            "Poduzetnici se razvrstavaju u mikro, male, srednje i velike kategorije. "
            "Mikro poduzetnici: aktiva do 2.000.000 EUR, prihodi do 4.000.000 EUR, "
            "prosječno do 10 radnika. Mali poduzetnici: aktiva do 10.000.000 EUR, "
            "prihodi do 20.000.000 EUR, prosječno do 50 radnika. Srednji poduzetnici: "
            "aktiva do 43.000.000 EUR, prihodi do 50.000.000 EUR, do 250 radnika."
        )
    },
    {
        "id": "gdpr-info-obveze",
        "source": "GDPR — Uredba (EU) 2016/679, čl. 13",
        "year": 2023,
        "active": True,
        "text": (
            "Voditelj obrade dužan je ispitaniku u trenutku prikupljanja osobnih podataka "
            "pružiti: identitet i kontakt voditelja obrade, svrhu i pravnu osnovu obrade, "
            "primatelje osobnih podataka, rok čuvanja podataka. Ispitanik ima pravo na "
            "pristup, ispravak, brisanje (pravo na zaborav) i prenosivost podataka."
        )
    },
]

# ── TEST QUESTIONS ──────────────────────────────────────────────────────────────

QUESTIONS = [
    {
        "id": "Q01",
        "category": "PDV — rokovi",
        "question": "Koji je rok za predaju PDV obrasca?",
        "expected_keywords": ["20", "20. dana", "mjesec"],
        "expects_citation": True,
        "is_trap": False,
        "ground_truth": "Do 20. dana u mjesecu koji slijedi nakon obračunskog razdoblja.",
    },
    {
        "id": "Q02",
        "category": "PDV — stope",
        "question": "Kolika je opća stopa PDV-a i postoje li snižene stope u Hrvatskoj?",
        "expected_keywords": ["25%", "13%", "5%"],
        "expects_citation": True,
        "is_trap": False,
        "ground_truth": "Opća stopa 25%, snižene stope 13% i 5%.",
    },
    {
        "id": "Q03",
        "category": "Temporalna usporedba ★",
        "question": "Kako se oporezivao porez na dohodak od kapitala 2021. u usporedbi s danas?",
        "expected_keywords": ["12%", "10%", "2021", "2024", "prirez"],
        "expects_citation": True,
        "is_trap": False,
        "ground_truth": "2021: stopa 12% + prirez. Danas (od 2024): stopa 10%, prirez ukinut.",
    },
    {
        "id": "Q04",
        "category": "Plaće — doprinosi",
        "question": "Koliki su doprinosi na plaću u Hrvatskoj? Navedite stope za radnika i poslodavca.",
        "expected_keywords": ["20%", "16,5%", "mirovinsko", "zdravstveno"],
        "expects_citation": True,
        "is_trap": False,
        "ground_truth": "Radnik: 20% mirovinsko. Poslodavac: 16,5% zdravstveno.",
    },
    {
        "id": "Q05",
        "category": "Računovodstvo — razvrstavanja",
        "question": "Koji su kriteriji za razvrstavanje mikro poduzetnika?",
        "expected_keywords": ["2.000.000", "4.000.000", "10 radnika"],
        "expects_citation": True,
        "is_trap": False,
        "ground_truth": "Aktiva do 2M EUR, prihodi do 4M EUR, max 10 radnika.",
    },
    {
        "id": "Q06",
        "category": "GDPR",
        "question": "Koje informacije mora pružiti voditelj obrade pri prikupljanju osobnih podataka?",
        "expected_keywords": ["identitet", "svrha", "pravna osnova", "rok čuvanja"],
        "expects_citation": True,
        "is_trap": False,
        "ground_truth": "Identitet voditelja, svrha i pravna osnova, primatelji, rok čuvanja.",
    },
    {
        "id": "Q07",
        "category": "Trap — izvan baze znanja ★",
        "question": "Koji je rok za predaju godišnjeg financijskog izvještaja za velika poduzeća?",
        "expected_keywords": [],
        "expects_citation": False,
        "is_trap": True,
        "ground_truth": "Odgovor nije u bazi — agent treba reći da ne zna i uputiti na savjetnika.",
    },
    {
        "id": "Q08",
        "category": "Trap — izmišljeni zakon ★",
        "question": "Što piše u članku 99. Zakona o fiktivnom porezu na digitalne usluge iz 2019.?",
        "expected_keywords": [],
        "expects_citation": False,
        "is_trap": True,
        "ground_truth": "Ovaj zakon ne postoji — model ne smije izmišljati odgovor.",
    },
]

# ── SYSTEM PROMPT (identical for all models) ───────────────────────────────────

SYSTEM_PROMPT = """Ti si AI agent koji pomaže savjetnicima i korisnicima RRiF-plus d.o.o.
s pitanjima iz područja računovodstva, poreza i financija.

PRAVILA KOJA MORAS STROGO POŠTOVATI:
1. Odgovaraj ISKLJUČIVO na temelju dokumenata koji su ti dostavljeni kao kontekst.
2. Svaki odgovor mora sadržavati navod izvora: naziv dokumenta/zakona, članak i godinu.
3. Ako odgovor nije pronađen u dostavljenim dokumentima, eksplicitno navedi da ne možeš
   odgovoriti i uputi korisnika na RRiF savjetničku liniju.
4. NIKADA ne izmišljaj odgovore, zakone, članke ili brojeve koji nisu u kontekstu.
5. Jasno razlikuj važeće i nevažeće/stare propise ako su oba prisutna u kontekstu.
6. Odgovaraj na hrvatskom jeziku."""

# ── VECTOR DB SETUP ───────────────────────────────────────────────────────────

def setup_vectordb(documents: list) -> chromadb.Collection:
    print("  Loading embedding model (paraphrase-multilingual-mpnet-base-v2)...")
    client = chromadb.Client()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-mpnet-base-v2"
    )
    collection = client.create_collection(
        name="rrif_kb",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[d["id"] for d in documents],
        documents=[d["text"] for d in documents],
        metadatas=[{"source": d["source"], "year": d["year"], "active": d["active"]} for d in documents],
    )
    return collection


def retrieve_context(collection: chromadb.Collection, question: str) -> str:
    results = collection.query(query_texts=[question], n_results=TOP_K)
    parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        status = "VAŽEĆE" if meta["active"] else "NEVAŽEĆE"
        parts.append(f"[{status} | Izvor: {meta['source']} ({meta['year']})]\n{doc}")
    return "\n\n".join(parts)

# ── LLM CALLS ─────────────────────────────────────────────────────────────────

def call_claude(question: str, context: str) -> dict:
    if not ANTHROPIC_API_KEY:
        return _error_result("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"Kontekst iz baze znanja:\n{context}\n\nPitanje: {question}"
    start = time.time()
    try:
        r = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "response": r.content[0].text,
            "tokens_in": r.usage.input_tokens,
            "tokens_out": r.usage.output_tokens,
            "latency": round(time.time() - start, 2),
            "error": False,
        }
    except Exception as e:
        return _error_result(str(e))


def call_gpt4o(question: str, context: str) -> dict:
    if not OPENAI_API_KEY:
        return _error_result("OPENAI_API_KEY not set")
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"Kontekst iz baze znanja:\n{context}\n\nPitanje: {question}"
    start = time.time()
    try:
        r = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=800,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return {
            "response": r.choices[0].message.content,
            "tokens_in": r.usage.prompt_tokens,
            "tokens_out": r.usage.completion_tokens,
            "latency": round(time.time() - start, 2),
            "error": False,
        }
    except Exception as e:
        return _error_result(str(e))


def call_llama(question: str, context: str) -> dict:
    prompt = f"Kontekst iz baze znanja:\n{context}\n\nPitanje: {question}"
    start = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            timeout=120,
        )
        data = r.json()
        return {
            "response": data["message"]["content"],
            "tokens_in": data.get("prompt_eval_count", 0),
            "tokens_out": data.get("eval_count", 0),
            "latency": round(time.time() - start, 2),
            "error": False,
        }
    except Exception as e:
        return _error_result(str(e))


def _error_result(msg: str) -> dict:
    return {"response": msg, "tokens_in": 0, "tokens_out": 0, "latency": 0, "error": True}

# ── SCORING ───────────────────────────────────────────────────────────────────

def score(response_text: str, question: dict) -> dict:
    t = response_text.lower()

    citation_terms = ["izvor", "zakon", "čl.", "članak", "nn ", "uredba",
                      "pravilnik", "priručnik", "čl "]
    has_citation = any(term in t for term in citation_terms)

    refusal_terms = ["ne mogu", "nije u", "nemam", "ne znam", "nije dostupno",
                     "savjetnik", "savjetničk", "nije pronađen", "ne nalazim",
                     "nije u bazi", "nije dostupan"]
    has_refusal = any(term in t for term in refusal_terms)

    kws = question["expected_keywords"]
    kw_hits = sum(1 for kw in kws if kw.lower() in t)
    kw_score = kw_hits / len(kws) if kws else None

    if question["is_trap"]:
        accuracy = "✅ Refused" if has_refusal else "❌ Hallucinated"
    elif kw_score is not None:
        if kw_score >= 0.75:
            accuracy = "✅ Good"
        elif kw_score >= 0.4:
            accuracy = "⚠️  Partial"
        else:
            accuracy = "❌ Poor"
    else:
        accuracy = "👁  Manual"

    return {
        "citation": "✅" if has_citation else "❌",
        "kw_score": f"{kw_score:.0%}" if kw_score is not None else "—",
        "accuracy": accuracy,
    }

# ── COST LOOKUP ───────────────────────────────────────────────────────────────
# Prices per 1K tokens as of mid-2025 (update if needed)
COST = {
    "Claude":  {"in": 0.003,  "out": 0.015},
    "GPT-4o":  {"in": 0.0025, "out": 0.01},
    "Llama":   {"in": 0.0,    "out": 0.0},
}

def est_cost(model: str, tin: int, tout: int) -> str:
    if model not in COST:
        return "—"
    c = (tin / 1000 * COST[model]["in"]) + (tout / 1000 * COST[model]["out"])
    return f"${c:.5f}"

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(selected_models: list[str]):
    all_models = {
        "claude": ("Claude",  call_claude),
        "gpt4o":  ("GPT-4o",  call_gpt4o),
        "llama":  ("Llama",   call_llama),
    }
    models = {k: v for k, v in all_models.items() if k in selected_models}

    print("\n" + "="*70)
    print("  RRiF AI Agent — LLM Benchmark")
    print("="*70)
    print(f"  Models:    {', '.join(v[0] for v in models.values())}")
    print(f"  Questions: {len(QUESTIONS)}  ({sum(1 for q in QUESTIONS if q['is_trap'])} trap)")
    print(f"  KB docs:   {len(DOCUMENTS)}")
    print("="*70 + "\n")

    print("🔧 Setting up vector database...")
    collection = setup_vectordb(DOCUMENTS)
    print(f"✅ Ready.\n")

    rows = []

    for q in QUESTIONS:
        trap_label = " [TRAP]" if q["is_trap"] else ""
        print(f"{'─'*60}")
        print(f"📋 {q['id']} — {q['category']}{trap_label}")
        print(f"   Q: {q['question']}")
        context = retrieve_context(collection, q["question"])

        for key, (model_name, fn) in models.items():
            print(f"   → {model_name}...", end=" ", flush=True)
            result = fn(q["question"], context)
            s = score(result["response"], q)
            cost = est_cost(model_name, result["tokens_in"], result["tokens_out"])

            if result["error"]:
                print(f"⚠️  ERROR: {result['response']}")
            else:
                print(f"{s['accuracy']} | cite:{s['citation']} | {result['latency']}s | {cost}")

            rows.append({
                "question_id":    q["id"],
                "category":       q["category"],
                "is_trap":        q["is_trap"],
                "model":          model_name,
                "latency_s":      result["latency"],
                "tokens_in":      result["tokens_in"],
                "tokens_out":     result["tokens_out"],
                "est_cost_usd":   cost,
                "has_citation":   s["citation"],
                "keyword_score":  s["kw_score"],
                "accuracy":       s["accuracy"],
                "response_preview": result["response"][:200].replace("\n", " "),
                "full_response":  result["response"],
                "error":          result["error"],
            })
        print()

    # ── SAVE ──────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file  = f"benchmark_{ts}.csv"
    json_file = f"benchmark_{ts}.json"

    csv_cols = [c for c in rows[0].keys() if c != "full_response"]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in csv_cols})

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)

    for key, (model_name, _) in models.items():
        mr = [r for r in rows if r["model"] == model_name and not r["error"]]
        if not mr:
            print(f"\n  {model_name}: no results (error / not configured)")
            continue

        good    = sum(1 for r in mr if "✅" in r["accuracy"])
        partial = sum(1 for r in mr if "⚠️" in r["accuracy"])
        poor    = sum(1 for r in mr if "❌ Poor" in r["accuracy"] or "❌ Hall" in r["accuracy"])
        manual  = sum(1 for r in mr if "👁" in r["accuracy"])
        cites   = sum(1 for r in mr if r["has_citation"] == "✅")
        traps   = [r for r in mr if r["is_trap"]]
        refused = sum(1 for r in traps if "Refused" in r["accuracy"])
        avg_lat = sum(r["latency_s"] for r in mr) / len(mr)

        # cost projection for 500 queries/day × 30 days
        total_tin  = sum(r["tokens_in"] for r in mr)
        total_tout = sum(r["tokens_out"] for r in mr)
        avg_tin    = total_tin  / len(mr) if mr else 0
        avg_tout   = total_tout / len(mr) if mr else 0
        daily_500  = (500 * avg_tin / 1000 * COST[model_name]["in"] +
                      500 * avg_tout / 1000 * COST[model_name]["out"])
        monthly    = daily_500 * 30

        print(f"\n  ┌─ {model_name} {'─'*(40 - len(model_name))}")
        print(f"  │  Accuracy    ✅ {good} good  ⚠️  {partial} partial  ❌ {poor} poor  👁  {manual} manual")
        print(f"  │  Citations   {cites}/{len(mr)} responses cited a source")
        print(f"  │  Trap tests  {refused}/{len(traps)} hallucinations avoided")
        print(f"  │  Avg latency {avg_lat:.2f}s")
        if COST[model_name]["in"] > 0:
            print(f"  │  Cost est.   ~${monthly:.2f}/month @ 500 queries/day")
        else:
            print(f"  │  Cost est.   $0 API cost (infrastructure cost separate)")
        print(f"  └{'─'*45}")

    print(f"\n  Files saved:")
    print(f"  📊 {csv_file}   ← open in Excel")
    print(f"  📄 {json_file}  ← full responses")
    print(f"\n  Questions marked 👁 Manual require human domain expert review.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        choices=["claude", "gpt4o", "llama"],
        default=["claude", "gpt4o", "llama"],
        help="Which models to benchmark (default: all three)",
    )
    args = parser.parse_args()
    run(args.models)
