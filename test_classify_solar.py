"""Diagnostic: compare ingest-time vs query-time classifier output on the
same text, to check whether the prompt gap explains the porezi/porezi
mis-tag on the solar panel article (RRiF 11/2024).

Run from your project root (where the rag/ package and .env live):
    python test_classify_solar.py
"""
import os
from dotenv import load_dotenv
load_dotenv()

from rag.classifier import classify as query_classify
from rag.ingest.pipeline import _classify_batch
import anthropic

# Representative excerpts from R241117.PDF (Petarčić, RRiF 11/2024,
# "Porezno motrište isporuke i ugradnje solarnih ploča u RH").
# Using a few different excerpts since chunking would have split this
# article into several pieces, and ingest only sees the first 600 chars
# of whatever chunk text it's given.
samples = {
    "intro_full_rates": (
        "Na isporuku i ugradnju solarnih ploča u Republici Hrvatskoj na "
        "privatne stambene objekte, prostore za stanovanje te javne i "
        "druge zgrade koje se koriste za aktivnosti od javnog interesa te "
        "isporuka i ugradnja solarnih ploča u blizini takvih objekata, "
        "prostora i zgrada primjenjuje se stopa PDV-a od 0%. U svim drugim "
        "slučajevima isporuka i ugradnja solarnih ploča oporeziva je "
        "PDV-om po stopi od 25% ili se na nju mora primijeniti tuzemni "
        "prijenos porezne obveze."
    ),
    "article2_pdv_mechanics": (
        "Primjenu stope PDV-a od 0% Zakon o PDV-u uređuje svega jednom "
        "rečenicom, koja glasi: PDV se obračunava i plaća po stopi od 0% "
        "na isporuku i ugradnju solarnih ploča na privatne stambene "
        "objekte, prostore za stanovanje te javne i druge zgrade koje se "
        "koriste za aktivnosti od javnog interesa te isporuku i ugradnju "
        "solarnih ploča u blizini takvih objekata, prostora i zgrada "
        "(čl. 38. st. 6.). Provedbu navedene odredbe propisuje čl. 47. "
        "st. 3. Pravilnika o PDV-u."
    ),
    "tuzemni_ppo_section": (
        "S obzirom na to da primatelji isporuke i ugradnje solarnih "
        "ploča mogu biti i porezni obveznici, bitno je istaknuti da se "
        "prilikom određivanja poreznog statusa prednost daje primjeni "
        "stope PDV-a od 0% u odnosu na primjenu tuzemnog prijenosa "
        "porezne obveze. To proizlazi iz čl. 151. st. 3. Pravilnika o "
        "PDV-u."
    ),
}

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=30.0)

for label, text in samples.items():
    print(f"\n{'=' * 70}\nSample: {label}\n{'=' * 70}")

    print("\n[query-time classifier — rag/classifier.py]")
    r = query_classify(text)
    print(f"  domain     : {r.domain}")
    print(f"  subdomains : {r.subdomains}")

    print("\n[ingest-time classifier — rag/ingest/pipeline.py]")
    batch = [(0, text)]
    result = _classify_batch(client, batch)
    print(f"  result: {result}")
