# Faza 1 — što je gotovo, što nedostaje

*Interna jednostranica · Duje · 9. 9. 2026. · podloga za sastanak 10. 9.*
*Izvori: TehSpec (sheet „F1 — Baza znanja"), Tehnički dio ponude v2.0 §3.1, Prijedlog kriterija prihvaćanja Faze 1*

## Rokovi

INFRA **15. 9.** (za 6 dana) · ingestija + metapodaci **18. 9.** · retrieval + reranker **25. 9.** · agent **1. 10.** · pilot sučelje + evaluacija **8. 10.** · **prihvaćanje F1 — 9. 10.**

## 1. Prag prihvaćanja se ne slaže sam sa sobom

| Dokument | Prag |
|---|---|
| Tehnički dio ponude §3.1 | ≥ 85 % prihvatljivih odgovora, trap refusal ≥ 95 % |
| TehSpec, sheet F1 | (Točno + Djelomično točno) ≥ 85 % na validacijskom setu |
| Kriteriji prihvaćanja, t. 5 | tablica **prazna**; navodi „postojeću odredbu o 90 % točnosti" |

**Zatvoriti 10. 9.** Sada postoji izmjerena brojka (90,3 % na našem setu) koja može poslužiti kao polazište. Bez upisanog praga prihvaćanje Faze 1 ostaje stvar tumačenja.

## 2. Ulazi koje čekamo od RRiF-a

| # | Materijal | Rok | Status |
|---|---|---|---|
| 1 | Interna literatura (PIP, RRIF, knjige, priručnici, pročišćeni zakoni) | 31. 8. | **kasni 9 dana** — pristup SharePointu se tek dogovara; u bazi samo 2014. i 2024. |
| 2 | Metapodaci po članku (XLS/CSV: kategorija + ključne riječi) | 31. 8. | **kasni 9 dana** — nije zatražen ni uzorak od 5–10 zapisa |
| 3 | Kalibracijski set | 20. 9. | nije stigao; radimo na vlastitih 41 + njihovih 27 pitanja iz PoC-a |
| 4 | Validacijski set (zamrznut) | 30. 9. | nije stigao; tražimo 150–200 pitanja umjesto 100, uz izvor po pitanju |

Sve četiri stavke su na kritičnom putu. Bez t. 1 i t. 2 komponente 1.2 i cijela kalibracija stoje.

## 3. Komponente 1.1 – 1.11

**Gotovo:** 1.1 extraction pipeline (prerađen 7. 9.) · 1.4 embedding + HNSW · 1.5 hybrid retrieval u jednom CTE upitu · 1.7 klasifikator (Haiku, temperatura 0, 100 % na golden setu) · 1.9 generator s obveznim citiranjem

**Uz zadršku na gotovim stavkama:**
- `to_tsvector('croatian', …)` iz scheme **ne postoji ni u jednom PG buildu** — maknuti prije prve migracije
- retrieval daje 20 kandidata → 10; specifikacija traži 50 → 10
- klasifikator: kategorije trebaju doći iz RRiF XLS-a (čeka t. 2); prag pouzdanosti nije implementiran kao gate; neispravan JSON na 2 od 27 stvarnih pitanja
- generator: `Citation.source` se puni iz `law_name`, koji je NULL na svakom chunku članka → objekti citata su vjerojatno prazni. **Popraviti prije nego RRiF vidi odgovor.** Ostaje i streaming, prompt caching, poštovanje `citable` flaga

**Djelomično:**
- **1.3 Temporalno versioniranje** — najveća strukturna rupa. Traži se `valid_from` / `valid_until` i prikaz **OD-DO**, a ne „nevažeće od" (ponuda §3.2 kaže suprotno — uskladiti). Danas je važenje otisnuto na chunkove i tvrdi neistinu. Prvi korak: tablica `documents` i pisanje `document_id` pri ingestiji.
- **1.10 Pilot sučelje** — postoji `index.html` + FastAPI s JWT loginom i feedback endpointom. Proći kroz popis: prikaz izvora, povijest upita, palac + komentar.
- **1.11 Evaluacijska infrastruktura** — harness je jak, ali vrti **jedan** set. Traže se dva odvojena (kalibracijski, validacijski) i izvještaj po kategorijama.

**Nije započeto:** 1.2 parser metapodataka (čeka RRiF) · **1.6 reranker preko EU API endpointa** (danas lokalni BGE; benchmark Cohere vs Jina na 50 upita; preduvjet — jedan rerank poziv po pitanju — riješen 5. 9., dakle odblokirano) · 1.8 Query Rewriter (radi se, necommitano od sinoć)

## 4. INFRA — najbliži rok, najmanje napravljeno

Hetzner AX41, PostgreSQL + pgvector, Docker, backup, monitoring do **15. 9.** Sve danas vrti na Omenu. U sheetu baza još stoji kao OTVORENO iako je odluka pala (samohostano, Unix socket).

**AX41 nema GPU.** BGE nad 50 kandidata na 6 jezgri su sekunde, ne milisekunde — odluka o serveru i odluka o rerankeru su ista odluka. Prijedlog ostaje GEX44 (~184 €/mj) barem za ingestiju i F3. Jedina stavka koja mijenja mjesečni trošak.

## 5. Za sastanak 10. 9.

1. **Prag prihvaćanja** — upisati brojke u t. 5 (85 ili 90, i po kojoj skali).
2. **Zakoni u opsegu F1?** Svih 6 poreznih pitanja daje točan odgovor, ali iz RRiF-ovih članaka, ne iz teksta zakona. Odgovor mijenja opseg i F1 i F2.
3. **Koje godine u bazi?** Danas 2014. i 2024., s deset godina rupe između.
4. **Materijali i metapodaci** — pristup SharePointu i uzorak XLS-a, oba su trebala stići 31. 8.
5. **Validacijski set** — 150–200 pitanja, izvor uz svako, mehanika zamrzavanja.
6. **Probno ocjenjivanje 31 odgovora** — dvije tablice su spremne, ~1 h po savjetniku.
