# Osvrt na TehSpec v2.0 — što zatvaramo, što se sudara

*Interni radni dokument · Duje · 4. 9. 2026. · podloga za odgovor Stipi*

Stipe traži komentar na sheet „Pregled". Ovaj tjedan smo mjerili postojeći PoC i tri
otvorena pitanja iz specifikacije sada imaju odgovor, a dvije „Odlučeno" stavke se
sudaraju s onim što smo izmjerili.

---

## 1. Otvorena pitanja koja možemo zatvoriti odmah

### 1.1. „Croatian" FTS konfiguracija u PostgreSQL-u — **ne postoji**

Otvoreno pitanje #1 i redak u schemi:

```sql
tsv tsvector GENERATED (to_tsvector('croatian', title || ' ' || body)) STORED
```

**Taj redak neće proći.** PostgreSQL ne isporučuje `croatian` konfiguraciju ni u
jednom buildu — nema je u `pg_ts_config`, nema hrvatskog Snowball stemmera.
Trenutni PoC zato vrti `to_tsvector('simple', …)` i morfologiju kompenzira ručno
održavanom listom stopwordova u `rag/retrieve/hybrid.py`.

Dvije opcije, obje testirane u planu:

- **hunspell rječnik** — datoteke u `$SHAREDIR/tsearch_data`, traži superusera i
  pristup datotečnom sustavu. Radi samo na samohostanom PG-u.
- **stemming u aplikaciji** — lematizacija u Pythonu (classla) u zaseban stupac,
  indeksiran s `simple`. Radi bilo gdje, uključujući managed PG.

Skripte za obje varijante su napisane (`rag/stem_hr.py`, `scripts/exp_stem_corpus.py`).
Prijedlog: testirati aplikacijsku varijantu prvo — jednostavnija je i ne veže nas
za samohostani PG.

→ **Status: zatvoreno. Treba samo maknuti `'croatian'` iz scheme prije nego netko
pokrene migraciju.**

### 1.2. Baza — samohostano ili managed: **samohostano**

Redak je označen kao OTVOREN. Tri stvari koje smo u međuvremenu saznali zatvaraju ga:

- **Nema postotka dostupnosti.** Tvoj mail od 2. 9. kaže: *„Uptime ne možemo ni
  definirati ni garantirati."* SLA je 20 h/mj održavanja, ne dostupnost. Time nestaje
  jedini ozbiljan argument za managed PG — automatski failover.
- **Baza je malena.** 12.749 chunkova danas ≈ 250 MB ukupno. Uz 10 godina građe
  (~127.000 chunkova) to je ~1,5 GB. Nema scenarija u kojem RAM postaje trošak.
- **Hrvatski FTS rječnik** (t. 1.1) traži pristup datotečnom sustavu ako idemo tim
  putem. Managed PG to zaključava.

Uz to: jedan podobrađivač umjesto dva, jedan DPA, sve u Njemačkoj — čišći papir
prema RRiF-ovom DPO-u, a tender ionako traži pohranu isključivo u EU.

→ **Status: zatvoreno. PostgreSQL na istom boxu, Unix socket.**

*Sitnica: schema kaže PG 16, postojeća baza je 17.9. Uskladiti prije INFRA faze.*

---

## 2. Dvije „Odlučeno" stavke koje se sudaraju s mjerenjima

### 2.1. Reranker preko API-ja **ne stane u latencijski budžet**

VOICE sheet daje **200 ms za „Retrieval + reranker"**, a F4 kriterij prihvaćanja
traži **prvi token < 2 s** na 500 upita.

Problem nije sam poziv, nego koliko ih ima. `_retrieve` danas ima troslojni
fallback (tight → domain → wide), a na svakoj razini zove `_top_rerank_score` prije
nego odluči ide li dalje — pa tek onda finalni rerank. **U najgorem slučaju četiri
sekvencijalna rerank poziva po pitanju.**

Lokalno je to jedan batchani forward pass i praktički besplatno. Preko API-ja su to
četiri mrežna round-tripa: na 200–500 ms svaki, **0,8–2,0 s samo na rerank**. Budžet
od 200 ms nije blizu.

Dvije stvari treba napraviti, i prva je obavezna bez obzira na izbor providera:

1. **Svesti rerank na jedan poziv po pitanju.** Fallback može odlučivati po broju
   kandidata, a kvalitetu provjeravati tek jednom na kraju.
2. Tek onda birati providera.

### 2.2. AX41-NVMe nema GPU, a e5 + BGE lokalno traže ga

Server je „Odlučeno": AX41-NVMe, Ryzen 5 3600, 6 jezgri, 64 GB RAM, bez GPU-a.
Na tom stroju:

- **e5-large embedding** na CPU-u: ~50–150 ms po upitu. Prihvatljivo.
- **BGE-reranker-v2-m3** nad 50 kandidata na 6 jezgri: **sekunde, ne milisekunde.**
  Specifikacija traži rerank ulaz top-50 → izlaz top-10, što je više nego danas
  (20 → 5).

Dakle lokalni reranker na tom boxu ne stane u budžet, a API reranker ne stane dok
se ne riješi t. 2.1. Kad se t. 2.1 riješi, oba postaju izvediva i izbor je čista
kalkulacija:

| | Trošak | Napomena |
|---|---|---|
| **GEX44** (RTX 4000 Ada, 20 GB) | ~184 €/mj | Vrti embedding + rerank + Whisper za F3. Sve ostaje u Njemačkoj, jedan DPA. |
| **API reranker** (Cohere/Jina) | po pozivu | Jedan mrežni skok u vrućoj putanji, još jedan podobrađivač. |

GEX44 usput rješava i F3: STT arhive na CPU-u je noć posla po seriji, na GPU-u
minute. Preporuka: **uzeti GEX44 barem za razdoblje ingestije i F3**, bez minimalnog
trajanja se otkazuje kad prestane trebati.

→ **Za odluku: ovo je jedina stavka u STACK-u koja mijenja mjesečni trošak, i
jedina koja može srušiti kriterij prihvaćanja F4.**

---

## 3. Što smo izmjerili ovaj tjedan, a mijenja sliku F1

### 3.1. Sastav korpusa: 98,5 % članci, 1,5 % zakon

| source_type | status | broj |
|---|---|---|
| članak | vazeci | 7.387 |
| članak | nevazeci | 5.174 |
| **zakon** | **vazeci** | **188** |

Na pitanje „Kolika je opća stopa PDV-a?" sustav vraća **pet članaka, četiri od njih
`nevazeci`**, a čl. 38. Zakona o PDV-u (koji u bazi postoji, ispravno označen i
`citable`) ne uđe ni među 20 kandidata. Jednostavno gubi na omjeru 67:1.

**Ovo traži razjašnjenje opsega.** Ponuda §4.2 kaže da baza pokriva i „pročišćene
tekstove zakona", a cijela F2 zove se „Dopuna baze pročišćenim propisima". Ako
zakoni ostaju u opsegu, treba retrieval koji jamči zakonske chunkove u kandidatskom
skupu kad je pitanje pravne naravi — inače ih 12.561 članak uvijek nadglasa.

Ako su zakoni ipak izvan opsega F1, to mora biti zapisano, jer validacijski set
koji piše RRiF gotovo sigurno sadrži pitanja o propisima.

### 3.2. Temporalni filtar propušta sve nevažeće članke

U `_build_where`:

```python
clauses.append(f"(source_type != 'članak' AND ({sql_time}) OR source_type = 'članak')")
```

Prioritet operatora daje `(nije članak AND vrijeme_ok) OR jest članak` — dakle
**svi članci zaobilaze temporalni filtar**, dok zakonski tekst mora zadovoljiti
uvjet. 5.174 nevažećih članaka (41 % korpusa) natječe se bez ikakvog vremenskog
ograničenja, i zato na pitanje o današnjoj stopi PDV-a prvi pogodak bude članak iz
2014. o prijelazu na 13 %.

Razlog zašto je iznimka uopće dodana: **svi članci imaju postavljen `valid_to`,
nijedan nema NULL** (provjereno), pa bi uvjet `valid_to IS NULL` obrisao cijeli
korpus. Ispravak nije „traži važeće" — to bi slomilo pitanja tipa „kolika je bila
minimalna plaća 2024." — nego poštovati `time_period` koji klasifikator ionako već
vraća.

Ovo je izravno vezano uz tenderski zahtjev *„jasno razlikovanje važećih i
nevažećih propisa"* i uz redak 1.3 u F1 sheetu. Trenutno razlikovanja u retrievalu
nema.

### 3.3. Metrika točnosti retrievala bila je neispravna

Stara `run_eval.py` mjerila je top-1 s **maksimumom od 16,2 %** — 31 od 37 pitanja
imalo je prazan `expected_articles`, pa su brojana kao promašaj po konstrukciji.
Uz to je mjerila članke koje je *generator odlučio citirati*, a ne ono što je
retrieval vratio, i `--skip-generation` uopće nije preskakao generiranje.

Popravljeno i pushano. Nova brojka na ispravljenoj metrici: **top-1 16,7 %, top-5
50 % (n=6)**. Šest pitanja je premalo za zaključak — i to je poanta sljedeće točke.

**Za brojku „retrieval top-1 15–23 %" iz kolovoškog pregleda: ne koristiti je više.
Nastala je istom greškom.**

### 3.4. Ablacija filtra po kategoriji — filtar pomaže

Testirano: retrieval bez domain/subdomain filtra daje **lošiji** rezultat
(top-1 0 % naspram 16,7 %). Uz klasifikator na 100 % i korpus ovog sastava, filtar
je jedna od rijetkih stvari koje korisno sužavaju izbor. **Ostaje kakav jest.**

---

## 4. Kalibracijski i validacijski set — dobro postavljeno, tri stvari nedostaju

F1 sheet već traži dva odvojena seta, s validacijskim zamrznutim i nevidljivim nama.
To je točno kako treba i pokriva glavni prigovor iz kolovoškog pregleda.

Nedostaje sljedeće:

**Veličina uzorka.** Specifikacija traži „minimalno 100 pitanja" ukupno, ali ne kaže
kako se dijele. Ako validacijski set ima 50 pitanja, pri stvarnoj točnosti od 85 %
interval pouzdanosti je **oko ±10 postotnih bodova** — sustav koji zadovoljava mjeri
se bilo gdje između 75 % i 95 %. Na 100 pitanja je ±7. **Tražiti 150–200 pitanja**,
i to sada dok ih savjetnici tek pišu.

Razlog nije samo preciznost: F1 kriterij predviđa iteracije, a **holdout je potrošen
čim vidimo koja su pitanja pala.** Ponovno mjerenje na istom setu je optimistično.
Sa 150+ pitanja stane treći set za ponovljeno mjerenje.

**Izvor po pitanju, ne samo odgovor.** Za svako pitanje treba i **koji RRiF-ov broj
i članak** savjetnik smatra ispravnim izvorom. Bez toga se retrieval na člancima ne
može ocijeniti — ostaje samo provjera ključnih riječi u generiranom tekstu, što je
najslabiji dio harnessa. Ako to stigne zajedno sa setom, ne košta ih ništa; ako
tražimo poslije, to je drugi krug.

**Mehanika zamrzavanja.** Tko dijeli set (RRiF, ne mi), stratificirano po
poddomenama i tipu pitanja, hash datoteke zapisan obostrano pri zamrzavanju i
provjeren pri mjerenju. Zadnje štiti nas jednako koliko i njih — sprječava naknadno
„mislili smo drugačije".

Uz to: **zajednički prolaz kroz odgovorivost prije zamrzavanja.** Savjetnici pišu
pitanja iz glave, ne iz arhive. Dio njih neće biti odgovoriv iz 10 godina RRiF-ovih
članaka. Klauzula o izuzeću vrijedi samo ako su takva pitanja identificirana
**prije** mjerenja.

---

## 5. Rizik na kriteriju F4: prvi token < 2 s

Lanac je danas serijski: klasifikator (Haiku) → retrieval → rerank → generator.
Sam klasifikator je mrežni poziv reda 1–2 s. Uz rerank (t. 2.1) i Sonnetov TTFT,
**2 s je vrlo tijesno**, a kriterij se mjeri na 500 upita.

Vrijedi ranije razmisliti o: keširanju klasifikacije za ponovljena pitanja,
paralelizaciji klasifikatora s embeddingom, ili preskakanju klasifikatora kad
retrieval bez filtra ionako daje isti skup.

---

## 6. Što bih poručio Stipi

**STACK je dobar. Tri retka treba promijeniti, jedno pitanje razjasniti.**

1. **`croatian` FTS config ne postoji** — maknuti iz scheme, ide hunspell ili
   stemming u aplikaciji (skripte spremne).
2. **Baza: samohostano, zatvoreno** — nema postotka dostupnosti, baza je ~1,5 GB,
   FTS rječnik traži datotečni pristup.
3. **Reranker: prvo svesti na jedan poziv po pitanju**, pa birati providera.
   Četiri poziva ne stanu u 200 ms.
4. **AX41 nema GPU** — s lokalnim rerankerom ne stane u „prvi token < 2 s".
   Prijedlog: GEX44 barem za ingestiju i F3.
5. **Razjasniti: jesu li zakoni u opsegu F1?** Ponuda kaže da jesu, F2 je cijela o
   propisima, a mi smo se u kolovozu dogovarali suprotno. Validacijski set to sigurno
   dira.
6. **Tražiti 150–200 pitanja umjesto 100, i izvor uz svaki odgovor** — sada, dok
   savjetnici pišu.

---

*Mjerenja: `eval/results/20260904_125952_baseline_fixed_retrieval_only.json`,
`20260904_130042_ablation_wide_retrieval_only_mode_wide.json`,
`scripts/exp_ann_recall.py` (recall@20 = 100 % na 29 pitanja).*
