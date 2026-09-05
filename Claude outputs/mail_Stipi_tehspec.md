# Nacrt odgovora Stipi — komentar na TehSpec

**Predmet:** Re: ▶INTERNAL◀ - RRIF AI Agent

---

Bok Stipe,

Prošao sam xls i doc. Spec je dobar, timeline je realan koliko može biti. Par
komentara na „Pregled", poredano po tome što blokira INFRA rok.

**1. Server — trebam odluku prije nego naručim**

AX41 nema GPU. E5 embedding na CPU-u je u redu (~100 ms po upitu), ali BGE reranker
nad 50 kandidata na 6 jezgri je sekunde, ne milisekunde. VOICE sheet daje 200 ms za
„retrieval + reranker", a F4 traži prvi token ispod 2 s na 500 upita.

Zasebno: trenutni `_retrieve` zove reranker **do četiri puta po pitanju** — troslojni
fallback provjerava kvalitetu na svakoj razini prije nego odluči ide li dalje. To
svodim na jedan poziv bez obzira na sve ostalo, ali ni s jednim pozivom lokalni
rerank na CPU-u ne stane u budžet.

Dvije opcije:

* **GEX44** (RTX 4000 Ada, 20 GB, ~184 €/mj, bez minimalnog trajanja) — vrti
  embedding, rerank i Whisper za F3. Sve ostaje u Njemačkoj, jedan DPA.
* **API reranker** (Cohere/Jina) — jedan mrežni skok u vrućoj putanji i još jedan
  podobrađivač na popisu.

Predlažem GEX44 barem za razdoblje ingestije i F3, pa vidimo treba li u produkciji.
STT arhive na CPU-u je noć posla po seriji, na GPU-u minute.

**2. `croatian` FTS konfiguracija ne postoji — maknuti iz scheme**

U schemi stoji `to_tsvector('croatian', title || ' ' || body)`. PostgreSQL nema
hrvatsku konfiguraciju ni u jednom buildu — migracija bi pukla na prvom pokretanju.

Ide ili hunspell rječnik (traži superusera i pristup datotečnom sustavu) ili
lematizacija u Pythonu u zaseban stupac. Skripte za obje varijante su napisane,
testiram sljedeći tjedan i javim koja je bolja.

**3. Baza: samohostano — zatvaram to pitanje**

Nema postotka dostupnosti (tvoj mail), baza je ~1,5 GB na 10 godina građe, a
hunspell varijanta traži datotečni pristup. Managed PG ne donosi ništa što nam
treba, a dodaje podobrađivača i drugi DPA. Ostaje na istom boxu, Unix socket.

Sitno: schema kaže PG 16, postojeća baza je 17.9. Uskladiti prije INFRA-e.

**4. Jesu li zakoni u opsegu F1? — ovo mi je najveća nepoznanica**

U kolovozu smo pričali da idu samo članci, a zakoni eventualno kasnije. Ali Ponuda
§4.2 kaže da baza pokriva i pročišćene tekstove zakona, a cijela F2 zove se
„Dopuna baze pročišćenim propisima".

Pitam jer sam ovaj tjedan mjerio. Baza je danas **12.561 članak i 188 chunkova
zakona**. Na pitanje „Kolika je opća stopa PDV-a?" sustav vrati pet članaka, četiri
od njih nevažeća, a čl. 38. Zakona o PDV-u — koji je u bazi, ispravno označen i
citable — ne uđe ni među 20 kandidata. Jednostavno gubi na omjeru 67:1.

Ako zakoni ostaju u opsegu, treba retrieval koji im jamči mjesto u kandidatskom
skupu kad je pitanje pravne naravi. Ako ne ostaju, to mora biti zapisano — jer
validacijski set koji pišu RRiF-ovi savjetnici gotovo sigurno dira propise.

**5. Jedan rok koji se ne poklapa**

Interno pravilo u F1 sheetu: *ako nakon 3 tjedna kalibracije nismo na ≥80 % → stop,
sastanak s RRiF-om o materijalima.*

Kalibracijski set stiže 20.09., tri tjedna je 11.10., a prihvaćanje F1 je 09.10.
Sigurnosni ventil se pali dva dana nakon što je trebao pomoći.

Ili set stiže ranije, ili pravilo prebacujemo na dva tjedna.

**6. Prema RRiF-u — dvije stvari, i to sada dok savjetnici tek pišu**

**Traži 150–200 pitanja umjesto 100.** Pri stvarnoj točnosti od 85 % i validacijskom
setu od 50 pitanja, interval pouzdanosti je oko ±10 postotnih bodova — sustav koji
zadovoljava izmjeri se bilo gdje između 75 % i 95 %. Uz to je holdout potrošen čim
vidimo koja su pitanja pala, pa za ponovljeno mjerenje nakon dorade treba treći set.
Sa 150+ pitanja to stane.

**I traži da uz svako pitanje dođe izvor** — koji RRiF-ov broj i članak savjetnik
smatra ispravnim. Bez toga retrieval na člancima ne mogu ocijeniti; ostaje samo
provjera ključnih riječi u tekstu odgovora, što je najslabiji dio evaluacije. Ako
dođe zajedno sa setom, njima ne košta ništa. Ako tražimo poslije, to je drugi krug.

I dalje čekam **uzorak XLS/CSV metapodataka** (dovoljno 5–10 redaka) — stoji kao
otvoreno pitanje #3 u tvom xls-u, a rok je bio 31.08. Ako su kategorije slobodan
tekst, normalizacija je posao koji nitko nije uračunao.

LP,
Duje
