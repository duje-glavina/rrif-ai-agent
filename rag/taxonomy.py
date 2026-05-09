"""Single source of truth for the RRiF two-level taxonomy.

Import this wherever you need domain/subdomain values — classifier,
reclassifier, retrieval query builder, eval grader.

Structure
---------
TAXONOMY : dict[domain -> dict[subdomain -> description]]
    Full hierarchy with human-readable descriptions for prompts.

SUBDOMAIN_TO_DOMAIN : dict[subdomain -> domain]
    Reverse lookup. Every subdomain maps to exactly one domain.

VALID_DOMAINS : frozenset[str]
VALID_SUBDOMAINS : frozenset[str]

RECENCY_TAGS : tuple[str, ...]
    Valid values for the recency_tag / status signal.

PROMOTION_THRESHOLD : int
    Minimum vazeci chunks before an ostalo leaf is promoted to its own domain.
"""
from __future__ import annotations

TAXONOMY: dict[str, dict[str, str]] = {
    "porezi": {
        "PDV": (
            "Porez na dodanu vrijednost — stope (25%, 13%, 5%), obrasci PDV/PDV-S/PDV-K, "
            "pretporez, oslobođenja, prag za upis u registar PDV-a, porezna osnovica"
        ),
        "dohodak": (
            "Porez na dohodak fizičkih osoba — osobni odbitak, porezne stope, "
            "dohodak od rada/kapitala/imovine, godišnja prijava PO, rezidenti i nerezidenti"
        ),
        "dobit": (
            "Porez na dobit pravnih osoba — porezna osnovica, stope (18%/10%), "
            "porezno priznatih rashodi, porezna prijava PD, porez po odbitku"
        ),
        "paušal": (
            "Paušalno oporezivanje — paušalni obrtnici, slobodna zanimanja, "
            "paušalni porez na dohodak, paušalni doprinosi"
        ),
        "porezi": (
            "Ostali porezi i davanja koja ne spadaju u PDV/dohodak/dobit/paušal — "
            "doprinosi za MIO i zdravstvo kao porezna davanja, trošarine, carine, "
            "porez na promet nekretnina, komunalni doprinosi"
        ),
    },

    "računovodstvo": {
        "knjiženje": (
            "Kontni plan, temeljnice, knjiženje poslovnih događaja, MRS/MSFI/HSFI, "
            "amortizacija, zalihe, dugotrajna imovina, potraživanja, obveze"
        ),
        "plaće": (
            "Obračun plaće, bruto/neto obračun, doprinosi iz plaće i na plaću, "
            "bolovanje, naknade plaće, putni nalozi, dnevnice, jubilarne nagrade, "
            "minimalac, porezna kartica"
        ),
        "revizija": (
            "Revizija financijskih izvještaja, MRevS standardi, revizorska izvješća, "
            "neovisnost revizora, Hrvatska revizorska komora"
        ),
        "fin_izv": (
            "Bilanca (POD-BIL), račun dobiti i gubitka (RDG), izvještaj o novčanom toku, "
            "godišnji financijski izvještaji, FINA, javna objava, konsolidacija"
        ),
        "računovodstvo": (
            "Ostalo iz računovodstva koje ne spada u knjiženje/plaće/reviziju/fin_izv — "
            "računovodstvene politike, procjene, organizacija računovodstva"
        ),
    },

    "pravo": {
        "radno_pravo": (
            "Zakon o radu, ugovori o radu, otkaz, otkazni rokovi, godišnji odmor, "
            "zaštita na radu, kolektivni ugovori, sindikati, radnička prava"
        ),
        "trg_pravo": (
            "Zakon o trgovačkim društvima (ZTD), osnivanje d.o.o./d.d., statut, "
            "organi društva, pripajanja, razdvajanja, stečaj, likvidacija"
        ),
        "inozemstvo": (
            "Uvoz i izvoz robe, INTRASTAT, devizno poslovanje, transferne cijene, "
            "ugovori o izbjegavanju dvostrukog oporezivanja, nerezidenti"
        ),
        "EU_propisi": (
            "Direktive i uredbe Europske unije, vijesti iz institucija EU, "
            "harmonizacija hrvatskog prava s EU pravom, implementacija direktiva"
        ),
        "pravo": (
            "Ostalo pravo koje ne spada u radno/trg/inozemstvo/EU — zaštita potrošača, "
            "tržišno natjecanje, opći propisi koji utječu na poslovanje, "
            "upravni postupak, prekršajno pravo"
        ),
    },

    "ostalo": {
        "proračun": (
            "Proračunski i izvanproračunski korisnici, proračunsko računovodstvo, "
            "javna nabava, financiranje iz proračuna, državna riznica"
        ),
        "NPO": (
            "Udruge, zaklade, vjerske zajednice, sindikati, ustanove, "
            "neprofitno računovodstvo, Zakon o financijskom poslovanju neprofitnih org."
        ),
        "tržište": (
            "Zaštita potrošača, tržišno natjecanje, označavanje proizvoda, "
            "e-trgovina, opći propisi koji uređuju tržišno poslovanje"
        ),
        "upravljanje": (
            "Korporativno upravljanje, interni akti, pravilnici, "
            "organizacijska struktura, compliance, interna kontrola"
        ),
        "stručne": (
            "Kratke stručne obavijesti, tumačenja Porezne uprave, pojašnjenja propisa, "
            "mišljenja ministarstava, odgovori na čitateljska pitanja"
        ),
        "ostalo": (
            "Sve što ne spada ni u jednu drugu kategoriju — opće informacije, "
            "oglasi, kazala, sadržaji publikacija"
        ),
    },
}

# ── Derived lookups ───────────────────────────────────────────────────────────

SUBDOMAIN_TO_DOMAIN: dict[str, str] = {
    subdomain: domain
    for domain, leaves in TAXONOMY.items()
    for subdomain in leaves
}

VALID_DOMAINS: frozenset[str] = frozenset(TAXONOMY.keys())
VALID_SUBDOMAINS: frozenset[str] = frozenset(SUBDOMAIN_TO_DOMAIN.keys())

# ── Recency tags ──────────────────────────────────────────────────────────────

RECENCY_TAGS = ("vazeci", "nevazeci", "najava")

# Mapping from old flat category values to new (domain, subdomain).
# Used by the migration script to do a fast bulk pre-fill before the
# LLM reclassification pass corrects the uncertain cases.
LEGACY_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    "PDV":                       ("porezi",        "PDV"),
    "porezi":                    ("porezi",        "porezi"),
    "računovodstvo":             ("računovodstvo", "računovodstvo"),
    "plaće":                     ("računovodstvo", "plaće"),
    "radno pravo":               ("pravo",         "radno_pravo"),
    "trgovačko pravo":           ("pravo",         "trg_pravo"),
    "revizija":                  ("računovodstvo", "revizija"),
    "proračun":                  ("ostalo",        "proračun"),
    "neprofitne organizacije":   ("ostalo",        "NPO"),
    "EU propisi":                ("pravo",         "EU_propisi"),
    "poslovanje s inozemstvom":  ("pravo",         "inozemstvo"),
    "tržište i propisi":         ("ostalo",        "tržište"),
    "novi propisi":              ("ostalo",        "ostalo"),   # recency → handled by tag
    "stručne informacije":       ("ostalo",        "stručne"),
    "upravljanje":               ("ostalo",        "upravljanje"),
    "ostalo":                    ("ostalo",        "ostalo"),
}

# Minimum vazeci chunks for an ostalo leaf to be promoted to its own domain
PROMOTION_THRESHOLD = 300

# ── Helpers ───────────────────────────────────────────────────────────────────

def prompt_taxonomy_block() -> str:
    """Return a formatted taxonomy string suitable for embedding in a system prompt."""
    lines = []
    for domain, leaves in TAXONOMY.items():
        lines.append(f'DOMAIN: "{domain}"')
        for subdomain, desc in leaves.items():
            lines.append(f'  subdomain: "{subdomain}" — {desc}')
        lines.append("")
    return "\n".join(lines).strip()


if __name__ == "__main__":
    print(f"Domains:    {sorted(VALID_DOMAINS)}")
    print(f"Subdomains: {sorted(VALID_SUBDOMAINS)}")
    print(f"\nTaxonomy prompt block:\n")
    print(prompt_taxonomy_block())
