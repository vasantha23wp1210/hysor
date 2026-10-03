# #!/usr/bin/env python3
# """
# ontology_population.py  —  UATIE-EDM Phase 1 · Step 6b
# ==============================================================
# Populate UATIE-EDM OWL Ontology with Annotated ASRS Incidents

# Tasks:
#   1. Load asrs_nlp_annotated.csv  (from Step 6a)
#   2. Load phraseology_ontology.ttl (from build_ontology_triples.py)
#   3. Create OWL individuals  → IncidentReport class
#   4. Populate object properties:
#        hasIntent            → IntentClass individual
#        hasHazard            → HazardType individual (multi-label)
#        hasPhaseOfFlight     → FlightPhase individual
#        hasDirectPhrasemeError   → Phraseology individual (strong link)
#        hasCommunicationContext  → Phraseology individual (weak link)
#   5. Export:
#        data/processed/asrs_incidents.ttl      ← incident triples
#        data/processed/uatie_edm_merged.ttl    ← phraseology + incidents merged

# Dependencies:
#   pip install rdflib pandas tqdm

# Usage:
#   python step6b_ontology_population.py
# """

# import os
# import re
# import hashlib
# import pandas as pd
# from tqdm import tqdm
# from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD
# from rdflib.namespace import SKOS

# # ── Paths ─────────────────────────────────────────────────────────────────────
# PROCESSED_DIR    = os.path.join("data", "processed")
# ANNOTATED_CSV    = os.path.join(PROCESSED_DIR, "asrs_nlp_annotated.csv")
# PHRASEOLOGY_TTL  = os.path.join(PROCESSED_DIR, "phraseology_ontology.ttl")
# OUTPUT_TTL       = os.path.join(PROCESSED_DIR, "asrs_incidents.ttl")
# MERGED_TTL       = os.path.join(PROCESSED_DIR, "uatie_edm_merged.ttl")

# # ── Namespaces ────────────────────────────────────────────────────────────────
# BASE     = "http://aviation.safety.ontology/"
# UATIE    = Namespace(BASE + "uatie#")
# INCIDENT = Namespace(BASE + "incident#")
# INTENT   = Namespace(BASE + "intent#")
# HAZARD   = Namespace(BASE + "hazard#")
# PHASE    = Namespace(BASE + "phase#")
# PHRAS    = Namespace(BASE + "phraseology#")

# # ── Helpers ───────────────────────────────────────────────────────────────────
# def slugify(text, maxlen=60):
#     text = re.sub(r"[^\w\s]", "", str(text))
#     text = re.sub(r"\s+", "_", text.strip())
#     return text[:maxlen].upper()

# def make_id(text, prefix="INC"):
#     """Stable short hash URI from narrative text."""
#     h = hashlib.md5(str(text).encode()).hexdigest()[:10].upper()
#     return f"{prefix}_{h}"

# # ══════════════════════════════════════════════════════════════════════════════
# # 1. LOAD ANNOTATED CSV
# # ══════════════════════════════════════════════════════════════════════════════
# print("=" * 65)
# print("  UATIE-EDM : Ontology Population")
# print("=" * 65)

# print("\n[1/5] Loading annotated corpus...")
# df = pd.read_csv(ANNOTATED_CSV, low_memory=False)
# df = df[df["intent"] != "Unclassified"].copy()
# df = df[df["summary"].notna()].reset_index(drop=True)
# print(f"  Classified records : {len(df):,}")
# print(f"  Intent classes     : {df['intent'].nunique()}")

# # ══════════════════════════════════════════════════════════════════════════════
# # 2. LOAD PHRASEOLOGY ONTOLOGY
# # ══════════════════════════════════════════════════════════════════════════════
# print("\n[2/5] Loading phraseology ontology...")
# phras_graph = Graph()
# if os.path.exists(PHRASEOLOGY_TTL):
#     phras_graph.parse(PHRASEOLOGY_TTL, format="turtle")
#     print(f"  Phraseology triples loaded : {len(phras_graph):,}")
# else:
#     print(f"  ⚠ Not found: {PHRASEOLOGY_TTL}")
#     print("    Continuing without phraseology linking.")

# # Collect existing phraseology URIs for linking
# existing_phrasemes = set(phras_graph.subjects(RDF.type, None))
# print(f"  Existing phraseme individuals : {len(existing_phrasemes):,}")

# # ══════════════════════════════════════════════════════════════════════════════
# # 3. BUILD INCIDENT GRAPH
# # ══════════════════════════════════════════════════════════════════════════════
# print("\n[3/5] Building incident OWL graph...")
# g = Graph()

# # Bind namespaces
# g.bind("uatie",    UATIE)
# g.bind("incident", INCIDENT)
# g.bind("intent",   INTENT)
# g.bind("hazard",   HAZARD)
# g.bind("phase",    PHASE)
# g.bind("phras",    PHRAS)
# g.bind("owl",      OWL)
# g.bind("rdfs",     RDFS)
# g.bind("skos",     SKOS)
# g.bind("xsd",      XSD)

# # ── Ontology declaration ──────────────────────────────────────────────────────
# onto_uri = URIRef(BASE + "uatie_incidents")
# g.add((onto_uri, RDF.type,           OWL.Ontology))
# g.add((onto_uri, RDFS.label,         Literal("UATIE-EDM Incident Reports", lang="en")))
# g.add((onto_uri, RDFS.comment,       Literal("Auto-populated from ASRS NLP Pipeline Step 6a", lang="en")))
# g.add((onto_uri, OWL.imports,        URIRef(BASE + "phraseology")))
# UATIE_hasIncidentID = UATIE["hasIncidentID"]
# g.add((UATIE_hasIncidentID, RDF.type,    OWL.DatatypeProperty))
# g.add((UATIE_hasIncidentID, RDFS.label,  Literal("hasIncidentID", lang="en")))
# g.add((UATIE_hasIncidentID, RDFS.domain, UATIE["IncidentReport"]))
# g.add((UATIE_hasIncidentID, RDFS.range,  XSD.string))
# g.add((UATIE_hasIncidentID, RDF.type, OWL.FunctionalProperty))

# # ── OWL Class declarations ────────────────────────────────────────────────────
# classes = {
#     "IncidentReport":   UATIE.IncidentReport,
#     "IntentClass":      UATIE.IntentClass,
#     "HazardType":       UATIE.HazardType,
#     "FlightPhase":      UATIE.FlightPhase,
# }
# for name, uri in classes.items():
#     g.add((uri, RDF.type,   OWL.Class))
#     g.add((uri, RDFS.label, Literal(name, lang="en")))

# # ── Object Property declarations ──────────────────────────────────────────────
# properties = {
#     UATIE.hasIntent:                (UATIE.IncidentReport, UATIE.IntentClass),
#     UATIE.hasHazard:                (UATIE.IncidentReport, UATIE.HazardType),
#     UATIE.hasPhaseOfFlight:         (UATIE.IncidentReport, UATIE.FlightPhase),
#     UATIE.hasDirectPhrasemeError:   (UATIE.IncidentReport, PHRAS.Phraseme),
#     UATIE.hasCommunicationContext:  (UATIE.IncidentReport, PHRAS.Phraseme),
#     UATIE.hasNarrativeText:         (UATIE.IncidentReport, None),   # DataProperty
#     UATIE.hasSourceDataset:         (UATIE.IncidentReport, None),
# }
# for prop_uri, (domain, range_uri) in properties.items():
#     prop_type = OWL.DatatypeProperty if range_uri is None else OWL.ObjectProperty
#     g.add((prop_uri, RDF.type,          prop_type))
#     g.add((prop_uri, RDFS.domain,       domain))
#     if range_uri:
#         g.add((prop_uri, RDFS.range,    range_uri))
#     label = str(prop_uri).split("#")[-1]
#     g.add((prop_uri, RDFS.label,        Literal(label, lang="en")))

# # ── Pre-create Intent individuals ─────────────────────────────────────────────
# intent_uris = {}
# for intent in df["intent"].unique():
#     uri = INTENT[slugify(intent)]
#     g.add((uri, RDF.type,   UATIE.IntentClass))
#     g.add((uri, RDF.type,   OWL.NamedIndividual))
#     g.add((uri, RDFS.label, Literal(intent.replace("_", " "), lang="en")))
#     intent_uris[intent] = uri

# # ── Pre-create FlightPhase individuals ────────────────────────────────────────
# phase_uris = {}
# for phase in df["phase_final"].dropna().unique():
#     uri = PHASE[slugify(str(phase))]
#     g.add((uri, RDF.type,   UATIE.FlightPhase))
#     g.add((uri, RDF.type,   OWL.NamedIndividual))
#     g.add((uri, RDFS.label, Literal(str(phase), lang="en")))
#     phase_uris[str(phase)] = uri

# # ── Pre-create HazardType individuals ─────────────────────────────────────────
# hazard_uri_map = {}
# all_hazards = set()
# for tags in df["hazard_tags"].dropna():
#     for h in str(tags).split("|"):
#         all_hazards.add(h.strip())
# for hazard in all_hazards:
#     uri = HAZARD[slugify(hazard)]
#     g.add((uri, RDF.type,   UATIE.HazardType))
#     g.add((uri, RDF.type,   OWL.NamedIndividual))
#     g.add((uri, RDFS.label, Literal(hazard.replace("H_", "").replace("_", " "), lang="en")))
#     hazard_uri_map[hazard] = uri

# # ── Phraseology link maps (two-tier) ──────────────────────────────────────────
# STRONG_PHRASEME_MAP = {
#     "readback":             PHRAS["READBACK"],
#     "read back":            PHRAS["READBACK"],
#     "phraseology":          PHRAS["STANDARD_PHRASEOLOGY"],
#     "callsign":             PHRAS["CALLSIGN"],
#     "squawk":               PHRAS["SQUAWK_CODE"],
#     "say again":            PHRAS["SAY_AGAIN"],
#     "wilco":                PHRAS["WILCO"],
#     "roger":                PHRAS["ROGER"],
#     "line up and wait":     PHRAS["LINE_UP_AND_WAIT"],
#     "cleared for takeoff":  PHRAS["CLEARED_FOR_TAKEOFF"],
#     "cleared to land":      PHRAS["CLEARED_TO_LAND"],
#     "words twice":          PHRAS["WORDS_TWICE"],
#     "acknowledge":          PHRAS["ACKNOWLEDGE"],
# }
# WEAK_PHRASEME_MAP = {
#     "frequency":   PHRAS["FREQUENCY_CHANGE"],
#     "clearance":   PHRAS["ATC_CLEARANCE"],
#     "clrnc":       PHRAS["ATC_CLEARANCE"],
#     "freq":        PHRAS["FREQUENCY_CHANGE"],
#     "contact":     PHRAS["CONTACT_INSTRUCTION"],
#     "monitor":     PHRAS["MONITOR_INSTRUCTION"],
#     "report":      PHRAS["POSITION_REPORT"],
#     "confirm":     PHRAS["CONFIRM_REQUEST"],
# }

# # ── Populate IncidentReport individuals ──────────────────────────────────────
# print(f"  Populating {len(df):,} IncidentReport individuals...")
# triple_count = 0

# for _, row in tqdm(df.iterrows(), total=len(df), desc="  Building triples"):
#     summary   = str(row["summary"])
#     intent    = str(row.get("intent", ""))
#     hazards   = str(row.get("hazard_tags", ""))
#     phase     = str(row.get("phase_final", "Unknown"))
#     source    = str(row.get("source", "ASRS"))
#     tl        = summary.lower()

#     # Create IncidentReport individual with stable URI
#     incident_id = str(row.get("incident_id", "UNK"))
#     inc_id  = make_id(summary)
#     inc_uri = INCIDENT[inc_id]

#     g.add((inc_uri, RDF.type,              UATIE.IncidentReport))
#     g.add((inc_uri, RDF.type,              OWL.NamedIndividual))
#     g.add((inc_uri, UATIE.hasNarrativeText,  Literal(summary, datatype=XSD.string)))
#     g.add((inc_uri, UATIE.hasSourceDataset,  Literal(source,  datatype=XSD.string)))
#     g.add((inc_uri, UATIE_hasIncidentID, Literal(incident_id, datatype=XSD.string)))
#     triple_count += 4
    

#     # hasIntent
#     if intent in intent_uris:
#         g.add((inc_uri, UATIE.hasIntent, intent_uris[intent]))
#         triple_count += 1

#     hazards = row['hazard_tags']

#     # hasHazard (multi-label)
#     if hazards and hazards != "nan":
#         for h in hazards.split("|"):
#             h = h.strip()
#             if h in hazard_uri_map:
#                 g.add((inc_uri, UATIE.hasHazard, hazard_uri_map[h]))
#                 triple_count += 1

    
#     # hasPhaseOfFlight
#     if phase and phase != "nan" and phase in phase_uris:
#         g.add((inc_uri, UATIE.hasPhaseOfFlight, phase_uris[phase]))
#         triple_count += 1

#     # Phraseology linking (Communication_Failure + ATC_Operational_Error only)
#     if intent in ("Communication_Failure", "ATC_Operational_Error"):
#         linked_strong = set()
#         linked_weak   = set()

#         for kw, phras_uri in STRONG_PHRASEME_MAP.items():
#             if kw in tl and phras_uri not in linked_strong:
#                 g.add((inc_uri, UATIE.hasDirectPhrasemeError, phras_uri))
#                 # Declare the phraseme individual if not already in phraseology graph
#                 g.add((phras_uri, RDF.type,   OWL.NamedIndividual))
#                 g.add((phras_uri, RDFS.label, Literal(kw.upper(), lang="en")))
#                 linked_strong.add(phras_uri)
#                 triple_count += 1

#         if not linked_strong:   # only use weak if no strong match
#             for kw, phras_uri in WEAK_PHRASEME_MAP.items():
#                 if kw in tl and phras_uri not in linked_weak:
#                     g.add((inc_uri, UATIE.hasCommunicationContext, phras_uri))
#                     g.add((phras_uri, RDF.type,   OWL.NamedIndividual))
#                     g.add((phras_uri, RDFS.label, Literal(kw.upper(), lang="en")))
#                     linked_weak.add(phras_uri)
#                     triple_count += 1

# print(f"  Total triples generated : {triple_count:,}")

# # ══════════════════════════════════════════════════════════════════════════════
# # 4. SERIALISE INCIDENT GRAPH
# # ══════════════════════════════════════════════════════════════════════════════
# print("\n[4/5] Serialising incident ontology...")
# g.serialize(destination=OUTPUT_TTL, format="turtle")
# print(f"  Saved : {OUTPUT_TTL}")
# print(f"  Size  : {os.path.getsize(OUTPUT_TTL)/1024/1024:.2f} MB")
# g.serialize(r"data\processed\uatie_edm_merged.ttl", format="turtle")
# print(f"✅ Serialization complete")
# print(f"✅ Total triples written : {triple_count:,}")
# print(f"✅ File saved            : data/processed/uatie_edm_merged.ttl")

# # ══════════════════════════════════════════════════════════════════════════════
# # 5. MERGE WITH PHRASEOLOGY ONTOLOGY
# # ══════════════════════════════════════════════════════════════════════════════
# print("\n[5/5] Merging with phraseology ontology...")
# merged = Graph()
# merged += phras_graph
# merged += g

# # Bind all namespaces in merged
# for prefix, ns in [("uatie",UATIE),("incident",INCIDENT),("intent",INTENT),
#                    ("hazard",HAZARD),("phase",PHASE),("phras",PHRAS)]:
#     merged.bind(prefix, ns)

# merged.serialize(destination=MERGED_TTL, format="turtle")
# print(f"  Saved : {MERGED_TTL}")
# print(f"  Total merged triples : {len(merged):,}")
# print(f"  Size  : {os.path.getsize(MERGED_TTL)/1024/1024:.2f} MB")

# # ══════════════════════════════════════════════════════════════════════════════
# # SUMMARY
# # ══════════════════════════════════════════════════════════════════════════════
# print("\n" + "=" * 65)
# print("  COMPLETE")
# print("=" * 65)

# comm_records  = (df["intent"] == "Communication_Failure").sum()
# atc_records   = (df["intent"] == "ATC_Operational_Error").sum()
# phras_triples = sum(1 for _ in g.triples((None, UATIE.hasDirectPhrasemeError, None)))
# ctx_triples   = sum(1 for _ in g.triples((None, UATIE.hasCommunicationContext, None)))

# print(f"  IncidentReport individuals  : {len(df):,}")
# print(f"  Intent individuals          : {len(intent_uris)}")
# print(f"  HazardType individuals      : {len(hazard_uri_map)}")
# print(f"  FlightPhase individuals     : {len(phase_uris)}")
# print(f"  Comm_Failure records        : {comm_records:,}")
# print(f"  ATC_Error records           : {atc_records:,}")
# print(f"  hasDirectPhrasemeError      : {phras_triples:,} triples (strong links)")
# print(f"  hasCommunicationContext     : {ctx_triples:,} triples (weak links)")
# print(f"\n  Output files:")
# print(f"    {OUTPUT_TTL}")
# print(f"    {MERGED_TTL}  ← import this into WebProtege")
# print("\nNext → rag_query_engine.py")
# print("=" * 65)
#!/usr/bin/env python3
"""
step6b_ontology_population.py  —  UATIE-EDM Phase 1 · Step 6b
==============================================================
Populate UATIE-EDM OWL Ontology with Annotated ASRS Incidents

Tasks:
  1. Load asrs_nlp_annotated.csv  (from Step 6a)
  2. Load phraseology_ontology.ttl (from build_ontology_triples.py)
  3. Create OWL individuals  → IncidentReport class
  4. Populate object properties:
       hasIntent                → IntentClass individual
       hasHazard                → HazardType individual (multi-label)
       hasPhaseOfFlight         → FlightPhase individual
       hasDirectPhrasemeError   → Phraseology individual (strong link)
       hasCommunicationContext  → Phraseology individual (weak link)
  5. Export:
       data/processed/asrs_incidents.ttl      ← incident triples only
       data/processed/uatie_edm_merged.ttl    ← phraseology + incidents merged

Dependencies:
  pip install rdflib pandas tqdm

Usage:
  python step6b_ontology_population.py
"""

import os
import re
import hashlib
import pandas as pd
from tqdm import tqdm
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD
from rdflib.namespace import SKOS


# ── Paths ──────────────────────────────────────────────────────────────────
PROCESSED_DIR   = os.path.join("data", "processed")
ANNOTATED_CSV   = os.path.join(PROCESSED_DIR, "asrs_nlp_annotated.csv")
PHRASEOLOGY_TTL = os.path.join(PROCESSED_DIR, "phraseology_ontology.ttl")
OUTPUT_TTL      = os.path.join(PROCESSED_DIR, "asrs_incidents.ttl")
MERGED_TTL      = os.path.join(PROCESSED_DIR, "uatie_edm_merged.ttl")


# ── Namespaces ─────────────────────────────────────────────────────────────
BASE     = "http://aviation.safety.ontology/"
UATIE    = Namespace(BASE + "uatie#")
INCIDENT = Namespace(BASE + "incident#")
INTENT   = Namespace(BASE + "intent#")
HAZARD   = Namespace(BASE + "hazard#")
PHASE    = Namespace(BASE + "phase#")
PHRAS    = Namespace(BASE + "phraseology#")


# ── Helpers ────────────────────────────────────────────────────────────────
def slugify(text, maxlen=60):
    text = re.sub(r"[^\w\s]", "", str(text))
    text = re.sub(r"\s+", "_", text.strip())
    return text[:maxlen].upper()


def make_id(text, prefix="INC"):
    """Stable short hash URI from narrative text."""
    h = hashlib.md5(str(text).encode()).hexdigest()[:10].upper()
    return f"{prefix}_{h}"


# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD ANNOTATED CSV  — ALL 113,468 records (no Unclassified filter)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  UATIE-EDM : Ontology Population  (Step 6b)")
print("=" * 65)

print("\n[1/5] Loading annotated corpus...")
df = pd.read_csv(ANNOTATED_CSV, low_memory=False)
df = df[df["summary"].notna()].reset_index(drop=True)   # only drop null summaries
print(f"  Total records      : {len(df):,}")
print(f"  Classified         : {(df['intent'] != 'Unclassified').sum():,}")
print(f"  Unclassified       : {(df['intent'] == 'Unclassified').sum():,}")
print(f"  Intent classes     : {df['intent'].nunique()}")
print(f"  Columns            : {df.columns.tolist()}")


# ══════════════════════════════════════════════════════════════════════════
# 2. LOAD PHRASEOLOGY ONTOLOGY
# ══════════════════════════════════════════════════════════════════════════
print("\n[2/5] Loading phraseology ontology...")
phras_graph = Graph()
if os.path.exists(PHRASEOLOGY_TTL):
    phras_graph.parse(PHRASEOLOGY_TTL, format="turtle")
    print(f"  Phraseology triples loaded    : {len(phras_graph):,}")
else:
    print(f"  ⚠ Not found: {PHRASEOLOGY_TTL}")
    print("    Continuing without phraseology linking.")

existing_phrasemes = set(phras_graph.subjects(RDF.type, None))
print(f"  Existing phraseme individuals : {len(existing_phrasemes):,}")


# ══════════════════════════════════════════════════════════════════════════
# 3. BUILD INCIDENT GRAPH
# ══════════════════════════════════════════════════════════════════════════
print("\n[3/5] Building incident OWL graph...")
g = Graph()

# Bind namespaces
g.bind("uatie",    UATIE)
g.bind("incident", INCIDENT)
g.bind("intent",   INTENT)
g.bind("hazard",   HAZARD)
g.bind("phase",    PHASE)
g.bind("phras",    PHRAS)
g.bind("owl",      OWL)
g.bind("rdfs",     RDFS)
g.bind("skos",     SKOS)
g.bind("xsd",      XSD)

# ── Ontology declaration ───────────────────────────────────────────────
onto_uri = URIRef(BASE + "uatie_incidents")
g.add((onto_uri, RDF.type,    OWL.Ontology))
g.add((onto_uri, RDFS.label,  Literal("UATIE-EDM Incident Reports", lang="en")))
g.add((onto_uri, RDFS.comment,Literal("Auto-populated from ASRS NLP Pipeline Step 6a", lang="en")))
g.add((onto_uri, OWL.imports, URIRef(BASE + "phraseology")))

UATIE_hasIncidentID = UATIE["hasIncidentID"]
g.add((UATIE_hasIncidentID, RDF.type,    OWL.DatatypeProperty))
g.add((UATIE_hasIncidentID, RDFS.label,  Literal("hasIncidentID", lang="en")))
g.add((UATIE_hasIncidentID, RDFS.domain, UATIE["IncidentReport"]))
g.add((UATIE_hasIncidentID, RDFS.range,  XSD.string))
g.add((UATIE_hasIncidentID, RDF.type,    OWL.FunctionalProperty))

# ── OWL Class declarations ────────────────────────────────────────────
classes = {
    "IncidentReport": UATIE.IncidentReport,
    "IntentClass":    UATIE.IntentClass,
    "HazardType":     UATIE.HazardType,
    "FlightPhase":    UATIE.FlightPhase,
}
for name, uri in classes.items():
    g.add((uri, RDF.type,   OWL.Class))
    g.add((uri, RDFS.label, Literal(name, lang="en")))

# ── Object Property declarations ──────────────────────────────────────
properties = {
    UATIE.hasIntent:               (UATIE.IncidentReport, UATIE.IntentClass),
    UATIE.hasHazard:               (UATIE.IncidentReport, UATIE.HazardType),
    UATIE.hasPhaseOfFlight:        (UATIE.IncidentReport, UATIE.FlightPhase),
    UATIE.hasDirectPhrasemeError:  (UATIE.IncidentReport, PHRAS.Phraseme),
    UATIE.hasCommunicationContext: (UATIE.IncidentReport, PHRAS.Phraseme),
    UATIE.hasNarrativeText:        (UATIE.IncidentReport, None),
    UATIE.hasSourceDataset:        (UATIE.IncidentReport, None),
}
for prop_uri, (domain, range_uri) in properties.items():
    prop_type = OWL.DatatypeProperty if range_uri is None else OWL.ObjectProperty
    g.add((prop_uri, RDF.type,    prop_type))
    g.add((prop_uri, RDFS.domain, domain))
    if range_uri:
        g.add((prop_uri, RDFS.range, range_uri))
    label = str(prop_uri).split("#")[-1]
    g.add((prop_uri, RDFS.label,  Literal(label, lang="en")))

# ── Pre-create Intent individuals ─────────────────────────────────────
intent_uris = {}
for intent in df["intent"].unique():
    uri = INTENT[slugify(intent)]
    g.add((uri, RDF.type,   UATIE.IntentClass))
    g.add((uri, RDF.type,   OWL.NamedIndividual))
    g.add((uri, RDFS.label, Literal(str(intent).replace("_", " "), lang="en")))
    intent_uris[intent] = uri

# ── Pre-create FlightPhase individuals ───────────────────────────────
phase_uris = {}
for phase in df["phase_final"].dropna().unique():
    uri = PHASE[slugify(str(phase))]
    g.add((uri, RDF.type,   UATIE.FlightPhase))
    g.add((uri, RDF.type,   OWL.NamedIndividual))
    g.add((uri, RDFS.label, Literal(str(phase), lang="en")))
    phase_uris[str(phase)] = uri

# ── Pre-create HazardType individuals ────────────────────────────────
hazard_uri_map = {}
all_hazards = set()
for tags in df["hazard_tags"].dropna():
    for h in str(tags).split("|"):
        h = h.strip()
        if h:
            all_hazards.add(h)
for hazard in all_hazards:
    uri = HAZARD[slugify(hazard)]
    g.add((uri, RDF.type,   UATIE.HazardType))
    g.add((uri, RDF.type,   OWL.NamedIndividual))
    g.add((uri, RDFS.label, Literal(hazard.replace("H_", "").replace("_", " "), lang="en")))
    hazard_uri_map[hazard] = uri

# ── Phraseology link maps (two-tier) ──────────────────────────────────
STRONG_PHRASEME_MAP = {
    "readback":            PHRAS["READBACK"],
    "read back":           PHRAS["READBACK"],
    "phraseology":         PHRAS["STANDARD_PHRASEOLOGY"],
    "callsign":            PHRAS["CALLSIGN"],
    "squawk":              PHRAS["SQUAWK_CODE"],
    "say again":           PHRAS["SAY_AGAIN"],
    "wilco":               PHRAS["WILCO"],
    "roger":               PHRAS["ROGER"],
    "line up and wait":    PHRAS["LINE_UP_AND_WAIT"],
    "cleared for takeoff": PHRAS["CLEARED_FOR_TAKEOFF"],
    "cleared to land":     PHRAS["CLEARED_TO_LAND"],
    "words twice":         PHRAS["WORDS_TWICE"],
    "acknowledge":         PHRAS["ACKNOWLEDGE"],
}
WEAK_PHRASEME_MAP = {
    "frequency": PHRAS["FREQUENCY_CHANGE"],
    "clearance": PHRAS["ATC_CLEARANCE"],
    "clrnc":     PHRAS["ATC_CLEARANCE"],
    "freq":      PHRAS["FREQUENCY_CHANGE"],
    "contact":   PHRAS["CONTACT_INSTRUCTION"],
    "monitor":   PHRAS["MONITOR_INSTRUCTION"],
    "report":    PHRAS["POSITION_REPORT"],
    "confirm":   PHRAS["CONFIRM_REQUEST"],
}

# ── Populate IncidentReport individuals ──────────────────────────────
print(f"  Populating {len(df):,} IncidentReport individuals...")
triple_count  = 0
intent_c      = 0
hazard_c      = 0
phase_c       = 0
literal_c     = 0
phras_strong  = 0
phras_weak    = 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="  Building triples"):
    summary = str(row["summary"])
    intent  = str(row.get("intent",      "Unclassified"))
    hazards = str(row.get("hazard_tags", ""))
    phase   = str(row.get("phase_final", "Unknown"))
    source  = str(row.get("source",      "ASRS"))
    tl      = summary.lower()

    inc_id  = make_id(summary)
    inc_uri = INCIDENT[inc_id]

    # Core triples — all records get these
    g.add((inc_uri, RDF.type,               UATIE.IncidentReport))
    g.add((inc_uri, RDF.type,               OWL.NamedIndividual))
    g.add((inc_uri, UATIE.hasNarrativeText, Literal(summary, datatype=XSD.string)))
    g.add((inc_uri, UATIE.hasSourceDataset, Literal(source,  datatype=XSD.string)))
    incident_id = str(row.get("incident_id", inc_id))
    g.add((inc_uri, UATIE_hasIncidentID,    Literal(incident_id, datatype=XSD.string)))
    triple_count += 5
    literal_c    += 3

    # hasIntent — classified + unclassified both get intent triple
    if intent in intent_uris:
        g.add((inc_uri, UATIE.hasIntent, intent_uris[intent]))
        triple_count += 1
        intent_c     += 1

    # hasHazard (multi-label) — all records
    if hazards and hazards != "nan":
        for h in hazards.split("|"):
            h = h.strip()
            if h in hazard_uri_map:
                g.add((inc_uri, UATIE.hasHazard, hazard_uri_map[h]))
                triple_count += 1
                hazard_c     += 1

    # hasPhaseOfFlight — all records
    if phase and phase != "nan" and phase in phase_uris:
        g.add((inc_uri, UATIE.hasPhaseOfFlight, phase_uris[phase]))
        triple_count += 1
        phase_c      += 1

    # Phraseology linking — Communication_Failure + ATC_Operational_Error only
    if intent in ("Communication_Failure", "ATC_Operational_Error"):
        linked_strong = set()
        linked_weak   = set()

        for kw, phras_uri in STRONG_PHRASEME_MAP.items():
            if kw in tl and phras_uri not in linked_strong:
                g.add((inc_uri, UATIE.hasDirectPhrasemeError, phras_uri))
                g.add((phras_uri, RDF.type,   OWL.NamedIndividual))
                g.add((phras_uri, RDFS.label, Literal(kw.upper(), lang="en")))
                linked_strong.add(phras_uri)
                triple_count += 1
                phras_strong += 1

        if not linked_strong:
            for kw, phras_uri in WEAK_PHRASEME_MAP.items():
                if kw in tl and phras_uri not in linked_weak:
                    g.add((inc_uri, UATIE.hasCommunicationContext, phras_uri))
                    g.add((phras_uri, RDF.type,   OWL.NamedIndividual))
                    g.add((phras_uri, RDFS.label, Literal(kw.upper(), lang="en")))
                    linked_weak.add(phras_uri)
                    triple_count += 1
                    phras_weak   += 1

print(f"\n  ── Triple Breakdown ──────────────────────────────")
print(f"  Rows processed           : {len(df):,}")
print(f"  Core/literal triples     : {literal_c:,}")
print(f"  Intent triples           : {intent_c:,}")
print(f"  Hazard triples           : {hazard_c:,}")
print(f"  Phase triples            : {phase_c:,}")
print(f"  Strong phraseme triples  : {phras_strong:,}")
print(f"  Weak phraseme triples    : {phras_weak:,}")
print(f"  Total loop triples       : {triple_count:,}")


# ══════════════════════════════════════════════════════════════════════════
# 4. SERIALISE INCIDENT GRAPH ONLY  →  asrs_incidents.ttl
# ══════════════════════════════════════════════════════════════════════════
print("\n[4/5] Serialising incident ontology...")
g.serialize(destination=OUTPUT_TTL, format="turtle")
print(f"  Saved : {OUTPUT_TTL}")
print(f"  Size  : {os.path.getsize(OUTPUT_TTL)/1024/1024:.2f} MB")


# ══════════════════════════════════════════════════════════════════════════
# 5. MERGE WITH PHRASEOLOGY  →  uatie_edm_merged.ttl
# ══════════════════════════════════════════════════════════════════════════
print("\n[5/5] Merging with phraseology ontology...")
merged = Graph()
merged += phras_graph
merged += g

for prefix, ns in [("uatie",    UATIE),
                   ("incident", INCIDENT),
                   ("intent",   INTENT),
                   ("hazard",   HAZARD),
                   ("phase",    PHASE),
                   ("phras",    PHRAS)]:
    merged.bind(prefix, ns)

merged.serialize(destination=MERGED_TTL, format="turtle")
print(f"  Saved : {MERGED_TTL}")
print(f"  Total merged triples : {len(merged):,}")
print(f"  Size  : {os.path.getsize(MERGED_TTL)/1024/1024:.2f} MB")


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
comm_records = (df["intent"] == "Communication_Failure").sum()
atc_records  = (df["intent"] == "ATC_Operational_Error").sum()

print("\n" + "=" * 65)
print("  COMPLETE")
print("=" * 65)
print(f"  IncidentReport individuals  : {len(df):,}")
print(f"  Intent individuals          : {len(intent_uris)}")
print(f"  HazardType individuals      : {len(hazard_uri_map)}")
print(f"  FlightPhase individuals     : {len(phase_uris)}")
print(f"  Comm_Failure records        : {comm_records:,}")
print(f"  ATC_Error records           : {atc_records:,}")
print(f"  hasDirectPhrasemeError      : {phras_strong:,}  triples (strong links)")
print(f"  hasCommunicationContext     : {phras_weak:,}  triples (weak links)")
print(f"  Total triples (incident g)  : {triple_count:,}")
print(f"  Total triples (merged TTL)  : {len(merged):,}")
print(f"\n  Output files:")
print(f"    {OUTPUT_TTL}")
print(f"    {MERGED_TTL}  ← import this into WebProtege")
print("\nNext → step6c_rag_query_engine.py")
print("=" * 65)
