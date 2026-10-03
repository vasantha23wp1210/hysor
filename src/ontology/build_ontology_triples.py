#!/usr/bin/env python3

import os
import re
import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD

# ── 0. Paths ─────────────────────────────────────────────────────────────────
#BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join("data", "processed")
#os.makedirs(PROCESSED_DIR, exist_ok=True)

INPUT_FILE    = os.path.join(PROCESSED_DIR, "CAA_ICAO_crossmap.csv")
TRIPLES_FILE  = os.path.join(PROCESSED_DIR, "phraseology_triples.csv")
TURTLE_FILE   = os.path.join(PROCESSED_DIR, "phraseology_ontology.ttl")

# ── 1. Load cross-map ─────────────────────────────────────────────────────────
print("Loading cross-map data...")
df = pd.read_csv(INPUT_FILE)
print(f"  Loaded {len(df)} records")

# ── 2. Define namespaces ──────────────────────────────────────────────────────
AVIATION = Namespace("http://aviation.safety.ontology/phraseology#")
ATC      = Namespace("http://aviation.safety.ontology/atc#")
ICAO_NS  = Namespace("http://aviation.safety.ontology/icao#")
CAA_NS   = Namespace("http://aviation.safety.ontology/caa#")
SKOS     = Namespace("http://www.w3.org/2004/02/skos/core#")

# ── 3. Helper: slugify text to safe URI fragment ──────────────────────────────
def slugify(text, max_len=60):
    text = re.sub(r"[^\w\s]", "", str(text))
    text = re.sub(r"\s+", "_", text.strip())
    return text[:max_len].upper()

# ── 4. Build RDF graph ────────────────────────────────────────────────────────
print("\nBuilding RDF ontology graph...")
g = Graph()
g.bind("aviation",  AVIATION)
g.bind("atc",       ATC)
g.bind("icao",      ICAO_NS)
g.bind("caa",       CAA_NS)
g.bind("skos",      SKOS)
g.bind("owl",       OWL)
g.bind("rdfs",      RDFS)
g.bind("xsd",       XSD)

# ── Ontology declaration
ontology_uri = URIRef("http://aviation.safety.ontology/phraseology")
g.add((ontology_uri, RDF.type,         OWL.Ontology))
g.add((ontology_uri, RDFS.label,       Literal("CAA ATC Phraseology Ontology", lang="en")))
g.add((ontology_uri, RDFS.comment,     Literal("Ontology for CAA ATC Section 8 phraseology cross-mapped with ICAO Doc 4444", lang="en")))

# ── Class declarations
classes = {
    "Phraseology":     "A standard aviation phraseology expression",
    "CAAPhraseology":  "A phraseology defined in CAA ATC Section 8",
    "ICAOPhraseology": "A phraseology defined in ICAO Doc 4444",
    "Circumstance":    "An operational condition that triggers a phraseology",
    "ATCService":      "An ATC service category (e.g. Aerodrome Control, Area Control)",
    "ATCSection":      "A numbered section within the ATC standards document",
    "Speaker":         "The role initiating the phraseology (ATC, Pilot, ATS Coordination)",
}
for cls_name, cls_comment in classes.items():
    cls_uri = AVIATION[cls_name]
    g.add((cls_uri, RDF.type,      OWL.Class))
    g.add((cls_uri, RDFS.label,    Literal(cls_name, lang="en")))
    g.add((cls_uri, RDFS.comment,  Literal(cls_comment, lang="en")))

# CAAPhraseology and ICAOPhraseology are subclasses of Phraseology
g.add((AVIATION["CAAPhraseology"],  RDFS.subClassOf, AVIATION["Phraseology"]))
g.add((AVIATION["ICAOPhraseology"], RDFS.subClassOf, AVIATION["Phraseology"]))

# ── Property declarations
properties = {
    "hasPhrase":         ("Data",   "The literal text of the phraseology"),
    "hasCircumstance":   ("Object", "Links a phraseology to its triggering circumstance"),
    "triggersPhrase":    ("Object", "Links a circumstance to the phraseology it triggers"),
    "belongsToSection":  ("Object", "Links a phraseology to its document section"),
    "belongsToService":  ("Object", "Links a phraseology to its ATC service category"),
    "hasSpeaker":        ("Object", "Links a phraseology to the speaker role"),
    "matchesICAO":       ("Object", "Links a CAA phraseology to its best ICAO equivalent"),
    "matchScore":        ("Data",   "Cosine similarity score between CAA and ICAO phrase"),
    "matchType":         ("Data",   "Classification of match quality"),
    "variableCount":     ("Data",   "Number of variable slots in the phraseology"),
    "sectionNumber":     ("Data",   "Section number in the source document"),
    "sectionTitle":      ("Data",   "Section title in the source document"),
}
for prop_name, (prop_type, prop_comment) in properties.items():
    prop_uri = AVIATION[prop_name]
    if prop_type == "Data":
        g.add((prop_uri, RDF.type,     OWL.DatatypeProperty))
    else:
        g.add((prop_uri, RDF.type,     OWL.ObjectProperty))
    g.add((prop_uri, RDFS.label,   Literal(prop_name, lang="en")))
    g.add((prop_uri, RDFS.comment, Literal(prop_comment, lang="en")))

# ── 5. Populate individuals ───────────────────────────────────────────────────
print("  Populating individuals from cross-map rows...")

# Cache service and section nodes to avoid duplicates
service_cache  = {}
section_cache  = {}
speaker_cache  = {}
icao_cache     = {}
triples_rows   = []

for _, row in df.iterrows():
    phrase_id   = slugify(row["Phraseology"])
    phrase_uri  = CAA_NS[f"PHRASE_{phrase_id}"]

    # Phraseology individual
    g.add((phrase_uri, RDF.type,               AVIATION["CAAPhraseology"]))
    g.add((phrase_uri, AVIATION["hasPhrase"],   Literal(str(row["Phraseology"]), datatype=XSD.string)))
    g.add((phrase_uri, AVIATION["matchScore"],  Literal(float(row["Match_Score"]), datatype=XSD.float)))
    g.add((phrase_uri, AVIATION["matchType"],   Literal(str(row["Match_Type"]), datatype=XSD.string)))
    g.add((phrase_uri, AVIATION["variableCount"], Literal(int(row["Variable_Count"]), datatype=XSD.integer)))
    g.add((phrase_uri, AVIATION["sectionNumber"], Literal(str(row["Section_Number"]), datatype=XSD.string)))
    g.add((phrase_uri, AVIATION["sectionTitle"],  Literal(str(row["Section_Title"]), datatype=XSD.string)))

    # Circumstance individual
    circ_id  = slugify(row["Circumstances"])
    circ_uri = AVIATION[f"CIRC_{circ_id}"]
    if circ_uri not in section_cache:
        g.add((circ_uri, RDF.type,    AVIATION["Circumstance"]))
        g.add((circ_uri, RDFS.label,  Literal(str(row["Circumstances"]), lang="en")))
    g.add((circ_uri, AVIATION["triggersPhrase"],   phrase_uri))
    g.add((phrase_uri, AVIATION["hasCircumstance"], circ_uri))

    # ATC Service individual
    svc_label = str(row["ATC_Service"])
    svc_id    = slugify(svc_label)
    svc_uri   = ATC[f"SVC_{svc_id}"]
    if svc_id not in service_cache:
        g.add((svc_uri, RDF.type,   AVIATION["ATCService"]))
        g.add((svc_uri, RDFS.label, Literal(svc_label, lang="en")))
        service_cache[svc_id] = svc_uri
    g.add((phrase_uri, AVIATION["belongsToService"], svc_uri))

    # Section individual
    sec_label = f"{row['Section_Number']} {row['Section_Title']}"
    sec_id    = slugify(sec_label)
    sec_uri   = ATC[f"SEC_{sec_id}"]
    if sec_id not in section_cache:
        g.add((sec_uri, RDF.type,   AVIATION["ATCSection"]))
        g.add((sec_uri, RDFS.label, Literal(sec_label, lang="en")))
        section_cache[sec_id] = sec_uri
    g.add((phrase_uri, AVIATION["belongsToSection"], sec_uri))

    # Speaker individual
    spk_label = str(row["Speaker"])
    spk_id    = slugify(spk_label)
    spk_uri   = ATC[f"SPEAKER_{spk_id}"]
    if spk_id not in speaker_cache:
        g.add((spk_uri, RDF.type,   AVIATION["Speaker"]))
        g.add((spk_uri, RDFS.label, Literal(spk_label, lang="en")))
        speaker_cache[spk_id] = spk_uri
    g.add((phrase_uri, AVIATION["hasSpeaker"], spk_uri))

    # ICAO match link
    icao_phrase = str(row["Best_ICAO_Match"])
    icao_id  = slugify(icao_phrase)
    icao_uri = ICAO_NS[f"PHRASE_{icao_id}"]
    if icao_id not in icao_cache:
        g.add((icao_uri, RDF.type,              AVIATION["ICAOPhraseology"]))
        g.add((icao_uri, AVIATION["hasPhrase"], Literal(icao_phrase, datatype=XSD.string)))
        icao_cache[icao_id] = icao_uri
    if str(row["Match_Type"]) != "No Match":
        g.add((phrase_uri, AVIATION["matchesICAO"], icao_uri))

    # Collect flat triples for CSV
    triples_rows.extend([
        (str(phrase_uri), "rdf:type",                        "aviation:CAAPhraseology"),
        (str(phrase_uri), "aviation:hasPhrase",              str(row["Phraseology"])),
        (str(phrase_uri), "aviation:hasCircumstance",        str(circ_uri)),
        (str(circ_uri),   "aviation:triggersPhrase",         str(phrase_uri)),
        (str(phrase_uri), "aviation:belongsToService",       str(svc_uri)),
        (str(phrase_uri), "aviation:belongsToSection",       str(sec_uri)),
        (str(phrase_uri), "aviation:hasSpeaker",             str(spk_uri)),
        (str(phrase_uri), "aviation:matchType",              str(row["Match_Type"])),
        (str(phrase_uri), "aviation:matchScore",             str(row["Match_Score"])),
    ])
    if str(row["Match_Type"]) != "No Match":
        triples_rows.append(
            (str(phrase_uri), "aviation:matchesICAO", str(icao_uri))
        )

# ── 6. Save Turtle ────────────────────────────────────────────────────────────
g.serialize(destination=TURTLE_FILE, format="turtle")
print(f"  RDF graph triples : {len(g)}")
print(f"  Saved TTL         : {TURTLE_FILE}")

# ── 7. Save flat triples CSV ──────────────────────────────────────────────────
triples_df = pd.DataFrame(triples_rows, columns=["Subject", "Predicate", "Object"])
triples_df.to_csv(TRIPLES_FILE, index=False)
print(f"  Saved triples CSV : {TRIPLES_FILE}")

# ── 8. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ONTOLOGY TRIPLE SUMMARY")
print("=" * 55)
print(f"  Total RDF triples       : {len(g)}")
print(f"  Flat triples (CSV rows) : {len(triples_df)}")
print(f"  CAA phrase individuals  : {len(df)}")
print(f"  ICAO phrase individuals : {len(icao_cache)}")
print(f"  Circumstance nodes      : {len(section_cache)}")
print(f"  ATC Service nodes       : {len(service_cache)}")
print(f"  Speaker nodes           : {len(speaker_cache)}")
print("=" * 55)
print("\nStep complete. Load phraseology_ontology.ttl into Protege or WebProtege.")
