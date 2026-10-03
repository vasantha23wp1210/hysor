#!/usr/bin/env python3
"""
asrs_nlp_pipeline.py  —  UATIE-EDM Phase 1 · Step 6a
=============================================================
ASRS Incident Narrative NLP Pipeline

Tasks:
  1. Load & clean fused_with_asrs.csv (summary column)
  2. Intent Classification  — keyword-rule classifier (7 intents)
  3. Hazard Tagging         — multi-label keyword tagger (9 hazard types)
  4. Phase-of-Flight Extraction — regex + keyword mapper
  5. Sentence-level Embeddings  — sentence-transformers (all-MiniLM-L6-v2)
  6. FAISS Index Build          — cosine similarity index
  7. RAG Demo Query             — top-5 retrieval for a sample query
  8. Outputs:
       data/processed/asrs_nlp_annotated.csv   ← annotated corpus
       data/processed/asrs_faiss.index          ← FAISS vector index
       data/processed/asrs_faiss_meta.csv       ← metadata for retrieved records

Dependencies (install once):
  pip install sentence-transformers faiss-cpu pandas numpy tqdm

Usage:
  python step6a_asrs_nlp_pipeline.py

  Optional: adjust SAMPLE_SIZE below to run on a subset first (e.g. 5000)
  Set SAMPLE_SIZE = None to process all 113K records (takes ~10 min).
"""

import os
import re
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR   = os.path.join("data", "processed")
INPUT_FILE      = os.path.join(PROCESSED_DIR, "fused_with_asrs.csv")
OUTPUT_CSV      = os.path.join(PROCESSED_DIR, "asrs_nlp_annotated.csv")
FAISS_INDEX     = os.path.join(PROCESSED_DIR, "asrs_faiss.index")
FAISS_META      = os.path.join(PROCESSED_DIR, "asrs_faiss_meta.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_SIZE     = None        # Set to None to run full 113K records
EMBED_MODEL     = "all-MiniLM-L6-v2"   # Fast, accurate, 384-dim
BATCH_SIZE      = 256         # Embedding batch size

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  UATIE-EDM · Step 6a: ASRS NLP Pipeline")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# PRE-LOAD: Verify input file exists before reading
# ══════════════════════════════════════════════════════════════
if not os.path.exists(INPUT_FILE):
    print(f"  ✗ Input file not found: {INPUT_FILE}")
    print("    Run fuse_datasets_fixed.py first to generate it.")
    raise SystemExit(1)

print("\n[1/7] Loading fused corpus...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"  Raw records : {len(df):,}")

print("\n  DEBUG source column check:")
print(f"  Column exists : {'source' in df.columns}")
print(f"  Unique values :\n{df['source'].value_counts().to_string()}")

# ── INCIDENT_ID BLOCK ─────────────────────────────────────────
# Assign a stable, source-prefixed incident ID to every row.
# Format:  ASRS_000001 | NTSB_000001 | CRASH_000001 | AIRLINE_000001
# Must be done BEFORE annotation so the same ID flows into:
#   → asrs_nlp_annotated.csv   (NLP output)
#   → asrs_faiss_meta.csv      (FAISS metadata)
#   → uatie_edm_merged.ttl     (ontology individuals via Step 6b)
# ─────────────────────────────────────────────────────────────
SOURCE_PREFIX_MAP = {
    "asrs"              : "ASRS",
    "ntsb_database.csv" : "NTSB",
    "plane_crashes.csv" : "CRASH",
    "airline_safety.csv": "AIRLINE",
}

df = df.reset_index(drop=True)

df.insert(0, "incident_id", [
    f"{SOURCE_PREFIX_MAP.get(str(df.at[i, 'source']).strip().lower(), 'UNK')}_{i:06d}"
    for i in range(len(df))
])

print(f"  incident_id assigned : {len(df):,}")
print(f"  Sample IDs           : {df['incident_id'].head(4).tolist()}")
print(f"  Prefix counts :\n{df['incident_id'].str.split('_').str[0].value_counts().to_string()}")
# ── END INCIDENT_ID BLOCK ─────────────────────────────────────

print("\n[1/7] Loading fused corpus...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"  Raw records : {len(df):,}")

# Filter: keep only rows with usable summary text
df = df[df["summary"].notna() & (df["summary"].astype(str).str.len() > 20)].copy()
df["summary"] = df["summary"].astype(str).str.strip()
df = df.reset_index(drop=True)
print(f"  After filter: {len(df):,} records with narratives")

if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"  Sampling    : {SAMPLE_SIZE:,} records for this run")

# ══════════════════════════════════════════════════════════════════════════════
# 2. INTENT CLASSIFICATION
#    7 intent categories aligned with UATIE-EDM ontology
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Intent classification...")

# Weight 2 = specific multi-word phrase | Weight 1 = single indicator word
INTENT_RULES_WEIGHTED = {
    "ATC_Operational_Error": {
        2: ["twr ctlr","atc instruction","issued clrnc","released dep","atc error",
            "incorrect clearance","wrong clearance","flow ctl","ctl instructions",
            "atc ctlr","operror","zob ctlr","zid ctlr","zny ctlr","zfw ctlr"],
        1: ["ctlr","controller","supvr","zob","zid","zny","zfw","zla","vectoring"]
    },
    "Emergency_Event": {
        2: ["declared emer","emer declared","declared emergency","eng fire","cabin smoke",
            "smoke in cockpit","bird strike","gear up lndg","returned dep arpt",
            "emergency descent","diverted due","rapid descent","smoke and fumes", "engine failure",
            "crashed after", "crash after", "forced landing", "emergency landing"],
        1: ["emer","mayday","smoke","fire","depressurization","blow-out"]
    },
    "Runway_Incursion": {
        2: ["runway incursion","crossed runway","entered runway","hold short",
            "near rwy incursion","runway crossing","wrong runway","entered active runway",
            "without clearance to cross"],
        1: ["incursion","txwy","lands on","short final","vacate runway"]
    },
    "Ground_Incident": {
        2: ["pushback damage","ramp collision","ground collision","taxiing into",
            "wingtip strike","nearly collided","wing strike on ground","congested ramp",
            "tow bar disconnect"],
        1: ["pushback","tow bar","fod","tug","ramp","collide","struck","gate"]
    },
    "Communication_Failure": {
        2: ["readback error","read back incorrect","wrong frequency","radio failure",
            "communication failure","lost comms","callsign confusion",
            "failed to respond","clrnc readback","phraseology error","lost communication"],
        1: ["readback","read back","phraseology","frequency","clrnc","freq"]
    },
    "Airspace_Violation": {
        2: ["airspace violation","tcas ra","resolution advisory","near midair","nmac",
            "violated airspace","unauthorized entry","conflict alert","level bust",
            "separation loss","airspace incursion"],
        1: ["tcas","separation","airspace","altdev","rnav","conflict","traffic alert"]
    },
    "IMC_Spatial_Disorientation": {
        2: ["vfr into imc","inadvertent imc","spatial disorientation","entered clouds",
            "encountered imc","lost visual reference","controlled flight into terrain",
            "wake vortex","missed approach","go-around"],
        1: ["disoriented","imc","cfit","wake turb"]
    },
    "Weather_Deviation": {
        2: ["wind shear","windshear","clear air turbulence","deteriorating wx",
            "severe turb","icing condition","weather divert"],
        1: ["turbulence","turb","icing","thunderstorm","lightning","pirep","sigmet",
            "visibility","ceiling","fog","weather","hail","deviat"]
    },
    "Crew_Resource_Management": {
        2: ["crew resource","situational awareness","sterile cockpit","missed callout",
            "crew coordination","challenge response","pilot incapacitation",
            "confusion between crew","low time fo"],
        1: ["crm","workload","fatigue","checklist","briefing","coordination","distraction"]
    },
    "Maintenance_Error": {
        2: ["airworthiness directive","incorrect part installed","dispatched with inop",
            "maint performed","replaced incorrectly","installed wrong","calibrated torque",
            "progressive inspection","interim repair","maint ctlr","maint history",
            "gear rigging","hydraulic leak","fuel leak","fuel imbalance", "engine failure during",
            "engine malfunction", "malfunction during", "engine failure on takeoff",
            "engine failure in flight"],
        1: ["maintenance","technician","maint","squawk","mel","deferred","replaced",
            "inspection","repair","write-up","unairworthy","discrepancy"]
    },
    "Procedural_Non_Compliance": {
        2: ["violated procedure","altitude bust","sop violation","failed to follow",
            "did not follow","wrong approach type","skipped checklist","fmgc error",
            "unauthorized descent","speed deviation","alt constraint"],
        1: ["hdg dev","alt dev","track hdg","omitted","non-standard","drifting","leveled at"]
    },
}

PRIORITY = [
    "ATC_Operational_Error","Emergency_Event","Runway_Incursion",
    "Ground_Incident","Communication_Failure","Airspace_Violation",
    "IMC_Spatial_Disorientation","Weather_Deviation",
    "Crew_Resource_Management","Maintenance_Error","Procedural_Non_Compliance"
]

def classify_intent(text):
    tl = text.lower()
    scores = {}
    for intent, weights in INTENT_RULES_WEIGHTED.items():
        scores[intent] = (
            sum(2 for kw in weights[2] if kw in tl) +
            sum(1 for kw in weights[1] if kw in tl)
        )
    best_score = max(scores.values())
    if best_score == 0:
        return "Unclassified"
    for intent in PRIORITY:
        if scores[intent] == best_score:
            return intent
    return max(scores, key=scores.get)

tqdm.pandas(desc="  Classifying")
df["intent"] = df["summary"].progress_apply(classify_intent)

dist = df["intent"].value_counts()
total = len(df)
print("\\n  Intent distribution:")
for intent, count in dist.items():
    pct = count / total * 100
    flag = " ← majority" if count == dist.max() else ""
    print(f"    {intent:<35} {count:>7,}  ({pct:.1f}%){flag}")
unclass_rate = df["intent"].eq("Unclassified").mean() * 100
print(f"\\n  Classified : {100-unclass_rate:.1f}%  |  Unclassified : {unclass_rate:.1f}%")

# ── Intent distribution summary ──────────────────────────────────────────────
dist  = df["intent"].value_counts()
total = len(df)
print("\n  Intent distribution:")
print(f"  {'Intent':<35} {'Count':>8}  {'%':>6}")
print(f"  {'-'*35} {'-'*8}  {'-'*6}")
for intent, count in dist.items():
    pct  = count / total * 100
    bar  = "█" * int(pct / 2)
    print(f"  {intent:<35} {count:>8,}  {pct:>5.1f}%  {bar}")

unclass_rate = df["intent"].eq("Unclassified").mean() * 100
print(f"\n  {'─'*56}")
print(f"  Total records    : {total:,}")
print(f"  Classified       : {total - dist.get('Unclassified', 0):,}  ({100-unclass_rate:.1f}%)")
print(f"  Unclassified     : {dist.get('Unclassified', 0):,}  ({unclass_rate:.1f}%)")
print(f"  Intent classes   : {df['intent'].nunique()} (incl. Unclassified)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. HAZARD TAGGING (multi-label)
#    9 hazard types → stored as pipe-separated string
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Hazard tagging (multi-label)...")

HAZARD_RULES = {
    "H_Communication":    ["communication", "readback", "phraseology", "radio", "clearance", "frequency"],
    "H_Runway":           ["runway", "taxiway", "ground", "incursion", "hold short", "landing"],
    "H_Airspace":         ["airspace", "separation", "tcas", "ra", "conflict", "traffic"],
    "H_Mechanical":       ["mechanical", "engine", "hydraulic", "landing gear", "inoperative", "failure"],
    "H_Weather":          ["weather", "turbulence", "wind shear", "icing", "fog", "visibility"],
    "H_HumanFactors":     ["fatigue", "distraction", "workload", "crm", "situational awareness", "error"],
    "H_Navigation":       ["navigation", "gps", "ils", "vor", "approach", "localizer", "glide"],
    "H_Fuel":             ["fuel", "low fuel", "fuel exhaustion", "fuel imbalance", "refuel"],
    "H_BirdStrike":       ["bird", "bird strike", "wildlife", "fod"],
}

def tag_hazards(text):
    text_lower = text.lower()
    tags = [h for h, kws in HAZARD_RULES.items() if any(kw in text_lower for kw in kws)]
    return "|".join(tags) if tags else "H_Unclassified"

tqdm.pandas(desc="  Tagging hazards")
df["hazard_tags"] = df["summary"].progress_apply(tag_hazards)
print(f"  Multi-hazard records: {(df['hazard_tags'].str.contains('|', regex=False)).sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PHASE OF FLIGHT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] Phase-of-flight extraction...")

PHASE_KEYWORDS = {
    "Taxi":       ["taxi", "taxiing", "taxied", "ground movement", "pushback"],
    "Takeoff":    ["takeoff", "take-off", "rotation", "v1", "rejected takeoff", "departure roll"],
    "Climb":      ["climb", "climbing", "initial climb", "departure climb"],
    "Cruise":     ["cruise", "cruising", "level flight", "en route", "enroute"],
    "Descent":    ["descent", "descending", "top of descent", "tod"],
    "Approach":   ["approach", "final", "ils", "localizer", "glideslope", "visual approach"],
    "Landing":    ["landing", "touchdown", "rollout", "flare", "go-around", "missed approach"],
    "Parked":     ["parked", "gate", "ramp", "chocks", "pushback"],
    "Unknown":    [],
}

def extract_phase(text):
    text_lower = text.lower()
    for phase, kws in PHASE_KEYWORDS.items():
        if phase == "Unknown":
            continue
        if any(kw in text_lower for kw in kws):
            return phase
    return "Unknown"

tqdm.pandas(desc="  Extracting phase")
df["phase_extracted"] = df["summary"].progress_apply(extract_phase)

# Use existing phase_of_flight from ASRS if available; else use extracted
if "phase_of_flight" in df.columns:
    df["phase_final"] = df["phase_of_flight"].fillna(df["phase_extracted"])
    df["phase_final"] = df["phase_final"].replace("", df["phase_extracted"])
else:
    df["phase_final"] = df["phase_extracted"]

# ══════════════════════════════════════════════════════════════════════════════
# 5. SENTENCE EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] Building sentence embeddings...")
print(f"  Model : {EMBED_MODEL}")
print(f"  Texts : {len(df):,} records")

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    texts = df["summary"].tolist()
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via dot product
        convert_to_numpy=True
    )
    print(f"  Embeddings shape : {embeddings.shape}")
    print(f"  Time taken       : {time.time()-t0:.1f}s")
    EMBEDDINGS_OK = True
except ImportError:
    print("  ⚠ sentence-transformers not installed.")
    print("    Run: pip install sentence-transformers")
    print("    Skipping embedding + FAISS steps.")
    embeddings = None
    EMBEDDINGS_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# 6. FAISS INDEX
# ══════════════════════════════════════════════════════════════════════════════
if EMBEDDINGS_OK:
    print("\n[6/7] Building FAISS index...")
    try:
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)       # Inner product = cosine (since normalized)
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, FAISS_INDEX)
        print(f"  FAISS index  : {index.ntotal:,} vectors · dim={dim}")
        print(f"  Saved to     : {FAISS_INDEX}")
        FAISS_OK = True
    except ImportError:
        print("  ⚠ faiss-cpu not installed. Run: pip install faiss-cpu")
        FAISS_OK = False
else:
    print("\n[6/7] FAISS skipped (no embeddings).")
    FAISS_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# 7. RAG DEMO QUERY
# ══════════════════════════════════════════════════════════════════════════════
DEMO_QUERY = "pilot failed to read back clearance correctly"

if FAISS_OK:
    print(f"\n[7/7] RAG Demo Query: \"{DEMO_QUERY}\"")
    q_emb = model.encode([DEMO_QUERY], normalize_embeddings=True).astype(np.float32)
    distances, indices = index.search(q_emb, 5)
    print("\n  Top-5 similar ASRS incidents:")
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
        row = df.iloc[idx]
        print(f"\n  [{rank}] Score={score:.4f} | Intent={row.get('intent','?')} | Phase={row.get('phase_final','?')}")
        print(f"       {str(row['summary'])[:200]}")
else:
    print("\n[7/7] RAG demo skipped.")

# ══════════════════════════════════════════════════════════════════════════════
# 8. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8/7] Saving outputs...")

output_cols = ["source", "summary", "intent", "hazard_tags", "phase_final"]
if "operator" in df.columns:   output_cols.insert(1, "operator")
if "location" in df.columns:   output_cols.insert(1, "location")
if "event_id" in df.columns:   output_cols.insert(1, "event_id")
output_cols = [c for c in output_cols if c in df.columns]

df[output_cols].to_csv(OUTPUT_CSV, index=False)
print(f"  Annotated CSV : {OUTPUT_CSV}")
print(f"  Records saved : {len(df):,}")

# Save FAISS metadata (record_id → index position for retrieval)
meta_df = df[["source", "summary", "intent", "hazard_tags", "phase_final"]].copy()
meta_df.index.name = "faiss_idx"
meta_df.to_csv(FAISS_META)
print(f"  FAISS meta    : {FAISS_META}")

print("\n" + "=" * 60)
print(" COMPLETE")
print("=" * 60)
print(f"  asrs_nlp_annotated.csv  → {len(df):,} records")
print(f"  Intent classes          → {df['intent'].nunique()}")
print(f"  Hazard types covered    → {len(HAZARD_RULES)}")
if FAISS_OK:
    print(f"  FAISS index size        → {index.ntotal:,} vectors")
print("\nNext: run step6b_ontology_population.py to push")
print("       annotated records into the UATIE-EDM OWL ontology.")
print("=" * 60)
