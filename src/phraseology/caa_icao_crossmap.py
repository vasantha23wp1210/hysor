#!/usr/bin/env python3

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── 0. Paths ────────────────────────────────────────────────────────────────
BASE_DIR   = "data/processed/"#os.path.dirname(os.path.abspath(__file__))
#OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed/")
#os.makedirs(OUTPUT_DIR, exist_ok=True)

CAA_FILE         = os.path.join(BASE_DIR, "CAA_ATC_section8_enriched.csv")
ICAO_AUTO_FILE   = os.path.join(BASE_DIR,   "icao_phraseology_auto.csv")
ICAO_MASTER_FILE = os.path.join(BASE_DIR,   "icao_phraseology_master.csv")
ICAO_DOC_FILE    = os.path.join(BASE_DIR,   "doc4444_12_3_ATC_phraseologies.csv")
OUTPUT_FILE      = os.path.join(BASE_DIR, "CAA_ICAO_crossmap.csv")

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
caa         = pd.read_csv(CAA_FILE)
icao_auto   = pd.read_csv(ICAO_AUTO_FILE)
icao_master = pd.read_csv(ICAO_MASTER_FILE)
icao_doc    = pd.read_csv(ICAO_DOC_FILE)

print(f"  CAA phrases   : {len(caa)}")
print(f"  ICAO auto     : {len(icao_auto)}")
print(f"  ICAO master   : {len(icao_master)}")
print(f"  ICAO doc4444  : {len(icao_doc)}")

# ── 2. Consolidate ICAO phrases ───────────────────────────────────────────────
print("\nConsolidating ICAO phrases...")
icao_phrases = pd.concat([
    icao_auto[["standard_phrase"]].rename(columns={"standard_phrase": "phraseology"}),
    icao_master[["standard_phrase"]].rename(columns={"standard_phrase": "phraseology"}),
    icao_doc[["phraseology"]]
], ignore_index=True).drop_duplicates(subset="phraseology")

print(f"  Total unique ICAO phrases: {len(icao_phrases)}")

# ── 3. Clean text for matching ────────────────────────────────────────────────
def clean_phrase(text):
    """Remove variable placeholders and normalise to uppercase."""
    text = re.sub(r"\(.*?\)", "", str(text))   # remove (variables)
    text = re.sub(r"\s+", " ", text)             # collapse whitespace
    return text.upper().strip()

icao_phrases["phraseology_clean"] = icao_phrases["phraseology"].apply(clean_phrase)
caa["phraseology_clean"]          = caa["Phraseology"].apply(clean_phrase)

# ── 4. TF-IDF cosine similarity ───────────────────────────────────────────────
print("\nBuilding TF-IDF vectors and computing similarity...")
corpus = list(icao_phrases["phraseology_clean"]) + list(caa["phraseology_clean"])

vectorizer  = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
tfidf_matrix = vectorizer.fit_transform(corpus)

icao_vecs = tfidf_matrix[:len(icao_phrases)]
caa_vecs  = tfidf_matrix[len(icao_phrases):]

sim_matrix = cosine_similarity(caa_vecs, icao_vecs)

# Best match per CAA phrase
best_idx   = sim_matrix.argmax(axis=1)
best_score = sim_matrix.max(axis=1)

caa["Best_ICAO_Match"] = icao_phrases["phraseology"].iloc[best_idx].values
caa["Match_Score"]     = best_score.round(3)

# ── 5. Classify match type ────────────────────────────────────────────────────
caa["Match_Type"] = pd.cut(
    best_score,
    bins=[-0.01, 0.30, 0.60, 0.85, 1.01],
    labels=["No Match", "Partial Match", "Close Match", "Exact/Near-Exact"]
)

# ── 6. Save output ────────────────────────────────────────────────────────────
output_cols = [
    "Section_Number", "Section_Title", "ATC_Service",
    "Circumstances", "Phraseology", "Speaker", "Variable_Count",
    "Best_ICAO_Match", "Match_Score", "Match_Type"
]
caa[output_cols].to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE}")

# ── 7. Summary report ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  CROSS-MAP SUMMARY")
print("=" * 55)
print(f"  Total CAA phrases         : {len(caa)}")
print(f"  Exact / Near-Exact match  : {(caa['Match_Type'] == 'Exact/Near-Exact').sum()}")
print(f"  Close Match               : {(caa['Match_Type'] == 'Close Match').sum()}")
print(f"  Partial Match             : {(caa['Match_Type'] == 'Partial Match').sum()}")
print(f"  No Match  (CAA-unique)    : {(caa['Match_Type'] == 'No Match').sum()}")

print("\n  CAA-Unique phrases by ATC Service:")
print(caa[caa["Match_Type"] == "No Match"]["ATC_Service"].value_counts().to_string())
print("=" * 55)

