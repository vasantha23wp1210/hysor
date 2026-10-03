#!/usr/bin/env python3
"""
rag_query_engine_updated.py
---------------------------
Updated UATIE-EDM RAG engine with:
- semantic + hybrid + ontology retrieval
- recency-aware ranking using year and month
- interactive CLI prompt mode
- single-query mode (--query)
- batch mode (--batch)
- Streamlit compatibility (--ui)

Recommended placement:
- project root: rag_query_engine.py
or
- src/rag/rag_query_engine.py
"""

import os
import re
import sys
import csv
import argparse
import warnings
import logging
import textwrap
import datetime as _dt
import subprocess
import numpy as np
import pandas as pd
from rdflib import Graph, Namespace

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..")) if os.path.basename(os.path.dirname(BASE_DIR)) == 'rag' else BASE_DIR
PROCESSED_DIR = os.path.join("data", "processed")
ANNOTATED_CSV = os.path.join(PROCESSED_DIR, "asrs_nlp_annotated.csv")
FAISS_INDEX = os.path.join(PROCESSED_DIR, "asrs_faiss.index")
FAISS_META = os.path.join(PROCESSED_DIR, "asrs_faiss_meta.csv")
MERGED_TTL = os.path.join(PROCESSED_DIR, "uatie_edm_merged.ttl")
CROSSMAP_CSV = os.path.join(PROCESSED_DIR, "caa_icao_crossmap.csv")
QUERY_LOG = os.path.join(PROCESSED_DIR, "rag_query_log.csv")
LAST_RESULTS = os.path.join(PROCESSED_DIR, "rag_last_results.csv")
APP_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "uatie_edm_app.py"),
    os.path.join(PROJECT_ROOT, "uatie_edm_app_updated.py"),
    os.path.join(PROJECT_ROOT, "src", "uatie_edm_app.py"),
]

TOP_K = 5
EMBED_MODEL = "all-MiniLM-L6-v2"
BASE_NS = "http://aviation.safety.ontology/"
SIM_WEIGHT = 0.65
RECENCY_WEIGHT = 0.35
RECENCY_MIN_YEAR = 1990
RECENCY_MAX_YEAR = _dt.datetime.now().year

INTENT_SLUG_MAP = {
    "atc": "ATC_OPERATIONAL_ERROR",
    "communication": "COMMUNICATION_FAILURE",
    "comm": "COMMUNICATION_FAILURE",
    "runway": "RUNWAY_INCURSION",
    "weather": "WEATHER_DEVIATION",
    "maintenance": "MAINTENANCE_ERROR",
    "maint": "MAINTENANCE_ERROR",
    "emergency": "EMERGENCY_EVENT",
    "ground": "GROUND_INCIDENT",
    "imc": "IMC_SPATIAL_DISORIENTATION",
    "crm": "CREW_RESOURCE_MANAGEMENT",
    "procedural": "PROCEDURAL_NON_COMPLIANCE",
    "airspace": "AIRSPACE_VIOLATION",
}

KW_INTENT_MAP = {
    "readback": "COMMUNICATION_FAILURE",
    "clearance": "COMMUNICATION_FAILURE",
    "phraseology": "COMMUNICATION_FAILURE",
    "frequency": "COMMUNICATION_FAILURE",
    "runway": "RUNWAY_INCURSION",
    "taxiway": "RUNWAY_INCURSION",
    "incursion": "RUNWAY_INCURSION",
    "weather": "WEATHER_DEVIATION",
    "turbulence": "WEATHER_DEVIATION",
    "icing": "WEATHER_DEVIATION",
    "engine": "MAINTENANCE_ERROR",
    "mechanical": "MAINTENANCE_ERROR",
    "maintenance": "MAINTENANCE_ERROR",
    "controller": "ATC_OPERATIONAL_ERROR",
    "ctlr": "ATC_OPERATIONAL_ERROR",
    "atc": "ATC_OPERATIONAL_ERROR",
    "smoke": "EMERGENCY_EVENT",
    "fire": "EMERGENCY_EVENT",
    "emergency": "EMERGENCY_EVENT",
    "bird": "EMERGENCY_EVENT",
    "fatigue": "CREW_RESOURCE_MANAGEMENT",
    "crm": "CREW_RESOURCE_MANAGEMENT",
    "workload": "CREW_RESOURCE_MANAGEMENT",
    "imc": "IMC_SPATIAL_DISORIENTATION",
    "cloud": "IMC_SPATIAL_DISORIENTATION",
    "disoriented": "IMC_SPATIAL_DISORIENTATION",
    "collision": "GROUND_INCIDENT",
    "pushback": "GROUND_INCIDENT",
    "ramp": "GROUND_INCIDENT",
    "airspace": "AIRSPACE_VIOLATION",
    "tcas": "AIRSPACE_VIOLATION",
    "separation": "AIRSPACE_VIOLATION",
}

RISK_MATRIX = {
    ("Emergency_Event", "Takeoff"): "CRITICAL",
    ("Emergency_Event", "Approach"): "CRITICAL",
    ("Emergency_Event", "Landing"): "CRITICAL",
    ("Runway_Incursion", "Takeoff"): "CRITICAL",
    ("Runway_Incursion", "Approach"): "HIGH",
    ("Runway_Incursion", "Taxi"): "HIGH",
    ("Imc_Spatial_Disorientation", "Cruise"): "HIGH",
    ("Imc_Spatial_Disorientation", "Descent"): "CRITICAL",
    ("Communication_Failure", "Approach"): "HIGH",
    ("Communication_Failure", "Takeoff"): "HIGH",
    ("Atc_Operational_Error", "Approach"): "HIGH",
    ("Weather_Deviation", "Approach"): "HIGH",
    ("Weather_Deviation", "Takeoff"): "HIGH",
    ("Maintenance_Error", "Takeoff"): "HIGH",
    ("Ground_Incident", "Taxi"): "MEDIUM",
    ("Airspace_Violation", "Cruise"): "MEDIUM",
    ("Crew_Resource_Management", "Unknown"): "MEDIUM",
    ("Procedural_Non_Compliance", "Unknown"): "LOW",
}

HAZARD_WEIGHTS = {
    "H_Runway": 3,
    "H_Mechanical": 3,
    "H_Fuel": 3,
    "H_Weather": 2,
    "H_Navigation": 2,
    "H_Airspace": 2,
    "H_BirdStrike": 2,
    "H_Communication": 1,
    "H_HumanFactors": 1,
    "H_Unclassified": 0,
}

RISK_ICONS = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}

FAISS_OK = EMBED_OK = ONTO_OK = SEMANTIC_OK = False
meta = None
df_full = None
index = None
model = None
g_onto = None
onto_graph = None
phraseology_df = pd.DataFrame()
_year_lookup = {}
_date_lookup = {}
_phraseme_map = None
session_log = []


def _extract_date_parts(date_val):
    if date_val is None:
        return None, None
    if isinstance(date_val, float) and pd.isna(date_val):
        return None, None
    s = str(date_val).strip()
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y"]:
        try:
            dt = _dt.datetime.strptime(s[:10], fmt)
            return dt.year, dt.month
        except Exception:
            pass
    m = re.search(r"\b(19|20)\d{2}\b", s)
    if m:
        return int(m.group(0)), None
    return None, None


def compute_recency_score(year, month=None):
    if year is None:
        return 0.5
    try:
        year = int(year)
    except Exception:
        return 0.5
    month = 6 if month in (None, "", "Unknown") else max(1, min(12, int(month)))
    recency_point = year + (month - 1) / 12.0
    min_point = RECENCY_MIN_YEAR
    max_point = RECENCY_MAX_YEAR + 11 / 12.0
    recency_point = max(min_point, min(max_point, recency_point))
    span = max_point - min_point
    return 1.0 if span == 0 else round((recency_point - min_point) / span, 4)


def compute_composite_score(sim_score, year, month=None):
    rec = compute_recency_score(year, month)
    return round(SIM_WEIGHT * float(sim_score) + RECENCY_WEIGHT * rec, 4)


def _first_existing(df, candidates, default=""):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def _safe_get(row, keys, default=""):
    for k in keys:
        if k in row.index:
            val = row.get(k)
            if val is not None and str(val).strip() not in {"", "nan", "None"}:
                return val
    return default


def _stable_incident_id_from_row(row, idx):
    for key in ["incident_id", "Incident_ID", "asrs_id", "ASRS_ID", "report_id", "id"]:
        if key in row.index:
            val = row.get(key)
            if val is not None and str(val).strip() not in {"", "nan", "None"}:
                return str(val).strip()
    txt = str(_safe_get(row, ["summary", "text", "narrative", "Narrative"], "")).strip()
    return f"FAISS_{idx}_{txt[:40].replace(',', ' ').replace(chr(10), ' ').strip()}" if txt else f"FAISS_{idx}"


def _norm_intent(intent):
    if not intent:
        return "Unknown"
    s = str(intent).strip().replace("-", "_").replace(" ", "_")
    parts = [p.capitalize() for p in s.split("_") if p]
    return "_".join(parts) if parts else "Unknown"


def compute_risk_score(intent, phase, hazards_str):
    key = (_norm_intent(intent), str(phase or "Unknown").strip())
    if key in RISK_MATRIX:
        return RISK_MATRIX[key]
    h_str = str(hazards_str or "").strip()
    hazard_list = h_str.split("|") if h_str and h_str.lower() not in ("—", "nan", "none") else []
    weight = sum(HAZARD_WEIGHTS.get(h.strip(), 0) for h in hazard_list)
    if weight >= 5:
        return "CRITICAL"
    if weight >= 3:
        return "HIGH"
    if weight >= 1:
        return "MEDIUM"
    return "LOW"


def _year_month_label(year, month):
    if year is None:
        return "Date=Unknown"
    if month is None:
        return f"Year={year}"
    return f"Date={year}-{int(month):02d}"


def clean_icao_phrase(raw_phrase):
    if not raw_phrase or raw_phrase == "—":
        return raw_phrase
    cleaned = re.sub(r"===\s*PAGE\s*\d+\s*===", "", str(raw_phrase))
    cleaned = re.sub(r"\d+[-–]\d+\s+[A-Z][A-Za-z\s]+\(PANS[^\)]*\)\s+\d{2}/\d{2}/\d{2}", "", cleaned)
    cleaned = re.sub(r"\b[a-z]{2}\)\s", " ", cleaned)
    cleaned = re.sub(r"\.{3,}", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if ";" in cleaned:
        cleaned = cleaned.split(";")[0].strip() + ";"
    if len(cleaned) > 120:
        cleaned = cleaned[:117].rsplit(" ", 1)[0] + "..."
    return cleaned


def initialize_resources(verbose=True):
    global meta, df_full, index, model, g_onto, onto_graph, phraseology_df
    global FAISS_OK, EMBED_OK, ONTO_OK, SEMANTIC_OK, _year_lookup, _date_lookup
    if meta is not None and df_full is not None and (FAISS_OK or ONTO_OK or not verbose):
        return
    if verbose:
        print("=" * 65)
        print(" UATIE-EDM · Updated RAG Query Engine")
        print("=" * 65)
    if verbose:
        print("\n[1/6] Loading FAISS metadata...")
    meta = pd.read_csv(FAISS_META, low_memory=False)
    if "faiss_idx" in meta.columns:
        meta = meta.set_index("faiss_idx")
    if verbose:
        print(f" Rows : {len(meta):,}")
    if verbose:
        print("[2/6] Loading annotated corpus...")
    df_full = pd.read_csv(ANNOTATED_CSV, low_memory=False)
    if verbose:
        print(f" Records : {len(df_full):,}")
       # ── build year/date lookup from FAISS meta (positionally aligned) ──
    _year_lookup = {}
    _date_lookup = {}

    # check meta first (has date column from patch)
    _meta_date_col = None
    for _c in ["date", "event_date", "report_date", "datetime"]:
        if _c in meta.columns:
            _meta_date_col = _c
            break

    if _meta_date_col:
        # meta is positionally aligned with FAISS index
        # use reset_index to get 0-based position
        for _pos, _mrow in meta.reset_index(drop=True).iterrows():
            _yr, _mo = _extract_date_parts(_mrow[_meta_date_col])
            if _yr is None:
                continue
            # index by FAISS position
            _year_lookup[str(_pos)] = _yr
            _date_lookup[str(_pos)] = (_yr, _mo)
            # index by incident_id / record_id
            for _idc in ["incident_id", "record_id", "event_id"]:
                if _idc in meta.columns:
                    _iv = str(_mrow.get(_idc, "")).strip()
                    if _iv and _iv not in ("nan", "None", ""):
                        _year_lookup[_iv] = _yr
                        _date_lookup[_iv] = (_yr, _mo)
                    break
        if verbose:
            print(f"  Date lookup : {len(_date_lookup):,} entries "
                  f"(from meta col={_meta_date_col})")
    else:
        # fallback to df_full if meta has no date
        date_col = _first_existing(
            df_full, ["date", "event_date", "report_date",
                      "incident_date", "Date", "EVENT_DATE"]
        )
        id_col = _first_existing(
            df_full, ["incident_id", "asrs_id", "id", "report_id"]
        )
        if date_col:
            for pos, row in df_full.iterrows():
                yr, mo = _extract_date_parts(row[date_col])
                if yr is None:
                    continue
                _year_lookup[str(pos)] = yr
                _date_lookup[str(pos)] = (yr, mo)
                if id_col:
                    _year_lookup[str(row[id_col])] = yr
                    _date_lookup[str(row[id_col])] = (yr, mo)
            if verbose:
                print(f"  Date lookup : {len(_date_lookup):,} entries "
                      f"(fallback df_full col={date_col})")
        else:
            if verbose:
                print("  ⚠ No date column found — recency will use unknown dates")
    if verbose:
        print("[3/6] Loading FAISS index...")
    try:
        import faiss
        index = faiss.read_index(FAISS_INDEX)
        FAISS_OK = True
        if verbose:
            print(f" Vectors : {index.ntotal:,} | dim={index.d}")
    except Exception as e:
        if verbose:
            print(f" ⚠ FAISS load failed: {e}")
        FAISS_OK = False
        index = None
    if verbose:
        print("[4/6] Loading embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        EMBED_OK = True
        if verbose:
            print(f" Model : {EMBED_MODEL} ✓")
    except Exception as e:
        if verbose:
            print(f" ⚠ sentence-transformers load failed: {e}")
        EMBED_OK = False
        model = None
    if verbose:
        print("[5/6] Loading merged ontology...")
    try:
        g_onto = Graph()
        g_onto.parse(MERGED_TTL, format="turtle")
        onto_graph = g_onto
        ONTO_OK = True
        if verbose:
            print(f" Triples : {len(g_onto):,}")
    except Exception as e:
        if verbose:
            print(f" ⚠ Ontology load failed: {e}")
        ONTO_OK = False
        g_onto = None
        onto_graph = None
    if verbose:
        print("[6/6] Loading phraseology crossmap...")
    if os.path.exists(CROSSMAP_CSV):
        phraseology_df = pd.read_csv(CROSSMAP_CSV, low_memory=False)
        if verbose:
            print(f" Entries : {len(phraseology_df):,}")
            print(f" Columns : {phraseology_df.columns.tolist()}")
    else:
        phraseology_df = pd.DataFrame()
        if verbose:
            print(f" ⚠ Not found: {CROSSMAP_CSV}")
    SEMANTIC_OK = FAISS_OK and EMBED_OK


def embed_query(text):
    if not EMBED_OK or model is None:
        raise RuntimeError("Embedding model not loaded.")
    return model.encode([text], normalize_embeddings=True).astype(np.float32)


def _get_phraseme_map():
    global _phraseme_map
    if _phraseme_map is not None:
        return _phraseme_map
    _phraseme_map = {}
    if not ONTO_OK or g_onto is None:
        return _phraseme_map
    sparql = f"""
    PREFIX uatie: <{BASE_NS}uatie#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?iid ?phraseme WHERE {{
      ?inc uatie:hasIncidentID ?iid .
      ?inc uatie:hasDirectPhrasemeError ?ph .
      ?ph rdfs:label ?phraseme .
    }}
    """
    try:
        _phraseme_map = {str(r.iid): str(r.phraseme) for r in g_onto.query(sparql)}
    except Exception:
        _phraseme_map = {}
    return _phraseme_map


def semantic_search(query, k=TOP_K):
    if not SEMANTIC_OK or index is None or meta is None:
        return []
    q_emb = embed_query(query)
    distances, indices = index.search(q_emb, k)
    results = []
    for rank, (idx, sim_score) in enumerate(zip(indices[0], distances[0]), 1):
        if idx == -1:
            continue
        if 0 <= idx < len(meta):
            row = meta.iloc[idx]
            inc_id = _stable_incident_id_from_row(row, idx)
            # position-first (most reliable — meta is positionally aligned with FAISS)
            year, month = _date_lookup.get(str(idx), (None, None))
            if year is None:
                year, month = _date_lookup.get(str(int(idx)), (None, None))
            if year is None:
                year, month = _date_lookup.get(inc_id, (None, None))
            comp_score = compute_composite_score(float(sim_score), year, month)
            results.append({
                "rank": rank,
                "score": round(float(sim_score), 4),
                "recency_score": compute_recency_score(year, month),
                "composite_score": comp_score,
                "year": int(year) if year else "Unknown",
                "month": int(month) if month else "Unknown",
                "intent": str(_safe_get(row, ["intent", "Intent"], "?")),
                "hazards": str(_safe_get(row, ["hazard_tags", "hazards", "Hazards"], "?")),
                "phase": str(_safe_get(row, ["phase_final", "phase", "Phase"], "?")),
                "source": str(_safe_get(row, ["source", "Source"], "ASRS")),
                "text": str(_safe_get(row, ["summary", "text", "narrative", "Narrative"], "")),
                "incident_id": inc_id,
                "phraseme_link": "—",
            })
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i
    return results


def hybrid_search(query, k=TOP_K):
    results = semantic_search(query, k)
    phras_map = _get_phraseme_map()
    for res in results:
        iid = res.get("incident_id", "")
        res["phraseme_link"] = phras_map.get(iid, "—")
    return results


def ontology_search(intent_slug, limit=TOP_K):
    if onto_graph is None:
        return []
    sparql = f"""
    PREFIX uatie: <{BASE_NS}uatie#>
    PREFIX intent: <{BASE_NS}intent#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?inc ?narrative ?phase (GROUP_CONCAT(DISTINCT ?hazardLabel; SEPARATOR="|") AS ?hazards)
    WHERE {{
      ?inc uatie:hasIntent intent:{intent_slug} .
      ?inc uatie:hasNarrativeText ?narrative .
      OPTIONAL {{ ?inc uatie:hasPhaseOfFlight ?pn . ?pn rdfs:label ?phase . }}
      OPTIONAL {{ ?inc uatie:hasHazard ?hn . ?hn rdfs:label ?hazardLabel . }}
    }}
    GROUP BY ?inc ?narrative ?phase
    LIMIT {int(limit)}
    """
    try:
        rows = onto_graph.query(sparql)
        hits = []
        for row in rows:
            hits.append({
                "rank": 0,
                "score": 1.0,
                "recency_score": 0.5,
                "composite_score": 1.0,
                "year": "Unknown",
                "month": "Unknown",
                "intent": intent_slug.replace("_", " ").title(),
                "phase": str(row.phase) if row.phase else "Unknown",
                "source": "SPARQL",
                "hazards": str(row.hazards) if row.hazards else "—",
                "text": str(row.narrative) if row.narrative else "",
                "incident_id": "",
                "phraseme_link": "—",
            })
        return hits
    except Exception:
        return []


def ontology_nl_search(query, limit=TOP_K):
    ql = query.lower()
    for kw, intent in KW_INTENT_MAP.items():
        if kw in ql:
            return intent, ontology_search(intent, limit)
    return None, []


def get_dynamic_recommendation(intent_slug, limit=3):
    if onto_graph is None or phraseology_df.empty or not intent_slug:
        return []
    intent_slug = str(intent_slug).strip().upper()
    sparql = f"""
    PREFIX uatie: <{BASE_NS}uatie#>
    PREFIX intent: <{BASE_NS}intent#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?phraseme ?label (COUNT(?inc) AS ?freq)
    WHERE {{
      ?inc uatie:hasIntent intent:{intent_slug} .
      {{ ?inc uatie:hasDirectPhrasemeError ?phraseme . }} UNION {{ ?inc uatie:hasCommunicationContext ?phraseme . }}
      ?phraseme rdfs:label ?label .
    }}
    GROUP BY ?phraseme ?label
    ORDER BY DESC(?freq)
    LIMIT {int(limit)}
    """
    try:
        rows = onto_graph.query(sparql)
    except Exception:
        return []
    recs = []
    cols = set(phraseology_df.columns)
    for row in rows:
        label = str(getattr(row, "label", "") or "").strip()
        frequency = int(getattr(row, "freq", 0) or 0)
        icao_phrase = label or "—"
        caa_phrase = "—"
        icao_ref = "ICAO Doc 4444"
        match_score = "—"
        speaker = "UNKNOWN"
        if label and "Best_ICAO_Match" in cols:
            match = phraseology_df[phraseology_df["Best_ICAO_Match"].astype(str).str.contains(re.escape(label), case=False, na=False)]
            if not match.empty:
                row_data = match.iloc[0]
                icao_phrase = str(row_data.get("Best_ICAO_Match", label) or label)
                caa_phrase = str(row_data.get("Phraseology", "—") or "—")
                section = str(row_data.get("Section_Number", "") or "").strip()
                sec_title = str(row_data.get("Section_Title", "") or "").strip()
                match_score = str(row_data.get("Match_Score", "—") or "—")
                speaker = str(row_data.get("Speaker", "UNKNOWN") or "UNKNOWN")
                if section and sec_title:
                    icao_ref = f"CAA §{section} — {sec_title}"
                elif section:
                    icao_ref = f"CAA §{section}"
        recs.append({
            "phrase": clean_icao_phrase(icao_phrase),
            "caa_phrase": caa_phrase,
            "icao_ref": icao_ref,
            "match_score": match_score,
            "speaker": speaker,
            "frequency": frequency,
        })
    return recs


def fmt_result(r):
    sim_pct = round(float(r.get("score", 0.0)) * 100, 1)
    risk = compute_risk_score(r.get("intent", "Unknown"), r.get("phase", "Unknown"), r.get("hazards", ""))
    body = textwrap.fill(str(r.get("text", ""))[:280], width=64, initial_indent=" ", subsequent_indent=" ")
    phras = r.get("phraseme_link", "—")
    phras_line = f"\n Phraseme : {phras}" if phras != "—" else ""
    return (
        f"\n [{r.get('rank', '?')}] Score={r.get('score', 0.0):.4f} ({sim_pct}%) Comp={r.get('composite_score', 0.0):.4f} "
        f"{_year_month_label(r.get('year'), r.get('month'))} Risk={RISK_ICONS.get(risk, '')} {risk}\n"
        f" Intent : {r.get('intent', 'Unknown')}\n"
        f" Phase : {r.get('phase', 'Unknown')} | Source={r.get('source', 'ASRS')}{phras_line}\n"
        f" Hazards : {r.get('hazards', '—')}\n"
        f"{body}\n {'─' * 61}"
    )


def log_query(mode, query, results):
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top = results[0]["text"][:120] if results else ""
    sc = results[0]["score"] if results else 0.0
    session_log.append({"timestamp": ts, "mode": mode, "query": query, "top_result": top, "score": sc})


def save_log():
    if not session_log:
        return
    pd.DataFrame(session_log).to_csv(QUERY_LOG, index=False)
    print(f"Log saved → {QUERY_LOG} ({len(session_log)} queries)")


def save_results(results):
    if not results:
        return
    pd.DataFrame(results).to_csv(LAST_RESULTS, index=False)
    print(f"Results saved → {LAST_RESULTS} ({len(results)} rows)")


def print_recommendations(intent_slug):
    recs = get_dynamic_recommendation(intent_slug)
    if not recs:
        print(f"\n ── No phraseology links found for {intent_slug}")
        return
    print("\n ── ICAO Phraseology Recommendation (from corpus) ──────")
    for j, rec in enumerate(recs, 1):
        print(f" [{j}] {rec['phrase']:<40} → {rec['frequency']:,} incidents")
        print(f" CAA phrase : {rec['caa_phrase']}")
        print(f" Speaker : {rec['speaker']}")
        print(f" Ref : {rec['icao_ref']}")
        print(f" Match score : {rec['match_score']}")
        print(f" {'─' * 56}")


def run_search(query, mode="hybrid", topk=TOP_K):
    if mode == "semantic":
        results = semantic_search(query, topk)
        matched_intent = None
    elif mode == "ontology":
        matched_intent, results = ontology_nl_search(query, topk)
    else:
        results = hybrid_search(query, topk)
        matched_intent = None
    return results, matched_intent


def process_query(query, mode="hybrid", topk=TOP_K, export=False):
    results, matched_intent = run_search(query, mode, topk)
    log_query(mode, query, results)
    print(f"\nQuery: {query}")
    print(f"Mode : {mode} | Hits={len(results)}")
    if matched_intent:
        print(f"Matched ontology intent: {matched_intent}")
    for r in results:
        print(fmt_result(r))
    if results:
        top_intent = str(results[0].get("intent", "")).replace(" ", "_").upper()
        print_recommendations(top_intent)
    if export:
        save_results(results)
    return results


def interactive_prompt(mode="hybrid", topk=TOP_K):
    print("\nEnter your query. Type 'exit' to quit, 'stats' for system status, 'save' to write the query log.")
    while True:
        try:
            query = input("\nUATIE-EDM query> ").strip()
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break
        if not query:
            continue
        ql = query.lower()
        if ql in {"exit", "quit", "q"}:
            break
        if ql == "save":
            save_log()
            continue
        if ql == "stats":
            print(f"Semantic ready: {SEMANTIC_OK} | Ontology ready: {ONTO_OK} | Meta rows: {0 if meta is None else len(meta)}")
            continue
        process_query(query, mode=mode, topk=topk, export=False)
    save_log()


def run_batch(batch_file, mode="hybrid", topk=TOP_K):
    if not os.path.exists(batch_file):
        raise FileNotFoundError(batch_file)
    with open(batch_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    all_rows = []
    for q in queries:
        results, matched = run_search(q, mode, topk)
        for r in results:
            all_rows.append({"query": q, "matched_intent": matched or "", **r})
    if all_rows:
        pd.DataFrame(all_rows).to_csv(LAST_RESULTS, index=False)
        print(f"Batch results saved → {LAST_RESULTS} ({len(all_rows)} rows)")


def launch_ui():
    for app_path in APP_CANDIDATES:
        if os.path.exists(app_path):
            subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=False)
            return
    raise FileNotFoundError("Could not find uatie_edm_app.py or uatie_edm_app_updated.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="single query text")
    parser.add_argument("--batch", type=str, help="path to text file with one query per line")
    parser.add_argument("--mode", choices=["hybrid", "semantic", "ontology"], default="hybrid")
    parser.add_argument("--topk", type=int, default=TOP_K)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--ui", action="store_true")
    args = parser.parse_args()

    initialize_resources(verbose=True)

    if args.ui:
        launch_ui()
        return
    if args.query:
        process_query(args.query, mode=args.mode, topk=args.topk, export=args.export)
        return
    if args.batch:
        run_batch(args.batch, mode=args.mode, topk=args.topk)
        return
    interactive_prompt(mode=args.mode, topk=args.topk)


if __name__ == "__main__":
    main()
