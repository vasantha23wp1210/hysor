#!/usr/bin/env python3
"""
analyze_semantic_late_runway_change_results.py
----------------------------------------------
Analyze semantic-search results CSV for the scenario:
late runway change / new runway assignment during approach.

Input:
- output/semantic_late_runway_change_incidents.csv

Outputs:
- output/semantic_late_runway_change_incidents_enriched.csv
- output/semantic_late_runway_change_stats.csv
- output/semantic_late_runway_change_hazard_summary.csv
- output/semantic_late_runway_change_scenario_summary.csv
"""

import re
from pathlib import Path
import pandas as pd

OUT_DIR = Path("output")
INPUT_CSV = OUT_DIR / "semantic_late_runway_change_incidents.csv"
ENRICHED_CSV = OUT_DIR / "semantic_late_runway_change_incidents_enriched.csv"
STATS_CSV = OUT_DIR / "semantic_late_runway_change_stats.csv"
HAZARD_CSV = OUT_DIR / "semantic_late_runway_change_hazard_summary.csv"
SCENARIO_CSV = OUT_DIR / "semantic_late_runway_change_scenario_summary.csv"

OUTCOME_PATTERNS = {
    "go_around": r"go around|go-around|missed approach",
    "unstable_approach": r"unstable approach|unstabilized approach|unstabilised approach",
    "wrong_runway": r"wrong runway",
    "runway_confusion": r"runway confusion|confused .* runway|wrong runway assigned",
    "fms_reprogramming": r"fms|mcdu|reprogram|re-program|arrival transition|briefing|setup changed",
    "atc_late_change": r"atc .* changed|approach .* changed|tower .* changed|cleared .* different runway|assigned .* different runway",
    "track_deviation": r"deviation|off course|localizer deviation|heading deviation|track deviation",
}

HAZARD_PATTERNS = {
    "unstable_approach_hazard": r"unstable approach|unstabilized approach|unstabilised approach|high sink rate|fast on approach",
    "runway_misalignment_hazard": r"wrong runway|wrong airport|misaligned|aligned with wrong runway|parallel runway confusion",
    "lateral_vertical_deviation_hazard": r"localizer deviation|glideslope deviation|track deviation|altitude deviation|heading deviation|off course",
    "automation_setup_hazard": r"fms|mcdu|reprogram|re-program|incorrect setup|wrong transition|approach not loaded|runway not loaded",
    "communication_hazard": r"misheard|did not hear|readback|hearback|miscommunication|communication problem|frequency congestion",
    "workload_hazard": r"high workload|task saturation|rushed|time pressure|behind the aircraft|workload increased",
    "go_around_hazard": r"go around|go-around|missed approach",
    "terrain_separation_hazard": r"tcas|gpws|egpws|pull up|terrain|loss of separation|traffic conflict",
}

REGEX_OUTCOMES = {k: re.compile(v, re.I) for k, v in OUTCOME_PATTERNS.items()}
REGEX_HAZARDS = {k: re.compile(v, re.I) for k, v in HAZARD_PATTERNS.items()}


def pick_text_col(df):
    for c in ["combined_text", "narrative", "summary", "synopsis", "text", "report"]:
        if c in df.columns:
            return c
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    return obj_cols[0] if obj_cols else None


def classify_scenario(text: str) -> str:
    t = str(text).lower()
    if "late runway" in t or "last minute runway" in t:
        return "Late runway change"
    if any(x in t for x in ["visual approach", "ils", "localizer", "glideslope"]):
        return "Approach reconfiguration"
    if any(x in t for x in ["new runway", "different runway", "another runway", "reassigned to runway"]):
        return "Runway reassignment"
    return "Runway change during approach"


def infer_phase(existing_phase: str, text: str) -> str:
    p = str(existing_phase or "").strip()
    if p and p.lower() not in {"", "unknown", "nan", "none"}:
        return p
    t = str(text).lower()
    if "final approach" in t or "on final" in t:
        return "final approach"
    if "initial approach" in t:
        return "initial approach"
    if "descent" in t:
        return "descent"
    if "approach" in t or "ils" in t or "visual approach" in t or "localizer" in t:
        return "approach"
    return "Unknown"


def build_stats(df):
    rows = []
    rows.append({"metric": "total_semantic_candidates", "category": "all", "value": int(len(df))})

    if "source" in df.columns:
        for val, cnt in df["source"].fillna("Unknown").value_counts().items():
            rows.append({"metric": "by_source", "category": str(val), "value": int(cnt)})

    if "scenario_type" in df.columns:
        for val, cnt in df["scenario_type"].fillna("Unknown").value_counts().items():
            rows.append({"metric": "by_scenario_type", "category": str(val), "value": int(cnt)})

    if "phase_inferred" in df.columns:
        for val, cnt in df["phase_inferred"].fillna("Unknown").value_counts().items():
            rows.append({"metric": "by_phase_inferred", "category": str(val), "value": int(cnt)})

    if "similarity_score" in df.columns and len(df):
        rows.extend([
            {"metric": "similarity_score", "category": "min", "value": float(df["similarity_score"].min())},
            {"metric": "similarity_score", "category": "mean", "value": float(df["similarity_score"].mean())},
            {"metric": "similarity_score", "category": "max", "value": float(df["similarity_score"].max())},
        ])

    for flag in OUTCOME_PATTERNS.keys():
        rows.append({"metric": "outcome_flag", "category": flag, "value": int(df[flag].fillna(False).sum())})

    rows.append({
        "metric": "avg_inferred_hazards_per_incident",
        "category": "all",
        "value": float(round(df["hazard_count_inferred"].mean(), 3)) if len(df) else 0.0,
    })

    return pd.DataFrame(rows)


def build_hazards(df):
    rows = []
    hazard_cols = [c for c in df.columns if c.endswith("_hazard")]
    for hc in hazard_cols:
        count = int(df[hc].fillna(False).sum())
        pct = round((count / len(df) * 100), 2) if len(df) else 0.0
        rows.append({"hazard": hc, "incident_count": count, "incident_pct": pct})
    return pd.DataFrame(rows).sort_values(["incident_count", "hazard"], ascending=[False, True])


def build_scenario_summary(df):
    rows = []
    if "scenario_type" not in df.columns:
        return pd.DataFrame(columns=["scenario_type", "count", "pct"])
    for val, cnt in df["scenario_type"].fillna("Unknown").value_counts().items():
        pct = round((cnt / len(df) * 100), 2) if len(df) else 0.0
        rows.append({"scenario_type": str(val), "count": int(cnt), "pct": pct})
    return pd.DataFrame(rows)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    text_col = pick_text_col(df)
    if text_col is None:
        raise ValueError("No text column found in semantic results CSV")

    if "phase" not in df.columns:
        df["phase"] = "Unknown"
    if "source" not in df.columns:
        df["source"] = "Unknown"

    df["scenario_type"] = df[text_col].astype(str).map(classify_scenario)
    df["phase_inferred"] = [infer_phase(p, t) for p, t in zip(df["phase"], df[text_col].astype(str))]

    for name, rx in REGEX_OUTCOMES.items():
        df[name] = df[text_col].astype(str).str.contains(rx, na=False)

    for name, rx in REGEX_HAZARDS.items():
        df[name] = df[text_col].astype(str).str.contains(rx, na=False)

    df["hazard_count_inferred"] = df[[c for c in df.columns if c.endswith("_hazard")]].sum(axis=1)
    df["narrative_preview"] = df[text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 500)

    stats = build_stats(df)
    hazards = build_hazards(df)
    scenarios = build_scenario_summary(df)

    df.to_csv(ENRICHED_CSV, index=False)
    stats.to_csv(STATS_CSV, index=False)
    hazards.to_csv(HAZARD_CSV, index=False)
    scenarios.to_csv(SCENARIO_CSV, index=False)

    print(f"Saved enriched incidents : {ENRICHED_CSV} ({len(df)} rows)")
    print(f"Saved stats             : {STATS_CSV} ({len(stats)} rows)")
    print(f"Saved hazards           : {HAZARD_CSV} ({len(hazards)} rows)")
    print(f"Saved scenario summary  : {SCENARIO_CSV} ({len(scenarios)} rows)")


if __name__ == "__main__":
    main()
