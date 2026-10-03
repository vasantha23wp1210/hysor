#!/usr/bin/env python3
"""
extract_late_runway_change_stats.py
-----------------------------------
Extract all candidate ASRS incidents related to late runway change / new runway
assignment during approach, then generate incident-level and statistics CSVs.
"""

import re
from pathlib import Path
import pandas as pd

INPUT_FILES = [
    r"data/processed/fused_with_asrs.csv",
    r"data/processed/asrs_for_fusion.csv",
]

OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

INCIDENTS_CSV = OUT_DIR / "late_runway_change_approach_incidents.csv"
STATS_CSV = OUT_DIR / "late_runway_change_approach_stats.csv"
SUMMARY_CSV = OUT_DIR / "late_runway_change_summary.csv"

PATTERNS_POSITIVE = [
    r"runway change",
    r"changed runway",
    r"change to runway",
    r"runway was changed",
    r"new runway",
    r"assigned runway",
    r"runway assignment changed",
    r"switched to runway",
    r"another runway",
    r"different runway",
    r"runway reassign",
    r"reassigned to runway",
    r"late runway",
    r"last minute runway",
    r"runway swap",
    r"change from runway",
    r"changed to runway",
]

PATTERNS_APPROACH = [
    r"approach",
    r"final",
    r"on final",
    r"final approach",
    r"visual approach",
    r"ils",
    r"localizer",
    r"glideslope",
    r"landing",
    r"arrival",
    r"descent",
]

PATTERNS_NEGATIVE = [
    r"takeoff",
    r"departure",
    r"departing",
    r"departed",
    r"taxi",
    r"pushback",
]

OUTCOME_PATTERNS = {
    "go_around": r"go around|go-around|missed approach",
    "unstable_approach": r"unstable approach|unstabilized approach|unstabilised approach",
    "wrong_runway": r"wrong runway",
    "runway_confusion": r"runway confusion|confused .* runway|wrong runway assigned",
    "fms_reprogramming": r"fms|mcdu|reprogram|re-program|arrival transition|briefing|setup changed",
    "atc_late_change": r"atc .* changed|approach .* changed|tower .* changed|cleared .* different runway|assigned .* different runway",
    "track_deviation": r"deviation|off course|localizer deviation|heading deviation|track deviation",
}

REGEX_POS = re.compile("|".join(PATTERNS_POSITIVE), re.I)
REGEX_APP = re.compile("|".join(PATTERNS_APPROACH), re.I)
REGEX_NEG = re.compile("|".join(PATTERNS_NEGATIVE), re.I)
REGEX_OUTCOMES = {k: re.compile(v, re.I) for k, v in OUTCOME_PATTERNS.items()}


def choose_text_col(df):
    cols = {c.lower(): c for c in df.columns}
    for cand in ["combined_text", "narrative", "summary", "synopsis", "text", "report"]:
        if cand in cols:
            return cols[cand]
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


def extract_candidates(file_path: str) -> tuple[pd.DataFrame, dict]:
    path = Path(file_path)
    if not path.exists():
        return pd.DataFrame(), {"file": file_path, "exists": False, "matches": 0}

    df = pd.read_csv(path, low_memory=False)
    text_col = choose_text_col(df)
    if text_col is None:
        return pd.DataFrame(), {"file": file_path, "exists": True, "matches": 0, "text_col": None}

    cols = {c.lower(): c for c in df.columns}
    phase_col = cols.get("phase_final") or cols.get("phase") or cols.get("phase_of_flight")
    intent_col = cols.get("intent")
    id_col = cols.get("incident_id") or cols.get("id") or cols.get("report_id")
    source_col = cols.get("source")

    s = df[text_col].astype(str)
    mask = s.str.contains(REGEX_POS, na=False) & s.str.contains(REGEX_APP, na=False)

    if phase_col:
        phase_s = df[phase_col].astype(str)
        mask = mask | (
            phase_s.str.contains("approach", case=False, na=False) &
            s.str.contains(REGEX_POS, na=False)
        )

    mask = mask & (~(s.str.contains(REGEX_NEG, na=False) & ~s.str.contains(REGEX_APP, na=False)))
    sub = df.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame(), {"file": file_path, "exists": True, "matches": 0, "text_col": text_col}

    out = pd.DataFrame({
        "source_file": file_path,
        "incident_id": sub[id_col].astype(str) if id_col else "",
        "source": sub[source_col].astype(str) if source_col else "ASRS",
        "intent": sub[intent_col].astype(str) if intent_col else "Unknown",
        "phase": sub[phase_col].astype(str) if phase_col else "Unknown",
        "narrative": sub[text_col].astype(str).str.replace(r"\s+", " ", regex=True),
    })

    out["scenario_type"] = out["narrative"].map(classify_scenario)
    out["phase_inferred"] = [infer_phase(p, t) for p, t in zip(out["phase"], out["narrative"])]

    for name, rx in REGEX_OUTCOMES.items():
        out[name] = out["narrative"].str.contains(rx, na=False)

    out["narrative_preview"] = out["narrative"].str.slice(0, 500)

    meta = {
        "file": file_path,
        "exists": True,
        "matches": len(out),
        "text_col": text_col,
    }
    return out, meta


def build_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({"metric": "total_candidates", "category": "all", "value": int(len(df))})

    for val, cnt in df["source"].fillna("Unknown").value_counts().items():
        rows.append({"metric": "by_source", "category": str(val), "value": int(cnt)})

    for val, cnt in df["scenario_type"].fillna("Unknown").value_counts().items():
        rows.append({"metric": "by_scenario_type", "category": str(val), "value": int(cnt)})

    for val, cnt in df["phase_inferred"].fillna("Unknown").value_counts().items():
        rows.append({"metric": "by_phase_inferred", "category": str(val), "value": int(cnt)})

    for flag in OUTCOME_PATTERNS.keys():
        rows.append({
            "metric": "outcome_flag",
            "category": flag,
            "value": int(df[flag].fillna(False).sum())
        })

    if "intent" in df.columns:
        for val, cnt in df["intent"].fillna("Unknown").value_counts().head(20).items():
            rows.append({"metric": "by_intent", "category": str(val), "value": int(cnt)})

    return pd.DataFrame(rows)


def main():
    all_parts = []
    summary = []

    for file in INPUT_FILES:
        part, meta = extract_candidates(file)
        summary.append(meta)
        if not part.empty:
            all_parts.append(part)

    if all_parts:
        final = pd.concat(all_parts, ignore_index=True)
        final = final.drop_duplicates(subset=["incident_id", "narrative"])
    else:
        final = pd.DataFrame(columns=[
            "source_file", "incident_id", "source", "intent", "phase",
            "scenario_type", "phase_inferred", "narrative",
        ])

    stats = build_stats(final) if not final.empty else pd.DataFrame(columns=["metric", "category", "value"])
    summary_df = pd.DataFrame(summary)

    final.to_csv(INCIDENTS_CSV, index=False)
    stats.to_csv(STATS_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print(f"Saved incidents : {INCIDENTS_CSV} ({len(final)} rows)")
    print(f"Saved stats     : {STATS_CSV} ({len(stats)} rows)")
    print(f"Saved summary   : {SUMMARY_CSV} ({len(summary_df)} rows)")


if __name__ == "__main__":
    main()
