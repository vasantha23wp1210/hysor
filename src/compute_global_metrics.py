#!/usr/bin/env python3
"""
compute_global_metrics.py
-------------------------
Compute corpus-level metrics for UATIE-EDM from asrs_nlp_annotated.csv.

Outputs:
- metrics/metrics_global.csv
"""

from pathlib import Path
import pandas as pd


PROCESSED_DIR = Path("data/processed")
ANNOTATED_CSV = PROCESSED_DIR / "asrs_nlp_annotated.csv"

OUT_DIR = Path("metrics")
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / "metrics_global.csv"


def first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def is_present_value(x):
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    if s in {"", "0", "false", "none", "nan", "unknown", "unclassified"}:
        return 0
    return 1


def main():
    if not ANNOTATED_CSV.exists():
        raise FileNotFoundError(f"Annotated CSV not found: {ANNOTATED_CSV}")

    df = pd.read_csv(ANNOTATED_CSV, low_memory=False)
    total = len(df)

    # ------------------------------------------------------------
    # Intent coverage
    # ------------------------------------------------------------
    intent_col = first_existing_column(df, ["intent", "intent_final", "intent_label"])

    if intent_col is None:
        intent_covered = 0
    else:
        s = df[intent_col].astype(str).str.strip().str.lower()
        intent_covered = (~s.isin(["", "unknown", "nan", "none", "unclassified"])).sum()

    # ------------------------------------------------------------
    # Phase coverage
    # ------------------------------------------------------------
    phase_col = first_existing_column(df, ["phase_final", "phase", "phase_of_flight"])

    if phase_col is None:
        phase_covered = 0
    else:
        s = df[phase_col].astype(str).str.strip().str.lower()
        phase_covered = (~s.isin(["", "unknown", "nan", "none"])).sum()

    # ------------------------------------------------------------
    # Hazard tagging density
    # Handles both:
    # - numeric/boolean hazard columns
    # - string-valued hazard columns like H_Airspace
    # ------------------------------------------------------------
    hazard_cols = [c for c in df.columns if c.lower().startswith("hazard")]

    if hazard_cols:
        hazard_df = df[hazard_cols].copy()

        # apply Series.map per column for compatibility
        hazard_presence = hazard_df.apply(lambda col: col.map(is_present_value))
        hazard_counts = hazard_presence.sum(axis=1)

        avg_hazards = float(hazard_counts.mean()) if total else 0.0
        incidents_with_any_hazard = int((hazard_counts > 0).sum())
        hazard_coverage_pct = float(round((incidents_with_any_hazard / total * 100), 2)) if total else 0.0
    else:
        avg_hazards = 0.0
        incidents_with_any_hazard = 0
        hazard_coverage_pct = 0.0

    # ------------------------------------------------------------
    # Output rows
    # ------------------------------------------------------------
    rows = [
        {
            "metric": "total_incidents",
            "category": "all",
            "value": int(total),
        },
        {
            "metric": "intent_coverage_count",
            "category": "all",
            "value": int(intent_covered),
        },
        {
            "metric": "intent_coverage_pct",
            "category": "all",
            "value": float(round((intent_covered / total * 100), 2)) if total else 0.0,
        },
        {
            "metric": "phase_coverage_count",
            "category": "all",
            "value": int(phase_covered),
        },
        {
            "metric": "phase_coverage_pct",
            "category": "all",
            "value": float(round((phase_covered / total * 100), 2)) if total else 0.0,
        },
        {
            "metric": "incidents_with_any_hazard_count",
            "category": "all",
            "value": int(incidents_with_any_hazard),
        },
        {
            "metric": "hazard_coverage_pct",
            "category": "all",
            "value": float(hazard_coverage_pct),
        },
        {
            "metric": "avg_hazards_per_incident",
            "category": "all",
            "value": float(round(avg_hazards, 3)),
        },
    ]

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_FILE, index=False)

    print(f"Saved global metrics to {OUT_FILE} ({len(out_df)} rows)")
    print(out_df)


if __name__ == "__main__":
    main()