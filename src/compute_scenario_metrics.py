#!/usr/bin/env python3
"""
compute_scenario_metrics.py
---------------------------
Aggregate scenario-level metrics for UATIE-EDM.
Assumes scenario-specific CSVs already exist (rule-based and semantic).

Inputs (expected in output/):
- late_runway_change_approach_incidents.csv
- late_runway_change_approach_stats.csv
- late_runway_change_hazard_summary.csv
- semantic_late_runway_change_incidents_enriched.csv
- semantic_late_runway_change_stats.csv
- semantic_late_runway_change_hazard_summary.csv

Outputs:
- metrics/metrics_scenarios.csv
"""

from pathlib import Path
import pandas as pd

OUT_DIR = Path("output")
METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)
OUT_FILE = METRICS_DIR / "metrics_scenarios.csv"

RULE_INCIDENTS = OUT_DIR / "late_runway_change_approach_incidents.csv"
RULE_STATS = OUT_DIR / "late_runway_change_approach_stats.csv"
RULE_HAZARDS = OUT_DIR / "late_runway_change_hazard_summary.csv"

SEM_INCIDENTS = OUT_DIR / "semantic_late_runway_change_incidents_enriched.csv"
SEM_STATS = OUT_DIR / "semantic_late_runway_change_stats.csv"
SEM_HAZARDS = OUT_DIR / "semantic_late_runway_change_hazard_summary.csv"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def main():
    rows = []

    rule_inc = load_csv(RULE_INCIDENTS)
    sem_inc = load_csv(SEM_INCIDENTS)

    n_rule = len(rule_inc)
    n_sem = len(sem_inc)

    if not rule_inc.empty and not sem_inc.empty:
        common_cols = [c for c in ["incident_id", "report_id", "narrative", "summary", "text"] if c in rule_inc.columns and c in sem_inc.columns]
        if common_cols:
            merged = rule_inc.merge(sem_inc, on=common_cols, how="inner")
            n_intersect = len(merged)
        else:
            n_intersect = 0
    else:
        n_intersect = 0

    rows.append({
        "scenario": "late_runway_change_approach",
        "metric": "rule_based_incidents",
        "category": "count",
        "value": int(n_rule),
    })

    rows.append({
        "scenario": "late_runway_change_approach",
        "metric": "semantic_incidents",
        "category": "count",
        "value": int(n_sem),
    })

    rows.append({
        "scenario": "late_runway_change_approach",
        "metric": "intersection_incidents",
        "category": "count",
        "value": int(n_intersect),
    })

    if n_rule:
        rows.append({
            "scenario": "late_runway_change_approach",
            "metric": "semantic_coverage_vs_rule",
            "category": "pct",
            "value": float(round((n_sem / n_rule * 100), 2)),
        })
    else:
        rows.append({
            "scenario": "late_runway_change_approach",
            "metric": "semantic_coverage_vs_rule",
            "category": "pct",
            "value": 0.0,
        })

    rule_stats = load_csv(RULE_STATS)
    for _, r in rule_stats.iterrows():
        rows.append({
            "scenario": "late_runway_change_approach",
            "metric": f"rule_{r['metric']}",
            "category": str(r['category']),
            "value": float(r['value']),
        })

    rule_haz = load_csv(RULE_HAZARDS)
    for _, r in rule_haz.iterrows():
        rows.append({
            "scenario": "late_runway_change_approach",
            "metric": "rule_hazard",
            "category": str(r['hazard']),
            "value": float(r['incident_count']),
        })

    sem_stats = load_csv(SEM_STATS)
    for _, r in sem_stats.iterrows():
        rows.append({
            "scenario": "late_runway_change_approach",
            "metric": f"sem_{r['metric']}",
            "category": str(r['category']),
            "value": float(r['value']),
        })

    sem_haz = load_csv(SEM_HAZARDS)
    for _, r in sem_haz.iterrows():
        rows.append({
            "scenario": "late_runway_change_approach",
            "metric": "sem_hazard",
            "category": str(r['hazard']),
            "value": float(r['incident_count']),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_FILE, index=False)
    print(f"Saved scenario metrics to {OUT_FILE} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
