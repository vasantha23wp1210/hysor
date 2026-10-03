#!/usr/bin/env python3
"""
fused_corpus_V4.py — UATIE-EDM Phase 1
======================================
Fusion of ASRS + fused NTSB/Airline crashes into a unified corpus,
with safe date normalization for partially specified NTSB dates.

Key fix:
- Never parse Event Year directly as a datetime/timestamp.
- Build NTSB datetime from Event Year / Event Month / Event Day explicitly.
- Preserve partial date precision using defaults only when needed.
- Handle NA values in year column safely without crashing.
"""

import os
import re
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")
NTSB_FILE     = os.path.join(PROCESSED_DIR, "fused_ntsb_airline_crashes.csv")
ASRS_FILE     = os.path.join(PROCESSED_DIR, "asrs_for_fusion.csv")
OUTPUT_FILE   = os.path.join(PROCESSED_DIR, "fused_corpus_V4.csv")
REPORT_FILE   = os.path.join(PROCESSED_DIR, "fusion_report_V4.txt")

os.makedirs(PROCESSED_DIR, exist_ok=True)

SCHEMA = [
    "record_id", "source", "event_id", "datetime", "location",
    "operator", "phase_of_flight", "primary_problem", "summary", "severity"
]


def clean_text(text):
    if pd.isna(text) or str(text).strip() in ("", "nan", "None"):
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:2000]


def clean_str_series(series, index):
    if series is None:
        return pd.Series("", index=index, dtype="object")
    return series.fillna("").astype(str).replace({"nan": "", "None": ""})


def pick_first_available(df, candidates, default=""):
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series(default, index=df.index)


def safe_int_series(series, index):
    if series is None:
        return pd.Series([pd.NA] * len(index), index=index, dtype="Int64")
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def build_ntsb_datetime(df):
    """
    Build a datetime string column from NTSB year/month/day columns.
    Handles NA values safely by only calling pd.to_datetime on rows
    where year is not null.
    """
    year  = safe_int_series(
        pick_first_available(df, ["Event Year",  "year"]),  df.index
    )
    month = safe_int_series(
        pick_first_available(df, ["Event Month", "month"]), df.index
    )
    day   = safe_int_series(
        pick_first_available(df, ["Event Day",   "day"]),   df.index
    )

    # ── determine precision BEFORE filling defaults ───────────────
    precision = pd.Series("unknown", index=df.index, dtype="object")
    has_year  = year.notna()
    has_month = has_year  & month.notna()
    has_day   = has_month & day.notna()

    precision[has_year  & ~has_month] = "year"
    precision[has_month & ~has_day]   = "month"
    precision[has_day]                = "day"

    # ── fill month / day defaults only for datetime assembly ─────
    month_filled = month.fillna(1)
    day_filled   = day.fillna(1)

    # ── clip to valid ranges to avoid out-of-range day errors ────
    month_filled = month_filled.clip(1, 12)
    day_filled   = day_filled.clip(1, 28)   # safe cap avoids Feb 29 issues

    # ── build datetime only for rows where year is not NA ────────
    # This avoids the "cannot convert NA to integer" crash in pandas
    result = pd.Series("", index=df.index, dtype="object")
    valid  = has_year

    if valid.sum() > 0:
        sub = pd.DataFrame({
            "year":  year[valid].astype("int64"),
            "month": month_filled[valid].astype("int64"),
            "day":   day_filled[valid].astype("int64"),
        }, index=df.index[valid])

        parsed = pd.to_datetime(sub, errors="coerce")

        # format output based on precision
        year_only  = valid & (precision == "year")
        month_only = valid & (precision == "month")
        day_level  = valid & (precision == "day")

        if year_only.sum() > 0:
            result[year_only] = (
                year[year_only]
                .astype("int64")
                .astype(str)
            )

        if month_only.sum() > 0:
            result[month_only] = (
                parsed[month_only]
                .dt.strftime("%Y-%m")
                .fillna("")
            )

        if day_level.sum() > 0:
            result[day_level] = (
                parsed[day_level]
                .dt.strftime("%Y-%m-%d")
                .fillna("")
            )

        # fallback for valid year rows that didn't match any precision bucket
        fallback = valid & (result == "") & has_year
        if fallback.sum() > 0:
            result[fallback] = year[fallback].astype("int64").astype(str)

    return result, precision


def ntsb_severity(row):
    try:
        fatal   = int(float(row.get("Total Fatal Injuries",   0) or 0))
        serious = int(float(row.get("Total Serious Injuries", 0) or 0))
        if fatal   > 0: return "Fatal"
        if serious > 0: return "Serious"
        return "Minor"
    except Exception:
        return "Unknown"


# ══════════════════════════════════════════════════════════════════
# NTSB
# ══════════════════════════════════════════════════════════════════

print("Loading NTSB dataset...")
ntsb = pd.read_csv(NTSB_FILE, low_memory=False)
print(f"  NTSB raw: {ntsb.shape[0]} rows x {ntsb.shape[1]} cols")

analysis = clean_str_series(
    pick_first_available(ntsb, ["Analysis"]), ntsb.index
)
summary2 = clean_str_series(
    pick_first_available(ntsb, ["summary"]), ntsb.index
)
ntsb["_narrative"] = (analysis + " " + summary2).str.strip()

ntsb_datetime, ntsb_date_precision = build_ntsb_datetime(ntsb)

location_series = clean_str_series(
    pick_first_available(ntsb, ["location", "City", "Address", "Place"]),
    ntsb.index,
)
operator_series = clean_str_series(
    pick_first_available(ntsb, ["operator", "airline"]),
    ntsb.index,
)
phase_series = clean_str_series(
    pick_first_available(ntsb, ["Broad Phase Of Flight", "phase_of_flight"]),
    ntsb.index,
)
problem_series = clean_str_series(
    pick_first_available(ntsb, ["type", "primary_problem", "Aircraft Damage"]),
    ntsb.index,
)
event_id_series = clean_str_series(
    pick_first_available(ntsb, ["Event Id", "event_id"]),
    ntsb.index,
)

ntsb_norm = pd.DataFrame({
    "record_id":       [f"NTSB_{i:06d}" for i in range(len(ntsb))],
    "source":          "NTSB",
    "event_id":        event_id_series,
    "datetime":        ntsb_datetime,
    "location":        location_series,
    "operator":        operator_series,
    "phase_of_flight": phase_series,
    "primary_problem": problem_series,
    "summary":         ntsb["_narrative"].apply(clean_text),
    "severity":        ntsb.apply(ntsb_severity, axis=1),
})

ntsb_norm = ntsb_norm[
    ntsb_norm["summary"].str.len() > 20
].reset_index(drop=True)

print(f"  NTSB normalized: {len(ntsb_norm)} rows with usable text")

# ── print NTSB date precision summary ────────────────────────────
prec_counts = ntsb_date_precision.value_counts(dropna=False)
print(f"  NTSB date precision: "
      + "  |  ".join(f"{k}={v:,}" for k, v in prec_counts.items()))


# ══════════════════════════════════════════════════════════════════
# ASRS
# ══════════════════════════════════════════════════════════════════

print("\nLoading ASRS dataset...")
asrs = pd.read_csv(ASRS_FILE, low_memory=False)
print(f"  ASRS raw: {asrs.shape[0]} rows x {asrs.shape[1]} cols")

asrs_norm = pd.DataFrame({
    "record_id": [f"ASRS_{i:07d}" for i in range(len(asrs))],
    "source":    "ASRS",
    "event_id":       clean_str_series(
        pick_first_available(asrs, ["event_id"]),        asrs.index),
    "datetime":       clean_str_series(
        pick_first_available(asrs, ["datetime"]),        asrs.index),
    "location":       clean_str_series(
        pick_first_available(asrs, ["location"]),        asrs.index),
    "operator":       clean_str_series(
        pick_first_available(asrs, ["operator"]),        asrs.index),
    "phase_of_flight":clean_str_series(
        pick_first_available(asrs, ["phase_of_flight"]), asrs.index),
    "primary_problem":clean_str_series(
        pick_first_available(asrs, ["primary_problem"]), asrs.index),
    "summary":        clean_str_series(
        pick_first_available(asrs, ["summary"]),         asrs.index
    ).apply(clean_text),
    "severity":       "Occurrence",
})

asrs_norm = asrs_norm[
    asrs_norm["summary"].str.len() > 20
].reset_index(drop=True)

print(f"  ASRS normalized: {len(asrs_norm)} rows with usable text")


# ══════════════════════════════════════════════════════════════════
# FUSE
# ══════════════════════════════════════════════════════════════════

print("\nFusing datasets...")
fused = pd.concat([ntsb_norm, asrs_norm], ignore_index=True)
fused = fused[SCHEMA]

# blank suspicious epoch dates from NTSB
bad_epoch_mask = (
    (fused["source"] == "NTSB") &
    fused["datetime"].astype(str).str.startswith("1970-01-01")
)
if bad_epoch_mask.any():
    print(f"  WARNING: {bad_epoch_mask.sum()} suspicious epoch-like "
          f"NTSB dates blanked")
    fused.loc[bad_epoch_mask, "datetime"] = ""

total      = len(fused)
asrs_count = (fused["source"] == "ASRS").sum()
ntsb_count = (fused["source"] == "NTSB").sum()
avg_len    = fused["summary"].str.len().mean()
phase_dist = fused["phase_of_flight"].value_counts().head(8)
sev_dist   = fused["severity"].value_counts()

fused.to_csv(OUTPUT_FILE, index=False)
print(f"\n  Saved: {OUTPUT_FILE}")


# ══════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════

report_lines = [
    "UATIE-EDM FUSION AUDIT REPORT V4",
    "=" * 50,
    f"Output file    : {OUTPUT_FILE}",
    f"Total records  : {total:,}",
    f"ASRS           : {asrs_count:,}",
    f"NTSB           : {ntsb_count:,}",
    f"Schema columns : {SCHEMA}",
    f"Avg summary len: {avg_len:.0f} chars",
    "",
    "NTSB datetime precision distribution:",
]
for k, v in ntsb_date_precision.value_counts(dropna=False).items():
    report_lines.append(f"  {str(k):<20}: {v:,}")

report_lines += ["", "Phase of Flight Distribution (top 8):"]
for phase, cnt in phase_dist.items():
    report_lines.append(f"  {str(phase):<35}: {cnt:,}")

report_lines += ["", "Severity Distribution:"]
for sev, cnt in sev_dist.items():
    report_lines.append(f"  {str(sev):<35}: {cnt:,}")

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"  Saved audit report: {REPORT_FILE}")

print("\n" + "=" * 55)
print("  FUSION COMPLETE")
print("=" * 55)
print(f"  fused_corpus_V4.csv  →  {total:,} records")
print(f"  ASRS: {asrs_count:,}  |  NTSB: {ntsb_count:,}")
print(f"  Avg summary length   :  {avg_len:.0f} chars")
print("  Ready for retrieval / index build")
print("=" * 55)