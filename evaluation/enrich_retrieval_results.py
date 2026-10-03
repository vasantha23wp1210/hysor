#!/usr/bin/env python3
"""
enrich_retrieval_results.py
---------------------------
Joins full narrative text from FAISS metadata into retrieval_results.csv
so that gold labeling can be done directly in the CSV.

Input:
- evaluation/retrieval_results.csv
- data/processed/asrs_faiss_meta.csv

Output:
- evaluation/retrieval_results_enriched.csv  (with narrative column)
"""

from pathlib import Path
import pandas as pd
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR     = PROJECT_ROOT / "evaluation"
PROCESSED    = PROJECT_ROOT / "data" / "processed"

RESULTS_CSV  = EVAL_DIR / "retrieval_results.csv"
META_CSV     = PROCESSED / "asrs_faiss_meta.csv"
ANNOTATED    = PROCESSED / "asrs_nlp_annotated.csv"
OUT_CSV      = EVAL_DIR  / "retrieval_results_enriched.csv"

# columns to pull into the enriched file
KEEP_META_COLS = [
    "summary", "narrative", "combined_text",
    "text", "synopsis", "report",
    "intent", "phase_final", "phase",
    "hazard_tags", "source", "incident_id",
]


def pick_text_col(df: pd.DataFrame) -> str | None:
    for c in ["summary", "narrative", "combined_text",
              "text", "synopsis", "report"]:
        if c in df.columns:
            return c
    return None


def extract_pseudo_text(pseudo_id: str) -> str:
    """
    Recover the text fragment embedded in a pseudo-ID like:
    Q001_R1_air carrier flight crew reported a late runway
    """
    parts = pseudo_id.split("_", 2)
    if len(parts) == 3:
        return parts[2].replace("_", " ").strip()
    return pseudo_id


def is_pseudo_id(incident_id: str) -> bool:
    return bool(re.match(r"^Q\d+_R\d+_", incident_id))


def load_meta() -> pd.DataFrame | None:
    for path in [META_CSV, ANNOTATED]:
        if path.exists():
            print(f"Loading metadata from: {path}")
            df = pd.read_csv(path, low_memory=False)
            print(f"  Rows: {len(df):,}  Columns: {df.columns.tolist()[:8]}...")
            return df
    return None


def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"retrieval_results.csv not found at {RESULTS_CSV}")

    results = pd.read_csv(RESULTS_CSV)
    print(f"Loaded retrieval results: {len(results):,} rows")
    print(f"Sample incident_ids:\n{results['incident_id'].head(5).tolist()}")

    meta = load_meta()

    # ── Strategy 1: real incident IDs → join on meta ──────────────
    if meta is not None:
        text_col = pick_text_col(meta)
        id_col   = None
        for c in ["incident_id", "asrs_id", "id", "report_id"]:
            if c in meta.columns:
                id_col = c
                break

        if id_col and text_col:
            print(f"Joining on meta['{id_col}'] → text from meta['{text_col}']")
            meta_slim = meta[[id_col, text_col]].copy()
            meta_slim = meta_slim.rename(
                columns={id_col: "incident_id_meta", text_col: "narrative"}
            )
            meta_slim["incident_id_meta"] = meta_slim["incident_id_meta"].astype(str)

            results["incident_id_str"] = results["incident_id"].astype(str)

            merged = results.merge(
                meta_slim,
                left_on="incident_id_str",
                right_on="incident_id_meta",
                how="left",
            )

            matched = merged["narrative"].notna().sum()
            print(f"  Matched narratives: {matched:,} / {len(merged):,}")
        else:
            merged = results.copy()
            merged["narrative"] = ""
    else:
        merged = results.copy()
        merged["narrative"] = ""

    # ── Strategy 2: pseudo-IDs → recover text from ID itself ──────
    pseudo_mask = merged["incident_id"].astype(str).apply(is_pseudo_id)
    no_text_mask = merged["narrative"].isna() | (merged["narrative"].astype(str).str.strip() == "")
    fill_mask = pseudo_mask & no_text_mask

    if fill_mask.sum() > 0:
        print(f"Recovering text from {fill_mask.sum():,} pseudo-IDs...")
        merged.loc[fill_mask, "narrative"] = (
            merged.loc[fill_mask, "incident_id"]
            .astype(str)
            .apply(extract_pseudo_text)
        )

    # ── Strategy 3: fallback positional join on FAISS index ───────
    still_empty = merged["narrative"].isna() | (merged["narrative"].astype(str).str.strip() == "")
    if still_empty.sum() > 0 and meta is not None:
        text_col = pick_text_col(meta)
        if text_col:
            print(f"Attempting positional fallback for {still_empty.sum():,} rows...")
            def get_by_rank(row):
                rank = int(row.get("rank", 1)) - 1
                if 0 <= rank < len(meta):
                    return str(meta.iloc[rank][text_col])
                return ""
            merged.loc[still_empty, "narrative"] = (
                merged[still_empty].apply(get_by_rank, axis=1)
            )

    # ── Clean up and write ─────────────────────────────────────────
    if "incident_id_str" in merged.columns:
        merged = merged.drop(columns=["incident_id_str"])
    if "incident_id_meta" in merged.columns:
        merged = merged.drop(columns=["incident_id_meta"])

    # truncate narrative for readability in CSV
    merged["narrative"] = (
        merged["narrative"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    merged["narrative_preview"] = merged["narrative"].str[:300]

    # reorder columns
    front = ["query_id", "rank", "incident_id", "score", "narrative_preview", "narrative"]
    rest  = [c for c in merged.columns if c not in front]
    merged = merged[front + rest]

    merged.to_csv(OUT_CSV, index=False)
    print(f"\nSaved enriched results to {OUT_CSV} ({len(merged):,} rows)")

    filled = (merged["narrative"].str.strip() != "").sum()
    print(f"Rows with narrative text: {filled:,} / {len(merged):,}")

    print("\nSample (first 3 rows narrative_preview):")
    for _, row in merged.head(3).iterrows():
        print(f"  [{row['query_id']} rank={row['rank']}] {row['narrative_preview'][:120]}")


if __name__ == "__main__":
    main()