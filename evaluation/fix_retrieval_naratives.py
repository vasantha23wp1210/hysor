#!/usr/bin/env python3
"""
fix_retrieval_narratives.py
---------------------------
Rebuilds retrieval_results_enriched.csv with FULL narrative text
by joining asrs_nlp_annotated.csv positionally via FAISS rank index.

Run once after build_retrieval_results.py.
"""

from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR     = PROJECT_ROOT / "evaluation"
PROCESSED    = PROJECT_ROOT / "data" / "processed"

RESULTS_CSV  = EVAL_DIR / "retrieval_results.csv"
ANNOTATED    = PROCESSED / "asrs_nlp_annotated.csv"
META_CSV     = PROCESSED / "asrs_faiss_meta.csv"
OUT_CSV      = EVAL_DIR  / "retrieval_results_enriched.csv"

# How many characters to show in preview column
PREVIEW_LEN  = 400


def pick_text_col(df: pd.DataFrame) -> str:
    for c in ["combined_text", "narrative", "summary",
              "synopsis", "text", "report"]:
        if c in df.columns:
            return c
    raise ValueError(f"No usable text column found. Columns: {df.columns.tolist()}")


def pick_id_col(df: pd.DataFrame) -> str | None:
    for c in ["incident_id", "asrs_id", "id", "report_id"]:
        if c in df.columns:
            return c
    return None


def extract_faiss_rank(pseudo_id: str) -> int | None:
    """
    Parse FAISS rank from pseudo-ID like Q001_R7_some text...
    Returns 0-based index (rank - 1).
    """
    m = re.match(r"^Q\d+_R(\d+)_", str(pseudo_id))
    if m:
        return int(m.group(1)) - 1   # convert 1-based rank to 0-based
    return None


def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Not found: {RESULTS_CSV}")
    if not ANNOTATED.exists():
        raise FileNotFoundError(f"Not found: {ANNOTATED}")

    print(f"Loading retrieval results from {RESULTS_CSV}...")
    results = pd.read_csv(RESULTS_CSV)
    print(f"  Rows: {len(results):,}")

    print(f"\nLoading annotated corpus from {ANNOTATED}...")
    corpus  = pd.read_csv(ANNOTATED, low_memory=False)
    print(f"  Rows: {len(corpus):,}")

    text_col = pick_text_col(corpus)
    id_col   = pick_id_col(corpus)
    print(f"  Text column : {text_col}")
    print(f"  ID   column : {id_col}")

    # ── Build lookup: real incident_id → full text ────────────────
    if id_col:
        id_to_text = dict(
            zip(corpus[id_col].astype(str), corpus[text_col].astype(str))
        )
    else:
        id_to_text = {}

    # ── Enrich each row ───────────────────────────────────────────
    narratives      = []
    real_ids        = []
    matched_real    = 0
    matched_positional = 0
    unmatched       = 0

    for _, row in results.iterrows():
        pseudo_id = str(row["incident_id"])
        narrative = ""
        real_id   = pseudo_id

        # Strategy 1: try real ID match
        if id_to_text and pseudo_id in id_to_text:
            narrative     = id_to_text[pseudo_id]
            matched_real += 1

        # Strategy 2: positional match via FAISS rank in pseudo-ID
        if not narrative:
            faiss_idx = extract_faiss_rank(pseudo_id)
            if faiss_idx is not None and 0 <= faiss_idx < len(corpus):
                narrative            = str(corpus.iloc[faiss_idx][text_col])
                if id_col:
                    real_id          = str(corpus.iloc[faiss_idx][id_col])
                matched_positional  += 1

        # Strategy 3: per-query rank fallback
        # Each query's rank-N result maps to corpus row (rank-1)
        if not narrative:
            rank = int(row.get("rank", 1)) - 1
            if 0 <= rank < len(corpus):
                narrative = str(corpus.iloc[rank][text_col])
                unmatched += 1

        narratives.append(narrative.replace("\n", " ").strip())
        real_ids.append(real_id)

    print(f"\nMatching summary:")
    print(f"  Real ID match       : {matched_real:,}")
    print(f"  Positional match    : {matched_positional:,}")
    print(f"  Rank fallback       : {unmatched:,}")

    # ── Build enriched dataframe ──────────────────────────────────
    enriched = results.copy()
    enriched["real_incident_id"] = real_ids
    enriched["narrative"]        = narratives
    enriched["narrative_preview"] = (
        enriched["narrative"].str[:PREVIEW_LEN]
    )

    # reorder columns
    front = [
        "query_id", "rank", "incident_id",
        "real_incident_id", "score",
        "narrative_preview", "narrative"
    ]
    rest  = [c for c in enriched.columns if c not in front]
    enriched = enriched[front + rest]

    enriched.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}  ({len(enriched):,} rows)")

    # ── Sanity check ──────────────────────────────────────────────
    filled = (enriched["narrative"].str.strip() != "").sum()
    print(f"Rows with non-empty narrative : {filled:,} / {len(enriched):,}")

    print("\nSample narratives (first 3 rows):")
    for _, row in enriched.head(3).iterrows():
        print(
            f"\n  [{row['query_id']} rank={row['rank']}]\n"
            f"  {row['narrative_preview'][:200]}"
        )


if __name__ == "__main__":
    main()