# save as: evaluation/make_pool_simple.py
"""
Simple pooled candidate builder.
Combines top 5 hybrid + top 5 BM25 per query.
"""

import pandas as pd
from pathlib import Path

EVAL_DIR = Path("evaluation")

# ── load files ────────────────────────────────────────────────────
print("Loading files...")

gold_q = pd.read_csv(EVAL_DIR / "gold_queries.csv")

try:
    hybrid = pd.read_csv(
        EVAL_DIR / "retrieval_results_enriched.csv",
        low_memory=False, encoding="latin-1"
    )
except Exception:
    hybrid = pd.read_csv(
        EVAL_DIR / "retrieval_results.csv",
        low_memory=False, encoding="latin-1"
    )

bm25   = pd.read_csv(EVAL_DIR / "bm25_results.csv")
exist  = pd.read_csv(EVAL_DIR / "gold_relevance.csv")

print(f"  Queries      : {len(gold_q)}")
print(f"  Hybrid rows  : {len(hybrid):,}")
print(f"  BM25 rows    : {len(bm25):,}")
print(f"  Labeled pairs: {len(exist):,}")

# ── find labeled query IDs ────────────────────────────────────────
labeled_qids = set(exist["query_id"].astype(str).unique())
print(f"  Labeled qids : {len(labeled_qids)}")

# ── existing label lookup ─────────────────────────────────────────
label_map = {
    (str(r["query_id"]), str(r["incident_id"])): int(r["relevance"])
    for _, r in exist.iterrows()
    if str(r["relevance"]).strip() not in ("", "nan")
}

# ── pick best narrative column ────────────────────────────────────
nar_col = None
for c in ["narrative_preview", "narrative", "summary", "text"]:
    if c in hybrid.columns:
        nar_col = c
        break

score_col = "score" if "score" in hybrid.columns else None
bm25_score_col = "score" if "score" in bm25.columns else None

rows = []

for _, q in gold_q.iterrows():
    qid   = str(q["query_id"])
    qtext = str(q.get("query_text", ""))

    # only process queries with existing labels
    if qid not in labeled_qids:
        continue

    # ── top 5 hybrid ─────────────────────────────────────────────
    h_sub = (
        hybrid[hybrid["query_id"].astype(str) == qid]
        .sort_values("rank")
        .head(5)
    )
    for _, r in h_sub.iterrows():
        inc_id = str(r["incident_id"])
        nar    = str(r[nar_col])[:350] if nar_col else inc_id
        score  = round(float(r[score_col]), 4) if score_col else ""
        key    = (qid, inc_id)
        rows.append({
            "query_id":         qid,
            "query_text":       qtext,
            "system":           "hybrid",
            "rank":             int(r["rank"]),
            "incident_id":      inc_id,
            "score":            score,
            "narrative_preview":nar,
            "relevance":        label_map.get(key, ""),
        })

    # ── top 5 BM25 ───────────────────────────────────────────────
    b_sub = (
        bm25[bm25["query_id"].astype(str) == qid]
        .sort_values("rank")
        .head(5)
    )
    for _, r in b_sub.iterrows():
        inc_id = str(r["incident_id"])
        score  = round(float(r[bm25_score_col]), 4) \
                 if bm25_score_col else ""

        # try to get narrative from hybrid enriched file
        h_match = hybrid[
            hybrid["incident_id"].astype(str) == inc_id
        ]
        if not h_match.empty and nar_col:
            nar = str(h_match.iloc[0][nar_col])[:350]
        else:
            # extract from pseudo-ID if available
            parts = inc_id.split("_", 2)
            nar   = parts[2].replace("_", " ")[:350] \
                    if len(parts) == 3 else inc_id

        key = (qid, inc_id)
        rows.append({
            "query_id":         qid,
            "query_text":       qtext,
            "system":           "bm25",
            "rank":             int(r["rank"]),
            "incident_id":      inc_id,
            "score":            score,
            "narrative_preview":nar,
            "relevance":        label_map.get(key, ""),
        })

# ── build dataframe ───────────────────────────────────────────────
pool = pd.DataFrame(rows)

# deduplicate keeping first occurrence
pool = pool.drop_duplicates(subset=["query_id","incident_id"])

# split into already labeled and needs labeling
pool["needs_label"] = (
    pool["relevance"].astype(str).str.strip() == ""
)

needs  = pool[pool["needs_label"]].drop(columns=["needs_label"])
filled = pool[~pool["needs_label"]].drop(columns=["needs_label"])

# ── save ──────────────────────────────────────────────────────────
pool.drop(columns=["needs_label"], errors="ignore").to_csv(
    EVAL_DIR / "pooled_candidates.csv", index=False
)
needs.to_csv(
    EVAL_DIR / "pooled_labeling_template.csv", index=False
)

print(f"\nPooled candidate set summary:")
print(f"  Total candidates      : {len(pool):,}")
print(f"  Queries covered       : {pool['query_id'].nunique()}")
print(f"  Already labeled       : {len(filled):,}")
print(f"  Need new label        : {len(needs):,}")
print(f"  Hybrid candidates     : {(pool['system']=='hybrid').sum():,}")
print(f"  BM25 candidates       : {(pool['system']=='bm25').sum():,}")
print(f"\nSaved:")
print(f"  evaluation/pooled_candidates.csv     ({len(pool)} rows)")
print(f"  evaluation/pooled_labeling_template.csv ({len(needs)} rows)")
print(f"\nOpen pooled_labeling_template.csv in Excel.")
print(f"Fill the 'relevance' column: 2=high 1=partial 0=none")
print(f"Save and run merge_pooled_labels.py to update gold_relevance.csv")