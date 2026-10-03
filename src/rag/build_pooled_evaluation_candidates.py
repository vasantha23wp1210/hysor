# save as: evaluation/build_pooled_candidates.py
"""
build_pooled_candidates.py
--------------------------
Creates a pooled candidate set combining hybrid + BM25 results
for fair comparative evaluation.

Output:
- evaluation/pooled_candidates.csv  (all candidates to label)
- evaluation/pooled_labeling_template.csv  (for manual labeling)
"""

import pandas as pd
from pathlib import Path

EVAL_DIR = Path("evaluation")

gold_q   = pd.read_csv(EVAL_DIR / "gold_queries.csv")
hybrid   = pd.read_csv(
    EVAL_DIR / "retrieval_results_enriched.csv",
    low_memory=False,
    encoding="latin-1"
)
bm25     = pd.read_csv(EVAL_DIR / "bm25_results.csv")
existing = pd.read_csv(EVAL_DIR / "gold_relevance.csv")

print(f"Hybrid results : {len(hybrid):,}")
print(f"BM25 results   : {len(bm25):,}")
print(f"Existing labels: {len(existing):,}")

# ── build pooled candidate set per query ──────────────────────────
# take top 5 from hybrid + top 5 from BM25 per query = 10 per query
TOP = 5
rows = []

labeled_qids = set(existing["query_id"].astype(str).unique())

for _, q in gold_q.iterrows():
    qid   = str(q["query_id"])
    qtext = str(q["query_text"])

    # only pool for queries that already have some labels
    if qid not in labeled_qids:
        continue

    # hybrid top 5
    h_sub = (
        hybrid[hybrid["query_id"].astype(str) == qid]
        .sort_values("rank")
        .head(TOP)
    )

    # bm25 top 5
    b_sub = (
        bm25[bm25["query_id"].astype(str) == qid]
        .sort_values("rank")
        .head(TOP)
    )

    for _, r in h_sub.iterrows():
        inc_id   = str(r["incident_id"])
        narrative = str(r.get("narrative_preview",
                              r.get("narrative", "")))[:300]
        rows.append({
            "query_id":         qid,
            "query_text":       qtext,
            "system":           "hybrid",
            "rank":             int(r["rank"]),
            "incident_id":      inc_id,
            "score":            r.get("score", ""),
            "narrative_preview":narrative,
            "relevance":        "",
        })

    for _, r in b_sub.iterrows():
        inc_id   = str(r["incident_id"])
        # get narrative from hybrid results if available
        h_match = hybrid[
            hybrid["incident_id"].astype(str) == inc_id
        ]
        if not h_match.empty:
            narrative = str(h_match.iloc[0].get(
                "narrative_preview",
                h_match.iloc[0].get("narrative", "")
            ))[:300]
        else:
            narrative = inc_id
        rows.append({
            "query_id":         qid,
            "query_text":       qtext,
            "system":           "bm25",
            "rank":             int(r["rank"]),
            "incident_id":      inc_id,
            "score":            r.get("score", ""),
            "narrative_preview":narrative,
            "relevance":        "",
        })

pool_df = pd.DataFrame(rows)

# deduplicate by query_id + incident_id
# keep both system labels for deduped rows
pool_df = pool_df.drop_duplicates(
    subset=["query_id", "incident_id"]
)

# mark which are already labeled
existing_pairs = set(
    zip(
        existing["query_id"].astype(str),
        existing["incident_id"].astype(str)
    )
)
pool_df["already_labeled"] = pool_df.apply(
    lambda r: (str(r["query_id"]), str(r["incident_id"]))
              in existing_pairs,
    axis=1
)

# pre-fill existing labels
label_map = {
    (str(r["query_id"]), str(r["incident_id"])): r["relevance"]
    for _, r in existing.iterrows()
}
pool_df["relevance"] = pool_df.apply(
    lambda r: label_map.get(
        (str(r["query_id"]), str(r["incident_id"])), ""
    ),
    axis=1
)

pool_df.to_csv(EVAL_DIR / "pooled_candidates.csv", index=False)

# labeling template — only rows needing new labels
needs_label = pool_df[
    (pool_df["relevance"] == "") |
    (pool_df["relevance"].isna())
]
needs_label.to_csv(
    EVAL_DIR / "pooled_labeling_template.csv", index=False
)

# summary
total      = len(pool_df)
pre_filled = pool_df["already_labeled"].sum()
to_label   = len(needs_label)
queries    = pool_df["query_id"].nunique()

print(f"\nPooled candidate set:")
print(f"  Total candidates  : {total:,}")
print(f"  Queries covered   : {queries}")
print(f"  Already labeled   : {pre_filled:,}")
print(f"  Still need label  : {to_label:,}")
print(f"\nFiles saved:")
print(f"  evaluation/pooled_candidates.csv")
print(f"  evaluation/pooled_labeling_template.csv")
print(f"\nOpen pooled_labeling_template.csv in Excel,")
print(f"fill the relevance column (2/1/0),")
print(f"then merge back into gold_relevance.csv")