#!/usr/bin/env python3
"""
score_retrieval.py
------------------
Compute IR evaluation metrics from gold labels.

Place this file in: evaluation/score_retrieval.py
Run  : python evaluation/score_retrieval.py
Input : evaluation/retrieval_results.csv
        evaluation/gold_relevance.csv
Output: evaluation/retrieval_metrics.csv   (per-query)
        evaluation/aggregate_metrics.json  (overall)
        prints aggregate summary to console
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
EVAL_DIR        = PROJECT_ROOT / "evaluation"
RESULTS_FILE    = EVAL_DIR / "retrieval_results.csv"
RELEVANCE_FILE  = EVAL_DIR / "gold_relevance.csv"
METRICS_OUT     = EVAL_DIR / "retrieval_metrics.csv"
AGGREGATE_OUT   = EVAL_DIR / "aggregate_metrics.json"

KS = [1, 3, 5, 10]
RELEVANCE_THRESHOLD = 1   # >= 1 counts as relevant


def precision_at_k(ranked_ids, relevant_set, k):
    top_k = ranked_ids[:k]
    hits  = sum(1 for i in top_k if i in relevant_set)
    return hits / k


def recall_at_k(ranked_ids, relevant_set, k):
    if not relevant_set:
        return 0.0
    top_k = ranked_ids[:k]
    hits  = sum(1 for i in top_k if i in relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(ranked_ids, relevant_set):
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_set:
            return 1.0 / i
    return 0.0


def average_precision(ranked_ids, relevant_set):
    if not relevant_set:
        return 0.0
    hits, score = 0, 0.0
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_set:
            hits  += 1
            score += hits / i
    return score / len(relevant_set)


def dcg_at_k(ranked_ids, graded: dict, k):
    score = 0.0
    for i, rid in enumerate(ranked_ids[:k], start=1):
        score += graded.get(rid, 0) / np.log2(i + 1)
    return score


def ndcg_at_k(ranked_ids, graded: dict, k):
    ideal = sorted(graded.values(), reverse=True)[:k]
    idcg  = sum(v / np.log2(i + 2) for i, v in enumerate(ideal))
    return dcg_at_k(ranked_ids, graded, k) / idcg if idcg > 0 else 0.0


def main():
    results_df  = pd.read_csv(RESULTS_FILE)
    relevance_df = pd.read_csv(RELEVANCE_FILE)

    # build relevance lookup: {query_id: {incident_id: score}}
    rel_lookup = {}
    for _, r in relevance_df.iterrows():
        qid = str(r.query_id)
        rel_lookup.setdefault(qid, {})[str(r.incident_id)] = int(r.relevance)

    query_rows = []
    for qid, grp in results_df.groupby("query_id"):
        qid = str(qid)
        ranked_ids = grp.sort_values("rank")["incident_id"].astype(str).tolist()
        graded     = rel_lookup.get(qid, {})
        relevant   = {iid for iid, score in graded.items() if score >= RELEVANCE_THRESHOLD}

        row = {"query_id": qid}
        for k in KS:
            row[f"P@{k}"]    = round(precision_at_k(ranked_ids, relevant, k), 4)
            row[f"R@{k}"]    = round(recall_at_k(ranked_ids, relevant, k), 4)
            row[f"nDCG@{k}"] = round(ndcg_at_k(ranked_ids, graded, k), 4)
        row["RR"]  = round(reciprocal_rank(ranked_ids, relevant), 4)
        row["AP"]  = round(average_precision(ranked_ids, relevant), 4)
        query_rows.append(row)

    metrics_df = pd.DataFrame(query_rows)
    metrics_df.to_csv(METRICS_OUT, index=False)
    print(f"[INFO] Per-query metrics saved → {METRICS_OUT}")

    agg = {}
    numeric_cols = [c for c in metrics_df.columns if c != "query_id"]
    for col in numeric_cols:
        agg[col] = round(float(metrics_df[col].mean()), 4)
    agg["MRR"]  = agg.pop("RR")
    agg["MAP"]  = agg.pop("AP")

    with open(AGGREGATE_OUT, "w") as f:
        json.dump(agg, f, indent=2)

    print("\n========== Aggregate Retrieval Metrics ==========")
    for metric, val in agg.items():
        print(f"  {metric:<12} {val:.4f}")
    print(f"=================================================")
    print(f"[DONE] Aggregate metrics saved → {AGGREGATE_OUT}")


if __name__ == "__main__":
    main()
