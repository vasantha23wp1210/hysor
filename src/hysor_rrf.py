#!/usr/bin/env python3
"""
hysor_rrf.py
------------
Sparse+dense fusion baseline for the HySOR camera-ready rework
(addresses Reviewer 3, Major Issue #2: "at least a sparse+dense hybrid
fusion variant (RRF), which is the standard state of the art today").

Builds a BM25 index over the same corpus used by the existing FAISS
dense index, then runs three conditions on the same gold set used by
hysor_ablation.py:

  1) BM25 only            — sparse keyword retrieval
  2) Dense only            — same as ablation Condition A (FAISS cosine)
  3) RRF (BM25 + Dense)    — reciprocal rank fusion of the two, no training

This script imports hysor_ablation.py directly, so it reuses the exact
same engine-loading code, gold-set loader, and metric functions
(P@k, nDCG@k, MRR, MAP) as your existing ablation study — the RRF
numbers will be directly comparable to Table 5 / Table 10 in the paper.

Requires:
  pip install rank-bm25

Usage (run from wherever hysor_ablation.py runs today):
  python hysor_rrf.py --gold evaluation/gold_relevance.csv \
                       --queries evaluation/queries.csv --topk 10

Output:
  - Prints a 3-row comparison table to console
  - Saves rrf_results.csv in the current directory
"""

import os
import sys
import argparse
import re
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Reuse everything from the existing ablation script: path/engine setup,
# gold-set loader, and metric functions. This guarantees the RRF numbers
# are computed the same way as Conditions A-D in Table 10, so they can
# sit in the same table without a methodology footnote.
import hysor_ablation as abl

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise SystemExit(
        "This script needs the rank-bm25 package.\n"
        "    pip install rank-bm25\n"
        "then re-run."
    )

RRF_K = 60  # standard RRF constant (Cormack et al., 2009); results are not
            # very sensitive to this value in the 20-100 range


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def build_bm25_index():
    """Build a BM25 index over the same corpus rows as the FAISS index,
    using the same stable incident IDs, so retrieved IDs line up with
    the dense retrieval and the gold-set relevance judgements."""
    abl._load_engine()
    print("Loading HySOR resources (this reuses your existing engine init)...")
    abl.eng.initialize_resources(verbose=True)
    eng = abl.eng

    if eng.meta is None:
        raise RuntimeError(
            "eng.meta is not loaded after initialize_resources() -- "
            "check that your engine config points at the right corpus."
        )

    corpus_ids, corpus_tokens = [], []
    for idx in range(len(eng.meta)):
        row = eng.meta.iloc[idx]
        inc_id = eng._stable_incident_id_from_row(row, idx)
        text = eng._safe_get(row, ["summary", "text", "narrative"], "")
        corpus_ids.append(inc_id)
        corpus_tokens.append(_tokenize(text))

    print(f"Building BM25 index over {len(corpus_ids):,} records ...")
    bm25 = BM25Okapi(corpus_tokens)
    print("BM25 index ready.")
    return bm25, corpus_ids


def retrieve_bm25_ranked(bm25, corpus_ids, query, k=50):
    """Return [(incident_id, bm25_score), ...] sorted desc, top-k."""
    scores = bm25.get_scores(_tokenize(query))
    order = np.argsort(scores)[::-1][:k]
    return [(corpus_ids[i], float(scores[i])) for i in order]


def retrieve_dense_ranked(query, k=50):
    """Return [(incident_id, sim_score), ...] sorted desc, top-k.
    Mirrors hysor_ablation.retrieve_semantic_only (Condition A) exactly."""
    results = abl._faiss_raw(query, k)
    results.sort(key=lambda x: x["sim_score"], reverse=True)
    return [(r["incident_id"], r["sim_score"]) for r in results[:k]]


def rrf_fuse(bm25_ranked, dense_ranked, k=10, rrf_k=RRF_K):
    """
    Reciprocal Rank Fusion: score(d) = sum over systems of 1 / (rrf_k + rank).
    No training, no score normalisation needed -- only the rank position
    within each system's list matters.
    """
    scores = {}
    for rank, (inc_id, _) in enumerate(bm25_ranked, start=1):
        scores[inc_id] = scores.get(inc_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, (inc_id, _) in enumerate(dense_ranked, start=1):
        scores[inc_id] = scores.get(inc_id, 0.0) + 1.0 / (rrf_k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [inc_id for inc_id, _ in fused[:k]]


def evaluate_condition(retrieve_ids_fn, gold_list, k):
    """Same metric set as hysor_ablation.evaluate(), reusing its
    metric functions directly so numbers are computed identically."""
    metrics = {m: [] for m in ["P@1", "P@3", "P@5", "P@10",
                                "nDCG@5", "nDCG@10", "MRR", "MAP"]}
    skipped = 0
    for entry in gold_list:
        query_text = entry.get("query_text")
        relevant = entry.get("relevant_ids", set())
        if not query_text or not relevant:
            skipped += 1
            continue
        retrieved = retrieve_ids_fn(query_text, k)
        metrics["P@1"].append(abl.precision_at_k(retrieved, relevant, 1))
        metrics["P@3"].append(abl.precision_at_k(retrieved, relevant, 3))
        metrics["P@5"].append(abl.precision_at_k(retrieved, relevant, 5))
        metrics["P@10"].append(abl.precision_at_k(retrieved, relevant, 10))
        metrics["nDCG@5"].append(abl.ndcg_at_k(retrieved, relevant, 5))
        metrics["nDCG@10"].append(abl.ndcg_at_k(retrieved, relevant, 10))
        metrics["MRR"].append(abl.reciprocal_rank(retrieved, relevant))
        metrics["MAP"].append(abl.average_precision(retrieved, relevant, k))
    if skipped:
        print(f"  \u26a0 Skipped {skipped} queries (no text or no relevant docs)")
    return {m: round(float(np.mean(v)), 4) for m, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="HySOR RRF (sparse+dense fusion) baseline")
    parser.add_argument("--gold", required=True, help="Path to gold_relevance.csv")
    parser.add_argument("--queries", default=None, help="Path to queries CSV")
    parser.add_argument("--topk", type=int, default=10,
                         help="Final results per query for metrics (default: 10)")
    parser.add_argument("--pool", type=int, default=50,
                         help="Depth retrieved from EACH system before fusing "
                              "(default: 50 -- standard RRF practice is to pool "
                              "deeper than the final top-k)")
    args = parser.parse_args()

    if not os.path.exists(args.gold):
        raise FileNotFoundError(f"Gold set not found: {args.gold}")

    gold_list = abl.load_gold(args.gold, args.queries)
    has_texts = sum(1 for g in gold_list if g["query_text"])
    print(f"Gold set: {len(gold_list)} queries ({has_texts} with query text)")
    if has_texts == 0:
        raise SystemExit(
            "No query texts found -- provide --queries path/to/queries.csv "
            "(same file you use for hysor_ablation.py)."
        )

    bm25, corpus_ids = build_bm25_index()

    def bm25_ids(query, k):
        ranked = retrieve_bm25_ranked(bm25, corpus_ids, query, k=args.pool)
        return [inc_id for inc_id, _ in ranked[:k]]

    def dense_ids(query, k):
        ranked = retrieve_dense_ranked(query, k=args.pool)
        return [inc_id for inc_id, _ in ranked[:k]]

    def rrf_ids(query, k):
        bm25_ranked = retrieve_bm25_ranked(bm25, corpus_ids, query, k=args.pool)
        dense_ranked = retrieve_dense_ranked(query, k=args.pool)
        return rrf_fuse(bm25_ranked, dense_ranked, k=k)

    conditions = [
        ("BM25 only",            bm25_ids),
        ("Dense only (Cond. A)", dense_ids),
        ("RRF (BM25 + Dense)",   rrf_ids),
    ]

    all_results = []
    for label, fn in conditions:
        print(f"\nRunning: {label} ...")
        m = evaluate_condition(fn, gold_list, k=args.topk)
        m["Condition"] = label
        all_results.append(m)
        print(f"  P@5={m['P@5']:.4f}  nDCG@10={m['nDCG@10']:.4f}  "
              f"MRR={m['MRR']:.4f}  MAP={m['MAP']:.4f}")

    cols = ["Condition", "P@1", "P@3", "P@5", "P@10", "nDCG@5", "nDCG@10", "MRR", "MAP"]
    df_out = pd.DataFrame(all_results)[cols]

    print("\n" + "=" * 90)
    print(f"RRF BASELINE RESULTS  (rrf_k={RRF_K}, pool depth={args.pool}, final top-k={args.topk})")
    print("=" * 90)
    print(df_out.to_string(index=False))
    print("=" * 90)

    df_out.to_csv("rrf_results.csv", index=False)
    print("\nSaved \u2192 rrf_results.csv")


if __name__ == "__main__":
    main()
