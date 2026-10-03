#!/usr/bin/env python3
"""
bm25_baseline.py
----------------
Implements BM25 baseline retrieval and compares against
UATIE-EDM hybrid semantic-ontological retrieval.

Outputs:
- evaluation/bm25_results.csv
- evaluation/bm25_scores.csv
- evaluation/comparison_table.csv
- prints comparison summary

Run:
    python evaluation/bm25_baseline.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
import json
from rank_bm25 import BM25Okapi

EVAL_DIR    = Path("evaluation")
PROCESSED   = Path("data/processed")

GOLD_Q      = EVAL_DIR / "gold_queries.csv"
GOLD_REL    = EVAL_DIR / "gold_relevance.csv"
HYBRID_RES  = EVAL_DIR / "retrieval_results_enriched.csv"
CORPUS_CSV  = PROCESSED / "asrs_faiss_meta.csv"

OUT_BM25_RESULTS  = EVAL_DIR / "bm25_results.csv"
OUT_BM25_SCORES   = EVAL_DIR / "bm25_scores.csv"
OUT_COMPARISON    = EVAL_DIR / "comparison_table.csv"

TOP_N           = 20
RELEVANCE_THRESH = 1
K_VALUES        = [1, 3, 5, 10]


# ── text helpers ──────────────────────────────────────────────────

def tokenize(text: str) -> list:
    """Simple whitespace + lowercase tokenizer."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


# ── metrics ───────────────────────────────────────────────────────

def compute_metrics(retrieved_ids: list,
                    gold_ids: set,
                    gold_count: int,
                    k_values: list) -> dict:
    metrics = {}

    if gold_count == 0:
        for k in k_values:
            metrics[f"P@{k}"] = np.nan
            metrics[f"R@{k}"] = np.nan
            metrics[f"nDCG@{k}"] = np.nan
        metrics["RR"]  = np.nan
        metrics["AP"]  = np.nan
        return metrics

    # Reciprocal Rank
    rr = 0.0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in gold_ids:
            rr = 1.0 / rank
            break
    metrics["RR"] = rr

    # Average Precision
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in gold_ids:
            hits += 1
            precision_sum += hits / rank
    metrics["AP"] = precision_sum / gold_count if gold_count else 0.0

    for k in k_values:
        top_k = retrieved_ids[:k]
        k_hits = sum(1 for d in top_k if d in gold_ids)

        metrics[f"P@{k}"] = k_hits / k
        metrics[f"R@{k}"] = k_hits / gold_count

        # nDCG@k
        dcg  = 0.0
        idcg = 0.0
        for i, doc_id in enumerate(top_k, 1):
            rel = 2 if doc_id in gold_ids else 0
            dcg += rel / np.log2(i + 1)
        ideal_rels = [2] * min(gold_count, k)
        for i, r in enumerate(ideal_rels, 1):
            idcg += r / np.log2(i + 1)
        metrics[f"nDCG@{k}"] = dcg / idcg if idcg > 0 else 0.0

    return metrics


def aggregate_metrics(per_query: list, k_values: list) -> dict:
    df = pd.DataFrame(per_query)
    agg = {}
    for k in k_values:
        agg[f"P@{k}"]    = df[f"P@{k}"].mean(skipna=True)
        agg[f"R@{k}"]    = df[f"R@{k}"].mean(skipna=True)
        agg[f"nDCG@{k}"] = df[f"nDCG@{k}"].mean(skipna=True)
    agg["MRR"] = df["RR"].mean(skipna=True)
    agg["MAP"] = df["AP"].mean(skipna=True)
    return agg


# ── load data ─────────────────────────────────────────────────────

def load_data():
    print("Loading gold queries...")
    q_df = pd.read_csv(GOLD_Q)
    print(f"  Queries: {len(q_df)}")

    print("Loading gold relevance judgments...")
    rel_df = pd.read_csv(GOLD_REL)
    rel_df["is_relevant"] = rel_df["relevance"] >= RELEVANCE_THRESH
    gold = (
        rel_df[rel_df["is_relevant"]]
        .groupby("query_id")["incident_id"]
        .apply(set).to_dict()
    )
    gold_counts  = {qid: len(s) for qid, s in gold.items()}
    labeled_qids = set(rel_df["query_id"].astype(str).unique())
    print(f"  Labeled queries : {len(labeled_qids)}")

    print("Loading corpus for BM25...")
    # read in chunks to handle encoding issues
    chunks = []
    try:
        for chunk in pd.read_csv(
            CORPUS_CSV,
            low_memory=False,
            chunksize=10000,
            encoding="utf-8",
            encoding_errors="replace",
        ):
            chunks.append(chunk)
        corpus_df = pd.concat(chunks, ignore_index=True)
    except Exception as e:
        print(f"  UTF-8 failed ({e}), trying latin-1...")
        chunks = []
        for chunk in pd.read_csv(
            CORPUS_CSV,
            low_memory=False,
            chunksize=10000,
            encoding="latin-1",
        ):
            chunks.append(chunk)
        corpus_df = pd.concat(chunks, ignore_index=True)
    print(f"  Corpus rows: {len(corpus_df):,}")

    text_col = None
    for c in ["summary", "text", "narrative", "combined_text"]:
        if c in corpus_df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(
            f"No text column found. Available: {corpus_df.columns.tolist()}"
        )
    print(f"  Text column: {text_col}")

    id_col = None
    for c in ["incident_id", "record_id", "faiss_idx"]:
        if c in corpus_df.columns:
            id_col = c
            break
    print(f"  ID column: {id_col}")

    print("Loading hybrid retrieval results...")
    try:
        hybrid_df = pd.read_csv(
            HYBRID_RES,
            low_memory=False,
            encoding="utf-8",
            encoding_errors="replace",
        )
    except Exception as e:
        print(f"  UTF-8 failed ({e}), trying latin-1...")
        hybrid_df = pd.read_csv(
            HYBRID_RES,
            low_memory=False,
            encoding="latin-1",
        )
    print(f"  Hybrid results: {len(hybrid_df):,}")

    required = {"query_id", "rank", "incident_id"}
    missing  = required - set(hybrid_df.columns)
    if missing:
        raise ValueError(
            f"Hybrid results missing columns: {missing}\n"
            f"Available: {hybrid_df.columns.tolist()}"
        )

    return (q_df, gold, gold_counts, labeled_qids,
            corpus_df, text_col, id_col, hybrid_df)

# ── build BM25 index ──────────────────────────────────────────────

def build_bm25(corpus_df, text_col):
    print("\nBuilding BM25 index...")
    texts = corpus_df[text_col].fillna("").astype(str).tolist()
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    print(f"  BM25 index built over {len(tokenized):,} documents")
    return bm25, tokenized


# ── BM25 retrieval ────────────────────────────────────────────────

def bm25_retrieve(bm25, query: str, corpus_df,
                  id_col: str, top_n: int = TOP_N):
    q_tokens = tokenize(query)
    scores   = bm25.get_scores(q_tokens)
    top_idx  = np.argsort(scores)[::-1][:top_n]

    results = []
    for rank, idx in enumerate(top_idx, 1):
        if id_col:
            inc_id = str(corpus_df.iloc[idx][id_col])
        else:
            inc_id = str(idx)
        results.append({
            "rank":        rank,
            "incident_id": inc_id,
            "score":       float(scores[idx]),
            "corpus_idx":  int(idx),
        })
    return results


# ── extract hybrid results per query ─────────────────────────────

def get_hybrid_results(hybrid_df, qid: str, top_n: int = TOP_N):
    sub = hybrid_df[hybrid_df["query_id"].astype(str) == qid]
    sub = sub.sort_values("rank").head(top_n)
    return sub["incident_id"].astype(str).tolist()


# ── keyword baseline ─────────────────────────────────────────────

def keyword_retrieve(query: str, corpus_df,
                     text_col: str, id_col: str,
                     top_n: int = TOP_N):
    """Simple keyword overlap baseline."""
    keywords = set(tokenize(query))
    texts    = corpus_df[text_col].fillna("").astype(str)

    scores = texts.apply(
        lambda t: len(keywords & set(tokenize(t))) / max(len(keywords), 1)
    )
    top_idx = scores.nlargest(top_n).index.tolist()

    results = []
    for rank, idx in enumerate(top_idx, 1):
        inc_id = str(corpus_df.iloc[idx][id_col]) if id_col else str(idx)
        results.append({
            "rank":        rank,
            "incident_id": inc_id,
            "score":       float(scores[idx]),
        })
    return results


# ── main ──────────────────────────────────────────────────────────

def main():
    (q_df, gold, gold_counts,
     labeled_qids, corpus_df,
     text_col, id_col, hybrid_df) = load_data()

    # build BM25 index
    bm25, _ = build_bm25(corpus_df, text_col)

    # only evaluate on labeled queries
    eval_queries = q_df[
        q_df["query_id"].astype(str).isin(labeled_qids)
    ]
    print(f"\nEvaluating on {len(eval_queries)} labeled queries...")

    bm25_rows    = []
    kw_rows      = []
    bm25_results = []

    bm25_per_q    = []
    kw_per_q      = []
    hybrid_per_q  = []

    for _, row in eval_queries.iterrows():
        qid   = str(row["query_id"])
        qtext = str(row["query_text"])

        gold_ids    = gold.get(qid, set())
        gold_count  = gold_counts.get(qid, 0)

        # BM25 retrieval
        b_results  = bm25_retrieve(bm25, qtext, corpus_df,
                                    id_col, TOP_N)
        b_ids      = [r["incident_id"] for r in b_results]

        for r in b_results:
            bm25_results.append({
                "query_id":   qid,
                "rank":       r["rank"],
                "incident_id":r["incident_id"],
                "score":      r["score"],
            })

        b_metrics = compute_metrics(b_ids, gold_ids,
                                     gold_count, K_VALUES)
        b_metrics["query_id"] = qid
        bm25_per_q.append(b_metrics)

        # Keyword baseline
        k_results  = keyword_retrieve(qtext, corpus_df,
                                       text_col, id_col, TOP_N)
        k_ids      = [r["incident_id"] for r in k_results]
        k_metrics  = compute_metrics(k_ids, gold_ids,
                                      gold_count, K_VALUES)
        k_metrics["query_id"] = qid
        kw_per_q.append(k_metrics)

        # Hybrid results
        h_ids     = get_hybrid_results(hybrid_df, qid, TOP_N)
        h_metrics = compute_metrics(h_ids, gold_ids,
                                     gold_count, K_VALUES)
        h_metrics["query_id"] = qid
        hybrid_per_q.append(h_metrics)

    # aggregate
    bm25_agg   = aggregate_metrics(bm25_per_q,   K_VALUES)
    kw_agg     = aggregate_metrics(kw_per_q,     K_VALUES)
    hybrid_agg = aggregate_metrics(hybrid_per_q, K_VALUES)

    # save BM25 results
    pd.DataFrame(bm25_results).to_csv(OUT_BM25_RESULTS, index=False)
    print(f"\nSaved BM25 results: {OUT_BM25_RESULTS}")

    # save per-query scores
    bm25_scores_df = pd.DataFrame(bm25_per_q)
    bm25_scores_df.to_csv(OUT_BM25_SCORES, index=False)
    print(f"Saved BM25 scores : {OUT_BM25_SCORES}")

    # build comparison table
    metrics_order = []
    for k in K_VALUES:
        metrics_order += [f"P@{k}", f"R@{k}", f"nDCG@{k}"]
    metrics_order += ["MRR", "MAP"]

    comp_rows = []
    for m in metrics_order:
        comp_rows.append({
            "Metric":           m,
            "Keyword_Baseline": round(kw_agg.get(m, np.nan), 4),
            "BM25_Baseline":    round(bm25_agg.get(m, np.nan), 4),
            "UATIE_EDM_Hybrid": round(hybrid_agg.get(m, np.nan), 4),
            "Delta_vs_BM25":    round(
                hybrid_agg.get(m, 0) - bm25_agg.get(m, 0), 4
            ),
        })

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT_COMPARISON, index=False)
    print(f"Saved comparison  : {OUT_COMPARISON}")

    # print comparison table
    print(f"\n{'='*65}")
    print("  Retrieval Comparison: Keyword vs BM25 vs UATIE-EDM Hybrid")
    print(f"  Evaluated on {len(eval_queries)} queries")
    print(f"{'='*65}")
    print(f"  {'Metric':<12} {'Keyword':>10} {'BM25':>10} "
          f"{'UATIE-EDM':>12} {'Delta':>8}")
    print(f"  {'-'*56}")
    for _, r in comp_df.iterrows():
        kw  = r["Keyword_Baseline"]
        bm  = r["BM25_Baseline"]
        hy  = r["UATIE_EDM_Hybrid"]
        dlt = r["Delta_vs_BM25"]
        sig = "↑" if dlt > 0.01 else ("↓" if dlt < -0.01 else "≈")
        print(f"  {r['Metric']:<12} {kw:>10.4f} {bm:>10.4f} "
              f"{hy:>12.4f} {sig}{abs(dlt):>6.4f}")
    print(f"{'='*65}")

    # summary
    print(f"\n  Key findings:")
    mrr_delta = hybrid_agg['MRR'] - bm25_agg['MRR']
    map_delta = hybrid_agg['MAP'] - bm25_agg['MAP']
    p5_delta  = hybrid_agg['P@5'] - bm25_agg['P@5']
    print(f"  MRR : BM25={bm25_agg['MRR']:.4f} → "
          f"UATIE-EDM={hybrid_agg['MRR']:.4f} "
          f"({'+'if mrr_delta>=0 else ''}{mrr_delta:.4f})")
    print(f"  MAP : BM25={bm25_agg['MAP']:.4f} → "
          f"UATIE-EDM={hybrid_agg['MAP']:.4f} "
          f"({'+'if map_delta>=0 else ''}{map_delta:.4f})")
    print(f"  P@5 : BM25={bm25_agg['P@5']:.4f} → "
          f"UATIE-EDM={hybrid_agg['P@5']:.4f} "
          f"({'+'if p5_delta>=0 else ''}{p5_delta:.4f})")


if __name__ == "__main__":
    main()