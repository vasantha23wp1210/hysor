#!/usr/bin/env python3
"""
hysor_ablation.py
-----------------
Ablation study for HySOR: measures the contribution of each component
by running four conditions against the same 100-query gold set.

Conditions:
  A) Semantic only       — FAISS cosine sim, no recency, no ontology
  B) Semantic + Recency  — FAISS + recency score R(y), no ontology
  C) Semantic + Ontology — FAISS + ontology enrichment, no recency
  D) Full HySOR          — FAISS + ontology + recency (alpha=0.65)

Gold set CSV format (your actual format):
  query_id    : e.g. Q001, Q002 …
  incident_id : e.g. ASRS_0105099
  relevance   : 0 = not relevant, 1 = relevant, 2 = highly relevant

You also need a queries CSV (--queries flag) with:
  query_id    : Q001 …
  query_text  : the actual text of the query

Usage:
  # Run full ablation (needs query texts):
  python hysor_ablation.py --gold evaluation/gold_relevance.csv \
                           --queries evaluation/queries.csv --topk 10

  # Sanity-check existing HySOR scores from gold file only (no query texts needed):
  python hysor_ablation.py --gold evaluation/gold_relevance.csv --sanity

Output:
  - Prints comparison table to console
  - Saves ablation_results.csv in current directory
"""

import os
import sys
import argparse
import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

# Find the project root — the folder that contains data/processed/
# Works whether the script is placed in project root OR in src/
def _find_project_root(start):
    d = start
    for _ in range(4):   # search up to 4 levels up
        if os.path.isdir(os.path.join(d, "data", "processed")):
            return d
        d = os.path.dirname(d)
    return start   # fallback: use script location

_PROJECT_ROOT = _find_project_root(_HERE)
os.chdir(_PROJECT_ROOT)   # all relative paths in rag_query_engine resolve from here
print(f"Project root: {_PROJECT_ROOT}")

for _candidate in [
    _HERE,
    os.path.join(_PROJECT_ROOT, "src"),
    os.path.join(_PROJECT_ROOT, "src", "rag"),
]:
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

# ── Import engine (only needed for live retrieval, not sanity check) ──────────
eng = None
def _load_engine():
    global eng
    if eng is not None:
        return
    try:
        import rag_query_engine as _eng
        eng = _eng
    except ImportError:
        try:
            from rag import rag_query_engine as _eng
            eng = _eng
        except ImportError:
            raise ImportError(
                "Cannot find rag_query_engine.py. "
                "Run from your project root (uatie-edm-phase1/)."
            )

ALPHA          = 0.65   # matches rag_query_engine.py SIM_WEIGHT
RECENCY_WEIGHT = 0.35   # matches rag_query_engine.py RECENCY_WEIGHT
RELEVANCE_THRESHOLD = 1   # relevance >= 1 counts as relevant


# ═════════════════════════════════════════════════════════════════════════════
# Gold-set loader  (handles your actual CSV format)
# ═════════════════════════════════════════════════════════════════════════════

def load_gold(gold_path, queries_path=None):
    """
    Returns a list of dicts: [{query_id, query_text, relevant_ids}, ...]
    relevant_ids = set of incident_ids with relevance >= RELEVANCE_THRESHOLD
    """
    df = pd.read_csv(gold_path)

    # Build per-query relevant set
    rel_col = "relevance" if "relevance" in df.columns else df.columns[-1]
    gold = {}
    for qid, grp in df.groupby("query_id"):
        rel_ids = set(
            grp.loc[grp[rel_col] >= RELEVANCE_THRESHOLD, "incident_id"].astype(str)
        )
        gold[str(qid)] = {"query_id": str(qid), "relevant_ids": rel_ids, "query_text": None}

    # Attach query texts if provided
    if queries_path and os.path.exists(queries_path):
        qdf = pd.read_csv(queries_path)
        # Accept: query_id + query_text OR query_id + query OR just two columns
        text_col = next(
            (c for c in ["query_text", "query", "text", "Query", "Query_Text"] if c in qdf.columns),
            qdf.columns[1] if len(qdf.columns) >= 2 else None
        )
        id_col = next(
            (c for c in ["query_id", "id", "qid"] if c in qdf.columns),
            qdf.columns[0]
        )
        if text_col:
            for _, row in qdf.iterrows():
                qid = str(row[id_col])
                if qid in gold:
                    gold[qid]["query_text"] = str(row[text_col]).strip()
        print(f"Query texts loaded from {queries_path}")
    else:
        print("⚠ No queries file provided — live retrieval will be skipped.")
        print("  Use --queries path/to/queries.csv to enable ablation retrieval.")

    return list(gold.values())


# ═════════════════════════════════════════════════════════════════════════════
# Sanity check: compute metrics from scores already IN the gold file
# ═════════════════════════════════════════════════════════════════════════════

def sanity_check(gold_path):
    """
    Compute P@k / nDCG@k / MRR / MAP using the scores already recorded
    in gold_relevance.csv — verifies the existing HySOR results without
    needing to re-run any retrieval.
    Tries three scoring columns: composite_score, score, recency_score.
    """
    df = pd.read_csv(gold_path)
    rel_col = "relevance" if "relevance" in df.columns else df.columns[-1]

    score_cols = [c for c in ["composite_score", "score", "recency_score"] if c in df.columns]
    results = []
    for score_col in score_cols:
        metrics = {m: [] for m in ["P@1","P@3","P@5","P@10","nDCG@5","nDCG@10","MRR","MAP"]}
        for qid, grp in df.groupby("query_id"):
            relevant = set(grp.loc[grp[rel_col] >= RELEVANCE_THRESHOLD, "incident_id"].astype(str))
            if not relevant:
                continue
            ranked = grp.sort_values(score_col, ascending=False)["incident_id"].astype(str).tolist()
            metrics["P@1"].append(precision_at_k(ranked, relevant, 1))
            metrics["P@3"].append(precision_at_k(ranked, relevant, 3))
            metrics["P@5"].append(precision_at_k(ranked, relevant, 5))
            metrics["P@10"].append(precision_at_k(ranked, relevant, 10))
            metrics["nDCG@5"].append(ndcg_at_k(ranked, relevant, 5))
            metrics["nDCG@10"].append(ndcg_at_k(ranked, relevant, 10))
            metrics["MRR"].append(reciprocal_rank(ranked, relevant))
            metrics["MAP"].append(average_precision(ranked, relevant, 10))
        row = {m: round(float(np.mean(v)), 4) for m, v in metrics.items()}
        row["Condition"] = f"Existing ({score_col})"
        results.append(row)

    cols = ["Condition","P@1","P@3","P@5","P@10","nDCG@5","nDCG@10","MRR","MAP"]
    df_out = pd.DataFrame(results)[cols]
    print("\n" + "=" * 95)
    print("SANITY CHECK — metrics from existing scores in gold_relevance.csv")
    print(f"(relevance threshold >= {RELEVANCE_THRESHOLD})")
    print("=" * 95)
    print(df_out.to_string(index=False))
    print("=" * 95)
    df_out.to_csv("sanity_check_results.csv", index=False)
    print("Saved → sanity_check_results.csv")


# ═════════════════════════════════════════════════════════════════════════════
# Component-isolated retrieval functions
# ═════════════════════════════════════════════════════════════════════════════

def _faiss_raw(query, k):
    """Return FAISS results with raw cosine scores — no composite scoring."""
    if not eng.SEMANTIC_OK or eng.index is None or eng.meta is None:
        return []
    q_emb = eng.embed_query(query)
    distances, indices = eng.index.search(q_emb, k)
    results = []
    for rank, (idx, sim_score) in enumerate(zip(indices[0], distances[0]), 1):
        if idx == -1:
            continue
        if 0 <= idx < len(eng.meta):
            row = eng.meta.iloc[idx]
            inc_id = eng._stable_incident_id_from_row(row, idx)
            year, month = eng._date_lookup.get(str(idx), (None, None))
            if year is None:
                year, month = eng._date_lookup.get(inc_id, (None, None))
            results.append({
                "incident_id": inc_id,
                "sim_score":   round(float(sim_score), 4),
                "recency_score": eng.compute_recency_score(year, month),
                "year": year,
                "month": month,
                "text": str(eng._safe_get(row, ["summary", "text", "narrative"], "")),
                "source": str(eng._safe_get(row, ["source", "Source"], "ASRS")),
            })
    return results


def retrieve_semantic_only(query, k=10):
    """Condition A: pure FAISS cosine similarity, sorted by sim_score."""
    results = _faiss_raw(query, k)
    results.sort(key=lambda x: x["sim_score"], reverse=True)
    return results[:k]


def retrieve_semantic_recency(query, k=10):
    """Condition B: FAISS + recency composite (alpha=0.65), no ontology."""
    results = _faiss_raw(query, k)
    for r in results:
        r["composite_score"] = round(
            ALPHA * r["sim_score"] + RECENCY_WEIGHT * r["recency_score"], 4
        )
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results[:k]


def retrieve_semantic_ontology(query, k=10):
    """Condition C: FAISS sorted by sim_score + ontology enrichment (phraseme), no recency."""
    results = _faiss_raw(query, k)
    # Add ontology phraseme links (same as hybrid_search)
    phras_map = eng._get_phraseme_map()
    for r in results:
        r["phraseme_link"] = phras_map.get(r["incident_id"], "—")
    # Sort by raw semantic similarity only (no recency)
    results.sort(key=lambda x: x["sim_score"], reverse=True)
    return results[:k]


def retrieve_full_hysor(query, k=10):
    """Condition D: Full HySOR — composite scoring (alpha=0.65) + ontology."""
    results = _faiss_raw(query, k)
    phras_map = eng._get_phraseme_map()
    for r in results:
        r["composite_score"] = round(
            ALPHA * r["sim_score"] + RECENCY_WEIGHT * r["recency_score"], 4
        )
        r["phraseme_link"] = phras_map.get(r["incident_id"], "—")
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results[:k]


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ═════════════════════════════════════════════════════════════════════════════

def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_ids)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, r in enumerate(top_k) if r in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(retrieved_ids, relevant_ids):
    for i, r in enumerate(retrieved_ids, 1):
        if r in relevant_ids:
            return 1.0 / i
    return 0.0


def average_precision(retrieved_ids, relevant_ids, k):
    hits, total = 0, 0.0
    for i, r in enumerate(retrieved_ids[:k], 1):
        if r in relevant_ids:
            hits += 1
            total += hits / i
    return total / len(relevant_ids) if relevant_ids else 0.0


def evaluate(retrieve_fn, gold_list, k=10, label=""):
    """gold_list: output of load_gold() — list of {query_id, query_text, relevant_ids}"""
    metrics = {m: [] for m in ["P@1", "P@3", "P@5", "P@10",
                                "nDCG@5", "nDCG@10", "MRR", "MAP"]}
    skipped = 0
    for entry in gold_list:
        query_text = entry.get("query_text")
        relevant   = entry.get("relevant_ids", set())
        if not query_text or not relevant:
            skipped += 1
            continue
        results   = retrieve_fn(query_text, k=k)
        retrieved = [r["incident_id"] for r in results]

        metrics["P@1"].append(precision_at_k(retrieved, relevant, 1))
        metrics["P@3"].append(precision_at_k(retrieved, relevant, 3))
        metrics["P@5"].append(precision_at_k(retrieved, relevant, 5))
        metrics["P@10"].append(precision_at_k(retrieved, relevant, 10))
        metrics["nDCG@5"].append(ndcg_at_k(retrieved, relevant, 5))
        metrics["nDCG@10"].append(ndcg_at_k(retrieved, relevant, 10))
        metrics["MRR"].append(reciprocal_rank(retrieved, relevant))
        metrics["MAP"].append(average_precision(retrieved, relevant, k))

    if skipped:
        print(f"  ⚠ Skipped {skipped} queries (no text or no relevant docs)")
    return {m: round(float(np.mean(v)), 4) for m, v in metrics.items()}


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HySOR Ablation Study")
    parser.add_argument("--gold",    required=True,
                        help="Path to gold_relevance.csv")
    parser.add_argument("--queries", default=None,
                        help="Path to queries CSV (columns: query_id, query_text)")
    parser.add_argument("--topk",   type=int, default=10,
                        help="Results to retrieve per query (default: 10)")
    parser.add_argument("--sanity", action="store_true",
                        help="Only compute metrics from existing gold scores (no retrieval needed)")
    args = parser.parse_args()

    if not os.path.exists(args.gold):
        raise FileNotFoundError(f"Gold set not found: {args.gold}")

    # ── Sanity check mode: no engine needed ──────────────────────────────────
    if args.sanity:
        sanity_check(args.gold)
        return

    # ── Full ablation mode ───────────────────────────────────────────────────
    gold_list = load_gold(args.gold, args.queries)
    has_texts = sum(1 for g in gold_list if g["query_text"])
    print(f"\nGold set : {len(gold_list)} queries ({has_texts} with query text)")
    print(f"Top-k    : {args.topk}")
    print(f"Alpha    : {ALPHA} (semantic) / {RECENCY_WEIGHT} (recency)")

    if has_texts == 0:
        print("\n✗ No query texts found. Cannot run live retrieval.")
        print("  Either provide --queries path/to/queries.csv")
        print("  Or run with --sanity to check existing gold scores.\n")
        return

    _load_engine()
    print("\nLoading HySOR resources...")
    eng.initialize_resources(verbose=True)
    print()

    conditions = [
        ("A — Semantic only",       retrieve_semantic_only),
        ("B — Semantic + Recency",  retrieve_semantic_recency),
        ("C — Semantic + Ontology", retrieve_semantic_ontology),
        ("D — Full HySOR",          retrieve_full_hysor),
    ]

    all_results = []
    for label, fn in conditions:
        print(f"Running: {label} ...")
        m = evaluate(fn, gold_list, k=args.topk, label=label)
        m["Condition"] = label
        all_results.append(m)
        print(f"  P@5={m['P@5']:.4f}  nDCG@10={m['nDCG@10']:.4f}  "
              f"MRR={m['MRR']:.4f}  MAP={m['MAP']:.4f}")

    cols = ["Condition","P@1","P@3","P@5","P@10","nDCG@5","nDCG@10","MRR","MAP"]
    df_out = pd.DataFrame(all_results)[cols]

    print("\n" + "=" * 90)
    print(f"ABLATION RESULTS  (α={ALPHA} semantic / {RECENCY_WEIGHT} recency)")
    print("=" * 90)
    print(df_out.to_string(index=False))
    print("=" * 90)

    df_out.to_csv("ablation_results.csv", index=False)
    print("\nSaved → ablation_results.csv")


if __name__ == "__main__":
    main()