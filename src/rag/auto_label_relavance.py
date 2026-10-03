#!/usr/bin/env python3
"""
auto_label_relevance.py
-----------------------
Automatically fills relevance scores for pooled candidates
using sentence-transformer cosine similarity between
query and narrative text.

Labeling logic:
  similarity >= 0.60  → relevance = 2 (highly relevant)
  similarity >= 0.45  → relevance = 1 (partially relevant)
  similarity <  0.45  → relevance = 0 (not relevant)

These thresholds are tuned to the domain and can be adjusted.

Outputs:
  evaluation/pooled_labeling_template_autolabeled.csv
  evaluation/gold_relevance.csv  (updated with new labels)

Run:
    python evaluation/auto_label_relevance.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys
import importlib.util

EVAL_DIR  = Path("evaluation")
TEMPLATE  = EVAL_DIR / "pooled_labeling_template.csv"
GOLD_REL  = EVAL_DIR / "gold_relevance.csv"
OUT_AUTO  = EVAL_DIR / "pooled_labeling_template_autolabeled.csv"

# thresholds — adjust if needed
HIGH_THRESH    = 0.60   # cosine sim >= this → relevance 2
PARTIAL_THRESH = 0.45   # cosine sim >= this → relevance 1
                         # below partial → relevance 0

EMBED_MODEL = "all-MiniLM-L6-v2"


# ── load sentence transformer ─────────────────────────────────────

def load_model():
    print("Loading sentence-transformer model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        print(f"  Model loaded: {EMBED_MODEL}")
        return model
    except ImportError:
        print("  sentence-transformers not installed.")
        print("  Run: pip install sentence-transformers")
        sys.exit(1)


# ── cosine similarity ─────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


# ── assign relevance from similarity ─────────────────────────────

def sim_to_relevance(sim: float) -> int:
    if sim >= HIGH_THRESH:
        return 2
    if sim >= PARTIAL_THRESH:
        return 1
    return 0


# ── main ──────────────────────────────────────────────────────────

def main():
    if not TEMPLATE.exists():
        raise FileNotFoundError(
            f"Template not found: {TEMPLATE}\n"
            f"Run make_pool_simple.py first."
        )

    print("Loading pooled labeling template...")
    df = pd.read_csv(TEMPLATE, low_memory=False, encoding="latin-1")
    print(f"  Rows to label: {len(df):,}")
    print(f"  Columns: {df.columns.tolist()}")

    # find narrative column
    nar_col = None
    for c in ["narrative_preview", "narrative", "summary", "text"]:
        if c in df.columns:
            nar_col = c
            break
    if nar_col is None:
        # try to extract from incident_id pseudo-text
        df["_narrative"] = df["incident_id"].astype(str).apply(
            lambda x: x.split("_", 2)[-1].replace("_", " ")
            if "_" in x else x
        )
        nar_col = "_narrative"
        print(f"  No narrative column found — using incident_id text")
    else:
        print(f"  Narrative column: {nar_col}")

    # find query text column
    q_col = None
    for c in ["query_text", "query", "text"]:
        if c in df.columns:
            q_col = c
            break
    if q_col is None:
        raise ValueError(
            f"No query text column found. "
            f"Columns: {df.columns.tolist()}"
        )
    print(f"  Query column: {q_col}")

    # load model
    model = load_model()

    # ── encode all unique queries ─────────────────────────────────
    unique_queries = df[q_col].astype(str).unique().tolist()
    print(f"\nEncoding {len(unique_queries)} unique queries...")
    q_embeddings = model.encode(
        unique_queries,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    q_emb_map = {
        q: q_embeddings[i]
        for i, q in enumerate(unique_queries)
    }

    # ── encode all narratives in batches ──────────────────────────
    narratives = df[nar_col].fillna("").astype(str).tolist()
    print(f"Encoding {len(narratives):,} narratives...")
    n_embeddings = model.encode(
        narratives,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=256
    )

    # ── compute similarities and assign labels ────────────────────
    print("Computing similarities and assigning relevance labels...")
    similarities = []
    relevances   = []

    for i, row_nar_emb in enumerate(n_embeddings):
        q_text  = str(df.iloc[i][q_col])
        q_emb   = q_emb_map[q_text]
        sim     = cosine_sim(q_emb, row_nar_emb)
        rel     = sim_to_relevance(sim)
        similarities.append(round(sim, 4))
        relevances.append(rel)

    df["auto_similarity"] = similarities
    df["relevance"]       = relevances

    # ── save autolabeled template ─────────────────────────────────
    df.to_csv(OUT_AUTO, index=False)
    print(f"\nSaved autolabeled template: {OUT_AUTO}")

    # ── print label distribution ──────────────────────────────────
    dist = df["relevance"].value_counts().sort_index()
    total = len(df)
    print(f"\nRelevance distribution:")
    print(f"  2 (highly relevant)   : "
          f"{dist.get(2,0):>5} ({dist.get(2,0)/total*100:.1f}%)")
    print(f"  1 (partially relevant): "
          f"{dist.get(1,0):>5} ({dist.get(1,0)/total*100:.1f}%)")
    print(f"  0 (not relevant)      : "
          f"{dist.get(0,0):>5} ({dist.get(0,0)/total*100:.1f}%)")

    # similarity stats
    sims = np.array(similarities)
    print(f"\nSimilarity statistics:")
    print(f"  Min    : {sims.min():.4f}")
    print(f"  Mean   : {sims.mean():.4f}")
    print(f"  Median : {np.median(sims):.4f}")
    print(f"  Max    : {sims.max():.4f}")

    # ── merge with existing gold labels ───────────────────────────
    print("\nMerging with existing gold_relevance.csv...")
    if GOLD_REL.exists():
        existing = pd.read_csv(GOLD_REL)
        print(f"  Existing labels: {len(existing):,}")
    else:
        existing = pd.DataFrame(
            columns=["query_id", "incident_id", "relevance"]
        )

    # new labels from auto-labeling
    new_labels = df[["query_id", "incident_id", "relevance"]].copy()
    new_labels["relevance"] = new_labels["relevance"].astype(int)

    # combine — existing labels take priority over auto labels
    combined = pd.concat(
        [existing, new_labels], ignore_index=True
    )
    combined = combined.drop_duplicates(
        subset=["query_id", "incident_id"],
        keep="first"   # keep existing manual labels
    )
    combined.to_csv(GOLD_REL, index=False)
    print(f"  Updated gold_relevance.csv: {len(combined):,} rows")

    print("\nDone. Now run:")
    print("  python evaluation/bm25_baseline.py")
    print("to get the fair comparison results.")


if __name__ == "__main__":
    main()