#!/usr/bin/env python3
"""
semantic_retrieve_all_late_runway_change.py
-------------------------------------------
Retrieve all semantically relevant records for the scenario:
"late runway change during approach"

Outputs:
- output/semantic_late_runway_change_incidents.csv
- output/semantic_late_runway_change_query_summary.csv
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
FAISS_INDEX = PROCESSED_DIR / "asrs_faiss.index"
FAISS_META = PROCESSED_DIR / "asrs_faiss_meta.csv"
ANNOTATED_CSV = PROCESSED_DIR / "asrs_nlp_annotated.csv"

OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

OUT_INCIDENTS = OUT_DIR / "semantic_late_runway_change_incidents.csv"
OUT_SUMMARY = OUT_DIR / "semantic_late_runway_change_query_summary.csv"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

QUERIES = [
    "late runway change during approach",
    "runway changed on final approach",
    "new runway assigned during approach",
    "different runway assigned by ATC during arrival",
    "approach changed to another runway",
    "switched to different landing runway close to landing",
    "ATC reassigned runway during final approach",
    "arrival runway change caused confusion",
    "runway reassignment during visual approach",
    "runway change required reprogramming FMS on approach",
]

SIMILARITY_THRESHOLD = 0.52
INITIAL_K = 20000


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def load_resources():
    import faiss
    from sentence_transformers import SentenceTransformer

    if not FAISS_INDEX.exists():
        raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX}")
    if not FAISS_META.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {FAISS_META}")
    if not ANNOTATED_CSV.exists():
        raise FileNotFoundError(f"Annotated CSV not found: {ANNOTATED_CSV}")

    index = faiss.read_index(str(FAISS_INDEX))
    meta = pd.read_csv(FAISS_META, low_memory=False)
    corpus = pd.read_csv(ANNOTATED_CSV, low_memory=False)

    if "faiss_idx" in meta.columns:
        meta = meta.set_index("faiss_idx")

    model = SentenceTransformer(EMBED_MODEL)
    return index, meta, corpus, model


def pick_text_column(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in ["combined_text", "narrative", "summary", "synopsis", "text", "report"]:
        if cand in lower:
            return lower[cand]
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if obj_cols:
        return obj_cols[0]
    raise ValueError("No usable text column found.")


def semantic_retrieve_all(index, meta, corpus, model, queries, threshold=0.62, initial_k=5000):
    text_col = pick_text_column(corpus)

    query_emb = model.encode(
        queries,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Assumes index was built with normalized vectors for cosine/IP-style retrieval.
    # If not, scores may need reinterpretation.
    D, I = index.search(query_emb, initial_k)

    all_rows = []
    summary_rows = []

    for q_idx, query in enumerate(queries):
        scores = D[q_idx]
        ids = I[q_idx]

        kept = 0
        for score, idx in zip(scores, ids):
            if idx == -1:
                continue
            if score < threshold:
                continue

            row = {}
            row["query"] = query
            row["similarity_score"] = float(score)
            row["faiss_idx"] = int(idx)

            if idx in meta.index:
                meta_row = meta.loc[idx]
                if hasattr(meta_row, "to_dict"):
                    for k, v in meta_row.to_dict().items():
                        row[f"meta_{k}"] = v

            if idx < len(corpus):
                corpus_row = corpus.iloc[idx]
                for c in corpus.columns:
                    row[c] = corpus_row[c]

            kept += 1
            all_rows.append(row)

        summary_rows.append({
            "query": query,
            "threshold": threshold,
            "initial_k": initial_k,
            "matches_kept": kept
        })

    if all_rows:
        out = pd.DataFrame(all_rows)
        sort_cols = ["similarity_score"]
        out = out.sort_values(sort_cols, ascending=False)

        dedup_cols = [c for c in ["incident_id", "report_id", "narrative", "summary", "text", "faiss_idx"] if c in out.columns]
        if dedup_cols:
            out = out.drop_duplicates(subset=dedup_cols)
    else:
        out = pd.DataFrame()

    summary = pd.DataFrame(summary_rows)
    return out, summary, text_col


def add_light_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    text_candidate_cols = [c for c in ["combined_text", "narrative", "summary", "synopsis", "text", "report"] if c in df.columns]
    if not text_candidate_cols:
        return df

    text_col = text_candidate_cols[0]
    s = df[text_col].astype(str).str.lower()

    approach_mask = (
        s.str.contains("approach", na=False) |
        s.str.contains("final", na=False) |
        s.str.contains("arrival", na=False) |
        s.str.contains("ils", na=False) |
        s.str.contains("visual approach", na=False) |
        s.str.contains("localizer", na=False)
    )

    runway_mask = (
        s.str.contains("runway", na=False) &
        (
            s.str.contains("change", na=False) |
            s.str.contains("changed", na=False) |
            s.str.contains("new runway", na=False) |
            s.str.contains("different runway", na=False) |
            s.str.contains("assigned", na=False) |
            s.str.contains("reassigned", na=False)
        )
    )

    return df.loc[approach_mask & runway_mask].copy()


def main():
    index, meta, corpus, model = load_resources()

    results, summary, text_col = semantic_retrieve_all(
        index=index,
        meta=meta,
        corpus=corpus,
        model=model,
        queries=QUERIES,
        threshold=SIMILARITY_THRESHOLD,
        initial_k=INITIAL_K
    )

    filtered = add_light_filters(results)

    summary["text_column_used"] = text_col
    summary["final_unique_rows_after_filter"] = len(filtered)

    filtered.to_csv(OUT_INCIDENTS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"Saved incidents : {OUT_INCIDENTS} ({len(filtered)} rows)")
    print(f"Saved summary   : {OUT_SUMMARY} ({len(summary)} rows)")


if __name__ == "__main__":
    main()