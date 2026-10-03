#!/usr/bin/env python3
"""
build_retrieval_results_v2.py
------------------------------
Generates retrieval_results.csv with REAL narratives and dates
captured directly at search time from the FAISS meta.
"""

from pathlib import Path
import pandas as pd
import sys
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR     = PROJECT_ROOT / "evaluation"
GOLD_QUERIES = EVAL_DIR / "gold_queries.csv"
OUT_RESULTS  = EVAL_DIR / "retrieval_results.csv"
OUT_ENRICHED = EVAL_DIR / "retrieval_results_enriched.csv"

TOP_N = 20

# ── load rag module ───────────────────────────────────────────────
def load_rag():
    for p in [
        PROJECT_ROOT / "src" / "rag" / "rag_query_engine.py",
        PROJECT_ROOT / "src" / "rag_query_engine.py",
        PROJECT_ROOT / "output" / "rag_query_engine.py",
        PROJECT_ROOT / "rag_query_engine.py",
    ]:
        if p.exists():
            spec   = importlib.util.spec_from_file_location("rag_query_engine", p)
            module = importlib.util.module_from_spec(spec)
            sys.modules["rag_query_engine"] = module
            spec.loader.exec_module(module)
            print(f"  Loaded engine from: {p}")
            return module
    raise FileNotFoundError("Cannot find rag_query_engine.py")


def main():
    rag = load_rag()
    print("Initializing resources...")
    rag.initialize_resources(verbose=True)

    if not GOLD_QUERIES.exists():
        raise FileNotFoundError(f"Not found: {GOLD_QUERIES}")

    q_df = pd.read_csv(GOLD_QUERIES)
    print(f"Loaded {len(q_df)} gold queries")

    rows     = []
    enriched = []

    for _, row in q_df.iterrows():
        qid   = str(row["query_id"])
        qtext = str(row["query_text"])
        print(f"  {qid}: {qtext}")

        # use hybrid search to get results
        results = rag.hybrid_search(qtext, k=TOP_N)

        for r in results:
            rank      = r.get("rank", 0)
            inc_id    = str(r.get("incident_id", "") or "")
            score     = float(r.get("score", 0.0))
            year      = r.get("year", "Unknown")
            text      = str(r.get("text", "") or "")
            intent    = str(r.get("intent", "") or "")
            phase     = str(r.get("phase", "") or "")
            hazards   = str(r.get("hazards", "") or "")
            source    = str(r.get("source", "") or "")
            rec_score = float(r.get("recency_score", 0.5))
            comp      = float(r.get("composite_score", score))

            # build stable incident_id from record_id or position
            if not inc_id or inc_id in ("", "nan", "None"):
                inc_id = f"{qid}_R{rank}"

            rows.append({
                "query_id":    qid,
                "rank":        rank,
                "incident_id": inc_id,
                "score":       score,
            })

            enriched.append({
                "query_id":          qid,
                "rank":              rank,
                "incident_id":       inc_id,
                "score":             score,
                "recency_score":     rec_score,
                "composite_score":   comp,
                "year":              year,
                "intent":            intent,
                "phase":             phase,
                "hazards":           hazards,
                "source":            source,
                "narrative_preview": text[:400],
                "narrative":         text,
            })

    # save both files
    pd.DataFrame(rows).to_csv(OUT_RESULTS, index=False)
    pd.DataFrame(enriched).to_csv(OUT_ENRICHED, index=False)

    print(f"\nSaved retrieval_results.csv    : {len(rows)} rows")
    print(f"Saved retrieval_results_enriched: {len(enriched)} rows")

    # show sample
    sample = pd.DataFrame(enriched)
    print("\nSample (first 3 rows):")
    for _, r in sample.head(3).iterrows():
        print(f"\n  [{r['query_id']} rank={r['rank']} "
              f"year={r['year']} score={r['score']:.4f}]")
        print(f"  {r['narrative_preview'][:150]}")


if __name__ == "__main__":
    main()