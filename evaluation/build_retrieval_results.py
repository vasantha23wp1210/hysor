#!/usr/bin/env python3
"""
build_retrieval_results.py
--------------------------
Generate evaluation/retrieval_results.csv by running the UATIE-EDM
search engine for each gold query.

Place this file in: evaluation/build_retrieval_results.py
Run  : python evaluation/build_retrieval_results.py
Output: evaluation/retrieval_results.csv
"""

import sys
import importlib.util
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR     = PROJECT_ROOT / "evaluation"
GOLD_QUERIES = EVAL_DIR / "gold_queries.csv"
OUT_RESULTS  = EVAL_DIR / "retrieval_results.csv"
TOP_N        = 20


def load_rag_module():
    candidates = [
        PROJECT_ROOT / "rag_query_engine.py",
        PROJECT_ROOT / "src" / "rag_query_engine.py",
        PROJECT_ROOT / "src" / "rag" / "rag_query_engine.py",
        PROJECT_ROOT / "output" / "rag_query_engine.py",
    ]
    for path in candidates:
        if path.exists():
            print(f"[INFO] Using rag_query_engine from: {path}")
            spec   = importlib.util.spec_from_file_location("rag_query_engine", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["rag_query_engine"] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find rag_query_engine.py in project root, src/, src/rag/, or output/.")


rag_module         = load_rag_module()
initialize_resources = rag_module.initialize_resources
hybrid_search        = rag_module.hybrid_search
semantic_search      = getattr(rag_module, "semantic_search", None)


def stable_incident_id(result: dict, query_id: str, rank: int) -> str:
    for key in ["incident_id", "Incident_ID", "asrs_id", "ASRS_ID", "report_id", "id"]:
        val = result.get(key)
        if val is not None and str(val).strip() not in {"", "nan", "None"}:
            return str(val).strip()
    text = str(result.get("text", "") or "").strip()
    if text:
        snippet = text[:40].replace("\n", " ").replace(",", " ").strip()
        return f"{query_id}_R{rank}_{snippet}"
    return f"{query_id}_R{rank}"


def retrieve(query_text: str, query_id: str, top_n: int):
    results = hybrid_search(query_text, k=top_n)
    if not results and semantic_search:
        results = semantic_search(query_text, k=top_n)
    rows = []
    for rank, r in enumerate(results, start=1):
        rows.append((rank, stable_incident_id(r, query_id, rank), float(r.get("score", 0.0) or 0.0)))
    return results, rows


def main():
    if not GOLD_QUERIES.exists():
        raise FileNotFoundError(f"gold_queries.csv not found at {GOLD_QUERIES}")

    print("[INFO] Initializing RAG backend ...")
    initialize_resources(verbose=True)

    q_df = pd.read_csv(GOLD_QUERIES)
    all_rows = []

    for i, (_, row) in enumerate(q_df.iterrows(), start=1):
        qid   = str(row["query_id"])
        qtext = str(row["query_text"])
        print(f"\nRunning retrieval for {qid}: {qtext}")

        raw, normalized = retrieve(qtext, qid, TOP_N)
        print(f"  raw={len(raw)}  normalized={len(normalized)}")

        if i <= 3 and raw:
            print("  sample keys:", list(raw[0].keys()))
            print("  sample incident_id:", repr(raw[0].get("incident_id")))

        for rank, inc_id, score in normalized:
            all_rows.append({"query_id": qid, "rank": rank, "incident_id": inc_id, "score": score})

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(OUT_RESULTS, index=False)
    print(f"\n[DONE] Saved {len(out_df)} rows → {OUT_RESULTS}")


if __name__ == "__main__":
    main()
