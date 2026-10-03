#!/usr/bin/env python3
"""
assist_gold_labeling.py
-----------------------
Interactive terminal tool for manually labeling retrieval results.

Place this file in: evaluation/assist_gold_labeling.py
Run  : python evaluation/assist_gold_labeling.py
Input : evaluation/retrieval_results.csv  +  evaluation/gold_queries.csv
Output: evaluation/gold_relevance.csv  (appended / updated incrementally)

Controls at the prompt
  2       highly relevant
  1       partially relevant
  0       not relevant
  Enter   skip (no label written for this incident)
  Ctrl-C  save progress and exit cleanly
"""

import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parents[1]
EVAL_DIR       = PROJECT_ROOT / "evaluation"
RESULTS_FILE   = EVAL_DIR / "retrieval_results.csv"
QUERIES_FILE   = EVAL_DIR / "gold_queries.csv"
RELEVANCE_FILE = EVAL_DIR / "gold_relevance.csv"

TOP_K = 10  # how many candidates to show per query


def load_existing_labels() -> dict:
    """Return {(query_id, incident_id): relevance_score} already labeled."""
    if not RELEVANCE_FILE.exists():
        return {}
    df = pd.read_csv(RELEVANCE_FILE)
    return {(str(r.query_id), str(r.incident_id)): int(r.relevance)
            for _, r in df.iterrows()}


def save_labels(labels: dict):
    rows = [{"query_id": qid, "incident_id": iid, "relevance": rel}
            for (qid, iid), rel in labels.items()]
    pd.DataFrame(rows).sort_values(["query_id", "incident_id"]).to_csv(RELEVANCE_FILE, index=False)


def prompt_label(query_id: str, incident_id: str) -> int | None:
    """
    Ask user for relevance label.
    Returns int (0/1/2) or None (skip).
    Ctrl-C is re-raised so main() can exit cleanly.
    Any other input is treated as skip.
    """
    short_id = incident_id[:60] + ("..." if len(incident_id) > 60 else "")
    while True:
        try:
            raw = input(
                f"    relevance for incident {short_id}\n"
                "    (2=high, 1=partial, 0=none, Enter=skip): "
            ).strip()
        except KeyboardInterrupt:
            raise   # handled by main()

        if raw == "":
            return None
        if raw in {"0", "1", "2"}:
            return int(raw)
        # anything else (accidental paste, wrong key, etc.) → skip
        print("    [WARN] Unrecognised input — treating as skip.")
        return None


def main():
    if not RESULTS_FILE.exists():
        sys.exit(f"[ERROR] {RESULTS_FILE} not found. Run build_retrieval_results.py first.")

    results_df = pd.read_csv(RESULTS_FILE)
    queries_df = pd.read_csv(QUERIES_FILE)

    labels = load_existing_labels()
    already = len(labels)
    print(f"[INFO] Loaded {already} existing labels from {RELEVANCE_FILE if RELEVANCE_FILE.exists() else '(none)'}")

    try:
        for _, qrow in queries_df.iterrows():
            qid   = str(qrow["query_id"])
            qtext = str(qrow["query_text"])

            top = (results_df[results_df["query_id"] == qid]
                   .sort_values("rank")
                   .head(TOP_K))

            if top.empty:
                continue

            # check if all candidates for this query are already labeled
            unlabeled = [r for _, r in top.iterrows()
                         if (qid, str(r.incident_id)) not in labels]
            if not unlabeled:
                continue  # fully labeled — skip query

            print(f"\n{'='*60}")
            print(f"Query {qid}: {qtext}")
            print("Top candidates:")
            for _, r in top.iterrows():
                status = labels.get((qid, str(r.incident_id)), "?")
                marker = f"[{status}]" if status != "?" else "   "
                print(f"  {marker} rank {int(r['rank']):2d}  id={str(r.incident_id)[:55]}  score={round(float(r['score']),4)}")

            for _, r in top.iterrows():
                inc_id = str(r.incident_id)
                if (qid, inc_id) in labels:
                    continue  # already labeled
                rel = prompt_label(qid, inc_id)
                if rel is not None:
                    labels[(qid, inc_id)] = rel
                    save_labels(labels)   # incremental save after every answer

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Progress saved.")

    save_labels(labels)
    new_labels = len(labels) - already
    print(f"\n[DONE] {new_labels} new labels added.  Total: {len(labels)} labels in {RELEVANCE_FILE}")


if __name__ == "__main__":
    main()
