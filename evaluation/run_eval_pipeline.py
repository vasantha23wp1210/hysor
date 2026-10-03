#!/usr/bin/env python3
"""
run_eval_pipeline.py
--------------------
One-shot evaluation pipeline runner.

Runs these steps in order:
  Step 1  generate_gold_queries.py   -- create gold_queries.csv if missing
  Step 2  build_retrieval_results.py -- run engine, save retrieval_results.csv
  Step 3  score_retrieval.py         -- compute metrics (requires gold_relevance.csv)

Usage
-----
  python evaluation/run_eval_pipeline.py            # full pipeline
  python evaluation/run_eval_pipeline.py --step 2   # single step

Place this file in: evaluation/run_eval_pipeline.py
"""

import sys
import argparse
import subprocess
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent

STEPS = {
    1: ("generate_gold_queries",   EVAL_DIR / "generate_gold_queries.py"),
    2: ("build_retrieval_results", EVAL_DIR / "build_retrieval_results.py"),
    3: ("score_retrieval",         EVAL_DIR / "score_retrieval.py"),
}


def run_step(step_num: int):
    name, path = STEPS[step_num]
    if not path.exists():
        print(f"[SKIP] {name}: file not found at {path}")
        return
    print(f"\n{'='*60}")
    print(f"  STEP {step_num}: {name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(path)], check=False)
    if result.returncode != 0:
        print(f"[WARN] Step {step_num} exited with code {result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=[1, 2, 3],
                        help="Run only a single step (1, 2, or 3)")
    args = parser.parse_args()

    if args.step:
        run_step(args.step)
    else:
        for s in sorted(STEPS):
            run_step(s)

    print("\n[DONE] Pipeline finished.")


if __name__ == "__main__":
    main()
