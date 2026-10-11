"""Recheck the paper's retrieval numbers (Tables 5, 6, 10, the RRF paragraph and Sect. 5.7).

Reference evaluation: src/hysor_rrf.py + src/hysor_ablation.py, imported unchanged.
The Table 5 protocol of src/rag/bm25_baseline.py is replayed with its own functions for comparison.
Inputs are only read; outputs go to recheck/output/.
"""
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TOPK = 10          # hysor_rrf.py / hysor_ablation.py default --topk
RRF_POOL = 50      # hysor_rrf.py default --pool
T5_TOP_N = 20      # bm25_baseline.py TOP_N
K_VALUES = [1, 3, 5, 10]
METRICS = ["P@1", "P@3", "P@5", "P@10", "nDCG@5", "nDCG@10", "MRR", "MAP"]
MANUAL_IDS = {f"Q{i:03d}" for i in range(1, 21)}

TASK = Path(__file__).resolve().parent
REPO = TASK.parent  # repo root
GOLD_REL = REPO / "evaluation" / "gold_relevance.csv"
GOLD_Q = REPO / "evaluation" / "gold_queries.csv"
HYBRID_T5 = REPO / "evaluation" / "retrieval_results_enriched.csv"
META = REPO / "data" / "processed" / "asrs_faiss_meta.csv"
OUT = TASK / "output"

# Values transcribed from the submitted paper (48.pdf).
# (item, system, subset, metric, value); subset "all" = every evaluated query.
PAPER = []
for sys_name, vals in {
    "HySOR": [0.896, 0.909, 0.891, 0.888, 0.899, 0.901, 0.934, 0.896],
    "BM25": [0.338, 0.338, 0.278, 0.210, 0.297, 0.246, 0.506, 0.088],
    "Keyword": [0.195, 0.134, 0.119, 0.083, 0.134, 0.104, 0.303, 0.029],
}.items():
    PAPER += [("Table 5", sys_name, "all", m, v) for m, v in zip(METRICS, vals)]
for bucket, (hp5, hmrr, bp5, bmrr) in {
    "runway_change": (0.921, 0.951, 0.312, 0.501),
    "phraseology": (0.874, 0.919, 0.278, 0.499),
    "unstable_approach": (0.893, 0.932, 0.271, 0.498),
    "runway_incursion": (0.912, 0.941, 0.298, 0.514),
    "weather": (0.902, 0.929, 0.291, 0.507),
    "maintenance": (0.881, 0.928, 0.263, 0.513),
    "crm_workload": (0.872, 0.919, 0.221, 0.508),
}.items():
    PAPER += [("Table 6", "HySOR", bucket, "P@5", hp5), ("Table 6", "HySOR", bucket, "MRR", hmrr),
              ("Table 6", "BM25", bucket, "P@5", bp5), ("Table 6", "BM25", bucket, "MRR", bmrr)]
T10_METRICS = ["P@1", "P@5", "nDCG@5", "nDCG@10", "MRR", "MAP"]
for cond, vals in {
    "A": [0.909, 0.883, 0.887, 0.891, 0.938, 0.457],
    "B": [0.896, 0.909, 0.911, 0.895, 0.939, 0.462],
    "C": [0.909, 0.883, 0.887, 0.891, 0.938, 0.457],
    "D": [0.896, 0.909, 0.911, 0.895, 0.939, 0.462],
}.items():
    PAPER += [("Table 10", cond, "all", m, v) for m, v in zip(T10_METRICS, vals)]
for sys_name, vals in {"BM25": [0.283, 0.502, 0.075], "A": [0.883, 0.938, 0.457],
                       "RRF": [0.678, 0.931, 0.264]}.items():
    PAPER += [("Sect. 5.8 RRF", sys_name, "all", m, v) for m, v in zip(["P@5", "MRR", "MAP"], vals)]
PAPER += [("Sect. 5.7", "HySOR", "Q001-Q020", "MRR", 0.921), ("Sect. 5.7", "HySOR", "Q001-Q020", "MAP", 0.874),
          ("Sect. 5.7", "BM25", "Q001-Q020", "MRR", 0.531), ("Sect. 5.7", "BM25", "Q001-Q020", "MAP", 0.103)]

# Paper system -> condition evaluated under each protocol (None = not applicable).
T5_COND = {"HySOR": "hysor_engine", "BM25": "bm25_baseline", "Keyword": "keyword"}
REF_COND = {"HySOR": "hysor_engine", "BM25": "bm25_rrf", "Keyword": "keyword",
            "A": "A", "B": "B", "C": "C", "D": "D", "RRF": "rrf"}


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    import torch
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def ref_metrics(abl, ids, relevant):
    ids = ids[:TOPK]
    return {
        "P@1": abl.precision_at_k(ids, relevant, 1), "P@3": abl.precision_at_k(ids, relevant, 3),
        "P@5": abl.precision_at_k(ids, relevant, 5), "P@10": abl.precision_at_k(ids, relevant, 10),
        "nDCG@5": abl.ndcg_at_k(ids, relevant, 5), "nDCG@10": abl.ndcg_at_k(ids, relevant, 10),
        "MRR": abl.reciprocal_rank(ids, relevant), "MAP": abl.average_precision(ids, relevant, TOPK),
    }


def t5_metrics(bm25_mod, ids, relevant):
    m = bm25_mod.compute_metrics(ids, relevant, len(relevant), K_VALUES)
    out = {k: m[k] for k in METRICS if k in m}
    out["MRR"], out["MAP"] = m["RR"], m["AP"]
    return out


def bm25_orders(scores, k):
    """Top-k corpus positions under three tie orders: numpy argsort default (what Phase 1 uses),
    and stable sorts breaking ties by lower / higher corpus position."""
    pos = np.arange(len(scores))
    return {
        "argsort_default": np.argsort(scores)[::-1][:k],
        "ties_low_index_first": np.lexsort((pos, -scores))[:k],
        "ties_high_index_first": np.lexsort((-pos, -scores))[:k],
    }


def main():
    set_seeds()
    sys.path.insert(0, str(REPO / "src"))
    import hysor_ablation as abl  # chdirs to the repo root
    import hysor_rrf as rrf
    bm25_mod = load_module("bm25_baseline", REPO / "src" / "rag" / "bm25_baseline.py")

    gold_list = abl.load_gold(str(GOLD_REL), str(GOLD_Q))
    evaluated = [g for g in gold_list if g["query_text"] and g["relevant_ids"]]
    bucket = pd.read_csv(GOLD_Q).set_index("query_id")["scenario_bucket"]

    bm25_r, corpus_ids = rrf.build_bm25_index()          # also initialises the engine
    meta = pd.read_csv(META, low_memory=False)
    bm25_b, _ = bm25_mod.build_bm25(meta, "summary")
    meta_ids = meta["incident_id"].astype(str).tolist()
    hyb = pd.read_csv(HYBRID_T5)

    rows, tie_rows = [], []
    for n, g in enumerate(evaluated, 1):
        qid, text, rel = g["query_id"], g["query_text"], g["relevant_ids"]
        print(f"[{n}/{len(evaluated)}] {qid}", flush=True)
        dense_ranked = rrf.retrieve_dense_ranked(text, k=RRF_POOL)
        lists = {
            "keyword": [r["incident_id"] for r in bm25_mod.keyword_retrieve(text, meta, "summary", "incident_id", T5_TOP_N)],
            "bm25_baseline": [r["incident_id"] for r in bm25_mod.bm25_retrieve(bm25_b, text, meta, "incident_id", T5_TOP_N)],
            "hysor_engine": bm25_mod.get_hybrid_results(hyb, qid, T5_TOP_N),
            "A": [r["incident_id"] for r in abl.retrieve_semantic_only(text, k=TOPK)],
            "B": [r["incident_id"] for r in abl.retrieve_semantic_recency(text, k=TOPK)],
            "C": [r["incident_id"] for r in abl.retrieve_semantic_ontology(text, k=TOPK)],
            "D": [r["incident_id"] for r in abl.retrieve_full_hysor(text, k=TOPK)],
        }
        sr = bm25_r.get_scores(rrf._tokenize(text))
        for variant, order in bm25_orders(sr, RRF_POOL).items():
            ranked = [(corpus_ids[i], float(sr[i])) for i in order]
            ids_b = [i for i, _ in ranked[:TOPK]]
            ids_f = rrf.rrf_fuse(ranked, dense_ranked, k=TOPK)
            if variant == "argsort_default":
                lists["bm25_rrf"], lists["rrf"] = ids_b, ids_f
            tie_rows += [dict(variant=variant, condition=c, query_id=qid, **ref_metrics(abl, ids, rel))
                         for c, ids in (("bm25_rrf", ids_b), ("rrf", ids_f))]
        sb = bm25_b.get_scores(bm25_mod.tokenize(text))
        for variant, order in bm25_orders(sb, T5_TOP_N).items():
            ids = [meta_ids[i] for i in order]
            tie_rows.append(dict(variant=variant, condition="bm25_baseline_t5", query_id=qid,
                                 **t5_metrics(bm25_mod, ids, rel)))
        for cond, ids in lists.items():
            rows.append(dict(protocol="reference", condition=cond, query_id=qid, bucket=bucket[qid],
                             **ref_metrics(abl, ids, rel)))
            if cond in ("keyword", "bm25_baseline", "hysor_engine"):
                rows.append(dict(protocol="table5_bm25_baseline_py", condition=cond, query_id=qid,
                                 bucket=bucket[qid], **t5_metrics(bm25_mod, ids, rel)))

    OUT.mkdir(exist_ok=True)
    per_q = pd.DataFrame(rows)
    per_q.to_csv(OUT / "per_query_metrics.csv", index=False, float_format="%.6f")
    ties = pd.DataFrame(tie_rows)
    ties.to_csv(OUT / "bm25_tie_order_per_query.csv", index=False, float_format="%.6f")

    def agg(df, subset):
        if subset == "Q001-Q020":
            df = df[df.query_id.isin(MANUAL_IDS)]
        elif subset != "all":
            df = df[df.bucket == subset]
        return df.groupby(["protocol", "condition"])[METRICS].mean(), df.query_id.nunique()

    subsets = ["all", "Q001-Q020"] + sorted(bucket.unique())
    agg_tables = {s: agg(per_q, s) for s in subsets}
    summary = {s: {"n_queries": int(n), "metrics": {f"{p}|{c}": {m: round(float(v), 4) for m, v in r.items()}
                                                    for (p, c), r in t.iterrows()}}
               for s, (t, n) in agg_tables.items()}

    comp = []
    for item, system, subset, metric, paper in PAPER:
        t, n = agg_tables[subset]
        out = {"item": item, "system": system, "subset": subset, "n_queries_evaluated": n,
               "metric": metric, "paper": paper}
        for proto, cmap, key in (("table5_bm25_baseline_py", T5_COND, "published_protocol"),
                                 ("reference", REF_COND, "reference_hysor_rrf")):
            cond = cmap.get(system) if (item in ("Table 5", "Table 6", "Sect. 5.7") or proto == "reference") else None
            val = round(float(t.loc[(proto, cond), metric]), 4) if cond and (proto, cond) in t.index else None
            out[key] = val
            out[f"{key}_matches"] = None if val is None else bool(abs(round(val, 3) - paper) < 1e-9)
        comp.append(out)
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "paper_vs_reproduced.csv", index=False)

    tie_agg = ties.groupby(["condition", "variant"])[METRICS].mean().round(4)
    tie_agg.to_csv(OUT / "bm25_tie_order_summary.csv")
    tie_spread = {c: {m: [float(tie_agg.loc[c][m].min()), float(tie_agg.loc[c][m].max())] for m in METRICS}
                  for c in tie_agg.index.get_level_values(0).unique()}

    committed = {
        "ablation_results.csv": (REPO / "ablation_results.csv", {"A — Semantic only": "A", "B — Semantic + Recency": "B",
                                                                 "C — Semantic + Ontology": "C", "D — Full HySOR": "D"}),
        "rrf_results.csv": (REPO / "rrf_results.csv", {"BM25 only": "bm25_rrf", "Dense only (Cond. A)": "A",
                                                       "RRF (BM25 + Dense)": "rrf"}),
    }
    ref_all = agg_tables["all"][0].loc["reference"]
    checks = {}
    for name, (path, cmap) in committed.items():
        df = pd.read_csv(path).set_index("Condition")
        checks[name] = {cond: {m: {"committed": float(df.loc[label, m]), "rerun": round(float(ref_all.loc[cond, m]), 4)}
                               for m in METRICS} for label, cond in cmap.items()}
    ct = pd.read_csv(REPO / "evaluation" / "comparison_table.csv").set_index("Metric")
    t5_all = agg_tables["all"][0].loc["table5_bm25_baseline_py"]
    checks["comparison_table.csv"] = {
        cond: {m: {"committed": float(ct.loc[m, col]), "rerun": round(float(t5_all.loc[cond, m]), 4)} for m in METRICS}
        for col, cond in (("UATIE_EDM_Hybrid", "hysor_engine"), ("BM25_Baseline", "bm25_baseline"),
                          ("Keyword_Baseline", "keyword"))}
    mismatched_checks = [f"{f}:{c}:{m}" for f, cs in checks.items() for c, ms in cs.items()
                         for m, v in ms.items() if abs(v["committed"] - v["rerun"]) > 1e-4]

    import faiss
    import rank_bm25
    import sentence_transformers
    import torch
    results = {
        "seed": SEED, "topk_reference": TOPK, "rrf_pool": RRF_POOL, "table5_top_n": T5_TOP_N,
        "relevance_threshold": abl.RELEVANCE_THRESHOLD,
        "queries_in_gold_set": len(gold_list), "queries_evaluated": len(evaluated),
        "queries_skipped_no_relevant": sorted(g["query_id"] for g in gold_list if not g["relevant_ids"]),
        "evaluated_per_bucket": {b: int(v) for b, v in pd.Series([bucket[g["query_id"]] for g in evaluated]).value_counts().sort_index().items()},
        "paper_numbers_checked": len(comp),
        "matches_reference_protocol": int(comp.reference_hysor_rrf_matches.fillna(False).sum()),
        "checked_reference_protocol": int(comp.reference_hysor_rrf.notna().sum()),
        "matches_published_protocol": int(comp.published_protocol_matches.fillna(False).sum()),
        "checked_published_protocol": int(comp.published_protocol.notna().sum()),
        "summary": summary,
        "bm25_tie_order_metric_range": tie_spread,
        "rerun_vs_committed_csv": checks,
        "rerun_vs_committed_mismatches": mismatched_checks,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "faiss": faiss.__version__, "rank_bm25": getattr(rank_bm25, "__version__", "0.2.2"),
                        "sentence_transformers": sentence_transformers.__version__, "torch": torch.__version__},
        "inputs_sha256": {str(p.relative_to(REPO)): sha256(p) for p in
                          [GOLD_REL, GOLD_Q, HYBRID_T5, META, REPO / "data" / "processed" / "asrs_faiss.index",
                           REPO / "src" / "hysor_rrf.py", REPO / "src" / "hysor_ablation.py",
                           REPO / "src" / "rag" / "bm25_baseline.py", REPO / "src" / "rag" / "rag_query_engine.py"]},
    }
    (TASK / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, comp, tie_agg)
    print(json.dumps({k: results[k] for k in ["queries_evaluated", "matches_reference_protocol", "checked_reference_protocol",
                                             "matches_published_protocol", "checked_published_protocol",
                                             "rerun_vs_committed_mismatches"]}, indent=2))


def fmt(v):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def write_report(R, comp, tie_agg):
    L = [
        "# HYSOR recheck: retrieval numbers in the paper vs. a rerun",
        "",
        "## Method",
        f"- Reference evaluation: `src/hysor_rrf.py` and `src/hysor_ablation.py`, imported unchanged: same gold loader, "
        f"relevance >= {R['relevance_threshold']}, top-{R['topk_reference']}, `P@k`, `nDCG@k` (binary gain), `MRR`, "
        f"`MAP` = AP@{R['topk_reference']} divided by all relevant labels of the query.",
        f"- Published Table 5 protocol replayed with `src/rag/bm25_baseline.py` functions: top-{R['table5_top_n']} lists, "
        "AP over the top-20 divided by all relevant labels.",
        "- HySOR in Tables 5/6 and Sect. 5.7 = the stored engine ranking `evaluation/retrieval_results_enriched.csv` "
        "(hybrid_search: FAISS top-20 re-ranked by C(s,y)). Table 10 Condition D re-ranks only the FAISS top-10.",
        "- BM25 under the reference protocol = `hysor_rrf.py` BM25 (`[a-z0-9]+` tokens). BM25 under the published protocol "
        "= `bm25_baseline.py` BM25 (drops 1-character tokens).",
        "- Labels: `evaluation/gold_relevance.csv` as committed. Q021–Q100 labels are MiniLM auto-labels "
        "(`auto_label_relavance.py`); see `phase2/gold_eval_rebuild` in the uatie-edm-phase1 repo.",
        f"- Queries evaluated: {R['queries_evaluated']} of {R['queries_in_gold_set']}. Skipped (no label >= 1): "
        f"{', '.join(R['queries_skipped_no_relevant'])}.",
        f"- Evaluated per bucket: {R['evaluated_per_bucket']}.",
        f"- Seed {R['seed']}. CPU only. Inputs pinned by SHA-256 in `results.json`.",
        "",
        "## Result",
        f"- Reference protocol: {R['matches_reference_protocol']} of {R['checked_reference_protocol']} paper numbers match at 3 d.p.",
        f"- Published Table 5 protocol: {R['matches_published_protocol']} of {R['checked_published_protocol']} "
        "applicable paper numbers match at 3 d.p.",
        f"- Rerun vs. committed result CSVs: {len(R['rerun_vs_committed_mismatches'])} differing values"
        + (f" ({', '.join(R['rerun_vs_committed_mismatches'])})." if R["rerun_vs_committed_mismatches"] else "."),
        "",
        "| Item | System | Subset | n | Metric | Paper | Published protocol | Reference (hysor_rrf) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in comp.itertuples():
        def cell(v, ok):
            if v is None or pd.isna(v):
                return "–"
            return fmt(v) + ("" if ok else " ✗")
        L.append(f"| {r.item} | {r.system} | {r.subset} | {r.n_queries_evaluated} | {r.metric} | {r.paper:.3f} | "
                 f"{cell(r.published_protocol, r.published_protocol_matches)} | "
                 f"{cell(r.reference_hysor_rrf, r.reference_hysor_rrf_matches)} |")
    L += ["", "✗ = differs from the paper at 3 d.p. – = not computed (protocol not applicable, or 0 evaluated queries in the subset).", "",
          "## BM25 tie order", "",
          "BM25 scores tie often (short summaries). Phase 1 orders ties with `np.argsort` (unstable). "
          "Metrics under three tie orders:", "",
          "| Condition | Tie order | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for (c, v), r in tie_agg.iterrows():
        L.append(f"| {c} | {v} | " + " | ".join(fmt(r[m]) for m in METRICS) + " |")
    L += ["", "## Failures and caveats",
          "- All numbers here score against the Phase 1 labels, including the MiniLM auto-labels for Q021–Q100. "
          "They are not an unbiased measure; `phase2/gold_eval_rebuild` (uatie-edm-phase1 repo) replaces those labels.",
          "- The recency score depends on the current year (engine uses `datetime.now().year`).", ""]
    (TASK / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
