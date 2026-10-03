"""Check every non-retrieval number in the paper against the artefacts in this repo.

Also checks input SHA-256 hashes (DATA_MANIFEST.csv) and the values hard-coded in the two figure scripts.
Writes verify/claims.csv and verify/results.json. Read-only on all inputs.
"""
import ast
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "verify"
P = ROOT / "data" / "processed"
MIB = 1024 * 1024


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_hashes():
    man = pd.read_csv(ROOT / "DATA_MANIFEST.csv")
    missing, bad = [], []
    for r in man.itertuples():
        f = ROOT / r.path
        if not f.exists():
            missing.append(r.path)
        elif sha256(f) != r.sha256:
            bad.append(r.path)
    return {"files": len(man), "missing": missing, "hash_mismatch": bad}


def script_lists(path, names):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def main():
    claims = []

    def add(item, claim, paper, measured, source, note="", tol=None):
        if isinstance(paper, float):
            if tol is None:  # half a unit in the paper's last printed decimal
                tol = 0.5 * 10 ** -len(repr(paper).split(".")[1]) + 1e-9
            ok = measured is not None and abs(paper - float(measured)) <= tol
        else:
            ok = paper == measured
        claims.append(dict(item=item, claim=claim, paper=paper, measured=measured,
                           status="match" if ok else "MISMATCH", source=source, note=note))

    # Corpus (Sect. 3, Fig. 1)
    v4 = pd.read_csv(P / "fused_corpus_v4.csv", usecols=["source"], low_memory=False)
    src = v4["source"].value_counts()
    add("Sect. 3 / Fig. 1", "fused records", 115461, len(v4), "data/processed/fused_corpus_v4.csv")
    add("Sect. 3 / Fig. 1", "ASRS records", 111484, int(src.get("ASRS", 0)), "data/processed/fused_corpus_v4.csv")
    add("Sect. 3 / Fig. 1", "NTSB records", 3977, int(src.get("NTSB", 0)), "data/processed/fused_corpus_v4.csv")
    fwa = pd.read_csv(P / "fused_with_asrs.csv", usecols=["source"], low_memory=False)
    add("Sect. 3 (pipeline input)", "rows in fused_with_asrs.csv (input to asrs_nlp_pipeline.py)", 115461, len(fwa),
        "data/processed/fused_with_asrs.csv",
        "by source: " + ", ".join(f"{k} {v}" for k, v in fwa["source"].value_counts().items()))

    # NLP annotation (Sect. 3.2)
    a = pd.read_csv(P / "asrs_nlp_annotated.csv", usecols=["intent", "phase_final", "hazard_tags"], low_memory=False)
    n = len(a)
    intent = a["intent"].astype(str).str.strip().str.lower()
    unclassified = intent.isin(["", "unknown", "nan", "none", "unclassified"])
    phase = a["phase_final"].astype(str).str.strip().str.lower()
    phase_cov = ~phase.isin(["", "unknown", "nan", "none"])
    hz = a["hazard_tags"].fillna("").astype(str).str.strip()
    add("Sect. 3.2", "annotated records", 113468, n, "data/processed/asrs_nlp_annotated.csv")
    add("Sect. 3.2", "intent-classified records", 69670, int((~unclassified).sum()), "asrs_nlp_annotated.csv intent")
    add("Sect. 3.2", "unclassified records", 43798, int(unclassified.sum()), "asrs_nlp_annotated.csv intent")
    add("Sect. 3.2", "intent coverage %", 60.92, round(100 * (~unclassified).mean(), 2), "asrs_nlp_annotated.csv intent")
    add("Sect. 3.2", "UNKNOWN intent %", 39.08, round(100 * unclassified.mean(), 2), "asrs_nlp_annotated.csv intent")
    add("Fig. 1", "intent classes", 13, int(a.loc[~unclassified, "intent"].nunique()), "asrs_nlp_annotated.csv intent",
        "distinct labels other than Unclassified")
    add("Sect. 3.2", "phase coverage %", 50.11, round(100 * phase_cov.mean(), 2), "asrs_nlp_annotated.csv phase_final")
    add("Fig. 1", "phase values (incl. Unknown)", 9, int(a["phase_final"].nunique()), "asrs_nlp_annotated.csv phase_final")
    add("Sect. 3.2", "hazard coverage %", 100.0, round(100 * (hz != "").mean(), 2), "asrs_nlp_annotated.csv hazard_tags")

    # FAISS index (Table 3)
    import faiss
    idx = faiss.read_index(str(P / "asrs_faiss.index"))
    add("Table 3", "FAISS vectors", 113468, int(idx.ntotal), "data/processed/asrs_faiss.index")
    add("Table 3", "FAISS dimension", 384, int(idx.d), "data/processed/asrs_faiss.index")
    add("Table 3", "FAISS index type", "IndexFlatL2", type(idx).__name__, "data/processed/asrs_faiss.index")
    add("Table 3", "FAISS index size MB (MiB)", 166.0, round((P / "asrs_faiss.index").stat().st_size / MIB, 1),
        "data/processed/asrs_faiss.index", "paper prints 166 MB; compared in MiB to the nearest unit", tol=0.5)

    # Ontology (Sect. 3.3, Table 3)
    import rdflib
    g = rdflib.Graph()
    g.parse(P / "uatie_edm_merged.ttl", format="turtle")
    add("Sect. 3.3 / Table 3", "merged ontology triples", 4880, len(g), "data/processed/uatie_edm_merged.ttl")
    add("Sect. 3.3 / Table 3", "merged ontology size MB (MiB)", 59.0,
        round((P / "uatie_edm_merged.ttl").stat().st_size / MIB, 1), "data/processed/uatie_edm_merged.ttl",
        "paper prints 59 MB; compared in MiB to the nearest unit", tol=0.5)

    # Scenario analysis (Tables 8, 9, Fig. 3)
    ms = pd.read_csv(ROOT / "metrics" / "metrics_scenarios.csv")
    ms = ms[ms.scenario == "late_runway_change_approach"]
    m = {(r.metric, r.category): r.value for r in ms.itertuples()}
    rule_n, sem_n = int(m[("rule_total_candidates", "all")]), int(m[("sem_total_semantic_candidates", "all")])
    add("Table 8", "rule-based incidents", 363, rule_n, "metrics/metrics_scenarios.csv")
    add("Table 8", "rule-based from ASRS", 361, int(m[("rule_by_source", "ASRS")]), "metrics/metrics_scenarios.csv")
    ntsb = int(m.get(("rule_by_source", "NTSB"), 0))
    other = {k: int(v) for (t, k), v in m.items() if t == "rule_by_source" and k not in ("ASRS",)}
    add("Table 8", "rule-based from NTSB", 2, ntsb, "metrics/metrics_scenarios.csv",
        f"non-ASRS rule-based sources: {other}")
    add("Table 8", "semantic incidents", 144, sem_n, "metrics/metrics_scenarios.csv")
    add("Table 8", "coverage ratio %", 39.67, round(100 * sem_n / rule_n, 2), "metrics/metrics_scenarios.csv")
    thr = re.search(r"SIMILARITY_THRESHOLD\s*=\s*([0-9.]+)",
                    (ROOT / "src" / "semantic_retrieve_all_late_runways.py").read_text(encoding="utf-8"))
    add("Table 8", "semantic threshold", 0.52, float(thr.group(1)), "src/semantic_retrieve_all_late_runways.py")
    phases = {k: int(v) for (t, k), v in m.items() if t == "rule_by_phase_inferred"}
    for label, key, paper in [("Approach", "approach", 38.6), ("Final", "final approach", 11.3),
                              ("Landing", "landing", 5.8), ("Unknown", "Unknown", 44.3)]:
        add("Sect. 6 (phase of 363)", f"{label} %", paper, round(100 * phases.get(key, 0) / rule_n, 1),
            "metrics/metrics_scenarios.csv rule_by_phase_inferred", f"all phase values: {phases}")
    t9 = [("Automation/FMS setup", "hazard", "automation_setup_hazard", 43, 27),
          ("Lateral/vertical deviation", "hazard", "lateral_vertical_deviation_hazard", 51, 27),
          ("Go-around hazard", "hazard", "go_around_hazard", 50, 25),
          ("Track deviation", "outcome_flag", "track_deviation", 58, 32),
          ("Unstable approach", "outcome_flag", "unstable_approach", 21, 11),
          ("ATC late runway change", "outcome_flag", "atc_late_change", 16, 18),
          ("Terrain/separation", "hazard", "terrain_separation_hazard", 22, 6)]
    for label, kind, key, pr, ps in t9:
        add("Table 9", f"{label} (rule)", pr, int(m.get((f"rule_{kind}", key), 0)), f"metrics_scenarios.csv rule_{kind}/{key}")
        add("Table 9", f"{label} (semantic)", ps, int(m.get((f"sem_{kind}", key), 0)), f"metrics_scenarios.csv sem_{kind}/{key}")

    # Figures: values hard-coded in the plotting scripts vs. the result files
    f2 = script_lists(ROOT / "src" / "rag" / "Plot_comparison.py", {"metrics", "keyword", "bm25", "proposed"})
    ct = pd.read_csv(ROOT / "evaluation" / "comparison_table.csv").set_index("Metric")
    for name, col in [("keyword", "Keyword_Baseline"), ("bm25", "BM25_Baseline"), ("proposed", "UATIE_EDM_Hybrid")]:
        for metric, val in zip(f2["metrics"], f2[name]):
            add("Fig. 2 (hard-coded)", f"{col} {metric}", val, float(ct.loc[metric, col]), "evaluation/comparison_table.csv")
    f3 = script_lists(ROOT / "src" / "rag" / "plot_hazard_analysis.py", {"rule_based", "semantic"})
    f3_keys = [("outcome_flag", "track_deviation"), ("outcome_flag", "fms_reprogramming"), ("hazard", "go_around_hazard"),
               ("hazard", "lateral_vertical_deviation_hazard"), ("hazard", "automation_setup_hazard"),
               ("outcome_flag", "unstable_approach"), ("outcome_flag", "atc_late_change"),
               ("hazard", "terrain_separation_hazard")]
    for (kind, key), r, s in zip(f3_keys, f3["rule_based"], f3["semantic"]):
        add("Fig. 3 (hard-coded)", f"{key} (rule)", r, int(m.get((f"rule_{kind}", key), 0)), "metrics/metrics_scenarios.csv")
        add("Fig. 3 (hard-coded)", f"{key} (semantic)", s, int(m.get((f"sem_{kind}", key), 0)), "metrics/metrics_scenarios.csv")

    OUT.mkdir(exist_ok=True)
    df = pd.DataFrame(claims)
    df.to_csv(OUT / "claims.csv", index=False)
    res = {"hashes": check_hashes(), "claims_checked": len(df), "claims_matching": int((df.status == "match").sum()),
           "mismatches": df[df.status != "match"][["item", "claim", "paper", "measured"]].astype(str).to_dict("records")}
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("hashes", "claims_checked", "claims_matching")}, indent=2))
    return res


if __name__ == "__main__":
    main()
