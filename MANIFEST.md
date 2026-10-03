# HYSOR manifest: paper item → script → inputs → outputs

Paths are relative to the repo root. Every Phase 1 script is an unmodified copy, kept at its original relative path so its
relative paths still resolve when run from the repo root. SHA-256 for every file: `DATA_MANIFEST.csv`.
Pass/fail per number: `REPORT.md` (non-retrieval) and `recheck/REPORT.md` (retrieval).

| Paper item | Script(s), in run order | Inputs | Outputs | Checked by |
|---|---|---|---|---|
| Abstract, Sect. 3, Fig. 1: 115,461 records (111,484 ASRS, 3,977 NTSB) | `src/tests/process_asrs_data.py` → `src/ontology/fuse_datasets.py` → `src/ontology/fuse_datasets_v4.py` | `data/raw/*` | `data/processed/asrs_for_fusion.csv`, `fused_ntsb_airline_crashes.csv`, `fused_with_asrs.csv`, `fused_corpus_v4.csv`, `fusion_report_v4.txt` | `verify.py` |
| Sect. 3.2, Fig. 1: 113,468 annotated records, intent / phase / hazard coverage | `src/rag/asrs_nlp_pipeline.py` → `src/compute_global_metrics.py` | `data/processed/fused_with_asrs.csv` | `data/processed/asrs_nlp_annotated.csv`, `asrs_faiss.index`, `asrs_faiss_meta.csv`, `metrics/metrics_global.csv` | `verify.py` |
| Sect. 3.3, Table 3: ontology triples and size | `src/phraseology/extract_phraseology_frm_CAA.py` → `src/phraseology/caa_icao_crossmap.py` → `src/ontology/build_ontology_triples.py` → `src/rag/ontology_population.py` | `data/raw/CAA_ATC_Section8.pdf`, `data/processed/CAA_ATC_section8_enriched.csv`, `asrs_nlp_annotated.csv` | `data/processed/CAA_ICAO_crossmap.csv`, `phraseology_triples.csv`, `phraseology_ontology.ttl`, `asrs_incidents.ttl`, `uatie_edm_merged.ttl` | `verify.py` |
| Table 3: FAISS type, dimension, size | `src/rag/asrs_nlp_pipeline.py` | as above | `data/processed/asrs_faiss.index` | `verify.py` |
| Sect. 4: HySOR engine and app | `src/rag/rag_query_engine.py`, `src/HySor_App.py` | `data/processed/*` | n/a | used by every retrieval script |
| Table 4: 100 gold queries in 7 buckets | `evaluation/generate_gold_queries.py` | n/a | `evaluation/gold_queries.csv` | `recheck/run.py` (query and bucket counts) |
| Sect. 5.1: relevance labels | manual labels (Q001–Q020); `src/rag/build_pooled_evaluation_candidates.py` → `src/rag/auto_label_relavance.py` → `src/rag/merge_pooled_labels.py` | `evaluation/retrieval_results.csv`, `bm25_results.csv` | `evaluation/gold_relevance.csv`, `pooled_*.csv` | `recheck/run.py` |
| Table 5, Fig. 2 | `evaluation/build_retrieval_results_v2.py` → `src/rag/bm25_baseline.py` → `src/rag/Plot_comparison.py` | `evaluation/gold_*.csv`, `data/processed/*` | `evaluation/retrieval_results*.csv`, `bm25_results.csv`, `bm25_scores.csv`, `comparison_table.csv`, `figures/retrieval_comparison.*` | `recheck/run.py`; `verify.py` (Fig. 2 values) |
| Table 6: per-bucket P@5 / MRR | **no script in the repo** | n/a | n/a | `recheck/run.py` computes it |
| Sect. 5.7: manual-subset numbers | **no script in the repo** | n/a | n/a | `recheck/run.py` computes it |
| Table 10: ablation | `src/hysor_ablation.py` (and `--sanity`) | `evaluation/gold_*.csv`, `data/processed/*` | `ablation_results.csv`, `src/sanity_check_results.csv` | `recheck/run.py` |
| Sect. 5.8: RRF | `src/hysor_rrf.py` | as above | `rrf_results.csv` | `recheck/run.py` (reference evaluation) |
| Tables 8–9, Fig. 3, Sect. 6 phase split | `src/extract_late_runway_change_analysis.py` → `src/semantic_retrieve_all_late_runways.py` → `src/analyze_semantic_late_runway_change_results.py` → `src/compute_scenario_metrics.py` → `src/rag/plot_hazard_analysis.py` | `data/processed/*` | `output/late_runway_*.csv`, `output/semantic_late_runway_*.csv`, `metrics/metrics_scenarios.csv`, `evaluation/figures/hazard_analysis.*` | `verify.py` |

## Known gaps (not fixed, recorded)
- **Figures 2 and 3 hard-code their numbers** in the plotting scripts and don't read the result CSVs. `verify.py` checks the hard-coded values against the CSVs.
- **The phraseology inputs are not in the repo:** `caa_icao_crossmap.py` needs `icao_phraseology_auto.csv`, `icao_phraseology_master.csv` and `doc4444_12_3_ATC_phraseologies.csv`, and none of them is present. So `CAA_ICAO_crossmap.csv` can be checked by hash but not regenerated.
- **Filename case:** `fuse_datasets_v4.py` writes `fused_corpus_V4.csv` / `fusion_report_V4.txt` (capital V), but the repo stores them with a lowercase `v4`. `rag_query_engine.py` reads `caa_icao_crossmap.csv` (lowercase). Both resolve the same on Windows but not on Linux.
- **The recency score uses the current year** (`datetime.now().year`), so Conditions B/D and the HySOR engine ranking can change in a later year.
- **`reproduce.py` checks results; it does not rebuild them from raw.** The scripts listed above overwrite files in `data/processed/`, `evaluation/` and `output/` when run, and the full rebuild from raw has not been re-run end to end.
- **Raw ASRS/NTSB/Kaggle/CAA files (`data/raw/`) and the processed CSV/TTL files contain incident narratives.** Check each source's terms before making the repo public.
