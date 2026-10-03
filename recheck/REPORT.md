# HYSOR recheck: retrieval numbers in the paper vs. a rerun

## Method
- Reference evaluation: `src/hysor_rrf.py` and `src/hysor_ablation.py`, imported unchanged: same gold loader, relevance >= 1, top-10, `P@k`, `nDCG@k` (binary gain), `MRR`, `MAP` = AP@10 divided by all relevant labels of the query.
- Published Table 5 protocol replayed with `src/rag/bm25_baseline.py` functions: top-20 lists, AP over the top-20 divided by all relevant labels.
- HySOR in Tables 5/6 and Sect. 5.7 = the stored engine ranking `evaluation/retrieval_results_enriched.csv` (hybrid_search: FAISS top-20 re-ranked by C(s,y)). Table 10 Condition D re-ranks only the FAISS top-10.
- BM25 under the reference protocol = `hysor_rrf.py` BM25 (`[a-z0-9]+` tokens). BM25 under the published protocol = `bm25_baseline.py` BM25 (drops 1-character tokens).
- Labels: `evaluation/gold_relevance.csv` as committed. Q021–Q100 labels are MiniLM auto-labels (`auto_label_relavance.py`); see `phase2/gold_eval_rebuild` in the uatie-edm-phase1 repo.
- Queries evaluated: 77 of 100. Skipped (no label >= 1): Q014, Q069, Q071, Q072, Q073, Q074, Q075, Q076, Q078, Q079, Q080, Q081, Q082, Q083, Q084, Q085, Q086, Q087, Q088, Q089, Q090, Q099, Q100.
- Evaluated per bucket: {'crm_workload': 8, 'phraseology': 20, 'runway_change': 19, 'runway_incursion': 9, 'unstable_approach': 20, 'weather': 1}.
- Seed 42. CPU only. Inputs pinned by SHA-256 in `results.json`.

## Result
- Reference protocol: 46 of 85 paper numbers match at 3 d.p.
- Published Table 5 protocol: 21 of 52 applicable paper numbers match at 3 d.p.
- Rerun vs. committed result CSVs: 12 differing values (rrf_results.csv:bm25_rrf:P@3, rrf_results.csv:bm25_rrf:nDCG@5, rrf_results.csv:bm25_rrf:nDCG@10, rrf_results.csv:bm25_rrf:MAP, rrf_results.csv:rrf:P@5, rrf_results.csv:rrf:nDCG@5, rrf_results.csv:rrf:nDCG@10, comparison_table.csv:bm25_baseline:P@3, comparison_table.csv:bm25_baseline:P@5, comparison_table.csv:bm25_baseline:nDCG@5, comparison_table.csv:bm25_baseline:nDCG@10, comparison_table.csv:bm25_baseline:MAP).

| Item | System | Subset | n | Metric | Paper | Published protocol | Reference (hysor_rrf) |
|---|---|---|---|---|---|---|---|
| Table 5 | HySOR | all | 77 | P@1 | 0.896 | 0.8961 | 0.8961 |
| Table 5 | HySOR | all | 77 | P@3 | 0.909 | 0.9091 | 0.9091 |
| Table 5 | HySOR | all | 77 | P@5 | 0.891 | 0.8909 | 0.8909 |
| Table 5 | HySOR | all | 77 | P@10 | 0.888 | 0.8883 | 0.8883 |
| Table 5 | HySOR | all | 77 | nDCG@5 | 0.899 | 0.8992 | 0.8992 |
| Table 5 | HySOR | all | 77 | nDCG@10 | 0.901 | 0.9013 | 0.9013 |
| Table 5 | HySOR | all | 77 | MRR | 0.934 | 0.9340 | 0.9340 |
| Table 5 | HySOR | all | 77 | MAP | 0.896 | 0.8957 | 0.4683 ✗ |
| Table 5 | BM25 | all | 77 | P@1 | 0.338 | 0.3377 | 0.3377 |
| Table 5 | BM25 | all | 77 | P@3 | 0.338 | 0.3333 ✗ | 0.3333 ✗ |
| Table 5 | BM25 | all | 77 | P@5 | 0.278 | 0.2805 ✗ | 0.2831 ✗ |
| Table 5 | BM25 | all | 77 | P@10 | 0.210 | 0.2104 | 0.2169 ✗ |
| Table 5 | BM25 | all | 77 | nDCG@5 | 0.297 | 0.2989 ✗ | 0.3000 ✗ |
| Table 5 | BM25 | all | 77 | nDCG@10 | 0.246 | 0.2461 | 0.2504 ✗ |
| Table 5 | BM25 | all | 77 | MRR | 0.506 | 0.5056 | 0.5021 ✗ |
| Table 5 | BM25 | all | 77 | MAP | 0.088 | 0.0878 | 0.0754 ✗ |
| Table 5 | Keyword | all | 77 | P@1 | 0.195 | 0.1948 | 0.1948 |
| Table 5 | Keyword | all | 77 | P@3 | 0.134 | 0.1342 | 0.1342 |
| Table 5 | Keyword | all | 77 | P@5 | 0.119 | 0.1195 | 0.1195 |
| Table 5 | Keyword | all | 77 | P@10 | 0.083 | 0.0831 | 0.0831 |
| Table 5 | Keyword | all | 77 | nDCG@5 | 0.134 | 0.1336 | 0.1336 |
| Table 5 | Keyword | all | 77 | nDCG@10 | 0.104 | 0.1036 | 0.1036 |
| Table 5 | Keyword | all | 77 | MRR | 0.303 | 0.3029 | 0.2948 ✗ |
| Table 5 | Keyword | all | 77 | MAP | 0.029 | 0.0293 | 0.0255 ✗ |
| Table 6 | HySOR | runway_change | 19 | P@5 | 0.921 | 0.9579 ✗ | 0.9579 ✗ |
| Table 6 | HySOR | runway_change | 19 | MRR | 0.951 | 1.0000 ✗ | 1.0000 ✗ |
| Table 6 | BM25 | runway_change | 19 | P@5 | 0.312 | 0.2316 ✗ | 0.2421 ✗ |
| Table 6 | BM25 | runway_change | 19 | MRR | 0.501 | 0.4433 ✗ | 0.4491 ✗ |
| Table 6 | HySOR | phraseology | 20 | P@5 | 0.874 | 0.9900 ✗ | 0.9900 ✗ |
| Table 6 | HySOR | phraseology | 20 | MRR | 0.919 | 1.0000 ✗ | 1.0000 ✗ |
| Table 6 | BM25 | phraseology | 20 | P@5 | 0.278 | 0.3500 ✗ | 0.3400 ✗ |
| Table 6 | BM25 | phraseology | 20 | MRR | 0.499 | 0.5187 ✗ | 0.5000 ✗ |
| Table 6 | HySOR | unstable_approach | 20 | P@5 | 0.893 | 0.8500 ✗ | 0.8500 ✗ |
| Table 6 | HySOR | unstable_approach | 20 | MRR | 0.932 | 0.9208 ✗ | 0.9208 ✗ |
| Table 6 | BM25 | unstable_approach | 20 | P@5 | 0.271 | 0.2700 ✗ | 0.2700 ✗ |
| Table 6 | BM25 | unstable_approach | 20 | MRR | 0.498 | 0.5597 ✗ | 0.5558 ✗ |
| Table 6 | HySOR | runway_incursion | 9 | P@5 | 0.912 | 0.8889 ✗ | 0.8889 ✗ |
| Table 6 | HySOR | runway_incursion | 9 | MRR | 0.941 | 0.9444 ✗ | 0.9444 ✗ |
| Table 6 | BM25 | runway_incursion | 9 | P@5 | 0.298 | 0.3778 ✗ | 0.3778 ✗ |
| Table 6 | BM25 | runway_incursion | 9 | MRR | 0.514 | 0.6003 ✗ | 0.6037 ✗ |
| Table 6 | HySOR | weather | 1 | P@5 | 0.902 | 0.2000 ✗ | 0.2000 ✗ |
| Table 6 | HySOR | weather | 1 | MRR | 0.929 | 0.2500 ✗ | 0.2500 ✗ |
| Table 6 | BM25 | weather | 1 | P@5 | 0.291 | 0.2000 ✗ | 0.2000 ✗ |
| Table 6 | BM25 | weather | 1 | MRR | 0.507 | 1.0000 ✗ | 1.0000 ✗ |
| Table 6 | HySOR | maintenance | 0 | P@5 | 0.881 | – | – |
| Table 6 | HySOR | maintenance | 0 | MRR | 0.928 | – | – |
| Table 6 | BM25 | maintenance | 0 | P@5 | 0.263 | – | – |
| Table 6 | BM25 | maintenance | 0 | MRR | 0.513 | – | – |
| Table 6 | HySOR | crm_workload | 8 | P@5 | 0.872 | 0.6750 ✗ | 0.6750 ✗ |
| Table 6 | HySOR | crm_workload | 8 | MRR | 0.919 | 0.7188 ✗ | 0.7188 ✗ |
| Table 6 | BM25 | crm_workload | 8 | P@5 | 0.221 | 0.1500 ✗ | 0.1750 ✗ |
| Table 6 | BM25 | crm_workload | 8 | MRR | 0.508 | 0.3176 ✗ | 0.3220 ✗ |
| Table 10 | A | all | 77 | P@1 | 0.909 | – | 0.9091 |
| Table 10 | A | all | 77 | P@5 | 0.883 | – | 0.8831 |
| Table 10 | A | all | 77 | nDCG@5 | 0.887 | – | 0.8872 |
| Table 10 | A | all | 77 | nDCG@10 | 0.891 | – | 0.8910 |
| Table 10 | A | all | 77 | MRR | 0.938 | – | 0.9380 |
| Table 10 | A | all | 77 | MAP | 0.457 | – | 0.4570 |
| Table 10 | B | all | 77 | P@1 | 0.896 | – | 0.8961 |
| Table 10 | B | all | 77 | P@5 | 0.909 | – | 0.9091 |
| Table 10 | B | all | 77 | nDCG@5 | 0.911 | – | 0.9114 |
| Table 10 | B | all | 77 | nDCG@10 | 0.895 | – | 0.8946 |
| Table 10 | B | all | 77 | MRR | 0.939 | – | 0.9387 |
| Table 10 | B | all | 77 | MAP | 0.462 | – | 0.4617 |
| Table 10 | C | all | 77 | P@1 | 0.909 | – | 0.9091 |
| Table 10 | C | all | 77 | P@5 | 0.883 | – | 0.8831 |
| Table 10 | C | all | 77 | nDCG@5 | 0.887 | – | 0.8872 |
| Table 10 | C | all | 77 | nDCG@10 | 0.891 | – | 0.8910 |
| Table 10 | C | all | 77 | MRR | 0.938 | – | 0.9380 |
| Table 10 | C | all | 77 | MAP | 0.457 | – | 0.4570 |
| Table 10 | D | all | 77 | P@1 | 0.896 | – | 0.8961 |
| Table 10 | D | all | 77 | P@5 | 0.909 | – | 0.9091 |
| Table 10 | D | all | 77 | nDCG@5 | 0.911 | – | 0.9114 |
| Table 10 | D | all | 77 | nDCG@10 | 0.895 | – | 0.8946 |
| Table 10 | D | all | 77 | MRR | 0.939 | – | 0.9387 |
| Table 10 | D | all | 77 | MAP | 0.462 | – | 0.4617 |
| Sect. 5.8 RRF | BM25 | all | 77 | P@5 | 0.283 | – | 0.2831 |
| Sect. 5.8 RRF | BM25 | all | 77 | MRR | 0.502 | – | 0.5021 |
| Sect. 5.8 RRF | BM25 | all | 77 | MAP | 0.075 | – | 0.0754 |
| Sect. 5.8 RRF | A | all | 77 | P@5 | 0.883 | – | 0.8831 |
| Sect. 5.8 RRF | A | all | 77 | MRR | 0.938 | – | 0.9380 |
| Sect. 5.8 RRF | A | all | 77 | MAP | 0.457 | – | 0.4570 |
| Sect. 5.8 RRF | RRF | all | 77 | P@5 | 0.678 | – | 0.6753 ✗ |
| Sect. 5.8 RRF | RRF | all | 77 | MRR | 0.931 | – | 0.9306 |
| Sect. 5.8 RRF | RRF | all | 77 | MAP | 0.264 | – | 0.2638 |
| Sect. 5.7 | HySOR | Q001-Q020 | 19 | MRR | 0.921 | 1.0000 ✗ | 1.0000 ✗ |
| Sect. 5.7 | HySOR | Q001-Q020 | 19 | MAP | 0.874 | 0.9644 ✗ | 0.4827 ✗ |
| Sect. 5.7 | BM25 | Q001-Q020 | 19 | MRR | 0.531 | 0.4433 ✗ | 0.4491 ✗ |
| Sect. 5.7 | BM25 | Q001-Q020 | 19 | MAP | 0.103 | 0.0541 ✗ | 0.0489 ✗ |

✗ = differs from the paper at 3 d.p. – = not computed (protocol not applicable, or 0 evaluated queries in the subset).

## BM25 tie order

BM25 scores tie often (short summaries). Phase 1 orders ties with `np.argsort` (unstable). Metrics under three tie orders:

| Condition | Tie order | P@1 | P@3 | P@5 | P@10 | nDCG@5 | nDCG@10 | MRR | MAP |
|---|---|---|---|---|---|---|---|---|---|
| bm25_baseline_t5 | argsort_default | 0.3377 | 0.3333 | 0.2805 | 0.2104 | 0.2989 | 0.2461 | 0.5056 | 0.0878 |
| bm25_baseline_t5 | ties_high_index_first | 0.3377 | 0.3377 | 0.2779 | 0.2104 | 0.2977 | 0.2464 | 0.5056 | 0.0881 |
| bm25_baseline_t5 | ties_low_index_first | 0.3377 | 0.3377 | 0.2805 | 0.2104 | 0.2990 | 0.2461 | 0.5056 | 0.0879 |
| bm25_rrf | argsort_default | 0.3377 | 0.3333 | 0.2831 | 0.2169 | 0.3000 | 0.2504 | 0.5021 | 0.0754 |
| bm25_rrf | ties_high_index_first | 0.3377 | 0.3290 | 0.2805 | 0.2169 | 0.2980 | 0.2501 | 0.5021 | 0.0752 |
| bm25_rrf | ties_low_index_first | 0.3377 | 0.3290 | 0.2805 | 0.2169 | 0.2980 | 0.2500 | 0.5021 | 0.0750 |
| rrf | argsort_default | 0.8961 | 0.7706 | 0.6753 | 0.5636 | 0.7268 | 0.6360 | 0.9306 | 0.2638 |
| rrf | ties_high_index_first | 0.8961 | 0.7706 | 0.6727 | 0.5636 | 0.7251 | 0.6359 | 0.9306 | 0.2637 |
| rrf | ties_low_index_first | 0.8961 | 0.7706 | 0.6779 | 0.5636 | 0.7289 | 0.6364 | 0.9306 | 0.2637 |

## Failures and caveats
- All numbers here score against the Phase 1 labels, including the MiniLM auto-labels for Q021–Q100. They are not an unbiased measure; `phase2/gold_eval_rebuild` (uatie-edm-phase1 repo) replaces those labels.
- The recency score depends on the current year (engine uses `datetime.now().year`).
