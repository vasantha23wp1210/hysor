# HYSOR: paper reproduction package

This folder holds every script, data file and result behind the paper *HySOR: Hybrid Semantic-Ontological Retrieval
for Aviation Safety Incident Analysis over Fused ASRS-NTSB Data*. The Phase 1 files come from
[uatie-edm-phase1](https://github.com/vasantha23wp1210/uatie-edm-phase1), copied unmodified at the same relative
paths, so each script runs from this repo's root exactly as it did there.

## Reproduce in one command
```
python reproduce.py
```
Run it from the repo root. It takes about 10 minutes on CPU. It is read-only on all inputs and writes:
- `REPORT.md` / `results.json`: corpus, NLP, FAISS, ontology, Table 8–9 and Fig. 2–3 numbers, paper vs. measured.
- `recheck/REPORT.md` / `recheck/results.json`: Tables 5, 6, 10, Sect. 5.7 and the RRF numbers, recomputed with
  the `src/hysor_rrf.py` evaluation (same gold loader and metrics as `src/hysor_ablation.py`). For comparison it also
  replays the published Table 5 protocol from `src/rag/bm25_baseline.py`. Per-query metrics are in `recheck/output/`.
- `verify/claims.csv`: one row per checked number.
- `_repro/evaluation/figures/`: Figures 2 and 3, regenerated.

## Setup (Python 3.11, CPU only)
```
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install -r requirements.txt
```
The 21 `kind=data` files in `DATA_MANIFEST.csv` (~1.1 GB, under `data/`) are stored with Git LFS: run `git lfs pull`.
If they are missing, copy them from a uatie-edm-phase1 checkout (after `git lfs pull` there); each SHA-256 is checked:
```
python tools/copy_data.py --from <path-to-uatie-edm-phase1>
```

## Run the HySOR app (Sect. 4)
```
streamlit run src/HySor_App.py --server.fileWatcherType none
```
Run it from the repo root and open http://localhost:8501. The backend (FAISS + the 989k-triple TTL) takes about a
minute to load. `--server.fileWatcherType none` only silences Streamlit's file-watcher warnings about the optional
`torchvision` module in `transformers`.

## Layout
| Path | Contents |
|---|---|
| `data/raw/` | ASRS, NTSB, Kaggle and UK CAA source files (external; check licences before a public release) |
| `data/processed/` | fused corpus, annotated corpus, FAISS index, ontology TTL files, phraseology crossmap |
| `src/` | Phase 1 pipeline, engine, baselines, ablation, RRF, scenario analysis and plotting scripts |
| `evaluation/` | gold queries, relevance labels, stored rankings, Table 5 results, figures |
| `output/`, `metrics/` | late-runway-change scenario outputs (Tables 8–9) and global/scenario metrics |
| `ablation_results.csv`, `rrf_results.csv`, `src/sanity_check_results.csv` | Table 10 and RRF outputs as committed |
| `verify.py`, `recheck/run.py`, `reproduce.py` | the checks |
| `MANIFEST.md` | paper item → script → inputs → outputs, plus known gaps |
| `DATA_MANIFEST.csv` | SHA-256, size, producing script and redistribution note for each of the 88 input files |

`.gitattributes` routes `data/raw/**` and `data/processed/**` through Git LFS.

## Rebuilding from raw
`MANIFEST.md` lists the scripts in run order. Run each one from the repo root. They overwrite files in `data/processed/`,
`evaluation/` and `output/`, so afterwards check `DATA_MANIFEST.csv` hashes with `python verify.py`.
`reproduce.py` does not run this rebuild, and it has not been tested end to end.
