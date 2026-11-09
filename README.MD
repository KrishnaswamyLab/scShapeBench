# scShapeBench

scShapeBench is a benchmark for evaluating graph extraction methods on single-cell shape data.

The repository now has two tracks:

- **Real data**: classify expert-annotated scRNA-seq graphs using fixed downstream models.
- **Synthetic data**: score recovered graphs against ground-truth graph structure and persistence.

The benchmark is method-agnostic. You provide graphs; scShapeBench evaluates them with a fixed pipeline.

---

## Overview

Real-data evaluation asks whether a method's graph makes expert shape labels predictable.

Synthetic evaluation asks whether a method recovers the contracted ground-truth graph, using:

- graph edit distance as the primary topology score
- persistence similarity as the primary persistence score
- additional diagnostics for debugging and analysis

---

## Installation

Python 3.10–3.12 (giotto-tda, used by the method baselines, has no 3.13 build).

```bash
git clone https://github.com/KrishnaswamyLab/scShapeBench
cd scShapeBench
uv sync --extra methods          # or: pip install -e ".[methods]"
source .venv/bin/activate
```

The core install is enough to *evaluate* submissions. The `methods` extra adds the
graph-extraction baselines (scReebTower, Scanpy PAGA, KeplerMapper, giotto-tda Mapper);
scReebTower (formerly dReeb) is installed from
[KrishnaswamyLab/scReebTower](https://github.com/KrishnaswamyLab/scReebTower).

---

## Data

All data is on Hugging Face at
[`scShape-Benchmark/scShapeBench`](https://huggingface.co/datasets/scShape-Benchmark/scShapeBench).
`scripts/download_data.py` keeps the repo paths, so files land under `./data/`:

```bash
python scripts/download_data.py                              # synthetic set → data/synthetic (46 MB)
python scripts/download_data.py --ids SCD-0010 SCD-0014      # chosen scRNAseq datasets → data/scRNAseq
python scripts/download_data.py --subset sample              # 9-dataset subset → data/sample (2.6 GB)
python scripts/download_data.py --subset scRNAseq            # all 105 datasets (~116 GB)
```

- **Synthetic**: 1000 samples, `sample_00000`–`sample_00999`, each a `.npz` (graph,
  positions, clean/noisy point cloud) plus a `.json` metadata sidecar. No
  `manifest.jsonl` is needed; it is rebuilt from the sidecars when absent.
- **Real**: one `.h5ad` per dataset with the cell embedding in `obsm["X_pca"]`.
  Expert labels ship in this repo under `benchmarks/scrnaseq/labels/`; the
  `high_support` label set is the default.

---

## Real-data submission format

One directory per method, one subdirectory per dataset:

```text
submissions/scRNAseq/
├── my_method/
│   ├── SCD-0001/
│   │   ├── nodes.csv          # node,x,y        (x, y are read; the id column name is free)
│   │   ├── edges.csv          # source,target,length
│   │   ├── h0_finite.csv      # optional: death column
│   │   └── h1_essential.csv   # optional: birth column
│   └── SCD-0002/ ...
└── another_method/ ...
```

- `source`/`target` index rows of `nodes.csv` (0-based). Extra columns are ignored.
- Folder names like `1` or `SC-001` are normalized to `SCD-0001`.
- Graphs without a label (and labels without a graph) are skipped with a warning.

`scripts/build_real_submissions.py` writes this layout for scReebTower, PAGA and Mapper
from the `.h5ad` files.

---

## Synthetic submission format

One JSON per sample named `sample_XXXXX.json` (or `XXXXX.json`), with:

- `nodes`: list of node ids
- `edges`: list of `[u, v]` pairs
- `node_positions`: `{node_id: [x, y, ...]}` (or a list aligned with `nodes`)

Every sample in the dataset needs a prediction; `evaluate` stops at the first missing one.

---

## Running the benchmark

### Real track

```bash
# 1. build graphs (skip if you bring your own submissions)
python scripts/build_real_submissions.py --data-dir data/scRNAseq --output-dir submissions/scRNAseq

# 2. evaluate every method folder
python run_benchmark.py --track real --real-mode all \
  --submissions-dir submissions/scRNAseq \
  --label-type high_support --splits splits/folds_v1.json \
  --output results/real --device cpu

# or one method
python run_benchmark.py --track real --real-mode single \
  --submission submissions/scRNAseq/paga \
  --label-type high_support --splits splits/folds_v1.json \
  --output results/real_paga --device cpu
```

### Synthetic track

Run the baselines and score them in one step:

```bash
python run_benchmark_synthetic.py run-methods \
  --dataset-dir data/synthetic \
  --methods screebtower paga mapper \
  --output-dir results/methods
```

Add `--limit N` for a quick check. Outputs are cached per sample, so reruns are cheap.

Score your own predictions:

```bash
python run_benchmark.py --track synthetic evaluate dataset \
  --dataset-dir data/synthetic \
  --predictions-dir my_predictions \
  --output-path results/synthetic_eval.json
```

Other synthetic commands: `generate` (make a new dataset), `evaluate sample`
(score one prediction), `run-swept-methods` (the default; `run-methods` with all
five baselines).

---

## Outputs

### Real track

`--real-mode all` writes the files below; `--real-mode single` writes only `results_*.json`.

- `summary_<label_type>.csv`
- `table_<label_type>.txt`
- `table_<label_type>_baselines.txt`
- `results_*.json`

Primary real-data metrics are:

- macro AUPRC
- macro AUC
- macro balanced accuracy

Secondary metrics include:

- macro F1
- exact match
- per-class scores

### Synthetic track

Synthetic dataset evaluation writes:

- a JSON report
- a LaTeX table sidecar with the headline scores
- optional manifold diagnostics

The paper-facing headline synthetic metrics are:

- graph edit distance
- persistence similarity

The synthetic method benchmark also writes:

- `results.csv`
- `summary.csv`
- `summary.tex`
- optional PDF figures

---

## Reproducibility

scShapeBench enforces:

- fixed splits on the real track
- fixed downstream models
- deterministic evaluation where possible
- explicit baselines on the real track
- strict graph/label ID alignment
- no training or tuning inside submissions

---

## Repository layout

```text
src/scshapebench/
├── evaluator/        # real-data evaluation backend
├── synthetic/        # synthetic generation and evaluation
├── models/           # fixed evaluation models
├── features/         # graph feature extraction
└── datasets/         # label loading and dataset helpers
```

Top-level entrypoints:

- `run_benchmark.py`
- `run_benchmark_all.py`
- `run_benchmark_single_cell.py`
- `run_benchmark_synthetic.py`

---

## Citation

If you use scShapeBench, please cite the paper associated with this repository.

---

## Common pitfalls

- missing graph files
- mismatched sample IDs
- malformed CSV columns
- using the wrong track or label type
- including training logic in submissions

---

## Contact

For questions or issues, open a GitHub issue.

---
