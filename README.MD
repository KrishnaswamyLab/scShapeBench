# scShapeBench

scShapeBench is a benchmark for evaluating graph extraction methods on single-cell shape data.

The repository now has two tracks:

- **Real data**: classify expert-annotated scRNA-seq graphs using fixed downstream models.
- **Synthetic data**: score recovered graphs against ground-truth graph structure and persistence.

The benchmark is method-agnostic; just add todes and edges csv, and optionally h0 and h1 persistence

---

## Overview

Real-data evaluation asks whether a method's graph makes expert shape labels predictable.

Synthetic evaluation asks whether a method recovers the underlying ground-truth graph, using:

- graph edit distance as the primary topology score
- persistence similarity as the primary persistence score
- additional diagnostics for debugging and analysis

---

## Installation

Python 3.10–3.12.

```bash
git clone https://github.com/KrishnaswamyLab/scShapeBench
cd scShapeBench
uv sync --extra methods          # or: pip install -e ".[methods]"
source .venv/bin/activate
```

The core install is enough to *evaluate* submissions. The `methods` extra adds the
graph-extraction baselines (scReebTower, Scanpy PAGA, KeplerMapper); scReebTower is
installed from
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
  16 of the 79 labelled datasets (including `SCD-0074` in the sample) have no
  `X_pca`; the builders skip them (see [docs/running_on_hpc.md](docs/running_on_hpc.md#datasets-without-x_pca)).
  Expert labels ship in this repo under `benchmarks/scrnaseq/labels/`. The
  commands below use the `high_support` label set; always pass `--label-type`,
  because the two real-track entrypoints default to different label sets.

---

## Tutorial: benchmark a kNN graph

This walks through the whole real track on the 9-dataset sample with the simplest
possible method: a k-nearest-neighbour graph (k = 2) over the cells. Run every
command from the repository root.

**1. Install and download the sample** (2.6 GB):

```bash
uv sync --extra methods && source .venv/bin/activate
python scripts/download_data.py --subset sample
```

**2. Build the graphs.** [`examples/knn_submission.py`](examples/knn_submission.py)
makes one node per cell, joins each cell to its 2 nearest neighbours in
`obsm["X_pca"]`, and writes one folder per dataset:

```bash
python examples/knn_submission.py --data-dir data/sample --output-dir submissions/scRNAseq/knn2
```

```text
SCD-0002: 8368 nodes, 13760 edges
...
SCD-0071: 18890 nodes, 33138 edges
SCD-0074: skipped (no obsm['X_pca'])
```

Each folder holds `nodes.csv`, `edges.csv`, `h0_finite.csv` and `h1_essential.csv`
(see [Real-data submission format](#real-data-submission-format)). To benchmark your
own method, replace the block marked *your method* in the script: it only needs
node positions and a list of `(source, target, length)` edges.

**3. Score them** (about 2 minutes on a laptop CPU):

```bash
python run_benchmark.py --track real --real-mode single \
  --submission submissions/scRNAseq/knn2 \
  --label-type high_support --splits splits/folds_v1.json \
  --output results/knn2 --device cpu
```

Each fixed model (`gnn`, `mlp`, `pi_svm`, `stats_svm`, `rf`) and four graph-free
baselines are trained and tested over the predefined folds. The console prints
mean ± std per shape class (`CLUSTERS`, `SINGLE_TRAJECTORY`, `MULTI_BRANCHING`,
`ARCHETYPAL`), and `results/knn2/results_<timestamp>.json` stores the same scores
plus per-fold values; the overall metrics are under
`results → <model> → __summary__` (`macro_auprc_mean`, `macro_auc_mean`,
`macro_balanced_accuracy_mean`).

**4. Compare against the baselines** (optional, about 4 minutes to build):

```bash
python scripts/build_real_submissions.py --data-dir data/sample --output-dir submissions/scRNAseq
python run_benchmark.py --track real --real-mode all \
  --submissions-dir submissions/scRNAseq \
  --label-type high_support --splits splits/folds_v1.json \
  --output results/all --device cpu
```

This scores every method folder (`knn2`, `paga`, `mapper`, `screebtower`) and also
writes `summary_high_support.csv` and LaTeX tables (see [Outputs](#outputs)).

> **The sample is for checking that everything runs, not for comparing methods.**
> It covers 8 of the 79 labelled datasets, so the scorer warns about missing labels
> and skips fold 3, whose only test dataset is `SCD-0074`. The results JSON records
> this under `metadata` (`n_evaluable_datasets`, `folds_evaluated`, `folds_skipped`).
> For real numbers, build graphs for all labelled datasets
> (`python scripts/download_data.py --subset scRNAseq`).

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

The fifth baseline, `gtda_mapper`, needs giotto-tda, which is not a project dependency:
its only Python 3.12 release pins scikit-learn 1.3.2, which would change the real
track's SVM scores. Run the synthetic baselines from a separate environment:

```bash
uv venv .venv-synthetic --python 3.12
uv pip install --python .venv-synthetic -e ".[methods]" giotto-tda==0.6.2
source .venv-synthetic/bin/activate
python run_benchmark_synthetic.py run-methods --dataset-dir data/synthetic \
  --methods screebtower screebtower_smooth paga mapper gtda_mapper --output-dir results/methods
```

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

### Reproducing the paper's real-track table

The paper's per-class table (GNN and MLP, Acc and F1 for Cl / ST / MT / A) uses the
**`union`** label set. Score the submissions with `--label-type union`, then print the
table in the paper's layout:

```bash
python run_benchmark.py --track real --real-mode all \
  --submissions-dir submissions/scRNAseq --label-type union \
  --splits splits/folds_v1.json --output results/union --device cpu
python scripts/paper_table.py results/union/summary_union.csv --methods mapper paga
```

```text
GNN MLP
Cl ST MT A Cl ST MT A
Method Acc F1 Acc F1 Acc F1 Acc F1 Acc F1 Acc F1 Acc F1 Acc F1
mapper .60±.07 .71±.06 ... .65±.08 .78±.07 .64±.09 .75±.08 .92±.07 .96±.04 .73±.11 .84±.08
```

Add `--models gnn mlp pi_svm stats_svm rf` for every model, or `--format markdown`.

What reproduces exactly, using the paper's graph files:

- **MLP, RF, baselines:** identical to the paper.
- **PI-SVM, Stats-SVM:** identical Acc/F1; AUC/AUPRC within 0.01, depending on the scikit-learn version.
- **GNN:** within about ±0.05, because GNN training is not bit-reproducible.

`tests/test_paper_reproduction.py` checks all of this:

```bash
uv run --frozen --extra methods --with pytest pytest -m "not slow"   # persistence files (seconds)
uv run --frozen --extra methods --with pytest pytest                  # + 10-graph regression (minutes)
SCSB_PAPER_SUBMISSIONS=/path/to/paper/submissions/scRNAseq \
  uv run --frozen --extra methods --with pytest pytest                # + full paper table
```

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


## Contact

For questions or issues, open a GitHub issue.

---
