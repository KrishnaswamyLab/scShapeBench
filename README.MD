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

```bash
pip install -e .
```

---

## Data

Real scRNA-seq annotations are loaded from the repository's benchmark data and are evaluated with the `high_support` label set by default.

Synthetic benchmark data is distributed separately as a release bundle and should be unpacked so the dataset directory contains:

- `manifest.jsonl`
- `dataset_config.json`
- `samples/*.npz`
- `samples/*.json`

The shipped synthetic bundle is the `new1000` release, reindexed to a contiguous `0..999` naming scheme.

---

## Real-data submission format

For the real scRNA-seq track, submit one graph per sample:

```text
my_submission/
├── SCD-0001/
│   ├── nodes.csv
│   ├── edges.csv
│   ├── h0_finite.csv      # optional
│   └── h1_essential.csv   # optional
├── SCD-0002/
│   ├── nodes.csv
│   └── edges.csv
└── ...
```

Requirements:

- one graph per sample
- sample IDs must match benchmark IDs after normalization
- `nodes.csv` columns: `id,x,y`
- `edges.csv` columns: `source,target,length`
- optional persistence files use one value per row:
  - `h0_finite.csv` with `death`
  - `h1_essential.csv` with `birth`

The loader accepts folder names like `1` or `SC-001` and normalizes them to `SCD-0001`.

---

## Synthetic submission format

For synthetic evaluation, submit one JSON file per sample, using the same graph payload schema as the generated benchmark data.

The evaluator reads:

- `nodes`
- `edges`
- `node_positions`

The rest of the benchmark bundle is used for truth loading and scoring.

---

## Running the benchmark

Use the unified dispatcher:

### Real track

All methods:

```bash
python run_benchmark.py \
  --track real \
  --real-mode all \
  --submissions-dir submissions/submissions/scRNAseq \
  --label-type high_support \
  --splits splits/folds_v1.json \
  --output results/high_support \
  --device cpu
```

Single submission:

```bash
python run_benchmark.py \
  --track real \
  --real-mode single \
  --submission submissions/submissions/scRNAseq/paga \
  --label-type high_support \
  --splits splits/folds_v1.json \
  --output results/high_support_single \
  --device cpu
```

### Synthetic track

Evaluate one prediction directory against the shipped dataset:

```bash
python run_benchmark.py \
  --track synthetic evaluate dataset \
  --dataset-dir /path/to/full_1000_swept_seed_20260422/dataset \
  --predictions-dir /path/to/predictions \
  --output-path results/synthetic_eval.json
```

Run the synthetic method benchmark:

```bash
python run_benchmark_synthetic.py run-methods \
  --dataset-dir /path/to/full_1000_swept_seed_20260422/dataset \
  --output-dir results/method_benchmark
```

---

## Outputs

### Real track

Real-data runs write:

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

## Example

See `examples/dummy_submission/` for a minimal real-data submission.

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
