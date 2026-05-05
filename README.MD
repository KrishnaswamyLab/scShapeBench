# scShapeBench

**scShapeBench** is a benchmark for evaluating **graph extraction methods** on single-cell shape datasets.

Unlike typical machine learning benchmarks, scShapeBench **does not evaluate end-to-end models**. Instead, it evaluates the *quality of graph representations* produced by different methods under a fixed, standardized downstream pipeline.

---

## 🔍 Overview

Given a dataset of samples, users apply their own **graph extraction method** to produce a graph for each sample.
scShapeBench then evaluates these graphs using:

* Fixed train/test splits
* Standard downstream models (MLP, SVM, GNN)
* Standard classification metrics (Accuracy, F1, AUC, Precision, Recall)

This ensures that **performance differences reflect graph quality, not training tricks**.

---

## 🧠 Benchmark Protocol

The benchmark implements the following function:

```
graphs → scores
```

Where:

* **Input**: a set of graphs (one per dataset sample)
* **Output**: classification performance across predefined tasks

Everything else is fixed and controlled by the benchmark.

---

## 📦 Submission Format

Users must provide a **submission directory** with one subdirectory per dataset:

```
my_submission/
├── SCD-0001/
│   ├── nodes.csv
│   ├── edges.csv
│   ├── h0_finite.csv       # optional
│   └── h1_essential.csv    # optional
├── SCD-0002/
│   ├── nodes.csv
│   └── edges.csv
└── ...
```

Sample directories may also be named with trailing integers such as `1` or
`SC-001`; the loader normalizes them to `SCD-0001`.

**Requirements:**

* One graph per sample
* Sample IDs must match benchmark dataset IDs after normalization
* No NaNs or invalid values
* `nodes.csv` columns: `id,x,y`
* `edges.csv` columns: `source,target,length`
* `h0_finite.csv` column: `death`, if persistence features are supplied
* `h1_essential.csv` column: `birth`, if persistence features are supplied

---

## 📥 Data

Datasets are hosted on HuggingFace. Download them before running the benchmark:

```bash
pip install -e .
python scripts/download_data.py --local-dir data/
```

This will populate `data/` with the raw cell shape samples. You can also specify a custom repo:

```bash
python scripts/download_data.py --repo-id your-username/scShapeBench --local-dir data/
```

---

## 🚀 Running the Benchmark

Install the package:

```bash
pip install -e .
```

Run evaluation on scRNA-seq data:

```bash
python run_benchmark_single_cell.py \
    --submission path/to/submission \
    --label-type confidence_score \
    --splits splits/folds_v1.json \
    --output results/
```

Run evaluation on synthetic data:

```bash
python run_benchmark_synthetic.py \
    --submission path/to/submission \
    --output results/
```

Optional:

```bash
--device cuda
```

---

## 📊 Output

Results are saved as JSON:

```json
{
  "metadata": {...},
  "results": {
    "mlp": {
      "accuracy_mean": 0.85,
      "f1_mean": 0.83,
      "auc_mean": 0.90
    },
    "gnn": {
      ...
    }
  }
}
```

All metrics are averaged across **fixed cross-validation splits**.

---

## 📁 Repository Structure

```
src/scshapebench/
├── evaluator/        # benchmark logic (runner, metrics, validation)
├── models/           # fixed evaluation models (MLP, SVM, GNN)
├── features/         # graph → feature extraction (for MLP/SVM)
├── schemas/          # graph format definition
```

Additional directories:

```
splits/               # predefined train/test splits
scripts/              # CLI entrypoints
examples/             # example submissions
```

---

## 🔒 Reproducibility

scShapeBench enforces:

* Fixed dataset splits
* Fixed evaluation models
* Deterministic evaluation pipeline
* Explicit always-negative, always-positive, train-prevalence, and random-prevalence baselines
* Strict graph/label ID alignment: unlabeled graphs and graphless labels are excluded from scoring and reported

Users **cannot modify training or evaluation procedures**.

---

## ❗ Design Principles

* **Method-agnostic**: users can use any graph extraction approach
* **Fair comparison**: identical downstream evaluation
* **Minimal interface**: only graphs are required
* **Strict validation**: malformed submissions are rejected

---

## 🧪 Example

See:

```
examples/dummy_submission/
```

for a minimal working submission.

---

## 📄 Citation

If you use scShapeBench, please cite:

```
@article{scshapebench2026,
  title={scShapeBench: A Benchmark for Graph Extraction Methods on Single-Cell Shape Data},
  author={...},
  year={2026}
}
```

---

## 🤝 Contributing

We welcome:

* New datasets
* Additional evaluation tasks
* Improvements to validation and metrics

Please open an issue or pull request.

---

## ⚠️ Common Pitfalls

* Missing graph files
* Mismatched sample IDs
* Incorrect tensor shapes
* Including training logic in submissions

---

## 📬 Contact

For questions or issues, please open a GitHub issue.

---
