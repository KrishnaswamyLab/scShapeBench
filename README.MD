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

Users must provide a **submission directory** with the following structure:

```
my_submission/
├── metadata.json
└── graphs/
    ├── sample_000.pt
    ├── sample_001.pt
    └── ...
```

---

### `metadata.json`

```json
{
  "method_name": "my_graph_method",
  "dataset_version": "v1",
  "notes": "optional description"
}
```

---

### Graph files (`.pt`)

Each file corresponds to a dataset sample and must contain a graph in PyTorch format:

```python
{
    "edge_index": LongTensor [2, E],
    "x": FloatTensor [N, D],        # optional node features
    "edge_attr": FloatTensor [E, D] # optional edge features
}
```

**Requirements:**

* One graph per sample
* File names must match dataset sample IDs
* No missing samples
* No NaNs or invalid values

---

## 🚀 Running the Benchmark

Install the package:

```bash
pip install -e .
```

Run evaluation:

```bash
python scripts/run_benchmark.py \
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
