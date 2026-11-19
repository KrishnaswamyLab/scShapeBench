# Running the real-data benchmark on an HPC cluster

This guide runs the real (scRNA-seq) track on the full labelled data, from a fresh
clone to a results table, and the [synthetic track](#synthetic-track) at the end. It uses Slurm; the [last section](#other-schedulers) maps the
few Slurm-specific lines to PBS and LSF.

The worked example scores three methods:

- `knn2`: the tutorial k-nearest-neighbour graph (k = 2, one node per cell), built here.
- `dreeb_k25`, `dreeb_k50`: graphs made earlier by another pipeline, imported as-is.

Replace either with your own method: build graphs from the `.h5ad` files (step 4) or
import graphs you already have (step 5).

## What runs where

| Step | Where | Cost on the reference run |
|---|---|---|
| 1. Install | login node (needs internet) | 9 min, 5.3 GB environment |
| 2. Download data | login or data-transfer node (needs internet) | 67 GB for the labelled datasets |
| 3. Make the dataset list | login node | seconds |
| 4. Build graphs | Slurm array, one task per dataset | kNN: ≤ 30 s and ≤ 0.6 GB per task |
| 5. Import existing graphs | small Slurm job | minutes |
| 6. Score | one Slurm job | 52 min, 2.7 GB on 8 CPUs for three methods |
| 7. Collect results | your laptop | 1 MB |

Scoring time is dominated by graphs with many nodes (one node per cell); methods that
produce a few hundred nodes per dataset score in about a minute.

## 0. Pick locations

Clusters usually have a small backed-up home, a larger project space and a purged
scratch space. Decide on three paths before you start:

```bash
export REPO=$HOME/scShapeBench              # code, environment, graphs, results (small)
export DATA_ROOT=/path/to/large/storage     # raw .h5ad files (67 GB, re-downloadable)
export HF_HOME=$DATA_ROOT/.cache/huggingface   # keep the Hugging Face cache out of home
```

Scratch is fine for `DATA_ROOT` because the files can be downloaded again, but check
its purge policy (often 30–60 days). Put the lines above in `~/.bashrc` so batch jobs
and new shells see them.

## 1. Install (login node)

Compute nodes often have no internet access, so install on a login node.
[uv](https://docs.astral.sh/uv/) installs in your home directory and needs no admin rights:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # → ~/.local/bin/uv
git clone https://github.com/KrishnaswamyLab/scShapeBench "$REPO"
cd "$REPO"
uv sync --extra methods                             # Python 3.12 + all baselines
mkdir -p logs                                       # Slurm writes job logs here
```

- `uv sync` downloads its own Python 3.12; your cluster's Python module is not used.
  Shared filesystems make it slow (about 9 minutes); that is normal.
- If home has a tight quota, set `UV_CACHE_DIR` to a larger location first.
- Without uv: `module load` a Python 3.10–3.12, create a virtualenv or conda env, then
  `pip install -e ".[methods]"`.

## 2. Download the data from Hugging Face

The benchmark scores the datasets that have expert labels (79 in the `high_support` set).
Download only those instead of all 105 (116 GB):

```bash
cd "$REPO" && source .venv/bin/activate
python -c "import json; print('\n'.join(json.load(open('benchmarks/scrnaseq/labels/high_support_labels.json'))['dataset_order']))" > labelled_ids.txt
python scripts/download_data.py --ids $(cat labelled_ids.txt) --local-dir "$DATA_ROOT"
# files land in $DATA_ROOT/data/scRNAseq/SCD-XXXX.h5ad
```

Large downloads can take hours. Run them inside `tmux` or `screen`, on your cluster's
data-transfer node if it has one, so a dropped connection does not stop them. Rerunning
the same command resumes and skips finished files. Expect 79 files.

## 3. Make the dataset list

Array jobs read one dataset ID per line:

```bash
ls "$DATA_ROOT/data/scRNAseq" | sed -n 's/\.h5ad$//p' | grep -Fxf labelled_ids.txt > ids.txt
wc -l < ids.txt      # 79
```

### Datasets without `X_pca`

The builders use the cell embedding in `obsm["X_pca"]`. Sixteen labelled datasets don't
have one: `SCD-0069 0070 0072 0074 0077 0080 0081 0084 0091 0096 0097 0107 0108 0109 0110 0111`.
Most have only `X_umap` (some also `X_scVI`; `SCD-0091` has only `X_tsne`). The builders
print `skipped` for these and move on, so a method built this way is scored on **63**
datasets. To use another embedding, pass `--rep` to `build_real_submissions.py`, or
change the block marked *your method* in `examples/knn_submission.py`.

Keep this in mind when comparing methods: a method scored on 78 datasets and one scored
on 63 are not directly comparable (see [step 6](#compare-on-the-same-datasets)).

## 4. Build graphs: one array task per dataset

`scripts/hpc/build_knn.sbatch` runs `examples/knn_submission.py` for one dataset per task:

```bash
cd "$REPO"
sbatch --array=1-$(wc -l < ids.txt) \
  --export=ALL,DATA_DIR=$DATA_ROOT/data/scRNAseq \
  scripts/hpc/build_knn.sbatch
```

It writes `submissions/scRNAseq/knn2/SCD-XXXX/{nodes,edges,h0_finite,h1_essential}.csv`.
Set `K`, `OUT_DIR` or `IDS_FILE` with `--export` to change the defaults. Add your
cluster's account or partition flags (`-A`, `-p`) on the `sbatch` line, not in the script.

- **Check:** `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS`. Finished datasets are
  cached, so resubmitting the array only rebuilds what is missing.
- **Rerun failures:** `sbatch --array=<failed indices> --mem=128G -t 1-00:00:00 ...`
  (command-line flags override the script's `#SBATCH` lines).
- **Other baselines:** `scripts/hpc/build_baselines.sbatch` builds the scReebTower, PAGA
  and Mapper graphs the same way (one folder per method under `submissions/scRNAseq/`):

  ```bash
  sbatch --array=1-$(wc -l < ids.txt) \
    --export=ALL,DATA_DIR=$DATA_ROOT/data/scRNAseq \
    scripts/hpc/build_baselines.sbatch
  ```

## 5. Import graphs you already have

If another pipeline already produced graphs, `scripts/import_graphs.py` repackages them.
It keeps `nodes.csv` (`x`, `y`) and `edges.csv` (`source`, `target`, `length`; extra
columns are dropped) and **recomputes** `h0_finite.csv` and `h1_essential.csv` from the
edge lengths, using the same code as the builders. Persistence files from the other
pipeline are ignored, so every method's persistence features are computed the same way.

```bash
srun --time=00:30:00 --mem=8G bash -c '
  source .venv/bin/activate
  for m in dreeb_k25 dreeb_k50; do
    python scripts/import_graphs.py --src /path/to/other_pipeline/$m \
      --dst submissions/scRNAseq/$m --ids $(cat ids.txt)
  done'
```

The source layout is `<src>/SCD-XXXX/{nodes.csv,edges.csv}`, and `source`/`target`
must index rows of `nodes.csv` from 0.

> Recomputing persistence changes the persistence-based models (`pi_svm`, `mlp`) when the
> other pipeline scaled edge lengths differently. In the reference run the imported dreeb
> graphs gave exactly the same `stats_svm` and `rf` scores as the original pipeline's own
> scoring, which confirms the graphs themselves were carried over unchanged.

## 6. Score

`scripts/hpc/score.sbatch` scores every method folder under `submissions/scRNAseq`
with the fixed splits and the `high_support` labels:

```bash
sbatch scripts/hpc/score.sbatch
# or start it automatically once the build array succeeds:
sbatch --dependency=afterok:<build array jobid> scripts/hpc/score.sbatch
```

Results go to `results/real/`. The defaults run on CPU (8 cores, 64 GB, 1 day), which is
plenty: the reference run used 2.7 GB. For a GPU node add your cluster's flags and
`DEVICE=cuda`, e.g. `sbatch -p <gpu partition> --gpus=1 --export=ALL,DEVICE=cuda scripts/hpc/score.sbatch`.
GNN scores can differ slightly between CPU and GPU and between torch versions.

### Compare on the same datasets

Each method is scored on the datasets it has graphs for. To compare a 78-dataset method
with a 63-dataset one fairly, score it again on the shared subset. Symlinks avoid copying:

```bash
mkdir -p submissions_common
for m in dreeb_k25 dreeb_k50; do
  mkdir -p submissions_common/$m
  for id in $(ls submissions/scRNAseq/knn2); do
    ln -sfn "$REPO/submissions/scRNAseq/$m/$id" submissions_common/$m/$id
  done
done
sbatch --export=ALL,SUBMISSIONS_DIR=submissions_common,OUTPUT_DIR=results/real_common scripts/hpc/score.sbatch
```

## 7. Collect the results

```bash
# from your laptop
rsync -az <user>@<cluster>:<REPO>/results/ ./results_hpc/
```

Each output folder holds:

- `summary_high_support.csv`: one row per method × model × class × metric. Rows with
  `class = __summary__` hold the headline scores: `macro_auprc`, `macro_auc` and
  `macro_balanced_accuracy` (mean and std over the 5 folds).
- `table_high_support.txt` and `table_high_support_baselines.txt`: LaTeX tables.
- `results_<timestamp>.json`: per-fold scores, plus `metadata` listing how many datasets
  were evaluated and which folds (if any) were skipped.

### Reference numbers for `knn2`

For checking your own run: `knn2` (k = 2 on `X_pca`, 63 datasets, CPU), mean ± std over
5 folds.

| Model | macro AUPRC | macro AUC | macro balanced acc. |
|---|---|---|---|
| gnn | 0.510 ± 0.12 | 0.579 ± 0.13 | 0.609 ± 0.11 |
| mlp | 0.470 ± 0.10 | 0.547 ± 0.11 | 0.512 ± 0.08 |
| pi_svm | 0.507 ± 0.10 | 0.601 ± 0.08 | 0.514 ± 0.03 |
| stats_svm | 0.546 ± 0.06 | 0.651 ± 0.05 | 0.574 ± 0.10 |
| rf | 0.614 ± 0.08 | 0.717 ± 0.07 | 0.586 ± 0.03 |
| baseline (random, train prevalence) | 0.432 ± 0.09 | 0.548 ± 0.07 | 0.488 ± 0.05 |

`mlp`, `rf` and the baselines are deterministic and should match to three decimals; the
SVM AUC/AUPRC can move by up to 0.01 with the scikit-learn version, and `gnn` may move by
a few hundredths with hardware and torch version.

## Synthetic track

The synthetic track needs no cluster (1000 small point clouds, 46 MB), but the five
baseline methods take about 17 s per sample, so splitting it into array tasks turns
5 hours into about 30 minutes.

The `gtda_mapper` baseline needs giotto-tda, which pins scikit-learn 1.3.2, so the
synthetic jobs run from their own environment (`VENV`, default `.venv`):

```bash
cd "$REPO"
uv venv .venv-synthetic --python 3.12
uv pip install --python .venv-synthetic -e ".[methods]" giotto-tda==0.6.2
source .venv-synthetic/bin/activate
python scripts/download_data.py                 # → data/synthetic (login node)
export VENV=.venv-synthetic
A=$(sbatch --parsable --array=1-10 scripts/hpc/synthetic_methods.sbatch)   # 100 samples per task
sbatch --dependency=afterok:$A scripts/hpc/synthetic_summary.sbatch
```

Step 1 runs every method on its chunk and caches each prediction under
`results/synthetic/cache/<method>/pointcloud/`. Step 2 reads the cache and writes:

- `summary.tex`: graph edit distance (GED, lower is better) and persistence similarity
  (higher is better) per method, mean ± std over samples.
- `summary.csv`: the same means plus diagnostics (cycle and branch recovery, match rates).
- `results.csv`: one row per sample and method.
- `eval_<method>.json`: the full `evaluate dataset` report for each method, including the
  Wasserstein distances that make up persistence similarity
  (`exp(−(0.5·W₂(H0 deaths) + 0.3·W₂(H1 births) + 0.2·|ΔH1 count|))`).

To score your own synthetic predictions instead, see "Synthetic submission format" in the README
and `run_benchmark.py --track synthetic evaluate dataset`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Job fails at once, no log | The `logs/` folder must exist before `sbatch`: `mkdir -p logs`. |
| `skipped (no obsm['X_pca'])` | Expected for 16 datasets; see [step 3](#datasets-without-x_pca). |
| `Warning: excluding N label(s) with no graph` | Labelled datasets without a graph are dropped from scoring; the count is in the results `metadata`. |
| `Warning: skipping fold k` | That fold has no test dataset with a graph (happens with small subsets). |
| `OUT_OF_MEMORY` / `TIMEOUT` in `sacct` | Rerun those array indices with more `--mem` / `-t`. |
| Downloads or installs fail in a job | Compute nodes lack internet; run steps 1–2 on a login or transfer node. |
| Home quota exceeded | Move `UV_CACHE_DIR`, `HF_HOME` and the data to project or scratch space. |

## Other schedulers

Only a few lines in `scripts/hpc/*.sbatch` are Slurm-specific:

| Slurm | PBS / Torque | LSF |
|---|---|---|
| `#SBATCH --cpus-per-task=4 --mem=32G --time=04:00:00` | `#PBS -l select=1:ncpus=4:mem=32gb` and `#PBS -l walltime=04:00:00` | `#BSUB -n 4 -R "rusage[mem=32G]" -W 04:00` |
| `sbatch --array=1-79` | `qsub -J 1-79` (Torque: `qsub -t 1-79`) | `bsub -J "knn[1-79]"` |
| `$SLURM_ARRAY_TASK_ID` | `$PBS_ARRAY_INDEX` (Torque: `$PBS_ARRAYID`) | `$LSB_JOBINDEX` |
| `$SLURM_SUBMIT_DIR` | `$PBS_O_WORKDIR` | `$LS_SUBCWD` |
| `--dependency=afterok:<id>` | `-W depend=afterok:<id>` | `-w "done(<id>)"` |

Without any scheduler, run the same commands in a loop over `ids.txt`.
