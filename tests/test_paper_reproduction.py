"""Guard the real-track results that the paper reports.

Three layers, fastest first:

1. ``test_write_persistence_reproduces_paper_files``: rebuilding the persistence files
   from the paper's paga/mapper graphs gives exactly the files the paper used.
2. ``test_ten_graphs_match_old_code``: scoring 10 of the paper's mapper/paga graphs
   (tests/fixtures/paper_graphs, two test datasets per fold of splits/folds_v1.json)
   gives the same numbers as the paper-era code (commit d81963f, ``main``) did.
   Expected values: tests/expected/old_code_10graphs_<label_type>.csv.
3. ``test_paper_table_*``: scoring ALL of the paper's mapper/paga graphs with union
   labels reproduces the paper's GNN/MLP table (tests/expected/paper_union_gnn_mlp.txt).
   Needs the full graph folders, so it only runs when SCSB_PAPER_SUBMISSIONS points to a
   directory holding ``mapper/`` and ``paga/`` (for example the original submissions on
   the cluster); otherwise it is skipped.

Ten graphs cannot reproduce the paper's numbers themselves (those are means over all
labelled datasets), which is why layer 2 compares with the old code and layer 3 with the
paper. Tolerances are explicit and listed in TOLERANCES:
GNN is not bit-reproducible, so its exact comparison is marked xfail rather than loosened.
"""

from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "paper_graphs"
EXPECTED = ROOT / "tests" / "expected"
METHODS = ["mapper", "paga"]

sys.path.insert(0, str(ROOT / "scripts"))
from build_real_submissions import write_persistence  # noqa: E402
from paper_table import load_summary, paper_rows  # noqa: E402

# Allowed |current - expected| per (model, metric); anything not listed must match exactly.
# SVM probability estimates (AUC/AUPRC only) differ between scikit-learn releases.
SVM_PROBABILITY_METRICS = {"auc", "auprc", "macro_auc", "macro_auprc", "micro_auc", "micro_auprc"}
TOLERANCES = {(model, metric): 0.01 for model in ("pi_svm", "stats_svm") for metric in SVM_PROBABILITY_METRICS}
EXACT = 1e-9
GNN_REASON = "GNN training is not bit-reproducible across environments"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _score(submissions_dir: Path, label_type: str, output: Path) -> Path:
    """Run the real-track scorer exactly as a user would; return the summary CSV."""
    subprocess.run(
        [
            sys.executable, str(ROOT / "run_benchmark.py"), "--track", "real", "--real-mode", "all",
            "--submissions-dir", str(submissions_dir), "--methods", *METHODS,
            "--label-type", label_type, "--splits", str(ROOT / "splits" / "folds_v1.json"),
            "--output", str(output), "--device", "cpu",
        ],
        cwd=ROOT, check=True,
    )
    return output / f"summary_{label_type}.csv"


# ---------------------------------------------------------------------------------------
# 1. Persistence files
# ---------------------------------------------------------------------------------------

FIXTURE_GRAPHS = sorted((m, p.name) for m in METHODS for p in (FIXTURES / m).iterdir() if p.is_dir())


@pytest.mark.parametrize("method,sample_id", FIXTURE_GRAPHS)
def test_write_persistence_reproduces_paper_files(tmp_path, method, sample_id):
    src = FIXTURES / method / sample_id
    n_nodes = len(_read_csv(src / "nodes.csv"))
    edges = [
        (int(r["source"]), int(r["target"]), float(r["length"]) if r["length"] else 0.0)
        for r in _read_csv(src / "edges.csv")
    ]
    write_persistence(tmp_path, list(range(n_nodes)), edges)

    for name, column in (("h0_finite.csv", "death"), ("h1_essential.csv", "birth")):
        ours = sorted(float(r[column]) for r in _read_csv(tmp_path / name))
        paper = sorted(float(r[column]) for r in _read_csv(src / name))
        assert len(ours) == len(paper), f"{name}: {len(ours)} points, paper has {len(paper)}"
        assert ours == pytest.approx(paper, rel=1e-9, abs=1e-12), name


# ---------------------------------------------------------------------------------------
# 2. Ten graphs vs the paper-era code
# ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module", params=["union", "high_support"])
def ten_graph_run(request, tmp_path_factory):
    label_type = request.param
    summary = _score(FIXTURES, label_type, tmp_path_factory.mktemp(f"ten_{label_type}"))
    return label_type, _read_csv(summary)


def _compare(current_rows, expected_rows, models):
    current = {(r["method"], r["model"], r["class"], r["metric"]): r for r in current_rows}
    failures = []
    for row in expected_rows:
        if row["model"] not in models:
            continue
        key = (row["method"], row["model"], row["class"], row["metric"])
        if key not in current:
            failures.append(f"{key}: missing from current results")
            continue
        tol = TOLERANCES.get((row["model"], row["metric"]), EXACT)
        for stat in ("mean", "std"):
            want, got = float(row[stat]), float(current[key][stat])
            if math.isnan(want) and math.isnan(got):
                continue
            if not abs(got - want) <= tol:
                failures.append(f"{key} {stat}: got {got:.6f}, old code {want:.6f} (tol {tol:g})")
    return failures


NON_GNN_MODELS = {
    "mlp", "pi_svm", "stats_svm", "rf",
    "baseline_all_zero", "baseline_all_one", "baseline_train_prevalence", "baseline_random_prevalence",
}


@pytest.mark.slow
def test_ten_graphs_match_old_code(ten_graph_run):
    label_type, rows = ten_graph_run
    expected = _read_csv(EXPECTED / f"old_code_10graphs_{label_type}.csv")
    failures = _compare(rows, expected, NON_GNN_MODELS)
    assert not failures, "\n".join(failures)


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason=GNN_REASON)
def test_ten_graphs_gnn_match_old_code(ten_graph_run):
    label_type, rows = ten_graph_run
    expected = _read_csv(EXPECTED / f"old_code_10graphs_{label_type}.csv")
    failures = _compare(rows, expected, {"gnn"})
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------------------
# 3. Full mapper/paga graphs vs the paper's table
# ---------------------------------------------------------------------------------------

PAPER_SUBMISSIONS = os.environ.get("SCSB_PAPER_SUBMISSIONS")


def _paper_table() -> dict[str, list[str]]:
    """method -> 8 GNN cells followed by 8 MLP cells (Acc, F1 for Cl, ST, MT, A), as printed in the paper."""
    table = {}
    for line in (EXPECTED / "paper_union_gnn_mlp.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            method, *cells = line.split()
            assert len(cells) == 16, f"{method}: expected 16 cells, found {len(cells)}"
            table[method] = cells
    return table


@pytest.fixture(scope="module")
def full_union_rows(tmp_path_factory):
    if not PAPER_SUBMISSIONS:
        pytest.skip("set SCSB_PAPER_SUBMISSIONS to a folder with the paper's mapper/ and paga/ graphs")
    summary = _score(Path(PAPER_SUBMISSIONS), "union", tmp_path_factory.mktemp("paper_union"))
    return {row[0]: row[1:] for row in paper_rows(load_summary(summary), METHODS, ["gnn", "mlp"])}


def _mismatches(ours: dict, model_slice: slice) -> list[str]:
    paper = _paper_table()
    out = []
    for method in METHODS:
        got_cells, want_cells = ours[method][model_slice], paper[method][model_slice]
        assert len(got_cells) == len(want_cells) == 8, f"{method}: compared {len(got_cells)} cells, expected 8"
        for i, (got, want) in enumerate(zip(got_cells, want_cells)):
            if got != want:
                out.append(f"{method} cell {i}: got {got}, paper {want}")
    return out


@pytest.mark.slow
def test_paper_table_mlp(full_union_rows):
    mismatches = _mismatches(full_union_rows, slice(8, 16))
    assert not mismatches, "\n".join(mismatches)


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason=GNN_REASON)
def test_paper_table_gnn(full_union_rows):
    mismatches = _mismatches(full_union_rows, slice(0, 8))
    assert not mismatches, "\n".join(mismatches)
