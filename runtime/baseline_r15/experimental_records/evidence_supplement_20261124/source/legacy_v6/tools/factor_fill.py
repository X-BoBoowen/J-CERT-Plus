"""Count occupied CHOLMOD entries on the recorded 40-commit trajectories."""
import csv
import json
from pathlib import Path

from cholmod_backend import CholmodFactor
from engine import Edit
from workloads import NAMES, candidates, load_graph, system


def main():
    data = Path(__file__).resolve().parents[1] / "data"
    with (data / "factor_fill.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("graph","epoch","n","matrix_lower_nnz","factor_nnz","fill_ratio",
                         "native_current_bytes","native_peak_bytes"))
        for name in NAMES:
            W = load_graph(name)
            A = system(W)
            edits = [Edit(*item) for item in candidates(W, 2500, 17)]
            with (data / f"orthogonal_{name}_2500.csv").open() as source:
                row = next(r for r in csv.DictReader(source) if r["b"] == "40")
            trajectory = json.loads(row["sequence"])
            if len(trajectory) != 40:
                raise ValueError("expected a 40-commit trace")
            lower_nnz = (A.nnz + A.shape[0]) // 2
            with CholmodFactor(A) as factor:
                for t in range(41):
                    if t in (0,20,40):
                        stored = factor.stored_entries()
                        writer.writerow((name,t,A.shape[0],lower_nnz,stored,stored/lower_nnz,
                                         factor.bytes,factor.peak_bytes))
                        output.flush()
                    if t < 40:
                        edit = edits[trajectory[t]]
                        factor.change(edit)
                        lower_nnz += int(not edit.flip)


def root_profile():
    """Record root fill independently of catalogs, opinions, or query schedules."""
    data = Path(__file__).resolve().parents[1] / "data"
    rows = []
    for name in NAMES:
        A = system(load_graph(name))
        input_lower = (A.nnz + A.shape[0]) // 2
        with CholmodFactor(A) as factor:
            entries = factor.stored_entries()
            rows.append((name,A.shape[0],input_lower,entries,entries/input_lower))
    with (data/"root_fill.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("graph","n","matrix_lower_nnz","factor_nnz","fill_ratio"))
        writer.writerows(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="store_true", help="profile the six unevaluated root operators")
    if parser.parse_args().root:
        root_profile()
    else:
        main()
