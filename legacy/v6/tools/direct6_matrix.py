"""Registered direct-plan comparison on all five direct-routed graphs."""
import argparse
import csv
import json
import random
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

from planned_matrix import OUTPUT, matrix_inputs
from direct6 import DirectPlanner, compile_residual
from direct6_slots import SlotPlanner
from fast_policy_kernel import FastCostPlanner

GRAPHS = ("Alpha", "OTC", "Conflict", "Bio", "Fin")


def trial(A, edits, s, mode, residual):
    start = perf_counter()
    options = dict(backend="cholmod", winner_slots=1, batch_transport=True)
    if mode == "planner":
        kernel = FastCostPlanner(A, edits, **options)
    else:
        planner = {"planner6": DirectPlanner, "slots6": SlotPlanner}[mode]
        kernel = planner(A, edits, residual=residual, **options)
    sequence, lower, upper, counts, rows = [], [], [], [], []
    parent = None
    try:
        for step in range(1, 41):
            before = sum(kernel.ledger[k] for k in "CFR")
            if parent is None:
                kernel.ledger["state_rhs"] += 1
                z, rho = kernel.solve(s)
            else:
                z, rho = kernel.solve_committed_state(s, parent)
            winner, lo = kernel.select_solved(s, z, rho, delta=1e-9)
            if winner is None:
                raise ValueError("registered persistent search stopped early")
            sequence.append(int(winner))
            lower.append(float(lo))
            upper.append(float(kernel.last_upper))
            counts.append(sum(kernel.ledger[k] for k in "CFR")-before)
            kernel.commit(winner)
            parent = z
            if step in (20, 40):
                rows.append(dict(b=step, seconds=perf_counter()-start, setup_seconds=0.,
                                 commits=step, flips=sum(edits[e].flip for e in sequence),
                                 sequence=json.dumps(sequence), winner_lower=json.dumps(lower),
                                 winner_upper=json.dumps(upper), k_rounds=json.dumps(counts),
                                 counters=json.dumps(dict(kernel.ledger), sort_keys=True),
                                 memory_bytes=kernel.memory_bytes(),
                                 factor_peak_bytes=kernel.factor.peak_bytes))
    finally:
        kernel.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", nargs="+", choices=GRAPHS, default=GRAPHS)
    parser.add_argument("--slots", action="store_true",
                        help="compare the fixed-slot operator to the fused residual planner")
    args = parser.parse_args()
    with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        residual = compile_residual(directory)
        for graph in args.graph:
            for m in (1000, 2500):
                prefix = "direct6_slots" if args.slots else "direct6"
                path = OUTPUT/f"{prefix}_{graph}_{m}.csv"
                if path.exists():
                    with path.open(newline="") as stream:
                        done = {(int(r["rep"]), r["mode"], int(r["b"]))
                                for r in csv.DictReader(stream)}
                else:
                    done = set()
                A, s, edits = matrix_inputs(graph, m, persistent=True)
                for rep in range(10):
                    modes = ["planner6", "slots6"] if args.slots else ["planner", "planner6"]
                    expected = {(rep, mode, b) for mode in modes for b in (20, 40)}
                    if expected <= done:
                        continue
                    if expected & done:
                        raise ValueError(f"incomplete paired repetition {path} {rep}")
                    random.Random(79+rep).shuffle(modes)
                    schedule = uuid4().hex
                    rows = []
                    for mode in modes:
                        for row in trial(A, edits, s, mode, residual):
                            rows.append(dict(graph=graph, n=A.shape[0], catalog_seed=17,
                                             opinion_seed=17, M=m, rep=rep, backend="cholmod",
                                             mode=mode, catalog_flips=sum(e.flip for e in edits),
                                             schedule=schedule, **row))
                    with path.open("a", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=rows[0])
                        if not stream.tell():
                            writer.writeheader()
                        writer.writerows(rows)
                print(graph, m, flush=True)


if __name__ == "__main__":
    main()
