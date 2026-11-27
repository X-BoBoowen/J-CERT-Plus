"""Uniform two-hop/negative catalog control with the matched study opinion."""
import argparse
import csv
import json
import random
from time import perf_counter

import numpy as np

from planned_matrix import OUTPUT, Edit
from workloads import load_graph, matched_inputs, system
from universe_index import build_universe
from fast_policy_kernel import FastCostPlanner, FastPolicyKernel
from cholmod_backend import CholmodFactor


def inputs(graph):
    W = load_graph(graph)
    s, _ = matched_inputs(W, 1000, seed=17)
    left, right, flip = build_universe(W)
    ids = np.random.default_rng(17).choice(len(left), 1000, replace=False)
    edits = [Edit(int(left[i]), int(right[i]), 1., bool(flip[i])) for i in ids]
    return system(W), s, edits


def run(graph, reps):
    A, s, edits = inputs(graph)
    path = OUTPUT / f"uniform_boundary_{graph}.csv"
    previous = list(csv.DictReader(path.open())) if path.exists() else []
    done = {(int(r["rep"]), r["backend"], r["mode"]) for r in previous}
    reference = json.loads(previous[0]["sequence"]) if previous else None
    for rep in range(reps):
        cells = [(backend, mode) for backend in ("cg", "cholmod")
                 for mode in ("cold", "planner") if (rep, backend, mode) not in done]
        random.Random(79+rep).shuffle(cells)
        for backend, mode in cells:
            start = perf_counter()
            factor = CholmodFactor(A) if backend == "cholmod" else None
            cls = FastCostPlanner if mode == "planner" else FastPolicyKernel
            kwargs = {} if mode == "planner" else {"cached": False}
            kernel = cls(A, edits, backend=backend, root_factor=factor, winner_slots=1, **kwargs)
            seq = []
            try:
                for _ in range(20):
                    e, _ = kernel.select(s, delta=1e-9)
                    if e is None:
                        break
                    kernel.commit(e)
                    seq.append(int(e))
                seconds = perf_counter()-start
                if reference is None:
                    reference = seq
                if seq != reference:
                    raise RuntimeError("uniform control decision mismatch")
                row = dict(graph=graph, M=len(edits), b=20, rep=rep, backend=backend,
                           mode=mode, seconds=seconds, flips=sum(edits[e].flip for e in seq),
                           catalog_flips=sum(e.flip for e in edits), sequence=json.dumps(seq),
                           counters=json.dumps(dict(kernel.ledger), sort_keys=True))
                with path.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=row)
                    if not f.tell():
                        writer.writeheader()
                    writer.writerow(row)
            finally:
                kernel.close()
                if factor is not None:
                    factor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", choices=("Alpha", "OTC"), required=True)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()
    run(args.graph, args.reps)
