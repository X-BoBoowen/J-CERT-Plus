"""Resumable matched full-graph policy comparisons, one numerical thread."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import csv
import json
import random
from pathlib import Path
from time import perf_counter

from engine import Edit
from workloads import NAMES, candidates, load_graph, opinions, system

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data"
DLL = ROOT.parent / "chol/ucrt64/bin/jcert_v2.dll"
if DLL.exists():
    os.environ.setdefault("JCERT_CHOLMOD_LIB", str(DLL))


def trial(A, edits, s, backend, mode, root_factor, setup_seconds):
    from policy_kernel import PolicyKernel
    start = perf_counter()
    kernel = PolicyKernel(A, edits, backend=backend, directional=False,
                          root_factor=root_factor, cached={"cold": False, "same": "same", "transport": True}[mode])
    sequence = []
    results = []
    try:
        stopped = False
        for step in range(1, 41):
            if not stopped:
                winner, gain = kernel.select(s, delta=1e-9)
                if winner is None:
                    stopped = True
                else:
                    kernel.commit(winner)
                    sequence.append(int(winner))
            if step in (20, 40):
                results.append(dict(b=step, seconds=setup_seconds + perf_counter()-start, setup_seconds=setup_seconds,
                                    commits=len(sequence), sequence=json.dumps(sequence),
                                    memory_bytes=kernel.memory_bytes(), factor_peak_bytes=kernel.factor.peak_bytes if kernel.factor else 0,
                                    counters=json.dumps(dict(kernel.ledger), sort_keys=True)))
    finally:
        kernel.close()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True, choices=NAMES)
    p.add_argument("--m", required=True, type=int, choices=[1000, 2500])
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--backend", choices=["cg", "cholmod"], nargs="+", default=["cg","cholmod"])
    p.add_argument("--memory", nargs=2, metavar=("BACKEND", "MODE"))
    args = p.parse_args()
    W = load_graph(args.graph)
    A = system(W)
    edits = [Edit(*e) for e in candidates(W, args.m, 17)]
    s = opinions(W, "unsigned_control", 31)
    path = OUTPUT / f"orthogonal_{args.graph}_{args.m}.csv"
    if args.memory:
        import psutil
        from cholmod_backend import CholmodFactor
        backend, mode = args.memory
        start = perf_counter()
        root = CholmodFactor(A) if backend == "cholmod" else None
        setup = perf_counter()-start if root is not None else 0.
        rows = trial(A, edits, s, backend, mode, root, setup)
        if os.name == "nt":
            peak = psutil.Process().memory_info().peak_wset
        else:
            import resource
            import sys
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform != "darwin":
                peak *= 1024
        record = dict(graph=args.graph, M=args.m, backend=backend, mode=mode, b=40,
                      process_peak_rss=peak, retained_bytes=rows[-1]["memory_bytes"],
                      clone_peak_bytes=rows[-1]["factor_peak_bytes"],
                      root_bytes=root.bytes if root is not None else 0,
                      root_peak_bytes=root.peak_bytes if root is not None else 0)
        if root is not None:
            root.close()
        path = OUTPUT / f"orthogonal_memory_{args.graph}_{args.m}_{backend}_{mode}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=record)
            writer.writeheader()
            writer.writerow(record)
        return
    previous = list(csv.DictReader(path.open())) if path.exists() else []
    done = {(int(r["rep"]), r["backend"], r["mode"]) for r in previous if r["b"] == "40"}
    for rep in range(args.reps):
        cells = [(k, v) for k in ("cg", "cholmod") for v in ("cold", "same", "transport")]
        random.Random(79 + rep).shuffle(cells)
        root_factor = None
        setup_seconds = 0.
        for backend, mode in cells:
            if backend not in args.backend or (rep, backend, mode) in done:
                continue
            if backend == "cholmod" and root_factor is None:
                from cholmod_backend import CholmodFactor
                start = perf_counter()
                root_factor = CholmodFactor(A)
                setup_seconds = perf_counter() - start
            rows = [dict(graph=args.graph, M=args.m, rep=rep, backend=backend,
                         mode=mode, **r) for r in trial(A, edits, s, backend, mode, root_factor if backend == "cholmod" else None, setup_seconds if backend == "cholmod" else 0.)]
            with path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                if stream.tell() == 0:
                    writer.writeheader()
                writer.writerows(rows)
            print(args.graph, args.m, rep, backend, mode, rows[-1]["seconds"], flush=True)


        if root_factor is not None:
            root_factor.close()

if __name__ == "__main__":
    main()




