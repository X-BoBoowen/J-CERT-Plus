"""Matched, resumable signed high-contention searches on frozen catalogs."""
import argparse
import csv
import json
import os
import random
from pathlib import Path
from time import perf_counter
from uuid import uuid4

# The planner needs the factor-entry counter exported by the diagnostic bridge.
bridge = Path(__file__).resolve().parents[2] / "chol/ucrt64/bin/jcert_diagnostic.dll"
if bridge.exists():
    os.environ.setdefault("JCERT_CHOLMOD_LIB", str(bridge))

from orthogonal import OUTPUT
from engine import Edit
from policy_kernel import PolicyKernel
from workloads import NAMES, load_graph, matched_inputs, system


def trial(A, edits, s, backend, mode, factor, setup, fast=False, persistent=False,
          batch_transport=False, native_pcg=None):
    start = perf_counter()
    kwargs = dict(backend=backend, root_factor=factor, winner_slots=1)
    if fast:
        from fast_policy_kernel import FastCostPlanner, FastPolicyKernel
        kernel_type = FastCostPlanner if mode == "planner" else FastPolicyKernel
    else:
        from cost_planner import CostPlanner
        kernel_type = CostPlanner if mode == "planner" else PolicyKernel
    if native_pcg is not None:
        from native_planner import NativeCostPlanner
        kernel_type = NativeCostPlanner
        kwargs['native_pcg'] = native_pcg
    if mode != "planner":
        kwargs["cached"] = {"cold": False, "same": "same"}[mode]
    elif batch_transport:
        kwargs["batch_transport"] = True
    kernel = kernel_type(A, edits, **kwargs)
    seq, rows, k_rounds = [], [], []
    stopped = False
    parent_z = None
    try:
        for step in range(1, 41):
            if not stopped:
                before = sum(kernel.ledger[k] for k in "CFR") if persistent else 0
                if persistent and mode == "planner":
                    if parent_z is None:
                        kernel.ledger["state_rhs"] += 1
                        z, rho = kernel.solve(s)
                    else:
                        z, rho = kernel.solve_committed_state(s, parent_z)
                    winner, _ = kernel.select_solved(s, z, rho, delta=1e-9)
                    parent_z = z
                else:
                    winner, _ = kernel.select(s, delta=1e-9)
                if persistent:
                    k_rounds.append(sum(kernel.ledger[k] for k in "CFR") - before)
                if winner is None:
                    stopped = True
                else:
                    kernel.commit(winner)
                    seq.append(int(winner))
            if step in (20, 40):
                rows.append(dict(b=step, seconds=setup+perf_counter()-start, setup_seconds=setup,
                                 commits=len(seq), flips=sum(bool(edits[e].flip) for e in seq),
                                 sequence=json.dumps(seq), counters=json.dumps(dict(kernel.ledger), sort_keys=True),
                                 memory_bytes=kernel.memory_bytes(),
                                 factor_peak_bytes=kernel.factor.peak_bytes if kernel.factor else 0))
                if persistent:
                    rows[-1]["k_rounds"] = json.dumps(k_rounds)
    finally:
        kernel.close()
    return rows


def matrix_inputs(graph, m, clustered=False, persistent=False):
    W = load_graph(graph)
    fixture = matched_inputs
    if clustered:
        from workloads import clustered_inputs
        fixture = clustered_inputs
    if persistent:
        from workloads import persistent_inputs
        fixture = persistent_inputs
    s, catalog = fixture(W, m, seed=17)
    return system(W), s, [Edit(*e) for e in catalog]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True, choices=NAMES)
    p.add_argument("--m", type=int, choices=(1000, 2500), required=True)
    p.add_argument("--fast", action="store_true", help="binary64 residual-certified local kernel")
    p.add_argument("--batch-transport", action="store_true",
                   help="run a fresh matrix with vectorized multi-candidate planner transport")
    p.add_argument("--shard", help="distinct output suffix for concurrent disjoint repetitions")
    p.add_argument("--first-rep", type=int, default=0)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--backend", nargs="+", choices=("cg","cholmod"), default=["cg","cholmod"])
    p.add_argument("--modes", nargs="+", choices=("cold","same","planner"), default=["cold","same","planner"])
    fixture = p.add_mutually_exclusive_group()
    fixture.add_argument("--clustered", action="store_true")
    fixture.add_argument("--matched", action="store_true")
    fixture.add_argument("--persistent", action="store_true",
                         help="persistent competition fixture; independent mutable factors")
    p.add_argument("--native-pcg", action="store_true", help="time the fused native iterative planner separately")
    args = p.parse_args()
    if args.persistent and not args.fast:
        p.error("--persistent requires --fast")
    if args.batch_transport and not args.persistent:
        p.error("--batch-transport requires --persistent --fast")
    native = None
    if args.native_pcg:
        if not (args.persistent and args.batch_transport and args.backend == ["cg"] and args.modes == ["planner"]):
            p.error("--native-pcg requires --persistent --fast --batch-transport --backend cg --modes planner")
        from tempfile import TemporaryDirectory
        from paired_pcg import compile_paired_pcg
        with TemporaryDirectory() as build:
            native = compile_paired_pcg(build)
            try:
                return run_matrix(args, native)
            finally:
                if os.name == 'nt':
                    from _ctypes import FreeLibrary
                    FreeLibrary(native.library._handle)
    return run_matrix(args, native)


def run_matrix(args, native):
    A, s, edits = matrix_inputs(args.graph, args.m, args.clustered, args.persistent)
    prefix = "planned_fast" if args.fast else "planned"
    if args.clustered:
        prefix = "clustered_fast" if args.fast else "clustered"
    if args.persistent:
        prefix = "persistent_batch" if args.batch_transport else "persistent_fast"
    if native is not None:
        prefix = "persistent_native"
    suffix = f".{args.shard}" if args.shard else ""
    path = OUTPUT / f"{prefix}_{args.graph}_{args.m}{suffix}.csv"
    done = set()
    sources = [path]
    if (args.clustered or args.persistent) and not args.shard:
        sources.extend(OUTPUT.glob(f"{prefix}_{args.graph}_{args.m}.*.csv"))
    for source in sources:
        if source.exists():
            with source.open() as f:
                done.update((int(r["rep"]),r["backend"],r["mode"]) for r in csv.DictReader(f) if r["b"] == "40")
    for rep in range(args.first_rep, args.reps):
        schedule = uuid4().hex if args.persistent else None
        factor = None
        setup = 0.
        cells = [(b,m) for b in args.backend for m in args.modes if (rep,b,m) not in done]
        random.Random(79+rep).shuffle(cells)
        try:
            for backend,mode in cells:
                if backend == "cholmod" and factor is None and not args.persistent:
                    from cholmod_backend import CholmodFactor
                    start = perf_counter()
                    factor = CholmodFactor(A)
                    setup = perf_counter()-start
                direct = backend == "cholmod"
                result = trial(A,edits,s,backend,mode,factor if direct else None,
                               setup if direct else 0., fast=args.fast, persistent=args.persistent,
                               batch_transport=args.batch_transport, native_pcg=native)
                rows = [dict(graph=args.graph,n=A.shape[0],catalog_seed=17,opinion_seed=17,M=args.m,rep=rep,backend=backend,mode=mode,
                             catalog_flips=sum(bool(e.flip) for e in edits),**row) for row in result]
                if args.persistent:
                    for row in rows:
                        row["schedule"] = schedule
                with path.open("a",newline="") as f:
                    writer = csv.DictWriter(f,fieldnames=rows[0])
                    if not f.tell():
                        writer.writeheader()
                    writer.writerows(rows)
                print(args.graph,args.m,rep,backend,mode,round(rows[-1]["seconds"],3),flush=True)
        finally:
            if factor is not None:
                factor.close()


if __name__ == "__main__":
    main()
