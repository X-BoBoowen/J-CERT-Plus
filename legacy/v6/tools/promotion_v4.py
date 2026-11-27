"""Ten reset long-horizon queries sharing a graph, catalog and certified root."""
import argparse
import csv
import json
import gc
import os
import tempfile
from threading import Event, Thread
from time import perf_counter

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from long_promotion import (DATA, OPINION_DEN, TAU, UNIT, Iterative, UpdatedFactor,
                            certified_state, search, seed_view)
from promotion_index_profile import ClassIndex, compile_index
from paired_pcg import compile_paired_pcg
from signed_promotion import build_screen, catalog, replace_rows, sign_audit
from workloads import NAMES, load_graph, opinions, system


METHODS = ("static", "adaptive", "full", "fixed")


def motif_sites(W, A):
    negative = (W < 0).tocsr()
    degree = A.diagonal()
    a = np.flatnonzero((np.diff(negative.indptr) == 2) & (degree < 7))[0]
    b, c = sorted(negative[a].indices, key=lambda v: (degree[v], v))
    return np.array([a, b, c]), np.array([round(768/degree[a]), -512, 96])


def query_opinion(W, family, seed=17, metadata=None):
    """Use one lexical negative-edge dipole blended with signed high-pass S1."""
    if family == "AlphaBlend":
        motif = query_opinion(W, "AlphaMotif", seed)
        signed = np.rint(OPINION_DEN*opinions(W, "S1", seed)).astype("int64")
        lower, remainder = np.divmod(31*motif+signed, 32)
        return lower + ((remainder > 16) | ((remainder == 16) & ((lower % 2) == 1)))
    if family == "AlphaMotif":
        A = system(W).astype("int64")
        sites, target = motif_sites(W, A)
        z = np.random.default_rng(seed).integers(-3, 4, size=A.shape[0])
        z[sites] = target
        rhs = A @ z
        return np.rint(rhs*min(1, OPINION_DEN/max(abs(rhs)))).astype("int64")
    if family == "AlphaClamp":
        A = system(W).astype("int64")
        sites, target = motif_sites(W, A)
        signed = np.rint(OPINION_DEN*opinions(W, "S1", seed)).astype("int64")
        started = perf_counter()
        y = UpdatedFactor(A).solve(signed.astype(float))
        aux_seconds = perf_counter()-started
        c = 3/np.abs(y).max()
        background = c*y
        rhs = c*signed + A[:, sites] @ (target-background[sites])
        scale = min(1, OPINION_DEN/np.abs(rhs).max())
        if metadata is not None:
            metadata.update(clamp_aux_seconds=aux_seconds, clamp_background_scale=c,
                            clamp_background_rms=float(np.sqrt(np.mean(background**2))),
                            clamp_background_max=float(np.abs(background).max()),
                            clamp_source_scale=scale)
        return np.rint(scale*rhs).astype("int64")
    negative = (W < 0).tocsr()
    i = int(np.flatnonzero(np.diff(negative.indptr))[0])
    j = int(min(negative[i].indices))
    s = 0.75*opinions(W, "S1", seed)
    s[i] += 0.25
    s[j] -= 0.25
    return np.rint(OPINION_DEN*s).astype("int64")


def output(name, kind, edition=4):
    return DATA / f"promotion_v{edition}_{name}_{kind}.csv"


def run(name, reps, broad, lib, paired, directional=True, edition=4, fine_screen=False):
    from promotion_root import bind_root

    if edition==6:
        from promotion_index_profile import FusedClassIndex
        index_type = FusedClassIndex
    else:
        index_type = ClassIndex
    matched = edition >= 5
    modes = {"static": "static", "adaptive": "promoted", "full": "full",
             "transport": "full", "fixed": "promoted"}
    methods = ("static", "adaptive", "full", "transport") if matched else METHODS
    indexed_methods = tuple(m for m in methods if m in ("adaptive", "fixed"))
    checked_methods = tuple(m for m in methods if m not in ("static", "full"))
    seed = 17
    tick = perf_counter()
    anchored = name in ("AlphaMotif", "AlphaBlend", "AlphaClamp")
    W = load_graph("Alpha" if anchored else name)
    metadata = {}
    if matched:
        from promotion_inputs5 import clamped_source
        s = clamped_source(W,name,seed,metadata)
    else:
        s = query_opinion(W, name, seed, metadata)
    graph_s = perf_counter()-tick
    tick = perf_counter()
    pool = catalog(W, name)
    catalog_s = perf_counter()-tick
    proof_unit = 1 << 48 if anchored or matched else UNIT
    tick = perf_counter()
    A = 16*system(W).astype("int64")
    fac = UpdatedFactor(A) if len(s) <= 10000 else Iterative(A, True, paired)
    solver = "sparse LU" if isinstance(fac, UpdatedFactor) else (
        "paired PCG" if paired is not None else "SciPy CG")
    if directional:
        label = name if paired is not None else name+"_scipy_directional"
    else:
        label = name+("_paired_box" if paired is not None else "_scipy")
    if fine_screen:
        label += "_fine"
        metadata["fine_screen"] = 1
    if edition==6:
        metadata.update(strict_screen=1,reuse_certified_rows=1)
    stage = label+"_running"
    state_start = perf_counter()
    state = certified_state(A, fac, s, proof_unit)
    root_state_s = perf_counter()-state_start
    root_iterations = getattr(fac, "iterations", 0)
    common_s = perf_counter()-tick
    tick = perf_counter()
    hot, _ = seed_view(pool, state[3], state[4], A.diagonal(), 64, False, 16)
    rank_s = perf_counter()-tick
    base = dict(A=A, fac=fac, state=state, hot=hot, common_seconds=common_s,
                rank_seconds=rank_s, index=None, index_bytes=0, index_seconds=0.,
                setup_seconds=rank_s)
    tick = perf_counter()
    index = index_type(lib, pool, state[3], state[4], 16)
    index_s = perf_counter()-tick
    roots = {method: base.copy() for method in methods}
    for mode in indexed_methods:
        # Posting lists and bins are shared. Each search starts a private
        # stamp array and uses the threshold policy bound to its root.
        roots[mode].update(index=index, index_bytes=index.nbytes, index_seconds=index_s,
                           setup_seconds=rank_s+index_s)
    bind_seconds = {}
    for method in methods:
        indexed = method in ("adaptive", "fixed")
        fixed_tau = TAU if method == "fixed" else None
        started = perf_counter()
        roots[method] = bind_root(W, s, pool, roots[method],
            "full" if method == "transport" else method,
            hot_size=64, operator_scale=16, proof_unit=proof_unit,
            index_factory=index_type if indexed else None, fixed_tau=fixed_tau)
        bind_seconds[method] = perf_counter()-started
    rounds = []
    runs = []
    reference = {}
    canonical = None
    proc = psutil.Process()
    for q in range(reps):
        paths = {}
        for method in methods[q%len(methods):]+methods[:q%len(methods)]:
            indexed = method in ("adaptive", "fixed")
            root = roots[method]
            gc.collect()
            peak = [proc.memory_info().rss]
            stop = Event()

            def sample():
                while not stop.wait(.015):
                    peak[0] = max(peak[0], proc.memory_info().rss)

            monitor = Thread(target=sample)
            monitor.start()
            try:
                rows, lo, hi, elapsed = search(W, s, pool, modes[method],
                    adaptive=indexed, operator_scale=16, dyadic_tau=True, trace=True,
                    broad_pass=broad, proof_unit=proof_unit, prepared_root=root,
                    woodbury_state=True, retain_winner=True, paired_pcg=paired,
                    directional=directional, transport=not matched or method != "full",
                    matched_screen=matched, fine_screen=fine_screen, strict_screen=edition==6,
                    index_factory=index_type if indexed else None)
            finally:
                stop.set()
                monitor.join()
            assert len(rows) == 15 and all(0 < r["low"] <= r["high"] for r in rows)
            if matched and method == "full":
                assert all(r["hits"] == r["transfers"] == 0 for r in rows)
                assert rows[-1]["fresh"] == sum(r["scores"] for r in rows)
            if edition==5 and modes[method] == "full":
                assert all(r["catalog_visits"] == len(pool[0]) for r in rows)
            ids = [r["winner_id"] for r in rows]
            assert len(set(ids)) == 15 and reference.setdefault(method, ids) == ids
            key = dict(graph=name, method=method, query=q, n=W.shape[0], universe=len(pool[0]),
                       seed=seed, proof_unit=proof_unit)
            for row in rows:
                for field in ("tau_units", "tau_denominator", "tau_grid_max", "tau_work"):
                    row.setdefault(field, "")
                rounds.append(dict(**key, **row))
            runs.append(dict(**key, graph_seconds=graph_s, catalog_seconds=catalog_s,
                common_seconds=common_s, root_state_seconds=root_state_s,
                root_state_iterations=root_iterations, rank_seconds=rank_s, index_seconds=root["index_seconds"],
                bind_seconds=bind_seconds[method], guarded=1,
                online_seconds=elapsed, peak_rss=peak[0], gain_low=lo, gain_high=hi,
                solver=solver, directional=int(directional), **metadata))
            if edition==6:
                runs[-1]["fused_threshold"] = int(indexed)
            paths[method] = rows
            print(name, q, method, round(elapsed, 3), flush=True)
        for mode in checked_methods:
            for a, b in zip(paths[mode], paths["full"]):
                assert a["winner_id"] == b["winner_id"]
                assert max(a["low"], b["low"]) <= min(a["high"], b["high"])
        if canonical is None:
            canonical = paths["adaptive"]
        replace_rows(output(stage, "runs", edition), runs)
        replace_rows(output(stage, "rounds", edition), rounds)
    audits, revived, winners, outsiders = sign_audit(W, s, pool, canonical, 16, proof_unit)
    if audits:
        replace_rows(output(label, "admissions", edition), audits)
    else:
        output(label, "admissions", edition).unlink(missing_ok=True)
    for kind in ("rounds", "runs"):
        output(stage, kind, edition).replace(output(label, kind, edition))
    print(name, "admissions/revived/winning/outsider", len(audits), revived, winners, outsiders,
          flush=True)


def check_v5(name, reps, fine_screen=False, *, edition=5):
    """Replay the cross-policy, accounting and sign checks on completed CSVs."""
    def records(kind):
        with output(name+"_fine" if fine_screen else name, kind, edition).open(newline="") as stream:
            return list(csv.DictReader(stream))

    runs, rounds, admissions = (records(k) for k in ("runs", "rounds", "admissions"))
    methods = ("static", "adaptive", "full", "transport")
    assert len(runs) == reps*len(methods) and len(rounds) == 15*len(runs)
    keys = {(row["method"], int(row["query"])) for row in runs}
    assert keys == {(m, q) for m in methods for q in range(reps)}
    initial_paths = {}
    for run in runs:
        assert run["graph"] == name and int(run["proof_unit"]) == 1 << 48
        assert bool(int(run.get("fine_screen", 0))) == fine_screen
        method, query = run["method"], run["query"]
        if edition==6:
            assert int(run["strict_screen"])==int(run["reuse_certified_rows"])==1
            assert int(run["fused_threshold"])==(method=="adaptive")
            assert int(run["guarded"])==int(run["directional"])==1
        path = [r for r in rounds if r["query"] == query and r["method"] == method]
        assert [int(r["epoch"]) for r in path] == list(range(1, 16))
        scan = sum(float(r[k]) for r in path for k in ("seed_screen_s", "cold_query_s", "cold_screen_s"))
        assert 0 < scan <= float(run["online_seconds"])
        ids = [int(r["winner_id"]) for r in path]
        assert len(set(ids)) == 15 and initial_paths.setdefault(method, ids) == ids
        for endpoint in ("low", "high"):
            assert int(run["gain_"+endpoint]) == sum(int(r[endpoint]) for r in path)
        previous = None
        for row in path:
            assert int(row["proof_unit"]) == 1 << 48
            assert int(row["transport_enabled"]) == (method != "full")
            assert 0 < int(row["low"]) <= int(row["high"])
            assert row["child_state_source"] in ("woodbury", "corrected")
            assert int(row["state_residual"]) >= 0 and int(row["child_state_residual"]) >= 0
            assert int(row["seed_visits"]) > 0
            scope = int(row["catalog_visits"])
            if edition==6:
                assert int(row["strict_rows"])==int(row["seed_visits"])+scope
                assert 0<=float(row["strict_setup_s"])<=float(row["seed_screen_s"])
            if method in ("full", "transport"):
                if edition==6:
                    assert int(row["strict_rows"])==int(row["universe"])-int(row["epoch"])+1
                else:
                    assert scope == int(row["universe"])
            elif method == "static":
                assert scope == 0
            else:
                assert scope == int(row["visited"])
            if previous is not None:
                assert row["state_residual"] == previous["child_state_residual"]
                assert row["state_source"] == previous["child_state_source"]
            previous = row
        if method == "full":
            assert all(int(r["hits"]) == int(r["transfers"]) == 0 for r in path)
            assert int(path[-1]["fresh"]) == sum(int(r["scores"]) for r in path)
        if method in ("adaptive", "transport"):
            cold = [r for r in rounds if r["query"] == query and r["method"] == "full"]
            for left, right in zip(path, cold):
                assert left["winner_id"] == right["winner_id"]
                assert max(int(left["low"]), int(right["low"])) <= min(int(left["high"]), int(right["high"]))
    receipt = {int(r["candidate"]): r for r in admissions}
    assert len(receipt) == len(admissions)
    certified = {}
    for r in rounds:
        if r["query"] == "0" and r["method"] == "adaptive":
            for key, endpoints in json.loads(r["admissions"]).items():
                e = int(key)
                assert e not in certified
                certified[e] = endpoints
                assert receipt[e]["epoch"] == r["epoch"]
                assert endpoints == [int(receipt[e][k]) for k in ("admission_low", "admission_high")]
    assert set(receipt) == set(certified)
    winners = set(initial_paths["adaptive"])
    negative = [r for e, r in receipt.items() if e in winners and r["root_high"]
                and int(r["root_high"]) < 0 and int(r["admission_low"]) > 0]
    assert negative and all(int(r["revived_at_admission"]) == 1 for r in negative)
    if fine_screen and output(name, "rounds", 5).exists():
        with output(name, "rounds", 5).open(newline="") as stream:
            coarse = [r for r in csv.DictReader(stream) if r["query"] == "0" and r["method"] == "full"]
        assert [int(r["winner_id"]) for r in coarse] == initial_paths["full"]
    print(name, "verified", len(runs), "cursors;", len(negative), "negative-at-root winners", flush=True)


def precision_records(names):
    """Measured screen-precision sensitivity over two complete ten-cursor cohorts."""
    from promotion_table_v4 import setup_cost
    records = []
    for graph in names:
        totals = {}
        cases = []
        for fine in (False, True):
            label = graph+"_fine" if fine else graph
            with output(label, "runs", 5).open(newline="") as stream:
                runs = list(csv.DictReader(stream))
            with output(label, "rounds", 5).open(newline="") as stream:
                rounds = list(csv.DictReader(stream))
            for method in ("static", "adaptive", "full", "transport"):
                group = sorted((r for r in runs if r["method"] == method), key=lambda r: int(r["query"]))
                assert [int(r["query"]) for r in group] == list(range(10))
                online = [float(r["online_seconds"]) for r in group]
                paths = [[r for r in rounds if r["method"] == method and int(r["query"]) == q]
                         for q in range(10)]
                assert all(len(path) == 15 for path in paths)
                setup = setup_cost(group[0])
                total = setup + sum(online)
                totals[fine, method] = total
                scans = [sum(float(r[k]) for r in path
                             for k in ("seed_screen_s", "cold_query_s", "cold_screen_s")) for path in paths]
                path = paths[0]
                cases.append(dict(graph=graph, fine_screen=int(fine), method=method, cursors=10,
                    setup_seconds=setup, median_online_seconds=float(np.median(online)),
                    median_scan_seconds=float(np.median(scans)), setup_plus_10_seconds=total,
                    fresh_scores=int(path[-1]["fresh"]), gram_hits=int(path[-1]["hits"]),
                    admitted=sum(int(r["promoted"]) for r in path)))
        for row in cases:
            fine, method = bool(row["fine_screen"]), row["method"]
            row["cold_over_dynamic_setup_plus_10"] = totals[fine, "full"]/totals[fine, "adaptive"]
            row["coarse_over_fine_setup_plus_10"] = totals[False, method]/totals[True, method]
        records.extend(cases)
    return records


def precision_summary(names):
    replace_rows(DATA/"promotion_v5_precision_summary.csv", precision_records(names))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", nargs="+", choices=(*NAMES, "AlphaMotif", "AlphaBlend", "AlphaClamp"),
                        default=None)
    parser.add_argument("--fine-screen", action="store_true", help="Apply the same cap on fine dyadic survivor coordinates and record separately.")
    parser.add_argument("--precision-summary", action="store_true", help="Summarize actual ten-cursor coarse and fine cohorts (default Alpha).")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check-v6", action="store_true", help="Check complete strict timing, row coverage, state continuity and sign receipts.")
    checks.add_argument("--check-v5", action="store_true", help="Check completed edition-5 timing and sign receipts.")
    parser.add_argument("--edition", type=int, choices=(4,5,6), default=4)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--scipy-control", action="store_true",
                        help="Use scalar SciPy CG and record a separate baseline.")
    parser.add_argument("--box-control", action="store_true",
                        help="Use the earlier interval score in a separate comparison.")
    args = parser.parse_args()
    if args.fine_screen and args.edition != 5 and not args.check_v5:
        parser.error("--fine-screen requires --edition 5")
    if args.edition >= 5 and (args.scipy_control or args.box_control
            or any(name not in NAMES for name in (args.graph or ()))):
        parser.error("Editions 5 and 6 use the six unmodified graphs, paired PCG and directional proofs")
    if args.reps < 1:
        parser.error("--reps must be positive")
    if args.precision_summary:
        precision_summary(args.graph or ("Alpha",))
        raise SystemExit(0)
    if args.check_v5 or args.check_v6:
        for name in args.graph or NAMES:
            check_v5(name, args.reps, args.fine_screen, edition=6 if args.check_v6 else 5)
        raise SystemExit(0)
    with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as tmp:
        broad = build_screen(tmp)
        lib = compile_index(tmp)
        paired = None if args.scipy_control else compile_paired_pcg(tmp)
        try:
            for name in args.graph or (NAMES if args.edition>=5 else (*NAMES,"AlphaClamp")):
                run(name, args.reps, broad, lib, paired, directional=not args.box_control, edition=args.edition, fine_screen=args.fine_screen)
        finally:
            if os.name == "nt":
                from _ctypes import FreeLibrary
                if paired is not None:
                    FreeLibrary(paired.library._handle)
                FreeLibrary(lib._handle)
                FreeLibrary(broad.library._handle)
