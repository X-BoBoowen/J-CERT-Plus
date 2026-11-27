"""Verify full-graph experiment records and replay the shipped decision witnesses."""
import argparse
from collections import Counter, defaultdict
import csv
from itertools import product
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from statistics import median

import numpy as np
from workloads import load_graph, system

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER / "data"
SIZES = dict(Alpha=3783, OTC=5881, Slashdot=82140, Conflict=768, Bio=1536, Fin=1024)
BACKENDS = ("cg", "cholmod")
MODES = ("cold", "same", "transport")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(name):
    with (DATA / name).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(rows and all(None not in r and None not in r.values() for r in rows),
            f"{name}: empty or malformed CSV")
    for row in rows:
        if "seconds" in row:
            require(0 < float(row["seconds"]) < math.inf, f"{name}: invalid duration")
    return rows


def cohort(rows, fields, expected):
    actual = Counter(tuple(r[k] for k in fields.split()) for r in rows)
    wanted = Counter(tuple(map(str, key)) for key in expected)
    require(actual == wanted, f"{fields}: missing {list((wanted-actual).items())[:2]}, "
            f"unexpected {list((actual-wanted).items())[:2]}")


def graphs():
    from workloads import root_flip_signs
    with (PAPER/"datasets/root_signs.csv").open(newline="") as f:
        sign_rows = list(csv.DictReader(f))
    cohort(sign_rows,"graph seed M",product(SIZES,(17,31),[2500]))
    for name, size in SIZES.items():
        W = load_graph(name)
        require(W.shape == (size, size) and W.has_canonical_format and W.nnz > 0
                and (W != W.T).nnz == 0 and not W.diagonal().any()
                and (abs(W.data) == 1).all(), f"{name}: invalid full graph")
        for row in (r for r in sign_rows if r["graph"] == name):
            require(tuple(int(row[k]) for k in ("positive","negative","unresolved"))
                    == tuple(root_flip_signs(W,int(row["seed"]))), f"{name}: root flip signs")


def uniform_controls():
    """Validate the historical unsigned-diffusion boundary on the full graphs."""
    for name,m in product(SIZES,(1000,2500)):
        rows = read(f"orthogonal_{name}_{m}.csv")
        cohort(rows,"graph M rep backend mode b",
               product([name],[m],range(10),BACKENDS,MODES,(20,40)))
        paths = {}
        for row in rows:
            b = int(row["b"])
            path = json.loads(row["sequence"])
            counts = Counter(json.loads(row["counters"]))
            require(paths.setdefault(b,path) == path and len(set(path)) == len(path)
                    and len(path) == int(row["commits"]) <= b and all(0 <= e < m for e in path)
                    and counts["state_rhs"] == min(b,len(path)+1)
                    and counts["screens"] == sum(m-i for i in range(counts["state_rhs"]))
                    and counts["refine_rhs"] == sum(counts[k+"_rhs"] for k in "CFR")
                    and counts["W_rhs"] == counts["winner_rhs"],
                    f"{name}/{m}: historical uniform control trace")
        require(paths[20] == paths[40][:20], f"{name}/{m}: historical prefix")


def uniform_boundary():
    from uniform_boundary import inputs
    for name in ("Alpha","OTC"):
        _,_,edits = inputs(name)
        rows = read(f"uniform_boundary_{name}.csv")
        cohort(rows,"graph M b rep backend mode",product([name],[1000],[20],range(10),BACKENDS,
                                                        ("cold","planner")))
        expected = json.loads(rows[0]["sequence"])
        require(len(expected) == len(set(expected)) <= 20
                and all(type(e) is int and 0 <= e < 1000 for e in expected), f"{name}: uniform path")
        for row in rows:
            counts = Counter(json.loads(row["counters"]))
            require(json.loads(row["sequence"]) == expected
                    and int(row["catalog_flips"]) == sum(bool(e.flip) for e in edits)
                    and int(row["flips"]) == sum(bool(edits[e].flip) for e in expected)
                    and counts["state_rhs"] == min(20,len(expected)+1)
                    and counts["refine_rhs"] == sum(counts[k+"_rhs"] for k in "CFR")
                    and counts["winner_rhs"] == counts["W_rhs"], f"{name}: matched uniform boundary")


def planned_matrix(clustered=False):
    """Check every timed run, including the two correlated prefix checkpoints."""
    from planned_matrix import matrix_inputs
    from policy_kernel import PolicyKernel
    prefix = "clustered" if clustered else "planned"
    fixture_label = "clustered" if clustered else "matched"
    rhs_count = 3 if clustered else 96
    modes = ("cold", "same", "planner")
    with (PAPER/f"datasets/{fixture_label}_setup.csv").open(newline="") as f:
        setup_rows = list(csv.DictReader(f))
    with (PAPER/f"datasets/{fixture_label}_setup_raw.csv").open(newline="") as f:
        setup_runs = list(csv.DictReader(f))
    cohort(setup_rows,"graph M seed",product(SIZES,(1000,2500),[17]))
    cohort(setup_runs,"graph M seed rep",product(SIZES,(1000,2500),[17],range(10)))
    require(all(int(r["calibration_rhs"]) == rhs_count and 0 < float(r["seconds"]) < math.inf
                for r in setup_runs), "signed setup repetitions")
    from workloads import NAMES
    rng = np.random.default_rng(48231)
    for name,m in product(NAMES,(1000,2500)):
        summary_row = next(r for r in setup_rows if r["graph"] == name and int(r["M"]) == m)
        samples = [float(r["seconds"]) for r in sorted(setup_runs,key=lambda r:int(r["rep"]))
                   if r["graph"] == name and int(r["M"]) == m]
        boot = np.median(rng.choice(samples,(20000,10)),axis=1)
        require(int(summary_row["reps"]) == 10 and np.allclose(
                [float(summary_row[k]) for k in ("q1_seconds","seconds","q3_seconds")],
                np.quantile(samples,[.25,.5,.75])) and np.allclose(
                [float(summary_row[k]) for k in ("median_ci_lo","median_ci_hi")],
                np.quantile(boot,[.025,.975])), f"{name}/{m}: setup timing intervals")
    fixture = {(r["graph"],int(r["M"])):float(r["seconds"]) for r in setup_rows}
    require(all(int(r["calibration_rhs"]) == rhs_count and 0 < float(r["seconds"]) < math.inf
                for r in setup_rows), "signed workload setup")
    summary = read(f"{prefix}_summary.csv")
    cohort(summary,"graph M b backend",product(SIZES,(1000,2500),(20,40),BACKENDS))
    for name, m in product(SIZES, (1000, 2500)):
        key = f"{name}_{m}"
        rows = read(f"{prefix}_fast_{key}.csv")
        quad = read(f"{prefix}_quad_{key}.csv")
        cohort(rows, "graph M rep backend mode b",
               product([name], [m], range(10), BACKENDS, modes, (20, 40)))
        cohort(quad, "graph M rep backend mode b",
               product([name], [m], range(10), ["cg"], ["quad"], (20, 40)))
        A,s,edits = matrix_inputs(name,m,clustered)
        require(A.shape == (SIZES[name],SIZES[name]) and len(s) == SIZES[name]
                and len(edits) == len({(e.i,e.j) for e in edits}) == m,
                f"{key}: regenerated graph and catalog dimensions")
        paths, schedules, setups, checkpoints = {}, {}, {}, {}
        for row in rows + quad:
            b, rep = int(row["b"]), int(row["rep"])
            backend, mode = row["backend"], row["mode"]
            path = json.loads(row["sequence"])
            require(isinstance(path,list) and len(set(path)) == len(path) == int(row["commits"])
                    and len(path) <= b and all(type(e) is int and 0 <= e < m for e in path),
                    f"{key}: invalid {mode} sequence")
            require(paths.setdefault(b,path) == path, f"{key}: {backend}/{mode} winner disagreement")
            elapsed = float(row["seconds"])
            checkpoints[b,rep,backend,mode] = elapsed
            counts = Counter(json.loads(row["counters"]))
            require(all(type(v) is int and v >= 0 for v in counts.values()), f"{key}: invalid ledger")
            if mode == "quad":
                require(counts["state_rhs"] == min(b,len(path)+1)
                        and counts["screens"] == sum(m-i for i in range(counts["state_rhs"]))
                        and counts["queries"] >= len(path) and counts["matvecs"] > 0,
                        f"{key}: quadrature schedule")
                continue
            direct = backend == "cholmod"
            flips = int(row["flips"])
            setup = float(row["setup_seconds"])
            require(0 <= setup < elapsed and (setup > 0) == direct
                    and setups.setdefault((backend,rep),setup) == setup, f"{key}: shared setup")
            require(int(row["n"]) == SIZES[name] and int(row["catalog_seed"]) == 17
                    and int(row["opinion_seed"]) == 17
                    and flips == sum(bool(edits[e].flip) for e in path)
                    and int(row["catalog_flips"]) == sum(bool(e.flip) for e in edits)
                    and counts["factor_down"] == direct*flips
                    and counts["factor_up"] == direct*len(path), f"{key}: rank updates")
            require(counts["factorizations"] == 0
                    and counts["factor_clones"] == direct + counts["planner_parent_clones"]
                    and counts["state_rhs"] == min(b,len(path)+1)
                    and counts["screens"] >= sum(m-i for i in range(counts["state_rhs"]))
                    and counts["refine_rhs"] == sum(counts[k+"_rhs"] for k in ("C","F","R"))
                    and counts["winner_rhs"] == counts["W_rhs"]
                    and all(counts[k] <= counts[k+"_rhs"] <= 2*counts[k] for k in "CFWR")
                    and (mode == "planner" or counts["transported"] == 0),
                    f"{key}: {backend}/{mode} solve ledger")
            if mode == "planner":
                require(counts["gate_requests"] == counts["gate_opens"] + counts["gate_misses"]
                        and counts["transported"] == counts["gate_opens"]
                        and counts["gate_requests"] >= 2*counts["W"]
                        and counts["gate_opens"] >= counts["W"]+counts["retained_hits"]
                        and counts["forecast_rows"] == counts["forecast_realized"]+counts["forecast_still_unrealized"]
                        and counts["planner_parent_clones"] == direct*counts["W"]
                        and counts["planner_undo_down"] == counts["planner_parent_clones"]
                        and counts["planner_undo_up"] <= counts["planner_undo_down"],
                        f"{key}: demand gate totals")
            schedule = tuple(counts[k] for k in ("state_rhs","factor_up","factor_down"))
            require(schedules.setdefault((backend,b),schedule) == schedule, f"{key}: solver schedule")
        if clustered:
            check_clustered_cell(name,m,rows,quad,paths,edits)
        require(paths[20] == paths[40][:20], f"{key}: non-prefix checkpoints")
        require(all(elapsed >= checkpoints[20,rep,backend,mode]
                    for (b,rep,backend,mode),elapsed in checkpoints.items() if b == 40),
                f"{key}: non-monotonic cumulative timing")
        root = PolicyKernel(A,edits,cached=False)
        try:
            z,rho = root.solve(s)
            signs = {}
            for e in paths[40]:
                lo,hi = root._interval(e,z,rho)
                signs[e] = -1 if hi < 0 else int(lo > 0)
        finally:
            root.close()
        for report in (r for r in summary if r["graph"] == name and int(r["M"]) == m):
            b,backend = int(report["b"]),report["backend"]
            selected = {r["mode"]:Counter(json.loads(r["counters"])) for r in rows
                        if r["backend"] == backend and int(r["b"]) == b and int(r["rep"]) == 0}
            cold,planner = selected["cold"], selected["planner"]
            ratios = sorted(checkpoints[b,i,backend,"cold"]/checkpoints[b,i,backend,"planner"]
                            for i in range(10))
            totals = {mode:sum(c[k+"_rhs"] for k in "CFWR") for mode,c in selected.items()}
            stats = dict(pairs=10,cold_rhs=totals["cold"],planner_rhs=totals["planner"],
                         saved_rhs=totals["cold"]-totals["planner"],ratio=median(ratios),
                         lo=ratios[1],hi=ratios[-2],net_pct=100*(1-totals["planner"]/totals["cold"]),
                         K=sum(cold[k] for k in "CFR")/len(paths[b]),gate_requests=planner["gate_requests"],
                         gate_opens=planner["gate_opens"], winner_rhs=planner["W_rhs"],
                         root_negative=sum(signs[e]<0 for e in paths[b]),
                         root_unresolved=sum(signs[e]==0 for e in paths[b]),
                         commits=len(paths[b]), flips=sum(bool(edits[e].flip) for e in paths[b]))
            stats.update({mode+"_seconds":median(checkpoints[b,i,backend,mode] for i in range(10))
                          for mode in modes})
            preparation = fixture[name,m]
            stats.update(fixture_seconds=preparation,inclusive_ratio=median(
                (preparation+checkpoints[b,i,backend,"cold"])/(preparation+checkpoints[b,i,backend,"planner"])
                for i in range(10)))
            require(all(math.isclose(float(report[k]),v) for k,v in stats.items()),
                    f"{key}: paired table or root-sign audit")


def sequencing():
    from full_sequencing import audit
    from scipy.sparse.linalg import splu
    for name in ("Alpha","OTC"):
        cases = json.loads((DATA/f"full_sequencing_{name}.json").read_text())
        rows = read(f"full_sequencing_times_{name}.csv")
        cohort(rows,"case repeat policy",product(range(10),range(10),("fresh","transport")))
        require(len(cases) == 10, f"{name}: case count")
        A = system(load_graph(name))
        factor = splu(A.tocsc())
        for i, case in enumerate(cases):
            require(case["graph"] == name and len(case["edits"]) == 15 and
                    sorted(case["order"]) == list(range(15)) and case["backtracks"] > 0,
                    f"{name}/{i}: invalid hard case")
            _, sep = audit(A,factor,case)
            trials = [r for r in rows if int(r["case"]) == i]
            require(all(math.isclose(float(r["sep"]),sep,rel_tol=1e-5,abs_tol=1e-12)
                        and float(r["setup"]) > 0 and float(r["search"]) > 0
                        for r in trials), f"{name}/{i}: audit/timing mismatch")


def stress_grid(write=False):
    tolerances=(1e-13,1e-10,1e-7,1e-5)
    path=PAPER/"sections/stress.tex"
    source=path.read_text(encoding="utf8")
    published=source.splitlines()
    for name in ("bio","alpha"):
        records=[]
        for w in (1,4,16,64):
            records.extend(read(f"directional_stress_{name}_{w}_grid.csv"))
        keys=[(int(r["w"]),int(r["b"]),float(r["rtol"])) for r in records]
        budgets=(15,20,30,50) if name=="bio" else (15,20)
        expected=product((1,4,16,64),budgets,tolerances if name=="bio" else (1e-7,))
        require(Counter(keys)==Counter(expected),f"{name}: stress grid is missing or duplicated")
        if name=="bio":
            data=dict(zip(keys,records,strict=True))
            nonzero={k for k,r in data.items() if int(r["R"])!=0}
            require(nonzero=={(64,b,1e-5) for b in budgets},"Bio: 60 of 64 cells must have zero reanchors")
            lines=[]
            for w,b in product((1,4,16,64),budgets):
                cells=[data[w,b,t] for t in tolerances]
                prefix=f"{w}&{cells[0]['beta']}" if b==budgets[0] else "&"
                lines.append("&".join([prefix,str(b),*(r["baseline_R"]+"/"+r["R"] for r in cells)])+r"\\")
            header=next(i for i,line in enumerate(published) if line.startswith(r"$w$&"))
            start=published.index(r"\hline",header)+1
            end=published.index(r"\hline",start)
            if write:
                published[start:end]=lines
                path.write_text("\n".join(published)+("\n" if source.endswith("\n") else ""),encoding="utf8")
            else:
                require(published[start:end]==lines,"Table V differs from frozen Bio measurements")


def directional_stress():
    for name,prefix in (("bio","flip_stress_"),("alpha","flip_stress_alpha_")):
        for table,dimensions in (("events",("rtol","epoch","edge")),
                                 ("steps",("rtol","epoch")),("grid",("rtol","b"))):
            old = read(prefix+table+".csv")
            for w in (1,4,16,64):
                rows = read(f"directional_stress_{name}_{w}_{table}.csv")
                keys = ("w",*dimensions)
                reference = {tuple(r[k] for k in keys):r for r in old if int(r["w"]) == w}
                cohort(rows," ".join(keys),reference)
                for row in rows:
                    before = reference[tuple(row[k] for k in keys)]
                    if table == "events":
                        observed,baseline,completed,same_state = (float(row[k]) for k in
                            ("radius","baseline_radius","completed","same_state_old"))
                        require(row["winner"] == before["winner"]
                                and math.isclose(baseline,float(before["radius"]),rel_tol=1e-9,abs_tol=1e-10)
                                and abs(float(before["score"])-float(before["true_gain"]))
                                    <= baseline+float(before["gain_ref_cap"])
                                and 0 <= float(before["error"])
                                    <= float(row["baseline_gram"])+float(before["ref_cap"])
                                and math.isclose(observed,min(completed,same_state),rel_tol=1e-12)
                                and float(row["rho_p"]) >= 0 and observed >= 0
                                and abs(float(row["score"])-float(row["true_gain"])) <=
                                    observed+float(row["gain_ref_cap"])
                                and 0 <= float(row["error"]) <= float(row["posterior"])+float(row["ref_cap"]),
                                f"{name}/{w}: directional enclosure")
                    elif table == "steps":
                        require(row["winner"] == before["winner"] and int(row["reanchors"]) >= 0
                                and float(row["margin"]) > 0
                                and (not row["max_rho_P"] or float(row["max_rho_P"]) >= 0),
                                f"{name}/{w}: directional winner")
                    else:
                        require(row["accepted"] == before["accepted"] and
                                int(row["blocks"]) == sum(int(row[k]) for k in "CWR")
                                and int(row["baseline_R"]) == int(before["R"]),
                                f"{name}/{w}: directional totals")
                if table == "grid":
                    steps = read(f"directional_stress_{name}_{w}_steps.csv")
                    for row in rows:
                        n = int(row["accepted"])
                        prefix_steps = [s for s in steps if s["rtol"] == row["rtol"]
                                        and int(s["epoch"]) <= n]
                        require(len(prefix_steps) == n
                                and int(row["R"]) == sum(int(s["reanchors"]) for s in prefix_steps)
                                and int(row["direct_fallbacks"]) == sum(int(s["direct_fallback"])
                                                                       for s in prefix_steps),
                                f"{name}/{w}: cumulative directional reanchors")


def branch_queries(prefixes=("branch_pinned","branch_s2")):
    from workloads import opinions, contention_candidates, matched_inputs
    budgets = [20,10,15,25]*3
    for name,prefix in product(("Alpha","OTC"),prefixes):
        rows = read(f"{prefix}_{name.lower()}.csv")
        cohort(rows,"rep level", product(range(10),range(4)))
        W = load_graph(name)
        if prefix == "branch_s2":
            edits = contention_candidates(W,1000,opinions(W,"S2",17),seed=17)
        else:
            _,edits = matched_inputs(W,1000,seed=17)
        expected = json.loads(rows[0]["paths"])
        if prefix == "branch_pinned":
            for control in ("branch","branch_fast"):
                control_rows = read(f"{control}_{name.lower()}.csv")
                cohort(control_rows,"rep level",product(range(10),range(4)))
                require(all(json.loads(r["paths"]) == expected and r["flips"] == rows[0]["flips"]
                            for r in control_rows), f"{name}: undo/rational sensitivity winners")
        require(len(expected) == len(budgets) and all(len(path) == len(set(path)) == b
                and all(type(e) is int and 0 <= e < 1000 for e in path)
                for b,path in zip(budgets,expected)), f"{name}: branching paths/budgets")
        states = {(q//6,tuple(path[:d])) for q,path in enumerate(expected) for d in range(len(path))}
        changes = {tuple(path[:d]) for path in expected for d in range(1,len(path))}
        flips = sum(bool(edits[e][3]) for path in expected for e in path)
        shared = {}
        for row in rows:
            counts = Counter(json.loads(row["counters"]))
            level = int(row["level"])
            require(json.loads(row["paths"]) == expected and int(row["flips"]) == flips,
                    f"{name}: branching winner mismatch")
            require(counts["state_rhs"] == (len(states) if level else sum(budgets))
                    and counts["factor_up"] == (len(changes) if level else sum(b-1 for b in budgets))
                    and counts["factor_down"] == (sum(bool(edits[p[-1]][3]) for p in changes) if level
                                else sum(bool(edits[e][3]) for path in expected for e in path[:-1]))
                    and counts["refine_rhs"] == sum(counts[k+"_rhs"] for k in "CFR")
                    and counts["W_rhs"] == counts["winner_rhs"]
                    and (level == 3 or counts["transported"] == 0), f"{name}/L{level}: branching schedule")
            if level and prefix != "branch_s2":
                schedule = tuple(counts[k] for k in ("factorizations","factor_clones","state_rhs",
                                                     "factor_up","factor_down","planner_undo_up","planner_undo_down"))
                require(counts["factor_clones"] == len(changes)+1
                        and counts["planner_parent_clones"] == counts["planner_undo_up"] == counts["planner_undo_down"] == 0
                        and shared.setdefault(row["rep"],schedule) == schedule,
                        f"{name}: L1--L3 state/factor schedule")


def main_witness():
    from main_witness import run
    for case in ("branch","persistent6","universe6"):
        result = run(case,1)
        stored = json.loads((DATA/f"main_witness_{case}.json").read_text())
        for key in ("case","plan","budget","pool","bytes","objective","denominator",
                    "epochs","performance_delta","certified_delta","root_gain"):
            require(stored[key] == result[key], f"{case}: main witness {key}")
        if case=="persistent6":
            require(stored["root_gains"]==result["root_gains"], "persistent root signs disagree")
        samples = np.asarray(stored["samples"])
        require(samples.shape == (10,3) and np.isfinite(samples).all() and (samples > 0).all()
                and np.allclose(stored["median_seconds"],np.median(samples,axis=0)),
                f"{case}: integer replay timings")
        draws = np.random.default_rng(17).integers(10,size=(10000,10))
        require(stored["independent_processes"] == 10 and stored["bootstrap_seed"] == 17
                and stored["bootstrap_resamples"] == len(draws)
                and stored["phases"] == ["input","index","online"]
                and np.allclose(stored["iqr_seconds"],np.quantile(samples,[.25,.75],axis=0).T)
                and np.allclose(stored["median_ci95_seconds"],np.quantile(
                    np.median(samples[draws],axis=1),[.025,.975],axis=0).T),
                f"{case}: proof timing quantiles")


def primary_intervals():
    from primary_intervals import FIELDS,timing_intervals
    expected = {tuple(map(str,r[:7])):r[7:] for r in timing_intervals()}
    rows = read("branch_timing_intervals.csv")
    cohort(rows, " ".join(FIELDS[:7]), expected)
    for r in rows:
        key = tuple(r[f] for f in FIELDS[:7])
        require(np.allclose([float(r[f]) for f in FIELDS[7:]],expected[key],rtol=1e-12,atol=0),
                f"{key}: median, quartiles or bootstrap CI")
        

def supporting_intervals():
    from primary_intervals import SUPPORT_FIELDS,support_intervals,scale_intervals,search_intervals
    fields = SUPPORT_FIELDS[:4]
    for filename,calculator in (("margin_timing_intervals.csv",support_intervals),
                                ("scale_timing_intervals.csv",scale_intervals),
                                ("search_timing_intervals.csv",search_intervals)):
        rows = read(filename)
        expected = {tuple(map(str,r[:4])):r[4:] for r in calculator()}
        cohort(rows," ".join(fields),expected)
        for row in rows:
            key = tuple(row[k] for k in fields)
            require(np.allclose([float(row[k]) for k in SUPPORT_FIELDS[4:]],
                                expected[key],rtol=1e-12,atol=0),
                    f"{key}: supporting-study timing interval")



def signed_promotion():
    from signed_promotion import catalog, sign_audit
    from workloads import opinions
    from long_promotion import OPINION_DEN
    motif = json.loads((DATA/"signed_promotion_motif.json").read_text())
    paths = motif["paths"]
    revived = motif["revival"]
    before,after = motif["direct_gain_intervals"]
    d,gram = motif["direct_gain_denominator"],motif["denominator"]
    require(d == 2**80 and before[0] <= before[1] < 0 < after[0] <= after[1]
            and max(before[0]*gram,revived["root_gain_low"]*d) <= min(before[1]*gram,revived["root_high"]*d)
            and max(after[0]*gram,revived["admission_low"]*d) <= min(after[1]*gram,revived["admission_high"]*d),
            "signed motif: direct/Gram root and admission energy intervals")
    ids = {mode:[r["winner_id"] for r in path] for mode,path in paths.items()}
    require(ids["full"] == ids["promoted"] and len(ids["full"]) == 15
            and ids["static"][0] == ids["full"][0] and ids["static"][1] != ids["full"][1]
            and revived["candidate"] == ids["full"][1] and revived["epoch"] == 2
            and revived["root_high"] < 0 < revived["admission_low"] <= revived["admission_high"]
            and revived["flip"] == revived["revived_at_admission"] == 1,
            "signed motif: root-negative outsider revival")
    methods = ("static","promoted","full")
    generic_runs = read("signed_generic_control_runs.csv")
    generic_rounds = read("signed_generic_control_rounds.csv")
    cohort(generic_runs,"graph method repetition",product(SIZES,["promoted"],[0]))
    cohort(generic_rounds,"graph method repetition epoch",product(SIZES,["promoted"],[0],range(1,16)))
    for prefix,reps in (("signed_promotion_",10),("signed_promotion_unit_",1)):
        summary = read(prefix+"summary_methods.csv")
        cohort(summary,"graph method reps",product(SIZES,methods,[reps]))
        for name,size in SIZES.items():
            W = load_graph(name)
            left,right,is_flip = catalog(W,name)
            universe = len(left)
            head = prefix+name+"_"
            if reps == 10:
                midpoint = read(f"signed_promotion_midpoint_{name}_rounds.csv")
                cohort(midpoint,"graph universe method seed insertion_denominator epoch",
                       product([name],[universe],["full"],[17],[4],range(1,16)))
                midpoint_ids = [int(r["winner_id"]) for r in midpoint]
                require(len(set(midpoint_ids)) == 15 and all(
                    0 <= k < universe and 0 < int(r["low"]) <= int(r["high"])
                    and r["opinion"] == "signed S1"
                    and (int(left[k]),int(right[k]),int(is_flip[k])) == (int(r["i"]),int(r["j"]),int(r["flip"]))
                    for k,r in zip(midpoint_ids,midpoint)), f"{name}: insertion-weight midpoint")
            runs,rounds = read(head+"runs.csv"),read(head+"rounds.csv")
            if reps == 10:
                require(all(r.get("index_policy") == "class-aware dyadic" for r in runs+rounds),
                        f"{name}: primary index implementation")
            common = ([name],[size],[universe],range(reps),methods)
            keys = "graph n universe repetition method"
            cohort(runs,keys,product(*common))
            cohort(rounds,keys+" epoch",product(*common,range(1,16)))
            paths,intervals,admissions = {},{},{}
            for run in runs:
                method,rep = run["method"],run["repetition"]
                rows = sorted((r for r in rounds if (r["method"],r["repetition"]) == (method,rep)),
                              key=lambda r:int(r["epoch"]))
                ids = [int(r["winner_id"]) for r in rows]
                require(len(set(ids)) == 15 and paths.setdefault(method,ids) == ids,
                        f"{name}: inconsistent signed path")
                require(all(0 <= k < universe and 0 < int(r["low"]) <= int(r["high"])
                            and (int(left[k]),int(right[k]),int(is_flip[k])) ==
                                (int(r["i"]),int(r["j"]),int(r["flip"]))
                            for k,r in zip(ids,rows)), f"{name}: invalid signed edit")
                require(int(run["peak_rss"]) > 0 and float(run["total_seconds"]) > 0
                        and int(run["gain_low"]) == sum(int(r["low"]) for r in rows)
                        and int(run["gain_high"]) == sum(int(r["high"]) for r in rows),
                        f"{name}: signed run totals")
                bounds = [(int(r["low"]),int(r["high"])) for r in rows]
                require(intervals.setdefault(method,bounds) == bounds, f"{name}: signed gain drift")
                if method == "promoted":
                    for r in rows:
                        tau = int(r["tau_units"])
                        limit = int(r["tau_grid_max"])
                        require(1 <= tau <= limit and tau & (tau-1) == 0 and limit & (limit-1) == 0
                                and int(r["tau_work"]) >= int(r["visited"])
                                and int(r["tau_denominator"]) == 2**20,
                                f"{name}: adaptive threshold outside dyadic grid")
                if method == "promoted" and rep == "0":
                    for r in rows:
                        entries = json.loads(r["admissions"])
                        require(len(entries) == int(r["promoted"]), f"{name}: admission count")
                        for k,pair in entries.items():
                            require(int(k) not in admissions, f"{name}: duplicate outsider")
                            admissions[int(k)] = (int(r["epoch"]),*pair)
            require(paths["promoted"] == paths["full"] and all(
                    max(a[0],b[0]) <= min(a[1],b[1]) for a,b in
                    zip(intervals["promoted"],intervals["full"])), f"{name}: full/promoted mismatch")
            if reps == 10:
                reference = sorted((r for r in generic_rounds if r["graph"] == name),
                                   key=lambda r:int(r["epoch"]))
                require([int(r["winner_id"]) for r in reference] == paths["full"]
                        and all(0 <= int(r["visited"]) <= int(r["tau_work"])
                                and r["algorithm"] == "generic dyadic root-cap index" for r in reference),
                        f"{name}: generic posting control")
            audit = read(head+"admissions.csv") if admissions else []
            cohort(audit,"candidate",[(k,) for k in admissions])
            opinion = np.rint(OPINION_DEN*opinions(W,"S1",17)).astype("int64")
            promoted = sorted((r for r in rounds if r["method"] == "promoted" and r["repetition"] == "0"),
                              key=lambda r:int(r["epoch"]))
            live_audit,_,_,outside = sign_audit(W,opinion,(left,right,is_flip),promoted,1 if reps == 1 else 16)
            live = {int(record["candidate"]):record for record in live_audit}
            require(set(live) == set(admissions), f"{name}: reconstructed signed admissions")
            revived,root_nonpositive = set(),set()
            for r in audit:
                k = int(r["candidate"])
                require(all(str(value) == r[field] for field,value in live[k].items()),
                        f"{name}: reconstructed signed root gain/admission certificate")
                lo,hi = int(r["admission_low"]),int(r["admission_high"])
                require(admissions[k] == (int(r["epoch"]),lo,hi) and lo <= hi,
                        f"{name}: first-admission interval")
                if r["root_test"] == "endpoint separation":
                    require(not is_flip[k] and int(r["root_strictly_positive"]) == 1
                            and int(r["root_abs_endpoint_difference_low_units"]) > 0 and not r["root_high"],
                            f"{name}: strict positive insertion sign")
                else:
                    require(r["root_test"] == "Gram interval"
                            and int(r["root_gain_low"]) <= int(r["root_high"])
                            and int(r["root_strictly_positive"]) == (int(r["root_gain_low"]) > 0),
                            f"{name}: root Gram sign")
                if not int(r["root_strictly_positive"]):
                    require(int(r["root_high"]) <= 0 and (lo > 0 or hi <= 0),
                            f"{name}: unresolved revival sign")
                    root_nonpositive.add(k)
                    if lo > 0:
                        revived.add(k)
                require(int(r["revived_at_admission"]) == (k in revived), f"{name}: revival class")
            full = set(paths["full"])
            metrics = dict(admitted=len(admissions),revived_admitted=len(revived),
                           revived_winners=len(full & root_nonpositive),
                           outsider_winners=outside,
                           mismatches=sum(a!=b for a,b in zip(paths["static"],paths["full"])),
                           flip_winners=sum(int(r["flip"]) for r in rounds
                                            if r["method"] == "full" and r["repetition"] == "0"))
            for r in (r for r in summary if r["graph"] == name):
                require(all(int(r[k]) == value for k,value in metrics.items()),
                        f"{name}: signed promotion summary")
                series = [run for run in runs if run["method"] == r["method"]]
                for field in ("graph_seconds","catalog_seconds","common_seconds","rank_seconds",
                              "index_seconds","online_seconds","total_seconds"):
                    values = [float(run[field]) for run in series]
                    require(all(math.isfinite(v) and v >= 0 for v in values)
                            and np.allclose([float(r[field+k]) for k in ("_q1","","_q3")],
                                            np.quantile(values,[.25,.5,.75])),
                            f"{name}: signed promotion timing quantiles")
                reference = {run["repetition"]:run for run in runs if run["method"] == "full"}
                rng = np.random.default_rng(108)
                for field in ("online_seconds","total_seconds"):
                    ratios = [float(reference[run["repetition"]][field])/float(run[field]) for run in series]
                    boot = np.median(rng.choice(ratios,(10000,reps)),axis=1)
                    key = "paired_full_over_method_"+field
                    require(math.isclose(float(r[key]),median(ratios))
                            and np.allclose([float(r[key+k]) for k in ("_ci_low","_ci_high")],
                                            np.quantile(boot,[.025,.975])),
                            f"{name}: paired promotion timing interval")
                full_gain = [sum(pair[j] for pair in intervals["full"]) for j in (0,1)]
                static_gain = [sum(pair[j] for pair in intervals["static"]) for j in (0,1)]
                loss = (100*(full_gain[0]-static_gain[1])/full_gain[0],
                        100*(full_gain[1]-static_gain[0])/full_gain[1]) if metrics["mismatches"] else (0,0)
                require(np.allclose([float(r[k]) for k in ("loss_pct_lo","loss_pct_hi")],loss),
                        f"{name}: static loss interval")


def root_fill():
    rows=read("root_fill.csv")
    cohort(rows,"graph",[(g,) for g in SIZES])
    for row in rows:
        graph=row["graph"]
        W=load_graph(graph)
        n=SIZES[graph]
        lower=n+W.nnz//2
        occupied=int(row["factor_nnz"])
        require(int(row["n"])==n and int(row["matrix_lower_nnz"])==lower
                and lower<=occupied<=n*(n+1)//2
                and math.isclose(float(row["fill_ratio"]),occupied/lower,rel_tol=1e-12),
                f"{graph}: root lower triangle or factor fill")
        if graph=="Slashdot":
            require((lower,occupied)==(580672,51160119) and round(occupied/lower,1)==88.1,
                    "Slashdot: documented factor fill")


def persistent_fixture_records():
    from workloads import NAMES,persistent_witness
    def fixture_file(name):
        with (PAPER/"datasets"/name).open(newline="") as stream:
            return list(csv.DictReader(stream))
    runs=fixture_file("persistent_setup_raw.csv")
    summary=fixture_file("persistent_setup.csv")
    witness=fixture_file("persistent_witness.csv")
    cohort(runs,"graph M seed calibration_rhs rep",product(NAMES,(1000,2500),[17],[84],range(10)))
    cohort(summary,"graph M seed calibration_rhs reps",product(NAMES,(1000,2500),[17],[84],[10]))
    cohort(witness,"graph M round",product(NAMES,(1000,2500),range(1,41)))
    rng=np.random.default_rng(48231)
    for name in NAMES:
        W=load_graph(name)
        for m in (1000,2500):
            def matches(r):
                return r["graph"]==name and int(r["M"])==m
            times=[float(r["seconds"]) for r in sorted(filter(matches,runs),key=lambda r:int(r["rep"]))]
            require(np.isfinite(times).all() and min(times)>0, f"{name}/{m}: invalid setup time")
            boot=np.median(rng.choice(times,(20000,10)),axis=1)
            expected=[*np.quantile(times,[.25,.5,.75]),*np.quantile(boot,[.025,.975])]
            row=next(filter(matches,summary))
            actual=[float(row[k]) for k in ("q1_seconds","seconds","q3_seconds","median_ci_lo","median_ci_hi")]
            require(np.allclose(actual,expected,rtol=1e-12,atol=0),f"{name}/{m}: fixture statistics")
            stored=sorted(filter(matches,witness),key=lambda r:int(r["round"]))
            for row,live in zip(stored,persistent_witness(W,m),strict=True):
                fields=list(row)[1:]
                for key,value in zip(fields,live,strict=True):
                    if key in ("root_gain_upper","current_gain_lower"):
                        require(math.isclose(float(row[key]),value,rel_tol=1e-8,abs_tol=1e-14),
                                f"{name}/{m}: fixture gain bound {key}")
                    else:
                        require(row[key]==str(value),f"{name}/{m}: fixture witness {key}")
            require(sum(int(r["root_negative"]) for r in stored[:20])>=2,
                    f"{name}/{m}: multiple root-negative winners in first checkpoint")


def persistent_plans():
    """Check every physical arm against the independently reconstructed fixture path."""
    from planned_report import catalogue_routes
    routes = catalogue_routes()
    with (PAPER/"datasets/persistent_witness.csv").open(newline="") as stream:
        witness = list(csv.DictReader(stream))
    for graph,m in product(SIZES,(1000,2500)):
        path = [int(r["winner"]) for r in sorted(witness,key=lambda r:int(r["round"]))
                if r["graph"]==graph and int(r["M"])==m]
        choices = {
            "persistent_batch": tuple(product(BACKENDS,("cold","same","planner"))),
            "persistent_quad": (("cg","quad"),),
            "linview_persistent": (("cholmod","linview"),),
            "linview_registered_persistent": (("cholmod","linview_registered"),),
        }
        if routes[graph][0]=="cg":
            choices["persistent_native"]=(("cg","planner"),)
        else:
            choices["direct6"] = tuple(product(("cholmod",),("planner","planner6")))
            choices["direct6_slots"] = tuple(product(("cholmod",),("planner6","slots6")))
        if graph in ("Alpha","Fin"):
            choices["linview_registered_lookup_persistent"]=(("cholmod","linview_registered_lookup"),)
        for prefix,arms in choices.items():
            stem=f"{prefix}_{graph}_{m}"
            rows=read(stem+".csv")
            for shard in DATA.glob(stem+".*.csv"):
                rows.extend(read(shard.name))
            cohort(rows,"graph M rep backend mode b",
                   ((graph,m,rep,backend,mode,b) for (backend,mode),rep,b in
                    product(arms,range(10),(20,40))))
            for r in rows:
                b=int(r["b"])
                require(int(r["commits"])==b and json.loads(r["sequence"])==path[:b],
                        f"{stem}: measured winner path disagrees with fixture")
                if prefix.startswith("linview_registered"):
                    c=json.loads(r["counters"])
                    setup,online,total,population,update=(float(r[k]) for k in
                        ("setup_seconds","online_seconds","seconds","population_seconds","registered_update_seconds"))
                    require(0<population<=setup and 0<update<=online and math.isclose(total,setup+online)
                            and r["population_reused"]==str("lookup" in prefix),
                            f"{stem}: registered setup, population or update timing boundary")
                    scores=b*m-b*(b-1)//2
                    require(c["registered_rows_populated"]==m and c["registered_rows_updated"]==b*m
                            and c["population_rhs"]>0 and c.get("registered_scores",0)==
                            (0 if "lookup" in prefix else scores)
                            and ("lookup" not in prefix or c["screens"]==scores),
                            f"{stem}: population, update or exact score count")


def graph_profiles():
    from workloads import profile,response_profile
    published = (PAPER/"sections/datasets.tex").read_text(encoding="utf8")
    with (PAPER/"datasets/grounded_response.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    cohort(rows,"network edge_count_sampled seed",product(SIZES,[64],[48231]))
    for r in rows:
        name = r["network"]
        W = load_graph(name)
        require(W.shape == (SIZES[name],SIZES[name]) and W.has_canonical_format
                and W.nnz>0 and (W != W.T).nnz==0 and not W.diagonal().any()
                and (abs(W.data)==1).all(), f"{name}: invalid complete signed operator")
        fields = ("max_degree","edge_count_sampled","seed","Q1","median","Q3","max_residual_error_bound")
        n,edges,neg,tri,unbalanced,diam = profile(W)
        line = (f"{name} & {n:,} & {edges:,} & {100*neg/edges:.1f}" + r"\% & " +
                f"{100*unbalanced/tri:.1f}" + r"\% & " + f"{diam:,}" + r"\\")
        require(published.splitlines().count(line)==1, f"{name}: published graph metadata")
        actual = response_profile(W)
        require(np.allclose([float(r[k]) for k in fields],actual,rtol=1e-8,atol=1e-13),
                f"{name}: grounded response quartiles or residual enclosure")


def clustered_matrix():
    from workloads import NAMES, clustered_witness
    with (PAPER/"datasets/clustered_witness.csv").open(newline="") as f:
        records = list(csv.DictReader(f))
    cohort(records,"graph M",product(NAMES,(1000,2500)))
    for r in records:
        g,m = r["graph"],int(r["M"])
        independent = clustered_witness(load_graph(g),m)
        require(list(map(int,list(r.values())[1:])) == list(independent),f"{g}/{m}: integer revival")
        require(int(r["root_hi"]) < 0 < int(r["child_lo"]) and int(r["two_hop_adds"]) >= 64
                and int(r["longer_adds"]) == (362 if (g,m)==("Bio",2500) else 0),
                f"{g}/{m}: distance-two catalog and root-negative winner")
    methods = read("clustered_methods.csv")
    cohort(methods,"graph M b backend mode",
           [(g,k,b,backend,mode) for g,k,b in product(SIZES,(1000,2500),(20,40))
            for backend,mode in [*product(BACKENDS,("cold","same","planner")),("quad","quad"),("lv","lv")]])
    planned_matrix(clustered=True)


def check_clustered_cell(name,m,rows,quad,paths,edits):
    with (PAPER/"datasets/clustered_witness.csv").open() as f:
        w = next(r for r in csv.DictReader(f) if r["graph"]==name and int(r["M"])==m)
    require(paths[40][:2]==[int(w["winner1"]),int(w["winner2"])],f"{name}/{m}: stale fixture")
    for rep in range(10):
        order = list(product(BACKENDS,("cold","same","planner")))
        random.Random(79+rep).shuffle(order)
        observed = [(r["backend"],r["mode"]) for r in rows if int(r["rep"])==rep and r["b"]=="40"]
        if name=="Slashdot" and (m==2500 or rep>0):
            for backend in BACKENDS:
                block = list(product([backend],("cold","same","planner")))
                random.Random(79+rep).shuffle(block)
                require([r for r in observed if r[0]==backend]==block,
                        f"{name}/{m}/{rep}: randomized within-backend block order")
        else:
            require(observed == order, f"{name}/{m}/{rep}: randomized six-method block order")
    lv = read(f"linview_clustered_{name}_{m}.csv")
    cohort(lv,"graph M rep backend mode b",product([name],[m],range(10),["cholmod"],["linview"],(20,40)))
    for r in lv:
        path = json.loads(r["sequence"])
        counts = Counter(json.loads(r["counters"]))
        n,b = int(r["n"]),int(r["b"])
        rank = sum(1+int(edits[e].flip) for e in path)
        require(n==SIZES[name] and path==paths[b] and len(path)==int(r["commits"])
                and int(r["history_bytes"])==16*n*rank and int(r["winner_bytes"])==0
                and int(r["memory_bytes"])>=int(r["history_bytes"])+int(r["candidate_bytes"])
                and counts["state_rhs"]==counts["factorizations"]==1
                and counts["base_action_rhs"]>=rank and int(r["peak_winner_bytes"])<=16*n,
                f"{name}/{m}: LINVIEW path, persistent factors or history")
    methods = read("clustered_methods.csv")
    from planned_report import independent_ratio
    for r in (r for r in methods if r["graph"]==name and int(r["M"])==m):
        b,backend,mode = r["b"],r["backend"],r["mode"]
        group = [x for x in (lv if backend=="lv" else quad if backend=="quad" else rows)
                 if x["b"]==b and (backend in ("lv","quad") or (x["backend"],x["mode"])==(backend,mode))]
        ref = [x for x in rows if (x["b"],x["backend"],x["mode"])==(b,"cholmod","planner")]
        group,ref = ({int(x["rep"]):x for x in arr} for arr in (group,ref))
        times = sorted(float(x["seconds"]) for x in group.values())
        if backend in ("quad","lv") or (name=="Slashdot" and backend=="cg"):
            ratio = independent_ratio(group,ref)
            require(r["pairs"]=="","external baselines require independent intervals")
        else:
            vals = sorted(float(group[i]["seconds"])/float(ref[i]["seconds"]) for i in range(10))
            ratio = (median(vals),vals[1],vals[-2])
            require(int(r["pairs"])==10,"paired ratio block count")
        summaries = [json.loads(x["counters"]) for x in group.values()]
        counts = {k:median(c.get(k,0) for c in summaries) for k in set().union(*summaries)}
        require(json.loads(r["counters"])==counts and int(r["reps"])==10
                and np.allclose([float(r[f]) for f in ("seconds","seconds_lo","seconds_hi",
                    "over_cholmod_planner","ratio_lo","ratio_hi")],
                    [median(times),times[1],times[-2],*ratio],rtol=1e-12,atol=0),
                f"{name}/{m}/{mode}: matrix uncertainty or accounting")


def moved_branches():
    from branch_table import render
    require((PAPER/"sections/branch_rows.tex").read_text()==render(), "branch rows differ from timed trials")
    from workloads import matched_inputs
    for name in ("Alpha","OTC"):
        rows = read(f"branch_move_{name.lower()}.csv")
        cohort(rows,"rep level",product(range(10),range(4)))
        paths = json.loads(rows[0]["paths"])
        _,edits = matched_inputs(load_graph(name),1000)
        budgets = [20,10,15,25]*3
        require(len(paths)==12 and all(len(p)==len(set(p))==b and all(
            type(e) is int and 0<=e<1000 and (d<3 or q%6==0 or (e+q)%7!=0) for d,e in enumerate(p))
            for q,(p,b) in enumerate(zip(paths,budgets))),f"{name}: restricted paths")
        states = {(q//6,tuple(p[:d])) for q,p in enumerate(paths) for d in range(len(p))}
        changes = {tuple(p[:d]) for p in paths for d in range(1,len(p))}
        flips = sum(edits[e][3] for p in paths for e in p)
        require(all(json.loads(r["paths"])==paths and int(r["flips"])==flips for r in rows),
                f"{name}: path or flip count differs across levels")
        for rep in range(10):
            block = [r for r in rows if int(r["rep"])==rep]
            order = list(range(4))
            random.Random(17+rep).shuffle(order)
            require([int(r["level"]) for r in block]==order, f"{name}: branch block order")
            counters = {int(r["level"]):Counter(json.loads(r["counters"])) for r in block}
            shared = set()
            for level,c in counters.items():
                roots = 1 if level else 12
                forks,moves,recoveries = (c[k] for k in ("fork_clones","moved_edges","planner_parent_clones"))
                require(c["root_clones"]==roots and c["factor_clones"]==roots+forks+recoveries
                        and c["factor_up"]==forks+moves==(len(changes) if level else sum(b-1 for b in budgets))
                        and c["factor_down"]==(sum(edits[p[-1]][3] for p in changes) if level else
                            sum(edits[e][3] for p in paths for e in p[:-1]))
                        and c["state_rhs"]==(len(states) if level else sum(budgets))
                        and c["refine_rhs"]==sum(c[k+"_rhs"] for k in "CFR")
                        and c["W_rhs"]==c["winner_rhs"] and c["planner_undo_down"]==recoveries
                        and (level==3 or recoveries==c["transported"]==0),
                        f"{name}/{level}: moved branch state/factor accounting")
                if level:
                    shared.add((roots,forks,moves,c["factor_up"],c["factor_down"],c["state_rhs"]))
            require(len(shared)==1 and counters[0]["refine_rhs"]==counters[1]["refine_rhs"],
                    f"{name}: unequal L1-L3 schedules or L0/L1 demand")


def promotion_v4(names=(*SIZES,"AlphaClamp","AlphaMotif"),reps=10,edition=4,*,fine=False):
    from promotion_v4 import METHODS,query_opinion
    from signed_promotion import catalog,sign_audit
    from promotion_inputs5 import clamped_source
    methods=("static","adaptive","full","transport") if edition==5 else METHODS
    for name in names:
        W=load_graph("Alpha" if name in ("AlphaMotif","AlphaBlend","AlphaClamp") else name)
        metadata={}
        s=(clamped_source if edition==5 else query_opinion)(W,name,metadata=metadata)
        pool=catalog(W,name)
        unit=1<<(48 if edition==5 or name in ("AlphaMotif","AlphaBlend","AlphaClamp") else 40)
        label=name+"_fine" if fine else name
        runs=read(f"promotion_v{edition}_{label}_runs.csv")
        rows=read(f"promotion_v{edition}_{label}_rounds.csv")
        keys="graph n universe seed proof_unit query method"
        common=([name],[len(s)],[len(pool[0])],[17],[unit],range(reps),methods)
        cohort(runs,keys,product(*common))
        cohort(rows,keys+" epoch",product(*common,range(1,16)))
        if name in (*SIZES,"AlphaClamp"):
            require(all(r.get("guarded")=="1" and 0<float(r["bind_seconds"])<math.inf for r in runs),
                    f"{name}: missing prepared-root protection or binding time")
            for field in ("graph_seconds","catalog_seconds","common_seconds","rank_seconds"):
                require(len({r[field] for r in runs})==1 and 0<float(runs[0][field])<math.inf,
                        f"{name}: shared preparation {field}")
            for method in methods:
                arm=[r for r in runs if r["method"]==method]
                require(len({(r["bind_seconds"],r["index_seconds"]) for r in arm})==1,
                        f"{name}/{method}: preparation drift across cursors")
        canonical={}
        for q in range(reps):
            block=[r for r in runs if int(r["query"])==q]
            require(tuple(r["method"] for r in block)==methods[q%4:]+methods[:q%4],
                    f"{name}: cursor schedule")
            bounds={}
            for run in block:
                method=run["method"]
                trace=sorted((r for r in rows if r["query"]==str(q) and r["method"]==method),
                             key=lambda r:int(r["epoch"]))
                ids=[int(r["winner_id"]) for r in trace]
                require(len(set(ids))==15 and all(0<=k<len(pool[0]) and
                        (int(r["i"]),int(r["j"]),int(r["flip"]))==tuple(int(x[k]) for x in pool)
                        and 0<int(r["low"])<=int(r["high"]) for k,r in zip(ids,trace)),
                        f"{name}/{method}: illegal winner or uncertified gain")
                if edition==5 or name=="AlphaClamp":
                    require(0<float(run["clamp_aux_seconds"])<=float(run["graph_seconds"])
                            and all(math.isclose(float(run[k]),value,rel_tol=1e-11)
                                    for k,value in metadata.items() if k!="clamp_aux_seconds"),
                            "AlphaClamp: uncharged auxiliary solve or background scaling")
                if edition==5:
                    require(int(run.get("fine_screen",0))==int(fine),f"{label}: screen precision")
                expected_solver = "paired PCG" if name=="Slashdot" else "sparse LU"
                require(run["directional"]=="1" and run["solver"]==expected_solver,
                        f"{name}: directional theorem or numerical backend")
                require(canonical.setdefault(method,ids)==ids and int(run["peak_rss"])>0
                        and 0<float(run["online_seconds"])<math.inf
                        and all(int(run[g])==sum(int(r[h]) for r in trace)
                                for g,h in (("gain_low","low"),("gain_high","high"))),
                        f"{name}/{method}: inconsistent run totals")
                for idx,r in enumerate(trace):
                    if idx:
                        require(r["state_source"]==trace[idx-1]["child_state_source"]
                                and r["state_residual"]==trace[idx-1]["child_state_residual"],
                                f"{name}: disconnected state transition")
                    else:
                        require(r["state_source"]=="shared" and float(r["state_s"])==0,
                                f"{name}: cursor root was not shared")
                    require(r["winner_response_source"] in ("recomputed","retained")
                            and int(r["winner_blocks"])==idx+1
                            and (method!="full" or int(r["visited"])==len(pool[0])-idx)
                            and (method!="fixed" or int(r["tau_units"])==256)
                            and all(0<=float(r[k])<math.inf for k in ("child_state_s","winner_correction_s")),
                            f"{name}: correction, full-scan or fixed threshold accounting")
                    require(int(r["state_residual"])>=0 and int(r["child_state_residual"])>=0
                            and int(r["promoted"])==len(json.loads(r["admissions"]))
                            and int(r["winner_blocks"])>=0, f"{name}: state/admission accounting")
                    if edition==5:
                        visits=int(r["catalog_visits"])
                        require(r["transport_enabled"]==str(int(method!="full"))
                                and int(r["seed_visits"])>=0 and visits>=0
                                and (method not in ("full","transport") or visits==len(pool[0]))
                                and (method!="static" or visits==0)
                                and (method!="adaptive" or visits==int(r["visited"])),
                                f"{name}/{method}: kernel, seed or actual catalog scan accounting")
                    if method in ("adaptive","fixed"):
                        t=int(r["tau_units"])
                        require(t>0 and t&(t-1)==0 and int(r["tau_work"])>=int(r["visited"])
                                and int(r["tau_denominator"])==2**20,f"{name}: posting threshold grid")
                bounds[method]=[(int(r["low"]),int(r["high"])) for r in trace]
            for method in ("adaptive",methods[-1]):
                require(canonical[method]==canonical["full"] and all(max(a,c)<=min(b,d)
                        for (a,b),(c,d) in zip(bounds[method],bounds["full"])),f"{name}: mixed full scan")
        reference=[r for r in rows if r["query"]=="0" and r["method"]=="adaptive"]
        audit,_,_,outside=sign_audit(W,s,pool,reference,16,unit)
        file=f"promotion_v{edition}_{label}_admissions.csv"
        published=read(file) if (DATA/file).exists() else []
        cohort(published,"candidate",[(r["candidate"],) for r in audit])
        live={r["candidate"]:r for r in audit}
        for r in published:
            require(all(r[k]==str(v) for k,v in live[int(r["candidate"])].items()),
                    f"{name}: admission interval differs from recomputed root sign")
        if edition==5:
            from promotion_v4 import check_v5
            check_v5(name,reps,fine_screen=fine)
            from long_promotion import Iterative, UpdatedFactor, certified_state, seed_view
            A=16*system(W).astype("int64")
            solver=UpdatedFactor(A) if len(s)<=10000 else Iterative(A)
            _,_,_,center,radius=certified_state(A,solver,s,unit)
            seed,_=seed_view(pool,center,radius,A.diagonal(),64,False,16)
            require(len(seed)==64 and set(canonical["static"])<=seed,
                    f"{name}: static path leaves actual top-64")
            require(all(int(r["seed_visits"])==64-int(r["epoch"])+1 for r in rows
                        if r["method"]=="static"), f"{name}: static screening count")
            a,b,c=(metadata[k] for k in ("clamp_center","clamp_first","clamp_target"))
            first,second=canonical["full"][:2]
            require({a,b}=={int(pool[0][first]),int(pool[1][first])}
                    and {a,c}=={int(pool[0][second]),int(pool[1][second])}
                    and second not in seed and live[second]["root_high"]<0<live[second]["admission_low"],
                    f"{name}: designated root-negative winning outsider")
            for method in methods:
                trace0=[r for r in rows if r["query"]=="0" and r["method"]==method]
                by_epoch={r["epoch"]:r for r in trace0}
                for row in (r for r in rows if r["method"]==method):
                    baseline=by_epoch[row["epoch"]]
                    require(all(row[k]==baseline[k] for k in ("winner_id","seed_visits","catalog_visits",
                            "scores","fresh","transfers","promoted","admissions","low","high")),
                            f"{name}/{method}: reset decision or work ledger changed")
        if name=="AlphaBlend":
            require(not audit and outside==0, "Alpha blend: failed stress control changed")
        if edition==5 or name in ("AlphaMotif","AlphaClamp"):
            require(outside>0 and any(r["candidate"] in canonical["full"] and r["flip"]
                    and r["root_high"]!="" and r["root_high"]<0<r["admission_low"] for r in audit),
                    "no strictly negative outside-root flip became a winner")


def precision_sensitivity():
    from promotion_v4 import precision_records
    expected=precision_records(("Alpha",))
    actual=read("promotion_v5_precision_summary.csv")
    fields=("graph","method","fine_screen")
    reference={tuple(str(row[k]) for k in fields):row for row in expected}
    cohort(actual," ".join(fields),reference)
    for row in actual:
        match=reference[tuple(row[k] for k in fields)]
        require(set(row)==set(match) and all(math.isclose(float(row[k]),value,rel_tol=1e-12)
                for k,value in match.items() if k not in fields),"Alpha precision summary mismatch")


def promotion_v6():
    subprocess.run([sys.executable,str(PAPER/"tools/promotion_v4.py"),"--check-v6"],check=True)


def promotion_v5():
    promotion_v4(SIZES,10,edition=5,fine=True)
    promotion_v4(("Alpha",),10,edition=5)
    traces=[]
    for label in ("Alpha","Alpha_fine"):
        rows=read(f"promotion_v5_{label}_rounds.csv")
        traces.append({(r["method"],r["query"],r["epoch"]):r for r in rows})
    for key,coarse in traces[0].items():
        fine=traces[1][key]
        require(fine["winner_id"]==coarse["winner_id"]
                and max(int(fine["low"]),int(coarse["low"]))<=min(int(fine["high"]),int(coarse["high"])),
                f"Alpha coarse/fine control: winner or gain interval {key}")


def blended_promotion_control():
    promotion_v4(("AlphaBlend",),1)


def s2_control():
    branch_queries(("branch_s2",))


def promotion_table():
    from promotion_table_v4 import NAMES,table_row,timing_statistics
    rows = read("promotion_v4_timing_summary.csv")
    expected = {(r["graph"],r["method"]):r for name in NAMES for r in timing_statistics(name)}
    cohort(rows,"graph method",expected)
    for row in rows:
        reference = expected[row["graph"],row["method"]]
        require(all(math.isclose(float(row[k]),float(v),rel_tol=1e-12,abs_tol=0)
                    for k,v in reference.items() if k not in ("graph","method")),
                "promotion timing interval, paired ratio or shared setup mismatch")
    source = (PAPER/"sections/promotion_results.tex").read_text(encoding="utf-8")
    published = source.split(r"\midrule",1)[1].split(r"\bottomrule",1)[0].strip()
    require(published == "\n".join(table_row(name) for name in NAMES),
            "promotion table differs from measured run and admission records")


def scalar_promotion():
    from promotion_v4 import METHODS
    name="Slashdot"
    base=read(f"promotion_v4_{name}_rounds.csv")
    indexed={(r["method"],r["query"],r["epoch"]):r for r in base}
    rounds=read(f"promotion_v4_{name}_scipy_rounds.csv")
    runs=read(f"promotion_v4_{name}_scipy_runs.csv")
    for records,epochs in ((runs,False),(rounds,True)):
        keys="graph query method"+(" epoch" if epochs else "")
        factors=([name],range(3),METHODS,*([range(1,16)] if epochs else []))
        cohort(records,keys,product(*factors))
    for r in rounds:
        other=indexed[r["method"],r["query"],r["epoch"]]
        require(r["winner_id"]==other["winner_id"] and
                max(int(r["low"]),int(other["low"]))<=min(int(r["high"]),int(other["high"])),
                "Slashdot: scalar and two-column PCG decisions or intervals differ")
    require(all(0<float(r["online_seconds"])<math.inf and int(r["peak_rss"])>0 for r in runs),
            "Slashdot: scalar PCG control costs")

    state = read("promotion_v4_Slashdot_state_control.csv")
    cohort(state,"epoch",product(range(1,16)))
    reference = {r["epoch"]:r for r in rounds if r["method"]=="adaptive" and r["query"]=="0"}
    for r in state:
        epoch = reference[r["epoch"]]
        require(all(r[k]==epoch[k] for k in ("winner_id","i","j","flip","proof_unit"))
                and r["woodbury_residual"]==epoch["child_state_residual"]
                and int(r["woodbury_iterations"])>=0 and int(r["cg_iterations"])>0
                and all(0<float(r[k])<math.inf for k in ("woodbury_s","cg_s"))
                and int(r["min_overlap_slack"])==int(r["woodbury_residual"])+int(r["cg_residual"])
                    -int(r["max_center_difference"])>=0,
                "Slashdot: state residuals or independent state-box intersection")


def scale_inputs():
    from scale_frontier import SIZES as sizes, ARMS, dataset, records
    for n in sizes:
        base, = read(f"scale_social_v5_{n}_audit.csv")
        union, = read(f"scale_social_v5_{n}_union_audit.csv")
        edges=int(base["edges"])
        require(int(base["n"])==int(union["n"])==n and int(union["edges"])==edges
                and int(base["shortcuts"])==n//32 and math.isclose(float(base["degree_mean"]),2*edges/n)
                and int(union["symbolic_nnz"])>=n+edges,
                f"{n}: graph or symbolic profile")
        methods=ARMS if n!=500000 else ("cholmod","registered")
        cohort(records(dataset(n)),"n method rep",product([n],methods,range(10)))
        if n in (10000,50000):
            cohort(read(f"scale_frontier_v6_{n}_deferred.csv"),"n method rep",
                   product([n],["deferred"],range(10)))


def scale_figure():
    import tempfile
    import pymupdf
    from unittest.mock import patch
    from scale_frontier import ARMS, SIZES as sizes, dataset, records, render_v6
    from native_scale_frontier import output
    stats = {}
    for n in sizes:
        native = records(output(n))
        direct = records(dataset(n))
        for method in ARMS:
            rows = native if method in ("cold","planner") else direct
            arm = [r for r in rows if r["method"]==method]
            cohort(arm,"n rep",product([n],range(10)))
            stats[n,method] = np.quantile(
                [[float(r["seconds"]),float(r["peak_mib"])] for r in arm],[.25,.5,.75],axis=0)
    # Render with the pinned plotting stack. Comparing pixels ignores PDF timestamps.
    with tempfile.TemporaryDirectory() as directory:
        target=Path(directory)
        (target/"figures").mkdir()
        with patch("scale_frontier.ROOT",target):
            render_v6(stats)
        name="figures/scale_views.pdf"
        with pymupdf.open(PAPER/name) as actual, pymupdf.open(target/name) as expected:
            require(len(actual)==len(expected)==1 and actual[0].rect==expected[0].rect,
                    "Figure 3 page bounds differ")
            require(actual[0].get_pixmap(dpi=200).samples==expected[0].get_pixmap(dpi=200).samples,
                    "Figure 3 differs from the four measured methods at four sizes")


def sparse_search():
    from sparse_search_report import render_table
    expected=render_table()
    for name in ("Alpha","OTC"):
        anchor=read(f"sparse_trap_{name}.csv")
        cohort(anchor,"mode repeat policy",product(("greedy","beam"),range(10),("cold","transport")))
        trap=json.loads((DATA/f"sparse_trap_{name}.json").read_text())
        require(trap["budget"]==2 and len(trap["edits"])==3, f"{name}: sparse trap fixture")
    published=(PAPER/"sections/sparse_search_rows.tex").read_text(encoding="utf8")
    require(published==expected,"search table differs from verified runs")


def margin_stress():
    rows=read("margin_stress.csv")
    targets=(.01,.0001,.000001,.00000001)
    cohort(rows,"target rep",product(targets,range(10)))
    for r in rows:
        t=float(r["target"])
        i=targets.index(t)
        lo,hi=float(r["gap_lower"]),float(r["gap_upper"])
        require(int(r["candidates"])==1152 and 0<lo<=t<=hi and max(t-lo,hi-t)<2e-13
                and (int(r["winner"]),int(r["runner_up"]))==(1051,1070)
                and int(r["state_refinements"])==(0,1,1,2)[i] and int(r["refinements"])==(0,2,2,4)[i]
                and 0<float(r["final_radius"])<=float(r["first_radius"])
                and all(0<float(r[k])<math.inf for k in ("setup_s","query_s")),
                "margin sweep failed to certify the requested gap or winner")

    path = read("margin_path.csv")
    cohort(path,"target step rep",product(targets,(1,2),range(10)))
    steps = {(r["target"],r["step"],r["rep"]):r for r in path}
    for r in path:
        step = int(r["step"])
        lo,hi = [float(r[k]) for k in ("gap_lower","gap_upper")]
        require(int(r["candidates"])==1153-step and int(r["winner"])==(1051,1070)[step-1]
                and 0<lo<=hi and float(r["winner_lower"])>max(0.,float(r["other_upper"]))
                and float(r["winner_upper"])>=float(r["winner_lower"]) and float(r["max_ref_radius"])>0
                and all(0<float(r[k])<math.inf for k in ("setup_s","query_s")),
                "margin sweep: child path or energy separation")
        if step==2:
            root=steps[r["target"],"1",r["rep"]]
            require(lo>float(root["gap_upper"]), "margin sweep: initial gap does not limit path")
        else:
            t = float(r["target"])
            require(lo<=t<=hi and max(t-lo,hi-t)<2e-13, "margin sweep: root gap")


def scale_study():
    from scale_views import summarize
    from view_memory import verify
    summarize()
    for graph in SIZES:
        verify(graph)


CHECKS = (graph_profiles,root_fill,persistent_fixture_records,persistent_plans,primary_intervals,supporting_intervals,moved_branches,
          promotion_v5,promotion_v6,precision_sensitivity,scale_inputs,scale_figure,sparse_search,margin_stress,stress_grid,directional_stress,main_witness)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=[f.__name__ for f in CHECKS])
    args = parser.parse_args()
    failures = []
    for check in CHECKS:
        if args.only and args.only != check.__name__:
            continue
        try:
            check()
        except (ValueError, KeyError, OSError, AssertionError, subprocess.CalledProcessError) as exc:
            failures.append(check.__name__)
            print(f"{check.__name__} failed: {type(exc).__name__}: {exc}", flush=True)
        else:
            print(check.__name__, "passed", flush=True)
    if failures:
        raise SystemExit("Failed checks: " + ", ".join(failures))



