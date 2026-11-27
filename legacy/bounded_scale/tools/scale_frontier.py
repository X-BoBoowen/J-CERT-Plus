"""Fixed-catalog scale study.

  python tools/scale_frontier.py --v6
  python tools/scale_frontier.py --v6 --check
  python tools/scale_frontier.py --v6 --report
  python tools/scale_frontier.py --v6 --n 10000 50000 --methods deferred

NativeCostPlanner with cached=False/True supplies the two primary PCG curves
from scale_frontier_v6_native_{n}.csv. The separate direct pair uses
FastCostPlanner with catalog-union CHOLMOD and RegisteredDelta with its own
root factor and eager table. Their rows are in scale_frontier_v6_{n}.csv.
Below 500k these direct rows share randomized blocks with a matched SciPy CG
pair. The position column is the slot in the seeded permutation. At 500k
the two direct arms run in its relative order, skipping the CG slots, and the
unchanged SciPy pair comes from scale_frontier_v5_500000.csv. --v6 --scipy
--report renders that sensitivity. Each child builds its graph and calibrates
its opinion, then times solver setup and 15 commits. The prebuilt kernels are
loaded on the host. The separate deferred controls use GlobalDelta on the
same inputs and are written to scale_frontier_v6_{n}_deferred.csv. Unfinished
workers stopped for the live 640 MiB memory reserve are retried. Only
completed paths enter the summaries. With no --v6 the v5 interface is used.
"""
import argparse
import csv
import gc
import json
import ctypes as ct
import os
import subprocess
import sys
from time import perf_counter, sleep

from scale_views import FastCostPlanner, ROOT, fixture, headroom_mib, np, peak_mib


def calibrated_rhs(A, edits):
    kernel = FastCostPlanner(A, edits, backend="cg", winner_slots=1)
    z = np.zeros(A.shape[0])
    try:
        for q, edge in enumerate(edits):
            _, gram, _ = kernel._fresh(q, "C")
            signs = [1., -1.] if edge.flip else [1.]
            pivot = gram + np.diag(signs) / edge.w
            unit_gain = np.linalg.inv(pivot)[0, 0]
            target = .1 * .70**(q//8) * (1 - .0005*(q%8))
            z[edge.i] = .5 * np.sqrt(target/unit_gain)
            z[edge.j] = -z[edge.i]
    finally:
        kernel.close()
    return A @ z


def trial(n, method, *, cg_factory=FastCostPlanner, rhs_transform=None,
          allow_early_stop=False):
    A, edits, _, meta = fixture(n)
    before = perf_counter()
    s = calibrated_rhs(A, edits)
    calibration_seconds = perf_counter() - before
    if rhs_transform is not None:
        s = rhs_transform(A, edits, s)
    gc.collect()
    if method in ("registered", "deferred"):
        from linview_delta import trial as delta_trial
        row, = delta_trial(A, edits, s, budgets=(15,), backend="cholmod",
                           registered=method == "registered")
        return dict(n=n, edges=meta["edges"], method=method, seconds=row["seconds"],
                    setup_seconds=row["setup_seconds"], calibration_seconds=calibration_seconds,
                    peak_mib=peak_mib(), sequence=row["sequence"], frontiers="",
                    counters=row["counters"])
    start = perf_counter()
    if method == "cholmod":
        from cholmod_backend import CholmodFactor
        from scale_views import factor_input
        os.environ["JCERT_SCALE_CONFIG"] = "union"
        factor = CholmodFactor(factor_input(A, edits))
        kernel = FastCostPlanner(A, edits, backend="cholmod", winner_slots=1,
                                 root_factor=factor, own_root_factor=True)
        kernel.ledger["factorizations"] += 1
    else:
        kernel = cg_factory(A, edits, backend="cg", winner_slots=1)
    kernel.cached = method != "cold"
    setup_seconds = perf_counter() - start
    path, frontiers = [], []
    try:
        kernel.ledger["state_rhs"] += 1
        z, rho = kernel.solve(s)
        for step in range(15):
            old = kernel.ledger["directional_scores"]
            e, _ = kernel.select_solved(s, z, rho, delta=1e-9)
            if e is None:
                if allow_early_stop:
                    break
                raise ArithmeticError("unexpected early termination")
            path.append(int(e))
            frontiers.append(kernel.ledger["directional_scores"] - old)
            kernel.commit(e)
            if step < 14:
                z, rho = kernel.solve_committed_state(s, z)
        return dict(n=n, edges=meta["edges"], method=method,
                    seconds=perf_counter()-start, setup_seconds=setup_seconds,
                    calibration_seconds=calibration_seconds, peak_mib=peak_mib(),
                    sequence=json.dumps(path), frontiers=json.dumps(frontiers),
                    counters=json.dumps(dict(kernel.ledger),sort_keys=True))
    finally:
        kernel.close()


def check(rows):
    pairs = {(rep, method) for rep in range(10) for method in ("planner", "cold")}
    if len(rows) != 20 or {(int(r["rep"]), r["method"]) for r in rows} != pairs:
        raise ValueError("expected exactly ten complete cold/planner pairs")
    frontiers = [8, 7, 6, 5, 4, 3, 2, 1, 7, 6, 6, 5, 4, 3, 2]
    common = dict(C=16, C_rhs=24, screens=375, directional_scores=sum(frontiers),
                  peak_winner_column_bytes=8000000)
    ledgers = {
        "planner": dict(F=11, F_rhs=14, refine_rhs=38, winner_rhs=5, state_rhs=8,
                        state_warm_hits=7, gate_requests=53, transported=42,
                        gate_opens=42, cg_iterations=814),
        "cold": dict(F=53, F_rhs=75, refine_rhs=99, winner_rhs=0, state_rhs=1,
                     state_warm_hits=14, gate_requests=0, transported=0,
                     gate_opens=0, cg_iterations=1593),
    }
    for row in rows:
        context = f"rep {row['rep']}, {row['method']}"
        if int(row["n"]) != 500000 or int(row["edges"]) != 1644524:
            raise ValueError(f"{context}: incorrect graph dimensions")
        if json.loads(row["sequence"]) != list(range(15)) or json.loads(row["frontiers"]) != frontiers:
            raise ValueError(f"{context}: winner or frontier sequence differs")
        counters = json.loads(row["counters"])
        for key, expected in (common | ledgers[row["method"]]).items():
            if counters.get(key, 0) != expected:
                raise ValueError(f"{context}: incorrect {key}")
        for key in ("seconds", "setup_seconds", "calibration_seconds", "peak_mib"):
            value = float(row[key])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{context}: invalid {key}")
        if float(row["setup_seconds"]) >= float(row["seconds"]):
            raise ValueError(f"{context}: setup exceeds query time")


def legacy():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500000)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--child", choices=["planner","cold"])
    mode.add_argument("--check", action="store_true", help="validate the published 500k table")
    parser.add_argument("--reps", type=int, default=1)
    args = parser.parse_args()
    path = ROOT / "data" / f"scale_frontier_v5_{args.n}.csv"
    if args.check:
        with path.open(newline="") as stream:
            check(list(csv.DictReader(stream)))
        print("Verified ten 500k frontier pairs.")
    elif args.child:
        print(json.dumps(trial(args.n, args.child)))
    else:
        rows = list(csv.DictReader(path.open(newline=""))) if path.exists() else []
        for rep in range(args.reps):
            paired = {}
            for method in np.random.default_rng(17+rep).permutation(["cold","planner"]):
                row = next((r for r in rows if int(r["rep"]) == rep and r["method"] == method), None)
                if row is None:
                    while min(headroom_mib()) < 1000:
                        print("Waiting for 1,000 MiB of process headroom.", flush=True)
                        sleep(20)
                    call = [sys.executable,__file__,"--n",str(args.n),"--child",method]
                    row = dict(rep=rep, **json.loads(subprocess.check_output(call,text=True,
                               creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))))
                    with path.open("a",newline="") as out:
                        writer = csv.DictWriter(out,fieldnames=row)
                        if out.tell() == 0:
                            writer.writeheader()
                        writer.writerow(row)
                paired[method] = json.loads(row["sequence"])
            if paired["cold"] != paired["planner"]:
                raise ArithmeticError("paired winner sequences differ")


SIZES = (10000, 50000, 100000, 500000)
ARMS = ("cold", "planner", "cholmod", "registered")


def dataset(n):
    return ROOT / "data" / f"scale_frontier_v6_{n}.csv"


def records(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def guarded_trial(n, method):
    from scale_views import read
    large = n == 500000 and method in ("cholmod", "registered", "deferred")
    needed = 1050 if n<100000 else 1350
    if large:
        measured_root = read(n,"cholmod","union") if method=="cholmod" else read(n,"linview","")
        # The clean archive omits these earlier timing files. In their
        # absence allow 2.25 GiB for the worker (observed peaks <2.1 GiB)
        # in addition to the live 640 MiB host reserve.
        needed = 640 + max((float(r["peak_mib"]) for r in measured_root), default=2304)
    while min(headroom_mib()) < needed:
        print("Waiting for benchmark headroom", n, method, flush=True)
        sleep(20)
    limit = int(min(6500,headroom_mib()[1]-640)) if large and os.name=="nt" else 0
    cmd = [sys.executable,__file__,"--v6","--n",str(n),"--child",method,
           "--limit-mib",str(limit)]
    with subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                          creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0)) as proc:
        start = perf_counter()
        while True:
            try:
                out,err = proc.communicate(timeout=.1 if large else 2)
                break
            except subprocess.TimeoutExpired:
                if min(headroom_mib())<640:
                    proc.kill()
                    proc.communicate()
                    print(f"Headroom fell below reserve; retrying {method} {n}",flush=True)
                    return None
                if perf_counter()-start>1800:
                    proc.kill()
                    proc.communicate()
                    raise TimeoutError(f"{method} {n}: 30 minute time limit")
        if proc.returncode:
            raise RuntimeError(err or out)
        return json.loads(out.strip().splitlines()[-1])


def measured(n,method):
    while True:
        row = guarded_trial(n,method)
        if row is not None:
            return row
        sleep(20)


def v6():
    p = argparse.ArgumentParser()
    p.add_argument("--v6",action="store_true")
    p.add_argument("--n",type=int,nargs="+",choices=SIZES,default=SIZES)
    p.add_argument("--methods",nargs="+",choices=(*ARMS,"deferred"),default=ARMS)
    p.add_argument("--child",choices=(*ARMS,"deferred"))
    p.add_argument("--limit-mib",type=int,default=0)
    p.add_argument("--reps",type=int,default=10)
    p.add_argument("--check",action="store_true")
    p.add_argument("--report",action="store_true")
    p.add_argument("--scipy",action="store_true",help="report the matched SciPy CG sensitivity")
    args = p.parse_args()
    if args.child:
        if args.limit_mib:
            from scale_views import BIN,MIB,symbolic_library
            diagnostic = BIN/"jcert_scale_auto.dll"
            if diagnostic.exists():
                os.environ["JCERT_CHOLMOD_LIB"]=str(diagnostic)
            lib = symbolic_library()
            lib.scale_limit.argtypes = [ct.c_size_t]
            if not lib.scale_limit(args.limit_mib*MIB):
                raise OSError("private commit limit unavailable")
        print(json.dumps(trial(args.n[0],args.child)))
        return
    if args.check or args.report:
        if args.check and not args.scipy:
            from native_scale_frontier import check as check_native
            for n in SIZES:
                check_native(n)
        summarize_v6(args.report,native=not args.scipy)
        return
    if "deferred" in args.methods and args.methods!=["deferred"]:
        p.error("run the optional deferred sensitivity separately with --methods deferred")
    for n in args.n:
        path = (ROOT/"data"/f"scale_frontier_v6_{n}_deferred.csv"
                if args.methods==["deferred"] else dataset(n))
        rows = records(path)
        for rep in range(args.reps):
            schedule = (*ARMS,"deferred") if "deferred" in args.methods else ARMS
            order = np.random.default_rng(871+rep).permutation(schedule)
            for pos,method in enumerate(order):
                if method not in args.methods:
                    continue
                if n==500000 and method in ("cold","planner"):
                    continue  # frozen paired measurements, validated by check()
                if any(int(r["rep"])==rep and r["method"]==method for r in rows):
                    continue
                print(n,rep,method,flush=True)
                row = dict(rep=rep,position=pos,**measured(n,method),batch="final")
                with path.open("a",newline="") as stream:
                    writer=csv.DictWriter(stream,fieldnames=row)
                    if stream.tell()==0:
                        writer.writeheader()
                    writer.writerow(row)
                rows.append(row)
    if args.methods==ARMS and not args.scipy and args.reps==10:
        subprocess.run([sys.executable,str(ROOT/"tools"/"native_scale_frontier.py"),
                        "--n",*map(str,args.n)],check=True,
                       creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))


def summarize_v6(render=False, *, native=True):
    from scale_views import read
    stats = {}
    print("CG backend:", "native PCG" if native else "SciPy CG")
    for n in SIZES:
        rows = records(dataset(n))
        draw=np.random.default_rng(651).integers(0,10,(20000,10))
        if n==500000:
            frozen = records(ROOT/"data"/"scale_frontier_v5_500000.csv")
            check(frozen)
            rows.extend(frozen)
        if native:
            scipy_paths={tuple(json.loads(r["sequence"])) for r in rows
                         if r["method"] in ("cold","planner")}
            native_rows=records(ROOT/"data"/f"scale_frontier_v6_native_{n}.csv")
            native_paths={tuple(json.loads(r["sequence"])) for r in native_rows}
            if len(scipy_paths)!=1 or scipy_paths!=native_paths:
                raise ValueError(f"native and SciPy certified paths differ or are missing at {n}")
            rows=[r for r in rows if r["method"] not in ("cold","planner")]+native_rows
        paths = set()
        edges = int(read(n,"audit")[0]["edges"])
        cells = {}
        for method in ARMS:
            arm = sorted((r for r in rows if r["method"]==method),key=lambda r:int(r["rep"]))
            if [int(r["rep"]) for r in arm]!=list(range(10)):
                raise ValueError(f"incomplete replications: {n} {method}")
            cells[method] = arm
            for r in arm:
                seq = tuple(json.loads(r["sequence"]))
                if (len(seq)!=15 or len(set(seq))!=15 or min(seq)<0 or max(seq)>=32 or
                        int(r["edges"])!=edges or int(r["n"])!=n):
                    raise ValueError(f"graph/path mismatch: {n} {method}")
                paths.add(seq)
                if not all(np.isfinite(float(r[k])) and float(r[k])>0
                           for k in ("setup_seconds","seconds","calibration_seconds","peak_mib")):
                    raise ValueError(f"invalid measurement: {n} {method}")
                if float(r["setup_seconds"])>=float(r["seconds"]):
                    raise ValueError("setup exceeds elapsed time")
                c=json.loads(r["counters"])
                if method=="registered":
                    if (c["registered_rows_updated"],c["registered_rows_populated"],
                            c["population_rhs"],c["registered_scores"])!=(480,32,48,375):
                        raise ValueError("incomplete registered view maintenance")
                else:
                    if c["screens"]!=375:
                        raise ValueError("incorrect active screening count")
                    if sum(json.loads(r["frontiers"]))!=c["directional_scores"]:
                        raise ValueError("frontier ledger mismatch")
                    if c["state_rhs"]+c.get("state_warm_hits",0)!=15:
                        raise ValueError("missing state")
                    hits=c.get("transported",0)
                    if (hits!=c.get("gram_hits",0) or (method=="cold")!=(hits==0)
                            or (method=="cholmod" and c["factorizations"]!=1)):
                        raise ValueError("transport or factor mismatch")
                if r.get("position") and not (native and method in ("cold","planner")):
                    scheduled=np.random.default_rng(871+int(r["rep"])).permutation(ARMS)
                    if scheduled[int(r["position"])]!=method:
                        raise ValueError("unexpected arm order")
            metrics=np.array([[float(r[k]) for k in ("seconds","peak_mib","setup_seconds",
                                                       "calibration_seconds")] for r in arm])
            metrics=np.column_stack((metrics,metrics[:,0]+metrics[:,3]))
            q=np.quantile(metrics,[.25,.5,.75],axis=0)
            stats[n,method]=q
            print(n,method,"[Q1,median,Q3] time,RSS,setup,calibration,inclusive",q.round(3).tolist())
            ci = np.quantile(np.median(metrics[draw,:2],axis=1),[.025,.975],axis=0)
            print("95% bootstrap median intervals, time and RSS",ci.round(3).T.tolist())
            work_keys = (("registered_rows_populated","registered_rows_updated","registered_scores")
                         if method=="registered" else
                         ("transported","gate_requests","refine_rhs","winner_rhs","state_rhs"))
            work={tuple(json.loads(r["counters"]).get(k,0) for k in work_keys) for r in arm}
            if len(work)!=1:
                raise ValueError(f"inconsistent repeated work: {n} {method}")
            print("measured work",dict(zip(work_keys,work.pop())))
        if len(paths)!=1:
            raise ValueError(f"certified paths disagree at {n}")
        deferred=records(ROOT/"data"/f"scale_frontier_v6_{n}_deferred.csv")
        if deferred:
            if (sorted(int(r["rep"]) for r in deferred)!=list(range(len(deferred))) or
                    any(r["method"]!="deferred" or int(r["edges"])!=edges or
                        tuple(json.loads(r["sequence"])) not in paths or
                        not all(np.isfinite(float(r[k])) and float(r[k])>0 for k in
                                ("seconds","peak_mib","setup_seconds","calibration_seconds"))
                        for r in deferred)):
                raise ValueError("invalid deferred sensitivity")
            print(n,"separate deferred sensitivity",len(deferred),"replicates, median seconds/RSS",
                  np.median([[float(r[k]) for k in ("seconds","peak_mib")]
                             for r in deferred],axis=0).round(3).tolist())
        for rep in range(10):
            cold=cells["cold"][rep]
            for method in ("planner","cholmod"):
                other=cells[method][rep]
                if json.loads(cold["frontiers"])!=json.loads(other["frontiers"]):
                    raise ValueError("frontier sizes differ")
        paired=np.array([float(a["seconds"])/float(b["seconds"])
                         for a,b in zip(cells["cold"],cells["planner"])])
        ci=np.quantile(np.median(paired[draw],axis=1),[.025,.975])
        print("paired cold/planner median ratio, 95% percentile bootstrap CI:",
              np.median(paired),ci.tolist(),"path",next(iter(paths)))
    if render:
        render_v6(stats,native)


def render_v6(stats,native=True):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"]=42
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(3.48,1.94))
    labels=("CG Cold","CG Planner","CHOLMOD Planner","LINVIEW-Delta")
    for j,(ax,axis_label) in enumerate(zip(axes,("Time (s)","Peak RSS (MiB)"))):
        for method,label,marker,linestyle in zip(ARMS,labels,"os^D",("--","-","-.",":")):
            low,mid,high=np.array([stats[n,method][:,j] for n in SIZES]).T
            ax.errorbar(SIZES,mid,yerr=[mid-low,high-mid],label=label,marker=marker,
                        linestyle=linestyle,lw=.9,ms=3,capsize=2,mfc="white")
        ax.set(xscale="log",yscale="log",ylabel=axis_label)
        ax.yaxis.labelpad=2
        ax.set_xticks(SIZES,["10k","50k","100k","500k"],rotation=30)
        ax.set_xticks([],minor=True)
        ax.yaxis.label.set_size(7)
        ax.tick_params(labelsize=6)
        ax.grid(axis="y",alpha=.25)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=2,
               fontsize=6,frameon=False,columnspacing=1.2,handlelength=2.0)
    fig.subplots_adjust(left=.155,right=.975,bottom=.21,top=.74,wspace=.57)
    output="scale_views.pdf" if native else "scale_views_scipy_sensitivity.pdf"
    fig.savefig(ROOT/"figures"/output)
    plt.close(fig)


if __name__ == "__main__":
    if "--v6" in sys.argv:
        v6()
    else:
        legacy()
