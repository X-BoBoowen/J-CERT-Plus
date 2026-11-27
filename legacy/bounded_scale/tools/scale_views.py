"""Structured signed preferential-attachment communities and hierarchical bridges.

Build scale_symbolic.c as a shared library linked to the same CHOLMOD as the
numeric bridge; point JCERT_SCALE_SYMBOLIC at it when outside the local layout.
"""
import argparse
import csv
import ctypes as ct
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import psutil
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from engine import Edit
from fast_policy_kernel import FastCostPlanner

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT.parent / "chol/ucrt64/bin"
DLL = BIN / "jcert_diagnostic.dll"
if DLL.exists():
    os.environ.setdefault("JCERT_CHOLMOD_LIB", str(DLL))
METHODS = ("cg", "cholmod", "linview")
SIZES = (10000, 50000, 100000, 500000)
MIB = 2**20


def fixture(n, audit=False):
    rng = np.random.default_rng(650101 + n)
    width = 128
    base = np.arange(0, n, width, dtype=np.int32)
    groups = np.arange(len(base))
    # Each occurrence in the urn represents one end of an existing edge.
    urn = np.empty((len(base), 8*width), dtype=np.int32)
    urn[:, :2] = [0, 1]
    used = 2
    left, right = [base], [base+1]
    for v in range(2, width):
        degree = min(v, 3 + int(v % 4 == 0))
        targets = np.empty((len(base), degree), dtype=np.int32)
        for k in range(degree):
            sample = urn[groups, rng.integers(used, size=len(base))]
            invalid = np.any(sample[:,None] == targets[:,:k], axis=1)
            while invalid.any():
                idx = groups[invalid]
                sample[idx] = urn[idx, rng.integers(used, size=len(idx))]
                invalid[idx] = np.any(sample[idx,None] == targets[idx,:k], axis=1)
            targets[:, k] = sample
            left.append(base+v)
            right.append(base+sample)
        urn[:,used:used+degree] = v
        urn[:,used+degree:used+2*degree] = targets
        used += 2*degree
    del urn
    u, v = np.concatenate(left), np.concatenate(right)
    valid = (u<n) & (v<n)
    u, v = u[valid], v[valid]
    # Eight independent cross-child endpoint pairs per binary merge.
    # Each bridge joins two previously disjoint child subgraphs.
    left, right = [], []
    span = width
    while span < n:
        for start in range(0, n-span, 2*span):
            stop = min(start+2*span, n)
            pairs = set()
            while len(pairs) < 8:
                pairs.add((int(rng.integers(start,start+span)), int(rng.integers(start+span,stop))))
            for a,b in sorted(pairs):
                left.append(a)
                right.append(b)
        span *= 2
    bridges = len(left)
    shortcut_count = n//32
    global_rank = rng.permutation(n)
    cumulative = np.cumsum(np.arange(1,n+1,dtype=float)**(-.8))
    existing = {(min(a,b),max(a,b)) for a,b in zip(left,right)}
    draws = 0
    while draws < shortcut_count:
        size = shortcut_count-draws
        sources = rng.integers(n,size=size)
        targets = global_rank[np.searchsorted(cumulative,rng.random(size)*cumulative[-1])]
        for a,b in zip(sources.tolist(),targets.tolist()):
            pair = (min(a,b),max(a,b))
            if a//width == b//width or pair in existing:
                continue
            existing.add(pair)
            left.append(a)
            right.append(b)
            draws += 1
    u = np.concatenate((u, np.asarray(left,dtype=np.int32)))
    v = np.concatenate((v, np.asarray(right,dtype=np.int32)))
    neg = rng.random(len(u)) < .3
    values = np.where(neg, .125, -.125)
    adj = sparse.coo_matrix((values,(u,v)),shape=(n,n)).tocsr()
    adj = adj + adj.T
    degree = np.diff(adj.indptr)
    A = adj + sparse.diags(1+.125*degree,format="csr")
    z = np.zeros(n)
    edits = []
    used_nodes = set()
    negatives = np.flatnonzero(neg)
    for q in range(32):
        flip = q%2 == 0
        while True:
            if flip:
                pos = rng.choice(negatives)
                i,j = int(u[pos]),int(v[pos])
            else:
                i,j = map(int,rng.integers(n,size=2))
                if i == j or adj[i,j]:
                    continue
            if i not in used_nodes and j not in used_nodes:
                break
        used_nodes.update([i,j])
        edits.append(Edit(i,j,.125,flip))
        z[i] = (1-q/48)/2
        z[j] = -z[i]
    meta = dict(n=n, edges=len(u), bridges=bridges, shortcuts=shortcut_count, degree_mean=degree.mean(),
                degree_p95=np.quantile(degree,.95), degree_max=int(max(degree)))
    if audit:
        sources = rng.choice(n,16,replace=False)
        adj.data.fill(1.)
        distances = [dijkstra(adj,indices=int(i),unweighted=True).sum()/(n-1) for i in sources]
        meta["mean_distance_16"] = np.mean(distances)
    return A, edits, A@z, meta


def symbolic_library():
    path = Path(os.environ.get("JCERT_SCALE_SYMBOLIC",BIN/"scale_symbolic.dll")).resolve()
    handles = [os.add_dll_directory(str(d)) for d in [path.parent,*map(Path,os.get_exec_path())] if d.is_absolute() and d.is_dir()] if os.name=="nt" else []
    try:
        return ct.CDLL(str(path))
    finally:
        for handle in handles:
            handle.close()


def symbolic(A):
    lib = symbolic_library()
    index = np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
    vector = np.ctypeslib.ndpointer(dtype=float,flags="C_CONTIGUOUS")
    lib.scale_symbolic.argtypes = [ct.c_int,index,index,vector,vector]
    lower = sparse.tril(A,format="csc")
    lower.sort_indices()
    out = np.zeros(6)
    if not lib.scale_symbolic(A.shape[0],lower.indptr,lower.indices,lower.data,out):
        raise MemoryError("CHOLMOD symbolic analysis failed")
    return dict(symbolic_nnz=int(out[0]), symbol_flops=out[1],
                symbolic_peak_mib=out[2]/MIB, ordering=int(out[3]),
                super_xsize=int(out[4]), super_maxcsize=int(out[5]),
                fill=out[0]/lower.nnz)


def peak_mib():
    if os.name == "nt":
        return psutil.Process().memory_info().peak_wset/MIB
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(MIB if sys.platform=="darwin" else 1024)


def factor_input(A,edits):
    """Optional symbolic union over the entire input catalog, before selection."""
    if os.environ.get("JCERT_SCALE_CONFIG")!="union":
        return A
    pairs = [(e.i,e.j) for e in edits if not e.flip]
    i,j = np.array(pairs,dtype=np.int32).T
    a = A.tocoo(copy=False)
    return sparse.coo_matrix((np.r_[a.data,np.zeros(2*len(i))],
                             (np.r_[a.row,i,j],np.r_[a.col,j,i])),shape=A.shape).tocsr()


def trial(n, method):
    A, edits, s, meta = fixture(n,method in ("audit","union_audit"))
    gc.collect()
    if method in ("audit","union_audit"):
        pattern = factor_input(A,edits) if method=="union_audit" else A
        return dict(**meta, **symbolic(pattern), peak_mib=peak_mib())
    if method=="factor":
        from cholmod_backend import CholmodFactor
        with CholmodFactor(factor_input(A,edits)) as factor:
            nnz = factor.stored_entries()
            return dict(n=n,numeric_nnz=nnz,numeric_fill=nnz/(n+meta["edges"]),
                        factor_peak_bytes=factor.peak_bytes,peak_mib=peak_mib())
    start = perf_counter()
    if method=="linview":
        from linview_delta import trial as lv_trial
        row, = lv_trial(A,edits,s,budgets=(15,),backend="cholmod")
        result = {k:row[k] for k in ("seconds","setup_seconds","counters","sequence","factor_peak_bytes")}
        result["history_bytes"] = row["history_bytes"]
    else:
        if method=="cholmod" and os.environ.get("JCERT_SCALE_CONFIG")=="union":
            from cholmod_backend import CholmodFactor
            root_factor = CholmodFactor(factor_input(A,edits))
            kernel = FastCostPlanner(A,edits,backend=method,winner_slots=1,
                                     root_factor=root_factor,own_root_factor=True)
            kernel.ledger["factorizations"] += 1
        else:
            kernel = FastCostPlanner(A,edits,backend=method,winner_slots=1)
        setup = perf_counter()-start
        root_entries = kernel.factor.stored_entries() if method=="cholmod" and n==500000 else None
        path = []
        try:
            kernel.ledger["state_rhs"] += 1
            z,rho = kernel.solve(s)
            state_radii = []
            for step in range(15):
                state_radii.append(float(rho))
                e,_ = kernel.select_solved(s,z,rho,delta=1e-9)
                if e is None:
                    raise ArithmeticError("unexpected early termination")
                kernel.commit(e)
                path.append(int(e))
                if step < 14:
                    z,rho = kernel.solve_committed_state(s,z)
            result = dict(seconds=perf_counter()-start,setup_seconds=setup,
                          counters=json.dumps(dict(kernel.ledger),sort_keys=True),
                          sequence=json.dumps(path),history_bytes=0,
                          factor_peak_bytes=kernel.factor.peak_bytes if kernel.factor else 0)
            if root_entries is not None:
                result["root_entries"] = root_entries
                # Independent terminal state check after the measurement stops.
                z,rho = kernel.solve_committed_state(s,z)
                state_radii.append(float(rho))
                result["state_radii"] = json.dumps(state_radii)
        except (ArithmeticError,MemoryError) as error:
            error.partial = dict(completed_commits=len(path),root_entries=root_entries,
                                 setup_seconds=setup)
            raise
        finally:
            kernel.close()
    return dict(n=n,edges=meta["edges"],method=method,peak_mib=peak_mib(),**result)


def fname(n,method,config=None):
    if config is None:
        config = os.environ.get("JCERT_SCALE_CONFIG")
    if config and (method.startswith(("cholmod","factor")) or
                   (config!="union" and method.startswith("linview"))):
        method += "_"+config
    return ROOT/"data"/f"scale_social_v5_{n}_{method}.csv"


def read(n,method,config=None):
    p = fname(n,method,config)
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def child(n,method,limit=0):
    command = [sys.executable,__file__,"--child",method,"--n",str(n),"--limit-mib",str(limit)]
    start = perf_counter()
    with subprocess.Popen(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0)) as proc:
        while True:
            try:
                stdout,stderr = proc.communicate(timeout=.02 if limit else None)
                break
            except subprocess.TimeoutExpired:
                physical,commit = headroom_mib()
                if min(physical,commit)>=640:
                    continue
                info = psutil.Process(proc.pid).memory_info()
                proc.kill()
                proc.communicate()
                return dict(status="failed",error="live system headroom reached 640 MiB reserve",
                            peak_mib=info.peak_wset/MIB,peak_commit_mib=info.peak_pagefile/MIB,
                            attempt_seconds=perf_counter()-start,
                            partial=dict(stop_physical_mib=physical,stop_commit_mib=commit))
        if proc.returncode:
            raise RuntimeError(stderr or stdout or f"child exited {proc.returncode}")
    # The native solver can print its allocation diagnosis to stdout.
    row = json.loads(stdout.strip().splitlines()[-1])
    if row.get("status")=="failed" and "CHOLMOD native status" in stderr:
        row["error"] += " (" + stderr.strip().splitlines()[-1] + ")"
    return row


def append(n,method,row):
    with fname(n,method).open("a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=row)
        if f.tell()==0:
            w.writeheader()
        w.writerow(row)


def headroom_mib():
    physical = psutil.virtual_memory().available/MIB
    if os.name != "nt":
        return physical,physical
    class MemoryStatus(ct.Structure):
        _fields_ = [("length",ct.c_uint32),("load",ct.c_uint32)] + [
            (name,ct.c_uint64) for name in ("total_phys","avail_phys","total_page","avail_page",
                                           "total_virtual","avail_virtual","avail_extended")]
    m = MemoryStatus()
    m.length = ct.sizeof(m)
    if not ct.windll.kernel32.GlobalMemoryStatusEx(ct.byref(m)):
        raise OSError("cannot determine system commit headroom")
    return physical,m.avail_page/MIB


def available_mib():
    return min(headroom_mib())


def run(args):
    for n in args.n:
        starting_limits = {method:len(read(n,method+"_limits")) for method in args.methods}
        keys = ["audit"]
        if os.environ.get("JCERT_SCALE_CONFIG")=="union":
            keys.append("union_audit")
        for key in keys:
            if not read(n,key):
                avail = available_mib()
                if avail < 640 + .0012*n:
                    raise MemoryError(f"symbolic preflight deferred at {avail:.0f} MiB available")
                append(n,key,child(n,key))
        audit = read(n,keys[-1])
        if args.audit:
            continue
        if args.numeric_audit:
            if not read(n,"factor"):
                needed = 640 + .0011*n + 48*int(audit[0]["symbolic_nnz"])/MIB
                if available_mib() < needed:
                    raise MemoryError(f"numeric occupancy audit needs {needed:.0f} MiB headroom")
                append(n,"factor",child(n,"factor"))
            continue
        for rep in range(args.reps):
            for method in np.random.default_rng(17+rep).permutation(args.methods):
                audit = read(n,keys[-1] if method=="cholmod" else "audit")
                if any(int(r["rep"])==rep for r in read(n,method)):
                    continue
                physical,commit = headroom_mib()
                avail = min(physical,commit)
                # Allow graph construction, working vectors, factor workspaces,
                # and up to two rank-two full-width slabs per LV commitment.
                reserve = 640
                budget = max(float(audit[0]["peak_mib"]),180+(.0011*n))
                if method!="cg":
                    budget += 48*int(audit[0]["symbolic_nnz"])/MIB
                if method=="linview":
                    budget += (4*15+12)*8*n/MIB
                limit = 0
                if n == 500000 and method != "cg" and os.name == "nt":
                    earlier = read(n,method)
                    # The kernel enforces private commit; the parent separately
                    # monitors physical and global commit headroom during work.
                    limit = int(min(budget,commit-reserve,args.max_commit_mib))
                    if limit < 450:
                        raise MemoryError(f"numeric probe deferred at {avail:.0f} MiB available")
                    # Reference receipts from a previous invocation do not
                    # suppress a new measurement on the current host.
                    attempts = read(n,method+"_limits")[starting_limits[method]:]
                    if any(int(r["limit_mib"]) >= limit*.98 and
                           ("live system" not in r["error"] or float(r["physical_mib"])>=physical*.98)
                           for r in attempts):
                        continue
                    if earlier and limit < max(float(r["peak_commit_mib"]) for r in earlier):
                        raise MemoryError(f"{method} n={n}: {limit} MiB below measured peak commit")
                elif avail < reserve+budget:
                    raise MemoryError(f"{method} n={n}: {avail:.0f} MiB available, estimated {budget:.0f} + {reserve} reserve")
                row = child(n,method,limit)
                if row.get("status")=="failed":
                    status = "resource_stop" if "live system" in row["error"] else "native_failure"
                    append(n,method+"_limits",dict(n=n,method=method,limit_mib=limit,
                           headroom_mib=avail,status=status,error=row["error"],
                           peak_mib=row["peak_mib"],attempt_seconds=row["attempt_seconds"],
                           physical_mib=physical,commit_mib=commit,
                           peak_commit_mib=row.get("peak_commit_mib",""),
                           partial=json.dumps(row.get("partial",{}))))
                    print(n,method,limit,row["error"],flush=True)
                    continue
                if limit:
                    row.update(limit_mib=limit,physical_mib=physical,commit_mib=commit)
                append(n,method,dict(rep=rep,**row))
                print(n,rep,method,round(row["seconds"],3),flush=True)


def check_attempts(n,method,config):
    attempts = read(n,method+"_limits",config)
    for a in attempts:
        if (int(a["n"])!=n or a["method"]!=method or a["status"] not in ("native_failure","resource_stop") or
                not a["error"] or not 0<int(a["limit_mib"])<=float(a.get("commit_mib") or a["headroom_mib"]) or
                not all(np.isfinite(float(a[k])) and float(a[k])>0
                        for k in ("attempt_seconds","peak_mib"))):
            raise ValueError("invalid resource-limited attempt")
        if (a["status"]=="resource_stop")!=("live system" in a["error"]):
            raise ValueError("native failure and monitor stop are distinct outcomes")
        if a.get("physical_mib") and not np.isclose(float(a["headroom_mib"]),
                min(float(a["physical_mib"]),float(a["commit_mib"]))):
            raise ValueError("inconsistent prelaunch headroom")
    return attempts


def summarize(direct_config="union"):
    stats = {}
    for n in SIZES:
        audit, = read(n,"audit")
        edges = int(audit["edges"])
        if not 6 <= 2*edges/n <= 8 or int(audit["shortcuts"])!=n//32:
            raise ValueError(f"invalid graph statistics for {n}")
        if not np.isclose(float(audit["fill"]),int(audit["symbolic_nnz"])/(edges+n)):
            raise ValueError("inconsistent factor fill")
        if direct_config=="union":
            union, = read(n,"union_audit")
            if (int(union["edges"])!=edges or
                    not np.isclose(float(union["fill"]),int(union["symbolic_nnz"])/(edges+n+16))):
                raise ValueError("inconsistent catalog-union symbolic fill")
        factors=read(n,"factor",direct_config)
        if n<500000 and len(factors)!=1:
            raise ValueError("missing measured root occupancy")
        for f in factors:
            if (int(f["n"])!=n or int(f["numeric_nnz"])<edges+n or
                    not np.isclose(float(f["numeric_fill"]),int(f["numeric_nnz"])/(edges+n))):
                raise ValueError("invalid measured numeric factor")
        paths=set()
        for method in METHODS:
            config = direct_config if method=="cholmod" else ""
            rows = read(n,method,config)
            for variant in ("","simplicial","unpacked","union"):
                check_attempts(n,method,variant)
            attempts = read(n,method+"_limits",config)
            if not rows and direct_config!="union" and n==500000 and method=="cholmod" and attempts:
                stats[n,method]=np.full((3,2),np.nan)
                continue
            if sorted(int(r["rep"]) for r in rows)!=list(range(10)):
                raise ValueError(f"incomplete {n} {method}")
            for r in rows:
                if int(r["n"])!=n or int(r["edges"])!=edges or r["method"]!=method:
                    raise ValueError("graph/method key mismatch")
                if not all(np.isfinite(float(r[k])) and float(r[k])>0
                           for k in ("seconds","peak_mib","setup_seconds")):
                    raise ValueError("nonpositive/nonfinite measurement")
                path=tuple(json.loads(r["sequence"]))
                if len(set(path))!=15 or len(path)!=15 or min(path)<0 or max(path)>=32:
                    raise ValueError("invalid path")
                paths.add(path)
                ledger=json.loads(r["counters"])
                if method=="linview":
                    if int(r["history_bytes"])!=16*n*ledger["committed_global_rank"]:
                        raise ValueError("missing full-width delta history")
                elif ledger["state_rhs"]+ledger.get("state_warm_hits",0)!=15:
                    raise ValueError("state solve ledger does not cover all decisions")
                if r.get("state_radii"):
                    radii=np.array(json.loads(r["state_radii"]))
                    if radii.shape!=(16,) or not np.all(np.isfinite(radii)&(radii>=0)&(radii<1.e-8)):
                        raise ValueError("root-to-terminal residual check failed")
                    if int(r["root_entries"])<n+edges:
                        raise ValueError("invalid root factor occupancy")
            if method=="cholmod" and n==500000 and len({r["root_entries"] for r in rows})!=1:
                raise ValueError("inconsistent measured 500k numeric root")
            stats[n,method]=np.quantile([[float(r[k]) for r in rows] for k in ("seconds","peak_mib")],
                                       [.25,.5,.75],axis=1)
        if len(paths)!=1:
            raise ValueError("sequence mismatch")
    return stats


def report(stats):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"]=42
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(3.48,1.72))
    for col,(ax,lab) in enumerate(zip(axes,["Time (s)","Peak RSS (MiB)"])):
        for method,label,marker in zip(METHODS,("CG + planner","CHOLMOD + planner","LINVIEW-Delta"),"os^"):
            q=np.array([stats[n,method][:,col] for n in SIZES]).T
            ax.errorbar(SIZES,q[1],yerr=[q[1]-q[0],q[2]-q[1]],label=label,
                        fmt=marker+"-",lw=.85,ms=3,capsize=2)
        ax.set(xscale="log",yscale="log",ylabel=lab)
        ax.set_xticks(SIZES,["10k","50k","100k","500k"],rotation=30)
        ax.set_xticks([],minor=True)
        ax.yaxis.label.set_size(7)
        ax.tick_params(labelsize=6)
        ax.grid(axis="y",alpha=.3)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=3,
               fontsize=5.8,frameon=False,columnspacing=.8)
    fig.subplots_adjust(left=.12,right=.975,bottom=.22,top=.80,wspace=.57)
    fig.savefig(ROOT/"figures"/"scale_views_v5_sensitivity.pdf")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--child",choices=(*METHODS,"audit","union_audit","factor"))
    p.add_argument("--n",type=int,nargs="+",choices=SIZES,default=SIZES)
    p.add_argument("--methods",nargs="+",choices=METHODS,default=METHODS)
    p.add_argument("--reps",type=int,default=10)
    p.add_argument("--audit",action="store_true")
    p.add_argument("--numeric-audit",action="store_true")
    p.add_argument("--report",action="store_true")
    p.add_argument("--check",action="store_true")
    p.add_argument("--limit-mib",type=int,default=0,
                   help="Windows private-commit ceiling for a numeric child")
    p.add_argument("--max-commit-mib",type=int,default=2**30,
                   help="upper bound on the 500k direct-root trial's Windows private commit")
    config = p.add_mutually_exclusive_group()
    config.add_argument("--packed",action="store_true",
                   help="sensitivity configuration using the original root sparsity pattern")
    config.add_argument("--simplicial",action="store_true",
                   help="use the alternate bridge built with JCERT_FORCE_SIMPLICIAL")
    config.add_argument("--unpacked",action="store_true",
                   help="use the alternate bridge built with JCERT_UPDATE_READY")
    config.add_argument("--union-pattern",action="store_true",
                   help="include the known candidate nonedges as structural zeros in the direct root")
    args=p.parse_args()
    if args.simplicial or args.unpacked:
        config = "simplicial" if args.simplicial else "unpacked"
        os.environ["JCERT_SCALE_CONFIG"] = config
        os.environ["JCERT_CHOLMOD_LIB"] = str(Path(os.environ.get(f"JCERT_{config.upper()}_LIB",
                                                  BIN/f"jcert_{config}.dll")).resolve())
    if args.union_pattern or (not args.child and not (args.packed or args.simplicial or args.unpacked)):
        os.environ["JCERT_SCALE_CONFIG"] = "union"
    if args.child:
        if args.limit_mib:
            diagnostic = BIN/"jcert_scale_auto.dll"
            if os.environ.get("JCERT_SCALE_CONFIG") in (None,"union") and diagnostic.exists():
                os.environ["JCERT_CHOLMOD_LIB"] = str(diagnostic.resolve())
            lib = symbolic_library()
            lib.scale_limit.argtypes = [ct.c_size_t]
            if not lib.scale_limit(args.limit_mib*MIB):
                raise OSError("could not enforce child commit limit")
        start = perf_counter()
        try:
            result = trial(args.n[0],args.child)
        except (ArithmeticError,MemoryError) as error:
            if not args.limit_mib:
                raise
            result = dict(status="failed",error=str(error),peak_mib=peak_mib(),
                          attempt_seconds=perf_counter()-start,
                          partial=getattr(error,"partial",{}))
        if args.limit_mib:
            result["peak_commit_mib"]=psutil.Process().memory_info().peak_pagefile/MIB
        print(json.dumps(result))
    elif args.report or args.check:
        stats=summarize(os.environ.get("JCERT_SCALE_CONFIG",""))
        if args.report:
            report(stats)
    else:
        run(args)






