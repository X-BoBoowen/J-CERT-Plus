"""Signed-S1 mixed-edit promotion with weighted prospective additions and unit control."""
import argparse
import csv
import ctypes as ct
import gc
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from long_promotion import (BLOCK, DATA, OPINION_DEN, SCREEN, UNIT, Iterative,
                            UpdatedFactor, Views, certified_state, dyadic_costs, search,
                            seed_view, write_rows)
from promotion import complete_pool
from universe_index import build_universe
from workloads import NAMES, load_graph, opinions, system


def build_screen(directory):
    """Compile the streamlined exhaustive first pass outside the measured region."""
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        raise RuntimeError("Install a C compiler for the measured full-scan control.")
    src = Path(__file__).with_name("promotion_screen.c")
    lib = Path(directory) / ("screen.dll" if os.name=="nt" else "screen.so")
    subprocess.run([cc, "-O3", "-fPIC", "-shared", str(src), "-o", str(lib)],
                   check=True, creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    library = ct.CDLL(str(lib))
    native = library.broad_pass
    native.restype = ct.c_int32
    native.argtypes = [ct.c_int32,ct.c_int32]+[ct.c_void_p]*4+[ct.c_int64]*3+[ct.c_void_p]
    buf = np.empty(BLOCK, dtype=np.int32)

    def scan(pool,x,r,scale,target,begin,end):
        if begin==0:
            arrays=(*pool,x)
            dtypes=(np.int32,np.int32,np.bool_,np.int64)
            if any(a.dtype!=dt or a.ndim!=1 or not a.flags.c_contiguous
                   for a,dt in zip(arrays,dtypes)):
                raise TypeError("The compiled pass requires contiguous typed catalog and state arrays.")
            if len(pool[0])>=2**31 or any(len(a)!=len(pool[0]) for a in pool):
                raise ValueError("The catalog exceeds the compiled 32-bit ID range or has incompatible arrays.")
            if max(abs(int(x.min())),abs(int(x.max())))+r > 2*SCREEN or scale>16:
                raise OverflowError("screen input out of range")
        count = native(begin,end,*(a.ctypes.data for a in pool), x.ctypes.data,
                       r,scale,target,buf.ctypes.data)
        return buf[:count].copy()
    scan.library = library
    return scan


def pathfile(name,control,suffix):
    tag = "unit_" if control else ""
    return DATA / f"signed_promotion_{tag}{name}_{suffix}.csv"


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def catalog(W,name):
    if not np.all(np.abs(W.data)==1):
        raise ValueError("The experiment requires unit signed input edges.")
    return complete_pool(W) if name in ("Conflict","Bio") else build_universe(W)


def run(name,reps,control,broad,*,index_factory=None,output_file=pathfile,generic_control=False):
    scale = 1 if control else 16
    methods = ["promoted"] if generic_control else ["static","promoted","full"]
    for rep in range(reps):
        existing = output_file(name,control,"runs")
        done = {r["method"] for r in read_rows(existing) if r["graph"]==name and int(r["repetition"])==rep} if existing.exists() else set()
        if done == set(methods):
            continue
        tick = perf_counter()
        W = load_graph(name)
        s = np.rint(OPINION_DEN*opinions(W,"S1",seed=17)).astype("int64")
        graph_seconds = perf_counter()-tick
        tick = perf_counter()
        pool = catalog(W,name)
        catalog_seconds = perf_counter()-tick
        for mode in methods[rep%3:]+methods[:rep%3]:
            if mode in done:
                continue
            gc.collect()
            process = psutil.Process()
            peak = [process.memory_info().rss]
            stop = Event()
            def sample():
                while not stop.wait(.015):
                    peak[0] = max(peak[0],process.memory_info().rss)
            monitor=Thread(target=sample)
            monitor.start()
            try:
                rows,lo,hi,elapsed = search(W,s,pool,mode,adaptive=mode=="promoted",
                    operator_scale=scale,dyadic_tau=True,trace=True,broad_pass=broad,
                    index_factory=index_factory if mode=="promoted" else None)
            finally:
                stop.set()
                monitor.join()
            assert len(rows)==15
            key=dict(graph=name,n=W.shape[0],universe=len(pool[0]),repetition=rep,method=mode)
            first=rows[0]
            online=elapsed-first["common_seconds"]-first["setup_seconds"]
            record=dict(**key,graph_seconds=graph_seconds,catalog_seconds=catalog_seconds,
                common_seconds=first["common_seconds"],rank_seconds=first["rank_seconds"],
                index_seconds=first["index_seconds"],online_seconds=online,
                total_seconds=graph_seconds+catalog_seconds+elapsed,peak_rss=peak[0],
                gain_low=lo,gain_high=hi,denominator=UNIT**2)
            for row in rows:
                row.setdefault("tau_units","")
                row.setdefault("tau_denominator","")
                row.setdefault("tau_grid_max","")
                row.setdefault("tau_work","")
            if generic_control:
                label="generic dyadic root-cap index"
                record["algorithm"]=label
                for row in rows:
                    row["algorithm"]=label
            write_rows(output_file(name,control,"rounds"),[dict(**key,**row) for row in rows])
            write_rows(existing,[record])
            print(name,"unit" if control else "repair",rep,mode,
                  sum(r['flip'] for r in rows),round(elapsed,2),flush=True)
        del W,pool
        gc.collect()


def sign_audit(W,s,pool,rows,scale,proof_unit=UNIT):
    """Classify every admitted outsider at the root and at first refinement."""
    A = scale*system(W).astype("int64")
    fac = UpdatedFactor(A) if W.shape[0]<=10000 else Iterative(A)
    _,x,r,xc,rc=certified_state(A,fac,s,proof_unit)
    root_hot,_=seed_view(pool,xc,rc,A.diagonal(),64,False,scale)
    root_view=Views(pool,proof_unit,scale)
    roots={}
    output=[]
    revived=set()
    for row in rows:
        admissions=json.loads(row["admissions"])
        for candidate,gain in admissions.items():
            k=int(candidate)
            assert k not in root_hot and k not in roots
            gap=abs(int(x[pool[0][k]])-int(x[pool[1][k]]))-2*r
            if not pool[2][k] and gap>0:
                # A positive update has strictly positive gain iff b^T z != 0.
                lo,hi=1,None  # qualitative certificate: strictly positive
            else:
                lo,hi=root_view.score(k,0,A,fac,x,r)
                lo,hi=lo*scale,hi*scale
            if lo<=0 and (hi is None or hi>0):
                raise ArithmeticError(f"unresolved root gain sign for {k}: {lo}, {hi}")
            roots[k]=(lo,hi)
            if hi is not None and hi<=0:
                if gain[0]>0:
                    revived.add(k)
                elif gain[1]>0:
                    raise ArithmeticError(f"unresolved admission gain sign for {k}")
            output.append(dict(epoch=int(row["epoch"]),candidate=k,flip=int(pool[2][k]),
                root_strictly_positive=int(lo>0),
                root_test="endpoint separation" if hi is None else "Gram interval",
                root_gain_low=lo if hi is not None else "",root_high=hi if hi is not None else "",
                root_abs_endpoint_difference_low_units=max(0,gap),state_denominator=proof_unit,
                admission_low=gain[0],admission_high=gain[1],revived_at_admission=int(k in revived)))
    winners=[int(row["winner_id"]) for row in rows]
    revived_winners=sum(k in roots and roots[k][1] is not None and roots[k][1]<=0 for k in winners)
    return output,len(revived),revived_winners,sum(k not in root_hot for k in winners)


def summarize(names,control,reps):
    out=[]
    for name in names:
        rr=read_rows(pathfile(name,control,"rounds"))
        runs=read_rows(pathfile(name,control,"runs"))
        assert len(runs)==reps*3 and len(rr)==reps*45
        paths={mode:[r for r in rr if r['method']==mode and r['repetition']=='0']
               for mode in ("full","promoted","static")}
        def ids(p):
            return [int(r['winner_id']) for r in p]
        for rep in range(reps):
            for method,p in paths.items():
                actual=[r for r in rr if r['method']==method and int(r['repetition'])==rep]
                assert ids(actual)==ids(p)
        assert ids(paths['promoted'])==ids(paths['full'])
        for p,f in zip(paths['promoted'],paths['full']):
            assert max(int(p['low']),int(f['low']))<=min(int(p['high']),int(f['high']))
        W=load_graph(name)
        s=np.rint(OPINION_DEN*opinions(W,"S1",17)).astype('int64')
        pool=catalog(W,name)
        audit,revived,revwins,outsiders=sign_audit(W,s,pool,paths['promoted'],1 if control else 16)
        audit_file=pathfile(name,control,'admissions')
        audit_file.unlink(missing_ok=True)
        if audit: write_rows(audit_file,audit)
        opposite=lambda r: (s[int(r["i"])]>=0)!=(s[int(r["j"])]>=0)
        taus=[int(r['tau_units']) for r in paths['promoted']]
        assert all(v>=1 and v & (v-1)==0 for v in taus)
        counts=dict(tau_min_units=min(taus),tau_max_units=max(taus),flip_winners=sum(int(r['flip']) for r in paths['full']),
            opposite_flip_winners=sum(int(r['flip']) and opposite(r) for r in paths['full']),
            opposite_add_winners=sum(not int(r['flip']) and opposite(r) for r in paths['full']),
            admitted=len(audit),
            revived_admitted=revived,revived_winners=revwins,outsider_winners=outsiders,
            mismatches=sum(a!=b for a,b in zip(ids(paths['static']),ids(paths['full']))))
        totals={m:[sum(int(r[k]) for r in p) for k in ('low','high')] for m,p in paths.items()}
        flo,fhi=totals['full']; slo,shi=totals['static']
        loss_lo,loss_hi=(100*(flo-shi)/flo,100*(fhi-slo)/fhi) if counts['mismatches'] else (0.,0.)
        for mode in paths:
            m=[r for r in runs if r['method']==mode]
            p=paths[mode]
            metrics={}
            for field in ('graph_seconds','catalog_seconds','common_seconds','rank_seconds',
                          'index_seconds','online_seconds','total_seconds'):
                data=[float(r[field]) for r in m]
                metrics[field]=np.median(data)
                metrics[field+'_q1'],metrics[field+'_q3']=np.quantile(data,[.25,.75])
            full={r['repetition']:r for r in runs if r['method']=='full'}
            rng=np.random.default_rng(108)
            for field in ('online_seconds','total_seconds'):
                ratios=np.asarray([float(full[r['repetition']][field])/float(r[field]) for r in m])
                key='paired_full_over_method_'+field
                metrics[key]=np.median(ratios)
                metrics[key+'_ci_low'],metrics[key+'_ci_high']=np.quantile(
                    np.median(rng.choice(ratios,size=(10000,len(ratios))),axis=1),[.025,.975])
            out.append(dict(graph=name,universe=len(pool[0]),method=mode,reps=reps,**counts,
                loss_pct_lo=loss_lo,loss_pct_hi=loss_hi,visited=sum(int(r['visited']) for r in p),
                peak_hot=max(64,max(int(r['view'])+1 for r in p)),
                peak_rss=max(int(r['peak_rss']) for r in m),**metrics))
    file=pathfile('summary',control,'methods')
    file.unlink(missing_ok=True)
    write_rows(file,out)


def write_table():
    data={(r["graph"],r["method"]):r for r in read_rows(pathfile("summary",False,"methods"))}
    control={r["graph"]:int(r["flip_winners"]) for r in read_rows(pathfile("summary",True,"methods"))
             if r["method"]=="full"}
    assert len(data)==18 and all(int(r["reps"])==10 for r in data.values())
    methods=("static","promoted","full")
    def fmt(v):
        return f"{v:.2f}" if v<10 else f"{v:.1f}"
    table=[r"""\begin{table*}[t]
\centering\small\setlength{\tabcolsep}{3pt}
\caption{Full graphs, 15 commits from 64 seeds. T contains negative edges and two-hop nonedges, A all negative edges and nonedges. Flips count winners at insertion weights $1/16,1/4,1$. Other columns use $1/16$. Miss/loss gives static action mismatches and cumulative gain loss (\%). A/W/R counts admissions, winners outside the seed set, and admissions with nonpositive root gain and positive current gain. Visits are distinct outsider inspections as a percentage of live candidate-epochs. Build is adaptive seed ranking plus cold-index construction. Online times are seconds for static (S), adaptive (D), and full (F). RSS is peak sampled adaptive working set (MiB).}
\label{tab:promotion}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrrr@{}}
\toprule
Graph and catalog size & Flips & Miss/loss & A/W/R & Visits (\%) & Build (s) & S (s) & D (s) & F (s) & RSS\\
\midrule"""]
    mid_total=0
    for name in NAMES:
        r=data[name,"promoted"]
        mid=read_rows(DATA/f"signed_promotion_midpoint_{name}_rounds.csv")
        assert len(mid)==15 and all(o["insertion_denominator"]=="4" and 0<int(o["low"])<=int(o["high"]) for o in mid)
        middle=sum(int(o["flip"]) for o in mid)
        mid_total+=middle
        n=int(r["universe"])
        lo,hi=float(r["loss_pct_lo"]),float(r["loss_pct_hi"])
        places=3
        while places and round(lo,places)!=round(hi,places): places-=1
        loss=f"{lo:.{places}f}" if hi else "0"
        runs=[o for o in read_rows(pathfile(name,False,"runs")) if o["method"]=="promoted"]
        build=np.median([float(o["rank_seconds"])+float(o["index_seconds"]) for o in runs])
        domain="A" if name in ("Conflict","Bio") else "T"
        fields=[f"{name} ({domain} {n:,})",f"{r['flip_winners']}/{middle}/{control[name]}",
                f"{r['mismatches']}/{loss}",f"{r['admitted']}/{r['outsider_winners']}/{r['revived_admitted']}",
                f"{100*int(r['visited'])/(15*n-105):.3f}",fmt(build)]
        fields.extend(fmt(float(data[name,m]["online_seconds"])) for m in methods)
        fields.append(f"{int(r['peak_rss'])/2**20:,.0f}")
        table.append(" & ".join(fields)+r"\\")
    table.append(r"""\bottomrule
\end{tabular*}
\end{table*}
\subsubsection*{Long-horizon cold promotion}
Signed high-pass S1 (seed 17, $2^{-16}$ input grid) fixes each query. A repair changes $-1$ to $+1$. Static searches the initial view. Adaptive and full share the current two-diagonal insertion screen, signed flip screen and lexical ties. Integer residuals on $H=16(I+L)$ and rational Gram intervals certify survivors using $\lVert H^{-1}\rVert_\infty\le1/16$. Adaptive minimizes posting lengths plus two conservative class-prefix lengths over $\tau_0 2^k$ ($\tau_0=1/4096$, minimum $2^{-20}$), retaining cutoff bins and deduplicating their union.""")
    sd=data["Slashdot","promoted"]
    raw=[o for o in read_rows(pathfile("Slashdot",False,"rounds")) if o["method"]=="promoted" and o["repetition"]=="0"]
    total="/".join(fmt(float(data["Slashdot",m]["total_seconds"])) for m in methods)
    ratio="paired_full_over_method_"
    def interval(metric):
        return (f"{float(sd[ratio+metric]):.2f} "
                f"[{float(sd[ratio+metric+'_ci_low']):.2f},{float(sd[ratio+metric+'_ci_high']):.2f}]")
    flips=sum(int(data[g,"full"]["flip_winners"]) for g in NAMES)
    admitted=sum(int(data[g,"promoted"]["admitted"]) for g in NAMES)
    revived=sum(int(data[g,"promoted"]["revived_winners"]) for g in NAMES)
    table.append(f"All 90 adaptive choices match full search. The three weights yield {flips}, {mid_total} and {sum(control.values())} repairs. "
        f"Among {admitted} admissions, {revived} nonpositive-root outsiders won. Slashdot inspects {int(sd['visited']):,} of {15*int(sd['universe'])-105:,} live candidate-epochs. "
        f"Its catalog and index occupy {int(raw[0]['catalog_bytes'])/2**20:.1f} and {int(raw[0]['index_bytes'])/2**20:.1f} MiB, with {max(int(o['view_bytes']) for o in raw)/1024:.1f} KiB of hot IDs and Grams. "
        f"Ten rotated repetitions give median S/D/F totals {total} s. Slashdot catalog construction takes {fmt(float(sd['catalog_seconds']))} s. "
        f"Adaptive root solve, seed ranking and cold indexing take {fmt(float(sd['common_seconds']))}, {fmt(float(sd['rank_seconds']))} and {fmt(float(sd['index_seconds']))} s. "
        f"The paired full/adaptive ratio is {interval('online_seconds')} online and {interval('total_seconds')} including setup (95\\% bootstrap intervals, 10,000 paired resamples). "
        "Per-method phase times, RSS, grid choices and gain bounds are in the artifact.\n")
    target=DATA.parent/"sections"/"promotion_results.tex"
    target.parent.mkdir(exist_ok=True)
    target.write_text("\n".join(table))


def replace_rows(path,rows):
    """Install a complete CSV beside its predecessor, including on Windows."""
    with tempfile.NamedTemporaryFile(mode="w",newline="",dir=path.parent,delete=False) as out:
        tmp=Path(out.name)
        writer=csv.DictWriter(out,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def finalize_index(names,reps):
    """Validate new matched triples before binding their canonical public names."""
    staged={}
    for name in names:
        tables={}
        for suffix in ("runs","rounds"):
            src=DATA/f"indexed_{name}_{suffix}.csv"
            dst=pathfile(name,False,suffix)
            if src.exists():
                rows=read_rows(src)
            else:
                rows=read_rows(dst)
                if not all(row.get("index_policy")=="class-aware dyadic" for row in rows):
                    raise FileNotFoundError(f"Missing new class-aware measurements: {src}")
            tables[suffix]=rows
        runs,rounds=tables["runs"],tables["rounds"]
        assert len(runs)==3*reps and len(rounds)==45*reps
        reference={}
        for rep in range(reps):
            paths={}
            for mode in ("static","promoted","full"):
                key=lambda r:(int(r["repetition"]),r["method"])==(rep,mode)
                run,=[r for r in runs if key(r)]
                trajectory=sorted((r for r in rounds if key(r)),key=lambda r:int(r["epoch"]))
                assert [int(r["epoch"]) for r in trajectory]==list(range(1,16))
                ids=[int(r["winner_id"]) for r in trajectory]
                assert len(set(ids))==15 and all(0<=k<int(run["universe"]) for k in ids)
                assert ids==reference.setdefault(mode,ids)
                assert all(0<int(r["low"])<=int(r["high"]) for r in trajectory)
                for c in ("low","high"):
                    assert sum(int(r[c]) for r in trajectory)==int(run["gain_"+c])
                if mode=="promoted":
                    for row in trajectory:
                        tau=int(row["tau_units"])
                        assert tau>0 and tau&(tau-1)==0 and tau<=int(row["tau_grid_max"])
                        assert int(row["tau_denominator"])==SCREEN
                        assert int(row["tau_work"])>=int(row["visited"])
                paths[mode]=trajectory
            for a,b in zip(paths["promoted"],paths["full"]):
                assert a["winner_id"]==b["winner_id"] and a["flip"]==b["flip"]
                assert max(int(a["low"]),int(b["low"]))<=min(int(a["high"]),int(b["high"]))
        staged[name]=tables
    # Keep one explicitly identified generic-schedule trajectory per graph if
    # one was actually measured locally. It is optional in a clean rerun.
    for suffix in ("runs","rounds"):
        control_path=DATA/f"signed_generic_control_{suffix}.csv"
        controls=read_rows(control_path) if control_path.exists() else []
        present={row["graph"] for row in controls}
        added=False
        for name in names:
            old=pathfile(name,False,suffix)
            if name not in present and old.exists():
                saved=[dict(**r,algorithm="generic dyadic root-cap index")
                       for r in read_rows(old) if not r.get("index_policy")
                       and r["method"]=="promoted" and r["repetition"]=="0"]
                controls.extend(saved)
                added=added or bool(saved)
        if added:
            replace_rows(control_path,controls)
    for name,tables in staged.items():
        for suffix,rows in tables.items():
            for row in rows:
                row["index_policy"]="class-aware dyadic"
            replace_rows(pathfile(name,False,suffix),rows)
    for name in names:
        for suffix in ("runs","rounds"):
            (DATA/f"indexed_{name}_{suffix}.csv").unlink(missing_ok=True)


def midpoint(name,broad):
    """One certified full path at the predeclared middle insertion price."""
    W=load_graph(name)
    intrinsic=opinions(W,"S1",seed=17)
    if not np.isfinite(intrinsic).all() or np.abs(intrinsic).max()>1:
        raise ValueError("Signed S1 input violated its bounded opinion contract")
    s=np.rint(OPINION_DEN*intrinsic).astype(np.int64)
    pool=catalog(W,name)
    rows,_,_,_=search(W,s,pool,"full",budget=15,hot_size=64,operator_scale=4,
                       trace=True,broad_pass=broad)
    assert len(rows)==15 and all(0<r["low"]<=r["high"] for r in rows)
    result=[dict(graph=name,universe=len(pool[0]),method="full",opinion="signed S1",
                 seed=17,insertion_denominator=4,denominator=UNIT**2,
                 **{k:r[k] for k in ("epoch","winner_id","i","j","flip","low","high")})
            for r in rows]
    path=DATA/f"signed_promotion_midpoint_{name}_rounds.csv"
    path.unlink(missing_ok=True)
    write_rows(path,result)
    print(name,"midpoint",sum(r["flip"] for r in rows),flush=True)


def revival_case(broad,index_factory):
    """Replay the existing anchored Alpha motif over its full mixed catalog."""
    W=load_graph("Alpha")
    A=system(W).astype("int64")
    negative=(W<0).tocsr()
    degree=A.diagonal()
    a=np.flatnonzero((np.diff(negative.indptr)==2)&(degree<7))[0]
    b,c=sorted(negative[a].indices,key=lambda v:(degree[v],v))
    z=np.random.default_rng(17).integers(-3,4,size=A.shape[0])
    z[[a,b,c]]=[round(1.5*512/degree[a]),-512,96]
    rhs=A@z
    s=np.rint(rhs*min(1,OPINION_DEN/max(abs(rhs)))).astype("int64")
    pool=catalog(W,"Alpha")
    proof_unit=1<<48
    paths={}
    for method in ("static","promoted","full"):
        paths[method]=search(W,s,pool,method,budget=15,adaptive=method=="promoted",
            operator_scale=16,dyadic_tau=True,trace=True,broad_pass=broad,proof_unit=proof_unit,
            index_factory=index_factory if method=="promoted" else None)[0]
    main=paths["promoted"]
    for method in ["full","promoted"]:
        assert [(r["i"],r["j"]) for r in paths[method][:2]]==[tuple(sorted((a,b))),tuple(sorted((a,c)))]
    assert [r['winner_id'] for r in main]==[r['winner_id'] for r in paths['full']]
    audit,revived,winners,_=sign_audit(W,s,pool,main,16,proof_unit)
    target=next(r for r in audit if r["candidate"]==main[1]["winner_id"])
    assert target["root_high"]<0<target["admission_low"] and revived>=1 and winners>=1
    # Direct edited-system energy enclosures are independent of the
    # rank-two Gram calculation and of the set of competing insertions.
    from flip_promotion_case import after,objective
    energies=[]
    for parent in (A,after(A,a,b)):
        parent_lo,parent_hi=objective(parent,s)
        child_lo,child_hi=objective(after(parent,a,c),s)
        energies.append([parent_lo-child_hi,parent_hi-child_lo])
    assert energies[0][1]<0<energies[1][0]
    result={"graph":"Alpha","catalog_size":len(pool[0]),"denominator":proof_unit**2,
            "index_policy":"class-aware dyadic","direct_gain_intervals":energies,
            "direct_gain_denominator":UNIT**2,
            "rule":"established Alpha motif; additions 1/16; flips -1 to +1",
            "revival":target,"revived_admitted":revived,"revived_winners":winners,"paths":paths}
    (DATA/"signed_promotion_motif.json").write_text(json.dumps(result,indent=2)+"\n")


def selftest(broad):
    from fractions import Fraction as F
    W=load_graph("Alpha",n=12)
    pool=complete_pool(W)
    rng=np.random.default_rng(77)
    s=rng.integers(-OPINION_DEN,OPINION_DEN,len(W.indptr)-1)
    for scale in (1,4,16):
        A=system(W).astype("int64")*scale
        q=[F(int(i),OPINION_DEN) for i in s]
        from sympy import Matrix, Rational
        rhs=Matrix([Rational(v.numerator,v.denominator) for v in q])
        current=Matrix(A.toarray())
        chosen=set()
        traces={}
        for mode in ("full","promoted"):
            rows,lo,hi,_=search(W,s,pool,mode,3,3,mode=="promoted",operator_scale=scale,
                                trace=True,dyadic_tau=True,broad_pass=broad)
            traces[mode]=rows
        assert [r["winner_id"] for r in traces["full"]]==[r["winner_id"] for r in traces["promoted"]]
        for row in traces["full"]:
            root=(rhs.T*current.inv()*rhs)[0]
            gains={}
            for k,(i,j,flip) in enumerate(zip(*pool)):
                if k in chosen: continue
                child=current.copy()
                i,j=int(i),int(j)
                if flip:
                    child[i,j]-=2*scale;child[j,i]-=2*scale
                else:
                    child[i,i]+=1;child[j,j]+=1;child[i,j]-=1;child[j,i]-=1
                gains[k]=(scale*(root-(rhs.T*child.inv()*rhs)[0]),child)
            winner=max(gains,key=lambda k:(gains[k][0],-k))
            gain,current=gains[winner]
            assert row["winner_id"]==winner and row["low"]<=gain*UNIT**2<=row["high"]
            chosen.add(winner)

    degree=np.asarray([4,2,3,1])
    drift=np.asarray([0,1,4,7])
    root_caps=np.asarray([32,24,8,-16])*SCREEN
    chunks=[(None,-root_caps[:2]),(None,-root_caps[2:])]
    for target in (0,8*SCREEN,56*SCREEN,100*SCREEN):
        choices,costs=dyadic_costs(degree,drift,chunks,target)
        assert choices.tolist()==[1,2,4,8]
        brute=[sum(int(d) for d,x in zip(degree,drift) if x>t)
               +sum(int(x)+8*SCREEN*t>=target for x in root_caps) for t in choices]
        assert costs.tolist()==brute and int(choices[np.argmin(costs)])==min(
            choices,key=lambda t:brute[choices.tolist().index(t)])

    # A disjoint tie cannot be settled by one endpoint transposition.
    from scipy.sparse import csr_matrix
    W=csr_matrix((4,4),dtype=np.int8)
    tied=complete_pool(W)
    s=np.asarray([OPINION_DEN,-OPINION_DEN,OPINION_DEN,-OPINION_DEN])
    rows=search(W,s,tied,"promoted",1,1,True,operator_scale=16,trace=True,dyadic_tau=True)[0]
    assert rows[0]["winner_id"]==0
    W=csr_matrix([[0,-1],[-1,0]])
    s=np.asarray([OPINION_DEN,OPINION_DEN])
    assert search(W,s,complete_pool(W),"full",1,1,operator_scale=16)[0]==[]


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--graph',nargs='+',choices=NAMES,default=NAMES)
    p.add_argument('--reps',type=int,default=10)
    p.add_argument('--unit-control',action='store_true')
    p.add_argument('--summary',action='store_true')
    p.add_argument('--table',action='store_true')
    p.add_argument('--finalize-index',action='store_true')
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--motif',action='store_true')
    p.add_argument('--midpoint',action='store_true')
    p.add_argument('--generic-control',action='store_true')
    args=p.parse_args()
    with threadpool_limits(limits=1):
        if args.finalize_index:
            finalize_index(args.graph,args.reps)
        elif args.table:
            write_table()
        elif args.summary:
            summarize(args.graph,args.unit_control,args.reps)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                broad=build_screen(tmp)
                lib=None
                try:
                    if args.midpoint:
                        for name in args.graph: midpoint(name,broad)
                    elif args.selftest: selftest(broad)
                    elif args.generic_control:
                        def generic_file(name,control,suffix):
                            return DATA/f"signed_generic_control_{suffix}.csv"
                        for name in args.graph:
                            run(name,1,False,broad,output_file=generic_file,generic_control=True)
                    elif args.unit_control:
                        for name in args.graph: run(name,args.reps,True,broad)
                    else:
                        from promotion_index_profile import ClassIndex,compile_index,indexed_file
                        lib=compile_index(tmp)
                        factory=lambda *params:ClassIndex(lib,*params)
                        if args.motif:
                            revival_case(broad,factory)
                        else:
                            for name in args.graph:
                                run(name,args.reps,False,broad,index_factory=factory,output_file=indexed_file)
                            finalize_index(args.graph,args.reps)
                finally:
                    if os.name=="nt":
                        from _ctypes import FreeLibrary
                        if lib is not None: FreeLibrary(lib._handle)
                        FreeLibrary(broad.library._handle)
