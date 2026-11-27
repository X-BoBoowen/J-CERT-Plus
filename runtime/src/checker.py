"""Independent full-catalog output replay with a dyadic variational certificate.

Reuse the R15 checker dyadic conversion; rebuild two-hop sets, operators, current
responses and certificates without importing the execution engine or its index.
"""
import argparse
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from math import isqrt
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT/"baseline_r15/experimental_records/legacy_v6/tools/main_witness_verify.py"
spec = importlib.util.spec_from_file_location("_r15_integer_replay", LEGACY)
r15 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r15)
need = r15.need
S = 65536
B = 16
D = 1 << 40
COARSE = 1 << 19


def opinion_hash(s):
    return hashlib.sha256(np.ascontiguousarray(s,dtype="<i8").tobytes()).hexdigest()


def catalog(W):
    n = W.shape[0]
    neighbors = [set(map(int,W.indices[W.indptr[i]:W.indptr[i+1]])) for i in range(n)]
    cols_i,cols_j,weights,flips = [],[],[],[]
    for i,adj in enumerate(neighbors):
        neg = {int(j): int(-float(w)*B) for j,w in zip(W.indices[W.indptr[i]:W.indptr[i+1]],W.data[W.indptr[i]:W.indptr[i+1]]) if w<0}
        second = set()
        for j in adj:
            second.update(neighbors[j])
        for j in sorted(second.difference(adj).union(neg)):
            if i < j:
                cols_i.append(i)
                cols_j.append(j)
                weights.append(neg.get(j,1))
                flips.append(j in neg)
    return tuple(np.array(col,dtype=dtype) for col,dtype in
                 zip((cols_i,cols_j,weights,flips),(np.int32,np.int32,np.int64,np.bool_)))


def operator(W):
    need(W.has_canonical_format and W.shape[0]==W.shape[1] and
         (W != W.T).nnz==0 and not np.any(W.diagonal()) and
         np.all(np.isfinite(W.data)) and np.all(W.data!=0),"invalid adjacency")
    nums,den = r15.dyadic(W.data)
    need(B%den==0 and max(map(abs,nums),default=0)*B//den <= 160,"unsupported weights")
    scaled = W.copy()
    scaled.data = np.array(nums,dtype=np.int64)*(B//den)
    diag = B+np.asarray(abs(scaled).sum(axis=1)).ravel()
    return (sparse.diags(diag,dtype=np.int64)-scaled).tocsr()


def changed(A,pool,k):
    i,j,w,f = (col[k].item() for col in pool)
    val = (0,-2*w,-2*w,0) if f else (w,-w,-w,w)
    return (A+sparse.coo_matrix((val,([i,i,j,j],[i,j,i,j])),shape=A.shape,dtype=np.int64)).tocsr()


def exact_energy(A,s):
    """Small-system reference uses exact inversion of the full integer operator."""
    import sympy as sy
    v = sy.Matrix(s.tolist())
    return F((v.T*sy.Matrix(A.toarray().tolist()).inv()*v)[0])*B/(S*S)


def reference_gains(W,s,chosen=()):
    pool = catalog(W)
    A = operator(W)
    for k in chosen:
        A = changed(A,pool,k)
    base = exact_energy(A,s)
    return [base-exact_energy(changed(A,pool,k),s) if k not in chosen else None
            for k in range(len(pool[0]))]


def response(A,s):
    rhs = np.asarray(s,dtype=np.float64)*(B/S)
    diag = A.diagonal().astype(np.float64)
    z,status = cg(A,rhs,rtol=1e-13,atol=0.,M=sparse.diags(1./diag))
    need(status==0 and np.all(np.isfinite(z)) and abs(z).max() <= 1.00001,"candidate response")
    return z


def dyadic_box(A,s,z,d):
    # For x=q/d, v(x)=2s'x-x'Ax <= Phi <= v(x)+||s-Ax||^2.
    # A here is B times the physical operator. All following bounds are exact.
    q = np.rint(d*z).astype(np.int64)
    qmax = int(abs(q).max())
    norm = int(abs(A).sum(axis=1).max())
    need(norm*qmax+B*d < 2**62,"certificate matvec range")
    source = s*(B*d//S)
    r = source-A@q
    total = B*sum(int(a)*int(b) for a,b in zip(q,source+r))
    rad = sum(int(x)**2 for x in r)
    return q,r,rad,F(total,(B*d)**2),F(total+rad,(B*d)**2)


def remaining_by_cap(A,s,z,pool,alive,threshold):
    # R15 main_witness_verify.keys + endpoint principal-block correction.
    # Integer caps here use units 1/(B*d²). We round radius upward and each
    # subtracted endpoint term downward. All candidates are explicitly scanned.
    d = COARSE
    q,r,r2,_,_ = dyadic_box(A,s,z,d)
    radius = -(-r2//B)
    i,j,w,f = pool
    a,b = q[i],q[j]
    wmax = int(w.max(initial=0))
    vmax = int(abs(q).max(initial=0))
    need((2*wmax*vmax)**2 < 2**62 and 4*wmax*vmax**2+radius < 2**62,"screen range")
    key = w*np.where(f,-4*a*b,(a-b)**2)
    change = w*abs(a-b)
    # r is exact signed parent residual in units 1/(B*d).
    pa = np.maximum(0,np.where(f,2*w*abs(b),change)-abs(r[i]))
    pb = np.maximum(0,np.where(f,2*w*abs(a),change)-abs(r[j]))
    shift = np.where(f,w,2*w)
    diag = A.diagonal()
    cap = key+radius-pa**2//(diag[i]+shift)-pb**2//(diag[j]+shift)
    target = threshold*(B*d*d)
    # Preserve exact ties for lexicographic checks.
    return np.flatnonzero(alive & (cap >= -(-target.numerator//target.denominator)))


class CatalogReplay:
    def __init__(self,W):
        self.W = W
        self.A = operator(W)
        self.pool = catalog(W)

    def verify(self,s,decisions,budget,*,delta=0,expected_opinion=None,gain_denominator=None):
        s = np.asarray(s)
        delta = F(delta)
        need(s.shape==(self.A.shape[0],) and s.dtype.kind=="i" and
             np.all((s>=-S)&(s<=S)) and delta>=0 and budget>=0,"request")
        if expected_opinion is not None:
            need(opinion_hash(s)==expected_opinion,"opinion binding")
        A = self.A
        p = self.pool
        chosen = []
        alive = np.ones(len(p[0]),dtype=bool)
        checked = refined = 0
        ended = None
        for row in decisions:
            need(ended is None,"decision after terminal status")
            status = row["status"]
            k = row.get("winner_id")
            if status=="budget_stop":
                need(k is None and len(chosen)==budget,"premature budget stop")
                ended = status
                continue
            need(len(chosen)<budget,"decision beyond budget")
            if status=="unresolved":
                need(k is None,"unresolved commit")
                ended = status
                continue
            need(status in ("accept","tolerance_stop"),"unknown decision")
            if status=="accept":
                need(type(k) is int and 0<=k<len(alive) and alive[k],"winner identity")
                for key,arr in (("i",p[0]),("j",p[1]),("flip",p[3])):
                    if key in row:
                        need(row[key]==arr[k],f"{key} endpoint binding")
            else:
                need(k is None,"stop with winner")
            checked += int(alive.sum())
            if A.shape[0]<=12:
                en = exact_energy(A,s)
                gains = {int(j):en-exact_energy(changed(A,p,int(j)),s) for j in np.flatnonzero(alive)}
                if k is not None:
                    lower = upper = gains[k]
                    need(lower>0 and all(x<=lower+delta and
                         (delta>0 or j>=k or x<lower) for j,x in gains.items()),"exact winner/tie")
                    child = changed(A,p,k)
                else:
                    need(all(x<=delta for x in gains.values()),"false tolerance stop")
                refined += len(gains)
            else:
                z = response(A,s)
                _,_,_,lo,hi = dyadic_box(A,s,z,D)
                if k is not None:
                    child = changed(A,p,k)
                    _,_,_,cl,ch = dyadic_box(child,s,response(child,s),D)
                    lower,upper = lo-ch,hi-cl
                    need(lower>0,"positive winner unproved")
                    threshold = lower+delta
                else:
                    threshold = delta
                for j in remaining_by_cap(A,s,z,p,alive,threshold):
                    j = int(j)
                    if j==k:
                        continue
                    rival = changed(A,p,j)
                    _,_,_,rl,_ = dyadic_box(rival,s,response(rival,s),D)
                    rival_upper = hi-rl
                    need(rival_upper<=threshold and (k is None or delta>0 or j>k or rival_upper<threshold),"unresolved rival")
                    refined += 1
            if k is None:
                ended = "tolerance_stop"
            else:
                if "low" in row or "high" in row:
                    need(gain_denominator is not None and F(row["low"],gain_denominator)<=lower and
                         upper<=F(row["high"],gain_denominator),"published gain enclosure")
                alive[k] = False
                chosen.append(k)
                A = child
        if ended is None:
            need(len(chosen)==budget,"missing terminal decision")
            ended = "budget_stop"
        return dict(catalog_size=len(alive),accepts=len(chosen),termination=ended,
                    checked_candidates=checked,refined_rivals_or_small_states=refined,plan=chosen)


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def worker_request(rep,request,s,graph_hash,*,kind="unit"):
    """Verify output decisions and bind root, parent chain, sums and termination."""
    pool = rep.pool
    A = rep.A.copy()
    A.sort_indices()
    root = hashlib.sha256("/".join([graph_hash,digest(A.indptr,A.indices,A.data),
        digest(pool[0],pool[1],pool[3]),"lexical-twohop-negative:v1",
        "superlu-lowrank:scale16:proof48:screen20" if kind=="unit"
        else "weighted-normalized:scale16:proof48:screen20"]).encode()).hexdigest()
    need(request["root_version_id"]==root and request["method"] in ("F","T","I","D"),"scope/root identity")
    current = root
    normalized = []
    lo = hi = 0
    for epoch,raw in enumerate(request["decisions"],1):
        row = dict(raw)
        need(row["epoch"]==epoch and row["root_version_id"]==root and row["parent_version_id"]==current,"parent/epoch binding")
        if row["status"]=="accepted":
            row["status"]="accept"
            current=hashlib.sha256(f'{current}/{row["winner_id"]}'.encode()).hexdigest()
            lo+=row["low"]
            hi+=row["high"]
        need(row["current_version_id"]==current,"child binding")
        normalized.append(row)
    need(request["current_version_id"]==current and request["gain_denominator"]==2**96 and
         lo==request["gain_low"] and hi==request["gain_high"],"final identity or total gain")
    result=rep.verify(s,normalized,request["budget"],delta=request["delta"],
                      expected_opinion=request["opinion_hash"],gain_denominator=request["gain_denominator"])
    need(result["accepts"]==request["commits"] and result["termination"]==request["status"],"terminal status/count")
    return result


def replay_file(path,indexes=None):
    from .common import opinions
    trace=json.loads(Path(path).read_text())
    graph_path=ROOT/f'data/projected/{trace["graph"]}_{trace["weight_mode"]}.npz'
    graph_hash=hashlib.sha256(graph_path.read_bytes()).hexdigest()
    need(graph_hash==trace["graph_hash"] and trace["method"]!="S","external graph/scope binding")
    rep=CatalogReplay(sparse.load_npz(graph_path))
    inputs=opinions(rep.A.shape[0],trace["stream"],trace["seed"],trace["requests_planned"])
    result=[]
    indexes=range(len(trace["requests"])) if indexes is None else indexes
    for k in indexes:
        req=trace["requests"][k]
        need(req["method"]==trace["method"] and req["budget"]==trace["budget"] and req["delta"]==trace["delta"],
             "request contract binding")
        result.append(dict(request=k+1,opinion_hash=opinion_hash(inputs[k]),
                    **worker_request(rep,req,inputs[k],graph_hash,kind=trace["weight_mode"])))
    return dict(trace_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),requests=result)


if __name__=="__main__":
    import resource
    from time import perf_counter
    p=argparse.ArgumentParser()
    p.add_argument("trace",type=Path)
    p.add_argument("--request",action="append",type=int,help="1-based request IDs")
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    start=perf_counter()
    result=replay_file(args.trace,None if args.request is None else [k-1 for k in args.request])
    result.update(checker_seconds=perf_counter()-start,checker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    payload=json.dumps(result,indent=2)+"\n"
    if args.output: args.output.write_text(payload)
    else: print(payload,end="")
