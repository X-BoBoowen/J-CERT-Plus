"""Prove natural-run S decisions on the independently reconstructed root shortlist."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
from scipy import sparse
from .checker import CatalogReplay, ROOT, digest, need, opinion_hash
from .common import opinions

UNIT=1<<48
SCREEN=1<<20
RATIO=UNIT//SCREEN


def root_seed(rep,s,request,k):
    # Scope registration requires the audited numerical root snapshot. Refuse to
    # infer membership from a bare list supplied by the timed service.
    snapshots=request.get("snapshots",[])
    need(snapshots and snapshots[0]["epoch"]==1,"missing audited root grid")
    shot=snapshots[0]
    need(shot["opinion_hash"]==opinion_hash(s),"root snapshot opinion")
    need(all(type(v) is int for v in shot["x"]),"dyadic response codes")
    x=np.asarray(shot["x"],dtype=np.int64)
    r=shot["r"]
    A=rep.A
    need(x.shape==s.shape and type(r) is int and r>=0,"root state shape")
    max_x=max(abs(int(x.min())),abs(int(x.max())))
    need(2*int(A.diagonal().max())*max_x+UNIT<2**63 and max_x<2**50,"root residual range")
    residual=A@x-s*(UNIT//65536)
    need(r==int(abs(residual).max()),"root residual mismatch")
    # The executor rounded a binary64 proposal at both UNIT and SCREEN.
    # At a halfway position of x on the finer grid, either adjacent coarse
    # value may have been obtained. Enclose the two possibilities below.
    half=x%RATIO==RATIO//2
    xc=np.rint(x.astype(np.float64)/RATIO).astype(np.int64)
    rc=(int(abs(x-xc*RATIO).max())+r+RATIO-1)//RATIO
    i,j,d,flip=rep.pool
    need(4*(int(abs(xc).max())+rc+1)**2*int(d.max(initial=1))<2**62,"key range")
    def keys(a,b):
        return d*np.where(flip,-4*a*b+4*rc*(abs(a)+abs(b)+rc),(abs(a-b)+2*rc)**2)
    key=keys(xc[i],xc[j])
    seed=sorted(map(int,np.argsort(-key,kind="stable")[:k]))
    if not half.any() or not seed or len(seed)==len(key):
        return seed
    # rc depends on |x-xc*RATIO|, which is identical for both choices at
    # halfway points. A robust separation proves the selected set under all
    # actual coarse-grid tie assignments. An inconclusive separation refuses.
    left=np.where(half,x//RATIO,xc)
    right=left+half.astype(np.int64)
    lo=key.copy()
    hi=key.copy()
    for aa in (left,right):
        for bb in (left,right):
            possible=keys(aa[i],bb[j])
            lo=np.minimum(lo,possible)
            hi=np.maximum(hi,possible)
    included=np.zeros(len(key),dtype=bool)
    included[seed]=True
    worst=min((int(lo[e]),-e) for e in seed)
    best_value=int(hi[~included].max())
    best_id=int(np.flatnonzero((hi==best_value)&~included)[0])
    need(worst>(best_value,-best_id),"ambiguous registered shortlist boundary")
    return seed


def verify_request(rep,s,req,graph_hash,mode,hot):
    need(req["method"]=="S" and req["opinion_hash"]==opinion_hash(s),"scope or opinion mismatch")
    seed=root_seed(rep,s,req,hot)
    need(seed==req["initial_seed_ids"],"incorrect registered shortlist")
    full=rep.pool
    A=rep.A.copy()
    A.sort_indices()
    salt="weighted-normalized:scale16:proof48:screen20" if mode=="weighted" else "superlu-lowrank:scale16:proof48:screen20"
    root=hashlib.sha256("/".join([graph_hash,digest(A.indptr,A.indices,A.data),
                         digest(full[0],full[1],full[3]),"lexical-twohop-negative:v1",salt]).encode()).hexdigest()
    need(req["root_version_id"]==root,"root binding")
    scope=copy.copy(rep)
    scope.pool=tuple(arr[seed] for arr in full)
    remap={e:k for k,e in enumerate(seed)}
    current=root
    decisions=[]
    low=high=0
    for t,d in enumerate(req["decisions"],1):
        row=dict(d)
        e=row["winner_id"]
        need(row["epoch"]==t and row["root_version_id"]==root and row["parent_version_id"]==current,"S parent/epoch")
        if e is not None:
            need(row["status"]=="accepted" and e in remap and row["proof_scope"]=="shortlist","unregistered S winner")
            row["status"]="accept"
            row["winner_id"]=remap[e]
            current=hashlib.sha256(f"{current}/{e}".encode()).hexdigest()
            low+=row["low"]
            high+=row["high"]
        need(row["current_version_id"]==current,"S child hash")
        decisions.append(row)
    need(current==req["current_version_id"] and (low,high)==(req["gain_low"],req["gain_high"]),"S terminal totals")
    need(req["gain_denominator"]==UNIT**2,"S gain units")
    checked=scope.verify(s,decisions,req["budget"],delta=req["delta"],gain_denominator=UNIT**2)
    need(checked["termination"]==req["status"] and checked["accepts"]==req["commits"],"S terminal status/count")
    return dict(scope="registered_root_shortlist_only",catalog_size=len(full[0]),shortlist_size=len(seed),
                registered_ids=seed,plan=[seed[j] for j in checked["plan"]],candidate_visits=checked["checked_candidates"],
                accepts=checked["accepts"],termination=checked["termination"],opinion_hash=opinion_hash(s))


def run(path):
    start=perf_counter()
    t=json.loads(path.read_text())
    need(t["method"]=="S" and t["audit"] and t["status"]=="completed","expected complete S audit")
    mode=t["weight_mode"]
    graph=ROOT/f'data/projected/{t["graph"]}_{mode}.npz'
    h=hashlib.sha256(graph.read_bytes()).hexdigest()
    need(h==t["graph_hash"],"projection checksum")
    rep=CatalogReplay(sparse.load_npz(graph))
    inputs=opinions(rep.A.shape[0],t["stream"],t["seed"],t["requests_planned"])
    need(len(inputs)==len(t["requests"]),"missing S requests")
    checked=[]
    for s,req in zip(inputs,t["requests"]):
        need(req["budget"]==t["budget"] and req["delta"]==t["delta"],"request binding")
        checked.append(verify_request(rep,s,req,h,mode,t["hot_size"]))
    return dict(trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),requests=checked,
                checker_seconds=perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("trace",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.write_text(json.dumps(run(a.trace),indent=2)+"\n")
