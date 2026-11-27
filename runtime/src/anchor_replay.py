"""Separate full-output and per-index-query proof replay for real A1 audit traces."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
from scipy import sparse

from .checker import CatalogReplay, ROOT, changed, digest, need, opinion_hash, worker_request
from .common import opinions
from .shortlist_checker import root_seed

UNIT=1<<48
B=16
SCREEN=1<<20
DIV=UNIT//SCREEN


def codes(items,n):
    need(isinstance(items,list) and len(items)==n and all(type(x) is int for x in items),"integer state codes")
    return np.array(items,dtype=np.int64)


def bound_abs(x):
    return max(abs(int(x.min())),abs(int(x.max())))


def verify_events(rep,s,req,request,anchors):
    """A complete independent catalog scan replaces both postings and sorted weight groups."""
    n=len(s)
    pool=rep.pool
    i,j,weights,flip=pool
    m=len(i)
    history=req["discovery"]
    events=history["epochs"]
    need(len(events)==len(req["decisions"]) and events,"one cold query per observed decision")
    first=events[0]
    current_x=codes(first["current_root_xc"],n)
    current_rc=first["current_root_rc"]
    need(type(current_rc) is int and current_rc>=0 and bound_abs(current_x)<2*SCREEN,"coarse root range")
    # Store only independently bound earlier request identities. A later trace
    # cannot declare an arbitrary new old root state or anchor weight table.
    anchors[request]=(opinion_hash(s),current_x,current_rc)
    origin=history["anchor_request"]
    need(type(origin) is int and 1<=origin<=request,"future/unseen anchor")
    origin_op,old,old_rc=anchors[origin]
    need(np.array_equal(codes(first["anchor_xc"],n),old) and first["anchor_rc"]==old_rc and
         history["anchor_opinion_hash"]==origin_op and history["anchor_age"]==request-origin,"old anchor identity")
    need(history["source_root_version"]==req["root_version_id"],"structural root identity")
    if history["action"] in ("rebuild","initial-build"):
        need(origin==request,"rebuild did not use new root")
    else:
        need(history["action"]=="retain" and origin<request,"retained anchor origin")
    if req["arm"]=="rebuild-always": need(origin==request,"rebuild policy")
    if req["arm"]=="retain-always": need(origin==1,"retain policy")
    if req["arm"]=="adaptive" and request>1:
        pred=history["predicted"]
        need(pred and (pred["estimated_extra_s"]<=pred["previous_build_s"])==(history["action"]=="retain"),
             "adaptive decision inconsistency")
    # Candidate-wise root upper bounds in physical scaled-operator SCORE-grid units.
    a,b=old[i],old[j]
    norm=bound_abs(old)+old_rc
    need(4*norm**2*int(weights.max())<2**62,"root key range")
    keys=weights*np.where(flip,-4*a*b+4*old_rc*(abs(a)+abs(b)+old_rc),
                          (abs(a-b)+2*old_rc)**2)
    A=rep.A
    chosen=set()
    visits=emitted=0
    parent=req["root_version_id"]
    for epoch,(e,d) in enumerate(zip(events,req["decisions"]),1):
        need(e["epoch"]==epoch and e["parent_version_id"]==d["parent_version_id"]==parent and
             e["anchor_request"]==origin and e["anchor_opinion_hash"]==origin_op,"epoch/anchor binding")
        x=codes(e["fine_x"],n)
        need(2*int(A.diagonal().max())*bound_abs(x)+UNIT<2**63,"fine residual overflow")
        residual=A@x-s*(UNIT//65536)
        raw=bound_abs(residual)
        radius=(raw+B-1)//B
        need(e["fine_radius"]==radius and e["raw_residual"]==raw,"fine root/child residual binding")
        xc=codes(e["current_xc"],n)
        need(bound_abs(xc)<2*SCREEN and bound_abs(x-xc*DIV)<=DIV//2,"coarse rounding")
        rc=(bound_abs(x-xc*DIV)+raw+DIV-1)//DIV
        need(e["current_rc"]==rc and (epoch>1 or
             (np.array_equal(xc,current_x) and rc==current_rc)),"current coarse state and radius")
        base=codes(e["base_drift"],n)
        need(np.array_equal(base,abs(xc-current_x)+rc+current_rc),"base root drift")
        if epoch==1:
            needed=(bound_abs(x-current_x*DIV)+radius+DIV-1)//DIV
            need(needed<=current_rc,"new root coarse error insufficient")
        drift=codes(e["drift"],n)
        # This floor on claimed drift accounts explicitly for the uncertainty
        # of both actual states. It applies also to a different old opinion.
        needed=(abs(x-old*DIV)+radius+DIV-1)//DIV+old_rc
        need(np.array_equal(drift,abs(xc-old)+rc+old_rc) and np.all(drift>=needed) and
             np.all(drift<=4*SCREEN) and e["max_drift"]==int(drift.max()),"invalid root/child cross-opinion drift")
        tau=e["tau"]
        need(type(tau) is int and tau>0 and tau&(tau-1)==0 and tau==d["tau"],"dyadic index threshold")
        target=e["target"]
        need(type(target) is int and 0<=target<=-(-d["low"]//(B*DIV**2)),"coarse threshold")
        shift=min(8*(SCREEN//B)*tau,8*norm*tau+4*tau**2)
        prefix=keys>=target-weights*shift
        moving=drift>tau
        prededup=int(prefix.sum())+int(moving[i].sum())+int(moving[j].sum())
        need(prededup==e["pre_dedup_visits"],"posting and group-visit witness")
        excluded=e["excluded_ids"]
        need(all(type(z) is int and 0<=z<m for z in excluded) and
             excluded==sorted(set(excluded)) and chosen.issubset(excluded),"noncanonical exclusions")
        seeds=e["seed_ids"]
        need(seeds==sorted(set(seeds)) and not chosen.intersection(seeds) and
             set(excluded)==chosen.union(seeds) and len(seeds)==d["seed_visits"],"hot/committed exclusions")
        if epoch==1: need(seeds==req["initial_seed_ids"],"initial seed scope")
        keep=prefix|moving[i]|moving[j]
        if excluded: keep[excluded]=False
        ids=np.flatnonzero(keep).astype(np.int32)
        need(len(ids)==e["returned_ids"]==d["cold_endpoint_visits"] and
             digest(ids)==e["returned_ids_sha256"],"cold query ID coverage")
        visits+=m
        emitted+=len(ids)
        k=d["winner_id"]
        if k is not None:
            chosen.add(k)
            A=changed(A,pool,k)
            parent=hashlib.sha256(f'{parent}/{k}'.encode()).hexdigest()
    return dict(request=request,arm=req["arm"],action=history["action"],origin=origin,epochs=len(events),
                index_rows_reconstructed=visits,cold_ids_returned=emitted,opinion_hash=opinion_hash(s))


def replay(path):
    start=perf_counter()
    trace=json.loads(path.read_text())
    graph=ROOT/f'data/projected/{trace["graph"]}_weighted.npz'
    graph_hash=hashlib.sha256(graph.read_bytes()).hexdigest()
    need(trace["graph_hash"]==graph_hash,"input graph binding")
    rep=CatalogReplay(sparse.load_npz(graph))
    requests=trace["requests"]
    need(trace.get("audit") and trace.get("status")=="completed" and trace.get("delta")==0 and
         trace.get("hot_size")==64 and len(requests)==trace.get("requests_planned") and
         len(requests)>=2 and trace["weight_mode"]=="weighted","complete registered A1 audit required")
    inputs=opinions(rep.A.shape[0],trace["stream"],trace["seed"],len(requests))
    result=[]
    anchors={}
    previous_origin=None
    for k,(req,s) in enumerate(zip(requests,inputs),1):
        history=req["discovery"]
        need(req["budget"]==trace["budget"] and req["query_position"]==k and
             trace["root_version_id"]==req["root_version_id"],"registered request contract")
        if k==1: need(history["action"]=="initial-build","absent first index")
        if history["action"]=="retain":
            need(history["anchor_request"]==previous_origin,"retained index was not active")
        previous_origin=history["anchor_request"]
        need(req["arm"]==trace["arm"] and req["method"]=="D" and req["delta"]==0,"A1 trial contract")
        need(req["initial_seed_ids"]==root_seed(rep,s,req,64),"current root top-64 mismatch")
        checked=worker_request(rep,req,s,graph_hash,kind="weighted")
        event=verify_events(rep,s,req,k,anchors)
        result.append(dict(**event,full_decisions=checked))
    return dict(trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),requests=result,
                checker_seconds=perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("trace",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.write_text(json.dumps(replay(args.trace),indent=2)+"\n")
