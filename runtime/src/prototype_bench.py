"""Serial, bounded feasibility workers. See protocol/PROTOTYPE_PROTOCOL.json."""
import argparse
from dataclasses import asdict
from fractions import Fraction as Q
from itertools import combinations
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from time import perf_counter as clock
import traceback

from .anchor import Anchor, Edit, Graph, Prices, State, choose_plan, tangent
from .implicit_catalog import ImplicitNonedges, pair_cap

ROOT=Path(__file__).resolve().parents[1]
N=64
TAU=1/32
def graph():
    return Graph(N,tuple((i,i+1,1 if i%4 else -1,1.) for i in range(0,N,2)))

def catalog(g):
    return tuple(Edit(i,j,'flip' if (i,j) in g.lookup else 'insert',1.)
                 for i,j in combinations(range(g.n),2) if g.lookup.get((i,j))!=(1,1.))

def stream(family):
    rng=random.Random(104729)
    s=[rng.randint(-65536,65536) for _ in range(N)]
    out=[]
    for q in range(4):
        if q and family=='slow':
            for i in rng.sample(range(N),math.ceil(N/10)):
                s[i]=round((9*s[i]+rng.randint(-65536,65536))/10)
        if q and family=='independent':
            s=[rng.randint(-65536,65536) for _ in range(N)]
        out.append(tuple(v/65536 for v in s))
    return out

class MatchingOracle:
    """Exact current matching responses, valid only for this generated root."""
    def __init__(self,g,s):
        self.s=tuple(map(Q,s))
        z=[None]*g.n
        for i,j,sign,w in g.edges:
            if w!=1. or (i^1)!=j: raise ValueError("matching fixture required")
            z[i]=(2*self.s[i]+sign*self.s[j])/3
            z[j]=(2*self.s[j]+sign*self.s[i])/3
        self.z=tuple(z)
        self.state=State.certify(g,s,tuple(map(float,z)))

    def score(self,e):
        i,j=e.i,e.j
        if e.kind=='flip':
            # Inverse before/after differs only in the signed off-diagonal entries.
            return -4*self.s[i]*self.s[j]/3
        # Distinct 2x2 blocks: effective resistance b^T A^-1 b=4/3.
        return 3*(self.z[i]-self.z[j])**2/7

def direct_gain(g,s,e):
    """Independent direct elimination on affected components before and after edit."""
    v=sorted({e.i,e.j,e.i^1,e.j^1})
    rhs=list(map(Q,(s[i] for i in v)))
    a=[[Q(i==j) for j in v] for i in v]
    for vertex in v:
        partner=vertex^1
        if vertex>=partner: continue
        i,j=v.index(vertex),v.index(partner)
        sign,w=g.lookup[vertex,partner]
        a[i][i]+=1; a[j][j]+=1
        a[i][j]-=sign; a[j][i]-=sign
    def energy(matrix):
        aug=[row[:]+[b] for row,b in zip(matrix,rhs)]
        for k in range(len(v)):
            p=aug[k][k]
            aug[k]=[t/p for t in aug[k]]
            for i in range(len(v)):
                if i!=k:
                    factor=aug[i][k]
                    aug[i]=[t-factor*h for t,h in zip(aug[i],aug[k])]
        return sum(b*row[-1] for b,row in zip(rhs,aug))
    before=energy(a)
    i,j=v.index(e.i),v.index(e.j)
    if e.kind=='insert':
        a[i][i]+=1; a[j][j]+=1
        a[i][j]-=1; a[j][i]-=1
    else:
        a[i][j]-=2; a[j][i]-=2
    return before-energy(a)

def oracle_winner(g,s,edges):
    best=Q(0); winner=None
    for e in edges:
        v=direct_gain(g,s,e)
        if v>best: winner,best=e,v
    return winner,best

def better(e,v,w,b):
    return v>b or (v==b and v>0 and (w is None or e<w))

def lower_float(v):
    x=float(v)
    return math.nextafter(x,-math.inf) if Q(x)>v else x

def anchor_query(g,edges,s,previous,warm,policy,prices):
    start=clock()
    o=MatchingOracle(g,s); st=o.state
    state_s=clock()-start
    manage=clock(); build_s=0
    if previous is None:
        start_build=clock(); a=Anchor(st,edges); build_s=clock()-start_build
    else:
        a=previous
    if warm is None:
        warm=max((rows[-1] for rows in a.groups.values()),key=lambda row:row[0],default=(0.,None))[1]
    t=clock(); v=o.score(warm) if warm else Q(0); score_s=clock()-t
    best,w=(v,warm) if v>0 else (Q(0),None)
    threshold=lower_float(best)
    if previous is None:
        action='initial-build'
    elif policy=='rebuild-always':
        action='rebuild'
    else:
        p=a.plan(st,threshold,TAU)
        action=choose_plan(policy,p,len(edges),prices)
    if action=='rebuild':
        t=clock(); a=Anchor(st,edges); build_s=clock()-t
    if action!='retain': p=a.plan(st,threshold,0.)
    t=clock(); candidates,cold=a.frontier(p); gather_s=clock()-t
    # Use actual drift for retained frontier, then exact current tangent on survivors.
    # No old gain, incumbent lower, or response is transferred.
    management_s=clock()-manage-score_s
    check_s=0.; scored=int(warm is not None)
    for e in sorted(candidates):
        if e==warm: continue
        t=clock()
        u=a.local(e,p.drift)
        if action=='retain' and u>=best: u=min(u,tangent(st,e).upper)
        needed=(u>=best)
        check_s+=clock()-t
        if needed:
            t=clock(); value=o.score(e); score_s+=clock()-t; scored+=1
            if better(e,value,w,best): w,best=e,value
    if not cold<best: raise ArithmeticError("missing strict cold certificate")
    price_start=clock()
    # One measured prefix gather item is a surrogate until actual hot postings are seen.
    update=Prices(build_s/max(1,len(edges)) if build_s else prices.rebuild,
                  gather_s/max(1,p.postings+p.prefix),
                  check_s/max(1,len(candidates)-1),score_s/max(1,scored),
                  scored/max(1,len(candidates)))
    management_s+=clock()-price_start
    wall=clock()-start
    row=dict(policy=policy,action=action,winner=asdict(w) if w else None,gain=str(best),wall=wall,
             state_s=state_s,management_s=management_s,build_s=build_s,gather_s=gather_s,
             check_s=check_s,response_s=score_s,other_s=wall-state_s-management_s-check_s-score_s,
             hot_vertices=len(p.hot),posting_visits=p.postings,prefix=p.prefix,cold_upper=cold if math.isfinite(cold) else None,
             frontier=len(candidates),score_calls=scored,prices_in=asdict(prices),prices_next=asdict(update))
    return a,w,update,row

def query_a2(g,s,method):
    start=clock()
    o=MatchingOracle(g,s); st=o.state
    blocked=g.lookup.keys(); legal=g.n*(g.n-1)//2-len(blocked)
    h=None; explicit=None
    if method=='explicit-sorted':
        explicit=sorted([(-pair_cap(st.x[i],st.x[j],st.radius,1.),(i,j)) for i,j in combinations(range(g.n),2) if (i,j) not in g.lookup])
    else:
        h=ImplicitNonedges(st.x,st.radius,1.,blocked)
    setup=clock()-start
    b=Q(0); winner=None; responses=0; checked_flips=0
    for i,j,sign,_ in g.edges:
        if sign<0:
            checked_flips+=1; e=Edit(i,j,'flip',1.)
            value=o.score(e); responses+=1
            if better(e,value,winner,b): winner,b=e,value
    k=0; tail=-math.inf
    while True:
        tail=h.tail if h else (-explicit[k][0] if k<len(explicit) else -math.inf)
        if tail==-math.inf or (method!='naive-heap' and (tail<b or (b==0 and tail<=0))): break
        item=h.pop() if h else (explicit[k][1],-explicit[k][0])
        if item is None: tail=-math.inf; break
        pair,u=item; k+=1
        if u<b: continue
        e=Edit(*pair,'insert',1.)
        value=o.score(e); responses+=1
        if better(e,value,winner,b): winner,b=e,value
    wall=clock()-start
    return dict(method=method,winner=asdict(winner) if winner else None,gain=str(b),
                wall=wall,setup_s=setup,decision_s=wall-setup,initial_heap=g.n-1 if h else 0,
                explicit_insert_keys=len(explicit) if explicit is not None else 0,legal_insertions=legal,
                checked_insertions=k,rejected=h.rejected if h else len(blocked),checked_flips=checked_flips,
                score_calls=responses,cold_upper=tail if math.isfinite(tail) else None)

def existing_d(g,s):
    import numpy as np
    from scipy import sparse
    base=ROOT/'baseline_r15/experimental_records/legacy_v6/tools'
    sys.path.insert(0,str(base))
    from long_promotion import search
    from promotion import complete_pool
    from promotion_index_profile import compile_index,FusedClassIndex
    from signed_promotion import build_screen
    with tempfile.TemporaryDirectory() as d:
        t=clock(); broad=build_screen(d); lib=compile_index(d); prep=clock()-t
        start=clock()
        ii,jj,ss,_=zip(*g.edges)
        w=sparse.coo_matrix((np.array(ss,dtype=np.int8),(ii,jj)),shape=(g.n,g.n)).tocsr()
        w=w+w.T
        pool=complete_pool(w)
        si=np.rint(np.array(s)*65536).astype('int64')
        setup=clock()-start
        rows,lo,hi,_=search(w,si,pool,'promoted',budget=1,hot_size=64,adaptive=True,
                  operator_scale=1,dyadic_tau=True,trace=True,broad_pass=broad,proof_unit=1<<48,
                  index_factory=lambda *args:FusedClassIndex(lib,*args),woodbury_state=True,retain_winner=True,
                  directional=True,transport=True,matched_screen=True,strict_screen=True)
        wall=clock()-start
        if not rows: raise ArithmeticError("D ended with no positive winner")
        r=rows[0]; win=Edit(int(r['i']),int(r['j']),'flip' if r['flip'] else 'insert',1.)
        return dict(method='existing-D',winner=asdict(win),gain=None,low=int(lo),high=int(hi),denominator=1<<96,
                    native_compile_load_s=prep,wall=wall,graph_catalog_s=setup,decision_including_commit_s=wall-setup,
                    raw_decision=r)

def run(route,emit):
    g=graph()
    if route=='A1':
        for fam,methods in [('slow',('rebuild-always','adaptive','retain-always')),('independent',('retain-always','adaptive','rebuild-always'))]:
            inputs=stream(fam); decisions=[]
            for policy in methods:
                t=clock(); edges=catalog(g); catalog_s=clock()-t
                a=w=None; prices=Prices(0,0,0,0)
                for idx,s in enumerate(inputs):
                    a,w,prices,row=anchor_query(g,edges,s,a,w,policy,prices)
                    decisions.append((idx,row))
                    emit(dict(route=route,family=fam,q=idx,catalog_s=catalog_s if not idx else 0,**row))
            for idx,s in enumerate(inputs):
                t=clock(); es=catalog(g); w,b=oracle_winner(g,s,es); checker_s=clock()-t
                verified={row['policy']:(row['winner']==asdict(w) and Q(row['gain'])==b) for q,row in decisions if q==idx}
                emit(dict(route=route,kind='independent_reference',family=fam,q=idx,winner=asdict(w),gain=str(b),checked=len(es),checker_s=checker_s,verified=verified))
    else:
        s=stream('slow')[0]; decisions=[]
        for method in ['naive-heap','explicit-sorted','existing-D','implicit-tail']:
            try:
                row=existing_d(g,s) if method=='existing-D' else query_a2(g,s,method)
                decisions.append(row); emit(dict(route=route,**row))
            except Exception as exc:
                emit(dict(route=route,method=method,error=repr(exc),traceback=traceback.format_exc()))
        t=clock(); es=catalog(g); w,b=oracle_winner(g,s,es); checker_s=clock()-t
        verified={}
        for row in decisions:
            value_ok=(Q(row['low'],row['denominator'])<=b<=Q(row['high'],row['denominator'])
                      if row['method']=='existing-D' else Q(row['gain'])==b)
            verified[row['method']]=row['winner']==asdict(w) and value_ok
        emit(dict(route=route,kind='independent_reference',winner=asdict(w),gain=str(b),checked=len(es),checker_s=checker_s,verified=verified))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--route',choices=['A1','A2'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    with args.output.open('x') as f:
        def emit(obj):
            f.write(json.dumps(obj,allow_nan=False,default=lambda x:x.item())+'\n'); f.flush()
        run(args.route,emit)
