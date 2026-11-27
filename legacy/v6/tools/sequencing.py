"""Conditioned safe-order search on observed operators; run from the paper directory."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from workloads import load_graph, system

DATA = Path(__file__).resolve().parents[1] / "data"
TOL = 1e-10


def local_set(W, rng):
    """Choose overlapping edits near a uniformly sampled observed negative edge."""
    i, j = W.nonzero()
    neg = np.flatnonzero((i < j) & (W.data < 0))
    pick = rng.choice(neg)
    q = {int(i[pick]), int(j[pick])}
    for _ in range(8):
        nbr = np.unique(np.concatenate([W[u:u+1].indices for u in sorted(q)]))
        choices = np.setdiff1d(nbr, list(q))
        if len(choices):
            q.add(int(rng.choice(choices)))
    flips, adds = [], []
    for u in sorted(q):
        for v in sorted(q):
            if v <= u:
                continue
            w = W[u, v]
            if w < 0:
                flips.append((u, v, float(-w), True))
            if w == 0:
                adds.append((u, v, 1., False))
    if len(flips) < 2 or len(adds) < 3:
        return []
    k = min(4, len(flips))
    if len(adds) < 7-k:
        return []
    return [flips[h] for h in rng.choice(len(flips), k, replace=False)] + [
        adds[h] for h in rng.choice(len(adds), 7-k, replace=False)]


def representation(A, edits):
    q = sorted({u for e in edits for u in e[:2]})
    D = []
    for u, v, w, flip in edits:
        e = np.eye(len(q))
        b, c = e[q.index(u)]-e[q.index(v)], e[q.index(u)]+e[q.index(v)]
        D.append(w*(np.outer(b,b) - (np.outer(c,c) if flip else 0)))
    rhs = np.zeros((A.shape[0],len(q)))
    rhs[q] = np.eye(len(q))
    G = splu(A.tocsc()).solve(rhs)[q]
    return q, np.asarray(D), G


def potential_forms(D, G):
    I = np.eye(len(G))
    H = []
    for mask in range(1 << len(D)):
        update = sum((d for h,d in enumerate(D) if mask >> h & 1), np.zeros_like(G))
        H.append(update @ np.linalg.solve(I+G@update,I))
    return np.asarray(H)


def search(scores, k, greedy=False, descend=None):
    full = (1 << k)-1
    seen = set()
    trace = []
    back = 0
    def visit(mask):
        nonlocal back
        seen.add(mask)
        trace.append(mask)
        if mask == full:
            return []
        gains = scores(mask)
        options = sorted((h for h in range(k) if not mask >> h & 1 and gains[h] > TOL),
                         key=lambda h: (-gains[h],h))
        for h in options[:1] if greedy else options:
            child = mask | (1 << h)
            if child in seen:
                continue
            if descend is not None and child != full:
                descend(mask,h)
            order = visit(child)
            if order is not None:
                return [h]+order
            back += 1
        return None
    order = visit(0)
    return order, trace, back


def find_cases(name, target, seed):
    rng = np.random.default_rng(seed)
    W = load_graph(name, n=400)
    A = system(W)
    cases = []
    proposals = 0
    tried = 0
    while len(cases) < target and proposals < 3000:
        proposals += 1
        edits = local_set(W,rng)
        if not edits:
            continue
        q,D,G = representation(A,edits)
        H = potential_forms(D,G)
        X = rng.normal(size=(512,len(q)))
        # Choose z_0 on the endpoints and set s=A z_0, then enforce the opinion box.
        C = A[:,q].tocsr()
        C = C[np.diff(C.indptr) > 0]
        scale = np.abs(C @ X.T).max(axis=0)
        X /= scale[:,None]
        F = X @ H.reshape(-1,len(q)).T
        F = np.einsum("bmi,bi->bm",F.reshape(len(X),len(H),len(q)),X)
        idx = 1 << np.arange(len(edits))
        masks = np.zeros(len(X),dtype=int)
        rows = np.arange(len(X))
        for _ in edits:
            values = F[rows[:,None],masks[:,None]|idx] - F[rows,masks,None]
            best = values.argmax(axis=1)
            masks |= np.where(values[rows,best]>TOL,idx[best],0)
        tried += len(X)
        locked = (F[:,idx] < -TOL).any(axis=1)
        for t in np.flatnonzero((masks != len(H)-1) & (F[:,-1] > TOL) & locked):
            score = lambda m: F[t,m|idx]-F[t,m]
            order,trace,B = search(score,len(edits))
            if order and 20 <= len(trace) <= 200 and B:
                case=dict(graph=name,edits=edits,q=q,z=X[t].tolist(),order=order,
                                  states=len(trace),backtracks=B,proposal=proposals,
                                  greedy_mask=int(masks[t]),candidate=int(t))
                try:
                    audit(A, case)
                except AssertionError:
                    continue
                cases.append(case)
                print(name,len(cases),proposals,len(trace),B,flush=True)
                break
    return cases, dict(graph=name,proposals=proposals,opinions_tested=tried)



def replay(A, case, reuse):
    from scipy.sparse import coo_matrix
    edits = case["edits"]
    q = case["q"]
    k = len(edits)
    cols = np.asarray([[q.index(e[0]), q.index(e[1])] for e in edits])
    weights = np.asarray([e[2] for e in edits])
    flips = np.asarray([e[3] for e in edits])
    z0 = np.asarray(case["z"])
    rhs = np.zeros((A.shape[0],len(q)))
    rhs[q] = np.eye(len(q))
    s = A[:,q]@z0
    gs = {}
    transitions = {}
    selected = {}
    margin = float("inf")
    def scores(mask):
        nonlocal margin
        if reuse and mask:
            # The DFS enters each subset once. Its scored parent registered this child.
            parent,h = transitions[mask]
            G,z,K = gs[parent]
            pair = cols[h]
            R = G[:,pair]
            G,z = G-R@K[h]@R.T,z-R@K[h]@z[pair]
        else:
            op = A
            if mask:
                rows,columns,data = [],[],[]
                for h,(i,j,w,flip) in enumerate(edits):
                    if mask >> h & 1:
                        rows.extend((i,i,j,j))
                        columns.extend((i,j,i,j))
                        data.extend((0,-2*w,-2*w,0) if flip else (w,-w,-w,w))
                op = A + coo_matrix((data,(rows,columns)),shape=A.shape).tocsc()
            factor = splu(op.tocsc())
            G = factor.solve(rhs)[q]
            z = factor.solve(s)[q]
        # K=(I+D G_pair)^(-1)D works for both singular additions and flips.
        D = np.zeros((k,2,2))
        D[:,0,0] = D[:,1,1] = np.where(flips,0,weights)
        D[:,1,0] = D[:,0,1] = np.where(flips,-2*weights,-weights)
        g = G[cols[:,:,None],cols[:,None,:]]
        K = np.linalg.solve(np.eye(2)+D@g,D)
        v = z[cols]
        gains = np.einsum("ki,kij,kj->k",v,K,v)
        if reuse:
            gs[mask] = G,z,K
            for h in range(k):
                if not mask >> h & 1:
                    transitions[mask|1<<h] = mask,h
        for h in range(k):
            if not mask >> h & 1:
                margin = min(margin, abs(gains[h]))
        selected[mask] = gains
        return gains
    order,trace,back = search(scores,k)
    greedy,gt,_ = search(lambda mask:selected[mask],k,True)
    assert greedy is None and gt[-1] == case["greedy_mask"]
    assert order == case["order"] and len(trace)==case["states"] and back==case["backtracks"]
    return margin



def audit(A,case,ranks=None):
    from scipy.sparse import coo_matrix
    from functools import lru_cache
    q = case["q"]
    s = A[:,q]@np.asarray(case["z"])
    eps = np.finfo(float).eps
    n = len(s)
    edits = case["edits"]
    @lru_cache(None)
    def objective(mask):
        rows,cols,vals = [],[],[]
        for h,(u,v,w,flip) in enumerate(edits):
            if mask >> h & 1:
                rows.extend([u,u,v,v])
                cols.extend([u,v,u,v])
                vals.extend([0,-2*w,-2*w,0] if flip else [w,-w,-w,w])
        M = A + coo_matrix((vals,(rows,cols)),shape=A.shape).tocsr()
        z = splu(M.tocsc()).solve(s)
        r = s-M@z
        k = 2+np.diff(M.indptr).max()
        # gamma_k < 2*k*eps here; inflate reductions and norms by another factor two.
        err_r = 4*k*eps*(np.abs(s)+abs(M)@np.abs(z))
        rad = 2*np.linalg.norm(s)*(np.linalg.norm(r)+np.linalg.norm(err_r))
        rad += 4*n*eps*(abs(s)@abs(z))
        return s@z,rad
    separation = []
    def scores(mask):
        f,err=objective(mask)
        result=np.zeros(len(edits))
        children=[]
        for h in range(len(edits)):
            if mask >> h & 1:
                continue
            g,rad=objective(mask|1<<h)
            gain=f-g
            lower=abs(gain)-(err+rad+eps*abs(gain))*2
            separation.append(lower)
            if gain>0:
                children.append((g,rad))
            result[h]=gain
        children.sort()
        for (g,r),(gg,rr) in zip(children,children[1:]):
            separation.append(gg-g-2*(rr+r+eps*abs(gg-g)))
        if ranks is not None:
            ranks[mask]=sorted((h for h,g in enumerate(result) if g>TOL),key=lambda h:(-result[h],h))
        return result
    order,trace,b=search(scores,len(edits))
    greedy,path,_=search(scores,len(edits),True)
    assert order==case["order"] and greedy is None and path[-1]==case["greedy_mask"]
    assert min(scores(0)) < -TOL
    assert len(trace)==case["states"] and b==case["backtracks"] and min(separation)>TOL
    return min(separation)


def benchmark(name):
    cases=json.loads((DATA/f"sequencing_{name}.json").read_text())["cases"]
    A=system(load_graph(name, n=400))
    with (DATA/f"sequencing_times_{name}.csv").open("w",newline="") as out:
        w=csv.writer(out)
        w.writerow(("case","trial","policy","seconds","min_abs_gain","audit_separation"))
        for t,case in enumerate(cases):
            sep=audit(A,case)
            for r in range(10):
                for reuse in ((True,False) if r%2 else (False,True)):
                    tick=time.perf_counter()
                    margin=replay(A,case,reuse)
                    w.writerow((t,r,"tree" if reuse else "scratch",time.perf_counter()-tick,margin,sep))
            out.flush()




def candidate_replay(A,case,reuse):
    from engine import Edit,ViewManager
    manager=ViewManager(A,[Edit(*e) for e in case["edits"]])
    k=len(manager.edits)
    s=A[:,case["q"]]@np.asarray(case["z"])
    states=[tuple(h for h in range(k) if mask >> h & 1) for mask in range(1<<k)]
    def scores(mask):
        state=states[mask]
        z=manager.solve(state,s)
        return [0. if h in state else manager.score(state,h,z) for h in range(k)]
    def descend(mask,h):
        manager.commit(states[mask],h)
    order,trace,b=search(scores,k,descend=descend if reuse else None)
    assert order==case["order"] and len(trace)==case["states"] and b==case["backtracks"]
    return manager.ledger


def benchmark_candidates(name):
    cases=json.loads((DATA/f"sequencing_{name}.json").read_text())["cases"]
    A=system(load_graph(name,n=400))
    with (DATA/f"sequencing_candidate_times_{name}.csv").open("w",newline="") as out:
        w=csv.writer(out)
        w.writerow(("case","trial","policy","seconds","factorizations","state_rhs","refine_rhs","winner_rhs","transported"))
        for t,case in enumerate(cases):
            audit(A,case)
            for r in range(10):
                for reuse in ((True,False) if r%2 else (False,True)):
                    tick=time.perf_counter()
                    ledger=candidate_replay(A,case,reuse)
                    w.writerow((t,r,"transport" if reuse else "fresh",time.perf_counter()-tick,
                                *(ledger[key] for key in ("factorizations","state_rhs","refine_rhs","winner_rhs","transported"))))
            out.flush()


def stack_replay(A,case,reuse,measure_memory=False,ranks=None):
    from engine import Edit
    from collections import Counter
    from scipy.sparse import coo_matrix
    edits=[Edit(*e) for e in case["edits"]]
    local=[e.local() for e in edits]
    q=[np.asarray([e.i,e.j]) for e in edits]
    k=len(edits)
    full=(1<<k)-1
    seen=set()
    ledger=Counter()
    s=A[:,case["q"]]@np.asarray(case["z"])
    def matrix_bytes(M):
        return M.data.nbytes+M.indptr.nbytes+M.indices.nbytes
    def visit(mask,M,z,grams,retained):
        seen.add(mask)
        if mask==full:
            return []
        fact=None
        factor_bytes=0
        def solve(rhs):
            nonlocal fact,factor_bytes
            if fact is None:
                fact=splu(M)
                ledger["factorizations"]+=1
                if measure_memory:
                    factor_bytes=matrix_bytes(fact.L)+matrix_bytes(fact.U)+fact.perm_r.nbytes+fact.perm_c.nbytes
            return fact.solve(rhs)
        if z is None:
            z=solve(s)
        responses={}
        def response(h):
            if h not in responses:
                U=local[h][0]
                rhs=np.zeros((len(s),U.shape[1]))
                rhs[q[h]]=U
                responses[h]=solve(rhs)
                ledger["response_rhs"]+=U.shape[1]
            return responses[h]
        options=[]
        for h in range(k):
            if mask >> h & 1:
                continue
            U,sign=local[h]
            if h not in grams:
                grams[h]=U.T@response(h)[q[h]]
            v=U.T@z[q[h]]
            alpha=np.linalg.solve(np.diag(sign)+grams[h],v)
            g=float(v@alpha)
            if g>TOL:
                options.append((-g,h,alpha))
            else:
                responses.pop(h,None)
        options.sort(key=lambda x:(x[0],x[1]))
        if ranks is not None:
            assert [item[1] for item in options]==ranks[mask]
        def working_bytes():
            return (retained+matrix_bytes(M)+factor_bytes+z.nbytes+
                    sum(x.nbytes for x in grams.values())+
                    sum(x.nbytes for x in responses.values())+
                    sum(x[2].nbytes for x in options))
        if measure_memory:
            ledger["peak_bytes"]=max(ledger["peak_bytes"],working_bytes())
        for _,h,alpha in options:
            child=mask|1<<h
            if child in seen:
                responses.pop(h,None)
                continue
            if child==full:
                seen.add(child)
                return [h]
            U,sign=local[h]
            Y=response(h)
            child_z=z-Y@alpha
            child_grams={}
            if reuse:
                inv=np.linalg.inv(np.diag(sign)+grams[h])
                for j,G in grams.items():
                    if j!=h:
                        cross=local[j][0].T@Y[q[j]]
                        child_grams[j]=G-cross@inv@cross.T
            e=edits[h]
            delta=U@np.diag(sign)@U.T
            delta=coo_matrix((delta.ravel(),(np.repeat(q[h],2),np.tile(q[h],2))),shape=M.shape)
            child_M=(M+delta).tocsc()
            del Y
            responses.pop(h,None)
            parent_bytes=working_bytes() if measure_memory else 0
            branch=visit(child,child_M,child_z,child_grams,parent_bytes)
            del child_M,child_z,child_grams
            if branch is not None:
                return [h]+branch
            ledger["backtracks"]+=1
        return None
    order=visit(0,A.tocsc(),None,{},0)
    assert order==case["order"] and len(seen)==case["states"] and ledger["backtracks"]==case["backtracks"]
    return ledger


def benchmark_stack(name):
    cases=json.loads((DATA/f"sequencing_{name}.json").read_text())["cases"]
    A=system(load_graph(name,n=400))
    with (DATA/f"sequencing_stack_times_{name}.csv").open("w",newline="") as out:
        w=csv.writer(out)
        w.writerow(("case","trial","policy","seconds","factorizations","response_rhs","peak_payload_bytes"))
        for t,case in enumerate(cases):
            ranks={}
            audit(A,case,ranks)
            peak={reuse:stack_replay(A,case,reuse,True,ranks)["peak_bytes"] for reuse in (True,False)}
            for r in range(10):
                for reuse in ((True,False) if r%2 else (False,True)):
                    tick=time.perf_counter()
                    ledger=stack_replay(A,case,reuse)
                    w.writerow((t,r,"transport" if reuse else "fresh",time.perf_counter()-tick,
                                ledger["factorizations"],ledger["response_rhs"],peak[reuse]))
            out.flush()


def summarize():
    section = [r"""\subsection{Conditioned sequencing with failed branches}
We form connected 400-node induced subgraphs of Bitcoin--Alpha and OTC.
Each proposal grows an endpoint neighborhood from an observed negative edge,
then samples seven overlapping legal edits, including two to four flips.
For each proposed set with enough legal edits we draw 512 Gaussian endpoint
vectors, extend each by zeros to obtain $z_0$, and scale $s=A_0z_0$ into the
opinion box. We retain at most one vector per proposal. Retention requires
at least one initially harmful flip, maximum-positive-gain greedy to stall,
and descending-gain, subset-memoized
depth-first search (DFS) to finish after 20--200 distinct states. These
outcomes describe conditioned examples. Their frequency depends on both
the proposal and the retention rule.

\begin{table}[t]
\centering\footnotesize
\caption{Conditioned sequencing, ten sets per graph. $B$ counts failed child
returns. Ratios compare fresh per-candidate solves to per-candidate transport.
Each set contributes its median paired time ratio over ten repetitions.
Quartiles and percentile 95\% bootstrap intervals summarize those ten set
medians (10,000 set resamples, seed 408).}\label{tab:seqhard}
\begin{tabular}{lrrrrr}\hline
Graph & States & $B$ & Stack ratio & IQR & 95\% CI\\\hline"""]
    stats=[]
    stackwork={policy:np.zeros(2,dtype=int) for policy in ("fresh","transport")}
    peaks={"fresh":0,"transport":0}
    controls=[]
    work={policy:np.zeros(2,dtype=int) for policy in ("fresh","transport")}
    for name in ("Alpha","OTC"):
        d=json.loads((DATA/f"sequencing_{name}.json").read_text())
        cases=d["cases"]
        rows=list(csv.DictReader((DATA/f"sequencing_candidate_times_{name}.csv").open()))
        for r in rows:
            if int(r["trial"])==0:
                work[r["policy"]]+=[int(r["refine_rhs"])+int(r["winner_rhs"]),int(r["factorizations"])]
        controls.append(np.median([np.median(np.array([float(r["seconds"]) for r in rows if r["policy"]=="fresh" and int(r["case"])==t])/np.array([float(r["seconds"]) for r in rows if r["policy"]=="transport" and int(r["case"])==t])) for t in range(len(cases))]))
        rows=list(csv.DictReader((DATA/f"sequencing_stack_times_{name}.csv").open()))
        for r in rows:
            if int(r["trial"])==0:
                pol=r["policy"]
                stackwork[pol]+=[int(r["response_rhs"]),int(r["factorizations"])]
                peaks[pol]=max(peaks[pol],int(r["peak_payload_bytes"]))
        by_key={(int(r["case"]),int(r["trial"]),r["policy"]):float(r["seconds"]) for r in rows}
        ratios=np.asarray([np.median([by_key[t,r,"fresh"]/by_key[t,r,"transport"]
                           for r in range(10)]) for t in range(len(cases))])
        boot=np.random.default_rng(408).choice(ratios,(10000,len(ratios)))
        interval=np.percentile(np.median(boot,axis=1),[2.5,97.5])
        def span(key):
            a=[case[key] for case in cases]
            return f"{min(a)}--{max(a)}"
        def pair(x):
            return f"{x[0]:.2f}--{x[1]:.2f}"
        lines=[name,span("states"),span("backtracks"),f"{np.median(ratios):.2f}",
               pair(np.percentile(ratios,[25,75])),pair(interval)]
        section.append(" & ".join(lines)+r"\\")
        stats.append(d["search"])
    section.append(r"""\hline\end{tabular}
\end{table}
Both policies lazily factor each active DFS state, retain its factor through
sibling visits, and discard it on return. They propagate the equilibrium as
$z'=z-YB^{-1}h$. Fresh re-solves each remaining candidate's response at every
state, retaining already solved positive-child columns for its active DFS
frame. Transport updates each candidate's small Gram from the accepted
response. Both use the same sparse LU backend and positive-gain tolerance.
Times include root setup and DFS and exclude validation and generation.
A separate replay brackets
$s^\top A_T^{-1}s$ using $\lambda_{\min}(A_T)\ge1$, the residual norm,
and floating-point dot-product allowances. It checks all gains and all
positive-child ordering comparisons on the visited states.""")
    section.append(f"The single-active-factor control produced fresh/transport ratios of {controls[0]:.2f} and {controls[1]:.2f}. Across the 20 sets, transport reduced candidate and winner response columns from "
                   f"{work['fresh'][0]:,} to {work['transport'][0]:,}. The backend retains one active factor. "
                   f"Revisiting parents for winner responses increased factorizations from {work['fresh'][1]} "
                   f"to {work['transport'][1]}, and the measured time ratios favor fresh solves.")
    section.append(f"With stack retention, fresh and transport required {stackwork['fresh'][1]} and {stackwork['transport'][1]} "
                   f"factorizations and {stackwork['fresh'][0]:,} and {stackwork['transport'][0]:,} response columns, respectively. "
                   f"Maximum retained active-stack numerical payloads were {peaks['fresh']/2**20:.2f} and {peaks['transport']/2**20:.2f} MiB, "
                   "including sparse operators, compressed LU factors, permutations and dense cached arrays. "
                   "These payload counts exclude allocator overhead and transient workspaces.")
    section.append("The two searches examined %s and %s opinion vectors, respectively."%
                   tuple(f'{d["opinions_tested"]:,}' for d in stats))
    (DATA.parent/"sections"/"sequencing.tex").write_text("\n".join(section)+"\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph",choices=["Alpha","OTC"])
    p.add_argument("--target",type=int,default=10)
    p.add_argument("--stack-bench",action="store_true")
    p.add_argument("--candidate-bench",action="store_true")
    p.add_argument("--bench",action="store_true")
    p.add_argument("--summarize",action="store_true")
    args = p.parse_args()
    if args.summarize:
        summarize()
        return
    if not args.graph:
        p.error("--graph is required for search and benchmark")
    with threadpool_limits(limits=1):
        if args.stack_bench:
            benchmark_stack(args.graph)
            return
        if args.candidate_bench:
            benchmark_candidates(args.graph)
            return
        if args.bench:
            benchmark(args.graph)
            return
        cases,stats=find_cases(args.graph,args.target,8192 if args.graph=="Alpha" else 9172)
        (DATA/f"sequencing_{args.graph}.json").write_text(json.dumps(dict(cases=cases,search=stats),indent=2))
    

if __name__ == "__main__":
    main()

