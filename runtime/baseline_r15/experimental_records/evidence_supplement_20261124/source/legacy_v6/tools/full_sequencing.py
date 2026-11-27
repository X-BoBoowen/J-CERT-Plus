"""Full-operator conditioned sequencing. Run from work/paper."""
import argparse
import csv
import json
from collections import defaultdict
from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from sequencing import DATA, TOL, potential_forms, search, stack_replay
from workloads import load_graph, system


def twins(W, excluded):
    groups = defaultdict(list)
    for u in np.flatnonzero(np.diff(W.indptr) == 1):
        if u not in excluded:
            row = W[u]
            groups[(row.indices[0], row.data[0])].append(int(u))
    pairs = [(a, b, 1., False) for group in groups.values()
             for a, b in zip(group[::2], group[1::2])]
    return pairs[:8]


def project(A, factor, q):
    rhs = np.zeros((A.shape[0], len(q)))
    rhs[q] = np.eye(len(q))
    R = factor.solve(rhs)
    G = R[q]
    return R, np.linalg.inv((G + G.T) / 2)


def deltas(edits, q):
    blocks = []
    for u, v, w, flip in edits:
        a, b = q.index(u), q.index(v)
        D = np.zeros((len(q), len(q)))
        D[a,b] = D[b,a] = -2*w if flip else -w
        D[a,a] = D[b,b] = 0 if flip else w
        blocks.append(D)
    return np.array(blocks)


def generate(name):
    W = load_graph(name)
    A = system(W)
    factor = splu(A.tocsc())
    ids = breadth_first_order(W, int(np.diff(W.indptr).argmax()), directed=False,
                              return_predecessors=False)
    proposals = json.loads((DATA/f"sequencing_{name}.json").read_text())["cases"]
    rng = np.random.default_rng(217)
    cases = []
    for source in proposals:
        core = [(int(ids[u]), int(ids[v]), w, f) for u,v,w,f in source["edits"]]
        q = sorted({u for e in core for u in e[:2]})
        G = project(A, factor, q)[0][q]
        H = potential_forms(deltas(core, q), G)
        satellites = twins(W, q)
        assert len(satellites) == 8
        edits = core + satellites
        idx = 1 << np.arange(7)
        found = False
        for batch in range(24):
            X = rng.normal(size=(512, len(q)))
            X /= (2*np.abs(A[:,q]@X.T).max(axis=0))[:,None]
            X = np.round(X*2**36)/2**36
            F = np.einsum("bmi,bi->bm",(X@H.reshape(-1,len(q)).T).reshape(len(X),128,len(q)),X)
            mask = np.zeros(len(X), dtype=int)
            rows = np.arange(len(X))
            for _ in core:
                gains = F[rows[:,None],mask[:,None]|idx]-F[rows,mask,None]
                best = gains.argmax(axis=1)
                mask |= np.where(gains[rows,best]>TOL,idx[best],0)
            for t in np.flatnonzero((mask!=127) & (F[:,-1]>TOL) & (F[:,idx]<-TOL).any(axis=1)):
                score = lambda m: F[t,m|idx]-F[t,m]
                order, trace, back = search(score,7)
                if order is None or not 20 <= len(trace) <= 100:
                    continue
                endpoints = q+[u for e in satellites for u in e[:2]]
                z = list(X[t])
                for j in range(8):
                    a = (24-j)/64
                    z.extend([a,-a])
                case = dict(graph=name, edits=edits, q=endpoints, z=z,
                            order=list(range(7,15))+order, states=8+len(trace),
                            backtracks=back, greedy_mask=(255<<7)|int(mask[t]),
                            source=len(cases), batch=batch, candidate=int(t))
                try:
                    audit(A, factor, case)
                except AssertionError:
                    continue
                cases.append(case)
                found = True
                print(name,len(cases),case["states"],flush=True)
                break
            if found:
                break
        if not found:
            raise RuntimeError("No admissible core for proposal")
    (DATA/f"full_sequencing_{name}.json").write_text(json.dumps(cases,indent=2))


def audit(A, factor, case):
    q = case["q"]
    edits = case["edits"]
    x0 = np.zeros(A.shape[0])
    x0[q] = case["z"]
    s = A@x0
    assert np.max(abs(s))<=1
    R, _ = project(A,factor,q)
    G = R[q]
    blocks = deltas(edits,q)
    eps = np.finfo(float).eps
    worst = []
    ranks = {}
    @lru_cache(None)
    def objective(mask):
        D = sum((block for j,block in enumerate(blocks) if mask>>j&1),np.zeros_like(G))
        y = np.linalg.solve(np.eye(len(q))+D@G,D@x0[q])
        x = x0-R@y
        dx = D@x[q]
        r = s-A@x
        r[q] -= dx
        # Bound both sparse and endpoint matvec roundoff componentwise.
        mag = abs(s)+abs(A)@abs(x)
        mag[q] += abs(D)@abs(x[q])
        k = max(int(np.diff(A.indptr).max()),len(q))+5
        r_bound = 2*np.linalg.norm(abs(r)+4*k*eps*mag)
        nz = np.flatnonzero(s)
        f = s[nz]@x[nz]
        rad = 2*np.linalg.norm(s)*r_bound + 4*len(nz)*eps*(abs(s[nz])@abs(x[nz]))
        return f,rad
    def scores(mask):
        f,r = objective(mask)
        gains = np.zeros(len(edits))
        pos = []
        for j in range(len(edits)):
            if mask>>j&1:
                continue
            g,t = objective(mask|1<<j)
            gains[j] = f-g
            worst.append(abs(f-g)-2*(r+t+eps*abs(f-g)))
            if gains[j]>TOL:
                pos.append((g,t,j))
        pos.sort()
        for (g,r1,_),(v,r2,_) in zip(pos,pos[1:]):
            worst.append(v-g-2*(r1+r2+eps*abs(v-g)))
        ranks[mask]=[j for _,_,j in pos]
        return gains
    order,trace,back = search(scores,len(edits))
    greedy,path,_ = search(scores,len(edits),True)
    assert order==case["order"] and back==case["backtracks"] and len(trace)==case["states"]
    assert greedy is None and path[-1]==case["greedy_mask"] and min(worst)>TOL
    return ranks,min(worst)


def benchmark(name):
    cases = json.loads((DATA/f"full_sequencing_{name}.json").read_text())
    A = system(load_graph(name))
    factor = splu(A.tocsc())
    with (DATA/f"full_sequencing_times_{name}.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["case","repeat","policy","setup","search","rhs","factors","sep"])
        for i,case in enumerate(cases):
            ranks,sep=audit(A,factor,case)
            q=case["q"]
            local=dict(case,q=list(range(len(q))),
                       edits=[(q.index(a),q.index(b),v,h) for a,b,v,h in case["edits"]])
            for rep in range(10):
                tick=perf_counter()
                _,S=project(A,splu(A.tocsc()),q)
                M=sparse.csc_matrix(S)
                setup=perf_counter()-tick
                for reuse in ((True,False) if rep%2 else (False,True)):
                    tick=perf_counter()
                    ledger=stack_replay(M,local,reuse,ranks=ranks)
                    elapsed=perf_counter()-tick
                    w.writerow([i,rep,"transport" if reuse else "fresh",setup,elapsed,
                                ledger["response_rhs"],ledger["factorizations"],sep])
            f.flush()


def summarize(name):
    cases = json.loads((DATA/f"full_sequencing_{name}.json").read_text())
    with (DATA/f"full_sequencing_times_{name}.csv").open() as f:
        rows = {(int(r["case"]),int(r["repeat"]),r["policy"]):r for r in csv.DictReader(f)}
    ratios = []
    for i in range(len(cases)):
        pairs = []
        for rep in range(10):
            a = rows[i,rep,"fresh"]
            b = rows[i,rep,"transport"]
            setup = float(a["setup"])
            fresh, transport = float(a["search"]), float(b["search"])
            pairs.append([fresh/transport,(setup+fresh)/(setup+transport)])
        ratios.append(np.median(pairs,axis=0))
    totals = np.asarray(ratios)[:,1]
    boot = np.random.default_rng(408).choice(totals,(10000,len(cases)))
    low, high = np.percentile(np.median(boot,axis=1),[2.5,97.5])
    spans = []
    for key in ("states","backtracks"):
        v = [c[key] for c in cases]
        spans.append(f"{min(v)}--{max(v)}")
    dfs,total = np.median(ratios,axis=0)
    print(" & ".join([name,*spans,f"{dfs:.2f}",f"{total:.3f}",f"{low:.3f}--{high:.3f}"])+r"\\")
    for policy in ("fresh","transport"):
        print(policy, "total response columns",sum(int(rows[i,0,policy]["rhs"]) for i in range(len(cases))))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("graph",choices=["Alpha","OTC"])
    p.add_argument("--bench",action="store_true")
    p.add_argument("--summarize",action="store_true")
    args=p.parse_args()
    with threadpool_limits(limits=1):
        (summarize if args.summarize else benchmark if args.bench else generate)(args.graph)


