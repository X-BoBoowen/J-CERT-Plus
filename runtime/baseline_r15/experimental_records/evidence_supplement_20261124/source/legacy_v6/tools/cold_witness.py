"""Independently certify budget-exhausting two-hop insertion and negative-flip queries."""
from math import isqrt
import argparse
import json
import time
from bisect import bisect_left
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
from standalone_verifier import exact_dot, require

D = 2**26


def inputs(graph, size):
    g = load_npz(graph).tocsr()
    require(g.has_canonical_format and g.shape[0] == g.shape[1] and
            (g != g.T).nnz == 0 and not g.diagonal().any() and
            np.all((g.data==1)|(g.data==-1)), "graph")
    g = g.astype(np.int64)
    if size == -1:
        neighbors=[set(map(int,g.indices[g.indptr[i]:g.indptr[i+1]]))
                   for i in range(g.shape[0])]
        edges=[]
        for i,adj in enumerate(neighbors):
            two=set()
            for j in adj:
                two.update(neighbors[j])
            two.difference_update(adj)
            start,end=g.indptr[i:i+2]
            negative=set(map(int,g.indices[start:end][g.data[start:end]<0]))
            for j in sorted(two | negative):
                if i<j:
                    edges.append((i,j,j in negative))
        return g,edges
    i, j = g.nonzero()
    edges = list(zip(i[(i < j) & (g.data < 0)].tolist(),
                     j[(i < j) & (g.data < 0)].tolist()))
    if size:
        ids = sorted(np.random.default_rng(61).choice(len(edges), size, replace=False))
        edges = [edges[k] for k in ids]
    return g, [(i,j,True) for i,j in edges]


def opinion(n,mixed):
    if mixed:
        return np.random.default_rng(61).integers(-32768,32769,n,dtype=np.int64)*(D//32768)
    return D*(1-2*(np.arange(n)%2))


def enclosure(g, q, edges, state, mixed=False):
    require(q.dtype == np.int32 and q.shape == (g.shape[0],), "codes")
    q = q.astype(np.int64)
    d = np.diff(g.indptr).astype(np.int64)
    require((4*int(d.max())+2*len(state)+2)*int(abs(q).max())+2*D < 2**61, "range")
    s = opinion(len(d),mixed)
    r = s - (1+d)*q + g@q
    for k in state:
        i,j,flip = edges[k]
        if flip:
            r[i] += 2*q[j]
            r[j] += 2*q[i]
        else:
            diff = q[i]-q[j]
            r[i] -= diff
            r[j] += diff
    lo = exact_dot(q,s+r)
    return lo, exact_dot(r,r)


def key(q, edge):
    i,j,flip = edge
    if not flip:
        return (int(q[i])-int(q[j]))**2
    return -4*int(q[i])*int(q[j])


def local_bound(q, edge, radius, degree):
    bound = key(q,edge) + radius
    i,j,flip = edge
    if flip:
        return bound
    # Restrict the child variational correction to its two endpoints.
    # diag(degree+3) dominates the child two-by-two principal block.
    error = isqrt(radius)+1
    a = max(0,abs(int(q[i])-int(q[j]))-error)**2
    return bound-a//(int(degree[i])+3)-a//(int(degree[j])+3)


def replay(graph, size, path, budget=15, *, memory_probe=None):
    start = time.perf_counter()
    g, edges = inputs(graph,size)
    with np.load(path,allow_pickle=False) as z:
        require(len(z.files)==len(set(z.files)), 'duplicate witness names')
        used={'root','cuts','plan','claims','objective_twice'}
        for field in ('plan','cuts'):
            require(z[field].shape==(budget,) and z[field].dtype.kind in 'iu',field)
        root = z['root']
        lo, radius = enclosure(g,root,edges,[],size==-1)
        incidence = [[] for _ in root]
        for k,(i,j,_) in enumerate(edges):
            incidence[i].append(k)
            incidence[j].append(k)
        require(int(abs(root.astype(np.int64)).max()) <= 2**29,'root index range')
        unsorted = np.fromiter((-key(root,e) for e in edges),dtype=np.int64)
        order = np.argsort(unsorted,kind='stable')
        keys = unsorted[order]
        root_memory = memory_probe() if memory_probe else None
        init = time.perf_counter()
        q, chosen, rows = root, [], []
        detail = []
        claims = []
        initial_lo, initial_radius = lo, radius
        degree = np.diff(g.indptr).copy()
        for t in range(budget):
            win = int(z['plan'][t])
            require(0 <= win < len(edges) and win not in chosen,'winner')
            used.add(f'r{t}e{win}')
            child = z[f'r{t}e{win}']
            low, rad = enclosure(g,child,edges,chosen+[win],size==-1)
            threshold = lo-low-rad
            require(threshold>0,'positive')
            # Cold endpoint drift is at most h integer quanta.
            h = D//1024
            drift = (4 if size != -1 else 8)*h*(2*int(np.abs(root.astype(np.int64)).max())+h)
            hot = np.flatnonzero(abs(q.astype(np.int64)-root)>h)
            cut = bisect_left(keys, -(threshold-radius-drift)+1)
            require(int(z['cuts'][t]) == cut,'cut')
            candidates = set(order[:cut].tolist())
            for i in hot:
                candidates.update(incidence[i])
            refined = 0
            for k in candidates.difference(chosen,[win]):
                if local_bound(q,edges[k],radius,degree) >= threshold:
                    used.add(f'r{t}e{k}')
                    rival,_ = enclosure(g,z[f'r{t}e{k}'],edges,chosen+[k],size==-1)
                    require(rival>low+rad,'rival')
                    refined += 1
            first_tail = int(-keys[cut])+radius+drift if cut<len(keys) else None
            require(first_tail is None or first_tail<threshold,'cold suffix')
            claim = dict(hot=hot.tolist(),drift=drift,cut=cut,
                         first_tail_key=int(-keys[cut]) if cut<len(keys) else None)
            claim.update(hot_threshold_quanta=h,decision_delta=0,r0_squared=initial_radius,rt_squared=radius,
                         winner_lower=threshold,Kcut=first_tail)
            claims.append(claim)
            detail.append(dict(parent_radius=radius,child_radius=rad,
                               winner_lower=threshold,cold_tail_upper=first_tail))
            rows.append([len(hot),sum(len(incidence[i]) for i in hot),cut,len(candidates),refined])
            i,j,flip = edges[win]
            if not flip:
                degree[i]+=1
                degree[j]+=1
            chosen.append(win)
            lo,radius,q = low,rad,child
        require(len(z['plan'])==budget,'budget')
        require(set(z.files)==used,'witness coverage')
        require(z['claims'].dtype==np.uint8,'claim encoding')
        def unique_fields(pairs):
            parsed=dict(pairs)
            require(len(parsed)==len(pairs),'duplicate screen field')
            return parsed
        actual=json.loads(z['claims'].tobytes(),object_pairs_hook=unique_fields)
        require(json.dumps(actual,sort_keys=True)==json.dumps(claims,sort_keys=True),'screen claims')
        obj=z['objective_twice']
        require(obj.shape==() and obj.dtype.kind in 'iu' and
                2*(initial_lo-lo-radius)<=int(obj)<=2*(initial_lo+initial_radius-lo),'objective')
    return dict(root_memory=root_memory,final_memory=memory_probe() if memory_probe else None,
                M=len(edges),root_seconds=init-start,replay_seconds=time.perf_counter()-init,
                objective_interval_numerators=[initial_lo-lo-radius,initial_lo+initial_radius-lo],
                claims=claims,denominator=D**2,epochs=detail,plan=chosen,rows=rows,bytes=Path(path).stat().st_size)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('graph'); p.add_argument('size',type=int); p.add_argument('log')
    a=p.parse_args()
    print(json.dumps(replay(a.graph,a.size,a.log)))
