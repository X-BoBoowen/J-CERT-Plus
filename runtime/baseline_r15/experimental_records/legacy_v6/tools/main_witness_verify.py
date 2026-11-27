"""Standalone integer decision replay. Input arrays and expected IDs come from the caller."""
from bisect import bisect_left
from math import isqrt
from time import perf_counter
import json
import numpy as np

D = 1 << 26


def need(condition, message):
    if not condition:
        raise ValueError(message)


def dyadic(values):
    ratios = [float(x).as_integer_ratio() for x in values]
    den = max((b for _, b in ratios), default=1)
    return [a*(den//b) for a,b in ratios], den


def verify(graph, source, pool, expected, budget, path, *, scale=D, revived=None, all_root_signs=False):
    tic = perf_counter()
    need(scale in (1<<26,1<<33), "response scale")
    need(graph.has_canonical_format and graph.shape[0] == graph.shape[1]
         and (graph != graph.T).nnz == 0 and not graph.diagonal().any(), "adjacency")
    i, j, weights, flip = pool
    n = graph.shape[0]
    need(i.dtype.kind in "iu" and j.dtype.kind in "iu" and flip.dtype == bool
         and all(a.shape == (len(i),) for a in pool) and len(i)>0, "catalog arrays")
    need(np.all((i>=0)&(i<j)&(j<n)) and np.all(weights>0)
         and len(np.unique(i.astype(np.int64)*n+j))==len(i), "catalog pairs")
    present = np.asarray(graph[i,j]).ravel()
    need(np.all(present == np.where(flip,-weights,0)), "edit legality")
    need(np.shape(source)==(n,) and np.all(np.abs(source)<=1), "source")
    need(len(expected)==budget and len(set(expected))==budget and all(type(k) is int and
         0<=k<len(i) for k in expected), "published budget/IDs")
    gnum, gd = dyadic(graph.data)
    winum, wd = dyadic(weights)
    B = max(gd, wd)
    # Sparse products and vectorized priorities are performed only after range checks.
    bound = max(map(abs,gnum),default=0)*(B//gd)
    wmax = max(winum)*(B//wd)
    rowcap = B+2*(int(np.diff(graph.indptr).max())*bound+budget*wmax)
    need(rowcap < 2**60, "weight range")
    g = graph.astype(np.int64)
    g.data = np.asarray(gnum,np.int64)*(B//gd)
    w = np.asarray(winum,np.int64)*(B//wd)
    diag = B+np.asarray(abs(g).sum(axis=1)).ravel()
    base_diag = diag.copy()
    su, S = dyadic(source)
    chosen = []

    def enclosure(codes, extra=None):
        need(codes.shape==(n,) and codes.dtype==np.int32, "response codes")
        q = codes.astype(np.int64)
        m = int(np.abs(q).max())
        need(rowcap*m < 2**61 and 16*wmax*(m+scale//1024)**2 < 2**62, "response range")
        aq = base_diag*q-g@q
        for k in chosen+([] if extra is None else [extra]):
            a,b,weight = int(i[k]),int(j[k]),int(w[k])
            if flip[k]:
                aq[a] -= 2*weight*q[b]
                aq[b] -= 2*weight*q[a]
            else:
                v = weight*(q[a]-q[b])
                aq[a] += v
                aq[b] -= v
        r = [B*scale*u-S*int(v) for u,v in zip(su,aq)]
        # Energy units are 1/(B D^2). Round the exact variational form outward.
        total = sum(int(v)*(B*scale*u+e) for v,u,e in zip(q,su,r))
        lower = total//S
        radius = -(-sum(e*e for e in r)//(B*S*S))
        upper = -(-total//S)+radius
        uncertainty = upper-lower
        return q,lower,upper,uncertainty

    def keys(q, ids):
        a,b = q[i[ids]],q[j[ids]]
        return w[ids]*np.where(flip[ids],-4*a*b,(a-b)**2)

    with np.load(path,allow_pickle=False) as log:
        needed = {"root","plan","claims"}
        need(log["plan"].dtype.kind in "iu" and log["plan"].tolist()==expected, "main winner sequence")
        q,lo,hi,rad = enclosure(log["root"])
        root = q.copy()
        initial = lo,hi
        root_gain = None
        root_gains = {}
        if all_root_signs:
            for k in expected:
                name = f"root_k{k}"
                needed.add(name)
                _,rl,ru,_ = enclosure(log[name],k)
                root_gains[k] = [lo-ru,hi-rl]
                need(hi<rl or lo>ru,"unresolved root sign")
            need(any(hi<0 for lo,hi in root_gains.values()),"no root-negative winner")
        if revived is not None:
            need(revived in expected[1:] and flip[revived], "revival target")
            needed.add("root_rival")
            _,rl,ru,_ = enclosure(log["root_rival"],revived)
            root_gain = [lo-ru,hi-rl]
            need(root_gain[1]<0, "negative root gain")
        priority = -keys(root,slice(None))
        order = np.argsort(priority,kind="stable")
        priority = priority[order]
        incidence=[[] for _ in root]
        for k,(a,b) in enumerate(zip(i,j)):
            incidence[a].append(k)
            incidence[b].append(k)
        h = min(scale//1024,max(1,int(abs(root).max())//256))
        drift = 8*wmax*h*(2*int(abs(root).max())+h)
        start = perf_counter()
        rows = []
        gains = []
        for t,win in enumerate(expected):
            def fetch(k):
                name = f"t{t}k{k}"
                needed.add(name)
                return enclosure(log[name],k)
            child,lower,upper,next_rad = fetch(win)
            threshold = lo-upper
            need(threshold>0, "positive exact decision")
            gains.append((threshold,hi-lower))
            hot = np.flatnonzero(abs(q-root)>h)
            cut = bisect_left(priority,-(threshold-rad-drift)+1)
            tail = int(-priority[cut])+rad+drift if cut<len(order) else None
            need(tail is None or tail<threshold,"cold exclusion")
            seen = set(order[:cut].tolist())
            for vertex in hot:
                seen.update(incidence[vertex])
            need(rad+4*wmax*int(abs(q).max())**2 < 2**62, "screen range")
            ids = np.array(sorted(seen.difference(chosen,[win])), dtype=int)
            a,b,weight = i[ids],j[ids],w[ids]
            delta = abs(q[a]-q[b])*weight
            # Triangle inequality for child residual; diagonal majorizes the
            # two-coordinate child block by its absolute off-diagonal.
            e = isqrt(B*rad)+1
            pa = np.maximum(0,np.where(flip[ids],2*weight*abs(q[b]),delta)-e)
            pb = np.maximum(0,np.where(flip[ids],2*weight*abs(q[a]),delta)-e)
            need(max(int(pa.max(initial=0)),int(pb.max(initial=0)))**2<2**62,"endpoint range")
            shift = np.where(flip[ids],weight,2*weight)
            caps = keys(q,ids)+rad-pa**2//(diag[a]+shift)-pb**2//(diag[b]+shift)
            rivals = ids[caps>=threshold]
            for rival in rivals:
                _,rl,_,_ = fetch(int(rival))
                need(rl>upper, "exact rival separation")
            rows.append(dict(winner=win,flip=bool(flip[win]),root_radius=initial[1]-initial[0],
                             radius=rad,h=h,hot=hot.tolist(),drift=drift,cut=int(cut),
                             tail=tail,threshold=threshold,delta=0,visited=len(seen),rivals=len(rivals)))
            if not flip[win]:
                diag[i[win]] += w[win]
                diag[j[win]] += w[win]
            chosen.append(win)
            q,lo,hi,rad = child,lower,upper,next_rad
        need(log["claims"].dtype==np.uint8 and log["claims"].tobytes()==
             json.dumps(rows,sort_keys=True,separators=(",",":")).encode(),"screen claims")
        need(len(log.files)==len(needed) and set(log.files)==needed,"witness names")
    return dict(epochs=rows, gains=gains, objective=[initial[0]-hi,initial[1]-lo],denominator=B*scale*scale,root_gain=root_gain,root_gains=root_gains,
                index_seconds=start-tic,replay_seconds=perf_counter()-start)




def precise_persistent_gains(graph, source, pool, plan):
    """Re-solve these unit-weight operators, verifying dyadic energy bounds exactly."""
    from scipy import sparse
    from scipy.sparse.linalg import cg
    from fractions import Fraction
    i,j,w,flip = pool
    need(np.all(w==1) and np.all(abs(graph.data)==1), "unit persistent weights")
    matrix = (sparse.diags(1+np.asarray(abs(graph).sum(axis=1)).ravel(),dtype=np.int64)-graph).astype(np.int64).tocsr()
    nums,den = dyadic(source)
    scale = 1<<48
    states = []
    for step in range(len(plan)+1):
        state,code = cg(matrix,source,rtol=1e-14,atol=0.)
        need(code==0 and np.all(np.isfinite(state)) and max(abs(state))<1, "persistent refinement")
        q = np.rint(scale*state).astype(np.int64)
        need(int(abs(q).max())*int(abs(matrix).sum(axis=1).max())<2**62,"refinement product range")
        product = matrix@q
        residual = [u*scale-den*int(v) for u,v in zip(nums,product)]
        energy = Fraction(sum(int(x)*(u*scale+r) for x,u,r in zip(q,nums,residual)),den*scale**2)
        radius = Fraction(sum(r*r for r in residual),(den*scale)**2)
        states.append((energy,energy+radius))
        if step<len(plan):
            k = plan[step]
            a,b = i[k],j[k]
            vals = (0,-2,-2,0) if flip[k] else (1,-1,-1,1)
            matrix = (matrix+sparse.coo_matrix((vals,([a,a,b,b],[a,b,a,b])),shape=matrix.shape)).tocsr()
    return [(a-d,b-c) for (a,b),(c,d) in zip(states,states[1:])]
