"""Tiny common-domain forest-to-gain interface. No online timing or confidence claim."""
from itertools import product
import json
import numpy as np


def generalized_forests(W):
    """Enumerate a directed successor per vertex (self = root). Keep only rooted
    components or components with one negative directed cycle. The empirical
    probe samples these maps uniformly as proposals and applies a self-normalized
    importance ratio with factors 2**(number of negative cycles).
    """
    n=len(W)
    options=[[i,*np.flatnonzero(row)] for i,row in enumerate(W)]
    factors=[]
    matrices=[]
    for succ in product(*options):
        signs=np.zeros((n,n))
        negative=set()
        ok=True
        for start in range(n):
            order=[]
            u=start
            sign=1
            while succ[u]!=u and u not in order:
                order.append(u)
                sign*=int(W[u,succ[u]])
                u=succ[u]
            if succ[u]==u:
                signs[start,u]=sign
            else:
                cycle=order[order.index(u):]
                if np.prod([int(W[v,succ[v]]) for v in cycle])>=0:
                    ok=False
                    break
                negative.add(frozenset(cycle))
        if ok:
            factors.append(2**len(negative))
            matrices.append(signs)
    return np.array(factors),np.array(matrices)


def rows(W,Q,s):
    z=Q@s
    symmetric=(Q+Q.T)/2
    out=[]
    for i in range(len(W)):
        for j in range(i+1,len(W)):
            if W[i,j]>0: continue
            flip=W[i,j]<0
            U=np.zeros((len(W),1+int(flip)))
            U[i,0],U[j,0]=1,-1
            if flip: U[i,1]=U[j,1]=1
            h=U.T@z
            pivot=U.T@symmetric@U+np.diag([1.,-1.] if flip else [1.])
            sigma=float(np.linalg.svd(pivot,compute_uv=False)[-1])
            out.append(dict(i=i,j=j,flip=bool(flip),min_pivot_singular_value=sigma,
                            gain=None if sigma<1e-10 else float(h@np.linalg.solve(pivot,h))))
    return out


def probe():
    W=np.array([[0,-1,1,0],[-1,0,1,0],[1,1,0,1],[0,0,1,0]])
    s=np.array([.875,-.625,.125,-.5])
    t=np.array([-.625,.375,1.,-.25])
    rng=np.random.default_rng(104729)
    out=[]
    for name,M in (("signed_root",W),("committed_flip",abs(W))):
        weight,mats=generalized_forests(M)
        Q=np.linalg.inv(np.diag(1+abs(M).sum(axis=1))-M)
        whole=np.einsum("k,kij->ij",weight,mats)/weight.sum()
        np.testing.assert_allclose(whole,Q,atol=1e-14)
        # Common unsigned unit-insertion subdomain on the committed graph.
        for v in (s,t):
            for e in rows(M,Q,v):
                child=M.copy()
                child[e["i"],e["j"]]=child[e["j"],e["i"]]=1
                gain=v@Q@v-v@np.linalg.solve(np.diag(1+abs(child).sum(axis=1))-child,v)
                np.testing.assert_allclose(e["gain"],gain,atol=1e-14)
        ids=rng.integers(0,len(weight),size=1024)
        for count in (128,1024):
            w=weight[ids[:count]]
            Qhat=np.einsum("k,kij->ij",w,mats[ids[:count]])/w.sum()
            for opname,v in (("s",s),("t",t)):
                truth,est=rows(M,Q,v),rows(M,Qhat,v)
                errors=[abs(b["gain"]-a["gain"]) for a,b in zip(truth,est) if b["gain"] is not None]
                off=~np.eye(len(M),dtype=bool)
                out.append(dict(graph=name,opinions=opname,samples=count,valid_maps=len(weight),
                    importance_mean_ratio=float(w.mean()/weight.mean()),
                    max_off_diagonal_error=float(abs(Qhat[off]-Q[off]).max()),
                    max_z_error=float(abs((Qhat-Q)@v).max()),max_gain_error=max(errors),
                    smallest_sample_pivot=min(b["min_pivot_singular_value"] for b in est),
                    unresolved_pivots=sum(b["gain"] is None for b in est),
                    exact_gains=truth,estimated_gains=est))
    return out


if __name__=="__main__":
    print(json.dumps(probe(),indent=2))
