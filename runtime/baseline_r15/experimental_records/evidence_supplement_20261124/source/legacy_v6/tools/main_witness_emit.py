"""Generate independent energy proofs using explicit edited sparse matrices."""
import json
from math import isqrt
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
from threadpoolctl import threadpool_limits

from main_witness_input import DATA, instance

SCALE = 2**26


def produce(case):
    case = {"universe6":"universe5", "persistent6":"persistent5"}.get(case,case)
    graph,s,(i,j,weights,flip),budget = instance(case,emitting=True)
    scale = 2**33 if case in ("motif","clamp","universe5") else SCALE
    denominator = max(float(x).as_integer_ratio()[1] for x in np.r_[weights,graph.data])
    dweight = np.rint(weights*denominator).astype(np.int64)
    mat = (sparse.diags(1+np.asarray(abs(graph).sum(axis=1)).ravel(),dtype=float)-graph).tocsr()
    matrix = (denominator*mat).astype(np.int64)
    ratios = [float(x).as_integer_ratio() for x in s]
    sd = max(b for _,b in ratios)
    sn = [a*(sd//b) for a,b in ratios]

    def response(m):
        a = m.astype(float)
        inverse = sparse.diags(1/a.diagonal())
        z,code = cg(a,denominator*s,M=inverse,rtol=2e-14,atol=0.)
        if code:
            raise ArithmeticError(code)
        v = np.rint(z*scale).astype(np.int32)
        aq = m@v.astype(np.int64)
        linear = sum(int(code)*src for code,src in zip(v,sn))
        quad = sum(int(code)*int(mv) for code,mv in zip(v,aq))
        en = 2*denominator*scale*linear-quad*sd
        sq = sum((int(mv)*sd-denominator*scale*src)**2 for mv,src in zip(aq,sn))
        low = en//sd
        high = (en+sd-1)//sd+(sq+denominator*sd**2-1)//(denominator*sd**2)
        return v,low,high

    def priorities(v):
        a,b=v[i].astype(np.int64),v[j].astype(np.int64)
        return np.where(flip,-4*a*b,(a-b)**2)*dweight

    def revised(k,parent):
        a,b,weight = i[k],j[k],dweight[k]
        if flip[k]:
            delta = sparse.coo_matrix(([-2*weight]*2,([a,b],[b,a])),shape=matrix.shape)
        else:
            delta = sparse.coo_matrix(([weight,-weight,-weight,weight],([a,b,a,b],[a,a,b,b])),
                                      shape=matrix.shape)
        return (parent+delta).tocsr()

    plan, claims = [],[]
    q,lo,hi = response(matrix)
    log = dict(root=q)
    root_matrix = matrix
    root = q.astype(np.int64)
    root_keys = priorities(root)
    root_radius = hi-lo
    h=min(scale//1024,max(1,int(abs(root).max())//256))
    drift=8*int(dweight.max())*h*(2*int(abs(root).max())+h)
    for epoch in range(budget):
        rad=hi-lo
        a,b=q[i].astype(np.int64),q[j].astype(np.int64)
        drop=dweight*abs(a-b)
        err=isqrt(denominator*rad)+1
        left=np.maximum(0,np.where(flip,2*dweight*abs(b),drop)-err)
        right=np.maximum(0,np.where(flip,2*dweight*abs(a),drop)-err)
        diagonal=matrix.diagonal()
        slack=np.where(flip,dweight,2*dweight)
        cap=priorities(q)+rad-left**2//(diagonal[i]+slack)-right**2//(diagonal[j]+slack)
        order=np.argsort(-cap,kind="stable")
        best=0
        children={}
        for ix in order:
            k=int(ix)
            if cap[k]<best:
                break
            if k in plan:
                continue
            value=response(revised(k,matrix))
            children[k]=value
            gain=lo-value[2]
            if gain>best:
                win,best=k,gain
        if best<=0:
            raise ArithmeticError("budget cannot be exhausted")
        hot=abs(q.astype(np.int64)-root)>h
        prefix=root_keys+rad+drift>=best
        seen=prefix|hot[i]|hot[j]
        unresolved=set(np.flatnonzero(seen&(cap>=best)))-set(plan)-{win}
        for k in unresolved|{win}:
            log[f"t{epoch}k{k}"]=children[k][0]
        if any(children[k][1]<=children[win][2] for k in unresolved):
            raise ArithmeticError(f"energy intervals overlap at epoch {epoch+1}")
        suffix=root_keys[~prefix]
        claims.append(dict(winner=win,flip=bool(flip[win]),root_radius=root_radius,radius=rad,
                           h=h,hot=np.flatnonzero(hot).tolist(),drift=drift,cut=int(prefix.sum()),
                           tail=int(suffix.max())+rad+drift if len(suffix) else None,
                           threshold=best,delta=0,visited=int(seen.sum()),rivals=len(unresolved)))
        if case=="persistent5":
            log[f"root_k{win}"]=response(revised(win,root_matrix))[0]
        if epoch==1 and case in ("motif","clamp","clustered","universe5"):
            log["root_rival"]=response(revised(win,root_matrix))[0]
        matrix=revised(win,matrix)
        q,lo,hi=children[win]
        plan.append(win)
    log["claims"]=np.frombuffer(json.dumps(claims,sort_keys=True,separators=(",",":")).encode(),np.uint8)
    log_case = "clamp" if case=="universe5" else case
    np.savez_compressed(DATA/f"main_witness_{log_case}.npz",**log,plan=plan)
    print(case,plan,flush=True)


if __name__=="__main__":
    import sys
    with threadpool_limits(1):
        produce(sys.argv[1])
