"""Fixed all-negative-edge interventions on the full Alpha operator."""
import csv
import json
from scipy.sparse import coo_matrix, triu
from scipy.sparse.linalg import spsolve
import numpy as np
from long_promotion import DATA, UNIT, SCREEN, OPINION_DEN, cap, search
from workloads import load_graph, system


def objective(A, s):
    """Enclose s^T A^-1 s on the UNIT^-2 grid using A >= I."""
    x = np.rint(UNIT*spsolve(A.astype(float).tocsc(), s/OPINION_DEN)).astype("int64")
    rhs = s*(UNIT//OPINION_DEN)
    residual = rhs - A@x
    low = sum(int(v)*int(w) for v,w in zip(x,rhs+residual))
    high = low+sum(int(v)**2 for v in residual)
    return low,high


def after(A,i,j):
    return A+coo_matrix(([-2,-2],([i,j],[j,i])),shape=A.shape).tocsr()


def run():
    W = load_graph("Alpha")
    A = system(W).astype("int64")
    edges = triu(W<0,k=1,format="csr").tocoo()
    pool = edges.row,edges.col,np.ones(edges.nnz,bool)
    ids = np.arange(edges.nnz)
    negative = (W<0).tocsr()
    degree = A.diagonal()
    centers = np.flatnonzero((np.diff(negative.indptr)==2)&(degree<7))[:5]
    found = []
    for a in centers:
        b,c = sorted(negative[a].indices,key=lambda v:(degree[v],v))
        z = np.random.default_rng(17).integers(-3,4,size=A.shape[0])
        z[[a,b,c]] = [round(1.5*512/degree[a]),-512,96]
        rhs = A@z
        s = np.rint(rhs*min(1,OPINION_DEN/max(abs(rhs)))).astype("int64")
        rows = search(W,s,pool,"full",budget=2)[0]
        found.append([tuple(sorted((a,b))),tuple(sorted((a,c)))] ==
                     [(r["i"],r["j"]) for r in rows])
        if len(found)==1:
            case = (int(a),int(b),int(c),s)
    a,b,c,s = case
    assert found[0]
    def idx(v):
        i,j = sorted((a,v))
        return int(np.flatnonzero((edges.row==i)&(edges.col==j))[0])
    z = spsolve(A.astype(float).tocsc(),s/OPINION_DEN)
    x = np.rint(z*UNIT).astype("int64")
    xc = np.rint(z*SCREEN).astype("int64")
    r = int(abs(A@x-s*(UNIT//OPINION_DEN)).max())
    d = UNIT//SCREEN
    rc = (int(abs(x-xc*d).max())+r+d-1)//d
    caps = cap(pool,ids,xc,rc,degree,False)
    rank = int(np.flatnonzero(np.argsort(-caps,kind="stable")==idx(c))[0])+1
    assert rank>64
    state = after(A,a,b)
    witness = []
    for parent in (A,state):
        low,high = objective(parent,s)
        child_low,child_high = objective(after(parent,a,c),s)
        witness.append([low-child_high,high-child_low])
    assert witness[0][1]<0<witness[1][0]
    records = []
    totals = {}
    paths = {}
    for mode in ("static","promoted","full"):
        rows,lo,hi,_ = search(W,s,pool,mode)
        assert len(rows)==15
        paths[mode] = [(r["i"],r["j"]) for r in rows]
        totals[mode] = lo,hi
        for r in rows:
            records.append(dict(method=mode,**{k:r[k] for k in
                                ("epoch","i","j","low","high","promoted","visited")}))
    assert paths["promoted"]==paths["full"]
    loss = [totals["full"][0]-totals["static"][1],totals["full"][1]-totals["static"][0]]
    ranks = np.argsort(np.argsort(-caps,kind="stable"))+1
    states = [A]
    cold_winners = []
    for epoch,(i,j) in enumerate(paths["full"],1):
        e = int(np.flatnonzero((edges.row==i)&(edges.col==j))[0])
        if ranks[e]>64 and epoch>1:
            intervals = []
            for parent in states[-2:]:
                lo,hi = objective(parent,s)
                child_lo,child_hi = objective(after(parent,i,j),s)
                intervals.append([lo-child_hi,hi-child_lo])
            cold_winners.append(dict(epoch=epoch,winner=[i,j],root_screen_rank=int(ranks[e]),
                                     preceding_action=paths["full"][epoch-2],
                                     prior_current_gain=intervals))
        states.append(after(states[-1],i,j))
    report = dict(cold_winners=cold_winners,centers=centers.tolist(),first_two_match=found,universe=edges.nnz,
                  first_id=idx(b),winner_id=idx(c),root_screen_rank=rank,
                  prior_current_gain=witness,static_loss=loss,denominator=UNIT**2)
    (DATA/"flip_promotion_case.json").write_text(json.dumps(report,indent=2)+"\n")
    with (DATA/"flip_promotion_case.csv").open("w",newline="") as stream:
        out = csv.DictWriter(stream,fieldnames=records[0])
        out.writeheader()
        out.writerows(records)


if __name__=="__main__":
    run()

