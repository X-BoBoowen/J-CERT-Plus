"""Exact unit-weight promotion experiment on enumerated two-hop/flip universes."""
import csv
import argparse
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path
from time import perf_counter

from sympy import Matrix

PAPER = Path(__file__).resolve().parents[1]
TAU = Q(1, 4096)


@dataclass
class State:
    epoch: int
    inverse: list
    z: list
    energy: Q


def state(A, opinions, epoch):
    H = [[Q(v) for v in row] for row in Matrix(A).inv(method="DM").tolist()]
    z = [sum(x*y for x, y in zip(row, opinions)) for row in H]
    return State(epoch, H, z, sum(x*y for x, y in zip(z, opinions)))


def edit(A, e):
    i, j, flip = e
    B = [row[:] for row in A]
    if not flip:
        B[i][i] += 1
        B[j][j] += 1
    B[i][j] -= 1 + flip
    B[j][i] -= 1 + flip
    return B


def upper(z, e):
    i, j, flip = e
    return -4*z[i]*z[j] if flip else (z[i]-z[j])**2


def gain(st, e):
    i, j, flip = e
    H, z = st.inverse, st.z
    b = 1 + H[i][i] + H[j][j] - 2*H[i][j]
    x = z[i] - z[j]
    if not flip:
        return x*x/b
    c = -1 + H[i][i] + H[j][j] + 2*H[i][j]
    d, y = H[i][i]-H[j][j], z[i]+z[j]
    return (c*x*x - 2*d*x*y + b*y*y)/(b*c-d*d)


def search(A, s, pool, mode, hot_size, budget):
    start = perf_counter()
    st = state(A, s, 0)
    initial = st.z
    cap = {e: upper(initial, e) for e in pool}
    order = sorted(pool, key=lambda e: (-cap[e], e))
    hot = set(order[:hot_size])
    cold = set(order[hot_size:])
    index = defaultdict(list)
    for e in cold:
        for i in e[:2]:
            index[i].append(e)
    chosen, records = [], []
    total = Q(0)
    cold_order = order[hot_size:]
    for t in range(budget):
        assert st.epoch == t
        winner, best, scores, inspected, promoted = None, Q(0), 0, 0, 0

        def compare(e):
            nonlocal winner, best, scores
            value = gain(st, e)
            scores += 1
            if value > best or (value == best and best > 0 and e < winner):
                winner, best = e, value

        for e in sorted((set(pool) if mode == "full" else hot).difference(chosen)):
            compare(e)
        if mode == "promoted":
            drift = [abs(a-b) for a, b in zip(st.z, initial)]
            visited = set()

            def examine(e):
                nonlocal inspected, promoted
                if e not in cold or e in visited:
                    return
                visited.add(e)
                inspected += 1
                i, j, _ = e
                if min(cap[e]+4*(drift[i]+drift[j]), upper(st.z, e)) >= best:
                    hot.add(e)
                    cold.remove(e)
                    promoted += 1
                    compare(e)

            for i, displacement in enumerate(drift):
                if displacement > TAU:
                    for e in index[i]:
                        examine(e)
            for e in cold_order:
                if cap[e]+8*TAU < best:
                    break
                examine(e)
        gain_record = str(best)
        if winner is not None:
            before = st.energy
            A = edit(A, winner)
            st = state(A, s, t+1)
            assert before-st.energy == best > 0
            total += best
            chosen.append(winner)
        records.append((t, winner, gain_record, promoted, inspected, scores))
        if winner is None:
            break
    return chosen, total, records, perf_counter()-start


def graph_case(name, n=18):
    from workloads import load_graph, opinions
    W = load_graph(name, n=n).sign().tocsr()
    n = W.shape[0]
    A = [[int(i == j) for j in range(n)] for i in range(n)]
    neighbors = []
    pool = []
    for i in range(n):
        neighbors.append(set(map(int, W[i:i+1].indices)))
        A[i][i] += len(neighbors[i])
        for j in neighbors[i]:
            A[i][j] = -int(W[i, j])
            if i < j and W[i, j] < 0:
                pool.append((i, j, True))
    for i in range(n):
        two = set().union(*(neighbors[j] for j in neighbors[i]))
        pool.extend((i, j, False) for j in two-neighbors[i] if i < j)
    s = [Q(round(float(x)*16), 16) for x in opinions(W, "S1", seed=17)]
    return A, s, sorted(pool)


def selftest():
    A = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
    s = [Q(-1), Q(1, 4), Q(1, 4)]
    pool = [(0, 1, True), (1, 2, True)]
    assert gain(state(A, s, 0), pool[1]) == Q(-3, 16)
    for mode in ("promoted", "full"):
        trace, _, rows, _ = search(A, s, pool, mode, 1, 2)
        assert trace == pool and rows[1][2] == "1/16"
    assert search(A, s, pool, "static", 1, 2)[0] == pool[:1]
    # Identical separate pairs test ties, an empty hot set, and zero stop.
    ident = [[int(i == j) for j in range(4)] for i in range(4)]
    pairs = [(0, 1, False), (2, 3, False)]
    assert search(ident, [1,-1,1,-1], pairs, "promoted", 0, 2)[0] == pairs
    assert not search(ident, [0]*4, pairs, "promoted", 0, 2)[0]
    st = state(A, s, 0)
    for e in pool + [(0,2,False)]:
        assert gain(st,e) == st.energy-state(edit(A,e), s,1).energy
    # Check rank-two interval transport against an independently refactored
    # rational inverse, including reuse by both a flip and an addition.
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import splu
    catalog = pool + [(0,2,False)]
    indices = tuple(np.array(col) for col in zip(*catalog))
    scale = 1<<24
    view = VerifiedGramViews(indices,scale)
    mat = csr_matrix(A)
    fac = splu(mat.astype(float).tocsc())
    rhs = np.array(s,dtype=float)
    z = fac.solve(rhs)
    def check_scores(epoch,live,exact,operator,factor):
        x = np.rint(factor.solve(rhs)*scale).astype("int64")
        r = int(abs(operator@x-np.rint(rhs*scale).astype("int64")).max())
        for e in live:
            lo,hi = view.score(e,epoch,operator,factor,x,r)
            assert lo <= gain(exact,catalog[e])*scale**2 <= hi
    check_scores(0,[0,1,2],st,mat,fac)
    view.advance(0,0,mat,fac,z)
    B = edit(A,catalog[0])
    mat = csr_matrix(B)
    fac = splu(mat.astype(float).tocsc())
    check_scores(1,[1,2],state(B,s,1),mat,fac)
    assert (view.fresh,view.hits,view.winner_blocks,view.transfers)==(3,2,1,2)
    # These disjoint flips tie through a component exchange, which is not a
    # single endpoint transposition. The exact fallback must select pair 0.
    W = csr_matrix([[0,-1,0,0],[-1,0,0,0],[0,0,0,-1],[0,0,-1,0]])
    pair_ids = (np.array([0,2]),np.array([1,3]),np.array([True,True]))
    op = np.array([1,-1,1,-1])*OPINION_DEN
    for use_all in (False,True):
        seq,_,_,_ = large_search(W,op,pair_ids,"promoted",budget=1,hot_size=2,all_pairs=use_all)
        assert seq == [0]
    seq,_,_,_ = large_search(W,np.zeros(4,dtype="int64"),pair_ids,"full",budget=1)
    assert seq==[]


def main():
    from workloads import NAMES
    selftest()
    data = PAPER/"data"
    with (data/"promotion_runs.csv").open("w", newline="") as rf, (data/"promotion_rounds.csv").open("w", newline="") as tf:
        runs, rounds = csv.writer(rf), csv.writer(tf)
        runs.writerow("graph n universe hot budget repetition method seconds objective_exact".split())
        rounds.writerow("graph repetition method epoch winner gain_exact promoted checked_cold exact_scores".split())
        for name in NAMES:
            A, s, pool = graph_case(name)
            k, b = min(8,len(pool)), 5
            methods = ["static", "promoted", "full"]
            for rep in range(10):
                paths = {}
                for mode in methods[rep%3:]+methods[:rep%3]:
                    seq, total, rows, elapsed = search(A, s, pool, mode, k, b)
                    paths[mode] = seq
                    runs.writerow([name,len(A),len(pool),k,b,rep,mode,elapsed,total])
                    rounds.writerows((name,rep,mode,*r) for r in rows)
                assert paths["promoted"] == paths["full"], name


# Integer units for x are 1/SCALE; energy and all compared bounds use
# 1/SCALE**2. Floating solves propose states; integer arithmetic verifies them.
SCALE = 1 << 24
OPINION_DEN = 1 << 16


def large_pool(W):
    import numpy as np
    from scipy.sparse import triu
    structure = W.astype(bool)
    two = (structure @ structure).astype("int32")
    two -= two.multiply(structure)
    two.eliminate_zeros()
    additions = triu(two, k=1, format="coo")
    flips = triu(W < 0, k=1, format="coo")
    i = np.r_[additions.row, flips.row].astype("int64")
    j = np.r_[additions.col, flips.col].astype("int64")
    f = np.r_[np.zeros(additions.nnz, bool), np.ones(flips.nnz, bool)]
    order = np.lexsort((j,i))
    return i[order], j[order], f[order]


# Closed rational intervals for the small endpoint matrices. Products and
# inverses have dimension at most two. Every transported bound is outward.
def add_iv(a,b):
    return a[0]+b[0],a[1]+b[1]


def neg_iv(a):
    return -a[1],-a[0]


def mul_iv(a,b):
    terms = [x*y for x in a for y in b]
    return min(terms),max(terms)


def inv_iv(a):
    if a[0]<=0<=a[1]:
        raise ArithmeticError("singular interval")
    return 1/a[1],1/a[0]


def transpose(a):
    return list(zip(*a))


def matmul_iv(a,b):
    def dot(x,y):
        low = high = 0
        for xi,yi in zip(x,y):
            lo,hi = mul_iv(xi,yi)
            low += lo
            high += hi
        return low,high
    return [[dot(x,y) for y in transpose(b)] for x in a]


def small_inverse(g):
    a = add_iv(g[0][0],(1,1))
    if len(g)==1:
        return [[inv_iv(a)]]
    b,c,d = g[0][1],g[1][0],add_iv(g[1][1],(-1,-1))
    determinant = add_iv(mul_iv(a,d),neg_iv(mul_iv(b,c)))
    scale = inv_iv(determinant)
    return [[mul_iv(el,scale) for el in row] for row in [[d,neg_iv(b)],[neg_iv(c),a]]]


class VerifiedGramViews:
    def __init__(self, pool, scale):
        self.pool = pool
        self.scale = scale
        self.directional_block = None
        self.entries = {}
        self.fresh = self.hits = self.winner_blocks = self.transfers = 0

    def project(self,e,columns,r):
        i,j,flip = self.pool
        p,q = columns[i[e]],columns[j[e]]
        rows = [p-q,p+q] if flip[e] else [p-q]
        def interval(value):
            return Q(int(value)-2*r,self.scale), Q(int(value)+2*r,self.scale)
        return [[interval(value) for value in row] for row in rows]

    def response(self,e,A,fac):
        import numpy as np
        i,j,flip = self.pool
        u,v = i[e],j[e]
        U = np.zeros((A.shape[0],2 if flip[e] else 1),dtype="int64")
        U[u,0],U[v,0] = 1,-1
        if flip[e]:
            U[u,1] = U[v,1] = 1
        Y = fac.solve(U.astype(float))
        X = np.rint(Y*self.scale).astype("int64")
        assert (2*A.shape[0]+1)*int(abs(X).max())+self.scale < 2**63
        r = int(abs(A@X-self.scale*U).max())
        return U,Y,X,r

    def score(self,e,t,A,fac,x,r):
        if e in self.entries:
            version,g = self.entries[e]
            assert version==t
            self.hits += 1
        else:
            _,_,X,rr = self.response(e,A,fac)
            self.fresh += 1
            g = self.project(e,X,rr)
            self.entries[e] = (t,g)
        h = self.project(e,x[:,None],r)
        if self.directional_block is not None:
            lo,hi = self.directional_block(g,transpose(h),self.scale)[0][0]
        else:
            try:
                inv = small_inverse(g)
            except ArithmeticError:
                # A loose transported view can be freshly anchored.
                _,_,X,rr = self.response(e,A,fac)
                self.fresh += 1
                g = self.project(e,X,rr)
                self.entries[e] = (t,g)
                inv = small_inverse(g)
            lo,hi = matmul_iv(matmul_iv(transpose(h),inv),h)[0][0]
        from math import floor,ceil
        return floor(lo*self.scale**2),ceil(hi*self.scale**2)

    def advance(self,e,t,A,fac,z):
        import numpy as np
        i,j,flip = self.pool
        u,v = i[e],j[e]
        U,Y,X,r = self.response(e,A,fac)
        self.winner_blocks += 1
        winner_gram = self.project(e,X,r)
        inv = small_inverse(winner_gram) if self.directional_block is None else None
        self.entries.pop(e,None)
        for candidate,(version,g) in self.entries.items():
            assert version==t
            c = self.project(candidate,X,r)
            if self.directional_block is None:
                correction = matmul_iv(matmul_iv(c,inv),transpose(c))
            else:
                correction = self.directional_block(winner_gram,c,self.scale)
            new_g = [[add_iv(v,neg_iv(w)) for v,w in zip(a,b)] for a,b in zip(g,correction)]
            self.entries[candidate] = (t+1,new_g)
            self.transfers += 1
        J = np.diag([1.,-1.] if flip[e] else [1.])
        step = np.linalg.solve(J+U[[u,v]].T@Y[[u,v]],U[[u,v]].T@z[[u,v]])
        return z-Y@step


class UnresolvedComparison(RuntimeError):
    """The supplied comparison precision did not certify an action or stop."""


def resolve_comparison(A,s,pool,evaluated,winner,best,scale):
    from math import ceil,floor
    overlapping = [e for e,hi in evaluated.items() if e!=winner and hi>=best]
    if winner is not None:
        if not overlapping:
            return winner,best
        if all(exchange_tie(A,s,pool,winner,e) for e in overlapping):
            return min([winner,*overlapping]),best
    elif all(v<=0 for v in evaluated.values()):
        return None,0
    if A.shape[0]>64:
        raise UnresolvedComparison("Overlapping gain intervals after screening; exact fallback size exceeds 64.")
    # A current full-state rational refactor resolves overlaps, including
    # ties involving disjoint actions with no single endpoint transposition.
    exact = state(A.toarray().tolist(),[Q(float(x)) for x in s],0)
    ids = sorted(overlapping+([winner] if winner is not None else []))
    value = Q(0)
    selected = None
    i,j,f = pool
    for e in ids:
        actual = gain(exact,(i[e],j[e],f[e]))
        if actual>value:
            selected,value = e,actual
    if selected is not None:
        evaluated[selected] = ceil(value*scale**2)
    return selected,floor(value*scale**2)


def exchange_tie(A,s,pool,a,b):
    """Prove equal gains by an operator/opinion-preserving transposition."""
    import numpy as np
    i,j,flip = pool
    if flip[a] != flip[b]:
        return False
    changed = sorted({int(i[a]),int(j[a])}^{int(i[b]),int(j[b])})
    if len(changed)!=2:
        return False
    u,v = changed
    if s[u]!=s[v] or A[u,u]!=A[v,v]:
        return False
    row = (A.getrow(u)-A.getrow(v)).tocsr()
    return all(k in (u,v) for k in row.indices)


def large_search(W, opinion_units, pool, mode, budget=3, hot_size=32, all_pairs=False):
    import numpy as np
    from scipy.sparse import diags, coo_matrix, csr_matrix
    from scipy.sparse.linalg import splu
    start = perf_counter()
    i, j, flip = pool
    n, m = W.shape[0], len(i)
    ids = np.arange(m, dtype="int32")
    A = (diags(1+np.asarray(abs(W).sum(axis=1)).ravel(), dtype=np.int64)-W).astype("int64").tocsr()
    active = np.ones(m, bool)
    on_view = np.zeros(m, bool)
    live_hot = set()
    s = opinion_units/OPINION_DEN
    trace, rows = [], []
    unit = (1 << 28) if all_pairs else SCALE
    tau = unit//4096
    views = VerifiedGramViews(pool,unit) if all_pairs and mode=="promoted" else None

    def enclosure(A, z, action=None):
        x = np.rint(z*unit).astype("int64")
        assert np.max(np.abs(x)) <= 2*unit
        # Each row absolute sum is at most 2n+1. The following guards bound
        # the two dot products and sparse integer matvec before evaluation.
        assert (2*n+1)*2*unit < 2**63 and n*OPINION_DEN*2*unit < 2**63
        r = A@x-opinion_units*(unit//OPINION_DEN)
        if action is not None:
            u, v = i[action], j[action]
            if flip[action]:
                r[u] -= 2*x[v]
                r[v] -= 2*x[u]
            else:
                d = x[u]-x[v]
                r[u] += d
                r[v] -= d
        radius = int(np.max(np.abs(r)))
        assert n*max(radius,2*unit)*radius < 2**63
        lower = int(opinion_units@x)*(unit//OPINION_DEN)-int(x@r)
        return x, radius, lower, lower+int(r@r)

    def current_cap(ch, x, r):
        u, v = x[i[ch]], x[j[ch]]
        # Unit inputs, guard all quadratic intermediate integer magnitudes.
        assert r < unit//2 and 30*unit**2 < 2**63
        base_cap = np.where(flip[ch], -4*u*v+4*r*(4*unit+r),
                        (np.abs(u-v)+2*r)**2)
        return base_cap

    initial_energy = None
    for t in range(budget):
        diag = A.diagonal()
        fac = splu(A.astype(float).tocsc())
        z = fac.solve(s)
        x, radius, phi_lo, phi_hi = enclosure(A,z)
        if t == 0:
            x0, r0 = x.copy(), radius
            initial_energy = final_energy = (phi_lo, phi_hi)
        if t == 0 and mode != "full":
            cap0 = current_cap(ids,x0,r0)
            order = np.argsort(-cap0, kind="stable").astype("int32")
            on_view[order[:hot_size]] = True
            live_hot.update(map(int, order[:hot_size]))
            cold_order = order[hot_size:]
            neg_keys = -cap0[cold_order]-8*unit*tau
            if mode == "promoted":
                incidence = csr_matrix((np.r_[ids+1,ids+1],
                    (np.r_[i,j], np.r_[ids,ids])), shape=(n,m))
        best, winner, runner_up = 0, None, 0
        evaluated = {}
        witnesses = {}
        promoted, inspected = 0, 0

        def score(e):
            if views is not None:
                return views.score(e,t,A,fac,x,radius)
            u, v = int(i[e]), int(j[e])
            U = np.zeros((n,2 if flip[e] else 1))
            U[u,0], U[v,0] = 1,-1
            if flip[e]:
                U[u,1] = U[v,1] = 1
            Y = fac.solve(U)
            J = np.diag([1.,-1.] if flip[e] else [1.])
            step = np.linalg.solve(J+U[[u,v]].T@Y[[u,v]], U[[u,v]].T@z[[u,v]])
            zz = z-Y@step
            _, _, lo, hi = enclosure(A,zz,e)
            witnesses[e] = (lo,hi)
            return phi_lo-hi, phi_hi-lo

        def visit(ch, cold=False):
            nonlocal best, winner, promoted, inspected
            if not len(ch):
                return
            bounds = current_cap(ch,x,radius)
            if all_pairs:
                # Addition cap (10), with Aij=0. Split quotient to avoid overflow.
                left, right = diag[i[ch]], diag[j[ch]]
                prod = left*right
                den = prod+left+right
                assert int(den.max())**2 < 2**63
                scaled = (bounds//den)*prod+((bounds%den)*prod+den-1)//den
                bounds = np.where(flip[ch],bounds,scaled)
            if cold:
                inspected += len(ch)
                bounds = np.minimum(bounds, cap0[ch]+4*unit*(drift[i[ch]]+drift[j[ch]]))
            for k in np.argsort(-bounds,kind="stable"):
                if bounds[k] < best and not (views is not None and not cold):
                    break
                e = int(ch[k])
                if cold:
                    promoted += 1
                    on_view[e] = True
                    live_hot.add(e)
                lo, hi = score(e)
                evaluated[e] = hi
                if lo > best or (lo == best and best > 0 and e < winner):
                    winner, best = e, lo

        hot_queries = ids[active] if mode == "full" else np.asarray(sorted(live_hot),dtype="int32")
        visit(hot_queries)
        if mode == "promoted":
            drift = abs(x-x0)+radius+r0
            moving = np.flatnonzero(drift>tau)
            # CSR stores one row per endpoint, containing the incident IDs.
            incident = incidence[moving].data-1 if len(moving) else np.array([], dtype=int)
            limit = np.searchsorted(neg_keys, -best, side="right")
            ch = np.union1d(incident,cold_order[:limit])
            visit(ch[active[ch] & ~on_view[ch]],cold=True)
        winner,best = resolve_comparison(A,s,pool,evaluated,winner,best,unit)
        if winner is None:
            break
        if views is not None:
            zz = views.advance(winner,t,A,fac,z)
            _,_,lo,hi = enclosure(A,zz,winner)
            if phi_lo-hi<=0:
                raise UnresolvedComparison("Winner energy check does not separate from zero.")
            assert phi_lo-hi<=evaluated[winner] and phi_hi-lo>=best
            final_energy = (lo,hi)
        else:
            final_energy = witnesses[winner]
        rows.append((t,(int(i[winner]),int(j[winner]),bool(flip[winner])),
                     best, evaluated[winner], promoted,inspected,len(evaluated)))
        trace.append(winner)
        if all_pairs:
            ledger = (views.fresh,views.hits,views.winner_blocks,views.transfers) if views else (0,0,0,0)
            rows[-1] += ledger
        active[winner] = False
        live_hot.discard(winner)
        u,v = i[winner],j[winner]
        w = (0,-2,-2,0) if flip[winner] else (1,-1,-1,1)
        A += coo_matrix((w,([u,u,v,v],[u,v,u,v])),shape=A.shape).tocsr()
    total = (initial_energy[0]-final_energy[1],initial_energy[1]-final_energy[0])
    return trace,total,rows,perf_counter()-start


def large_experiment():
    import numpy as np
    from workloads import load_graph, opinions
    name = "Alpha"
    W = load_graph(name).sign().tocsr()
    pool = large_pool(W)
    s = np.rint(OPINION_DEN*opinions(W,"S1",seed=17)).astype("int64")
    methods = ["static","promoted","full"]
    data = PAPER/"data"
    with (data/"promotion_large_runs.csv").open("w",newline="") as rf, (data/"promotion_large_rounds.csv").open("w",newline="") as tf:
        runs, rounds = csv.writer(rf),csv.writer(tf)
        runs.writerow("graph n universe hot budget repetition method seconds numerator_low numerator_high denominator".split())
        rounds.writerow("graph repetition method epoch winner numerator_low numerator_high promoted checked_cold refinements".split())
        for rep in range(10):
            paths = {}
            for mode in methods[rep%3:]+methods[:rep%3]:
                path,total,rows,elapsed = large_search(W,s,pool,mode)
                paths[mode] = path
                runs.writerow([name,W.shape[0],len(pool[0]),32,3,rep,mode,elapsed,*total,SCALE**2])
                rounds.writerows((name,rep,mode,*row) for row in rows)
                rf.flush()
                tf.flush()
            assert paths["full"] == paths["promoted"]


def complete_pool(W):
    import numpy as np
    n = W.shape[0]
    allowed = np.triu(np.ones((n,n),dtype=bool), k=1)
    positive = (W>0).tocoo()
    allowed[positive.row,positive.col] = False
    i,j = np.nonzero(allowed)
    negative = np.zeros((n,n),dtype=bool)
    negatives = (W<0).tocoo()
    negative[negatives.row,negatives.col] = True
    return i.astype("int32"),j.astype("int32"),negative[i,j]


def brute_search(W,s,pool,budget=3):
    """Independent FP64 all-gain comparison; verify its IDs against certified search."""
    import numpy as np
    from scipy.sparse.linalg import splu
    from workloads import system
    start = perf_counter()
    A = system(W)
    n = A.shape[0]
    H = np.empty((n,n))
    factor = splu(A.tocsc())
    for first in range(0,n,32):
        col = np.arange(first,min(first+32,n))
        rhs = np.zeros((n,len(col)))
        rhs[col,np.arange(len(col))] = 1.
        H[:,col] = factor.solve(rhs)
    i,j,flip = pool
    trace,rows = [],[]
    total = 0.
    for t in range(budget):
        z = H@s
        diag = H.diagonal()
        best, winner = 0,None
        for start_id in range(0,len(i),65536):
            stop_id = min(start_id+65536,len(i))
            sl = slice(start_id,stop_id)
            u,v = i[sl],j[sl]
            a,d,cross = diag[u],diag[v],H[u,v]
            x,y = z[u]-z[v],z[u]+z[v]
            b = 1+a+d-2*cross
            gains = x*x/b
            f = flip[sl]
            if f.any():
                c = (-1+a+d+2*cross)[f]
                h = (a-d)[f]
                gains[f] = (c*x[f]**2-2*h*x[f]*y[f]+b[f]*y[f]**2)/(b[f]*c-h*h)
            for chosen in trace:
                if start_id<=chosen<stop_id:
                    gains[chosen-start_id] = -np.inf
            k = int(np.argmax(gains))
            # Distinguish true symmetry from close rounded scores.
            near = np.flatnonzero(gains >= gains[k]-1e-13)
            for other in near:
                if other<k and exchange_tie(A,s,pool,start_id+k,start_id+int(other)):
                    k = int(other)
            if winner is not None and abs(gains[k]-best)<1e-13 and exchange_tie(A,s,pool,winner,start_id+k):
                continue
            if gains[k]>best:
                best,winner = float(gains[k]),start_id+k
        assert winner is not None
        u,v,f = int(i[winner]),int(j[winner]),bool(flip[winner])
        rows.append((t,(u,v,f),best,len(i)-t))
        trace.append(winner)
        total += best
        from scipy.sparse import coo_matrix
        w = (0,-2,-2,0) if f else (1,-1,-1,1)
        A = (A + coo_matrix((w,([u,u,v,v],[u,v,u,v])),shape=A.shape)).tocsr()
        Y = H[:,u]-H[:,v]
        if f:
            Y = np.column_stack([Y,H[:,u]+H[:,v]])
            G = np.vstack([Y[u]-Y[v],Y[u]+Y[v]])
            H -= (Y@np.linalg.inv(np.diag([1.,-1.])+G))@Y.T
        else:
            H -= np.outer(Y,Y)/(1+Y[u]-Y[v])
    return trace,total,rows,perf_counter()-start


def all_pairs_experiment():
    import numpy as np
    from workloads import load_graph,opinions
    W = load_graph("Alpha").sign().tocsr()
    pool = complete_pool(W)
    s = np.rint(OPINION_DEN*opinions(W,"S1",seed=17)).astype("int64")
    methods = ["static","promoted","brute"]
    data = PAPER/"data"
    with (data/"promotion_all_runs.csv").open("w",newline="") as rf, (data/"promotion_all_rounds.csv").open("w",newline="") as tf:
        runs,rounds = csv.writer(rf),csv.writer(tf)
        runs.writerow("graph n universe hot budget repetition method seconds objective_low objective_high".split())
        rounds.writerow("repetition method epoch winner gain_low gain_high promoted checked_cold score_evaluations cumulative_view_blocks cumulative_cache_hits cumulative_winner_blocks cumulative_transfers".split())
        for rep in range(10):
            paths = {}
            for mode in methods[rep%3:]+methods[:rep%3]:
                if mode == "brute":
                    path,total,rows,elapsed = brute_search(W,s/OPINION_DEN,pool)
                    vals = (total,total)
                    rounds.writerows((rep,mode,t,e,g,g,0,0,count,0,0,0,0) for t,e,g,count in rows)
                else:
                    path,total,rows,elapsed = large_search(W,s,pool,mode,all_pairs=True)
                    vals = [v/(1 << 56) for v in total]
                    rounds.writerows((rep,mode,t,e,lo/(1 << 56),hi/(1 << 56),p,c,q,*ledger) for t,e,lo,hi,p,c,q,*ledger in rows)
                paths[mode] = path
                runs.writerow(["Alpha",W.shape[0],len(pool[0]),32,3,rep,mode,elapsed,*vals])
                rf.flush()
                tf.flush()
            assert paths["promoted"] == paths["brute"]


def timing_summary():
    """Timing intervals for each fixed input and method, resampling repetitions."""
    import numpy as np
    folder = PAPER / "data"
    rng = np.random.default_rng(108)
    with (folder / "promotion_summary.csv").open("w", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(["graph","n","universe","method","repetitions","q25_ms","median_ms",
                      "q75_ms","ci_low_ms","ci_high_ms"])
        for name in ("promotion_runs", "promotion_large_runs", "promotion_all_runs"):
            groups = defaultdict(list)
            with (folder / (name + ".csv")).open() as raw:
                for row in csv.DictReader(raw):
                    key = tuple(row[k] for k in ("graph", "n", "universe", "method"))
                    groups[key].append(1000 * float(row["seconds"]))
            for key, times in sorted(groups.items()):
                if len(times) != 10:
                    raise ValueError(f"Incomplete timing series {key}")
                draws = rng.choice(times, (10000, len(times)))
                ci = np.quantile(np.median(draws, axis=1), [.025, .975])
                out.writerow([*key, len(times), *np.quantile(times, [.25, .5, .75]), *ci])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--summary", action="store_true")
    options = parser.parse_args()
    if options.summary:
        timing_summary()
    elif options.all_pairs:
        all_pairs_experiment()
    elif options.large:
        large_experiment()
    else:
        main()













