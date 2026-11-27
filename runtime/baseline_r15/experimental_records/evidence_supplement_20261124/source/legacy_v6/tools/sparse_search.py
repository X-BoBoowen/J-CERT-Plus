"""Direct full sparse signed search on conditioned Bitcoin fixtures."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import csv
import copy
from fractions import Fraction
import json
from collections import Counter
from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from cholmod_backend import CholmodFactor
from engine import Edit
from sequencing import DATA, TOL
from workloads import load_graph, system
from fast_policy_kernel import FastPolicyKernel


class SearchKernel(FastPolicyKernel):
    """A DFS stack shares one sparse operator and restores it alongside the factor."""
    def prepare_stack(self):
        matrix = self.A.tocoo()
        additions = ~self.flip
        i,j = self.i[additions],self.j[additions]
        # Retain explicit zeros at both orientations of every possible insertion.
        self.A = sparse.csr_matrix(
            (np.r_[matrix.data,np.zeros(2*len(i))],
             (np.r_[matrix.row,i,j],np.r_[matrix.col,j,i])),shape=self.A.shape)
        self.A.sort_indices()
        self._refresh()
        slots = []
        for e in self.edits:
            positions = []
            for a,b in ((e.i,e.i),(e.i,e.j),(e.j,e.i),(e.j,e.j)):
                start,end = self.A.indptr[a:a+2]
                pos = start + np.searchsorted(self.A.indices[start:end],b)
                assert pos<end and self.A.indices[pos]==b
                positions.append(pos)
            slots.append(positions)
        self.slots = np.array(slots)
        d = self.w*(~self.flip)
        off = -self.w*(1+self.flip.astype(int))
        self.deltas = np.column_stack((d,off,off,d))

    def change_entries(self,e,direction):
        # A's grid check guarantees exact entrywise addition and reversal.
        slots = self.slots[e]
        values = self.A.data[slots] + direction*self.deltas[e]
        self.A.data[slots] = values
        self.absA.data[slots] = abs(values)
        for vertex,value in ((self.i[e],values[0]),(self.j[e],values[3])):
            self.diag[vertex] = value
            self.invdiag[vertex] = 1/value

    def _apply_sparse_edit(self,e):
        self.change_entries(e,1)


def fixture(W, case):
    A = system(W)
    z = np.zeros(A.shape[0])
    z[case["q"]] = case["z"]
    s = A @ z
    edits = [Edit(*e) for e in case["edits"]]
    for e in edits:
        assert W[e.i, e.j] == (-e.w if e.flip else 0)
    assert np.max(abs(s)) <= 1 + 1e-12
    return A, s, edits


def independent(A, s, edits):
    """Refactor the assembled operator; bound energy error using A >= I."""
    deltas = []
    for e in edits:
        i, j = e.i, e.j
        d = 0.0 if e.flip else e.w
        off = -e.w * (2 if e.flip else 1)
        deltas.append(sparse.csr_matrix(([d,off,off,d],([i,i,j,j],[i,j,i,j])),shape=A.shape))
    @lru_cache(None)
    def energy(mask):
        M = A
        for j, delta in enumerate(deltas):
            if mask >> j & 1:
                M = M + delta
        with CholmodFactor(M) as f:
            x = f.solve(s)
        eps = np.finfo(float).eps
        roundoff = (np.diff(M.indptr) + 3) * eps * (abs(s) + abs(M) @ abs(x))
        residual = np.linalg.norm(abs(s - M @ x) + 2 * roundoff)
        radius = 2 * (np.linalg.norm(s) * residual + len(s) * eps * (abs(s) @ abs(x)))
        return float(s @ x), float(radius)
    return energy


class PreparedSearch:
    """One catalog's validated, read-only root and pristine numeric factor."""
    def __init__(self,W,case):
        tick = perf_counter()
        self.case = case
        self.A,self.s,self.edits = fixture(W,case)
        self.prototype = SearchKernel(self.A,self.edits,backend="cg",winner_slots=len(self.edits))
        self.prototype.prepare_stack()
        for matrix in (self.prototype.A,self.prototype.absA):
            for array in (matrix.data,matrix.indices,matrix.indptr):
                array.flags.writeable = False
        self.prototype.backend = "cholmod"
        self.structure_s = perf_counter()-tick
        tick = perf_counter()
        self.prototype.factor = CholmodFactor(self.A)
        self.factor_s = perf_counter()-tick

    def consumer(self,reuse):
        tick = perf_counter()
        root = copy.copy(self.prototype)
        for field in ("A","absA"):
            matrix = copy.copy(getattr(root,field))
            matrix.data = matrix.data.copy()
            setattr(root,field,matrix)
        root.diag = root.diag.copy()
        root.invdiag = root.invdiag.copy()
        root.active = root.active.copy()
        root.seen = root.seen.copy()
        root.rows,root.stale = {},set()
        root.clear_winner_columns()
        root.cached = True if reuse else "same"
        root.ledger = Counter()
        state_s = perf_counter()-tick
        tick = perf_counter()
        root.factor = self.prototype.factor.clone()
        return root,state_s,perf_counter()-tick

    def close(self):
        self.prototype.close()


def query(prepared,root,mode,reuse,audit=None):
    import math

    tick = perf_counter()
    A,s,edits,case = prepared.A,prepared.s,prepared.edits,prepared.case
    root.rows.clear()
    root.stale.clear()
    root.active.fill(False)
    root.seen.fill(False)
    root.clear_winner_columns()
    root.ledger.clear()
    k = len(edits)
    budget = case.get("budget", k)
    counts = Counter()
    seen = set()
    counts["min_margin"] = float("inf")
    counts["min_rank_slack"] = float("inf")
    # The operator, catalog, per-row residual factors and edit slots are shared
    # across stack nodes. Charge these once and track the node-specific payloads.
    shared_bytes = sum(array.nbytes for array in (
        root.i,root.j,root.w,root.flip,root.off,root.diag,root.invdiag,root.gamma,
        root.A.data,root.A.indices,root.A.indptr,root.absA.data,root.absA.indices,root.absA.indptr))
    input_bytes = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes + s.nbytes
                   + root.slots.nbytes + root.deltas.nbytes + shared_bytes)

    def sample(node,parents):
        local = node.memory_bytes() - node.factor.bytes - shared_bytes
        counts["peak_bytes"] = max(counts["peak_bytes"],input_bytes+parents+local+node.factor.peak_bytes)
        return local

    def rank(options,limit):
        # The caller supplies rigorous intervals of a single objective. A
        # selected lower bound is within TOL of every remaining upper bound.
        available = options.copy()
        ranked = []
        for _ in range(min(limit,len(available))):
            best_upper = max(row[2] for row in available)
            permitted = [row for row in available if math.nextafter(best_upper-row[1],math.inf) <= TOL]
            if not permitted:
                raise ArithmeticError("uncertified ordering")
            opt = min(permitted,key=lambda row:row[0])
            counts["min_rank_slack"] = min(counts["min_rank_slack"],TOL-(best_upper-opt[1]))
            counts["rank_checks"] += 1
            available.remove(opt)
            ranked.append(opt)
        return ranked

    def expand(node,mask,parents):
        seen.add(mask)
        counts["expansions"] += 1
        z,rho = node.solve(s)
        counts["state_rhs"] += 1
        options = []
        for j in range(k):
            if mask >> j & 1:
                continue
            captured = []
            lo,hi = node._interval(j,z,rho,capture=captured)
            if not (lo>TOL or hi<=TOL):
                lo,hi = node._interval(j,z,rho,refresh=True,capture=captured)
            if not (lo>TOL or hi<=TOL):
                raise ArithmeticError("uncertified safety classification")
            margin = lo-TOL if lo>TOL else TOL-hi
            counts["min_margin"] = min(counts["min_margin"],margin)
            if audit is not None:
                p,r = audit(mask)
                h,t = audit(mask | (1<<j))
                # Exact dyadic arithmetic for independent interval comparisons.
                p,r,h,t = map(Fraction,(p,r,h,t))
                g = p-h
                assert (g-r-t > TOL) if lo>TOL else (g+r+t <= TOL)
                assert hi >= g-r-t and lo <= g+r+t
                counts["max_error"] = max(counts["max_error"], abs((lo+hi)/2-float(g)))
            if lo>TOL:
                if reuse and captured:
                    node._retain_winner(j,captured[-1])
                options.append((j,lo,hi))
        # The equilibrium is alive while the candidates are scored.
        sample(node,parents+z.nbytes)
        return options

    def descend(node,j):
        child = copy.copy(node)
        for field in ("active","seen","rows","stale","_winner_columns"):
            setattr(child,field,getattr(node,field).copy())
        child.commit(j)
        counts["updates"] += 1
        return child

    def undo(node,j):
        node.factor.undo(edits[j])
        node.change_entries(j,-1)
        counts["undos"] += 1

    def walk(node,mask,parents):
        seen.add(mask)
        sample(node,parents)
        if mask.bit_count() == budget:
            return []
        options = expand(node,mask,parents)
        if not options:
            return None if mode=="dfs" else []
        parent_mem = parents + sample(node,parents)
        retained = rank(options,k if mode=="dfs" else 3 if mode=="beam" else 1)
        if mode=="beam" and budget-mask.bit_count()>1:
            pairs = []
            continuations = {}
            for j,lo,hi in retained:
                child = descend(node,j)
                second = expand(child,mask|(1<<j),parent_mem)
                continuations[j] = second
                if second:
                    # Max over every certified positive continuation. Enclose
                    # the best two-step objective even when second moves tie.
                    lo = math.nextafter(lo+max(r[1] for r in second),-math.inf)
                    hi = math.nextafter(hi+max(r[2] for r in second),math.inf)
                pairs.append((j,lo,hi))
                undo(node,j)
                del child
            retained = rank(pairs,1)
            if budget-mask.bit_count()==2:
                first = retained[0][0]
                second = rank(continuations[first],1)
                if not second:
                    return [first]
                last = second[0][0]
                seen.add(mask | (1<<first) | (1<<last))
                return [first,last]
        for j,_,_ in retained:
            nxt = mask | (1<<j)
            if mode=="dfs" and nxt in seen:
                continue
            if nxt.bit_count()==budget:
                seen.add(nxt)
                return [j]
            child = descend(node,j)
            path = walk(child,nxt,parent_mem)
            undo(node,j)
            if path is not None:
                return [j]+path
            del child
            counts["backtracks"] += 1
        return None if mode=="dfs" else []

    path = walk(root,0,0)
    counts["response_rhs"] = root.ledger["refine_rhs"] + root.ledger["winner_rhs"]
    counts["transported"] = root.ledger["transported"]
    assert counts["undos"] == counts["updates"]
    return path, dict(counts,states=len(seen)), perf_counter()-tick


def run(W,case,mode,reuse,audit=None):
    prepared = PreparedSearch(W,case)
    root,_,_ = prepared.consumer(reuse)
    try:
        return query(prepared,root,mode,reuse,audit)
    finally:
        root.close()
        prepared.close()


TIMING = "structure_s root_factor_s consumer_state_s factor_clone_s setup_s first_query_s ten_query_s".split()


def paired_queries(prepared,mode):
    consumers = {reuse: prepared.consumer(reuse) for reuse in (False,True)}
    rows = []
    try:
        for rep in range(10):
            for reuse in ((True,False) if rep%2 else (False,True)):
                root,state,clone = consumers[reuse]
                path,c,online = query(prepared,root,mode,reuse)
                setup = prepared.structure_s + prepared.factor_s + state + clone
                timing = dict(structure_s=prepared.structure_s,root_factor_s=prepared.factor_s,
                              consumer_state_s=state,factor_clone_s=clone,setup_s=setup)
                rows.append((rep,reuse,path,c,online,timing))
        for reuse in consumers:
            group = [row for row in rows if row[1]==reuse]
            setup = group[0][-1]["setup_s"]
            for *_,timing in group:
                timing.update(first_query_s=setup+group[0][4],
                              ten_query_s=setup+sum(row[4] for row in group))
    finally:
        for root,_,_ in consumers.values():
            root.close()
    return rows


FIELDS = "case repeat mode policy seconds path gain states expansions state_rhs response_rhs factors updates undos backtracks peak_bytes max_error min_margin min_rank_slack rank_checks factor_clones transported".split() + TIMING


def experiment(task):
    from sparse_search_verify import replay
    graph,i,case = task
    W = load_graph(graph)
    A,s,edits = fixture(W,case)
    oracle = independent(A,s,edits)
    rows = []
    prepared = PreparedSearch(W,case)
    for mode in ("dfs","greedy"):
        ref,check,_ = run(W,case,mode,True,oracle)
        assert ref == replay(case,oracle,mode)
        if mode == "dfs":
            assert ref == case["order"] and check["states"] == case["states"]
        if mode == "greedy":
            assert sum(1<<j for j in ref) == case["greedy_mask"]
        gain = oracle(0)[0] - oracle(sum(1<<j for j in ref))[0]
        for r,reuse,path,c,t,timing in paired_queries(prepared,mode):
            assert path == ref
            row = {key:c.get(key,0) for key in FIELDS}
            row.update(case=i,repeat=r,mode=mode,policy="transport" if reuse else "cold",
                       seconds=t,path=" ".join(map(str,path)),gain=gain,max_error=check["max_error"],
                       factors=1,factor_clones=1,**timing)
            rows.append(row)
    prepared.close()
    return rows


def execute(graph,workers):
    from concurrent.futures import ProcessPoolExecutor
    cases = json.loads((DATA/f"full_sequencing_{graph}.json").read_text())
    tasks = [(graph,i,case) for i,case in enumerate(cases)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        with (DATA/f"sparse_deep_{graph}.csv").open("w",newline="") as out:
            writer = csv.DictWriter(out,FIELDS)
            writer.writeheader()
            for rows in pool.map(experiment,tasks):
                writer.writerows(rows)
                out.flush()
                print(graph,rows[0]["case"],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph",choices=("Alpha","OTC"))
    parser.add_argument("--workers",type=int,default=1)
    args = parser.parse_args()
    with threadpool_limits(1):
        execute(args.graph,args.workers)







