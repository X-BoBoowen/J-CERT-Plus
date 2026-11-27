"""Independent safe-order replay using only independently assembled full energies."""
import argparse
import csv
import json
from fractions import Fraction
from functools import lru_cache

from sparse_search import DATA, TOL, fixture, independent
from workloads import load_graph


def replay(case, energies, mode):
    @lru_cache(None)
    def oracle(mask):
        return tuple(map(Fraction,energies(mask)))
    k = len(case["edits"])
    budget = case.get("budget",k)
    seen = set()

    def best(rows, limit):
        selected = []
        rows = list(rows)
        while rows and len(selected)<limit:
            hi = max(x[2] for x in rows)
            viable = [x for x in rows if hi-x[1] < TOL]
            if not viable:
                raise AssertionError("independent ranking residual too wide")
            row = min(viable,key=lambda x:x[0])
            selected.append(row)
            rows.remove(row)
        return selected

    def safe(mask):
        p,r = oracle(mask)
        rows = []
        for j in range(k):
            if mask>>j & 1:
                continue
            h,t = oracle(mask | (1<<j))
            lo,hi = p-h-r-t,p-h+r+t
            assert lo>TOL or hi<TOL
            if lo>TOL:
                rows.append((j,lo,hi))
        return rows

    def visit(mask):
        seen.add(mask)
        if mask.bit_count()==budget:
            return []
        ranks = best(safe(mask),{"dfs":k,"beam":3,"greedy":1}[mode])
        if mode=="beam" and budget-mask.bit_count()>1:
            root,rr = oracle(mask)
            pairs = []
            for j,lo,hi in ranks:
                first = mask | (1<<j)
                choices = safe(first)
                if choices:
                    # Directly compare complete two-edit energy differences.
                    ends = [oracle(first|(1<<h)) for h,*_ in choices]
                    lo = root-rr-min(f+r for f,r in ends)
                    hi = root+rr-min(f-r for f,r in ends)
                pairs.append((j,lo,hi))
            ranks = best(pairs,1)
        for j,_,_ in ranks:
            child = mask | (1<<j)
            if mode=="dfs" and child in seen:
                continue
            path = visit(child)
            if path is not None:
                return [j,*path]
        return None if mode=="dfs" else []
    return visit(0)


def exhaustive_deep(A,s,edits,oracle,case):
    """All 2^15 subsets via the exact invariant twin difference eigenspaces."""
    import numpy as np
    k = len(edits)
    assert k == 15
    occupied = {u for e in edits[:7] for u in (e.i,e.j)}
    satellite_gains = []
    for e in edits[7:]:
        assert not e.flip and e.i not in occupied and e.j not in occupied
        occupied.update((e.i,e.j))
        difference = np.zeros(A.shape[0])
        difference[e.i],difference[e.j] = 1,-1
        action = A @ difference
        lam = Fraction(action[e.i])
        assert lam>0 and np.array_equal(action,float(lam)*difference)
        projection = Fraction(s[e.i])-Fraction(s[e.j])
        gain = projection**2/2*(1/lam-1/(lam+2*Fraction(e.w)))
        assert gain>TOL
        satellite_gains.append(gain)
    core = [tuple(map(Fraction,oracle(m))) for m in range(128)]
    safe_core = []
    for m,(v,r) in enumerate(core):
        safe = []
        for j in range(7):
            if m & (1<<j):
                continue
            h,t = core[m | (1<<j)]
            assert v-h-r-t>TOL or v-h+r+t<=TOL
            if v-h-r-t>TOL:
                safe.append(j)
        safe_core.append(safe)
    reachable = bytearray(1<<k)
    reachable[0] = 1
    for m in range(1<<k):
        if reachable[m]:
            for j in safe_core[m&127]+[j for j in range(7,k) if not m&(1<<j)]:
                reachable[m|(1<<j)] = 1
    assert reachable[-1] and reachable[case["greedy_mask"]]
    assert not safe_core[case["greedy_mask"]&127] and case["greedy_mask"]>>7==255
    final,radius = core[-1]
    root,root_radius = core[0]
    return root-final+sum(satellite_gains),radius+root_radius


def check_anchor(graph,W):
    case = json.loads((DATA/f"sparse_trap_{graph}.json").read_text())
    oracle = independent(*fixture(W,case))
    with (DATA/f"sparse_trap_{graph}.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert case["budget"] == 2 and len(case["edits"])==3
    for mode in ("greedy","beam"):
        path = replay(case,oracle,mode)
        root,r = oracle(0)
        end,t = oracle(sum(1<<j for j in path))
        gain = Fraction(root)-Fraction(end)
        matches = [row for row in rows if row["mode"]==mode]
        assert len(matches)==20
        for row in matches:
            assert list(map(int,row["path"].split()))==path
            assert abs(gain-Fraction(float(row["gain"]))) <= Fraction(r)+Fraction(t)


def check(graph):
    W = load_graph(graph)
    check_anchor(graph,W)
    with (DATA/f"sparse_deep_{graph}.csv").open() as f:
        rows = list(csv.DictReader(f))
    cases = json.loads((DATA/f"full_sequencing_{graph}.json").read_text())
    for i,case in enumerate(cases):
        system = fixture(W,case)
        oracle = independent(*system)
        total,error = exhaustive_deep(*system,oracle,case)
        for mode in ("dfs","greedy"):
            path = replay(case,oracle,mode)
            matching = [r for r in rows if int(r["case"])==i and r["mode"]==mode]
            if mode=="dfs":
                for row in matching:
                    assert abs(Fraction(float(row["gain"]))-total) <= error+oracle((1<<15)-1)[1]
            assert len(matching)==20 and all(list(map(int,r["path"].split()))==path for r in matching)
    from sparse_traps import certify_optimum
    with (DATA/f"sparse_trap_cohort_{graph}.csv").open() as f:
        rows = list(csv.DictReader(f))
    traps = json.loads((DATA/f"sparse_trap_cohort_{graph}.json").read_text())
    for i,case in enumerate(traps):
        oracle = independent(*fixture(W,case))
        optimum,gap = certify_optimum(oracle,len(case["edits"]))
        for mode in ("greedy","beam"):
            path = replay(case,oracle,mode)
            gain = oracle(0)[0] - oracle(sum(1<<j for j in path))[0]
            matching = [r for r in rows if int(r["case"])==i and r["mode"]==mode]
            assert len(matching)==20
            for row in matching:
                assert list(map(int,row["path"].split()))==path
                # Fresh factorizations may round differently across BLAS builds.
                assert abs(float(row["optimum_gain"])-optimum)<gap/4
                assert abs(float(row["gain"])-gain)<gap/4
                assert int(row["optimal"])==int(path==[0,1])


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph",choices=("Alpha","OTC"))
    args = parser.parse_args()
    check(args.graph)



