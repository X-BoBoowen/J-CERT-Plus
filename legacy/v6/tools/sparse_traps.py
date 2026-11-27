"""Construct budget-two traps on unmodified full Bitcoin operators."""
import argparse
import csv
import json
from fractions import Fraction
from itertools import combinations, permutations

import numpy as np
from threadpoolctl import threadpool_limits

from full_sequencing import twins
from sparse_search import DATA, TOL, fixture, independent, run, PreparedSearch, paired_queries, TIMING
from workloads import load_graph, system


def construct(W, pair, d):
    """Fit controlled opinions to a pair of incident flips and a twin insertion."""
    q = sorted(set(pair[0][:2]+pair[1][:2]))
    allq = q+d[:2]
    template = dict(q=allq,edits=[*pair,d],z=[0.]*len(allq))
    A,_,edits = fixture(W,template)
    from cholmod_backend import CholmodFactor
    from scipy import sparse
    # Only used to choose controlled opinions, outside the timed solver.
    selectors = sparse.eye(A.shape[0],format="csc")[:,q].toarray()
    forms = []
    for mask in (1,2,3):
        M = A.copy().tolil()
        for j,e in enumerate(edits[:2]):
            if mask>>j & 1:
                M[e.i,e.j] -= 2*e.w
                M[e.j,e.i] -= 2*e.w
        src = A @ selectors
        with CholmodFactor(M.tocsr()) as f:
            forms.append(src.T @ (selectors-f.solve(src)))
    rng = np.random.default_rng(581)
    for _ in range(24):
        X = rng.uniform(-1,1,size=(8192,len(q)))
        ga,ge,both = [np.einsum("ij,jk,ik->i",X,H,X) for H in forms]
        for a,b,swap in ((ga,ge,False),(ge,ga,True)):
            candidates = np.flatnonzero((a > 1e-5) & (b < -1e-4) & (both > 8*a))
            if not len(candidates):
                continue
            h = candidates[0]
            # Unit antisymmetric twin opinion, projected directly into s=A z.
            template["z"] = [0.]*len(q)+[1.,-1.]
            s = A[:,allq] @ template["z"]
            oracle = independent(A,s,edits)
            unit = oracle(0)[0]-oracle(4)[0]
            amp = np.sqrt(2*a[h]/unit)
            z = np.r_[X[h],amp,-amp]
            scale = 0.5 / np.max(abs(A[:,allq] @ z))
            template["z"] = (z*scale).tolist()
            if swap:
                template["edits"][:2] = template["edits"][:2][::-1]
            return template



def cohort(W, graph):
    cases = json.loads((DATA/f"full_sequencing_{graph}.json").read_text())
    pairs = {}
    distractors = {}
    for case in cases:
        flips = [e for e in case["edits"][:7] if e[3]]
        for a,b in combinations(flips, 2):
            if set(a[:2]) & set(b[:2]):
                pairs.setdefault(tuple(sorted((tuple(a),tuple(b)))), (a,b))
        for d in case["edits"][7:]:
            distractors.setdefault(tuple(d), d)
    anchor = json.loads((DATA/f"sparse_trap_{graph}.json").read_text())
    anchor_edits = set(map(tuple,anchor["edits"]))
    traps = []
    for d in distractors.values():
        for pair in pairs.values():
            if set(d[:2]) & set(pair[0][:2]+pair[1][:2]):
                continue
            if set(map(tuple,(*pair,d))) == anchor_edits:
                continue
            trap = construct(W,pair,d)
            if trap is None:
                continue
            # The second opportunity has disjoint endpoints. With s=A z, a
            # flip pair's isolated energy change depends only on its endpoint z.
            for second_pair in pairs.values():
                if set(trap["q"]) & set(second_pair[0][:2]+second_pair[1][:2]):
                    continue
                secondary = construct(W,second_pair,d)
                if secondary is not None:
                    break
            else:
                continue
            first_oracle = independent(*fixture(W,trap))
            second_oracle = independent(*fixture(W,secondary))
            relative = min((.35*(first_oracle(0)[0]-first_oracle(m)[0]) /
                            (second_oracle(0)[0]-second_oracle(m)[0]))**.5 for m in (1,3))
            strong_amplitude = trap["z"][-2]
            trap["q"].extend(secondary["q"][:-2])
            trap["z"].extend(relative*x for x in secondary["z"][:-2])
            trap["edits"].extend(secondary["edits"][:2])
            # Disjoint twin differences provide ten independent safe decoys.
            while len(trap["edits"]) < 15:
                extra = twins(W,set(trap["q"]))
                if not extra:
                    raise RuntimeError("insufficient disjoint same-sign twins")
                for e in extra[:15-len(trap["edits"])]:
                    fraction = .25 + .05*(len(trap["edits"])-5)
                    amplitude = strong_amplitude*np.sqrt(fraction/2)
                    trap["edits"].append(e)
                    trap["q"].extend(e[:2])
                    trap["z"].extend([amplitude,-amplitude])
            max_input = np.max(abs(system(W)[:,trap["q"]] @ trap["z"]))
            trap["z"] = (np.asarray(trap["z"])/max(1.,max_input)).tolist()
            trap["budget"] = 2
            traps.append(trap)
            if len(traps) == 10:
                return traps
    raise RuntimeError("insufficient distinct certified trap candidates")


def certify_optimum(oracle,k):
    """Compare all feasible paths of length zero, one, or two on full matrices."""
    masks = [0, *(1<<i for i in range(k)), *((1<<i)|(1<<j) for i,j in combinations(range(k),2))]
    energies = {mask: tuple(map(Fraction,oracle(mask))) for mask in masks}
    def gain(before, after):
        x, r = energies[before]
        y, t = energies[after]
        return x-y, r+t
    feasible = {(): (Fraction(0), Fraction(0))}
    for a,b in permutations(range(k),2):
        first = 1<<a
        final = first|(1<<b)
        g,r = gain(0,first)
        h,t = gain(first,final)
        assert abs(g-Fraction(TOL)) > r and abs(h-Fraction(TOL)) > t
        if g-r > TOL:
            feasible[(a,)] = g,r
            if h-t > TOL:
                feasible[a,b] = gain(0,final)
    best,r = feasible[0,1]
    gap = min(best-r-(v+e) for path,(v,e) in feasible.items() if path!=(0,1))
    assert gap > 0
    # Check even the terminal subsets that lack a safe ordering.
    for mask in masks:
        if mask == 3:
            continue
        value, radius = gain(0,mask)
        assert best-r > value+radius
    for before,after,sign in ((0,1,1),(0,2,-1),(0,4,1),(1,3,1),(0,8,1),(0,16,-1),(8,24,1)):
        value, radius = gain(before,after)
        assert sign*value-radius > TOL
    for a, e in ((0,1),(3,4)):
        single,sr = gain(0,1<<a)
        joint,jr = gain(0,(1<<a)|(1<<e))
        assert joint-jr > single+sr+TOL
    for j in range(5,k):
        value,radius = gain(0,1<<j)
        assert value-radius > TOL and value+radius < gain(0,1)[0]-gain(0,1)[1]
    return float(best), float(gap)


def benchmark(graph):
    W = load_graph(graph)
    traps = json.loads((DATA/f"sparse_trap_cohort_{graph}.json").read_text())
    assert len({tuple(sorted(map(tuple,c["edits"]))) for c in traps}) == 10
    counters = "states state_rhs response_rhs peak_bytes min_margin min_rank_slack transported".split()
    fields = ("case mode repeat policy path gain optimum_gain optimum_gap optimal "
              "root_a root_e root_d e_after_a seconds").split()
    fields += TIMING + ["factors","factor_clones","updates","undos"] + counters + ["secondary_a", "secondary_e", "secondary_after_a"]
    with (DATA/f"sparse_trap_cohort_{graph}.csv").open("w",newline="") as out:
        w = csv.DictWriter(out,fields)
        w.writeheader()
        for i,trap in enumerate(traps):
            A,s,edits = fixture(W,trap)
            oracle = independent(A,s,edits)
            optimum,gap = certify_optimum(oracle,len(edits))
            prepared = PreparedSearch(W,trap)
            for mode in ("greedy","beam"):
                ref,_,_ = run(W,trap,mode,True,oracle)
                assert ref == ([2,0] if mode=="greedy" else [0,1])
                for r,reuse,path,c,t,timing in paired_queries(prepared,mode):
                    assert path == ref
                    w.writerow(dict(case=i,mode=mode,repeat=r,policy="transport" if reuse else "cold",
                                    path=" ".join(map(str,path)),gain=oracle(0)[0]-oracle(sum(1<<j for j in path))[0],
                                    optimum_gain=optimum,optimum_gap=gap,optimal=int(path==[0,1]),
                                    root_a=oracle(0)[0]-oracle(1)[0],root_e=oracle(0)[0]-oracle(2)[0],
                                    root_d=oracle(0)[0]-oracle(4)[0],e_after_a=oracle(1)[0]-oracle(3)[0],
                                    seconds=t,**timing,factors=1,factor_clones=1,updates=c["updates"],undos=c["undos"],secondary_a=oracle(0)[0]-oracle(8)[0],
                                    secondary_e=oracle(0)[0]-oracle(16)[0],secondary_after_a=oracle(8)[0]-oracle(24)[0],
                                    **{key:c[key] for key in counters}))
            prepared.close()
            out.flush()
            print(graph,i,flush=True)


def benchmark_anchor(graph):
    """Replay the frozen three-edit gain control using the same shared root as the cohort."""
    trap=json.loads((DATA/f"sparse_trap_{graph}.json").read_text())
    W=load_graph(graph)
    oracle=independent(*fixture(W,trap))
    root=oracle(0)[0]
    evidence=dict(root_a=root-oracle(1)[0],root_e=root-oracle(2)[0],root_d=root-oracle(4)[0],
                  e_after_a=oracle(1)[0]-oracle(3)[0])
    prepared=PreparedSearch(W,trap)
    rows=[]
    try:
        for mode in ("greedy","beam"):
            for rep,reuse,path,c,elapsed,_ in paired_queries(prepared,mode):
                if path!=([2,0] if mode=="greedy" else [0,1]):
                    raise ValueError(f"{graph} {mode}: frozen anchor winner changed")
                rows.append(dict(mode=mode,repeat=rep,policy="transport" if reuse else "cold",
                                 path=" ".join(map(str,path)),gain=root-oracle(sum(1<<j for j in path))[0],
                                 **evidence,seconds=elapsed,**{k:c[k] for k in
                                 ("states","state_rhs","response_rhs","peak_bytes","min_margin","min_rank_slack")}))
    finally:
        prepared.close()
    with (DATA/f"sparse_trap_{graph}.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graphs", nargs="*", default=["Alpha","OTC"])
    with threadpool_limits(1):
        for graph in parser.parse_args().graphs:
            benchmark_anchor(graph)
            benchmark(graph)

