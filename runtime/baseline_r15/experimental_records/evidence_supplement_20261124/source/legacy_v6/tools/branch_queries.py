"""Four matched levels on twelve full-graph branching queries."""
import argparse
import copy
import csv
import json
import random
import os
from pathlib import Path
from collections import Counter, defaultdict
from time import perf_counter

diagnostic = Path(__file__).resolve().parents[2]/"chol/ucrt64/bin/jcert_diagnostic.dll"
if diagnostic.exists():
    os.environ.setdefault("JCERT_CHOLMOD_LIB",str(diagnostic))

from orthogonal import OUTPUT
import numpy as np
from cost_planner import CostPlanner
from policy_kernel import PolicyKernel
from fast_policy_kernel import FastPolicyKernel,FastCostPlanner
from engine import Edit
from workloads import contention_candidates, load_graph, opinions, system


class PinnedPlanner(FastCostPlanner):
    def _historical(self, pending):
        if pending.operator[-1] is None:
            return super()._historical(pending)
        current = self._operator()
        iterations = self.last_iterations
        try:
            self._bind_operator(pending.operator)
            return self._fresh(pending.winner, "W")
        finally:
            self._bind_operator(current)
            self.last_iterations = iterations


def branch(k):
    child = copy.copy(k)
    for name in ('active','seen','rows','stale','_winner_columns'):
        setattr(child,name,getattr(k,name).copy())
    child.ledger = Counter()
    if k.factor:
        child.factor = k.factor.clone()
        child.ledger['factor_clones'] += 1
    if isinstance(child,CostPlanner):
        child.pending = None
    return child


def trial(template, queries, level, pin, move=False):
    start = perf_counter()
    paths = [[] for _ in queries]
    total = Counter()
    live = [template.factor]
    memory = Counter()
    def sample(k):
        if move:
            extra = k.ledger.pop('peak_planner_clone_bytes', 0)
            memory['peak_native_envelope_bytes'] = max(
                memory['peak_native_envelope_bytes'], sum(f.peak_bytes for f in live) + extra)
            memory['peak_live_factors'] = max(memory['peak_live_factors'], len(live) + bool(extra))
    def visit(k, ids, depth):
        groups = defaultdict(list)
        solved = {}
        for q in ids:
            s, budget, pool, family = queries[q]
            if family not in solved:
                solved[family] = k.solve(s)
                k.ledger['state_rhs'] += 1
            eligible = pool if depth >= 3 else None
            winner,_ = k.select_solved(s,*solved[family],eligible,1e-9)
            if winner is not None:
                paths[q].append(int(winner))
                if len(paths[q]) < budget:
                    groups[winner].append(q)
        sample(k)
        for i,(winner,children) in enumerate(groups.items()):
            child = k if (move or not pin) and i == len(groups)-1 else branch(k)
            if move:
                if child is k:
                    k.ledger['moved_edges'] += 1
                else:
                    live.append(child.factor)
                    child.ledger['fork_clones'] += 1
            child.commit(winner)
            if child is not k and isinstance(child,PinnedPlanner) and child.pending and child.pending.operator:
                child.pending.operator = k._operator()
                child.pending.replay_work = 0
            visit(child,children,depth+1)
        if (pin and not move) or not groups:
            total.update(k.ledger)
            if move:
                live.pop()
            k.close()
    def root_visit(ids):
        k = branch(template)
        if move:
            live.append(k.factor)
            k.ledger['root_clones'] += 1
        visit(k,ids,0)
    if level:
        root_visit(list(range(len(queries))))
    else:
        for q in range(len(queries)):
            root_visit([q])
    total.update(memory)
    return dict(seconds=perf_counter()-start,paths=json.dumps(paths),
                flips=sum(template.flip[e] for path in paths for e in path),
                counters=json.dumps(dict(total)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph',choices=['Alpha','OTC'],required=True)
    parser.add_argument('--reps',type=int,default=10)
    parser.add_argument('--audit-revivals',action='store_true')
    parser.add_argument('--s2-control',action='store_true')
    parser.add_argument('--rational-control',action='store_true')
    parser.add_argument('--summary',action='store_true')
    parser.add_argument('--undo-control',action='store_true')
    parser.add_argument('--move',action='store_true',help='transfer the factor to the last child; copy only to explore siblings')
    args = parser.parse_args()
    strict = not (args.undo_control or args.rational_control or args.s2_control)
    prefix = 'branch_s2' if args.s2_control else 'branch'
    if not (args.rational_control or args.s2_control):
        prefix = 'branch_pinned' if strict else 'branch_fast'
    if args.move:
        if not strict:
            parser.error('--move uses the binary64 calibrated catalog')
        prefix = 'branch_move'
    dest = OUTPUT/f'{prefix}_{args.graph.lower()}.csv'
    prev = list(csv.DictReader(dest.open())) if dest.exists() else []
    if args.summary:
        groups = {l:{int(r["rep"]):r for r in prev if int(r["level"])==l} for l in range(4)}
        reference = prev[0]["paths"]
        if len(prev)!=4*args.reps or any(row["paths"]!=reference for row in prev):
            raise ValueError("incomplete or mismatched paired study")
        if args.move:
            with (OUTPUT/f'branch_pinned_{args.graph.lower()}.csv').open() as f:
                if next(csv.DictReader(f))['paths'] != reference:
                    raise ValueError("ownership transfer changed the reference paths")
        if strict:
            schedule = ("factor_clones","factor_up","factor_down","state_rhs",
                        "planner_parent_clones","planner_undo_up","planner_undo_down")
            if args.move:
                schedule = ("root_clones","fork_clones","moved_edges","factor_up","factor_down","state_rhs")
            for rep in range(args.reps):
                counts = [json.loads(groups[level][rep]["counters"]) for level in range(4)]
                if len({tuple(c.get(k,0) for k in schedule) for c in counts[1:]}) != 1:
                    raise ValueError("shared factor or state schedule differs")
                if counts[0]["refine_rhs"] != counts[1]["refine_rhs"]:
                    raise ValueError("cold candidate RHS mismatch")
                if args.move:
                    for level, c in enumerate(counts):
                        roots = 1 if level else len(json.loads(reference))
                        forks = c.get("fork_clones",0)
                        recovery = c.get("planner_parent_clones",0)
                        if (c["root_clones"] != roots
                                or c["factor_clones"] != roots + forks + recovery
                                or c["factor_up"] != forks + c["moved_edges"]
                                or recovery != c.get("planner_undo_down",0)
                                or recovery > c.get("W",0)
                                or (level < 3 and recovery)):
                            raise ValueError("incorrect ownership or recovery accounting")
        for l,rows in groups.items():
            if set(rows)!=set(range(args.reps)):
                raise ValueError("missing paired repetition")
            c = json.loads(rows[0]["counters"])
            cand = c["refine_rhs"]+c.get("winner_rhs",0)
            print(l,round(float(np.median([float(r["seconds"]) for r in rows.values()])),3),
                  c["state_rhs"],cand,cand+c["state_rhs"])
        ratios = sorted(float(groups[2][i]["seconds"])/float(groups[3][i]["seconds"]) for i in range(args.reps))
        print("L2/L3 median and ordered second/ninth", np.median(ratios), ratios[1], ratios[-2])
        return
    W = load_graph(args.graph)
    A = system(W)
    if args.s2_control:
        s = opinions(W,'S2',17)
        edits = contention_candidates(W,1000,s,seed=17)
    else:
        from workloads import matched_inputs
        s,edits = matched_inputs(W,1000,seed=17)
    edits = [Edit(*e) for e in edits]
    rng = np.random.default_rng(81)
    perturbed = s.copy()
    if args.s2_control:
        perturbed = np.clip(s+.025*rng.normal(size=len(s)),-1,1)
    else:
        active = np.flatnonzero(s)
        ids = rng.choice(active,min(12,len(active)),replace=False)
        perturbed[ids] *= 1+rng.uniform(-.001,.001,size=len(ids))
        perturbed = np.clip(perturbed,-1,1)
    opinions_list = [s,perturbed]
    budgets = [20,10,15,25]*3
    idx = np.arange(len(edits))
    queries = [(opinions_list[q//6],budgets[q],None if q%6==0 else idx[(idx+q)%7!=0],q//6) for q in range(12)]
    if args.rational_control or args.s2_control:
        base,planner = PolicyKernel,CostPlanner
    else:
        base,planner = FastPolicyKernel,PinnedPlanner if strict else FastCostPlanner
    kernels = [cls(A,edits,cached=cache,backend='cholmod',directional=True,winner_slots=int(level==3))
               for level,(cls,cache) in enumerate([(base,False),(base,False),(base,'same'),(planner,True)])]
    if args.audit_revivals:
        paths = json.loads(prev[0]['paths'])
        revived = 0
        for (opinion,_,_,_), path in zip(queries,paths):
            kernel = branch(kernels[2])
            nonpositive = set()
            for j,e in enumerate(path):
                z,rho = kernel.solve(opinion)
                revived += int(e in nonpositive)
                for f in path[j+1:]:
                    if kernel._interval(f,z,rho)[1] <= 0:
                        nonpositive.add(f)
                kernel.commit(e)
            kernel.close()
        print(args.graph,'revived winners',revived)
        for kernel in kernels:
            kernel.close()
        return
    done = {(int(r['rep']),int(r['level'])) for r in prev}
    expected = prev[0]['paths'] if prev else None
    for rep in range(args.reps):
        levels = list(range(4))
        random.Random(17+rep).shuffle(levels)
        for level in levels:
            if (rep,level) in done:
                continue
            row = dict(rep=rep,level=level,**trial(kernels[level],queries,level,strict and level>0,args.move))
            expected = expected or row['paths']
            if expected != row['paths']:
                raise RuntimeError('paired paths differ')
            with dest.open('a',newline='') as f:
                writer = csv.DictWriter(f,fieldnames=row)
                if not f.tell():
                    writer.writeheader()
                writer.writerow(row)
            print(args.graph,rep,level,row['seconds'],flush=True)
    for kernel in kernels:
        kernel.close()

if __name__ == '__main__':
    main()
