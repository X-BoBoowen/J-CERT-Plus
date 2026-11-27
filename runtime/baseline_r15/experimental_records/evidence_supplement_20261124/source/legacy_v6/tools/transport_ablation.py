"""Paired scalar/batch transport ablation on two fixed persistent catalogs."""
import argparse
import csv
import json
import random

import numpy as np

from planned_matrix import OUTPUT, matrix_inputs, trial


def run():
    rows = []
    for graph in ("Alpha","OTC"):
        A,s,edits = matrix_inputs(graph,1000,persistent=True)
        for rep in range(10):
            cells = [(backend,batch) for backend in ("cg","cholmod") for batch in (False,True)]
            random.Random(79+rep).shuffle(cells)
            trace = None
            solve_work = {}
            for order,(backend,batch) in enumerate(cells):
                result = trial(A,edits,s,backend,"planner",None,0.,fast=True,
                               persistent=True,batch_transport=batch)[-1]
                seq = json.loads(result["sequence"])
                if len(seq)!=40 or (trace is not None and seq!=trace):
                    raise ValueError("ablation trajectory mismatch")
                trace = seq
                counters = json.loads(result["counters"])
                work = [counters.get(k,0) for k in ("C_rhs","F_rhs","R_rhs","W_rhs",
                                                       "state_rhs","transported","gate_requests")]
                if work!=solve_work.setdefault(backend,work):
                    raise ValueError("ablation solve or gate counts differ")
                computed,used,unused = [counters.get("batch_"+k,0) for k in
                                        ("computed","consumed","unused")]
                if computed!=used+unused or (batch and not used):
                    raise ValueError("ablation batch ledger")
                rows.append(dict(graph=graph,M=1000,b=40,rep=rep,order=order,
                                 backend=backend,batch=int(batch),seconds=result["seconds"],
                                 sequence=result["sequence"],counters=result["counters"]))
    with (OUTPUT/"persistent_transport_ablation.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    check()


def check():
    with (OUTPUT/"persistent_transport_ablation.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = {(r["graph"],r["backend"],int(r["rep"]),int(r["batch"])) for r in rows}
    required = {(g,b,r,t) for g in ("Alpha","OTC") for b in ("cg","cholmod")
                for r in range(10) for t in (0,1)}
    if keys!=required or len(keys)!=len(rows):
        raise ValueError("ablation completeness")
    for graph in ("Alpha","OTC"):
        reference = None
        for rep in range(10):
            group = [r for r in rows if r["graph"]==graph and int(r["rep"])==rep]
            cells = [(b,t) for b in ("cg","cholmod") for t in (False,True)]
            random.Random(79+rep).shuffle(cells)
            solve_work = {}
            for r in group:
                seq = json.loads(r["sequence"])
                if reference is None:
                    reference = seq
                if seq!=reference or len(set(seq))!=40 or r["M"]!="1000" or r["b"]!="40":
                    raise ValueError("ablation checkpoint")
                batch = int(r["batch"])
                if cells[int(r["order"])]!=(r["backend"],batch) or not float(r["seconds"])>0:
                    raise ValueError("ablation schedule or time")
                c = json.loads(r["counters"])
                work = tuple(c.get(k,0) for k in ("C_rhs","F_rhs","R_rhs","W_rhs","state_rhs",
                                                  "transported","gate_requests"))
                if work!=solve_work.setdefault(r["backend"],work):
                    raise ValueError("ablation work mismatch")
                used = c.get("batch_consumed",0)
                if c.get("batch_computed",0)!=used+c.get("batch_unused",0) or bool(used)!=bool(batch):
                    raise ValueError("ablation speculative ledger")
    for graph in ("Alpha","OTC"):
        for backend in ("cg","cholmod"):
            cell = [r for r in rows if r["graph"]==graph and r["backend"]==backend]
            cell.sort(key=lambda r:int(r["rep"]))
            scalar,batch = [np.array([float(r["seconds"]) for r in cell if int(r["batch"])==i])
                            for i in (0,1)]
            ratios = scalar/batch
            indices = np.random.default_rng(17).integers(10,size=(10000,10))
            ci = np.quantile(np.median(ratios[indices],axis=1),[.025,.975])
            print(graph,backend,np.median(ratios),ci)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true",help="verify the saved paired measurements")
    if parser.parse_args().check:
        check()
    else:
        run()
