"""Live numerical payload and per-process high-water marks on matched signed catalogs."""
import argparse
import csv
from fractions import Fraction
import json
import os
import subprocess
import sys

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from fast_policy_kernel import FastCostPlanner
from linview_delta import GlobalDelta
from planned_matrix import OUTPUT, matrix_inputs
from scale_views import peak_mib
from workloads import NAMES


class Payload:
    """Count unique buffer allocations, and numeric scalar payloads in views."""
    def __init__(self):
        self.seen = set()

    def count(self, value):
        if value is None or id(value) in self.seen:
            return 0
        if isinstance(value, np.ndarray):
            if isinstance(value.base, np.ndarray):
                return self.count(value.base)
            self.seen.add(id(value))
            return value.nbytes + (sum(self.count(x) for x in value.flat)
                                   if value.dtype == object else 0)
        if isinstance(value, (tuple, list)):
            return sum(self.count(x) for x in value)
        self.seen.add(id(value))
        if isinstance(value, Fraction):
            return sum(max(1, (abs(x).bit_length()+7)//8)
                       for x in (value.numerator, value.denominator))
        if isinstance(value, (int, float, np.number)):
            return 8
        raise TypeError(type(value))

    def csr(self, matrix):
        return self.count((matrix.data, matrix.indices, matrix.indptr))


def measure(k, method, *, persistent=False, state=None):
    p = Payload()
    local = winner = history = aux = historical_operator = 0
    if method == "planner":
        ticket = k.pending
        tables = [k.rows] + ([ticket.rows] if ticket else [])
        local = sum(p.count(row) for table in tables for row in table.values())
        winner = sum(p.count(r) for r in k._winner_columns.values())
        if persistent:
            winner += p.count(getattr(k, "_state_response", None))
            aux += p.count(state)
            if hasattr(k, "_state_edge"):
                aux += 8
        if ticket:
            winner += p.count(ticket.response)
            aux += p.count(ticket.pivot)
            # Fixed-width equivalents for winner id, rank, cost, request and forecast fields.
            aux += 6*8
        aux += p.count(getattr(k, "_transport_trial", None))
        if ticket and ticket.operator:
            A, absA, gamma, _, invdiag, _ = ticket.operator
            for current in (k.A, k.absA):
                p.csr(current)
            p.count((k.gamma, k.invdiag))
            historical_operator = p.csr(A) + p.csr(absA) + p.count((gamma, invdiag))
    else:
        # The per-row block cursor is an integer field, independent of interning.
        local = sum(p.count((G, err))+8 for G, err, _ in k.local.values())
        winner = sum(p.count(r) for r in k._winner_columns.values())
        history = sum(p.count(block) for block in k.blocks)
        for current in (k.current, k.current_abs):
            p.csr(current)
        p.count(k.current_gamma)
        historical_operator = p.csr(k.A)+p.csr(k.absA)+p.count(k.gamma)
        aux = p.count(k.z)
    return dict(local=local, winner=winner, history=history, auxiliary=aux,
                views=local+winner+history+aux, historical_operator=historical_operator,
                with_operator=local+winner+history+aux+historical_operator)


def replay(graph, method, persistent):
    A, s, edits = matrix_inputs(graph, 2500, clustered=not persistent, persistent=persistent)
    k = (FastCostPlanner(A, edits, backend="cholmod", winner_slots=1,
                         batch_transport=persistent)
         if method == "planner" else GlobalDelta(A, edits, "cholmod", s))
    rows = []
    state = None
    try:
        for step in range(1, 41):
            if method == "linview":
                e = k.choose(s)
            elif persistent:
                if state is None:
                    state, radius = k.solve(s)
                    k.ledger["state_rhs"] += 1
                else:
                    state, radius = k.solve_committed_state(s, state)
                e = k.select_solved(s, state, radius, delta=1e-9)[0]
            else:
                e = k.select(s, delta=1e-9)[0]
            if e is None:
                raise ValueError("unexpected early stop")
            for phase in ("pre", "post"):
                if phase == "post":
                    k.commit(e)
                rows.append(dict(graph=graph, method=method, n=A.shape[0], step=step,
                                 phase=phase, edit=int(e), rank=1+int(edits[e].flip),
                                 **measure(k, method, persistent=persistent, state=state),
                                 factor=k.factor.bytes,
                                 peak_rss_mib=peak_mib()))
                if persistent:
                    rows[-1]["factor_peak_bytes"] = k.factor.peak_bytes
                    if method == "planner":
                        assert k._batch is None and k._lookahead is None
                        rows[-1]["batch_computed"] = k.ledger["batch_computed"]
                        rows[-1]["batch_consumed"] = k.ledger["batch_consumed"]
    finally:
        k.close()
    rows[-1]["peak_rss_mib"] = peak_mib()
    with path_for(graph, method, persistent).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def path_for(graph, method, persistent):
    prefix = "view_memory_persistent" if persistent else "view_memory"
    return OUTPUT / f"{prefix}_{graph}_{method}.csv"


def verify(graph, persistent=False):
    traces = []
    for method in ("planner", "linview"):
        with path_for(graph, method, persistent).open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 80
        path = []
        rank = 0
        for i, r in enumerate(rows):
            assert r["graph"] == graph and r["method"] == method
            assert int(r["step"]) == 1+i//2 and r["phase"] == ("post" if i%2 else "pre")
            assert int(r["views"]) == sum(int(r[key]) for key in ("local","winner","history","auxiliary"))
            assert int(r["with_operator"]) == int(r["views"]) + int(r["historical_operator"])
            if persistent:
                assert int(r["auxiliary"]) >= 8 * int(r["n"])
                assert int(r["factor_peak_bytes"]) >= int(r["factor"]) > 0
                if i:
                    assert float(r["peak_rss_mib"]) >= float(rows[i-1]["peak_rss_mib"])
            if i%2:
                assert r["edit"] == rows[i-1]["edit"]
                rank += int(r["rank"])
                path.append(int(r["edit"]))
            expected = (16*int(r["n"])*rank+8*((i+1)//2)) if method=="linview" else 0
            assert int(r["history"]) == expected
        traces.append(path)
        if persistent and method == "planner":
            batch = {key: int(rows[-1][key]) for key in ("batch_computed", "batch_consumed")}
            assert batch["batch_computed"] >= batch["batch_consumed"] > 0
        print(graph,method,"view,view+historical op,process MiB",
              [round(max(float(r[key]) for r in rows)/scale, 4)
               for key,scale in (("views",2**20),("with_operator",2**20),("peak_rss_mib",1))])
        if method == "linview":
            fixture = "persistent" if persistent else "clustered"
            timing = OUTPUT / f"linview_{fixture}_{graph}_2500.csv"
            if timing.exists():
                with timing.open() as f:
                    for raw in csv.DictReader(f):
                        if raw["b"] == "40":
                            assert path == json.loads(raw["sequence"])
                            assert int(raw["history_bytes"]) + 320 == expected
    assert traces[0] == traces[1] and len(set(traces[0])) == 40
    prefix = "persistent_batch" if persistent else "clustered_fast"
    for timing in OUTPUT.glob(f"{prefix}_{graph}_2500*.csv"):
        with timing.open() as f:
            for raw in csv.DictReader(f):
                if raw["b"] == "40" and raw["mode"] == "planner":
                    assert traces[0] == json.loads(raw["sequence"])
                    if persistent and raw["backend"] == "cholmod":
                        counters = json.loads(raw["counters"])
                        assert all(value == counters.get(key, 0) for key, value in batch.items())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", nargs="+", choices=NAMES, default=NAMES)
    p.add_argument("--child", choices=("planner","linview"))
    p.add_argument("--check", action="store_true")
    p.add_argument("--persistent", action="store_true")
    args = p.parse_args()
    for graph in args.graph:
        if args.check:
            verify(graph, args.persistent)
        elif args.child:
            replay(graph, args.child, args.persistent)
        else:
            for method in ("planner","linview"):
                flags = ["--persistent"] if args.persistent else []
                subprocess.run([sys.executable, __file__, "--graph",graph,"--child",method,*flags],
                               check=True, creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            verify(graph, args.persistent)


if __name__ == "__main__":
    main()

