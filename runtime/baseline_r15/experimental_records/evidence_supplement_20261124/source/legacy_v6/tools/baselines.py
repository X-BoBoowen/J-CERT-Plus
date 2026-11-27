"""Matched greedy comparisons. Run from any directory with py -3.11 baselines.py."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from collections import Counter
from contextlib import ExitStack
from fractions import Fraction as Q
from pathlib import Path
import csv
import random
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from engine import Edit, ViewManager
from certified_engine import CertifiedViewManager, Unresolved

DATA = Path(__file__).resolve().parents[1] / "data"
DELTA = 1e-9


def update(A, edit):
    q = [edit.i, edit.j]
    w = edit.w
    D = np.array([[0.,-2*w],[-2*w,0.]]) if edit.flip else np.array([[w,-w],[-w,w]])
    return A + sparse.coo_matrix((D.ravel(), (np.repeat(q, 2), np.tile(q, 2))),
                                 shape=A.shape).tocsc()


def quad(A, edit, z, incumbent, ledger, iterate=None):
    """Bound g = z'Dz - (Dz)'(A+D)^-1(Dz) by Gauss and left Radau."""
    idx = [edit.i, edit.j]
    U, signs = edit.local()
    D = (U * signs) @ U.T
    v = np.zeros_like(z)
    v[idx] = D @ z[idx]
    a, mass = float(z[idx] @ v[idx]), float(v @ v)
    if mass == 0:
        return 0., 0.
    q = v / np.sqrt(mass)
    old, beta = np.zeros_like(v), 0.
    gauss, corner, anchor = 0., 1., 0.5
    pivot, shifted = 1., 1.
    direction = np.zeros_like(v) if iterate is not None else None
    for step in range(min(80, len(v))):
        p = A @ q
        p[idx] += D @ q[idx]
        p -= beta * old
        d = q @ p
        p -= d * q
        # LDL recurrences for T_k, T_k-anchor*I and (T_k^-1)[1,k].
        prev = pivot
        pivot = d - beta**2 / prev
        shifted = d - anchor - beta**2 / shifted
        if min(pivot, shifted) <= 0:
            break
        cross = 1. if step == 0 else -beta * corner
        if iterate is not None:
            direction = q - (beta/prev)*direction
            iterate += (np.sqrt(mass)*cross/pivot)*direction
        gauss += cross**2 / pivot
        corner = cross / pivot
        beta = np.linalg.norm(p)
        schur = anchor + beta**2 * (1/shifted - 1/pivot)
        radau = gauss + (beta * corner)**2 / schur
        low, high = a - mass * radau, a - mass * gauss
        pad = 2e-13 * (1 + abs(a) + mass * radau)
        low, high = low-pad, high+pad
        ledger["matvecs"] += 1
        if high < incumbent or high-low < 2e-10 or beta < 1e-14:
            return low, high
        old, q = q, p / beta
    return -np.inf, np.inf


def energy_interval(A, x, s, max_row):
    """Variational bound on s' A^-1 s for symmetric A >= I.

    All envelopes and their combinations use dyadic rational arithmetic. Sparse
    row accumulation and full dots use a conservative binary64 gamma bound.
    """
    n = len(x)
    u, tiny = Q(1, 2**53), Q(1, 2**1074)
    k = 2*int(np.max(np.diff(A.indptr)))+2
    xcap = Q(float(np.max(abs(x))))
    scap = Q(float(np.max(abs(s))))
    r = s - A @ x
    rcap = Q(float(np.max(abs(r))))
    component = k*(u*(scap+max_row*xcap)+tiny)/(1-k*u)
    kdot = 2*n+2
    doterr = kdot*(u*n*xcap*(scap+rcap)+2*tiny)/(1-kdot*u)
    mid = Q(float(x @ s)) + Q(float(x @ r))
    error = doterr + n*xcap*component
    return mid-error, mid+error+n*(rcap+component)**2


class CertifiedQuad(CertifiedViewManager):
    def select(self, state, s, delta=0., pool=None):
        self.direct_fallback = False
        try:
            return super().select(state,s,delta,pool)
        except Unresolved:
            self.direct_fallback = True
            return super().select(state,s,delta,pool)

    def solve(self, state, s):
        z = super().solve(state, s)
        self.opinions = s
        self.current_energy = energy_interval(self.A, z, s, self.max_row)
        return z

    def _interval(self, state, e, z, rho):
        edit = self.edits[e]
        correction = np.zeros_like(z)
        quad(self.A, edit, z, -np.inf, self.ledger, iterate=correction)
        trial = update(self.A, edit)
        lo, hi = energy_interval(trial, z-correction, self.opinions, self.max_row)
        self.ledger["queries"] += 1
        if self.direct_fallback:
            direct = splu(trial).solve(self.opinions)
            lo, hi = energy_interval(trial, direct, self.opinions, self.max_row)
            self.ledger["fallbacks"] += 1
        a,b = self.current_energy
        return a-hi, b-lo

    def commit(self,state,e):
        return tuple(sorted((*state,e)))


class CertifiedChol(CertifiedViewManager):
    """Identical certified selection, with fresh factor responses every epoch."""
    def __init__(self, A, edits, resources):
        from cholmod_backend import CholmodFactor
        super().__init__(A, edits)
        self.factor = resources.enter_context(CholmodFactor(A))
        self.A = self.base
        self._active = ()
        self.ledger["factorizations"] += 1

    def _activate(self, state):
        if tuple(state) != self._active:
            raise ValueError("factor backend requires sequential states")

    def commit(self,state,e):
        edit = self.edits[e]
        then = time.perf_counter()
        self.factor.change(edit)
        self.ledger["modify_ms"] += 1000*(time.perf_counter()-then)
        self.ledger["updates"] += 1
        self.ledger["downdates"] += int(edit.flip)
        self.A = update(self.A,edit)
        self._active = tuple(sorted((*state,e)))
        self.views.clear()
        self.errors.clear()
        return self._active


def reference(A, edits, s, budget):
    result = []
    for _ in range(budget):
        phi = s @ splu(A).solve(s)
        scores = [(phi - s @ splu(update(A, edit)).solve(s), e)
                  for e, edit in enumerate(edits) if e not in result]
        best, e = max(scores, key=lambda pair: (pair[0], -pair[1]))
        if best <= DELTA:
            break
        result.append(e)
        A = update(A, edits[e])
    return result


def run(method, base, edits, s, budget, diagnostics=False):
    with ExitStack() as resources:
        return _run(method, base, edits, s, budget, diagnostics, resources)


def _run(method, base, edits, s, budget, diagnostics, resources):
    A = base
    if method == "Certified":
        manager = CertifiedViewManager(base, edits)
    elif method == "CholCert":
        manager = CertifiedChol(base, edits, resources)
    elif method == "QuadCert":
        manager = CertifiedQuad(base, edits)
    else:
        manager = ViewManager(base, edits)
    counts = Counter()
    if method == "Chol":
        from cholmod_backend import CholmodFactor
        L = resources.enter_context(CholmodFactor(base))
    result, state = [], ()
    for _ in range(budget):
        if method in ("J", "Certified", "CholCert", "QuadCert"):
            best, _ = manager.select(state, s, DELTA)
            if best is None:
                break
            state = manager.commit(state, best)
            result.append(best)
            continue
        # Reuse the published screen on the current matrix without allocating
        # an additional factor merely for its four local coefficients.
        manager.A, manager._active = A, state
        manager.diagonal = A.diagonal()
        manager.offdiag = np.asarray(A[manager.i, manager.j]).reshape(-1)
        if method == "Quad":
            lu = splu(A)
            if diagnostics:
                counts["factor_bytes"] = max(counts["factor_bytes"], factor_bytes(lu))
            z = lu.solve(s)
        else:
            z = L.solve(s)
        ids = np.array([e for e in range(len(edits)) if e not in state], dtype=int)
        _, upper_bounds = manager._screens(state, ids, z)
        bounds = zip(upper_bounds, ids)
        best, lower = None, 0.
        intervals = {}
        for upper, e in sorted(bounds, key=lambda item: (-item[0], item[1])):
            if upper < lower - 1e-12*(1+abs(lower)):
                break
            edit = edits[e]
            counts["queries"] += 1
            if method == "Quad":
                lo, hi = quad(A, edit, z, lower, counts)
                intervals[e] = lo, hi
                if lo > lower:
                    best, lower = e, lo
            else:
                U, signs = edit.local()
                q = [edit.i, edit.j]
                rhs = np.zeros((len(s), len(signs)))
                rhs[q] = U
                Y = L.solve(rhs)
                h = U.T @ z[q]
                g = h @ np.linalg.solve(np.diag(signs) + U.T @ Y[q], h)
                if g > lower:
                    best, lower = e, g
        if method == "Quad":
            contenders = [e for e, (_, hi) in intervals.items() if hi >= max(lower, DELTA)]
            if len(contenders) > 1 or (contenders and lower <= DELTA):
                lower, best = 0., None
                for e in contenders:
                    g = s @ (z - splu(update(A, edits[e])).solve(s))
                    counts["fallbacks"] += 1
                    if g > lower:
                        best, lower = e, g
        if best is None or lower <= DELTA:
            break
        edit = edits[best]
        if method == "Chol":
            then = time.perf_counter()
            L.change(edit)
            counts["updates"] += 1
            counts["downdates"] += int(edit.flip)
            counts["modify_ms"] += 1000*(time.perf_counter() - then)
        A = update(A, edit)
        result.append(best)
        state = tuple(sorted(result))
    counts.update(manager.ledger)
    if method == "Chol" and diagnostics:
        counts["factor_bytes"] = L.bytes
        counts["peak_bytes"] = L.peak_bytes
    if method == "CholCert" and diagnostics:
        counts["factor_bytes"] = manager.factor.bytes
        counts["peak_bytes"] = manager.factor.peak_bytes
    if method in ("J","Certified","QuadCert") and diagnostics:
        counts["factor_bytes"] = factor_bytes(manager.factor)
        counts["gram_bytes"] = manager.gram_bytes()
    return result, counts


def factor_bytes(lu):
    return sum(a.nbytes for M in (lu.L, lu.U) for a in (M.data, M.indices, M.indptr))


def summarize():
    """Per fixed input, empirical quartiles and exact sign-test median CI."""
    columns = ("n", "graph", "scenario", "method", "cohort", "p25_ms", "median_ms",
               "p75_ms", "ci_low_ms", "ci_high_ms", "matches")
    with (DATA/"baselines_summary.csv").open("w",newline="") as file:
        writer = csv.DictWriter(file,fieldnames=columns)
        writer.writeheader()
        for n in (64,512):
            for suffix in ("","_certified"):
                grouped = {}
                with (DATA/f"baselines_{n}{suffix}.csv").open() as raw:
                    for row in csv.DictReader(raw):
                        key = (row["graph"],row["scenario"],row["method"])
                        grouped.setdefault(key,[]).append(row)
                for (graph,scenario,method), runs in grouped.items():
                    vals = sorted(float(r["ms"]) for r in runs)
                    if len(vals) != 10:
                        raise ValueError("ten completed repetitions required")
                    q1,med,q3 = np.quantile(vals,[.25,.5,.75])
                    writer.writerow(dict(n=n,graph=graph,scenario=scenario,method=method,
                        cohort="certified" if suffix else "mixed",p25_ms=q1,median_ms=med,
                        p75_ms=q3,ci_low_ms=vals[1],ci_high_ms=vals[-2],
                        matches=sum(int(r["matches"]) for r in runs)))


def main():
    from workloads import NAMES, load_graph, system, candidates, opinions
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=[64,512], default=64)
    parser.add_argument("--certified", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        return summarize()
    n = args.n
    methods = ["CholCert","QuadCert","Certified"] if args.certified else ["Chol","Quad","J","Certified"]
    M, budget = (24,4) if n == 64 else (128,6)
    suffix = "_certified" if args.certified else ""
    fields = ("graph scenario rep method n m budget ms matches chosen factor_bytes peak_bytes gram_bytes "
              "queries matvecs fallbacks updates downdates modify_ms").split()
    with (DATA/f"baselines_{n}{suffix}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for name in NAMES:
            W = load_graph(name, n=n)
            A = system(W).tocsc()
            edits = [Edit(*row) for row in candidates(W, M, seed=17)]
            for scenario in ("S1", "S2", "S3"):
                s = opinions(W, scenario, seed=31)
                oracle = reference(A, edits, s, budget)
                memory = {}
                for rep in range(-1, 10):
                    order = methods.copy()
                    random.Random(2026 + rep).shuffle(order)
                    for method in order:
                        start = time.perf_counter()
                        result, ledger = run(method, A, edits, s, budget, diagnostics=(rep == -1))
                        ms = 1000*(time.perf_counter()-start)
                        if rep == -1:
                            memory[method] = {k:v for k,v in ledger.items() if k.endswith("_bytes")}
                        else:
                            ledger.update(memory[method])
                            writer.writerow(dict(ledger, graph=name, scenario=scenario, rep=rep,
                                method=method, n=A.shape[0], m=len(edits), budget=budget, ms=ms,
                                matches=int(result == oracle), chosen=" ".join(map(str,result))))
                file.flush()
                print(name, scenario, flush=True)


if __name__ == "__main__":
    main()


