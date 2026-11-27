"""Full-graph certified Quad-Refine comparator with fresh Lanczos per candidate."""
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from engine import Edit
from workloads import NAMES, candidates, load_graph, opinions, system

DATA = Path(__file__).resolve().parents[1] / "data"
U = 2.0**-53
DELTA = 1e-9


def enclosure(A, edit, z, rho, incumbent, ledger, scale, rowcap, rowlen):
    """Lanczos on S(A+D)S. Accept comparisons only by a residual enclosure.

    For y approximating (A+D)^-1 v, r=v-(A+D)y, the inverse quadratic
    is y'v+y'r+r'(A+D)^-1 r. The final term lies in [0,||r||^2].
    All bounds below are inflated for binary64 roundoff; n*u < 1e-8.
    """
    n = len(z)
    w = edit.w
    if n*U >= 1e-8 or not 0 < w <= 4 or (edit.flip and w != 1):
        raise ValueError("expected positive weights up to four and unit flips")
    i, j = edit.i, edit.j
    if edit.flip:
        vi, vj = -2*z[j], -2*z[i]
    else:
        vi = w*(z[i]-z[j])
        vj = -vi
    a = z[i]*vi + z[j]*vj
    mag = abs(z[i])+abs(z[j])
    # ||D|| <= 2w. Account for rounding in v and z'Dz and the state residual.
    dv = w*(2*rho + 8*U*mag)
    vn = abs(vi)+abs(vj)
    dataerr = 1.01*(rho*(2*vn+4*w*rho) + dv*(2*vn+dv)
                    + 32*U*w*mag*mag + 1e-298)
    v = np.zeros(n)
    v[i], v[j] = vi, vj
    q = scale*v
    norm = np.linalg.norm(q)
    if norm == 0:
        return -dataerr, dataerr
    q /= norm
    old = np.zeros(n)
    direction = np.zeros(n)
    x = np.zeros(n)
    pivot = shifted = 1.
    beta = gauss = 0.
    corner = 1.
    # Signed diagonal dominance gives A+D >= I.
    anchor = 0.5*np.min(scale)**2
    lo, hi = -np.inf, np.inf

    def apply(y):
        p = A @ y
        if edit.flip:
            p[i] -= 2*y[j]
            p[j] -= 2*y[i]
        else:
            diff = w*(y[i]-y[j])
            p[i] += diff
            p[j] -= diff
        ledger["matvecs"] += 1
        return p

    for step in range(min(256, n)):
        p = scale*apply(scale*q) - beta*old
        d = q @ p
        p -= d*q
        prev = pivot
        pivot = d-beta**2/prev
        shifted = d-anchor-beta**2/shifted
        if min(pivot, shifted) <= 0:
            break
        cross = 1. if step == 0 else -beta*corner
        direction = q - (beta/prev)*direction
        x += (norm*cross/pivot)*direction
        gauss += cross**2/pivot
        corner = cross/pivot
        beta = np.linalg.norm(p)
        schur = anchor + beta**2*(1/shifted-1/pivot)
        radau = gauss+(beta*corner)**2/schur
        # Gauss / left-Radau guide stopping. Roundoff cannot decide a winner.
        narrow = norm**2*(radau-gauss) < 2e-11
        if (step in (0, 2, 5) or (step+1)%8 == 0 or narrow
                or a-norm**2*gauss < incumbent or beta == 0 or step+1 == n):
            y = scale*x
            r = v-apply(y)
            yn, rn = np.linalg.norm(y), np.linalg.norm(r)
            # ||fl(r)-r|| <= gamma_k (||v|| + || |A|+|D| || ||y||);
            # symmetry bounds the latter operator norm by rowcap.
            er = 1.01*(2*rowlen+16)*U*(vn+rowcap*yn)+1e-298
            res = 1.01*(rn+er)+1e-298
            yv = vi*y[i]+vj*y[j]
            yr = y @ r
            mid = a-yv-yr
            err = 1.01*(dataerr+yn*er+(2*n+16)*U*yn*rn
                        +16*U*(abs(a)+vn*yn+abs(yr))+1e-298)
            lo = max(lo, np.nextafter(mid-err-res**2, -np.inf))
            hi = min(hi, np.nextafter(mid+err, np.inf))
            if lo > hi:
                raise ArithmeticError("inconsistent quadrature enclosure")
            if hi < incumbent or narrow or beta == 0 or hi-lo < 2e-11:
                break
        old, q = q, p/beta
    return lo, hi


def trial(A, edits, s, delta=DELTA/2):
    from policy_kernel import PolicyKernel
    start = perf_counter()
    k = PolicyKernel(A, edits, backend="cg", cached=False)
    seq, results, ledger = [], [], Counter()
    stopped = False
    try:
        for step in range(1, 41):
            if not stopped:
                ledger["state_rhs"] += 1
                z, rho = k.solve(s)
                ids = np.array(sorted(set(range(len(edits)))-set(seq)), dtype=int)
                caps = np.asarray(k.upper(ids, z, rho))
                order = np.lexsort((ids, -caps))
                diag = k.A.diagonal()
                scale = 1/np.sqrt(diag)
                rowcap = 2*diag.max()+2*max(1., k.w.max(initial=0))
                rowlen = int(np.diff(k.A.indptr).max())
                best, lower, intervals = None, 0., {}
                ledger["screens"] += len(ids)
                for ix in order:
                    e, cap = int(ids[ix]), caps[ix]
                    if cap <= 0 or (best is not None and cap <= np.nextafter(lower+delta, -np.inf)):
                        break
                    ledger["queries"] += 1
                    lo, hi = enclosure(k.A, edits[e], z, rho, lower, ledger,
                                       scale, rowcap, rowlen)
                    intervals[e] = min(hi, cap)
                    if lo > lower:
                        best, lower = e, lo
                if any(hi > np.nextafter(lower+delta, -np.inf) for e, hi in intervals.items() if e != best):
                    raise ArithmeticError("unresolved quadrature decision")
                if best is None:
                    if any(hi > 0 for hi in intervals.values()):
                        raise ArithmeticError("unresolved stop")
                    stopped = True
                else:
                    k.commit(best)
                    seq.append(best)
            if step in (20, 40):
                counts = Counter(k.ledger)
                counts.update(ledger)
                results.append(dict(b=step, seconds=perf_counter()-start,
                                    commits=len(seq), sequence=json.dumps(seq),
                                    counters=json.dumps(dict(counts), sort_keys=True)))
    finally:
        k.close()
    return results


def check():
    """Compare with exact rational objective differences on small signed graphs."""
    from fractions import Fraction
    from baselines import update

    def exact_energy(matrix, rhs):
        rows = [[Fraction(float(v)) for v in row] + [Fraction(float(b))]
                for row, b in zip(matrix.toarray(), rhs)]
        for j, row in enumerate(rows):
            pivot = row[j]
            row[:] = [v/pivot for v in row]
            for i, other in enumerate(rows):
                if i != j:
                    c = other[j]
                    rows[i] = [v-c*w for v, w in zip(other, row)]
        return sum(Fraction(float(b))*row[-1] for b, row in zip(rhs, rows))

    from policy_kernel import PolicyKernel
    for name in NAMES:
        W = load_graph(name, 12)
        A = system(W)
        s = opinions(W, "S1", 31)
        edits = [Edit(*e) for e in candidates(W, 12, 17)]
        k = PolicyKernel(A, edits, backend="cg", cached=False)
        try:
            z, rho = k.solve(s)
            phi = exact_energy(A, s)
            weighted = [Edit(e.i, e.j, w, False) for e in edits if not e.flip
                        for w in (0.125, 1.375, 4.)]
            for edit in [*edits, *weighted]:
                gain = phi-exact_energy(update(A, edit), s)
                lo, hi = enclosure(A, edit, z, rho, 0., Counter(),
                                   1/np.sqrt(A.diagonal()), 2*A.diagonal().max()+2*max(1.,edit.w),
                                   max(np.diff(A.indptr)))
                if not Fraction(float(lo)) <= gain <= Fraction(float(hi)):
                    raise ArithmeticError((name, edit, gain, lo, hi))
        finally:
            k.close()
    print("Exact rational unit and dyadic-weight interval comparisons passed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", choices=NAMES, nargs="+", default=NAMES)
    p.add_argument("--m", type=int, nargs="+", default=[1000,2500])
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--check", action="store_true")
    p.add_argument("--catalog", choices=["uniform", "matched", "clustered", "persistent"], default="uniform")
    args = p.parse_args()
    if args.check:
        return check()
    prefix = {"uniform": "full_quad", "matched": "planned_quad", "clustered": "clustered_quad",
              "persistent": "persistent_quad"}[args.catalog]
    for name in args.graph:
        W = load_graph(name)
        A = system(W)
        for m in args.m:
            path = DATA/f"{prefix}_{name}_{m}.csv"
            done = set()
            if path.exists():
                with path.open() as stream:
                    done = {int(r["rep"]) for r in csv.DictReader(stream) if r["b"] == "40"}
            pending = sorted(set(range(args.reps))-done)
            if not pending:
                continue
            if args.catalog == "persistent":
                from workloads import persistent_inputs
                s, catalog = persistent_inputs(W, m, 17)
            elif args.catalog == "clustered":
                from workloads import clustered_inputs
                s, catalog = clustered_inputs(W, m, seed=17)
            elif args.catalog == "matched":
                from workloads import matched_inputs
                s, catalog = matched_inputs(W, m, seed=17)
            else:
                s = opinions(W, "unsigned_control", 31)
                catalog = candidates(W, m, 17)
            edits = [Edit(*e) for e in catalog]
            for rep in pending:
                rows = [dict(graph=name, M=m, rep=rep, backend="cg", mode="quad", **r)
                        for r in trial(A, edits, s, DELTA/2 if args.catalog == "uniform" else DELTA)]
                with path.open("a", newline="") as stream:
                    out = csv.DictWriter(stream, fieldnames=rows[0])
                    if stream.tell() == 0:
                        out.writeheader()
                    out.writerows(rows)
                print(name, m, rep, rows[-1]["seconds"], flush=True)


if __name__ == "__main__":
    main()
