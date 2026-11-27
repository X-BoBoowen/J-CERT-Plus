"""Two-decision margin sweep on all native Bitcoin-Alpha negative-to-positive flips."""
import argparse
import csv
import os
from pathlib import Path
from time import perf_counter

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, splu

from flip_stress import vec_cap
from workloads import load_graph, system

ROOT = Path(__file__).resolve().parents[1]
EPS = np.finfo(float).eps
J = np.diag([1., -1.])
TOLS = (1e-3, 1e-6, 1e-10, 1e-14)


def solve(A, M, b, tol):
    x, info = cg(A, b, M=M, rtol=tol, atol=0.)
    if info:
        raise RuntimeError(info)
    return x


def residual(A, absA, b, x):
    # Each row dot has <= d products. A >= I gives a state error bound.
    d = np.diff(A.indptr)
    errors = abs(b - A @ x) + (d + 8)*EPS*(abs(b) + absA @ abs(x))
    return vec_cap(errors)


def gram(A, absA, M, edges, tol):
    blocks, errors = [], []
    for i, j in edges:
        block, caps = [], []
        for sign in (-1, 1):
            b = np.zeros(A.shape[0])
            b[i], b[j] = 1, sign
            x = solve(A, M, b, tol)
            caps.append(residual(A, absA, b, x))
            block.append([x[i]-x[j], x[i]+x[j]])
        G = np.array(block).T
        blocks.append(G)
        errors.append((np.sqrt(2)*vec_cap(np.asarray(caps)) + 32*EPS)*(1+32*EPS))
    return np.array(blocks), np.array(errors)


def scores(edges, z, state_error, G, gram_error):
    i, j = edges.T
    h = np.column_stack((z[i]-z[j], z[i]+z[j]))
    B = G + J
    p = np.linalg.solve(B, h[..., None])[..., 0]
    Bp = (B @ p[..., None])[..., 0]
    hp = np.sum(h*p, axis=1)
    pBp = np.sum(p*Bp, axis=1)
    center = 2*hp - pBp
    # Completed-square identity for symmetric true B. beta=1+2w=3.
    pn = vec_cap(p, axis=1)
    mag = (abs(B) @ abs(p[..., None]))[..., 0]
    rhs_error = np.sqrt(2)*state_error + 32*EPS*(1+abs(z[i])+abs(z[j]))
    matrix_error = gram_error + 16*EPS*(1+vec_cap(G.reshape(-1,4), axis=1))
    defect = vec_cap(h-Bp, axis=1) + 32*EPS*vec_cap(abs(h)+mag, axis=1)
    rad = (2*rhs_error*pn + matrix_error*pn**2 +
           3*(defect + rhs_error + matrix_error*pn)**2 +
           128*EPS*(1+np.sum(abs(p)*(abs(h)+mag), axis=1)))*(1+128*EPS)
    return center, rad


def query(A, absA, M, s, edges, initial):
    active = np.arange(len(edges))
    initial_radius = None
    refreshed = 0
    for stage, tol in enumerate(TOLS):
        z = solve(A, M, s, tol)
        G, E = initial if stage == 0 else gram(A, absA, M, edges[active], tol)
        refreshed += len(active) if stage else 0
        center, rad = scores(edges[active], z, residual(A, absA, s, z), G, E)
        winner = int(np.argmax(center))
        if initial_radius is None:
            initial_radius = dict(zip(active, rad))
        lower, upper = np.nextafter(center-rad, -np.inf), np.nextafter(center+rad, np.inf)
        keep = upper >= lower.max()
        if keep.sum() == 1 and lower[winner] > 0:
            chosen = int(active[winner])
            return dict(winner=chosen, first_radius=initial_radius[chosen],
                        final_radius=rad[winner], refinements=refreshed,
                        state_refinements=stage, final_rtol=tol)
        active = active[keep]
    raise ArithmeticError("Unresolved winner after final tolerance")


def oracle(A, absA, M, edges, s, runner_edge=None):
    # Independently solve each edited equilibrium. Rank-two Gram data is unused.
    z = solve(A, M, s, 1e-13)
    base_error = residual(A, absA, s, z)
    support = np.flatnonzero(s)
    mid, widths = [], []
    sn = vec_cap(s)
    for i, j in edges:
        edit = A.copy()
        edit[i,j] = edit[j,i] = -1
        x = solve(edit, M, s, 1e-13)
        err = residual(edit, absA, s, x)
        m = s[support] @ (z[support]-x[support])
        magnitude = abs(s[support]) @ (abs(z[support])+abs(x[support]))
        r = (sn*(err+base_error) + (4*len(support)+16)*EPS*magnitude)*(1+32*EPS)
        mid.append(m)
        widths.append(r)
    mid, widths = np.array(mid), np.array(widths)
    lower = np.nextafter(mid-widths,-np.inf)
    upper = np.nextafter(mid+widths,np.inf)
    a = int(np.argmax(lower))
    others = np.arange(len(edges)) != a
    if runner_edge is not None:
        b = int(np.flatnonzero(np.all(edges == runner_edge,axis=1))[0])
        assert b != a and lower[b] > upper[others & (np.arange(len(edges)) != b)].max()
    # max of lower/upper endpoints encloses the runner-up gain even in ties.
    lead_lo = np.nextafter(lower[a]-upper[others].max(),-np.inf)
    lead_hi = np.nextafter(upper[a]-lower[others].max(),np.inf)
    assert lead_lo > 0 and lower[a] > 0
    return a, dict(gap_lower=lead_lo, gap_upper=lead_hi,
                   winner_lower=lower[a], winner_upper=upper[a],
                   other_upper=upper[others].max(), max_ref_radius=widths.max())


def main(refresh_first=False):
    W = load_graph("Alpha")
    A = system(W)
    absA = abs(A)
    M = sparse.diags(1/A.diagonal())
    u = sparse.triu(W, 1, format='coo')
    edges = np.column_stack((u.row[u.data<0], u.col[u.data<0]))
    # Low-degree contenders within the giant component; all other edges compete.
    pairs = np.array([[1580,3598],[2100,3630]])
    exact = splu(A.tocsc())
    coefficients = []
    for i, j in pairs:
        b = np.zeros((A.shape[0],2))
        b[i], b[j] = [1,1], [-1,1]
        G = b.T @ exact.solve(b)
        coefficients.append(4*np.linalg.inv(J+G)[0,0])
    start = perf_counter()
    initial = gram(A, absA, M, edges, TOLS[0])
    setup = perf_counter()-start
    rows, first_rows = [], []
    child_initial = None
    for target in (1e-2, 1e-4, 1e-6, 1e-8):
        z = np.zeros(A.shape[0])
        for pair, gain, coeff in zip(pairs, (.04,.04-target), coefficients):
            z[pair] = np.sqrt(gain/coeff)*np.array([1,-1])
        s = A @ z
        assert max(abs(s)) <= 1
        current, catalog = A, edges
        original_ids = np.arange(len(edges))
        for step in (1,2):
            a, reference = oracle(current,absA,M,catalog,s,pairs[1] if step == 1 else None)
            if step == 1:
                lo,hi = reference['gap_lower'],reference['gap_upper']
                assert lo <= target <= hi and hi-lo < 4e-13
                assert np.array_equal(catalog[a],pairs[0])
                blocks, setup_time = initial, setup
            else:
                assert reference['gap_lower'] > hi
                assert np.array_equal(catalog[a],pairs[1])
                if child_initial is None:
                    start = perf_counter()
                    child_initial = gram(current,absA,M,catalog,TOLS[0])
                    child_setup = perf_counter()-start
                blocks, setup_time = child_initial, child_setup
            for rep in range(10):
                start = perf_counter()
                result = query(current,absA,M,s,catalog,blocks)
                elapsed = perf_counter()-start
                assert result['winner'] == a
                result['winner'] = original_ids[a]
                rows.append(dict(target=target,step=step,rep=rep,candidates=len(catalog),
                                 **reference,setup_s=setup_time,query_s=elapsed,**result))
                if step == 1:
                    b = int(np.flatnonzero(np.all(edges==pairs[1],axis=1))[0])
                    first_rows.append(dict(target=target,rep=rep,candidates=len(edges),
                                           gap_lower=lo,gap_upper=hi,runner_up=b,
                                           setup_s=setup,query_s=elapsed,**result))
            print(target,step,reference,result,flush=True)
            if step == 1:
                current = A.copy()
                i,j = edges[a]
                current[i,j] = current[j,i] = -1
                original_ids = original_ids[original_ids != a]
                catalog = edges[original_ids]
    outputs = [('margin_path',rows)]
    if refresh_first:
        outputs.append(('margin_stress',first_rows))
    for name, records in outputs:
        with (ROOT/'data'/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh-first',action='store_true',
                        help='also regenerate the separately published first-decision timings')
    main(parser.parse_args().refresh_first)
