"""Independent interval certificate for the bounded scale winner traces.

Only graph construction is shared with the experiment. SciPy CG supplies guesses;
integer residuals and strict diagonal dominance certify root endpoint enclosures.
Outward-rounded mpmath interval arithmetic then propagates signed rank-one edits.
No planner scores, response columns, or ranking code are consumed.
"""
import argparse
import csv
import json
from pathlib import Path

from scale_views import ROOT, fixture, np
from scipy.sparse.linalg import cg, LinearOperator
from mpmath import iv

SCALE = 1 << 46


def require(condition, message):
    if not condition:
        raise ValueError(message)


def interval(numerator, denominator=1):
    return iv.mpf(int(numerator)) / int(denominator)


def root_enclosures(matrix, source, endpoints):
    """Varah's infinity-norm bound: ||A^-1||_inf <= 1 for this fixture.

    A*8 and quantized guesses are integers; every sparse multiplication is
    checked against int64 overflow. Rounding the source adds at most 1/(2D).
    """
    scaled = matrix * 8
    require(np.array_equal(scaled.data, np.rint(scaled.data)), "non-dyadic graph")
    integer = scaled.astype(np.int64)
    require((integer != integer.T).nnz == 0, "symmetric root matrix")
    rownorm = np.asarray(abs(integer).sum(axis=1)).ravel()
    require(np.all(2*integer.diagonal()-rownorm >= 8), "diagonal dominance")
    diagonal = matrix.diagonal()
    preconditioner = LinearOperator(matrix.shape, matvec=lambda v: v/diagonal)

    def solve(rhs, unit=None):
        guess, code = cg(matrix, rhs, M=preconditioner, rtol=2e-14, atol=0.)
        require(code == 0 and np.all(np.isfinite(guess)), "CG convergence")
        require(float(np.max(abs(guess)))*SCALE < 2**61, "quantization range")
        codes = np.rint(guess*SCALE).astype(np.int64)
        require(int(np.max(abs(codes)))*int(rownorm.max()) < 2**62, "integer product range")
        product = integer @ codes
        if unit is None:
            require(np.max(abs(rhs))*SCALE < 2**60, "source range")
            rounded = np.rint(rhs*SCALE).astype(np.int64)
            residual = 8*rounded-product
            radius = interval(int(np.max(abs(residual)))+4, 8*SCALE)
        else:
            residual = -product
            residual[unit] += 8*SCALE
            radius = interval(int(np.max(abs(residual))), 8*SCALE)
        return [interval(v, SCALE)+iv.mpf([-1, 1])*radius for v in codes[endpoints]]

    y = solve(source)
    columns = []
    for j, node in enumerate(endpoints):
        rhs = np.zeros(matrix.shape[0])
        rhs[node] = 1.
        columns.append(solve(rhs, int(node)))
        if (j+1) % 16 == 0:
            print(f"certified {j+1}/{len(endpoints)} endpoint columns", flush=True)
    gram = [[columns[j][i] for j in range(len(endpoints))] for i in range(len(endpoints))]
    return y, gram


def rank_update(y, gram, a, b, sign, coefficient, commit=False):
    """u=e_a+sign*e_b; update A by coefficient*u*u^T."""
    h = y[a]+sign*y[b]
    g = gram[a][a]+sign*(gram[a][b]+gram[b][a])+gram[b][b]
    denominator = 1+coefficient*g
    require(denominator.a > 0, "unresolved positive update denominator")
    alpha = coefficient/denominator
    gain = alpha*h*h
    if not commit:
        # Candidate flip needs just its own two coordinates after first update.
        indices = [a,b]
    else:
        indices = range(len(y))
    v = [gram[i][a]+sign*gram[i][b] for i in indices]
    result_y = [y[i]-alpha*v[k]*h for k,i in enumerate(indices)]
    result_g = [[gram[i][j]-alpha*v[k]*v[l] for l,j in enumerate(indices)]
                for k,i in enumerate(indices)]
    return gain, result_y, result_g


def candidate(y, gram, a, b, weight, flip):
    gain, yy, gg = rank_update(y, gram, a, b, -1, weight)
    if flip:
        other, _, _ = rank_update(yy, gg, 0, 1, 1, -weight)
        gain += other
    return gain


def verify(n, input_path, traces_path):
    iv.dps = 45
    matrix, original_edits, _, _ = fixture(n)
    with np.load(input_path, allow_pickle=False) as data:
        source = data['s'].copy()
        ii, jj, ww, ff = [data[key].copy() for key in ('edit_i','edit_j','edit_w','edit_flip')]
        require(int(data['n']) == n and source.shape == (n,), "input size")
    require(np.all(np.isfinite(source)) and np.max(abs(source)) <= 1, "bounded source")
    require(len(ii) == 32 and len(set(zip(ii,jj))) == 32, "catalog size")
    require([(e.i,e.j,e.w,e.flip) for e in original_edits] == list(zip(ii,jj,ww,ff)), "saved catalog differs")
    with traces_path.open(newline='') as stream:
        traces = list(csv.DictReader(stream))
    require(bool(traces), "missing timing traces")
    paths = sorted(set(tuple(json.loads(row['sequence'])) for row in traces))
    require(all(len(path) == 15 and len(set(path)) == 15 and all(type(k) is int and 0 <= k < 32 for k in path) for path in paths), "15-step traces required")
    endpoints = np.unique(np.concatenate((ii,jj)))
    positions = {int(v):k for k,v in enumerate(endpoints)}
    aa, bb = [positions[int(v)] for v in ii], [positions[int(v)] for v in jj]
    weights = [iv.mpf(float(w)) for w in ww]
    root_y, root_gram = root_enclosures(matrix, source, endpoints)
    output = []
    delta = iv.mpf('0.000000001')
    for path_id,path in enumerate(paths):
        y, gram = root_y, root_gram
        selected = set()
        trajectory = interval(0)
        for step, winner in enumerate(path):
            gains = {k:candidate(y,gram,aa[k],bb[k],weights[k],ff[k]) for k in range(32) if k not in selected}
            win = gains[winner]
            rival_upper = max(value.b for k,value in gains.items() if k != winner)
            margin = win.a-rival_upper
            require(win.a > 0, f"{n} step {step}: positivity unresolved")
            require(margin > -delta.a, f"{n} step {step}: delta optimality unresolved: {margin}")
            gain, y, gram = rank_update(y,gram,aa[winner],bb[winner],-1,weights[winner],True)
            if ff[winner]:
                extra,y,gram = rank_update(y,gram,aa[winner],bb[winner],1,-weights[winner],True)
                gain += extra
            trajectory += gain
            selected.add(winner)
            output.append(dict(n=n,path=path_id,step=step,winner=winner,candidates=len(gains),
                               delta='1e-9',gain_lower=str(win.a),gain_upper=str(win.b),
                               margin_lower=str(margin.a),trajectory_lower=str(trajectory.a),
                               trajectory_upper=str(trajectory.b),status='interval_certified'))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, nargs='+', default=[10000,50000,100000,500000])
    parser.add_argument('--output', type=Path, default=ROOT/'data'/'scale_bounded_verified.csv')
    args = parser.parse_args()
    rows = []
    for n in args.n:
        rows.extend(verify(n,ROOT/'data'/f'scale_bounded_input_{n}.npz',ROOT/'data'/f'scale_bounded_{n}.csv'))
        print(f'{n}: all distinct published 15-step paths certified', flush=True)
    with args.output.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
