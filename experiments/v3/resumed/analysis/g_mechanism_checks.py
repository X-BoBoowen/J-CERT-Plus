"""Run changed G checks and the three fixed empty-graph diagnostic sizes."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import os
import sys
import time
import unittest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    runtime = args.runtime.resolve()
    sys.path.insert(0, str(runtime))
    os.environ['V3_ORACLE'] = str(args.oracle.resolve())
    from scipy import sparse
    import numpy as np
    from src.a2_session import A2Session
    from src.common import DEN
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_v3_g')
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tested.wasSuccessful():
        raise SystemExit(1)
    report = dict(scope='constructed mechanism diagnosis, not real-data Q4 timing',
                  tests_run=tested.testsRun, tests_passed=True, seed=None,
                  sizes=[8, 32, 128], runs=[], independent_fixture_checks=[])
    for n in report['sizes']:
        for method in ('H', 'G'):
            start = time.perf_counter()
            W = sparse.csr_matrix((n, n), dtype=np.float64)
            s = np.array([-DEN] + [-63*DEN//64]*(n//2-1) +
                         [63*DEN//64]*(n//2-1) + [DEN], np.int64)
            input_seconds = time.perf_counter()-start
            constructed = time.perf_counter()
            service = A2Session(W, method, hot_size=64, timers=True)
            setup_seconds = time.perf_counter()-constructed
            request_start = time.perf_counter()
            result = service.request(s, budget=1, delta=0, audit=False)
            request_seconds = time.perf_counter()-request_start
            total = time.perf_counter()-start
            decision = result['decisions'][0]
            if result['status'] != 'budget_stop' or decision['winner_id'] != n-1:
                raise AssertionError('wrong constructed decision')
            if not F(decision['low'], result['gain_denominator']) <= F(2, 9) <= F(decision['high'], result['gain_denominator']):
                raise AssertionError('published interval missed physical exact value')
            row = dict(n=n, method=method, logical_candidates=n*(n-1)//2,
                       opinion_sha256=hashlib.sha256(s.tobytes()).hexdigest(),
                       input_construct_seconds=input_seconds, structure_seconds=setup_seconds,
                       request_seconds=request_seconds, total_seconds=total,
                       input_scalar_opinion_entries=n, graph_csr_nnz=0,
                       physical_expected_gain='2/9', result=result)
            report['runs'].append(row)
        checker_start = time.perf_counter()
        opinions = [F(int(x), DEN) for x in s]
        parent = sum((x*x for x in opinions), F(0))
        maximum, winner, count = F(-1), None, 0
        w = F(1, 16)
        for i in range(n):
            for j in range(i+1, n):
                si, sj = opinions[i], opinions[j]
                # Independently invert the direct physical child 2x2 block.
                determinant = (1+w)*(1+w)-w*w
                zi = ((1+w)*si+w*sj)/determinant
                zj = (w*si+(1+w)*sj)/determinant
                child = parent-si*si-sj*sj+si*zi+sj*zj
                gain, cid = parent-child, i*n+j
                if gain > maximum or (gain == maximum and (winner is None or cid < winner)):
                    maximum, winner = gain, cid
                count += 1
        if (maximum, winner) != (F(2, 9), n-1):
            raise AssertionError('independent physical fixture check disagrees')
        report['independent_fixture_checks'].append(dict(n=n, pairs=count,
            maximum=str(maximum), canonical_id=winner, checker_seconds=time.perf_counter()-checker_start,
            generic_linear_rhs=0, scope='exact direct two-coordinate physical child plus unchanged coordinates'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed', tests=tested.testsRun, fixed_diagnostic_runs=len(report['runs']),
                         output=str(args.output))))


if __name__ == '__main__':
    main()
