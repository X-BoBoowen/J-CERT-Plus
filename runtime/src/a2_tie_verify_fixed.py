"""Independent complete-catalog replay of the ten preselected diagnostic outputs.

This is validation work, not an algorithm timing experiment. The inputs are the
original fixed failures; the checker ignores execution candidate/frontier lists.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from .a2_checker import A2CatalogReplay, worker_request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    inventory = json.loads((ROOT/'reports/TIE_FIXED_STATE_INVENTORY.json').read_text())
    rows = sorted((ROOT/'raw/a2_tie_fixed_replay').glob('[0-9][0-9]_*.json'))
    if len(rows) != 10:
        raise ValueError('exactly ten fixed request outputs required')
    expected = {row['state_id'] for row in inventory['states']}
    present = {json.loads(path.read_text())['state']['state_id'] for path in rows}
    if present != expected:
        raise ValueError('fixed-state roster changed')
    report = dict(study='independent full-catalog validation of fixed failure revisions',
                  algorithm_timing=False, original_confirmation_status='failed',
                  checker_sha256=sha(ROOT/'src/a2_checker.py'),
                  numerical_checker_sha256=sha(ROOT/'src/checker.py'),
                  inventory_sha256=sha(ROOT/'reports/TIE_FIXED_STATE_INVENTORY.json'),
                  selected_states=sorted(expected), cases=[], status='running')
    start = perf_counter()
    for path in rows:
        raw = json.loads(path.read_text())
        state, request = raw['state'], raw['request']
        graph, fixture = ROOT/state['graph_file'], ROOT/state['fixture_file']
        if sha(graph) != state['graph_sha256'] or sha(fixture) != state['fixture_sha256']:
            raise ValueError('fixed graph or opinion bytes changed')
        with np.load(fixture, allow_pickle=False) as loaded:
            opinion = loaded['opinions'][state['opinion_row_zero_based']].copy()
        replay = A2CatalogReplay(sparse.load_npz(graph))
        proof = worker_request(replay, request, opinion, state['graph_sha256'])
        row = dict(state_id=state['state_id'], raw_file=str(path.relative_to(ROOT)),
                   raw_sha256=sha(path), graph_sha256=state['graph_sha256'],
                   fixture_sha256=state['fixture_sha256'], proof=proof)
        report['cases'].append(row)
        print(json.dumps(dict(state_id=state['state_id'],
              certified=proof['full_catalog_decisions_certified'],
              accepts=proof['accepts'], checked_candidates=proof['checked_candidates'],
              witnesses=proof['structural_tie_witnesses_verified'])), flush=True)
        # Preserve independently completed work even if a later case fails.
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(output)
    report['status'] = ('passed' if all(row['proof']['full_catalog_decisions_certified']
                                       for row in report['cases']) else 'verification_gap')
    report['wall_seconds'] = perf_counter()-start
    report['checked_candidate_decision_rows'] = sum(row['proof']['checked_candidates']
                                                    for row in report['cases'])
    report['decisions'] = sum(len(row['proof']['decision_checks']) for row in report['cases'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT/'reports/TIE_FIXED_INDEPENDENT_VERIFICATION.json')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run(args.output)
    raise SystemExit(0 if result['status'] == 'passed' else 1)
