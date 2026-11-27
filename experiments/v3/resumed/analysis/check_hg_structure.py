"""Independent current-version legality and actual-witness checks, no solves.

Uses the independent V2 checker operator reconstruction and full sparse
conjugacy, never the execution access/score/resolver. Does not establish global
optimality, independent positivity, or containment of all reported intervals.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(x, message):
    if not x:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    w = args.workspace.resolve()
    runtime = w / 'v3_runtime'
    sys.path.insert(0, str(runtime))
    import numpy as np
    from scipy import sparse
    from src.a2_checker import A2CatalogReplay, _TieProofs, _change, _version
    start = time.perf_counter()
    freeze = json.loads((w / 'v3/resumed/protocols/HG_FROZEN.json').read_text())
    report = dict(status='running', workers=0, requests=0, accepted_decisions=0,
                  failed_terminal_rows=0, witnesses=0, unique_relations=[],
                  independent_linear_solves=0, full_decisions_numerically_replayed=0,
                  scope='Independent exact physical parent/edit/version and all actual declared permutation relations. No all-catalog gain/positivity or published-interval enclosure claim.',
                  records=[], source_files={}, raw_files={})
    for path in [Path(__file__).resolve(), runtime / 'src/a2_checker.py', runtime / 'src/checker.py']:
        report['source_files'][str(path)] = sha(path)
    graphs, opinions, unique = {}, {}, {}
    try:
        for job in freeze['jobs']:
            graph, stream = job['graph'], job['stream']
            raw = w / freeze['current_raw_dir'] / job['output_file']
            trace = json.loads(raw.read_text())
            report['raw_files'][str(raw)] = sha(raw)
            if graph not in graphs:
                graph_path = runtime / freeze['input_graphs'][graph]['weighted_path']
                need(sha(graph_path) == freeze['input_graphs'][graph]['weighted_sha256'], 'graph hash')
                graphs[graph] = A2CatalogReplay(sparse.load_npz(graph_path))
            key = (graph, stream)
            if key not in opinions:
                fixture = next(f for f in freeze['input_fixtures'] if (f['graph'], f['stream']) == key)
                with np.load(runtime / 'v2' / fixture['file'], allow_pickle=False) as z:
                    need(z.files == ['opinions'], 'original opinion archive layout')
                    opinions[key] = z['opinions'].copy()
            replay = graphs[graph]
            root = replay.root_version_id(trace['graph_hash'])
            need(root == trace['root_version_id'], 'root binding')
            report['workers'] += 1
            for request in trace['requests']:
                q = request['query_position']
                s = opinions[key][q-1]
                need(hashlib.sha256(np.ascontiguousarray(s).tobytes()).hexdigest() == request['opinion_hash'], 'opinion binding')
                A, chosen, parent = replay.A, set(), root
                row_report = dict(file=str(raw), q=q, witnesses=0, accepts=0, terminal=request['status'])
                for epoch, decision in enumerate(request['decisions'], 1):
                    need(decision['epoch'] == epoch and decision['parent_version_id'] == parent, 'parent/epoch binding')
                    proofs = _TieProofs(replay, A, s, chosen, parent)
                    proofs.read(decision)
                    for witness in decision.get('comparison_resolution', {}).get('witnesses', []):
                        identity = hashlib.sha256(json.dumps(witness, sort_keys=True).encode()).hexdigest()
                        unique.setdefault(identity, dict(witness=witness, occurrences=0))['occurrences'] += 1
                        row_report['witnesses'] += 1
                        report['witnesses'] += 1
                    if decision['status'] == 'accepted':
                        candidate = proofs.candidate(decision['winner_id'])
                        i, j, d, flip = candidate
                        need((decision['i'], decision['j'], decision['d'], bool(decision['flip'])) == (i, j, d, flip), 'edit binding')
                        need(type(decision['low']) is int and type(decision['high']) is int
                             and 0 < decision['low'] <= decision['high'], 'declared positive interval syntax')
                        parent = _version(parent, decision['winner_id'])
                        need(decision['current_version_id'] == parent, 'child version binding')
                        A = _change(A, candidate)
                        chosen.add(decision['winner_id'])
                        row_report['accepts'] += 1
                        report['accepted_decisions'] += 1
                    else:
                        need(decision['current_version_id'] == parent, 'failed/stop parent identity')
                        if decision['status'] == 'unresolved':
                            report['failed_terminal_rows'] += 1
                need(len(chosen) == request['commits'] and parent == request['current_version_id'], 'request terminal binding')
                report['requests'] += 1
                report['records'].append(row_report)
        report['unique_relations'] = [dict(sha256=k, **v) for k, v in unique.items()]
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report['checker_seconds'] = time.perf_counter()-start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ('status','workers','requests','accepted_decisions','witnesses','checker_seconds')}))


if __name__ == '__main__':
    main()
