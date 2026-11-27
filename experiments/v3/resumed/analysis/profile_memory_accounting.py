"""One frozen representative-flow profile of nonoverlapping accounting calls.

No instrumentation is disabled, no access policy is changed and these times
are diagnostic, not additional formal repeats. Both methods use fresh sessions.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    w = a.workspace.resolve()
    runtime = w / 'v3_runtime'
    protocol = json.loads(a.protocol.read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != protocol['script_sha256']:
        raise ValueError('profile script changed')
    sys.path.insert(0, str(runtime))
    import numpy as np
    from scipy import sparse
    from src.a2_session import A2Session, lp
    from src.a2_access import Access
    from src.a2_catalog import AllCatalog
    from src.g_access import GAccess
    from src.access_bench import load_fixture
    freeze = json.loads((w / protocol['freeze']).read_text())
    for file, digest in freeze['source_files_sha256'].items():
        if hashlib.sha256((runtime/file).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen runtime source changed: '+file)
    report = dict(status='running', protocol=protocol, rows=[],
                  scope='one representative development stream; timed reporting wrappers retain all work; no new formal repeat or instrumentation-free speed ratio')
    state = dict(method=None, rows=None, depth=0)
    patches = []
    def install(owner, attribute, label):
        original = getattr(owner, attribute)
        def measured(*args, **kwargs):
            if state['depth']:
                raise RuntimeError('profile accounting scopes unexpectedly overlap')
            state['depth'] += 1
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter()-start
                state['depth'] -= 1
                slot = state['rows'][label]
                slot['seconds'] += elapsed
                slot['calls'] += 1
        setattr(owner, attribute, measured)
        patches.append((owner, attribute, original))
    install(Access, 'observe_frontier', 'H_frontier_deep_memory_report')
    install(GAccess, '_observe_memory', 'G_frontier_deep_memory_report')
    install(AllCatalog, 'memory', 'catalog_deep_memory_report')
    install(lp, 'store_bytes', 'cache_and_output_deep_memory_report')
    try:
        fixture = next(f for f in freeze['input_fixtures'] if
                       (f['graph'], f['stream']) == (protocol['graph'], protocol['stream']))
        opinions = load_fixture(fixture, runtime/'v2')
        spec = freeze['input_graphs'][protocol['graph']]
        for method in protocol['method_order']:
            state['method'] = method
            state['rows'] = defaultdict(lambda:dict(seconds=0., calls=0))
            start = time.perf_counter()
            W = sparse.load_npz(runtime/spec['weighted_path'])
            service = A2Session(W, method, hot_size=64, timers=True,
                                graph_hash=spec['weighted_sha256'], graph_name=protocol['graph'])
            setup = time.perf_counter()-start
            requests = []
            for position, s in enumerate(opinions, 1):
                begun = time.perf_counter()
                request = service.request(s, budget=15, delta=0, audit=False)
                elapsed = time.perf_counter()-begun
                request['profile_request_wall_seconds'] = elapsed
                requests.append(request)
            total = setup + sum(r['profile_request_wall_seconds'] for r in requests)
            memory = sum(v['seconds'] for v in state['rows'].values())
            original_job = next(j for j in freeze['jobs'] if
                (j['graph'], j['stream'], j['method']) == (protocol['graph'], protocol['stream'], method))
            original = json.loads((w / freeze['current_raw_dir'] / original_job['output_file']).read_text())
            for r, old in zip(requests, original['requests']):
                observed = [(d['winner_id'],d['parent_version_id'],d['current_version_id']) for d in r['decisions'] if d['status']=='accepted']
                expected = [(d['winner_id'],d['parent_version_id'],d['current_version_id']) for d in old['decisions'] if d['status']=='accepted']
                if observed != expected or r['status'] != old['status']:
                    raise ValueError('profile changed the actual decision trajectory')
            report['rows'].append(dict(method=method, graph=protocol['graph'], stream=protocol['stream'],
                structure_seconds=setup, inclusive_diagnostic_service_seconds=total,
                separately_timed_accounting=state['rows'], accounting_sum_seconds=memory,
                diagnostic_service_residual_after_accounting_subtraction=total-memory,
                residual_scope='arithmetic residual of this instrumented diagnostic, not an observed uninstrumented execution',
                accepted_trace_matches_formal=True, requests=requests))
        report['status'] = 'completed'
    finally:
        for owner, attribute, original in reversed(patches):
            setattr(owner, attribute, original)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','scope')}))


if __name__ == '__main__':
    main()
