"""Frozen paired H/G worker; no V2 freeze bypass and no new input generation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def environment():
    return dict(python=sys.version, platform=platform.platform(),
                packages={name: importlib.metadata.version(name) for name in
                          ('numpy', 'scipy', 'psutil', 'threadpoolctl', 'sympy')})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--job', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen = json.loads(args.freeze.read_text())
    job = frozen['jobs'][args.job]
    if job['position'] != args.job or job['method'] not in ('H', 'G'):
        raise ValueError('unregistered method or job')
    from .measurement_policy import worker_policy
    measurement_policy=worker_policy(frozen,job)
    started = time.perf_counter()
    trace = dict(job=job, status='starting', requests=[], prefixes={},
                 freeze_sha256=sha(args.freeze), declared_runtime_commit=frozen['runtime_commit'],
                 active_phase='binding', metering=dict(solve_calls=0, scalar_rhs=0, solve_seconds=0.),
                 measurement_policy=measurement_policy,memory_observation_schema=measurement_policy,
                 run_kind=frozen['run_kind'],
                 resource_observation_scope='ru_maxrss read at each checkpoint before JSON serialization; '
                    'last checkpoint does not guarantee inclusion of its own serialization peak; '
                    'supervisor RSS/VMS are separate 100ms sampled maxima')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def checkpoint():
        trace['process_wall_seconds'] = time.perf_counter()-started
        trace['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        tmp = args.output.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(trace, allow_nan=False)+'\n')
        os.replace(tmp, args.output)
    checkpoint()
    try:
        observed_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        trace['actual_runtime_commit'] = observed_commit
        if observed_commit != frozen['runtime_commit']:
            raise ValueError('runtime commit differs from freeze')
        source_digest = hashlib.sha256(json.dumps(frozen['source_files_sha256'], sort_keys=True).encode()).hexdigest()
        if source_digest != frozen['source_digest']:
            raise ValueError('frozen source digest does not match its member map')
        if environment() != frozen['environment']:
            raise ValueError('environment differs from current V3 freeze')
        for filename, digest in frozen['source_files_sha256'].items():
            if sha(ROOT / filename) != digest:
                raise ValueError('changed frozen source: '+filename)
        if sha(ROOT / 'v2/INPUTS.json') != frozen['v2_input_manifest_sha256']:
            raise ValueError('changed V2 input manifest')
        limits = frozen['resources']
        if resource.getrlimit(resource.RLIMIT_AS)[0] != limits['memory_bytes']:
            raise ValueError('hard memory limit differs from paired protocol')
        if sorted(os.sched_getaffinity(0)) != [limits['cpu']]:
            raise ValueError('affinity differs from paired protocol')
        manifest = json.loads((ROOT / 'v2/INPUTS.json').read_text())
        spec = manifest['graphs'][job['graph']]
        fixture = next(f for f in manifest['fixtures']
                       if (f['graph'], f['stream']) == (job['graph'], job['stream']))
        from scipy.sparse import load_npz
        from threadpoolctl import threadpool_info, threadpool_limits
        from .a2_session import A2Session, lp
        from .a2_bench import _check_seconds
        from .access_bench import load_fixture
        trace['binding_and_import_seconds'] = time.perf_counter()-started
        input_start = time.perf_counter()
        opinions = load_fixture(fixture, ROOT / 'v2')
        graph_path = ROOT / spec['weighted_path']
        if sha(graph_path) != spec['weighted_sha256']:
            raise ValueError('graph file hash differs from retained input')
        trace['opinion_and_graph_binding_seconds'] = time.perf_counter()-input_start
        trace.update(graph_hash=spec['weighted_sha256'], fixture=fixture,
                     graph=job['graph'], stream=job['stream'], method=job['method'], repeat_id=job['repeat_id'],
                     environment=environment(), threadpools=threadpool_info(),
                     input_manifest_sha256=frozen['v2_input_manifest_sha256'],
                     source_digest=frozen['source_digest'], n=spec['n'], catalog_size=spec['catalog_size'])
        original = lp.UpdatedFactor
        class MeteredFactor(original):
            def __init__(self, A):
                trace['active_phase'] = 'factor_or_precond_init'
                checkpoint()
                start = time.perf_counter()
                try:
                    super().__init__(A)
                finally:
                    trace['metering']['factor_attempt_seconds'] = time.perf_counter()-start
                    checkpoint()
            def solve(self, rhs):
                trace['metering']['solve_calls'] += 1
                trace['metering']['scalar_rhs'] += 1 if rhs.ndim == 1 else rhs.shape[1]
                start = time.perf_counter()
                try:
                    return super().solve(rhs)
                finally:
                    trace['metering']['solve_seconds'] += time.perf_counter()-start
        lp.UpdatedFactor = MeteredFactor
        status_map = dict(budget_stop='completed_budget', tolerance_stop='certified_stop',
                          unresolved='unresolved', memory_error='memory_limit_exceeded',
                          error='runtime/certificate_error')
        with threadpool_limits(limits=1):
            trace['enforced_threadpools'] = threadpool_info()
            trace['active_phase'] = 'graph_read_and_setup'
            checkpoint()
            service_start = time.perf_counter()
            read_start = time.perf_counter()
            W = load_npz(graph_path)
            graph_load = time.perf_counter()-read_start
            service = A2Session(W, job['method'], hot_size=64, timers=True,
                                graph_hash=spec['weighted_sha256'], graph_name=job['graph'],
                                measurement_policy=measurement_policy)
            structural = time.perf_counter()-service_start
            phases = dict(service.setup_phases)
            phases['graph_load'] = graph_load
            phases['other'] += structural-sum(phases.values())
            _check_seconds(phases, structural, 'V3 structural')
            if (service.n, service.m) != (spec['n'], spec['catalog_size']):
                raise ValueError('catalog dimensions changed')
            trace.update(structural_seconds=structural, structural_phases=phases,
                         root_version_id=service.root_id, operator_hash=service.operator_hash,
                         catalog_hash=service.catalog_hash, root_factor_nnz=service.factor.base.nnz,
                         service_accounted_seconds=structural)
            del W
            for q, opinion in enumerate(opinions, 1):
                trace['active_phase'] = f'request_{q}'
                checkpoint()
                start = time.perf_counter()
                request = service.request(opinion, budget=15, delta=0, audit=False)
                elapsed = time.perf_counter()-start
                _check_seconds(request['phases'], request['request_seconds'], 'V3 internal request')
                wrapper = elapsed-request['request_seconds']
                request['phases']['reset_restore'] += wrapper
                request['reset_opinion_seconds'] += wrapper
                request.update(query_position=q, request_seconds=elapsed, wrapper_seconds=wrapper,
                               result_status=status_map.get(request['status'], 'runtime/certificate_error'))
                if request['opinion_hash'] != fixture['opinion_hashes'][q-1]:
                    raise ValueError('opinion binding differs')
                _check_seconds(request['phases'], elapsed, 'V3 full request')
                trace['requests'].append(request)
                trace['service_accounted_seconds'] += elapsed
                trace['prefixes'][str(q)] = trace['service_accounted_seconds']
                trace['status'] = 'in_progress'
                checkpoint()
                if request['result_status'] == 'runtime/certificate_error':
                    break
            states = [r['result_status'] for r in trace['requests']]
            trace['status'] = ('completed_budget' if len(states) == 4 and all(s == 'completed_budget' for s in states)
                               else 'certified_stop' if len(states) == 4 and all(s in ('completed_budget', 'certified_stop') for s in states)
                               else next((s for s in states if s not in ('completed_budget', 'certified_stop')), 'runtime/certificate_error'))
            trace['complete_Q4'] = trace['status'] in ('completed_budget', 'certified_stop') and len(states) == 4
            trace['complete_service_seconds'] = trace['service_accounted_seconds'] if trace['complete_Q4'] else None
            trace['partial_work_seconds'] = None if trace['complete_Q4'] else trace['service_accounted_seconds']
            trace['service_plus_input_binding_seconds'] = trace['service_accounted_seconds'] + trace['opinion_and_graph_binding_seconds']
            trace['timing_scope'] = ('service_accounted_seconds preserves V2 graph-load/setup+all actually executed request costs; '
                                     'opinion/input validation, full process wall and metered solve costs are separate. '
                                     'Metering overlaps the service phases and is never added twice.')
            trace['active_phase'] = 'complete'
    except Exception as exc:
        trace.update(status='memory_limit_exceeded' if isinstance(exc, MemoryError) else 'runtime/certificate_error',
                     complete_Q4=False, complete_service_seconds=None,
                     partial_work_seconds=trace.get('service_accounted_seconds'),
                     partial_cost_scope='known returned-request/setup cost only; an interrupted active request may be absent; use supervisor process wall separately',
                     failure=repr(exc), traceback=traceback.format_exc())
    finally:
        checkpoint()
    print(json.dumps(dict(status=trace['status'], graph=job['graph'], stream=job['stream'],
                          method=job['method'], output=str(args.output))), flush=True)
    return 1 if trace['status'] == 'runtime/certificate_error' else 0


if __name__ == '__main__':
    sys.exit(main())
