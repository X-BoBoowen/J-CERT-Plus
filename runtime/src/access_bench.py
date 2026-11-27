"""Freeze input blocks and run the preregistered access controls serially."""
import argparse
from decimal import Decimal, ROUND_FLOOR
import fcntl
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import tempfile
import time
import traceback

from .anchor_service_bench import save, sha

ROOT = Path(__file__).resolve().parents[1]
GRAPHS = ('Alpha', 'OTC')
STREAMS = ('slow', 'independent', 'signed-neighbor')
CONTROLS = ('rebuild-always', 'retain-always')
GUARD = 'budget-guard'
PREFIXES = (1, 2, 4, 8, 16)
LIMITS = dict(worker_seconds=90, rss_mib=2048, batch_seconds=1800,
              rss_sample_seconds=0.025)
TERMINAL_REQUESTS = ('budget_stop', 'tolerance_stop')
THREAD_ENV = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def ordered_subset(values, permitted):
    return (isinstance(values, list) and bool(values)
            and all(value in permitted for value in values)
            and len(set(values)) == len(values))


def validate_protocol(protocol):
    if not isinstance(protocol, dict):
        raise ValueError('expected a JSON protocol object')
    cfg = dict(id='access-bench-v1', repeat_count=1, cpu=0,
               prefixes=list(PREFIXES)) | protocol
    if (not isinstance(cfg['id'], str) or
            not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', cfg['id'])):
        raise ValueError('invalid protocol ID')
    if not ordered_subset(cfg.get('graphs'), GRAPHS) or not ordered_subset(cfg.get('streams'), STREAMS):
        raise ValueError('invalid registered graph or stream order')
    seeds = cfg.get('seeds')
    if (not isinstance(seeds, list) or not seeds or
            any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError('seeds must be distinct nonnegative integers')
    if (type(cfg.get('requests')) is not int or cfg['requests'] != 16 or
            type(cfg.get('budget')) is not int or cfg['budget'] != 15 or
            cfg['prefixes'] != list(PREFIXES) or
            type(cfg['cpu']) is not int or cfg['cpu'] != 0 or
            type(cfg['repeat_count']) is not int or cfg['repeat_count'] < 1):
        raise ValueError('expected Q16, edit budget 15, registered prefixes, CPU 0 and positive repeat_count')
    coefficients = cfg.get('coefficients')
    if (not isinstance(coefficients, list) or len(coefficients) > 3 or
            any(not finite_positive(value) for value in coefficients) or
            len(set(coefficients)) != len(coefficients)):
        raise ValueError('expected at most three distinct, positive budget coefficients')
    methods = cfg.get('methods')
    if not isinstance(methods, list) or not methods:
        raise ValueError('registration requires methods')
    labels = set()
    controls = []
    guards = []
    for method in methods:
        if not isinstance(method, dict):
            raise ValueError('expected a method object')
        label, arm = method.get('label'), method.get('arm')
        if (not isinstance(label, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', label)
                or label in labels):
            raise ValueError('method labels must be distinct safe filename components')
        labels.add(label)
        if arm in CONTROLS and 'coefficient' not in method:
            controls.append(arm)
        elif arm == GUARD and finite_positive(method.get('coefficient')):
            guards.append(method['coefficient'])
        else:
            raise ValueError('unknown arm or invalid coefficient for method')
    if set(controls) != set(CONTROLS) or len(controls) != len(CONTROLS):
        raise ValueError('register one instance of each rebuild and retain control')
    if len(guards) != len(coefficients) or set(guards) != set(coefficients):
        raise ValueError('guard methods must enumerate the registered coefficients once')
    custom_limits = cfg.get('limits', {})
    if not isinstance(custom_limits, dict) or custom_limits.keys() - LIMITS.keys():
        raise ValueError('unknown resource limit')
    cfg['limits'] = LIMITS | custom_limits
    if any(not finite_positive(value) for value in cfg['limits'].values()):
        raise ValueError('resource limits must be positive and finite')
    return cfg


def read_protocol(path):
    contents = path.read_bytes()
    return validate_protocol(json.loads(contents)), hashlib.sha256(contents).hexdigest()


def jobs(cfg):
    cells = list(product(cfg['graphs'], cfg['streams'], cfg['seeds']))
    methods = cfg['methods']
    for repeat in range(cfg['repeat_count']):
        for cell_position, (graph, stream, seed) in enumerate(cells):
            pivot = (repeat + cell_position) % len(methods)
            for method in methods[pivot:] + methods[:pivot]:
                job = dict(graph=graph, stream=stream, seed=seed, repeat_id=repeat,
                           arm=method['arm'], label=method['label'])
                if 'coefficient' in method:
                    job['coefficient'] = method['coefficient']
                yield job


def source_hashes():
    sources = ['access_bench', 'anchor_session', 'anchor_service_bench', 'session',
               'weighted_adapter', 'common', 'holdout_stream', 'confirmation_replay', 'checker']
    paths = [ROOT / 'src' / (name + '.py') for name in sources]
    coverage = ROOT / 'src/coverage_budget.py'
    if coverage.exists():
        paths.append(coverage)
    evidence = ROOT / 'baseline_r15/experimental_records/evidence_supplement_20261124'
    paths.append(evidence / 'diagnostic/common.py')
    paths.extend(sorted((evidence / 'source/legacy_v6/tools').glob('*.py')))
    paths.extend(evidence / 'diagnostic/build' / name for name in ('screen.so', 'index.so'))
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def matrix_hashes(matrix, expected_shape):
    import numpy as np

    if (matrix.dtype != np.dtype('<i8') or tuple(matrix.shape) != tuple(expected_shape) or
            not matrix.flags.c_contiguous or np.any(matrix < -65536) or np.any(matrix > 65536)):
        raise ValueError('invalid int64 opinion matrix layout, dimensions or range')
    return (hashlib.sha256(matrix.tobytes()).hexdigest(),
            [hashlib.sha256(row.tobytes()).hexdigest() for row in matrix])


def load_fixture(record, manifest_directory):
    import numpy as np

    path = manifest_directory / record['file']
    if sha(path) != record['npz_sha256']:
        raise ValueError('fixture file digest mismatch')
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ['opinions']:
            raise ValueError('unexpected fixture archive members')
        matrix = archive['opinions']
    observed = matrix_hashes(matrix, record['shape'])
    if observed != (record['matrix_sha256'], record['opinion_hashes']):
        raise ValueError('fixture matrix or row digest mismatch')
    return matrix


def prepare(protocol_path, manifest_path, fixture_directory=None):
    import numpy as np
    from scipy.sparse import load_npz
    from threadpoolctl import threadpool_limits

    from .common import opinions
    from .confirmation_replay import reference_stream
    from .holdout_stream import signed_neighbor_opinions

    cfg, protocol_hash = read_protocol(protocol_path)
    sources = source_hashes()
    directory = fixture_directory or ROOT / 'data/access' / cfg['id']
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() or directory.exists():
        raise FileExistsError('refusing to replace a fixture freeze')
    fixtures = []
    graphs = {}
    with tempfile.TemporaryDirectory(dir=directory.parent) as temporary, threadpool_limits(limits=1):
        stage = Path(temporary) / 'fixtures'
        stage.mkdir()
        for graph in cfg['graphs']:
            paths = {kind: ROOT / 'data/projected' / f'{graph}_{kind}.npz'
                     for kind in ('unit', 'weighted')}
            hashes = {kind: sha(path) for kind, path in paths.items()}
            metadata = json.loads((ROOT / 'data/projected' / f'{graph}_metadata.json').read_text())
            if any(hashes[kind] != metadata[kind + '_sha256'] for kind in paths):
                raise ValueError(f'projected graph checksum mismatch for {graph}')
            unit = load_npz(paths['unit'])
            weighted = load_npz(paths['weighted'])
            if (unit.format != 'csr' or not unit.has_canonical_format or
                    (unit != unit.T).nnz or unit.diagonal().any() or
                    not np.all(np.abs(unit.data) == 1) or (weighted.sign() != unit).nnz):
                raise ValueError(f'invalid projected sign support for {graph}')
            n = unit.shape[0]
            graphs[graph] = dict(weighted_path=str(paths['weighted'].relative_to(ROOT)),
                                 weighted_sha256=hashes['weighted'],
                                 unit_path=str(paths['unit'].relative_to(ROOT)),
                                 unit_sha256=hashes['unit'], n=n)
            for stream, seed in product(cfg['streams'], cfg['seeds']):
                vecs = (signed_neighbor_opinions(unit, cfg['requests'], seed)
                        if stream == 'signed-neighbor' else
                        opinions(n, stream, seed, cfg['requests']))
                matrix = np.stack(vecs)
                reference = reference_stream(unit, stream, seed, cfg['requests'])
                if not np.array_equal(matrix, reference):
                    raise ValueError(f'independent input regeneration mismatch: {graph}/{stream}/{seed}')
                matrix_hash, row_hashes = matrix_hashes(matrix, (cfg['requests'], n))
                filename = f'{graph}_{stream}_{seed}.npz'
                np.savez(stage / filename, opinions=matrix)
                fixtures.append(dict(
                    graph=graph, stream=stream, seed=seed,
                    fixture_id=f"{graph}/{stream}/{seed}/Q{cfg['requests']}",
                    file=os.path.relpath(directory / filename, manifest_path.parent),
                    npz_sha256=sha(stage / filename), matrix_sha256=matrix_hash,
                    opinion_hashes=row_hashes, shape=list(matrix.shape),
                    independently_matched_rows=cfg['requests'],
                ))
        if sha(protocol_path) != protocol_hash or source_hashes() != sources:
            raise ValueError('protocol or source changed during fixture preparation')
        if manifest_path.exists() or directory.exists():
            raise FileExistsError('fixture freeze was concurrently created')
        stage.rename(directory)
    manifest = dict(schema_version=1, protocol_id=cfg['id'], protocol_sha256=protocol_hash,
                    source_files_sha256=sources, graphs=graphs, fixtures=fixtures,
                    verified=True, independent_opinion_rows=sum(r['shape'][0] for r in fixtures))
    save(manifest_path, manifest)
    return manifest


def load_freeze(protocol_path, manifest_path):
    cfg, protocol_hash = read_protocol(protocol_path)
    contents = manifest_path.read_bytes()
    manifest = json.loads(contents)
    if (manifest.get('schema_version') != 1 or manifest.get('verified') is not True or
            manifest.get('protocol_id') != cfg['id'] or manifest.get('protocol_sha256') != protocol_hash or
            manifest.get('source_files_sha256') != source_hashes()):
        raise ValueError('prepared protocol, source freeze or independent input verification mismatch')
    grid = set(product(cfg['graphs'], cfg['streams'], cfg['seeds']))
    records = {(r['graph'], r['stream'], r['seed']): r for r in manifest['fixtures']}
    if (set(records) != grid or len(manifest['fixtures']) != len(grid) or
            set(manifest['graphs']) != set(cfg['graphs']) or
            manifest['independent_opinion_rows'] != len(grid) * cfg['requests']):
        raise ValueError('prepared fixture grid does not match registration')
    for (graph, stream, seed), row in records.items():
        if (row['fixture_id'] != f"{graph}/{stream}/{seed}/Q{cfg['requests']}" or
                row['shape'] != [cfg['requests'], manifest['graphs'][graph]['n']] or
                len(row['opinion_hashes']) != cfg['requests'] or
                row['independently_matched_rows'] != cfg['requests']):
            raise ValueError('prepared fixture identity or independent row coverage mismatch')
    return cfg, manifest, records, hashlib.sha256(contents).hexdigest()


def catalog_visits(coefficient, catalog_size):
    return int((Decimal(str(coefficient)) * catalog_size).to_integral_value(rounding=ROUND_FLOOR))


def worker(protocol_path, manifest_path, output, position, expected_manifest_hash):
    if output.exists():
        raise FileExistsError(output)
    for key in THREAD_ENV:
        os.environ[key] = '1'
    started = time.monotonic()
    trace = dict(status='starting', failure_reason=None, schedule_position=position, requests=[], prefixes={})

    def checkpoint():
        trace['worker_observed_wall_seconds'] = time.monotonic() - started
        trace['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        save(output, trace)

    try:
        cfg, manifest, records, manifest_hash = load_freeze(protocol_path, manifest_path)
        if manifest_hash != expected_manifest_hash:
            raise ValueError('worker manifest differs from the batch freeze')
        schedule = list(jobs(cfg))
        if not 0 <= position < len(schedule):
            raise ValueError('unregistered schedule position')
        job = schedule[position]
        graph, stream, seed = (job[k] for k in ('graph', 'stream', 'seed'))
        trace.update(job, protocol_id=cfg['id'], protocol_sha256=manifest['protocol_sha256'],
                     source_files_sha256=manifest['source_files_sha256'],
                     input_manifest_sha256=manifest_hash, requests_planned=cfg['requests'],
                     budget=cfg['budget'], method='D', weight_mode='weighted', delta=0,
                     hot_size=64, audit=False, cpu=cfg['cpu'],
                     stream_id=f'{graph}/weighted/{stream}/{seed}')
        os.sched_setaffinity(0, {cfg['cpu']})
        trace['affinity'] = sorted(os.sched_getaffinity(0))
        checkpoint()

        from scipy.sparse import load_npz
        from threadpoolctl import threadpool_info, threadpool_limits

        from .anchor_session import AnchorSession
        from .common import Ledger

        with threadpool_limits(limits=1):
            trace['threadpools'] = threadpool_info()
            start = time.monotonic()
            spec = manifest['graphs'][graph]
            graph_path = ROOT / spec['weighted_path']
            gh = sha(graph_path)
            record = records[graph, stream, seed]
            if gh != spec['weighted_sha256']:
                raise ValueError('weighted graph digest mismatch')
            matrix = load_fixture(record, manifest_path.parent)
            trace.update(graph_hash=gh, fixture_id=record['fixture_id'],
                         fixture_npz_sha256=record['npz_sha256'], fixture_matrix_sha256=record['matrix_sha256'],
                         fixture_and_graph_validate_seconds=time.monotonic() - start)
            ledger = Ledger(True)
            with ledger.phase('graph_load'):
                W = load_npz(graph_path)
            if W.shape != (spec['n'], spec['n']):
                raise ValueError('weighted graph dimensions do not match the freeze')
            sess = AnchorSession(W, job['arm'], graph_hash=gh, ledger=ledger)
            del W
            raw_limit = None
            if job['arm'] == GUARD:
                raw_limit = catalog_visits(job['coefficient'], sess.m)
                sess.bridge.visit_budget = raw_limit
            trace.update(status='prepared', n=sess.W.shape[0], catalog_size=sess.m,
                         operator_hash=sess.operator_hash, catalog_hash=sess.catalog_hash,
                         root_version_id=sess.root_id, visit_budget=raw_limit,
                         structural_seconds=sess.setup_seconds, structural_phases=sess.setup_phases,
                         structural_weight_grouping_bytes=sess.weight_group_bytes,
                         service_accounted_seconds=sess.setup_seconds)
            cumulative = sess.setup_seconds
            checkpoint()
            for q, opinion in enumerate(matrix, 1):
                start = time.monotonic()
                req = sess.request(opinion, budget=cfg['budget'])
                elapsed = time.monotonic() - start
                overhead = elapsed - req['request_seconds']
                req['phases']['reset_restore'] += overhead
                req['reset_opinion_seconds'] += overhead
                req.update(request_seconds=elapsed, query_position=q, stream_id=trace['stream_id'])
                trace['requests'].append(req)
                if (req['opinion_hash'] != record['opinion_hashes'][q - 1] or
                        abs(sum(req['phases'].values()) - elapsed) > 1e-7 or
                        abs(req['reset_opinion_seconds'] + req['online_seconds'] - elapsed) > 1e-7):
                    raise ValueError('request opinion or accounting mismatch')
                cumulative += elapsed
                trace['service_accounted_seconds'] = cumulative
                if q in cfg['prefixes']:
                    trace['prefixes'][str(q)] = cumulative
                trace['status'] = 'in_progress'
                checkpoint()
            trace['status'] = ('completed' if all(req['status'] in TERMINAL_REQUESTS for req in trace['requests'])
                               else 'completed_with_failures')
    except Exception as error:
        trace.update(status='memory_error' if isinstance(error, MemoryError) else 'error',
                     failure_reason=repr(error), traceback=traceback.format_exc())
    finally:
        checkpoint()
    return trace


def supervise(command, logfile, worker_deadline, batch_deadline, limits):
    import psutil

    started = time.monotonic()
    with logfile.open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        observed = psutil.Process(child.pid)
        peak = 0
        stop = None
        try:
            while child.poll() is None:
                try:
                    peak = max(peak, observed.memory_info().rss)
                except psutil.NoSuchProcess:
                    break
                now = time.monotonic()
                if peak > limits['rss_mib'] * 1024**2:
                    stop = 'rss_cap'
                elif now >= batch_deadline:
                    stop = 'batch_cap'
                elif now >= worker_deadline:
                    stop = 'timeout'
                if stop:
                    child.kill()
                    break
                time.sleep(min(limits['rss_sample_seconds'], worker_deadline - now, batch_deadline - now))
        finally:
            if child.poll() is None and sys.exc_info()[0] is not None:
                child.kill()
            returncode = child.wait()
    return returncode, time.monotonic() - started, peak, stop


def batch(protocol_path, manifest_path, output):
    cfg, manifest, _, manifest_hash = load_freeze(protocol_path, manifest_path)
    limits = cfg['limits']
    lock_path = ROOT / 'logs/.measurement.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError('refusing to replace an existing batch')
        schedule = list(jobs(cfg))
        save(output / 'FROZEN_JOBS.json', dict(jobs=schedule, cpu=cfg['cpu'], limits=limits,
             protocol_sha256=manifest['protocol_sha256'], input_manifest_sha256=manifest_hash,
             source_files_sha256=manifest['source_files_sha256']))
        started = time.monotonic()
        batch_deadline = started + limits['batch_seconds']
        results = dict(jobs=[], limits=limits, batch_wall_seconds=0., status='in_progress')
        for pos, job in enumerate(schedule):
            filename = f"{pos:03d}_{job['graph']}_{job['stream']}_{job['seed']}_{job['label']}.json"
            path = output / filename
            row = dict(position=pos, job=job, result=filename, status='batch_cap_not_started',
                       returncode=None, process_wall_seconds=None,
                       external_sampled_peak_rss_bytes=None, completed_requests=0)
            now = time.monotonic()
            if now < batch_deadline:
                command = [
                    sys.executable, '-m', 'src.access_bench', 'worker',
                    '--protocol', str(protocol_path.resolve()), '--manifest', str(manifest_path.resolve()),
                    '--output', str(path.resolve()), '--schedule-position', str(pos),
                    '--manifest-sha256', manifest_hash,
                ]
                try:
                    code, wall, peak, stop = supervise(command, path.with_suffix('.log'),
                         now + limits['worker_seconds'], batch_deadline, limits)
                    partial = json.loads(path.read_text()) if path.exists() else {}
                    if partial.get('process_peak_rss_bytes', 0) > limits['rss_mib'] * 1024**2:
                        stop = stop or 'rss_cap'
                    status = partial.get('status', 'exit_before_checkpoint')
                    if status in ('starting', 'prepared', 'in_progress') or (code and status == 'completed'):
                        status = 'worker_exit'
                    row.update(status=stop or status, returncode=code, process_wall_seconds=wall,
                               external_sampled_peak_rss_bytes=peak,
                               completed_requests=len(partial.get('requests', [])))
                    if stop or status in ('exit_before_checkpoint', 'worker_exit'):
                        partial.update(job, status=row['status'], last_worker_status=partial.get('status'),
                                       schedule_position=pos, failure_reason=stop or f'worker exited with {code}')
                        partial.setdefault('requests', [])
                        partial.setdefault('prefixes', {})
                        save(path, partial)
                except OSError as error:
                    row.update(status='process_launch_error', failure_reason=repr(error))
                    save(path, dict(job, status=row['status'], schedule_position=pos,
                                    failure_reason=repr(error), requests=[], prefixes={}))
            else:
                save(path, dict(job, status=row['status'], schedule_position=pos,
                                failure_reason='registered batch cap', requests=[], prefixes={}))
            results['jobs'].append(row)
            results['batch_wall_seconds'] = time.monotonic() - started
            save(output / 'SCHEDULE_RESULTS.json', results)
            print(f"{pos + 1}/{len(schedule)} {row['status']} {filename}", flush=True)
        results['status'] = ('completed' if all(r['status'] == 'completed' for r in results['jobs'])
                             else 'completed_with_failures')
        save(output / 'SCHEDULE_RESULTS.json', results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    for mode in ('prepare', 'batch', 'worker'):
        cmd = modes.add_parser(mode)
        cmd.add_argument('--protocol', required=True, type=Path)
        cmd.add_argument('--manifest', required=True, type=Path)
        if mode == 'prepare':
            cmd.add_argument('--fixture-dir', type=Path)
        else:
            cmd.add_argument('--output', required=True, type=Path)
        if mode == 'worker':
            cmd.add_argument('--schedule-position', required=True, type=int)
            cmd.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepared = prepare(args.protocol, args.manifest, args.fixture_dir)
        print(f"{len(prepared['fixtures'])} fixtures, {prepared['independent_opinion_rows']} independently regenerated rows")
        return 0
    if args.mode == 'batch':
        result = batch(args.protocol, args.manifest, args.output)
    else:
        result = worker(args.protocol, args.manifest, args.output, args.schedule_position,
                        args.manifest_sha256)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
