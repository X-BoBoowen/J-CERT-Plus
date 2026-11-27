"""Frozen A2 all-catalog inputs and serial bounded D/S/H service experiment.

The scheduler, atomic checkpoints, RSS supervisor, input digests and thread
policy reuse the accepted access benchmark. Only root runs real experiments.
"""
import argparse
import fcntl
import hashlib
from importlib import metadata
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import sys
import tempfile
import time
import traceback

from .access_bench import (THREAD_ENV, TERMINAL_REQUESTS, load_fixture,
                           matrix_hashes, supervise)
from .anchor_service_bench import save, sha

ROOT = Path(__file__).resolve().parents[1]
GRAPHS = ('Alpha', 'OTC')
STREAMS = ('slow', 'independent', 'signed-neighbor')
METHODS = ({'method': 'D', 'label': 'D-all'},
           {'method': 'S', 'label': 'Stream-all'},
           {'method': 'H', 'label': 'Implicit-all'})
LIMITS = dict(worker_seconds=180, rss_mib=2048, batch_seconds=2700,
              rss_sample_seconds=0.025)
DOMAIN = {
    'catalog': 'every root nonedge insertion plus every root negative-edge sign flip; root positive edges excluded',
    'insertion_weight_numerator': 1, 'insertion_weight_denominator': 16,
    'flip_weight': 'natural absolute root edge weight',
    'candidate_id': 'canonical signed int64 e=i*n+j, 0<=i<j<n',
    'catalog_size': 'n*(n-1)//2 - root_positive_edge_count',
    'reset': 'every opinion starts at the same structural root; empty request Grams; no answer cache',
    'commits': 'up to 15 continuous commits, each selected pair illegal for the rest of the request',
    'tie': 'exact canonical candidate ID order; preserve equality competition for positive incumbents',
}
SEED_DERIVATION = "int.from_bytes(SHA256(UTF8('A2|{seed_label}|2026-11-25|{graph}|{family}')).digest()[:8], 'big')"
RUNTIME_SOURCES = ('a2_bench', 'a2_catalog', 'a2_numeric', 'a2_access', 'a2_session',
                   'access_bench', 'anchor_service_bench', 'session', 'weighted_adapter', 'common')
INPUT_SOURCES = ('common', 'holdout_stream', 'confirmation_replay', 'checker')
PACKAGES = ('numpy', 'scipy', 'psutil', 'threadpoolctl', 'sympy')


def seed_label(label, graph, stream):
    if label not in ('development', 'confirmation-1', 'confirmation-2'):
        raise ValueError('unregistered seed label')
    if graph not in GRAPHS or stream not in STREAMS:
        raise ValueError('unregistered graph or family')
    return f'A2|{label}|2026-11-25|{graph}|{stream}'


def seed_for(label, graph, stream):
    return int.from_bytes(hashlib.sha256(seed_label(label, graph, stream).encode('utf-8')).digest()[:8], 'big')


def validate_protocol(protocol):
    if not isinstance(protocol, dict):
        raise ValueError('expected a JSON protocol object')
    cfg = dict(protocol)
    if not isinstance(cfg.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', cfg['id']):
        raise ValueError('invalid protocol ID')
    stage = cfg.get('stage')
    labels = {'development': ['development'],
              'confirmation': ['confirmation-1', 'confirmation-2']}.get(stage)
    if labels is None or cfg.get('seed_labels') != labels or cfg.get('date') != '2026-11-25':
        raise ValueError('unregistered development/confirmation seed labels or date')
    if cfg.get('graphs') != list(GRAPHS) or cfg.get('streams') != list(STREAMS):
        raise ValueError('all six graph/family cells must be retained in registered order')
    fixed = dict(requests=4, budget=15, delta=0, hot_size=64, repeat_count=3,
                 cpu=0, supervisor_cpu=8, numerical_threads=1)
    if any(type(cfg.get(k)) is not int or cfg[k] != v for k, v in fixed.items()):
        raise ValueError('A2 requires Q4, b15, delta0, hot64, three repeats, CPU0/supervisor8/thread1')
    if cfg.get('prefixes') != [1, 4] or cfg.get('methods') != list(METHODS):
        raise ValueError('A2 requires Q1/Q4 and the fixed D/S/H methods')
    expected_limits = LIMITS | {'batch_seconds': 2700 if stage == 'development' else 3600}
    if cfg.get('limits') != expected_limits:
        raise ValueError('unregistered resource limits')
    if cfg.get('domain') != DOMAIN or cfg.get('seed_derivation') != SEED_DERIVATION:
        raise ValueError('candidate domain or seed derivation differs from registration')
    if stage == 'confirmation':
        standard = cfg.get('confirmation_standard')
        if (not isinstance(standard, dict) or
                any(not standard.get(k) for k in ('claim', 'metric', 'acceptance'))):
            raise ValueError('confirmation requires a claim, metric and acceptance frozen before input generation')
    return cfg


def read_protocol(path):
    contents = Path(path).read_bytes()
    return validate_protocol(json.loads(contents)), hashlib.sha256(contents).hexdigest()


def jobs(cfg):
    cells = list(product(cfg['graphs'], cfg['streams'], cfg['seed_labels']))
    methods = cfg['methods']
    for repeat in range(cfg['repeat_count']):
        for cell_position, (graph, stream, label) in enumerate(cells):
            pivot = (repeat + cell_position) % len(methods)
            for method in methods[pivot:] + methods[:pivot]:
                yield dict(graph=graph, stream=stream, seed_label=label,
                           seed=seed_for(label, graph, stream), repeat_id=repeat, **method)


def job_key(job):
    return tuple(job[k] for k in ('graph', 'stream', 'seed_label', 'repeat_id', 'method'))


def result_filename(position, job):
    return f"{position:03d}_{job['graph']}_{job['stream']}_{job['seed_label']}_{job['label']}.json"


def source_hashes():
    """Freeze the actual numerical runtime and native sources/binaries, not analysis."""
    paths = [ROOT / 'src' / f'{name}.py' for name in RUNTIME_SOURCES]
    evidence = ROOT / 'baseline_r15/experimental_records/evidence_supplement_20261124'
    paths.append(evidence / 'diagnostic/common.py')
    legacy = evidence / 'source/legacy_v6/tools'
    for suffix in ('*.py', '*.c', '*.h'):
        paths.extend(sorted(legacy.glob(suffix)))
    paths.extend(evidence / 'diagnostic/build' / name for name in ('screen.so', 'index.so'))
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def input_source_hashes():
    paths = [ROOT / 'src' / f'{name}.py' for name in INPUT_SOURCES]
    paths.append(ROOT / 'baseline_r15/experimental_records/legacy_v6/tools/main_witness_verify.py')
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def environment():
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                platform=platform.platform(), packages={name: metadata.version(name) for name in PACKAGES})


def _new_json(path, value):
    """Publish a frozen manifest atomically, refusing even concurrent replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', prefix=path.name + '.', suffix='.tmp',
                                     dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _graph_record(graph):
    import numpy as np
    from scipy.sparse import load_npz

    paths = {kind: ROOT / 'data/projected' / f'{graph}_{kind}.npz' for kind in ('unit', 'weighted')}
    hashes = {kind: sha(path) for kind, path in paths.items()}
    meta_path = ROOT / 'data/projected' / f'{graph}_metadata.json'
    record = json.loads(meta_path.read_text())
    if any(hashes[kind] != record[kind + '_sha256'] for kind in paths):
        raise ValueError(f'projected graph checksum mismatch for {graph}')
    unit, weighted = (load_npz(paths[kind]) for kind in ('unit', 'weighted'))
    if (unit.format != 'csr' or not unit.has_canonical_format or
            unit.shape[0] != unit.shape[1] or not 1 <= unit.shape[0] <= 10000 or
            (unit != unit.T).nnz or unit.diagonal().any() or
            not np.all(np.abs(unit.data) == 1) or
            weighted.format != 'csr' or not weighted.has_canonical_format or
            (weighted != weighted.T).nnz or weighted.diagonal().any() or
            (weighted.sign() != unit).nnz or not np.isfinite(weighted.data).all() or
            not np.all(2 * weighted.data == np.rint(2 * weighted.data))):
        raise ValueError(f'invalid signed natural half-integer weighted projection for {graph}')
    n = unit.shape[0]
    positive = int(np.count_nonzero(unit.data > 0)) // 2
    negative = int(np.count_nonzero(unit.data < 0)) // 2
    nonedges = n * (n - 1) // 2 - unit.nnz // 2
    spec = dict(n=n, root_positive_edges=positive, root_negative_edges=negative,
                root_nonedges=nonedges, catalog_size=nonedges + negative,
                metadata_path=str(meta_path.relative_to(ROOT)), metadata_sha256=sha(meta_path))
    for kind, path in paths.items():
        spec[kind + '_path'] = str(path.relative_to(ROOT))
        spec[kind + '_sha256'] = hashes[kind]
    return spec, unit


def prepare(protocol_path, manifest_path, fixture_directory=None):
    protocol_path, manifest_path = Path(protocol_path), Path(manifest_path)
    cfg, protocol_hash = read_protocol(protocol_path)
    directory = Path(fixture_directory) if fixture_directory else ROOT / 'data/a2' / cfg['id']
    if manifest_path.exists() or directory.exists():
        raise FileExistsError('refusing to replace a fixture freeze')
    # No source imports, graph loads or input generation happen before this guard.
    import numpy as np
    from threadpoolctl import threadpool_limits
    from .common import opinions
    from .confirmation_replay import reference_stream
    from .holdout_stream import signed_neighbor_opinions

    sources, input_sources, env = source_hashes(), input_source_hashes(), environment()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    directory.parent.mkdir(parents=True, exist_ok=True)
    fixtures, graphs = [], {}
    with tempfile.TemporaryDirectory(dir=directory.parent) as temporary, threadpool_limits(limits=1):
        stage = Path(temporary) / 'fixtures'
        stage.mkdir()
        for graph in cfg['graphs']:
            spec, unit = _graph_record(graph)
            graphs[graph] = spec
            for stream, label in product(cfg['streams'], cfg['seed_labels']):
                seed = seed_for(label, graph, stream)
                rows = (signed_neighbor_opinions(unit, cfg['requests'], seed)
                        if stream == 'signed-neighbor' else
                        opinions(spec['n'], stream, seed, cfg['requests']))
                matrix = np.stack(rows)
                reference = reference_stream(unit, stream, seed, cfg['requests'])
                if not np.array_equal(matrix, reference):
                    raise ValueError(f'independent input regeneration mismatch: {graph}/{stream}/{label}')
                matrix_hash, row_hashes = matrix_hashes(matrix, (cfg['requests'], spec['n']))
                filename = f'{graph}_{stream}_{label}.npz'
                np.savez(stage / filename, opinions=matrix)
                fixtures.append(dict(graph=graph, stream=stream, seed_label=label, seed=seed,
                    seed_utf8_label=seed_label(label, graph, stream),
                    fixture_id=f'{graph}/{stream}/{label}/{seed}/Q4',
                    file=os.path.relpath(directory / filename, manifest_path.parent),
                    npz_sha256=sha(stage / filename), matrix_sha256=matrix_hash,
                    opinion_hashes=row_hashes, shape=list(matrix.shape), independently_matched_rows=4))
        if (sha(protocol_path) != protocol_hash or source_hashes() != sources or
                input_source_hashes() != input_sources or environment() != env):
            raise ValueError('protocol, source or environment changed during input preparation')
        if manifest_path.exists() or directory.exists():
            raise FileExistsError('fixture freeze was concurrently created')
        # mkdir is exclusive; rename over an empty competing directory would not be.
        # A failed publication leaves its partial directory for inspection and may
        # not be silently regenerated under this freeze name.
        directory.mkdir()
        for path in sorted(stage.iterdir()):
            path.rename(directory / path.name)
    selected_label = min(cfg['seed_labels'])
    selection = [dict(graph=g, stream='slow', seed_label=selected_label,
                      seed=seed_for(selected_label, g, 'slow'), method='H',
                      repeat_id=0, query_position=1) for g in cfg['graphs']]
    manifest = dict(schema_version=2, protocol_id=cfg['id'], stage=cfg['stage'],
                    protocol_sha256=protocol_hash, source_files_sha256=sources,
                    input_source_files_sha256=input_sources, environment=env,
                    domain=DOMAIN, graphs=graphs, fixtures=fixtures, verified=True,
                    independent_opinion_rows=len(fixtures) * 4,
                    independent_replay_selection=selection)
    _new_json(manifest_path, manifest)
    return manifest


def load_freeze(protocol_path, manifest_path, *, check_sources=True):
    cfg, protocol_hash = read_protocol(protocol_path)
    contents = Path(manifest_path).read_bytes()
    manifest = json.loads(contents)
    if (manifest.get('schema_version') != 2 or manifest.get('verified') is not True or
            manifest.get('protocol_id') != cfg['id'] or manifest.get('stage') != cfg['stage'] or
            manifest.get('protocol_sha256') != protocol_hash or manifest.get('domain') != DOMAIN):
        raise ValueError('prepared protocol/domain or independent input verification mismatch')
    if check_sources and (manifest.get('source_files_sha256') != source_hashes() or
                          manifest.get('input_source_files_sha256') != input_source_hashes() or
                          manifest.get('environment') != environment()):
        raise ValueError('frozen source or runtime environment mismatch')
    grid = set(product(cfg['graphs'], cfg['streams'], cfg['seed_labels']))
    records = {(r['graph'], r['stream'], r['seed_label']): r for r in manifest['fixtures']}
    if (set(records) != grid or len(manifest['fixtures']) != len(grid) or
            set(manifest['graphs']) != set(cfg['graphs']) or
            manifest.get('independent_opinion_rows') != len(grid) * cfg['requests']):
        raise ValueError('prepared fixture grid does not match registration')
    for (graph, stream, label), row in records.items():
        seed = seed_for(label, graph, stream)
        n = manifest['graphs'][graph]['n']
        if (row['seed'] != seed or row['seed_utf8_label'] != seed_label(label, graph, stream) or
                row['fixture_id'] != f'{graph}/{stream}/{label}/{seed}/Q4' or
                row['shape'] != [cfg['requests'], n] or
                len(row['opinion_hashes']) != cfg['requests'] or row['independently_matched_rows'] != 4):
            raise ValueError('prepared fixture identity or independent row coverage mismatch')
    selected_label = min(cfg['seed_labels'])
    expected_selection = [dict(graph=g, stream='slow', seed_label=selected_label,
                              seed=seed_for(selected_label, g, 'slow'), method='H',
                              repeat_id=0, query_position=1) for g in cfg['graphs']]
    if manifest.get('independent_replay_selection') != expected_selection:
        raise ValueError('independent replay selection changed')
    return cfg, manifest, records, hashlib.sha256(contents).hexdigest()


def _base_trace(cfg, manifest, manifest_hash, job, position):
    return dict(job, schema_version=2, status='starting', failure_reason=None,
                schedule_position=position, requests=[], prefixes={},
                protocol_id=cfg['id'], protocol_sha256=manifest['protocol_sha256'],
                source_files_sha256=manifest['source_files_sha256'],
                input_source_files_sha256=manifest['input_source_files_sha256'],
                input_manifest_sha256=manifest_hash, environment=manifest['environment'],
                requests_planned=cfg['requests'], budget=cfg['budget'], delta=0,
                hot_size=64, audit=False, cpu=cfg['cpu'], weight_mode='weighted',
                domain=DOMAIN, stream_id=f"{job['graph']}/weighted/{job['stream']}/{job['seed_label']}/{job['seed']}")


def _check_seconds(phases, elapsed, label):
    if (type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed < 0 or
            not isinstance(phases, dict) or
            any(type(v) not in (float, int) or not math.isfinite(v) or v < -1e-8 for v in phases.values()) or
            abs(sum(phases.values()) - elapsed) > 1e-7):
        raise ValueError(label + ' accounting mismatch')


def worker(protocol_path, manifest_path, output, position, expected_manifest_hash):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    for name in THREAD_ENV:
        os.environ[name] = '1'
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
        trace.update(_base_trace(cfg, manifest, manifest_hash, job, position))
        os.sched_setaffinity(0, {cfg['cpu']})
        trace['affinity'] = sorted(os.sched_getaffinity(0))
        trace['thread_environment'] = {name: os.environ[name] for name in THREAD_ENV}
        checkpoint()

        from scipy.sparse import load_npz
        from threadpoolctl import threadpool_info, threadpool_limits
        from .a2_session import A2Session

        with threadpool_limits(limits=1):
            trace['threadpools'] = threadpool_info()
            start = time.monotonic()
            spec = manifest['graphs'][job['graph']]
            graph_path = ROOT / spec['weighted_path']
            graph_hash = sha(graph_path)
            if graph_hash != spec['weighted_sha256']:
                raise ValueError('weighted graph digest mismatch')
            record = records[job['graph'], job['stream'], job['seed_label']]
            matrix = load_fixture(record, Path(manifest_path).parent)
            trace.update(graph_hash=graph_hash, fixture_id=record['fixture_id'],
                         fixture_npz_sha256=record['npz_sha256'], fixture_matrix_sha256=record['matrix_sha256'],
                         fixture_and_graph_validate_seconds=time.monotonic() - start)
            start = time.perf_counter()
            W = load_npz(graph_path)
            if W.shape != (spec['n'], spec['n']):
                raise ValueError('weighted graph dimensions differ from the freeze')
            graph_load_seconds = time.perf_counter() - start
            session = A2Session(W, job['method'], hot_size=cfg['hot_size'], timers=True,
                                graph_hash=graph_hash, graph_name=job['graph'])
            del W
            elapsed = time.perf_counter() - start
            phases = dict(session.setup_phases)
            _check_seconds(phases, session.setup_seconds, 'session structural')
            phases['graph_load'] = phases.get('graph_load', 0.) + graph_load_seconds
            phases['other'] = phases.get('other', 0.) + elapsed - sum(phases.values())
            _check_seconds(phases, elapsed, 'full structural')
            if session.n != spec['n'] or session.m != spec['catalog_size']:
                raise ValueError('session dimensions/catalog count differ from the full registered domain')
            trace.update(status='prepared', n=session.n, catalog_size=session.m,
                         operator_hash=session.operator_hash, catalog_hash=session.catalog_hash,
                         root_version_id=session.root_id, structural_seconds=elapsed,
                         structural_phases=phases, session_setup_seconds=session.setup_seconds,
                         service_accounted_seconds=elapsed)
            cumulative = elapsed
            checkpoint()
            for q, opinion in enumerate(matrix, 1):
                start = time.perf_counter()
                request = session.request(opinion, budget=cfg['budget'], delta=0, audit=False)
                elapsed = time.perf_counter() - start
                _check_seconds(request['phases'], request['request_seconds'], 'request internal')
                overhead = elapsed - request['request_seconds']
                if overhead < -1e-7:
                    raise ValueError('request wall exceeds its outer call')
                request['phases']['reset_restore'] = request['phases'].get('reset_restore', 0.) + overhead
                request['reset_opinion_seconds'] += overhead
                request.update(request_seconds=elapsed, query_position=q, stream_id=trace['stream_id'],
                               wrapper_seconds=overhead)
                trace['requests'].append(request)
                if (request['opinion_hash'] != record['opinion_hashes'][q - 1] or
                        abs(request['reset_opinion_seconds'] + request['online_seconds'] - elapsed) > 1e-7):
                    raise ValueError('request opinion or reset/online accounting mismatch')
                _check_seconds(request['phases'], elapsed, 'request full')
                cumulative += elapsed
                trace['service_accounted_seconds'] = cumulative
                if q in cfg['prefixes']:
                    trace['prefixes'][str(q)] = cumulative
                trace['status'] = 'in_progress'
                checkpoint()
            trace['status'] = ('completed' if all(r['status'] in TERMINAL_REQUESTS for r in trace['requests'])
                               else 'completed_with_failures')
    except Exception as error:
        trace.update(status='memory_error' if isinstance(error, MemoryError) else 'error',
                     failure_reason=repr(error), traceback=traceback.format_exc())
    finally:
        checkpoint()
    return trace


def batch(protocol_path, manifest_path, output):
    cfg, manifest, _, manifest_hash = load_freeze(protocol_path, manifest_path)
    output = Path(output)
    limits = cfg['limits']
    lock_path = ROOT / 'logs/.measurement.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('refusing to replace an existing batch; no retries')
        output.mkdir(parents=True, exist_ok=True)
        os.sched_setaffinity(0, {cfg['supervisor_cpu']})
        schedule = list(jobs(cfg))
        _new_json(output / 'FROZEN_JOBS.json', dict(schema_version=2, jobs=schedule,
                  cpu=cfg['cpu'], supervisor_cpu=cfg['supervisor_cpu'], limits=limits,
                  supervisor_affinity=sorted(os.sched_getaffinity(0)),
                  domain=DOMAIN, environment=manifest['environment'],
                  protocol_sha256=manifest['protocol_sha256'], input_manifest_sha256=manifest_hash,
                  source_files_sha256=manifest['source_files_sha256'],
                  input_source_files_sha256=manifest['input_source_files_sha256']))
        started = time.monotonic()
        deadline = started + limits['batch_seconds']
        results = dict(schema_version=2, jobs=[dict(position=pos, job=job,
                  result=result_filename(pos, job), status='registered_not_started',
                  returncode=None, process_wall_seconds=None,
                  external_sampled_peak_rss_bytes=None, completed_requests=0)
                  for pos, job in enumerate(schedule)], limits=limits,
                  supervisor_affinity=sorted(os.sched_getaffinity(0)),
                  batch_wall_seconds=0., status='in_progress')
        save(output / 'SCHEDULE_RESULTS.json', results)
        for pos, job in enumerate(schedule):
            row = results['jobs'][pos]
            path = output / row['result']
            now = time.monotonic()
            partial = _base_trace(cfg, manifest, manifest_hash, job, pos)
            if now < deadline:
                command = [sys.executable, '-m', 'src.a2_bench', 'worker',
                           '--protocol', str(Path(protocol_path).resolve()),
                           '--manifest', str(Path(manifest_path).resolve()),
                           '--output', str(path.resolve()), '--schedule-position', str(pos),
                           '--manifest-sha256', manifest_hash]
                try:
                    code, wall, peak, stop = supervise(command, path.with_suffix('.log'),
                            now + limits['worker_seconds'], deadline, limits)
                    if path.exists():
                        try:
                            partial.update(json.loads(path.read_text()))
                        except (ValueError, TypeError) as error:
                            row.update(status='invalid_checkpoint', failure_reason=repr(error),
                                       returncode=code, process_wall_seconds=wall,
                                       external_sampled_peak_rss_bytes=peak)
                            # Keep the invalid raw bytes; the scheduler is the failure record.
                            partial = None
                    if partial is not None:
                        if partial.get('process_peak_rss_bytes', 0) > limits['rss_mib'] * 1024**2:
                            stop = stop or 'rss_cap'
                        status = partial.get('status', 'exit_before_checkpoint')
                        if status in ('starting', 'prepared', 'in_progress') or (code and status == 'completed'):
                            status = 'worker_exit'
                        row.update(status=stop or status, returncode=code, process_wall_seconds=wall,
                                   external_sampled_peak_rss_bytes=peak,
                                   completed_requests=len(partial.get('requests', [])))
                        if stop or status in ('exit_before_checkpoint', 'worker_exit'):
                            partial.update(status=row['status'], last_worker_status=partial.get('status'),
                                           failure_reason=stop or f'worker exited with {code}')
                            save(path, partial)
                except OSError as error:
                    row.update(status='process_launch_error', failure_reason=repr(error))
                    partial.update(status=row['status'], failure_reason=repr(error))
                    save(path, partial)
            else:
                row['status'] = 'batch_cap_not_started'
                partial.update(status=row['status'], failure_reason='registered batch cap')
                save(path, partial)
            results['batch_wall_seconds'] = time.monotonic() - started
            save(output / 'SCHEDULE_RESULTS.json', results)
            print(f"{pos + 1}/{len(schedule)} {row['status']} {row['result']}", flush=True)
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
        result = prepare(args.protocol, args.manifest, args.fixture_dir)
        print(f"{len(result['fixtures'])} frozen fixtures; {result['independent_opinion_rows']} independently regenerated rows")
        return 0
    result = (batch(args.protocol, args.manifest, args.output) if args.mode == 'batch' else
              worker(args.protocol, args.manifest, args.output, args.schedule_position, args.manifest_sha256))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
