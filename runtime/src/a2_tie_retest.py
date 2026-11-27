"""One revision retest on diagnosed fixed inputs; never an unseen confirmation.

The archived confirmation protocol, inputs, roster and failed judgment are read
only. This module creates a separate source binding while preserving every old
fixture record, including its relative path. It reuses the existing serial
benchmark and analyzer code with private function-global bindings; neither
module nor the D/S/H access implementations are modified by this adapter.
Only root launches the actual freeze and bounded 108-worker batch.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import types

from . import a2_bench as bench


ROOT = bench.ROOT
STUDY_LABEL = 'revision retest on diagnosed fixed inputs'
PROTOCOL_ID = 'A2-exact-tie-retest-v1'
STAGE = 'revision_retest'
ORIGINAL_PATHS = {
    'protocol': 'protocol/A2_CONFIRMATION.json',
    'inputs': 'protocol/A2_CONFIRMATION_INPUTS.json',
    'admission': 'protocol/A2_CONFIRMATION_ADMISSION.json',
    'judgment': 'reports/A2_CONFIRMATION_JUDGMENT.json',
    'frozen_jobs': 'raw/a2_confirmation/FROZEN_JOBS.json',
}
ALLOWED_MODIFIED = ('src/a2_session.py', 'src/weighted_adapter.py')
ADDED_SOURCES = ('src/a2_tie_retest.py', 'src/exact_ties.py')
RETAINED_FIELDS = ('domain', 'graphs', 'fixtures', 'verified',
                   'independent_opinion_rows', 'independent_replay_selection',
                   'input_source_files_sha256', 'environment')
BINDING_FIELDS = ('protocol_id', 'stage', 'protocol_sha256', 'source_files_sha256',
                  'input_source_files_sha256', 'environment')


def _with_bindings(function, **bindings):
    """Reuse an unchanged function body without patching shared module globals."""
    result = types.FunctionType(function.__code__, function.__globals__ | bindings,
                                function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def read_protocol(path):
    contents = Path(path).read_bytes()
    cfg = json.loads(contents)
    if (not isinstance(cfg, dict) or cfg.get('id') != PROTOCOL_ID or
            cfg.get('stage') != STAGE or cfg.get('study_label') != STUDY_LABEL or
            cfg.get('unseen_confirmation') is not False or 'confirmation_standard' in cfg):
        raise ValueError('expected a separately labeled diagnosed-input revision retest')
    # Reuse the registered fixed-policy validator. The temporary confirmation
    # stage exists only for validation and is never published or returned.
    compatible = dict(cfg, stage='confirmation', confirmation_standard={
        'claim': STUDY_LABEL, 'metric': 'full structural plus Q4 service',
        'acceptance': 'descriptive retest only; the original confirmation remains failed'})
    bench.validate_protocol(compatible)
    original = cfg.get('original_confirmation', {})
    if not isinstance(original, dict) or original.get('status') != 'failed':
        raise ValueError('the original confirmation status must remain failed')
    for key, expected in ORIGINAL_PATHS.items():
        row = original.get(key)
        if (not isinstance(row, dict) or row.get('path') != expected or
                not isinstance(row.get('sha256'), str) or
                not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise ValueError('missing exact original provenance: ' + key)
    scope = cfg.get('revision_scope', {})
    if (scope.get('allowed_modified_original_runtime_sources') != list(ALLOWED_MODIFIED) or
            scope.get('added_runtime_sources') != list(ADDED_SOURCES) or
            scope.get('seed_label_aliases') != {'C1': 'confirmation-1', 'C2': 'confirmation-2'}):
        raise ValueError('revision scope or original seed-label identity changed')
    return cfg, hashlib.sha256(contents).hexdigest()


def source_hashes():
    """The original runtime set plus the shared exact resolver and this adapter."""
    return bench.source_hashes() | {path: bench.sha(ROOT / path) for path in ADDED_SOURCES}


def _same_manifest_directory(manifest_path):
    if Path(manifest_path).resolve().parent != (ROOT / ORIGINAL_PATHS['inputs']).resolve().parent:
        raise ValueError('new manifest must be a sibling in the same directory as the original; '
                         'fixture paths are retained verbatim')


def _original(cfg, *, verify_inputs=True):
    """Verify historical bytes and input identities, explicitly excluding only
    the old runtime binding from *current* source validation.

    The excluded historical binding is preserved verbatim in the new manifest;
    all current sources and environment are validated separately below.
    """
    references = cfg['original_confirmation']
    for key, relative in ORIGINAL_PATHS.items():
        if bench.sha(ROOT / relative) != references[key]['sha256']:
            raise ValueError('original archived provenance digest changed: ' + key)
    original_cfg, original, _, original_hash = bench.load_freeze(
        ROOT / ORIGINAL_PATHS['protocol'], ROOT / ORIGINAL_PATHS['inputs'], check_sources=False)
    judgment = json.loads((ROOT / ORIGINAL_PATHS['judgment']).read_text())
    if judgment.get('status') != 'failed':
        raise ValueError('original confirmation must remain failed')
    frozen = json.loads((ROOT / ORIGINAL_PATHS['frozen_jobs']).read_text())
    schedule = list(bench.jobs(cfg))
    if (original_cfg['id'] != 'A2-confirmation-v1' or original_cfg['stage'] != 'confirmation' or
            len(original['fixtures']) != 12 or len(schedule) != 108 or
            list(bench.jobs(original_cfg)) != schedule or frozen.get('jobs') != schedule):
        raise ValueError('original twelve-cell/108-worker rotated roster differs')
    expected = {
        'schema_version': 2, 'cpu': 0, 'supervisor_cpu': 8, 'supervisor_affinity': [8],
        'limits': cfg['limits'], 'domain': original['domain'],
        'protocol_sha256': original['protocol_sha256'], 'input_manifest_sha256': original_hash,
        'source_files_sha256': original['source_files_sha256'],
        'input_source_files_sha256': original['input_source_files_sha256'],
        'environment': original['environment'],
    }
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise ValueError('original scheduler provenance or resource policy changed')
    if verify_inputs:
        for graph, spec in original['graphs'].items():
            actual, _ = bench._graph_record(graph)
            if actual != spec:
                raise ValueError('original graph metadata or digest changed: ' + graph)
        for row in original['fixtures']:
            # Existing loader checks NPZ SHA, exact int64 shape, matrix and every
            # row digest. It never regenerates or writes an input file.
            bench.load_fixture(row, (ROOT / ORIGINAL_PATHS['inputs']).parent)
    return original


def _check_revision_scope(sources, inputs, env, original):
    old = original['source_files_sha256']
    if set(sources) != set(old) | set(ADDED_SOURCES):
        raise ValueError('revision source scope is not the original runtime plus the two declared modules')
    changed = {path for path in old if sources[path] != old[path]}
    if not changed <= set(ALLOWED_MODIFIED):
        raise ValueError('unauthorized source changes outside common resolver scope: ' +
                         ', '.join(sorted(changed - set(ALLOWED_MODIFIED))))
    if inputs != original['input_source_files_sha256']:
        raise ValueError('input sources must remain unchanged; no regeneration is authorized')
    if env != original['environment']:
        raise ValueError('runtime environment differs from the original confirmation')


def _historical_binding(original):
    return {key: copy.deepcopy(original[key]) for key in BINDING_FIELDS}


def _label(value, cfg):
    value.update(stage=STAGE, study_label=STUDY_LABEL, unseen_confirmation=False,
                 original_confirmation_status='failed',
                 original_confirmation=copy.deepcopy(cfg['original_confirmation']))
    return value


def prepare(protocol_path, manifest_path):
    """Publish only a new manifest; reuse the exact archived inputs in place."""
    protocol_path, manifest_path = Path(protocol_path), Path(manifest_path)
    if manifest_path.exists():
        raise FileExistsError('refusing to replace an existing retest freeze')
    _same_manifest_directory(manifest_path)
    cfg, protocol_hash = read_protocol(protocol_path)
    original = _original(cfg)
    sources, inputs, env = source_hashes(), bench.input_source_hashes(), bench.environment()
    _check_revision_scope(sources, inputs, env, original)
    manifest = _label(copy.deepcopy(original), cfg)
    manifest.update(protocol_id=cfg['id'], protocol_sha256=protocol_hash,
                    source_files_sha256=sources, input_source_files_sha256=inputs, environment=env,
                    original_source_binding=_historical_binding(original),
                    inputs_reused_without_regeneration=True, retained_fixture_count=12,
                    retained_worker_count=108)
    if (bench.sha(protocol_path) != protocol_hash or source_hashes() != sources or
            bench.input_source_hashes() != inputs or bench.environment() != env or
            _original(cfg) != original):
        raise ValueError('protocol, sources, environment or original inputs changed during retest preparation')
    bench._new_json(manifest_path, manifest)
    return manifest


def load_freeze(protocol_path, manifest_path, *, verify_inputs=True):
    _same_manifest_directory(manifest_path)
    cfg, manifest, records, manifest_hash = _with_bindings(
        bench.load_freeze, read_protocol=read_protocol, source_hashes=source_hashes)(
            protocol_path, manifest_path)
    original = _original(cfg, verify_inputs=verify_inputs)
    if (any(manifest.get(key) != original[key] for key in RETAINED_FIELDS) or
            manifest.get('original_source_binding') != _historical_binding(original) or
            manifest.get('original_confirmation') != cfg['original_confirmation'] or
            manifest.get('original_confirmation_status') != 'failed' or
            manifest.get('study_label') != STUDY_LABEL or manifest.get('unseen_confirmation') is not False or
            manifest.get('inputs_reused_without_regeneration') is not True or
            manifest.get('retained_fixture_count') != 12 or manifest.get('retained_worker_count') != 108):
        raise ValueError('retest did not retain exact original fixtures, history and explicit binding provenance')
    _check_revision_scope(manifest['source_files_sha256'], manifest['input_source_files_sha256'],
                          manifest['environment'], original)
    return cfg, manifest, records, manifest_hash


def _base_trace(cfg, manifest, manifest_hash, job, position):
    return _label(bench._base_trace(cfg, manifest, manifest_hash, job, position), cfg)


def _supervise(command, *args, **kwargs):
    if command[1:4] != ['-m', 'src.a2_bench', 'worker']:
        raise ValueError('unexpected original worker command')
    command = list(command)
    command[2] = 'src.a2_tie_retest'
    return bench.supervise(command, *args, **kwargs)


def worker(protocol_path, manifest_path, output, position, expected_manifest_hash):
    if Path(output).exists():
        raise FileExistsError(output)
    cfg, _ = read_protocol(protocol_path)
    # Full input verification occurs in prepare and the batch preflight. The
    # unchanged worker verifies its graph and active NPZ/matrix/row hashes after
    # pinning CPU0; do not add unpinned graph loads to its freeze preamble.
    return _with_bindings(bench.worker,
        load_freeze=lambda p, m: load_freeze(p, m, verify_inputs=False), _base_trace=_base_trace,
        save=lambda path, value: bench.save(path, _label(value, cfg)))(
            protocol_path, manifest_path, output, position, expected_manifest_hash)


def batch(protocol_path, manifest_path, output):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('refusing to replace an existing batch; no retries')
    cfg, _ = read_protocol(protocol_path)
    return _with_bindings(bench.batch, load_freeze=load_freeze, _base_trace=_base_trace,
        supervise=_supervise,
        save=lambda path, value: bench.save(path, _label(value, cfg)),
        _new_json=lambda path, value: bench._new_json(path, _label(value, cfg)))(
            protocol_path, manifest_path, output)


def analyze(protocol_path, directory, output, manifest_path):
    """Use the normal full-roster analyzer, with the explicit retest binding."""
    if Path(output).exists():
        raise FileExistsError(output)
    from . import a2_analyze
    cfg, _ = read_protocol(protocol_path)
    return _with_bindings(a2_analyze.analyze, read_protocol=read_protocol, load_freeze=load_freeze,
        save=lambda path, value: bench._new_json(path, _label(value, cfg)))(
            protocol_path, directory, output, manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    for mode in ('prepare', 'batch', 'worker', 'analyze'):
        cmd = modes.add_parser(mode)
        cmd.add_argument('--protocol', required=True, type=Path)
        cmd.add_argument('--manifest', required=True, type=Path)
        if mode != 'prepare':
            cmd.add_argument('--output', required=True, type=Path)
        if mode == 'worker':
            cmd.add_argument('--schedule-position', required=True, type=int)
            cmd.add_argument('--manifest-sha256', required=True)
        if mode == 'analyze':
            cmd.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        result = prepare(args.protocol, args.manifest)
        print(f"{STUDY_LABEL}: {len(result['fixtures'])} exact archived fixtures reused; "
              '108 registered workers; original confirmation remains failed')
        return 0
    if args.mode == 'analyze':
        result = analyze(args.protocol, args.directory, args.output, args.manifest)
        return 0 if result['correctness_accounting_pass'] else 1
    result = (batch(args.protocol, args.manifest, args.output) if args.mode == 'batch' else
              worker(args.protocol, args.manifest, args.output,
                     args.schedule_position, args.manifest_sha256))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
