"""Synthetic-only tests of the A2 freeze, roster and complete-cost denominator."""
import copy
from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.a2_bench import (jobs, load_fixture, prepare, seed_for, seed_label,
                          validate_protocol, load_freeze, _base_trace, result_filename,
                          THREAD_ENV, DOMAIN, worker)
from src.a2_analyze import ACCESS_COUNTS, _compare_outputs, analyze, summarize, validate_request


def protocol(stage='development'):
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / 'protocol/A2_DEVELOPMENT.json').read_text())
    if stage == 'confirmation':
        cfg.update(id='A2-confirmation-v1', stage=stage,
                   seed_labels=['confirmation-1', 'confirmation-2'])
        cfg['limits']['batch_seconds'] = 3600
        cfg['confirmation_standard'] = {
            'claim': 'Synthetic confirmation standard for protocol validation.',
            'metric': 'full structural plus all four request seconds',
            'acceptance': {'test_only': True},
        }
    return validate_protocol(cfg)


def synthetic_observations(cfg):
    result = []
    for pos, job in enumerate(jobs(cfg)):
        scale = {'D': 4., 'S': 2., 'H': 1.}[job['method']]
        trace = dict(job, status='completed', structural_seconds=scale,
                     prefixes={'1': 2. * scale, '4': 5. * scale},
                     service_accounted_seconds=5. * scale,
                     process_peak_rss_bytes=100,
                     requests=[dict(status='budget_stop', request_seconds=scale,
                                    counters={'response_blocks': 2, 'candidate_rhs': 3,
                                              'gram_materializations': 2},
                                    access={'epochs': []}, memory={}) for _ in range(4)])
        result.append({'position': pos, 'job': job, 'trace': trace,
                       'scheduler': {'status': 'completed', 'returncode': 0,
                                     'process_wall_seconds': 6. * scale,
                                     'external_sampled_peak_rss_bytes': 100},
                       'valid': True, 'issues': []})
    return result


def synthetic_request():
    from scipy import sparse
    W = sparse.csr_matrix(np.array([[0., -.5, 1.], [-.5, 0., 0.], [1., 0., 0.]]))
    root, opinion = 'a' * 64, 'b' * 64
    event = dict.fromkeys(ACCESS_COUNTS, 0)
    event.update(epoch=1, parent_version_id=root, source='H', frontier_rebuilds=1,
                 remaining_upper_coarse=None, source_complete=True)
    decision = dict(epoch=1, root_version_id=root, parent_version_id=root,
                    current_version_id=root, status='tolerance_stop', winner_id=None,
                    low=0, high=None, live_catalog=2, proof_scope='full_catalog',
                    endpoint_checked=0, endpoint_excluded=0, scored=0,
                    scored_rival_max_high=None, cold_upper=None, competitor_max_upper=None)
    trace = dict(root_version_id=root, stream_id='synthetic', method='H')
    fixture = dict(opinion_hashes=[opinion] * 4)
    spec = dict(n=3, catalog_size=2)
    request = dict(query_position=1, opinion_hash=opinion, stream_id='synthetic', method='H',
                   budget=15, delta=0, initial_gram_count=0, initial_seed_ids=[1, 5],
                   initial_seed_order=[1, 5], seed_rule='paid-joint-frontier-first64/heap-order-recorded',
                   root_version_id=root, current_version_id=root,
                   status='tolerance_stop', commits=0, gain_low=0, gain_high=0,
                   gain_denominator=256, request_seconds=.3, phases={'other': .3},
                   online_seconds=.2, online_phases={'other': .2}, reset_opinion_seconds=.1,
                   counters=dict(response_blocks=0, candidate_rhs=0, gram_materializations=0,
                                 endpoint_checked=0),
                   memory=dict(catalog_materialized=False, candidate_array_rows=0,
                               full_root_caps_created_rows=0, m_sized_structures=[]),
                   decisions=[decision], access={'epochs': [event]})
    return request, trace, fixture, spec, W


def synthetic_commit_request():
    request, trace, fixture, spec, W = synthetic_request()
    parent = trace['root_version_id']
    child = hashlib.sha256(f'{parent}/1'.encode()).hexdigest()
    commit = copy.deepcopy(request['decisions'][0])
    commit.update(status='accepted', winner_id=1, i=0, j=1, flip=1,
                  d=8, low=10, high=12, current_version_id=child)
    stop = copy.deepcopy(request['decisions'][0])
    stop.update(epoch=2, parent_version_id=child, current_version_id=child, live_catalog=1)
    event = copy.deepcopy(request['access']['epochs'][0])
    event.update(epoch=2, parent_version_id=child)
    request.update(current_version_id=child, commits=1, gain_low=10, gain_high=12,
                   decisions=[commit, stop])
    request['access']['epochs'].append(event)
    return request, trace, fixture, spec, W


def comparison_rows(requests):
    return [dict(position=position, job=job, trace={'requests': [request]}, valid=True, issues=[])
            for position, (job, request) in enumerate(zip(list(jobs(protocol()))[:3], requests, strict=True))]


def partial_request(accepted=0, status='memory_error'):
    values = synthetic_commit_request() if accepted else synthetic_request()
    request, trace, fixture, spec, W = values
    request['status'] = request['decisions'][-1]['status'] = status
    request['decisions'][-1]['failure_reason'] = status + ': synthetic injection'
    request['access']['epochs'][-1]['source_complete'] = False
    return values


def two_commit_request():
    request, trace, fixture, spec, W = synthetic_commit_request()
    parent = request['current_version_id']
    child = hashlib.sha256(f'{parent}/5'.encode()).hexdigest()
    commit = copy.deepcopy(request['decisions'][0])
    commit.update(epoch=2, parent_version_id=parent, current_version_id=child,
                  winner_id=5, i=1, j=2, flip=0, d=1, low=20, high=21, live_catalog=1)
    stop = copy.deepcopy(request['decisions'][-1])
    stop.update(epoch=3, parent_version_id=child, current_version_id=child, live_catalog=0)
    request['decisions'][-1:] = [commit, stop]
    event = copy.deepcopy(request['access']['epochs'][-1])
    event.update(epoch=3, parent_version_id=child)
    request['access']['epochs'].append(event)
    request.update(current_version_id=child, commits=2, gain_low=30, gain_high=33)
    return request, trace, fixture, spec, W


@contextmanager
def synthetic_bundle():
    """Consistent synthetic records only; these are not numerical proof records."""
    from scipy import sparse
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        projected = root / 'data/projected'
        projected.mkdir(parents=True)
        W = synthetic_request()[-1]
        for graph in ('Alpha', 'OTC'):
            hashes = {}
            for kind, matrix in (('unit', W.sign()), ('weighted', W)):
                path = projected / f'{graph}_{kind}.npz'
                sparse.save_npz(path, matrix)
                hashes[kind + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            (projected / f'{graph}_metadata.json').write_text(json.dumps(hashes))
        registration, manifest_path = root / 'protocol.json', root / 'manifest.json'
        cfg = protocol()
        registration.write_text(json.dumps(cfg))
        sources, env = {'synthetic': 'f' * 64}, {'synthetic': True}
        with patch('src.a2_bench.ROOT', root), patch('src.a2_analyze.ROOT', root), \
                patch('src.a2_bench.source_hashes', return_value=sources), \
                patch('src.a2_bench.input_source_hashes', return_value=sources), \
                patch('src.a2_bench.environment', return_value=env):
            manifest = prepare(registration, manifest_path, root / 'fixtures')
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            fixtures = {(r['graph'], r['stream'], r['seed_label']): r for r in manifest['fixtures']}
            output = root / 'raw'
            output.mkdir()
            roster = list(jobs(cfg))
            scheduler = dict(schema_version=2, jobs=[], limits=cfg['limits'],
                             supervisor_affinity=[8], batch_wall_seconds=1., status='completed')
            for position, job in enumerate(roster):
                fixture = fixtures[job['graph'], job['stream'], job['seed_label']]
                spec = manifest['graphs'][job['graph']]
                trace = _base_trace(cfg, manifest, manifest_hash, job, position)
                trace.update(status='completed', n=3, catalog_size=2,
                             graph_hash=spec['weighted_sha256'], fixture_id=fixture['fixture_id'],
                             fixture_npz_sha256=fixture['npz_sha256'],
                             fixture_matrix_sha256=fixture['matrix_sha256'],
                             root_version_id='a' * 64, operator_hash='c' * 64, catalog_hash='d' * 64,
                             affinity=[0], threadpools=[{'num_threads': 1}],
                             thread_environment={name: '1' for name in THREAD_ENV},
                             structural_seconds=1., structural_phases={'other': 1.},
                             process_peak_rss_bytes=100)
                cumulative = 1.
                for q in range(1, 5):
                    request = synthetic_request()[0]
                    request.update(method=job['method'], query_position=q,
                                   opinion_hash=fixture['opinion_hashes'][q - 1], stream_id=trace['stream_id'])
                    event = request['access']['epochs'][0]
                    event.update(source=job['method'], frontier_rebuilds=int(job['method'] != 'D'))
                    if job['method'] == 'D':
                        request['seed_rule'] = 'paid-global-root-caps-top64/canonical-ties'
                        request['memory'].update(catalog_materialized=True, candidate_array_rows=2,
                                                 full_root_caps_created_rows=2, m_sized_structures=['catalog'])
                    trace['requests'].append(request)
                    cumulative += request['request_seconds']
                    if q in (1, 4):
                        trace['prefixes'][str(q)] = cumulative
                trace['service_accounted_seconds'] = cumulative
                filename = result_filename(position, job)
                (output / filename).write_text(json.dumps(trace))
                scheduler['jobs'].append(dict(position=position, job=job, result=filename,
                                              status='completed', returncode=0,
                                              process_wall_seconds=3., completed_requests=4,
                                              external_sampled_peak_rss_bytes=100))
            frozen = dict(schema_version=2, jobs=roster, cpu=0, supervisor_cpu=8,
                          limits=cfg['limits'], supervisor_affinity=[8], domain=DOMAIN,
                          environment=env, protocol_sha256=manifest['protocol_sha256'],
                          input_manifest_sha256=manifest_hash, source_files_sha256=sources,
                          input_source_files_sha256=sources)
            (output / 'FROZEN_JOBS.json').write_text(json.dumps(frozen))
            (output / 'SCHEDULE_RESULTS.json').write_text(json.dumps(scheduler))
            yield registration, manifest_path, output, root / 'analysis.json'


class A2BenchTests(unittest.TestCase):
    def test_first_failed_request_does_not_invalidate_longer_successful_outputs(self):
        partial = partial_request(1)[0]
        completed = two_commit_request()[0]
        rows = comparison_rows([partial, copy.deepcopy(completed), copy.deepcopy(completed)])
        self.assertEqual(_compare_outputs(rows), [])
        self.assertTrue(all(row['valid'] for row in rows))

    def test_short_first_partial_does_not_hide_disagreement_between_completed_requests(self):
        rows = comparison_rows([partial_request(0)[0], synthetic_commit_request()[0], two_commit_request()[0]])
        issues = _compare_outputs(rows)
        self.assertTrue(issues)
        self.assertTrue(rows[0]['valid'])
        self.assertTrue(rows[1]['valid'])
        self.assertFalse(rows[2]['valid'])

    def test_different_partial_progress_compares_only_shared_accepted_prefix(self):
        longest = two_commit_request()[0]
        longest['status'] = longest['decisions'][-1]['status'] = 'memory_error'
        longest['access']['epochs'][-1]['source_complete'] = False
        rows = comparison_rows([partial_request(0, 'unresolved')[0], partial_request(1)[0], longest])
        self.assertEqual(_compare_outputs(rows), [])
        self.assertTrue(all(row['valid'] for row in rows))

    def test_conflicting_accepted_partial_prefix_still_fails(self):
        partial = partial_request(1)[0]
        partial['decisions'][0]['winner_id'] = 5
        completed = two_commit_request()[0]
        rows = comparison_rows([partial, copy.deepcopy(completed), copy.deepcopy(completed)])
        self.assertTrue(_compare_outputs(rows))
        self.assertFalse(rows[0]['valid'])
        self.assertTrue(rows[1]['valid'])
        self.assertTrue(rows[2]['valid'])

    def test_partial_positive_commit_cannot_extend_past_completed_tolerance_stop(self):
        partial = two_commit_request()[0]
        partial['status'] = partial['decisions'][-1]['status'] = 'memory_error'
        partial['access']['epochs'][-1]['source_complete'] = False
        completed = synthetic_commit_request()[0]
        rows = comparison_rows([partial, copy.deepcopy(completed), copy.deepcopy(completed)])
        self.assertTrue(_compare_outputs(rows))
        self.assertFalse(rows[0]['valid'])
        self.assertTrue(rows[1]['valid'])
        self.assertTrue(rows[2]['valid'])

    def test_incomplete_failed_frontier_build_may_report_zero_completed_rebuilds(self):
        request, trace, fixture, spec, W = partial_request(1)
        request['access']['epochs'][-1]['frontier_rebuilds'] = 0
        validate_request(request, trace, fixture, 1, spec, W)
        request['access']['epochs'][-1]['source_complete'] = True
        with self.assertRaisesRegex(ValueError, 'frontier'):
            validate_request(request, trace, fixture, 1, spec, W)

    def test_session_memory_errors_count_once_and_leave_complete_streaming_pairs(self):
        with synthetic_bundle() as (registration, manifest, raw, output), redirect_stdout(io.StringIO()):
            path = raw / 'SCHEDULE_RESULTS.json'
            scheduler = json.loads(path.read_text())
            first = scheduler['jobs'][0]
            trace_path = raw / first['result']
            trace = json.loads(trace_path.read_text())
            trace['status'] = first['status'] = 'completed_with_failures'
            first['returncode'] = 1
            scheduler['status'] = 'completed_with_failures'
            for request in trace['requests'][:2]:
                request['status'] = request['decisions'][-1]['status'] = 'memory_error'
                request['decisions'][-1]['failure_reason'] = 'MemoryError: synthetic injection'
                request['access']['epochs'][-1]['source_complete'] = False
            path.write_text(json.dumps(scheduler))
            trace_path.write_text(json.dumps(trace))
            result = analyze(registration, raw, output, manifest)
            self.assertEqual(result['validation_issues'], [])
            self.assertEqual(result['complete_workers_by_method'], {'D': 17, 'S': 18, 'H': 18})
            self.assertEqual(result['reference_resource_failures']['D'], 1)
            self.assertEqual(result['H_completed_against_reference_resource_failure']['D'], 1)
            self.assertEqual(result['aggregate_comparisons']['H_over_S'], 1.)
            self.assertEqual(result['aggregate_comparisons']['H_over_better_reference'], 1.)
            self.assertEqual(len(result['workers']), 54)
            # Numerical unresolved is a distinct nonresource outcome.
            for request in trace['requests'][:2]:
                request['status'] = request['decisions'][-1]['status'] = 'unresolved'
                request['decisions'][-1]['failure_reason'] = 'unresolved: synthetic injection'
            trace_path.write_text(json.dumps(trace))
            result = analyze(registration, raw, output, manifest)
            self.assertEqual(result['validation_issues'], [])
            self.assertEqual(result['reference_resource_failures']['D'], 0)
            self.assertEqual(result['H_completed_against_reference_resource_failure']['D'], 0)
            self.assertEqual(result['aggregate_comparisons']['H_over_S'], 1.)

    def test_seed_uses_exact_utf8_label_and_big_endian_first_eight_bytes(self):
        label = 'A2|development|2026-11-25|Alpha|slow'
        self.assertEqual(seed_label('development', 'Alpha', 'slow'), label)
        expected = int(hashlib.sha256(label.encode('utf-8')).hexdigest()[:16], 16)
        self.assertEqual(seed_for('development', 'Alpha', 'slow'), expected)
        self.assertNotEqual(seed_for('confirmation-1', 'Alpha', 'slow'), expected)
        self.assertNotEqual(seed_for('development', 'OTC', 'slow'), expected)
        with self.assertRaises(ValueError):
            seed_for('unregistered-seed', 'Alpha', 'slow')

    def test_development_has_exact_rotated_54_job_roster(self):
        schedule = list(jobs(protocol()))
        self.assertEqual(len(schedule), 54)
        self.assertEqual([j['method'] for j in schedule[:3]], ['D', 'S', 'H'])
        self.assertEqual([j['method'] for j in schedule[3:6]], ['S', 'H', 'D'])
        self.assertEqual([j['method'] for j in schedule[18:21]], ['S', 'H', 'D'])
        self.assertEqual(len({(j['graph'], j['stream'], j['seed_label'],
                               j['repeat_id'], j['method']) for j in schedule}), 54)
        for row in schedule:
            self.assertEqual(row['seed'], seed_for(row['seed_label'], row['graph'], row['stream']))

    def test_confirmation_has_exact_rotated_108_job_roster(self):
        schedule = list(jobs(protocol('confirmation')))
        self.assertEqual(len(schedule), 108)
        self.assertEqual({j['seed_label'] for j in schedule},
                         {'confirmation-1', 'confirmation-2'})
        self.assertEqual([j['method'] for j in schedule[36:39]], ['S', 'H', 'D'])
        self.assertEqual(len({(j['graph'], j['stream'], j['seed_label'],
                               j['repeat_id'], j['method']) for j in schedule}), 108)

    def test_protocol_cannot_drop_cells_repeats_or_resource_limits(self):
        base = protocol()
        bad = [dict(base, graphs=['Alpha']), dict(base, streams=['slow']),
               dict(base, repeat_count=2), dict(base, requests=16),
               dict(base, methods=base['methods'][:2]), dict(base, delta=1),
               dict(base, hot_size=63), dict(base, cpu=1),
               dict(base, limits=dict(base['limits'], worker_seconds=181))]
        for cfg in bad:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                validate_protocol(cfg)
        confirm = protocol('confirmation')
        confirm.pop('confirmation_standard')
        with self.assertRaisesRegex(ValueError, 'confirmation'):
            validate_protocol(confirm)

    def test_fixture_rejects_changed_file_or_reordered_rows(self):
        values = np.array([[-65536, 4], [5, 65536]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'opinions.npz'
            np.savez(path, opinions=values)
            record = dict(file=path.name, shape=[2, 2],
                          npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          matrix_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                          opinion_hashes=[hashlib.sha256(row.tobytes()).hexdigest() for row in values])
            np.testing.assert_array_equal(load_fixture(record, root), values)
            with self.assertRaisesRegex(ValueError, 'matrix or row'):
                load_fixture(dict(record, opinion_hashes=record['opinion_hashes'][::-1]), root)
            np.savez(path, opinions=values[::-1].copy())
            with self.assertRaisesRegex(ValueError, 'file'):
                load_fixture(record, root)

    def test_preparation_refuses_existing_manifest_or_fixture_directory_before_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registration = root / 'protocol.json'
            registration.write_text(json.dumps(protocol()))
            manifest = root / 'manifest.json'
            fixtures = root / 'fixtures'
            manifest.write_text('frozen sentinel')
            with self.assertRaises(FileExistsError):
                prepare(registration, manifest, fixtures)
            self.assertEqual(manifest.read_text(), 'frozen sentinel')
            manifest.unlink()
            fixtures.mkdir()
            with self.assertRaises(FileExistsError):
                prepare(registration, manifest, fixtures)

    def test_synthetic_fixture_freeze_is_independently_regenerated_and_immutable(self):
        from scipy import sparse
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projected = root / 'data/projected'
            projected.mkdir(parents=True)
            graph = sparse.csr_matrix(np.array([[0., -.5, 1.], [-.5, 0., 0.], [1., 0., 0.]]))
            for name in ('Alpha', 'OTC'):
                hashes = {}
                for kind, matrix in (('unit', graph.sign()), ('weighted', graph)):
                    path = projected / f'{name}_{kind}.npz'
                    sparse.save_npz(path, matrix)
                    hashes[kind + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                (projected / f'{name}_metadata.json').write_text(json.dumps(hashes))
            registration = root / 'protocol.json'
            registration.write_text(json.dumps(protocol()))
            manifest = root / 'manifest.json'
            sources = {'synthetic': 'f' * 64}
            with patch('src.a2_bench.ROOT', root), patch('src.a2_bench.source_hashes', return_value=sources), \
                    patch('src.a2_bench.input_source_hashes', return_value=sources), \
                    patch('src.a2_bench.environment', return_value={'synthetic': True}):
                frozen = prepare(registration, manifest, root / 'fixtures')
                self.assertEqual(frozen['independent_opinion_rows'], 24)
                self.assertEqual(len(frozen['fixtures']), 6)
                self.assertEqual(frozen['graphs']['Alpha']['catalog_size'], 2)
                cfg, reloaded, records, _ = load_freeze(registration, manifest)
                self.assertEqual(reloaded, frozen)
                self.assertEqual(len(records), 6)
                first_bytes = manifest.read_bytes()
                with self.assertRaises(FileExistsError):
                    prepare(registration, manifest, root / 'fixtures')
                self.assertEqual(manifest.read_bytes(), first_bytes)
                record = frozen['fixtures'][0]
                matrix = load_fixture(record, root)
                self.assertEqual(matrix.shape, (4, 3))
                with patch('src.a2_bench.source_hashes', return_value={'changed': '0' * 64}):
                    with self.assertRaisesRegex(ValueError, 'source'):
                        load_freeze(registration, manifest)
    def test_summary_compares_full_service_to_both_references_and_better(self):
        cfg = protocol()
        report = summarize(cfg, synthetic_observations(cfg))
        self.assertEqual(report['planned_workers'], 54)
        self.assertEqual(report['complete_workers'], 54)
        self.assertEqual(len(report['cells']), 6)
        self.assertEqual(report['full_schedule_totals_seconds']['H'], 90.)
        self.assertEqual(report['aggregate_comparisons']['H_over_D'], .25)
        self.assertEqual(report['aggregate_comparisons']['H_over_S'], .5)
        self.assertEqual(report['aggregate_comparisons']['H_over_better_reference'], .5)
        self.assertEqual(report['cells'][0]['comparisons']['Q1']['H_over_S']['ratios'], [.5] * 3)
        self.assertEqual(report['cells'][0]['comparisons']['Q4']['H_over_D']['ratios'], [.25] * 3)

    def test_failures_remain_in_fixed_denominator_and_partial_is_never_complete(self):
        cfg = protocol()
        observations = synthetic_observations(cfg)
        failed = next(row for row in observations if row['job']['method'] == 'D')
        failed['trace'].update(status='timeout', requests=failed['trace']['requests'][:1],
                               prefixes={'1': 8.}, service_accounted_seconds=8.)
        failed['scheduler'].update(status='timeout', returncode=-9, process_wall_seconds=180.)
        report = summarize(cfg, observations)
        self.assertEqual(report['planned_workers'], 54)
        self.assertEqual(report['complete_workers'], 53)
        self.assertEqual(report['status_counts']['timeout'], 1)
        self.assertIsNone(report['full_schedule_totals_seconds']['D'])
        self.assertIsNone(report['aggregate_comparisons']['H_over_D'])
        self.assertEqual(report['aggregate_comparisons']['H_over_S'], .5)
        self.assertEqual(report['aggregate_comparisons']['H_over_better_reference'], .5)
        self.assertEqual(report['reference_resource_failures']['D'], 1)
        self.assertEqual(len(report['workers']), 54)
        cell = next(c for c in report['cells'] if c['graph'] == failed['job']['graph']
                    and c['stream'] == failed['job']['stream'])
        self.assertEqual(cell['comparisons']['Q4']['H_over_D']['paired_count'], 2)
        self.assertEqual(cell['comparisons']['Q4']['H_over_D']['planned_pairs'], 3)
        self.assertIsNone(cell['comparisons']['Q4']['H_over_D']['full_cell_ratio'])
        self.assertEqual(report['observed_service_seconds_including_partial']['D'], 348.)

    def test_unresolved_and_missing_workers_cannot_be_counted_as_completed(self):
        cfg = protocol()
        observations = synthetic_observations(cfg)
        observations[0]['trace']['requests'][0]['status'] = 'unresolved'
        missing = observations.pop()
        report = summarize(cfg, observations)
        self.assertEqual(report['planned_workers'], 54)
        self.assertEqual(len(report['workers']), 54)
        self.assertEqual(report['complete_workers'], 52)
        self.assertEqual(report['status_counts']['missing_result'], 1)
        self.assertIsNone(report['full_schedule_totals_seconds'][missing['job']['method']])

    def test_q4_checkpoint_from_failed_process_is_not_a_complete_cost_pair(self):
        cfg = protocol()
        observations = synthetic_observations(cfg)
        failed = next(row for row in observations if row['job']['method'] == 'S')
        failed['trace']['status'] = 'timeout'
        failed['scheduler'].update(status='timeout', returncode=-9)
        report = summarize(cfg, observations)
        cell = next(c for c in report['cells'] if c['graph'] == failed['job']['graph']
                    and c['stream'] == failed['job']['stream'])
        comparison = cell['comparisons']['Q4']['H_over_S']
        self.assertEqual(comparison['paired_count'], 2)
        self.assertIsNone(comparison['pairs'][0]['reference_seconds'])
        # That matched repetition must use completed D, never the failed S Q4 checkpoint.
        better = cell['comparisons']['Q4']['H_over_better_reference']['pairs'][0]
        self.assertEqual(better['reference_method'], 'D')
        self.assertEqual(better['ratio'], .25)

    def test_request_validator_accepts_sparse_domain_and_continuous_chain(self):
        for values in (synthetic_request(), synthetic_commit_request()):
            request, trace, fixture, spec, W = values
            validate_request(request, trace, fixture, 1, spec, W)
        # A negative weighted-flip bound may be the only remaining frontier head.
        request['access']['epochs'][-1]['remaining_upper_coarse'] = -1
        validate_request(request, trace, fixture, 1, spec, W)

    def test_request_validator_rejects_hidden_catalog_and_stale_frontier(self):
        request, trace, fixture, spec, W = synthetic_commit_request()
        hidden = copy.deepcopy(request)
        hidden['memory']['candidate_array_rows'] = 2
        with self.assertRaisesRegex(ValueError, 'M-sized'):
            validate_request(hidden, trace, fixture, 1, spec, W)
        stale = copy.deepcopy(request)
        stale['access']['epochs'][1]['parent_version_id'] = trace['root_version_id']
        with self.assertRaisesRegex(ValueError, 'version'):
            validate_request(stale, trace, fixture, 1, spec, W)

    def test_request_validator_rejects_wrong_canonical_id_and_positive_edge(self):
        request, trace, fixture, spec, W = synthetic_commit_request()
        changed = copy.deepcopy(request)
        changed['decisions'][0]['winner_id'] = 3  # (1,0) is not canonical.
        with self.assertRaisesRegex(ValueError, 'domain'):
            validate_request(changed, trace, fixture, 1, spec, W)
        changed = copy.deepcopy(request)
        changed['initial_seed_ids'] = changed['initial_seed_order'] = [2]  # Root positive (0,2) is excluded.
        with self.assertRaisesRegex(ValueError, 'domain'):
            validate_request(changed, trace, fixture, 1, spec, W)

    def test_request_validator_rejects_weight_and_seed_order_corruption(self):
        request, trace, fixture, spec, W = synthetic_commit_request()
        altered = copy.deepcopy(request)
        altered['decisions'][0]['d'] = 1  # A natural weight-1/2 flip has scaled d=8.
        with self.assertRaisesRegex(ValueError, 'weight'):
            validate_request(altered, trace, fixture, 1, spec, W)
        altered = copy.deepcopy(request)
        altered['initial_seed_order'] = [1, 1]
        with self.assertRaisesRegex(ValueError, 'seed order'):
            validate_request(altered, trace, fixture, 1, spec, W)

    def test_request_validator_rejects_partial_budget_stop_and_bad_accounting(self):
        request, trace, fixture, spec, W = synthetic_request()
        request['status'] = 'budget_stop'
        request['decisions'] = []
        request['access']['epochs'] = []
        with self.assertRaisesRegex(ValueError, 'partial budget'):
            validate_request(request, trace, fixture, 1, spec, W)
        request, trace, fixture, spec, W = synthetic_request()
        request['phases']['other'] = .1
        with self.assertRaisesRegex(ValueError, 'accounting'):
            validate_request(request, trace, fixture, 1, spec, W)

    def test_all_54_synthetic_records_and_resource_failure_validate_without_dropping_jobs(self):
        with synthetic_bundle() as (registration, manifest, raw, output), redirect_stdout(io.StringIO()):
            result = analyze(registration, raw, output, manifest)
            self.assertEqual(result['validation_issues'], [])
            self.assertEqual(result['complete_workers'], 54)
            scheduler = json.loads((raw / 'SCHEDULE_RESULTS.json').read_text())
            row = scheduler['jobs'][0]
            trace_path = raw / row['result']
            trace = json.loads(trace_path.read_text())
            trace.update(status='timeout', failure_reason='timeout', requests=trace['requests'][:1],
                         prefixes={'1': 1.3}, service_accounted_seconds=1.3)
            trace_path.write_text(json.dumps(trace))
            row.update(status='timeout', returncode=-9, process_wall_seconds=180., completed_requests=1)
            scheduler['status'] = 'completed_with_failures'
            (raw / 'SCHEDULE_RESULTS.json').write_text(json.dumps(scheduler))
            result = analyze(registration, raw, output, manifest)
            self.assertEqual(result['validation_issues'], [])
            self.assertTrue(result['correctness_accounting_pass'])
            self.assertEqual(result['complete_workers'], 53)
            self.assertEqual(len(result['workers']), 54)
            self.assertIsNone(result['aggregate_comparisons']['H_over_D'])
            self.assertEqual(result['aggregate_comparisons']['H_over_S'], 1.)

    def test_roster_tampering_invalidates_full_schedule_comparisons(self):
        with synthetic_bundle() as (registration, manifest, raw, output), redirect_stdout(io.StringIO()):
            path = raw / 'FROZEN_JOBS.json'
            frozen = json.loads(path.read_text())
            frozen['jobs'] = frozen['jobs'][:-1]
            path.write_text(json.dumps(frozen))
            result = analyze(registration, raw, output, manifest)
            self.assertFalse(result['correctness_accounting_pass'])
            self.assertEqual(len(result['workers']), 54)
            self.assertTrue(any('roster' in issue for issue in result['validation_issues']))
            self.assertIsNone(result['aggregate_comparisons']['H_over_S'])

    def test_invalid_json_object_retains_the_entire_registered_denominator(self):
        with synthetic_bundle() as (registration, manifest, raw, output), redirect_stdout(io.StringIO()):
            first = json.loads((raw / 'SCHEDULE_RESULTS.json').read_text())['jobs'][0]
            (raw / first['result']).write_text('[]')
            result = analyze(registration, raw, output, manifest)
            self.assertFalse(result['correctness_accounting_pass'])
            self.assertEqual(len(result['workers']), 54)
            self.assertEqual(result['complete_workers'], 53)
            self.assertEqual(result['aggregate_comparisons']['H_over_S'], 1.)

    def test_corrupt_request_shape_is_reported_without_losing_other_workers(self):
        with synthetic_bundle() as (registration, manifest, raw, output), redirect_stdout(io.StringIO()):
            first = json.loads((raw / 'SCHEDULE_RESULTS.json').read_text())['jobs'][0]
            path = raw / first['result']
            trace = json.loads(path.read_text())
            trace['requests'][0] = 'corrupt request'
            path.write_text(json.dumps(trace))
            result = analyze(registration, raw, output, manifest)
            self.assertFalse(result['correctness_accounting_pass'])
            self.assertEqual(len(result['workers']), 54)
            self.assertEqual(result['complete_workers'], 53)
            self.assertEqual(result['aggregate_comparisons']['H_over_S'], 1.)

    def test_actual_small_worker_reconciles_complete_cost_and_protects_checkpoint(self):
        initial_affinity = os.sched_getaffinity(0)
        initial_env = {name: os.environ.get(name) for name in THREAD_ENV}
        try:
            with synthetic_bundle() as (registration, manifest, raw, output):
                cfg, frozen, records, manifest_hash = load_freeze(registration, manifest)
                path = raw.parent / 'actual_worker.json'
                trace = worker(registration, manifest, path, 0, manifest_hash)
                self.assertEqual(trace['status'], 'completed', trace.get('failure_reason'))
                self.assertEqual(len(trace['requests']), 4)
                self.assertGreaterEqual(trace['structural_seconds'], trace['session_setup_seconds'])
                self.assertGreater(trace['structural_phases']['graph_load'], 0.)
                self.assertAlmostEqual(sum(trace['structural_phases'].values()), trace['structural_seconds'])
                self.assertAlmostEqual(trace['prefixes']['4'], trace['structural_seconds'] +
                                       sum(q['request_seconds'] for q in trace['requests']))
                W = synthetic_request()[-1]
                job = list(jobs(cfg))[0]
                fixture = records[job['graph'], job['stream'], job['seed_label']]
                for q, request in enumerate(trace['requests'], 1):
                    validate_request(request, trace, fixture, q, frozen['graphs'][job['graph']], W)
                with self.assertRaises(FileExistsError):
                    worker(registration, manifest, path, 0, manifest_hash)
        finally:
            os.sched_setaffinity(0, initial_affinity)
            for name, value in initial_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == '__main__':
    unittest.main()
