"""Read-only archive checks and temporary retest manifests; no benchmark workers."""
import copy
from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src import a2_analyze, a2_bench

try:
    from src import a2_tie_retest as retest
except ImportError:
    retest = None


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def fixed_input_copy():
    """Copy archived bytes, never generate a replacement opinion matrix."""
    original = json.loads((ROOT / 'protocol/A2_CONFIRMATION_INPUTS.json').read_text())
    cfg = json.loads((ROOT / 'protocol/A2_TIE_RETEST.json').read_text())
    relative_paths = {'protocol/A2_TIE_RETEST.json'}
    relative_paths.update(row['path'] for row in cfg['original_confirmation'].values()
                          if isinstance(row, dict))
    relative_paths.update(str((Path('protocol') / row['file']).as_posix())
                          for row in original['fixtures'])
    relative_paths.update(spec[key] for spec in original['graphs'].values()
                          for key in ('metadata_path', 'unit_path', 'weighted_path'))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for relative in relative_paths:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        sources = copy.deepcopy(original['source_files_sha256'])
        sources.update({'src/a2_session.py': '1' * 64,
                        'src/weighted_adapter.py': '2' * 64,
                        'src/a2_tie_retest.py': '3' * 64,
                        'src/exact_ties.py': '4' * 64})
        # Only source/environment lookups are substituted. Archive JSON, graph
        # files, fixture bytes, matrix/row hashes and the roster are all real.
        with patch.object(retest, 'ROOT', root), patch.object(a2_bench, 'ROOT', root), \
                patch.object(a2_analyze, 'ROOT', root), \
                patch.object(retest, 'source_hashes', return_value=sources), \
                patch.object(a2_bench, 'input_source_hashes',
                             return_value=original['input_source_files_sha256']), \
                patch.object(a2_bench, 'environment', return_value=original['environment']):
            yield root, original, sources


class A2TieRetestTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(retest, 'fixed-input revision retest harness is not implemented')

    def test_reuses_exact_fixture_records_and_original_rotated_roster(self):
        # Catches any regenerated opinion, re-rooted fixture path, renamed seed,
        # reordered method, missing cell, or discarded historical source binding.
        with fixed_input_copy() as (root, original, sources):
            protocol = root / 'protocol/A2_TIE_RETEST.json'
            manifest_path = root / 'protocol/A2_TIE_RETEST_INPUTS.json'
            originals = {p: digest(p) for p in root.rglob('*') if p.is_file()}
            manifest = retest.prepare(protocol, manifest_path)
            cfg, loaded, records, manifest_hash = retest.load_freeze(protocol, manifest_path)
            archived_jobs = json.loads((root / 'raw/a2_confirmation/FROZEN_JOBS.json').read_text())['jobs']
            self.assertEqual(list(a2_bench.jobs(cfg)), archived_jobs)
            self.assertEqual(len(archived_jobs), 108)
            self.assertEqual(len(records), 12)
            self.assertEqual(loaded, manifest)
            self.assertEqual(manifest_hash, digest(manifest_path))
            self.assertEqual(manifest['fixtures'], original['fixtures'])
            self.assertEqual(manifest['graphs'], original['graphs'])
            self.assertEqual(manifest['input_source_files_sha256'], original['input_source_files_sha256'])
            self.assertEqual(manifest['source_files_sha256'], sources)
            self.assertEqual(manifest['original_source_binding']['source_files_sha256'],
                             original['source_files_sha256'])
            self.assertEqual(manifest['stage'], 'revision_retest')
            self.assertEqual(manifest['study_label'], 'revision retest on diagnosed fixed inputs')
            self.assertFalse(manifest['unseen_confirmation'])
            self.assertTrue(manifest['inputs_reused_without_regeneration'])
            self.assertEqual(manifest['original_confirmation']['status'], 'failed')
            self.assertEqual({p: digest(p) for p in originals}, originals)

    def test_rejects_changed_fixture_bytes_before_freeze(self):
        with fixed_input_copy() as (root, original, _):
            fixture = root / 'protocol' / original['fixtures'][0]['file']
            with fixture.open('ab') as handle:
                handle.write(b'changed archive bytes')
            manifest = root / 'protocol/A2_TIE_RETEST_INPUTS.json'
            with self.assertRaisesRegex(ValueError, 'checksum|digest|hash'):
                retest.prepare(root / 'protocol/A2_TIE_RETEST.json', manifest)
            self.assertFalse(manifest.exists())

    def test_rejects_missing_fixture_without_regeneration(self):
        with fixed_input_copy() as (root, original, _):
            fixture = root / 'protocol' / original['fixtures'][0]['file']
            fixture.unlink()
            with self.assertRaises(FileNotFoundError):
                retest.prepare(root / 'protocol/A2_TIE_RETEST.json',
                               root / 'protocol/A2_TIE_RETEST_INPUTS.json')
            self.assertFalse(fixture.exists())

    def test_rejects_modified_original_protocol_manifest_roster_or_failed_status(self):
        for relative in ('protocol/A2_CONFIRMATION.json', 'protocol/A2_CONFIRMATION_INPUTS.json',
                         'raw/a2_confirmation/FROZEN_JOBS.json',
                         'reports/A2_CONFIRMATION_JUDGMENT.json'):
            with self.subTest(relative=relative), fixed_input_copy() as (root, _, _):
                path = root / relative
                data = json.loads(path.read_text())
                data['status'] = 'passed'
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'original|archived'):
                    retest.prepare(root / 'protocol/A2_TIE_RETEST.json',
                                   root / 'protocol/A2_TIE_RETEST_INPUTS.json')

    def test_refuses_wrong_fixture_location_and_existing_manifest(self):
        with fixed_input_copy() as (root, _, _):
            protocol = root / 'protocol/A2_TIE_RETEST.json'
            existing = root / 'protocol/existing.json'
            existing.write_bytes(b'preserve these bytes\n')
            with self.assertRaises(FileExistsError):
                retest.prepare(protocol, existing)
            self.assertEqual(existing.read_bytes(), b'preserve these bytes\n')
            with self.assertRaisesRegex(ValueError, 'same directory|sibling'):
                retest.prepare(protocol, root / 'elsewhere/manifest.json')

    def test_rejects_renamed_seed_labels_and_relaxed_resource_caps(self):
        for field, bad in (('seed_labels', ['C1', 'C2']), ('budget', 16),
                           ('limits', {'worker_seconds': 181, 'rss_mib': 2048,
                                       'batch_seconds': 3600, 'rss_sample_seconds': .025})):
            with self.subTest(field=field), fixed_input_copy() as (root, _, _):
                path = root / 'protocol/A2_TIE_RETEST.json'
                cfg = json.loads(path.read_text())
                cfg[field] = bad
                path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):
                    retest.read_protocol(path)

    def test_rejects_changed_access_source_and_runtime_environment(self):
        with fixed_input_copy() as (root, original, sources):
            sources['src/a2_access.py'] = '5' * 64
            with self.assertRaisesRegex(ValueError, 'scope|unchanged|unauthorized'):
                retest.prepare(root / 'protocol/A2_TIE_RETEST.json',
                               root / 'protocol/A2_TIE_RETEST_INPUTS.json')
            sources['src/a2_access.py'] = original['source_files_sha256']['src/a2_access.py']
            with patch.object(a2_bench, 'environment', return_value={'changed': True}):
                with self.assertRaisesRegex(ValueError, 'environment'):
                    retest.prepare(root / 'protocol/A2_TIE_RETEST.json',
                                   root / 'protocol/A2_TIE_RETEST_INPUTS.json')

    def test_load_rejects_changed_fixture_or_new_source_binding(self):
        for mutation in ('fixture', 'resolver', 'manifest'):
            with self.subTest(mutation=mutation), fixed_input_copy() as (root, original, sources):
                protocol = root / 'protocol/A2_TIE_RETEST.json'
                path = root / 'protocol/A2_TIE_RETEST_INPUTS.json'
                retest.prepare(protocol, path)
                if mutation == 'fixture':
                    fixture = root / 'protocol' / original['fixtures'][0]['file']
                    fixture.write_bytes(b'not the frozen NPZ')
                elif mutation == 'resolver':
                    sources['src/exact_ties.py'] = '6' * 64
                else:
                    manifest = json.loads(path.read_text())
                    manifest['fixtures'][0]['file'] = 'substituted.npz'
                    path.write_text(json.dumps(manifest))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    retest.load_freeze(protocol, path)

    def test_existing_worker_batch_and_analysis_outputs_are_not_overwritten(self):
        with fixed_input_copy() as (root, _, _):
            protocol = root / 'protocol/A2_TIE_RETEST.json'
            manifest = root / 'protocol/A2_TIE_RETEST_INPUTS.json'
            retest.prepare(protocol, manifest)
            existing = root / 'already-written.json'
            existing.write_bytes(b'keep me\n')
            raw = root / 'raw/retest'
            raw.mkdir()
            sentinel = raw / 'do-not-retry'
            sentinel.write_bytes(b'no rerun\n')
            with self.assertRaises(FileExistsError):
                retest.worker(protocol, manifest, existing, 0, digest(manifest))
            with self.assertRaises(FileExistsError):
                retest.batch(protocol, manifest, raw)
            with self.assertRaises(FileExistsError):
                retest.analyze(protocol, raw, existing, manifest)
            self.assertEqual(existing.read_bytes(), b'keep me\n')
            self.assertEqual(sentinel.read_bytes(), b'no rerun\n')

    def test_worker_checks_new_manifest_binding_before_affinity_or_graph_loading(self):
        # No worker process or numerical session is launched. Deliberately reject
        # a wrong manifest binding before the old worker reaches CPU0/session work.
        with fixed_input_copy() as (root, _, _):
            protocol = root / 'protocol/A2_TIE_RETEST.json'
            manifest = root / 'protocol/A2_TIE_RETEST_INPUTS.json'
            retest.prepare(protocol, manifest)
            with patch.object(a2_bench, '_graph_record',
                              side_effect=AssertionError('pre-affinity graph load')):
                result = retest.worker(protocol, manifest, root / 'rejected-worker.json', 0, '0' * 64)
            self.assertEqual(result['status'], 'error')
            self.assertIn('worker manifest differs', result['failure_reason'])
            self.assertEqual(result['requests'], [])
            self.assertEqual(result['study_label'], 'revision retest on diagnosed fixed inputs')
            self.assertEqual(result['original_confirmation_status'], 'failed')

    def test_supervisor_routes_only_worker_module_and_retains_resource_arguments(self):
        # Replacing the process-launch boundary is necessary to avoid a real run.
        command = ['python', '-m', 'src.a2_bench', 'worker', '--schedule-position', '31']
        limits = {'worker_seconds': 180, 'rss_mib': 2048, 'batch_seconds': 3600,
                  'rss_sample_seconds': .025}
        def no_process(*args):
            return args
        with patch.object(a2_bench, 'supervise', side_effect=no_process):
            actual = retest._supervise(command, Path('worker.log'), 181., 3601., limits)
        self.assertEqual(actual, (['python', '-m', 'src.a2_tie_retest', 'worker',
                                   '--schedule-position', '31'], Path('worker.log'), 181., 3601., limits))
        self.assertEqual(command[2], 'src.a2_bench')

    def test_source_hashes_bind_new_exact_tie_module_and_entry_point(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/exact_ties.py').write_bytes(b'exact certificate module\n')
            (root / 'src/a2_tie_retest.py').write_bytes(b'retest entry point\n')
            with patch.object(retest, 'ROOT', root), \
                    patch.object(a2_bench, 'source_hashes', return_value={'old.py': 'a' * 64}):
                actual = retest.source_hashes()
            self.assertEqual(actual, {'old.py': 'a' * 64,
                'src/exact_ties.py': digest(root / 'src/exact_ties.py'),
                'src/a2_tie_retest.py': digest(root / 'src/a2_tie_retest.py')})

    def test_normal_analyzer_retains_missing_full_roster_and_retest_label(self):
        # Missing results exercise the real analyzer, without timing or replay.
        with fixed_input_copy() as (root, _, _), redirect_stdout(io.StringIO()):
            protocol = root / 'protocol/A2_TIE_RETEST.json'
            manifest = root / 'protocol/A2_TIE_RETEST_INPUTS.json'
            retest.prepare(protocol, manifest)
            raw = root / 'raw/retest-empty'
            raw.mkdir()
            output = root / 'analysis.json'
            result = retest.analyze(protocol, raw, output, manifest)
            self.assertEqual(result['planned_workers'], 108)
            self.assertEqual(result['complete_workers'], 0)
            self.assertEqual(len(result['cells']), 12)
            self.assertFalse(result['correctness_accounting_pass'])
            self.assertEqual(result['study_label'], 'revision retest on diagnosed fixed inputs')
            self.assertEqual(result['original_confirmation_status'], 'failed')
            self.assertFalse(result['unseen_confirmation'])
            self.assertTrue(all(value is None for value in result['aggregate_comparisons'].values()))
            self.assertEqual(json.loads(output.read_text()), result)


if __name__ == '__main__':
    unittest.main()
