import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.access_bench import catalog_visits, jobs, load_fixture, validate_protocol


def protocol():
    return {
        'id': 'test',
        'graphs': ['Alpha', 'OTC'],
        'streams': ['slow', 'independent', 'signed-neighbor'],
        'seeds': [262147, 262151],
        'requests': 16,
        'budget': 15,
        'repeat_count': 2,
        'coefficients': [0.125, 0.25, 0.5],
        'methods': [
            {'arm': 'rebuild-always', 'label': 'rebuild'},
            {'arm': 'retain-always', 'label': 'retain'},
            {'arm': 'budget-guard', 'label': 'guard_0.125', 'coefficient': 0.125},
            {'arm': 'budget-guard', 'label': 'guard_0.25', 'coefficient': 0.25},
            {'arm': 'budget-guard', 'label': 'guard_0.5', 'coefficient': 0.5},
        ],
    }


class AccessBenchTests(unittest.TestCase):
    def test_schedule_preserves_and_rotates_all_registered_cells(self):
        schedule = list(jobs(validate_protocol(protocol())))
        self.assertEqual(len(schedule), 120)
        labels = ['rebuild', 'retain', 'guard_0.125', 'guard_0.25', 'guard_0.5']
        self.assertEqual([j['label'] for j in schedule[:5]], labels)
        self.assertEqual([j['label'] for j in schedule[5:10]], labels[1:] + labels[:1])
        self.assertEqual([j['label'] for j in schedule[60:65]], labels[1:] + labels[:1])
        self.assertEqual(len({(j['seed'], j['stream'], j['graph'], j['repeat_id'], j['label'])
                              for j in schedule}), 120)

    def test_duplicate_seed_cannot_silently_repeat_an_input_block(self):
        with self.assertRaisesRegex(ValueError, 'seeds'):
            validate_protocol(protocol() | {'seeds': [262147, 262147]})

    def test_guard_budget_is_decimal_floor_in_catalog_raw_visits(self):
        self.assertEqual(catalog_visits(0.125, 552388), 69048)
        self.assertEqual(catalog_visits(0.5, 1201725), 600862)
        self.assertEqual(catalog_visits(0.29, 100), 29)

    def test_fixture_rejects_changed_file_and_reordered_row_binding(self):
        values = np.array([[-65536, 4], [5, 65536]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'opinions.npz'
            np.savez(path, opinions=values)
            record = {
                'file': path.name,
                'shape': [2, 2],
                'npz_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'matrix_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
                'opinion_hashes': [hashlib.sha256(row.tobytes()).hexdigest() for row in values],
            }
            np.testing.assert_array_equal(load_fixture(record, root), values)
            with self.assertRaisesRegex(ValueError, 'matrix or row'):
                load_fixture(record | {'opinion_hashes': record['opinion_hashes'][::-1]}, root)
            np.savez(path, opinions=values[::-1].copy())
            with self.assertRaisesRegex(ValueError, 'file'):
                load_fixture(record, root)


if __name__ == '__main__':
    unittest.main()
