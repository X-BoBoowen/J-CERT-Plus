import unittest

import numpy as np
from scipy import sparse

from src.a2_session import A2Session
from src.common import DEN


class A2TieIntegrationTests(unittest.TestCase):
    def test_all_access_methods_record_the_same_exact_current_parent_tie(self):
        W = sparse.csr_matrix((17, 17), dtype=float)
        s = np.zeros(17, np.int64)
        s[0], s[1], s[2] = DEN, -DEN, -DEN
        signatures = []
        for method in ('D', 'S', 'H'):
            session = A2Session(W, method, hot_size=1, timers=False)
            result = session.request(s, budget=1)
            self.assertEqual(result['commits'], 1, (method, result))
            row = result['decisions'][0]
            self.assertEqual(row['winner_id'], 1)
            self.assertGreater(row['low'], 0)
            resolution = row['comparison_resolution']
            self.assertEqual(resolution['canonical_winner_id'], 1)
            self.assertEqual(resolution['parent_version_id'], row['parent_version_id'])
            self.assertEqual(len(resolution['witnesses']), 1)
            proof = resolution['witnesses'][0]
            self.assertEqual(proof['operator_sha256'], session.operator_hash)
            self.assertEqual(proof['opinion_sha256'], result['opinion_hash'])
            self.assertEqual(proof['swap'], [1, 2])
            self.assertEqual(result['counters']['structural_tie_decisions'], 1)
            signatures.append((row['winner_id'], row['current_version_id']))
        self.assertTrue(all(item == signatures[0] for item in signatures))

    def test_partial_large_tie_family_is_reported_and_does_not_commit(self):
        W = sparse.csr_matrix((17, 17), dtype=float)
        s = np.array([DEN if i % 2 else -DEN for i in range(17)], np.int64)
        result = A2Session(W, 'H', hot_size=1, timers=False).request(s, budget=1)
        self.assertEqual(result['status'], 'unresolved')
        self.assertEqual(result['commits'], 0)
        evidence = result['decisions'][0]['comparison_resolution']
        self.assertTrue(evidence['witnesses'])
        self.assertTrue(evidence['unresolved_ids'])
        self.assertNotIn('canonical_winner_id', evidence)
        self.assertEqual(result['counters']['structural_tie_decisions'], 0)


if __name__ == '__main__':
    unittest.main()
