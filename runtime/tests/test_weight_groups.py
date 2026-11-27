import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from src.anchor_session import ARMS, AnchorSession
from src.session import Session
from src.weighted_adapter import Index


def weighted_fixture():
    # Catalog IDs: flip (0,1), insert (0,2), flip (1,2), insert (1,3).
    return sparse.csr_matrix([[0., -1.5, 0., 0.], [-1.5, 0., -2.5, 0.],
                              [0., -2.5, 0., .5], [0., 0., .5, 0.]])


class WeightGroupsTests(unittest.TestCase):
    def test_grouping_runs_once_for_multiple_weighted_index_builds(self):
        unique = np.unique
        grouping_calls = []

        def observe_grouping(values, *args, **kwargs):
            if np.array_equal(values, [24, 1, 40, 1]):
                grouping_calls.append(values)
            return unique(values, *args, **kwargs)

        with patch.object(np, 'unique', observe_grouping):
            session = Session(weighted_fixture(), 'I', weight_mode='weighted')
            session.request(np.array([100, 20, -10, 10], np.int64), budget=0)
            session.request(np.array([-10, 20, -10, 10], np.int64), budget=0)
        self.assertEqual(len(grouping_calls), 1)
        self.assertIs(session.weight_groups, session.kernel.prepare_weight_groups())

    def test_rebuild_reads_new_caps_and_preserves_canonical_ties(self):
        session = Session(weighted_fixture(), 'D', weight_mode='weighted')
        x = np.zeros(4, np.int64)
        first = Index(session.kernel, x, 0, session.inc, session.degree,
                      caps=np.array([12, 9, 15, 3], np.int64))
        tied = Index(session.kernel, x, 0, session.inc, session.degree,
                     caps=np.array([0, 5, 1, 5], np.int64))

        def triples(index):
            return [(d, keys.tolist(), ids.tolist()) for d, keys, ids in index.groups]

        self.assertEqual(triples(first), [(1, [3, 9], [3, 1]), (24, [12], [0]), (40, [15], [2])])
        self.assertEqual(triples(tied), [(1, [5, 5], [1, 3]), (24, [0], [0]), (40, [1], [2])])
        self.assertEqual([(d, ids.tolist()) for d, ids in session.weight_groups.members],
                         [(1, [1, 3]), (24, [0]), (40, [2])])
        self.assertEqual(set(first.build_phases), {'key_gather', 'stable_sort'})
        self.assertGreater(first.build_phases['key_gather'], 0)
        self.assertGreater(first.build_phases['stable_sort'], 0)

    def test_group_members_catalog_and_weights_are_immutable(self):
        session = Session(weighted_fixture(), 'D', weight_mode='weighted')
        groups = session.weight_groups
        members = groups.members[0][1]
        for values in (members, *session.pool, session.kernel.d):
            with self.assertRaises(ValueError):
                values[0] = 0
            with self.assertRaises(ValueError):
                values.flags.writeable = True
        self.assertEqual(groups.nbytes, 4 * np.dtype(np.int32).itemsize)
        self.assertEqual(session.weight_group_bytes, groups.nbytes)
        self.assertGreater(session.setup_phases['structural_weight_grouping'], 0)

    def test_replaced_structural_arrays_are_rejected(self):
        opinion = np.zeros(4, np.int64)
        for target, field in (('kernel', 'd'), ('kernel', 'pool'), ('session', 'pool')):
            with self.subTest(target=target, field=field):
                session = Session(weighted_fixture(), 'D', weight_mode='weighted')
                obj = session.kernel if target == 'kernel' else session
                old = getattr(obj, field)
                new = tuple(a.copy() for a in old) if field == 'pool' else old.copy()
                setattr(obj, field, new)
                with self.assertRaisesRegex(ValueError, 'weight group binding'):
                    session.request(opinion, budget=0)

    def test_structural_stride_tampering_is_rejected(self):
        for array_index in range(4):
            with self.subTest(array_index=array_index):
                session = Session(weighted_fixture(), 'D', weight_mode='weighted')
                values = (*session.pool, session.kernel.d)[array_index]
                values.strides = (0,)
                with self.assertRaisesRegex(ValueError, 'weight group binding'):
                    session.request(np.zeros(4, np.int64), budget=0)

    def test_every_anchor_arm_reuses_group_preparation(self):
        s = np.array([100, 20, -10, 10], np.int64)
        for arm in ARMS:
            with self.subTest(arm=arm):
                session = AnchorSession(weighted_fixture(), arm)
                groups = session.weight_groups
                baseline = session.request(s, budget=0)
                later = session.request(s[::-1].copy(), budget=0)
                self.assertIs(groups, session.bridge.kernel.prepare_weight_groups())
                self.assertGreater(baseline['index_build_phases']['stable_sort'], 0)
                self.assertEqual(later['memory']['structural_weight_grouping'], groups.nbytes)
                if arm == 'rebuild-always':
                    self.assertGreater(later['index_build_phases']['stable_sort'], 0)
                else:
                    self.assertEqual(later['index_build_phases']['stable_sort'], 0)

    def test_empty_catalog_and_nonindexed_structural_cost(self):
        session = Session(sparse.csr_matrix((2, 2)), 'D', weight_mode='weighted')
        self.assertEqual(session.weight_groups.members, ())
        self.assertEqual(session.weight_group_bytes, 0)
        result = session.request(np.zeros(2, np.int64), budget=1)
        self.assertEqual(result['commits'], 0)
        self.assertEqual(result['initial_seed_ids'], [])
        self.assertEqual(result['index_build_phases'], {'key_gather': 0., 'stable_sort': 0.})
        for method in 'FTS':
            nonindexed = Session(weighted_fixture(), method, weight_mode='weighted')
            self.assertIsNone(nonindexed.weight_groups)
            self.assertEqual(nonindexed.weight_group_bytes, 0)
            self.assertEqual(nonindexed.setup_phases.get('structural_weight_grouping', 0.), 0.)
