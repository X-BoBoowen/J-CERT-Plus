import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from scipy import sparse

from src import anchor_session as access
from src.checker import CatalogReplay, worker_request
from src.common import opinions
from src.weighted_adapter import Index
from test_anchor_session import graph


class CoverageBudgetTests(unittest.TestCase):
    def setUp(self):
        self.W = graph()
        self.s, self.t = opinions(8, 'independent', 104729, 2)

    def test_late_fallback_preserves_current_decision_and_grams(self):
        guard = access.AnchorSession(self.W, 'budget-guard', visit_budget=15, hot_size=3)
        guard.request(self.s, budget=4)
        old_index = guard.bridge.anchor.index
        source_by_gather = []
        original = access.gather

        def observe(index, *args, **kwargs):
            source_by_gather.append(index)
            return original(index, *args, **kwargs)

        with patch.object(access, 'gather', observe):
            result = guard.request(self.t, budget=4, audit=True)
        d = result['discovery']
        events = d['epochs']
        self.assertEqual([e['action'] for e in events], ['retain', 'fallback', 'fresh', 'fresh'])
        self.assertEqual([e['consumed'] for e in events], [15] * 4)
        self.assertEqual(events[0]['pre_dedup_visits'], 15)
        self.assertEqual(d['fallback_count'], 1)
        self.assertIs(source_by_gather[0], old_index)
        fresh = guard.bridge.anchor.index
        self.assertTrue(all(source is fresh for source in source_by_gather[1:]))
        self.assertEqual(len(source_by_gather), len(events))
        switch = events[1]
        self.assertEqual(switch['rejected_plan']['anchor_request'], 1)
        self.assertEqual(switch['rejected_plan']['pre_dedup_visits'], 15)
        self.assertEqual(switch['anchor_request'], 2)
        self.assertEqual(switch['parent_version_id'], result['decisions'][0]['current_version_id'])
        self.assertEqual(switch['anchor_xc'], switch['current_root_xc'])
        self.assertNotEqual(switch['anchor_xc'], switch['current_xc'])
        self.assertGreater(d['sorted_index_overlap_bytes'], d['retained_key_coordinate_bytes'])
        self.assertEqual(d['root_caps_retention_peak_bytes'], 8 * guard.m)
        self.assertEqual(d['root_caps_retained_at_end_bytes'], 0)

        rebuild = access.AnchorSession(self.W, 'rebuild-always', hot_size=3)
        reference = rebuild.request(self.t, budget=4)
        self.assertEqual([entry['winner_id'] for entry in result['decisions']], [5, 4, 9, 1])
        self.assertEqual(result['initial_seed_ids'], reference['initial_seed_ids'])
        self.assertEqual(result['gain_low'], reference['gain_low'])
        self.assertNotEqual(events[0]['pre_dedup_visits'], reference['discovery']['epochs'][0]['pre_dedup_visits'])
        for key in ('gram_hits', 'transfers', 'candidate_rhs', 'response_blocks'):
            self.assertEqual(result['counters'][key], reference['counters'][key])
        self.assertGreater(result['counters']['transfers'], 0)
        self.assertEqual(worker_request(CatalogReplay(self.W), result, self.t,
                                        guard.graph_hash, kind='weighted')['accepts'], 4)

        guard.bridge.visit_budget = 1000
        following = guard.request(self.t, budget=1)
        self.assertEqual(following['discovery']['initial_anchor_request'], 2)
        self.assertEqual(following['discovery']['epochs'][0]['action'], 'retain')
        self.assertEqual(following['discovery']['epochs'][0]['consumed_before'], 0)

    def test_denied_initial_old_plan_does_not_reach_gather(self):
        guard = access.AnchorSession(self.W, 'budget-guard', visit_budget=0, hot_size=3)
        guard.request(self.s, budget=0)
        old = guard.bridge.anchor.index
        calls = []
        original = access.gather

        def observe(index, *args, **kwargs):
            calls.append(index)
            return original(index, *args, **kwargs)

        with patch.object(access, 'gather', observe):
            result = guard.request(self.t, budget=1)
        event = result['discovery']['epochs'][0]
        self.assertEqual(event['action'], 'fallback')
        self.assertEqual(event['fallback_reason'], 'visit_budget_exceeded')
        self.assertEqual(event['rejected_plan']['pre_dedup_visits'], 15)
        self.assertEqual(event['consumed'], 0)
        self.assertEqual(len(calls), 1)
        self.assertIsNot(calls[0], old)
        self.assertNotIn('returned_ids_sha256', event)

    def test_replaced_source_anchor_is_rejected_even_with_matching_fields(self):
        guard = access.AnchorSession(self.W, 'budget-guard', visit_budget=100)
        guard.request(self.s, budget=0)
        guard.bridge.anchor = replace(guard.bridge.anchor)
        with self.assertRaisesRegex(ValueError, 'anchor binding mismatch'):
            guard.request(self.t, budget=1)

    def test_live_decision_binding_cannot_change_between_preview_and_gather(self):
        original = access.Cursor.preview
        for field in ('opinion', 'parent'):
            with self.subTest(field=field):
                guard = access.AnchorSession(self.W, 'budget-guard', visit_budget=100)
                guard.request(self.s, budget=0)

                def change(cursor, *args):
                    ticket = original(cursor, *args)
                    setattr(cursor.bridge, field, 'changed')
                    return ticket

                with patch.object(access.Cursor, 'preview', change), \
                        self.assertRaisesRegex(ValueError, 'decision binding mismatch'):
                    guard.request(self.t, budget=1)

    def test_zero_budget_admits_empty_old_catalog(self):
        W = sparse.csr_matrix((2, 2))
        zero = np.zeros(2, np.int64)
        guard = access.AnchorSession(W, 'budget-guard', visit_budget=0)
        guard.request(zero, budget=0)
        result = guard.request(zero, budget=1)
        e = result['discovery']['epochs'][0]
        self.assertEqual((e['action'], e['pre_dedup_visits'], e['consumed']), ('retain', 0, 0))
        self.assertEqual(e['group_tails'], [])
        self.assertEqual(result['status'], 'tolerance_stop')

    def test_fallback_preserves_ties_nonpositive_stops_and_unresolved_overlap(self):
        tie_graph = sparse.csr_matrix([[0., -1.5, 0.], [-1.5, 0., -1.5], [0., -1.5, 0.]])
        guard = access.AnchorSession(tie_graph, 'budget-guard', visit_budget=0, hot_size=0)
        zero = np.zeros(3, np.int64)
        tie = np.array([65536, -65536, 65536], np.int64)
        guard.request(zero, budget=0)
        winner = guard.request(tie, budget=1)
        self.assertEqual(winner['discovery']['fallback_count'], 1)
        self.assertEqual(winner['decisions'][0]['winner_id'], 0)
        self.assertTrue(winner['decisions'][0]['exact_overlap_resolution'])
        worker_request(CatalogReplay(tie_graph), winner, tie, guard.graph_hash, kind='weighted')
        for s in (zero, np.full(3, 65536, np.int64)):
            stop = guard.request(s, budget=1)
            self.assertEqual(stop['status'], 'tolerance_stop')
            self.assertEqual(stop['commits'], 0)
            worker_request(CatalogReplay(tie_graph), stop, s, guard.graph_hash, kind='weighted')

        larger = sparse.block_diag((tie_graph, sparse.csr_matrix((14, 14))), format='csr')
        guard = access.AnchorSession(larger, 'budget-guard', visit_budget=0, hot_size=0)
        guard.request(np.zeros(17, np.int64), budget=0)
        unresolved = guard.request(np.r_[tie, np.zeros(14, np.int64)], budget=1)
        self.assertEqual(unresolved['discovery']['fallback_count'], 1)
        self.assertEqual(unresolved['status'], 'unresolved')
        self.assertEqual(unresolved['commits'], 0)

    def test_constant_drift_and_decreasing_target_increase_necessary_visits(self):
        from src.coverage_budget import VisitBudget
        W = sparse.diags([np.ones(5), np.ones(5)], [-1, 1], shape=(6, 6), format='csr')
        session = access.AnchorSession(W, 'budget-guard', visit_budget=3).service
        x = np.ones(6, np.int64)
        caps = np.array([0, 3, 6, 9], np.int64)
        index = Index(session.kernel.kernel, x, 0, session.inc, session.degree, caps=caps)
        drift = np.zeros(6, np.int64)
        budget = VisitBudget(3)
        expected = ([], [3], [2, 3], [1, 2, 3], [0, 1, 2, 3])
        for target, ids, admitted in zip((22, 21, 18, 15, 12), expected,
                                         (True, True, True, False, False)):
            probe = access.plan(index, drift, target)
            self.assertEqual(probe.costs[probe.best], len(ids))
            self.assertEqual(access.gather(index, drift, probe, set())[0].tolist(), ids)
            self.assertEqual(budget.reserve(probe.costs[probe.best]), admitted)
        self.assertEqual(budget.consumed, 3)
        self.assertTrue(VisitBudget(0).reserve(0))

    def test_ticket_rejects_replacement_changed_decision_and_reuse(self):
        from src.coverage_budget import Decision, PlanTickets
        guard = access.AnchorSession(self.W, 'budget-guard', visit_budget=10)
        guard.request(self.s, budget=0)
        anchor = guard.bridge.anchor
        decision = Decision(2, 'opinion', 'parent', 1, 10)
        drift = np.zeros(8, np.int64)
        probe = access.plan(anchor.index, drift, decision.target)
        tickets = PlanTickets()
        ticket = tickets.issue(decision, anchor, probe, drift, set())
        for bad, current, source in ((replace(ticket), decision, anchor),
                                     (ticket, replace(decision, target=11), anchor),
                                     (ticket, replace(decision, parent='stale'), anchor),
                                     (ticket, decision, replace(anchor))):
            with self.assertRaisesRegex(ValueError, 'plan binding mismatch'):
                tickets.take(bad, current, source)
        self.assertIs(tickets.take(ticket, decision, anchor), ticket)
        with self.assertRaisesRegex(ValueError, 'plan binding mismatch'):
            tickets.take(ticket, decision, anchor)
