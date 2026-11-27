"""The A2 verifier is tested against full exact catalogs, never H emissions."""
import ast
import copy
from fractions import Fraction
import hashlib
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse

from src.a2_checker import A2CatalogReplay, reference_gains, worker_request
from src.checker import opinion_hash


def graph():
    return sparse.csr_matrix([
        [0., -1.5, 0., 0., 0.],
        [-1.5, 0., -2., 0., 0.],
        [0., -2., 0., .5, 0.],
        [0., 0., .5, 0., 0.],
        [0., 0., 0., 0., 0.],
    ])


def exact_decisions(W, s, budget):
    chosen, decisions = [], []
    for _ in range(budget):
        gains = reference_gains(W, s, chosen)
        winner = min(gains, key=lambda e: (-gains[e], e)) if gains else None
        if winner is None or gains[winner] <= 0:
            decisions.append(dict(status="tolerance_stop", winner_id=None))
            break
        decisions.append(dict(status="accepted", winner_id=winner))
        chosen.append(winner)
    return decisions


class A2CheckerTests(unittest.TestCase):
    def setUp(self):
        self.W = graph()
        self.s = np.array([65536, -20000, -32000, 5000, 17000], np.int64)

    def test_independent_complete_sparse_catalog_and_bounded_blocks(self):
        rep = A2CatalogReplay(self.W, block_size=3)
        observed = []
        for ids, pool in rep.iter_blocks():
            self.assertLessEqual(len(ids), 3)
            self.assertEqual(ids.dtype, np.dtype("int64"))
            observed.extend((int(e), int(i), int(j), int(d), bool(f))
                            for e, i, j, d, f in zip(ids, *pool))
        expected = []
        for i in range(5):
            for j in range(i + 1, 5):
                w = self.W[i, j]
                if w <= 0:
                    expected.append((i * 5 + j, i, j, 1 if w == 0 else int(-16 * w), w < 0))
        self.assertEqual(observed, expected)
        self.assertEqual(rep.m, 9)
        self.assertIn((4, 0, 4, 1, False), observed)  # isolated; outside two-hop domain
        self.assertEqual(rep.candidate(1), (0, 1, 24, True))
        self.assertEqual(rep.candidate(7), (1, 2, 32, True))
        remaining = [int(e) for ids, _ in rep.iter_blocks(excluded={1, 4}) for e in ids]
        self.assertEqual(remaining, [r[0] for r in expected if r[0] not in {1, 4}])

    def test_exact_continuous_trajectory_and_request_reset(self):
        rep = A2CatalogReplay(self.W, block_size=2)
        for s in (self.s, self.s, -self.s, self.s[::-1].copy()):
            decisions = exact_decisions(self.W, s, 3)
            checked = rep.verify(s, decisions, 3, expected_opinion=opinion_hash(s))
            accepts = sum(r["status"] == "accepted" for r in decisions)
            self.assertEqual(checked["accepts"], accepts)
            self.assertEqual(checked["plan"], [r["winner_id"] for r in decisions if r["status"] == "accepted"])
            self.assertEqual(checked["checked_candidate_decision_rows"],
                             sum(rep.m - t for t in range(len(decisions))))
            self.assertEqual(checked["independent_rhs"], 0)
            self.assertTrue(checked["full_catalog_decisions_certified"])
        with self.assertRaisesRegex(ValueError, "opinion"):
            rep.verify(self.s, [], 0, expected_opinion=opinion_hash(-self.s))

    def test_canonical_exact_tie_is_endpoint_id(self):
        W = sparse.csr_matrix([[0., -1., 0.], [-1., 0., -1.], [0., -1., 0.]])
        s = np.array([65536, -65536, 65536], np.int64)
        gains = reference_gains(W, s)
        self.assertEqual(gains[1], gains[5])
        self.assertGreater(gains[1], 0)
        rep = A2CatalogReplay(W)
        self.assertEqual(rep.verify(s, [dict(status="accepted", winner_id=1)], 1)["plan"], [1])
        with self.assertRaisesRegex(ValueError, "winner/tie"):
            rep.verify(s, [dict(status="accepted", winner_id=5)], 1)

    def test_bad_winner_positive_edge_duplicate_and_endpoint_binding(self):
        rep = A2CatalogReplay(self.W)
        good = exact_decisions(self.W, self.s, 1)[0]
        bad = next(e for e in reference_gains(self.W, self.s) if e != good["winner_id"])
        with self.assertRaisesRegex(ValueError, "winner/tie"):
            rep.verify(self.s, [good | {"winner_id": bad}], 1)
        with self.assertRaisesRegex(ValueError, "positive root"):
            rep.verify(self.s, [good | {"winner_id": 13}], 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            rep.verify(self.s, [good, good], 2)
        with self.assertRaisesRegex(ValueError, "endpoint"):
            rep.verify(self.s, [good | {"i": 42}], 1)
        for invalid in (-1, 0, 25, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                rep.verify(self.s, [good | {"winner_id": invalid}], 1)

    def test_stop_unresolved_and_terminal_semantics(self):
        rep = A2CatalogReplay(self.W)
        stop = dict(status="tolerance_stop", winner_id=None)
        self.assertEqual(rep.verify(np.zeros(5, np.int64), [stop], 3)["termination"], "tolerance_stop")
        self.assertEqual(rep.verify(self.s, [], 0)["termination"], "budget_stop")
        unresolved = dict(status="unresolved", winner_id=None)
        result = rep.verify(self.s, [unresolved], 3)
        self.assertFalse(result["full_catalog_decisions_certified"])
        self.assertFalse(result["termination_certified"])
        self.assertEqual(result["checked_candidates"], 0)
        with self.assertRaisesRegex(ValueError, "false tolerance"):
            rep.verify(self.s, [stop], 3)
        with self.assertRaisesRegex(ValueError, "premature budget"):
            rep.verify(self.s, [dict(status="budget_stop", winner_id=None)], 3)
        with self.assertRaisesRegex(ValueError, "terminal"):
            rep.verify(self.s, [unresolved, stop], 3)
        with self.assertRaisesRegex(ValueError, "unresolved commit"):
            rep.verify(self.s, [unresolved | {"winner_id": 1}], 3)
        with self.assertRaisesRegex(ValueError, "missing terminal"):
            rep.verify(self.s, [], 3)

    def test_exact_gain_binding_and_numerical_complete_scan(self):
        exact = A2CatalogReplay(self.W)
        good = exact_decisions(self.W, self.s, 1)[0]
        gain = reference_gains(self.W, self.s)[good["winner_id"]]
        denominator = 1 << 96
        scaled = gain * denominator
        published = good | {"low": scaled.numerator // scaled.denominator,
                            "high": -(-scaled.numerator // scaled.denominator)}
        exact.verify(self.s, [published], 1, gain_denominator=denominator)
        with self.assertRaisesRegex(ValueError, "published gain"):
            exact.verify(self.s, [published | {"low": 0, "high": 0}], 1,
                         gain_denominator=denominator)
        numeric = A2CatalogReplay(self.W, block_size=2, exact_limit=0)
        checked = numeric.verify(self.s, [published], 1, gain_denominator=denominator)
        self.assertEqual(checked["checked_candidates"], exact.m)
        self.assertGreaterEqual(checked["independent_rhs"], 2)
        self.assertEqual(sum(r["checked_candidates"] for r in checked["decision_checks"]), exact.m)
        self.assertLessEqual(checked["max_block_rows"], 2)
        trajectory = exact_decisions(self.W, self.s, 3)
        checked = numeric.verify(self.s, trajectory, 3)
        self.assertEqual(checked["checked_candidates"], 9 + 8 + 7)
        self.assertEqual(checked["independent_rhs_by_role"]["parent"], 1)
        self.assertEqual(checked["independent_rhs_by_role"]["winner_child"], 3)
        with self.assertRaisesRegex(ValueError, "published gain"):
            numeric.verify(self.s, [published | {"low": 0, "high": 0}], 1,
                           gain_denominator=denominator)

    def test_numerical_zero_stop_uses_complete_caps_without_candidate_solves(self):
        W = sparse.csr_matrix((14, 14), dtype=float)
        rep = A2CatalogReplay(W, block_size=5)
        checked = rep.verify(np.zeros(14, np.int64),
                             [dict(status="tolerance_stop", winner_id=None)], 1)
        self.assertEqual(checked["checked_candidates"], 91)
        self.assertEqual(checked["independent_rhs"], 1)
        self.assertEqual(checked["endpoint_pruned_rows"], 91)

    def test_complete_positive_graph_and_single_node(self):
        for W in (sparse.csr_matrix(np.ones((4, 4)) - np.eye(4)),
                  sparse.csr_matrix((1, 1), dtype=float)):
            rep = A2CatalogReplay(W, block_size=2)
            self.assertEqual(rep.m, 0)
            self.assertEqual(list(rep.iter_blocks()), [])
            result = rep.verify(np.zeros(W.shape[0], np.int64),
                                [dict(status="tolerance_stop", winner_id=None)], 1)
            self.assertEqual(result["checked_candidates"], 0)

    def test_worker_root_versions_totals_and_mutations(self):
        rep = A2CatalogReplay(self.W)
        denominator = 1 << 96
        graph_hash = "test-graph-digest"
        root = rep.root_version_id(graph_hash)
        current = root
        decisions = exact_decisions(self.W, self.s, 2)
        chosen = []
        low = high = 0
        for epoch, row in enumerate(decisions, 1):
            parent = current
            if row["status"] == "accepted":
                e = row["winner_id"]
                gain = reference_gains(self.W, self.s, chosen)[e] * denominator
                row["low"] = gain.numerator // gain.denominator
                row["high"] = -(-gain.numerator // gain.denominator)
                low += row["low"]
                high += row["high"]
                current = hashlib.sha256(f"{parent}/{e}".encode()).hexdigest()
                chosen.append(e)
            row.update(epoch=epoch, root_version_id=root, parent_version_id=parent,
                       current_version_id=current)
        request = dict(root_version_id=root, current_version_id=current,
                       method="H", budget=2, delta=0, opinion_hash=opinion_hash(self.s),
                       decisions=decisions, commits=len(chosen), gain_low=low,
                       gain_high=high, gain_denominator=denominator, status="budget_stop")
        self.assertEqual(worker_request(rep, request, self.s, graph_hash)["accepts"], 2)
        for field, value in (("root_version_id", "wrong"), ("current_version_id", root),
                             ("gain_low", low + 1), ("commits", 1),
                             ("opinion_hash", opinion_hash(-self.s))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                worker_request(rep, request | {field: value}, self.s, graph_hash)
        modified = copy.deepcopy(request)
        modified["decisions"][1]["parent_version_id"] = root
        with self.assertRaisesRegex(ValueError, "parent"):
            worker_request(rep, modified, self.s, graph_hash)

    def test_checker_has_no_execution_catalog_or_service_import(self):
        source = Path(__file__).resolve().parents[1] / "src/a2_checker.py"
        tree = ast.parse(source.read_text())
        imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any("a2_catalog" in n or "a2_session" in n or "a2_access" in n
                             or "a2_numeric" in n for n in imports))

    def test_actual_small_service_schema_replays_in_all_three_arms(self):
        from src.a2_session import A2Session
        rep = A2CatalogReplay(self.W, block_size=2)
        for method in ("D", "S", "H"):
            service = A2Session(self.W, method, timers=False)
            request = service.request(self.s, budget=3)
            verified = worker_request(rep, request, self.s, service.graph_hash)
            self.assertEqual(verified["accepts"], 3, method)
            self.assertTrue(verified["full_catalog_decisions_certified"], method)
            self.assertEqual(verified["checked_candidate_decision_rows"], 24, method)


if __name__ == "__main__":
    unittest.main()
