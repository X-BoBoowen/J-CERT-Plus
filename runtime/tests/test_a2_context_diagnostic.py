"""Tiny synthetic checks only; execute only while the formal batch is idle."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from src.a2_context_diagnostic import compare_snapshots, match_request, reproduce_request
from src.a2_numeric import EndpointKernel
from src.a2_session import A2Session
from src.weighted_adapter import WeightedViews


def tiny():
    W = np.zeros((6, 6), float)
    for i, j, w in ((0, 1, -2.), (1, 2, 1.5), (2, 3, -.5), (3, 4, 3.), (4, 5, -1.)):
        W[i, j] = W[j, i] = w
    s = np.array([40000, -21000, 12000, -49000, 8000, 50000], np.int64)
    service = A2Session(sparse.csr_matrix(W), "H", hot_size=1, timers=False)
    measured = service.request(s, budget=2)
    return sparse.csr_matrix(W), s, measured, service.graph_hash


class A2ContextDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.W, cls.s, cls.measured, cls.graph_hash = tiny()
        cls.audited, cls.reproduction = reproduce_request(
            cls.W, cls.s, cls.measured, cls.graph_hash, hot_size=1)

    def test_same_root_last_contexts_paid_root_index_and_no_new_responses(self):
        with patch.object(EndpointKernel, "views", side_effect=AssertionError("new numerical views")), \
                patch.object(WeightedViews, "response", side_effect=AssertionError("new response")):
            report = compare_snapshots(self.W, self.s, self.measured, self.audited, self.graph_hash)
        self.assertEqual(report["status"], "completed", report)
        self.assertEqual(report["selected_epochs"], [1, 2])
        self.assertTrue(self.reproduction["outputs_match_recorded"])
        self.assertGreater(self.reproduction["response_blocks"], 0)
        self.assertEqual(report["additional_numerical_response_rhs"], 0)
        paid = report["paid_D_root_setup"]
        self.assertEqual(paid["root_caps_created_rows"], report["catalog_size"])
        self.assertTrue(paid["retained_across_selected_contexts"])
        root_digest = paid["root_center_sha256"]
        for context in report["contexts"]:
            self.assertEqual(len(context["arms"]), 3)
            arms = {row["method"]: row for row in context["arms"]}
            self.assertEqual(arms["S"]["legal_emits"], context["cold_unprocessed_candidates"])
            for row in context["arms"]:
                self.assertEqual(row["context_id"], context["context_id"])
                self.assertEqual(row["incumbent_lower_fine"], context["incumbent_lower_fine"])
                self.assertEqual(row["excluded_ids_sha256"], context["excluded_ids_sha256"])
                self.assertEqual(row["numerical_response_rhs"], 0)
                self.assertEqual(row["threshold_updates"], 0)
                self.assertEqual(row["endpoint_rows"], row["legal_emits"])
                self.assertLessEqual(row["strict_survivors"], arms["S"]["strict_survivors"])
            self.assertEqual(arms["D"]["raw_work"]["root_center_sha256"], root_digest)
            self.assertEqual(arms["D"]["raw_work"]["root_index_epoch"], 1)
        self.assertEqual(len(report["contexts"][1]["chosen_ids"]), 1)

    def test_changed_output_and_opinion_are_rejected(self):
        changed = copy.deepcopy(self.measured)
        changed["decisions"][0]["winner_id"] += 1
        with self.assertRaisesRegex(ValueError, "output mismatch"):
            match_request(changed, self.audited)
        changed = copy.deepcopy(self.measured)
        changed["access"]["epochs"][0]["cold_start_lower_fine"] += 1
        with self.assertRaisesRegex(ValueError, "access mismatch"):
            match_request(changed, self.audited)
        with self.assertRaisesRegex(ValueError, "opinion"):
            compare_snapshots(self.W, -self.s, self.measured, self.audited, self.graph_hash)

    def test_missing_last_snapshot_is_reported_without_substitution(self):
        audited = copy.deepcopy(self.audited)
        audited["snapshots"] = [audited["snapshots"][0]]
        report = compare_snapshots(self.W, self.s, self.measured, audited, self.graph_hash)
        self.assertEqual(report["selected_epochs"], [1, 2])
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["contexts"][0]["status"], "completed")
        self.assertEqual(report["contexts"][1]["status"], "unavailable")
        self.assertEqual(report["contexts"][1]["roles"], ["last_reached"])
        self.assertEqual(report["contexts"][1]["arms"], [])


if __name__ == "__main__":
    unittest.main()
