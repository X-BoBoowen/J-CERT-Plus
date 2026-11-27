"""Targeted G output rejection tests using the existing independent checker.

The missing-block test deliberately injects a fault into PairBlocks._push.  It
does not report an observed production defect.  A2CatalogReplay independently
regenerates the legal catalog and rebuilds operators; it does not trust G's
source_complete flag, emitted candidates, partition, or bounds.  These tests
do not make PairBlocks a public authenticated certificate API.

Run only through the root V3 numerical-budget scheduler.  Preparing this file
does not constitute an executed test.
"""
import copy
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from src.a2_checker import A2CatalogReplay, worker_request
from src.a2_session import A2Session
from src.common import DEN
from src.g_access import PairBlocks


def graph(n, edges):
    rows, columns, values = [], [], []
    for i, j, weight in edges:
        rows.extend((i, j))
        columns.extend((j, i))
        values.extend((weight, weight))
    return sparse.csr_matrix((values, (rows, columns)), shape=(n, n), dtype=float)


class GIndependentRejections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This deterministic mixed fixture is also used by test_v3_g.py.  Keep
        # one authenticated actual G trace as the control for both mutations.
        cls.W = graph(6, ((0, 1, -1.5), (1, 3, 2), (2, 3, -0.5),
                          (3, 5, 1), (0, 4, 2.5), (2, 5, -2)))
        cls.s = np.array((DEN, -3 * DEN // 4, DEN // 2, -DEN,
                          3 * DEN // 16, 7 * DEN // 8), dtype=np.int64)
        service = A2Session(cls.W, "G")
        cls.graph_hash = service.graph_hash
        cls.control = service.request(cls.s, budget=2)
        if (cls.control["method"] != "G" or cls.control["commits"] != 2 or
                len(cls.control["decisions"]) != 2 or
                any(row["status"] != "accepted" for row in cls.control["decisions"])):
            raise AssertionError("G rejection-test control did not produce two accepted decisions")
        proof = worker_request(A2CatalogReplay(cls.W, block_size=2), cls.control,
                               cls.s, cls.graph_hash)
        if not proof["full_catalog_decisions_certified"] or proof["accepts"] != 2:
            raise AssertionError("unaltered G control must pass independent full-catalog replay")

    def test_g_request_rejects_stale_second_parent(self):
        """Mutate only the second decision's parent to the genuine old root."""
        request = copy.deepcopy(self.control)
        second = request["decisions"][1]
        self.assertNotEqual(second["parent_version_id"], request["root_version_id"])
        second["parent_version_id"] = request["root_version_id"]
        with self.assertRaisesRegex(ValueError, "parent/epoch binding"):
            worker_request(A2CatalogReplay(self.W, block_size=2), request,
                           self.s, self.graph_hash)

    def test_g_request_rejects_wrong_physical_edit_weight(self):
        """The checker derives weight from its own root graph and candidate ID."""
        request = copy.deepcopy(self.control)
        first = request["decisions"][0]
        self.assertIsInstance(first["d"], int)
        first["d"] += 1  # One incorrect 1/16 physical-weight unit; no other edit.
        with self.assertRaisesRegex(ValueError, "weight endpoint binding"):
            worker_request(A2CatalogReplay(self.W, block_size=2), request,
                           self.s, self.graph_hash)

    def test_fault_injected_missing_root_block_false_stop_is_rejected(self):
        """TEST-ONLY FAULT INJECTION: omit G's root block on a positive case."""
        W = graph(2, ())
        s = np.array((DEN, -DEN), dtype=np.int64)

        # Show the exact same unmodified input has one certified positive edit.
        control_service = A2Session(W, "G")
        control = control_service.request(s, budget=1)
        control_proof = worker_request(A2CatalogReplay(W, block_size=1), control,
                                       s, control_service.graph_hash)
        self.assertEqual(control["commits"], 1)
        self.assertTrue(control_proof["full_catalog_decisions_certified"])
        self.assertEqual(control_proof["decision_checks"][0]["independent_gain_interval"],
                         [[2, 9], [2, 9]])

        original_push = PairBlocks._push
        omitted = []

        def omit_root_block(blocks, a, b, c, d):
            if (a, b, c, d) == (0, blocks.catalog.n, 0, blocks.catalog.n):
                omitted.append((a, b, c, d))
                return None
            return original_push(blocks, a, b, c, d)

        # Patch only execution's partition construction.  Restore it before
        # independent replay so the failure's scope is explicit and isolated.
        with patch.object(PairBlocks, "_push", new=omit_root_block):
            faulty_service = A2Session(W, "G")
            faulty = faulty_service.request(s, budget=1)
        self.assertEqual(omitted, [(0, 2, 0, 2)])
        self.assertEqual(faulty["status"], "tolerance_stop")
        self.assertEqual(faulty["commits"], 0)
        self.assertEqual(faulty["decisions"][0]["status"], "tolerance_stop")
        self.assertTrue(faulty["access"]["epochs"][0]["source_complete"])
        self.assertGreater(faulty["access"]["epochs"][0]["g_global_insertion_upper_coarse"], 0)

        # The authentic checker scans the entire legal domain and finds the
        # omitted insertion's positive exact gain despite the claimed stop.
        with self.assertRaisesRegex(ValueError, "false tolerance stop"):
            worker_request(A2CatalogReplay(W, block_size=1), faulty,
                           s, faulty_service.graph_hash)


if __name__ == "__main__":
    unittest.main(verbosity=2)
