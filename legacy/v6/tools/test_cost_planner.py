"""Exercise zero-, one-, and multiple-request edges through singleton decisions."""
import os
from pathlib import Path
import unittest

import numpy as np

from cost_planner import CostPlanner, gate_costs
from cholmod_backend import CholmodFactor
from engine import Edit
from policy_kernel import PolicyKernel


class PlannerTests(unittest.TestCase):
    def exercise(self, backend, retained):
        n = 80
        A = np.eye(n)
        edits = [Edit(0,1,1), Edit(2,3,1,True), Edit(4,5,1,True), Edit(6,7,1,True)]
        for edit in edits[1:]:
            i,j = edit.i,edit.j
            A[np.ix_([i,j],[i,j])] += 1
        s = np.tile([1.,-1.], n//2)
        kernel = CostPlanner(A, edits, backend=backend)
        reference = PolicyKernel(A, edits, cached=False, backend=backend)
        try:
            # A singleton query certifies its member and genuinely demands a
            # Gram. Query ordering establishes which one-slot response survives.
            for f in ([1,2,3,0] if retained else [0,1,2,3]):
                self.assertEqual(kernel.select(s,[f])[0],f)
            kernel.commit(0)
            reference.commit(0)
            self.assertEqual(kernel.ledger["W_rhs"],0)
            self.assertLess(PolicyKernel.memory_bytes(kernel),kernel.memory_bytes())
            for f in [1,2,3]:
                self.assertEqual(kernel.select(s,[f])[0],reference.select(s,[f])[0])
                self.assertLessEqual(kernel.select(s,[f])[1],4.0)
                if f==1:
                    self.assertEqual(kernel.ledger["W"],0)
                    self.assertEqual(kernel.ledger["transported"],int(retained))
            self.assertEqual(kernel.ledger["W_rhs"],0 if retained else 1)
            self.assertEqual(kernel.ledger["F_rhs"],0 if retained else 2)
            self.assertEqual(kernel.ledger["gate_requests"],3)
            self.assertEqual(kernel.ledger["transported"],3 if retained else 2)
            self.assertEqual(kernel.ledger["winner_column_hits"],int(retained))
            self.assertIsNone(kernel.pending)
            # The previous parent was exhausted. Subsequent zero-demand edges
            # must not consume a historical RHS.
            kernel.commit(1)
            before = kernel.ledger["W_rhs"]
            kernel.commit(2)
            self.assertEqual(kernel.ledger["W_rhs"],before)
        finally:
            kernel.close()
            reference.close()

    def test_cg(self):
        for retained in [False,True]:
            with self.subTest(retained=retained):
                self.exercise("cg",retained)

    @unittest.skipUnless(os.environ.get("JCERT_CHOLMOD_LIB") or
                         Path(__file__).with_name("jcert_cholmod.dll").exists(),
                         "CHOLMOD wrapper unavailable")
    def test_cholmod(self):
        for retained in [False,True]:
            with self.subTest(retained=retained):
                self.exercise("cholmod",retained)

    def test_cost_gate(self):
        self.assertFalse(gate_costs(2,1,1000,1000,False,1)[0])
        self.assertTrue(gate_costs(2,1,1000,1000,False,2)[0])
        self.assertFalse(gate_costs(2,2,1000,1000,False,3)[0])
        self.assertFalse(gate_costs(1,1,128,1000,True,1)[0])
        self.assertTrue(gate_costs(1,1,129,1000,True,1)[0])

    def test_equal_rank_competition(self):
        for flip in [False,True]:
            n = 512
            A = np.eye(n)
            edits = [Edit(2*i,2*i+1,0.5,flip) for i in range(8)]
            if flip:
                for edit in edits:
                    ids = [edit.i,edit.j]
                    A[np.ix_(ids,ids)] += edit.w
            s = np.tile([1.,-1.], n//2)
            for backend in (["cg","cholmod"] if os.environ.get("JCERT_CHOLMOD_LIB") else ["cg"]):
                with self.subTest(flip=flip,backend=backend):
                    k = CostPlanner(A,edits,backend=backend)
                    try:
                        for e in range(8):
                            self.assertEqual(k.select(s,[e])[0],e)
                        k.commit(0)
                        for e in range(1,8):
                            self.assertEqual(k.select(s,[e])[0],e)
                            if e == 1:
                                self.assertEqual(k.ledger["W_rhs"],0)
                                self.assertEqual(k.ledger["planner_parent_clones"],0)
                        self.assertEqual(k.ledger["W_rhs"],1+int(flip))
                        self.assertGreater(k.ledger["forecast_rows"],0)
                        self.assertEqual(k.ledger["forecast_still_unrealized"],0)
                    finally:
                        k.close()

    @unittest.skipUnless(os.environ.get("JCERT_CHOLMOD_LIB"), "CHOLMOD wrapper unavailable")
    def test_factor_reverse(self):
        s = np.arange(1.,7.)
        for flip in [False,True]:
            edit = Edit(1,4,0.75,flip)
            A = np.eye(6)
            A[np.ix_([0,1],[0,1])] += [[1,-1],[-1,1]]
            if flip:
                A[np.ix_([1,4],[1,4])] += edit.w
            with CholmodFactor(A) as factor:
                before = factor.solve(s)
                factor.change(edit)
                after = factor.solve(s)
                with factor.clone() as rollback:
                    rollback.undo(edit)
                    np.testing.assert_allclose(rollback.solve(s),before,rtol=1e-14)
                np.testing.assert_array_equal(factor.solve(s),after)


if __name__ == "__main__":
    unittest.main()
