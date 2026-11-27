"""Rational checks of full-scale endpoint, transport, and refusal paths."""
import unittest
from fractions import Fraction as Q

import numpy as np
import sympy as sp

from engine import Edit
from policy_kernel import CertificateError, PolicyKernel


class PolicyTests(unittest.TestCase):
    def test_mixed_transport_and_screens(self):
        A = np.array([[3,2,0,0],[2,4,1,0],[0,1,2,0],[0,0,0,1]])
        edits = [Edit(0,1,2,True),Edit(1,2,1,True),Edit(0,3,.5)]
        rhs = np.array([1.,-.25,.5,-1.])
        kernel = PolicyKernel(A,edits)
        s = sp.Matrix([sp.Rational(x) for x in rhs])
        for requested in [0,2,1]:
            H = sp.Matrix(kernel.A.toarray()).applyfunc(sp.Rational).inv()
            z,rho = kernel.solve(rhs)
            error = H*s-sp.Matrix(z).applyfunc(sp.Rational)
            self.assertLessEqual((error.T*error)[0],sp.Rational(rho)**2)
            ids = np.flatnonzero(~kernel.active)
            cap = kernel.upper(ids,z,rho)
            for k,e in enumerate(ids):
                a = edits[e]
                V = sp.zeros(4,1+int(a.flip))
                V[a.i,0],V[a.j,0] = 1,-1
                if a.flip:
                    V[a.i,1],V[a.j,1] = 1,1
                G = V.T*H*V
                B = G + sp.diag(*([1,-1] if a.flip else [1]))/sp.Rational(a.w)
                h = V.T*H*s
                gain = (h.T*B.inv()*h)[0]
                self.assertLessEqual(gain,sp.Rational(cap[k]))
                lo,hi = kernel._interval(e,z,rho)
                self.assertLessEqual(sp.Rational(lo),gain)
                self.assertLessEqual(gain,sp.Rational(hi))
                center,radius = kernel.rows[e]
                entry_error = G-sp.Matrix(center).applyfunc(sp.Rational)
                # Exact PSD test for ||G-center||_2 <= radius.
                M = sp.eye(G.rows)*sp.Rational(radius)**2-entry_error.T*entry_error
                self.assertTrue(M.is_positive_semidefinite)
            kernel.commit(requested)

    def test_parent_scoped_winner_retention(self):
        A = np.eye(6)
        edits = [Edit(0,1,1),Edit(2,3,1),Edit(4,5,1)]
        rhs = np.array([1.,-1.,.75,-.75,.5,-.5])
        ordinary = PolicyKernel(A,edits)
        reuse = PolicyKernel(A,edits,winner_slots=2)
        # Two independent cursor decisions share a parent and leave two
        # exact fresh winner responses ready for different child branches.
        for pool in ([1],[0]):
            self.assertEqual(ordinary.select(rhs,pool),reuse.select(rhs,pool))
        self.assertEqual(reuse.winner_column_bytes(),2*6*8)
        ordinary.commit(0)
        reuse.commit(0)
        self.assertEqual(ordinary.ledger["W"],1)
        self.assertEqual(reuse.ledger["W"],0)
        self.assertEqual(reuse.ledger["winner_column_hits"],1)
        self.assertEqual(reuse.winner_column_bytes(),0)
        np.testing.assert_array_equal(ordinary.rows[1][0],reuse.rows[1][0])
        self.assertEqual(ordinary.rows[1][1],reuse.rows[1][1])
        decision = reuse.select(rhs)
        self.assertEqual(ordinary.select(rhs),decision)
        # After commitment this winner is a transported Gram hit, so
        # its parent-version columns must be obtained at the child.
        reuse.transition_columns(decision[0])
        self.assertEqual(reuse.ledger["W"],1)

    def test_tie_refusal_and_zero(self):
        edits = [Edit(0,1,3),Edit(2,3,3)]
        for cached in (True,False,"same"):
            kernel = PolicyKernel(np.eye(4),edits,cached=cached)
            with self.assertRaises(CertificateError):
                kernel.select([1,-1,1,-1])
            winner,lo = kernel.select([1,-1,1,-1],delta=1e-8)
            self.assertIn(winner,(0,1))
            self.assertLess(Q(lo),Q(12,7))
            self.assertGreater(Q(kernel.last_upper),Q(12,7))
            self.assertEqual(kernel.select([0,0,0,0]),(None,0.))

    def test_reanchor_preserves_winner_columns(self):
        # Two separate copies have identical local Grams 5/3. Scaling one
        # RHS scales only its gain, by the square of the scale factor.
        A = np.kron(np.eye(2),[[2.,0.,-1.],[0.,1.,0.],[-1.,0.,2.]])
        edits = [Edit(0,1,1),Edit(3,4,1)]
        block_rhs = np.array([1.,-1.,.5])
        for stale,scale in ((0,.999),(1,.999),(0,1.)):
            with self.subTest(stale=stale,scale=scale):
                kernel = PolicyKernel(A,edits,winner_slots=1)
                try:
                    kernel.rows[stale] = np.array([[5/3]]),1.5
                    kernel.stale.add(stale)
                    rhs = np.r_[block_rhs,scale*block_rhs]
                    if scale==1.:
                        with self.assertRaises(CertificateError):
                            kernel.select(rhs)
                        self.assertEqual(kernel.winner_column_bytes(),0)
                    else:
                        winner,_ = kernel.select(rhs)
                        self.assertEqual(winner,0)
                        kernel.commit(winner)
                        self.assertEqual(kernel.ledger["W"],0)
                        self.assertEqual(kernel.ledger["winner_column_hits"],1)
                    self.assertEqual(kernel.ledger["R"],1)
                finally:
                    kernel.close()


if __name__ == "__main__":
    unittest.main()

