"""Exact check of the declared Gram view through rank-one and rank-two commits."""
import unittest

import numpy as np
import sympy as sp

from engine import Edit
from linview_delta import RegisteredDelta, residual


class RegisteredDeltaTests(unittest.TestCase):
    def test_unqueried_rows_are_current(self):
        A = np.array([[3,2,0,0],[2,4,1,0],[0,1,2,0],[0,0,0,1]])
        edits = [Edit(0,1,2,True),Edit(1,2,1,True),Edit(0,3,.5)]
        rhs = np.array([1.,-.25,.5,-1.])
        kernel = RegisteredDelta(A,edits,"cg",rhs)
        try:
            for step, winner in enumerate([2,0,None]):
                inverse = sp.Matrix(kernel.current.toarray()).applyfunc(sp.Rational).inv()
                self.assertEqual(kernel.ledger["registered_rows_updated"],step*len(edits))
                self.assertEqual(kernel.ledger["refinements"],0)
                rho = residual(kernel.current,kernel.current_abs,kernel.current_gamma,rhs,kernel.z)
                lo,hi = kernel.score_all(np.arange(len(edits)),rho)
                for e,edit in enumerate(edits):
                    d = 1+int(edit.flip)
                    V = sp.zeros(4,d)
                    V[edit.i,0],V[edit.j,0] = 1,-1
                    if edit.flip:
                        V[edit.i,1],V[edit.j,1] = 1,1
                    exact = V.T*inverse*V
                    defect = exact-sp.Matrix(kernel.grams[e,:d,:d]).applyfunc(sp.Rational)
                    margin = sp.eye(d)*sp.Rational(kernel.errors[e])**2-defect.T*defect
                    self.assertTrue(margin.is_positive_semidefinite)
                    if not kernel.active[e]:
                        h = V.T*inverse*sp.Matrix(rhs).applyfunc(sp.Rational)
                        signs = sp.diag(*([1,-1] if edit.flip else [1]))
                        gain = (h.T*(exact+signs/sp.Rational(edit.w)).inv()*h)[0]
                        self.assertLessEqual(sp.Rational(lo[e]),gain)
                        self.assertLessEqual(gain,sp.Rational(hi[e]))
                before = kernel.ledger["registered_scores"]
                kernel.choose(rhs)
                self.assertEqual(kernel.ledger["registered_scores"]-before,len(edits)-step)
                self.assertEqual(kernel.ledger["screens"],0)
                if winner is not None:
                    kernel.commit(winner)
        finally:
            kernel.close()


if __name__ == "__main__":
    unittest.main()
