"""Compare both certificates against rational gains and Gram corrections."""
import os
import unittest
from fractions import Fraction

import numpy as np
import sympy as sp

from engine import Edit
from policy_kernel import PolicyKernel


class DirectionalKernelTests(unittest.TestCase):
    def check_backend(self, backend):
        A = np.array([[9,8,0,0],[8,10,1,0],[0,1,2,0],[0,0,0,1]])
        edits = [Edit(0,1,8,True),Edit(1,2,1,True),Edit(0,3,.5)]
        rhs = np.array([.75,-.125,.5,-1.])
        kernels = [PolicyKernel(A,edits,backend,directional=flag) for flag in (False,True)]
        old,new = kernels
        try:
            z,rho = new.solve(rhs)
            for e,edit in enumerate(edits):
                # Deliberate, certified nonsymmetry and common input balls.
                _,G,err = new._fresh(e,"C")
                perturb = Fraction(1,8192)
                G[0,-1] += perturb
                for kernel in kernels:
                    kernel._store(e,G,err+perturb)
                lo,hi = old._interval(e,z,rho)
                dl,dh = new._interval(e,z,rho)
                self.assertTrue(lo <= dl <= dh <= hi)
                V = sp.zeros(4,1+int(edit.flip))
                V[edit.i,0],V[edit.j,0] = 1,-1
                if edit.flip:
                    V[edit.i,1],V[edit.j,1] = 1,1
                H = sp.Matrix(A).inv()
                h = V.T*H*sp.Matrix(rhs).applyfunc(sp.Rational)
                B = V.T*H*V + sp.diag(*([1,-1] if edit.flip else [1]))/sp.Rational(edit.w)
                gain = (h.T*B.inv()*h)[0]
                self.assertTrue(sp.Rational(dl) <= gain <= sp.Rational(dh))
            for kernel in kernels:
                kernel.commit(0)
            H = sp.Matrix(new.A.toarray()).applyfunc(sp.Rational).inv()
            for e in new.rows:
                center,radius = new.rows[e]
                self.assertLessEqual(radius,old.rows[e][1])
                edit = edits[e]
                V = sp.zeros(4,1+int(edit.flip))
                V[edit.i,0],V[edit.j,0] = 1,-1
                if edit.flip:
                    V[edit.i,1],V[edit.j,1] = 1,1
                E = V.T*H*V-sp.Matrix(center).applyfunc(sp.Rational)
                self.assertTrue((sp.eye(E.rows)*sp.Rational(radius)**2-E.T*E).is_positive_semidefinite)
            self.assertEqual(new.ledger["directional_scores"],3)
            self.assertEqual(new.ledger["directional_blocks"],2)
        finally:
            for kernel in kernels:
                kernel.close()

    def test_cg(self):
        self.check_backend("cg")

    @unittest.skipUnless(os.environ.get("JCERT_CHOLMOD_LIB"),"native CHOLMOD unavailable")
    def test_cholmod(self):
        self.check_backend("cholmod")
