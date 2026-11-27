"""Rational oracles for the optional scalar score and one-row transport."""
from fractions import Fraction as Q
import os
import random
import unittest

import numpy as np

from certified_engine import inverse, norm_cap, qarray
from engine import Edit
from fast_policy_kernel import FastPolicyKernel, score_interval


class FastCertificates(unittest.TestCase):
    def test_exact_score_with_uncertain_gram_and_state(self):
        rng = random.Random(417)
        for w in (0.125,0.75,1.,8.,64.):
            for flip in (False,True):
                for _ in range(40):
                    # A diagonal SPD base, with the second signed direction
                    # formed by a legal flip when requested.
                    diag = np.diag([Q(1+rng.randrange(5)),Q(1+rng.randrange(5))])
                    V = qarray([[1,1],[-1,1]]) if flip else qarray([[1],[-1]])
                    if flip:
                        diag += Q(w)*qarray([[1,1],[1,1]])
                    H = inverse(diag)
                    G = V.T@H@V
                    center = np.array(G,dtype=float)
                    center[0,-1] += 2.**-21
                    eg = float(norm_cap(qarray(center)-G))*(1+1e-12)
                    x,y = (rng.randrange(-200,200)/32 for _ in range(2))
                    delta = Q(1,1<<29)
                    st = np.array([Q(x)+delta,Q(y)-delta],dtype=object)
                    h = V.T@st
                    B = G.copy()
                    B[0,0] += 1/Q(w)
                    if flip:
                        B[1,1] -= 1/Q(w)
                    gain = h@inverse(B)@h
                    lo,hi = score_interval(x,y,w,flip,center,eg,float(2*delta))
                    self.assertLessEqual(Q(lo),gain)
                    self.assertLessEqual(gain,Q(hi))

    def transport(self,backend):
        A = np.array([[5,4,0],[4,6,1],[0,1,2]])
        edits = [Edit(0,1,4,True),Edit(1,2,1,True),Edit(0,2,.25)]
        for winner,other in ((0,1),(1,2),(2,0)):
            kernel = FastPolicyKernel(A,edits,backend=backend,winner_slots=1)
            try:
                z,rho = kernel.solve([1.,-.5,.25])
                kernel._interval(other,z,rho)
                kernel.commit(winner)
                center,radius = kernel.rows[other]
                # Exact 3-by-3 inverse from cofactors via the rational oracle.
                import sympy as sp
                H = sp.Matrix(kernel.A.toarray()).applyfunc(sp.Rational).inv()
                e = edits[other]
                V = sp.zeros(3,1+int(e.flip))
                V[e.i,0],V[e.j,0] = 1,-1
                if e.flip:
                    V[e.i,1],V[e.j,1] = 1,1
                E = V.T*H*V-sp.Matrix(center).applyfunc(sp.Rational)
                self.assertTrue((sp.eye(E.rows)*sp.Rational(radius)**2-E.T*E).is_positive_semidefinite)
            finally:
                kernel.close()

    @unittest.skipUnless(os.environ.get("JCERT_CHOLMOD_LIB"),"native CHOLMOD unavailable")
    def test_child_factor_recovery_and_warm_state(self):
        import sympy as sp
        from fast_policy_kernel import FastCostPlanner
        from cholmod_backend import CholmodFactor
        for flip in (True,False):
            n = 512
            A = np.eye(n)
            if flip:
                A[:2,:2] += 2
            edits = [Edit(0,1,2,flip)]+[Edit(0,k,.5) for k in range(2,15)]
            rhs = np.arange(n,dtype=float)/16
            for retain in (False,True):
                factor = CholmodFactor(A)
                k = FastCostPlanner(A,edits,"cholmod",root_factor=factor,own_root_factor=True)
                try:
                    z,rho = k.solve(rhs)
                    for e in range(len(edits)):
                        k._interval(e,z,rho)
                    if retain:
                        k._retain_winner(0,k._fresh(0,"W"))
                    k.commit(0)
                    current,r = k.solve_committed_state(rhs,z)
                    H = sp.Matrix(k.A.toarray()[:15,:15]).applyfunc(sp.Rational).inv()
                    exact = H*sp.Matrix(rhs[:15]).applyfunc(sp.Rational)
                    diff = exact-sp.Matrix(current[:15]).applyfunc(sp.Rational)
                    self.assertLessEqual(diff.dot(diff),sp.Rational(r)**2)
                    self.assertEqual(k.ledger["state_warm_hits"],int(retain))
                    for e in range(1,len(edits)):
                        k._interval(e,current,r)
                        center,rad = k.rows[e]
                        j = edits[e].j
                        gram = H[0,0]+H[j,j]-2*H[0,j]
                        self.assertLessEqual(abs(gram-sp.Rational(center[0,0])),sp.Rational(rad))
                    self.assertEqual(k.ledger["parent_recoveries"],int(not retain))
                    self.assertEqual(k.ledger["factor_clones"],0)
                    self.assertGreater(k.ledger["transported"],0)
                finally:
                    k.close()

    def test_batch_mixed_gram_oracle(self):
        import sympy as sp
        from fast_policy_kernel import FastCostPlanner
        A = np.eye(64)
        edits = [Edit(0,1,2,True),Edit(0,2,.5),Edit(0,3,.25,True),
                 Edit(1,4,1.),Edit(2,4,1.,True)]
        for edge in edits:
            if edge.flip:
                ix = [edge.i,edge.j]
                A[np.ix_(ix,ix)] += edge.w
        for winner in (0,1):
            k = FastCostPlanner(A,edits,batch_transport=True)
            try:
                q,rho = k.solve(np.arange(64.)/64)
                for e in range(len(edits)):
                    k._interval(e,q,rho)
                k._retain_winner(winner,k._fresh(winner,"W"))
                k.commit(winner)
                ids = np.array([e for e in range(len(edits)) if e!=winner])
                order = np.arange(len(ids))
                cap = np.ones(len(ids))
                H = sp.Matrix(k.A[:5,:5].toarray()).applyfunc(sp.Rational).inv()
                for idx,e in enumerate(ids[:-1]):
                    k._lookahead = (ids,order,cap,idx,0.)
                    k._interval(e,q,rho)
                    V = sp.zeros(5,1+int(k.flip[e]))
                    V[k.i[e],0],V[k.j[e],0] = 1,-1
                    if k.flip[e]:
                        V[k.i[e],1] = V[k.j[e],1] = 1
                    G,rad = k.rows[e]
                    E = V.T*H*V-sp.Matrix(G).applyfunc(sp.Rational)
                    self.assertTrue((sp.eye(E.rows)*sp.Rational(rad)**2-E.T*E).is_positive_semidefinite)
                k._discard_batch()
                self.assertEqual(k.ledger["batch_calls"],1)
                self.assertEqual(k.ledger["batch_computed"],4)
                self.assertEqual(k.ledger["batch_consumed"],3)
                self.assertEqual(k.ledger["batch_unused"],1)
                self.assertNotIn(ids[-1],k.rows)
            finally:
                k.close()

    def test_cg_transport(self):
        self.transport('cg')

    def test_global_delta_late_projection_and_retained_state(self):
        """Rows first requested at different versions must see every pending delta."""
        import sympy as sp
        from linview_delta import GlobalDelta, residual

        source = np.array([[2,1,0,0],[1,3,1,0],[0,1,2,0],[0,0,0,1]])
        edits = [Edit(0,1,1,True), Edit(1,2,1,True),
                 Edit(0,3,.5), Edit(2,3,.25)]
        opinion = np.array([1.,-.5,.25,-.75])
        rhs = sp.Matrix([sp.Rational(x) for x in opinion])
        exact = sp.Matrix(source)
        kernel = GlobalDelta(source, edits, 'cg', opinion)

        def change(edit):
            basis = sp.zeros(4, 1 + int(edit.flip))
            basis[edit.i,0], basis[edit.j,0] = 1,-1
            if edit.flip:
                basis[edit.i,1] = basis[edit.j,1] = 1
            signature = sp.diag(1,-1) if edit.flip else sp.eye(1)
            return sp.Rational(edit.w)*basis*signature*basis.T

        try:
            for commit, requests in ((0,[2]),(2,[1,3]),(1,[1,3]),(None,[3])):
                cap = residual(kernel.current,kernel.current_abs,kernel.current_gamma,
                               opinion,kernel.z)
                equilibrium = exact.inv()*rhs
                for approx,truth in zip(kernel.z,equilibrium):
                    self.assertLessEqual(abs(sp.Rational(approx)-truth),sp.Rational(cap))
                for e in requests:
                    edited = exact + change(edits[e])
                    gain = (rhs.T*(equilibrium-edited.inv()*rhs))[0]
                    lo,hi = kernel._interval(e,kernel.z,cap)
                    self.assertLessEqual(sp.Rational(lo),gain)
                    self.assertLessEqual(gain,sp.Rational(hi))
                if commit is not None:
                    kernel.commit(commit)
                    exact += change(edits[commit])
                    before = kernel.z.copy()
                    with self.assertRaises(ValueError):
                        kernel.commit(commit)
                    np.testing.assert_array_equal(kernel.z,before)
        finally:
            kernel.close()

    @unittest.skipUnless(os.environ.get("JCERT_CHOLMOD_LIB"),"native CHOLMOD unavailable")
    def test_cholmod_transport(self):
        self.transport('cholmod')
