"""Exact rational checks of certified decisions and transported Gram enclosures."""
import unittest

import numpy as np
import sympy as sp

from certified_engine import CertifiedViewManager, Unresolved
from engine import Edit


class CertifiedEngineTests(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[3,2,0,0],[2,4,1,0],[0,1,2,0],[0,0,0,1]])
        self.edits = [Edit(0,1,2,True),Edit(1,2,1,True),Edit(0,3,.5)]
        self.manager = CertifiedViewManager(self.A,self.edits)

    def operator(self, state):
        A = sp.Matrix(self.A)
        for eid in state:
            e = self.edits[eid]
            w = sp.Rational(e.w)
            d = -2*w if e.flip else -w
            A[e.i,e.j] += d
            A[e.j,e.i] += d
            if not e.flip:
                A[e.i,e.i] += w
                A[e.j,e.j] += w
        return A

    def test_each_branch_and_opinion(self):
        for e in range(3):
            self.manager.gram((),e)
        for state in [(),self.manager.commit((),0),self.manager.commit((),2)]:
            H = self.operator(state).inv()
            for e,G in self.manager.views[state].items():
                ed = self.edits[e]
                V = sp.zeros(4,1+int(ed.flip))
                V[ed.i,0],V[ed.j,0] = 1,-1
                if ed.flip:
                    V[ed.i,1],V[ed.j,1] = 1,1
                exact = V.T*H*V
                rad = sp.Rational(self.manager.errors[state,e])
                for a,b in zip(exact,G.flat):
                    self.assertLessEqual(abs(a-sp.Rational(b)),rad)
            for opinion in [[1,-.25,.5,-1],[.625,.75,-1,.5]]:
                s = sp.Matrix([sp.Rational(v) for v in opinion])
                gains = {e:(s.T*(H-self.operator((*state,e)).inv())*s)[0]
                         for e in range(3) if e not in state}
                eid,lo = self.manager.select(state,opinion,1e-8)
                for e,(left,right) in self.manager.last_intervals.items():
                    self.assertLessEqual(gains[e],right)
                    if left is not None:
                        self.assertLessEqual(left,gains[e])
                if eid is None:
                    self.assertLessEqual(max(gains.values()),0)
                else:
                    self.assertGreater(gains[eid],0)
                    self.assertLessEqual(sp.Rational(lo),gains[eid])
                    self.assertLessEqual(max(gains.values())-gains[eid],sp.Rational(1e-8))
        self.assertGreater(self.manager.ledger["transported"],0)

    def test_tie_and_refusal(self):
        m = CertifiedViewManager(np.eye(4),[Edit(0,1,3),Edit(2,3,3)])
        with self.assertRaises(Unresolved):
            m.select((),[1,-1,1,-1])
        self.assertEqual(m.select((),[1,-1,1,-1],1e-8)[0],0)
        self.assertEqual(m.select((),[1,-1,1,-1],pool=[]),(None,0.))

    def test_illegal_operator(self):
        with self.assertRaises(ValueError):
            CertifiedViewManager([[1,1],[1,1]],[Edit(0,1,1,True)])
        with self.assertRaises(ValueError):
            CertifiedViewManager(np.eye(2),[Edit(0,1,1,True)])


if __name__ == "__main__":
    unittest.main()
