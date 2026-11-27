import unittest
from fractions import Fraction
import numpy as np
from scipy.sparse import csr_matrix
from src.checker import CatalogReplay, reference_gains, opinion_hash


class CheckerTests(unittest.TestCase):
    def setUp(self):
        self.W = csr_matrix([[0.,-1.5,0.,0.],[-1.5,0.,-2.,0.],[0.,-2.,0.,.5],[0.,0.,.5,0.]])
        self.s = np.array([65536,-20000,-32000,5000],dtype=np.int64)

    def test_entire_weighted_trajectory_resets_switches_and_tamper(self):
        rep = CatalogReplay(self.W)
        for s in (self.s, self.s, -self.s, self.s[::-1].copy()):
            gains = reference_gains(self.W,s)
            best = max(range(len(gains)),key=gains.__getitem__)
            checked = rep.verify(s,[dict(status="accept",winner_id=best)],budget=1)
            self.assertGreater(checked["checked_candidates"],1)
            self.assertEqual(checked["accepts"],1)
            with self.assertRaises(ValueError):
                rep.verify(s,[dict(status="accept",winner_id=(best+1)%len(gains))],budget=1)
        with self.assertRaisesRegex(ValueError,"opinion"):
            rep.verify(self.s,[],budget=0,expected_opinion=opinion_hash(-self.s))

    def test_stop_tie_unresolved_budget(self):
        W = csr_matrix([[0.,-1.,0.],[-1.,0.,-1.],[0.,-1.,0.]])
        rep = CatalogReplay(W)
        zeros = np.zeros(3,np.int64)
        self.assertEqual(rep.verify(zeros,[dict(status="tolerance_stop",winner_id=None)],1)["accepts"],0)
        self.assertEqual(rep.verify(zeros,[],0)["termination"],"budget_stop")
        self.assertEqual(rep.verify(zeros,[dict(status="unresolved",winner_id=None)],1)["termination"],"unresolved")
        s = np.array([65536,-65536,65536],dtype=np.int64)
        g = reference_gains(W,s)
        self.assertEqual(g[0],g[-1])
        self.assertGreater(g[0],0)
        rep.verify(s,[dict(status="accept",winner_id=0)],1)
        with self.assertRaises(ValueError):
            rep.verify(s,[dict(status="accept",winner_id=2)],1)
        rep.verify(s,[dict(status="accept",winner_id=2)],1,delta=Fraction(1,100))
        with self.assertRaises(ValueError):
            rep.verify(s,[dict(status="tolerance_stop",winner_id=None)],1)
