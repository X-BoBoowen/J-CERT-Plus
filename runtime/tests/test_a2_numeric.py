import unittest
from collections import defaultdict
from fractions import Fraction
import numpy as np
from scipy import sparse

from src.common import Ledger, UNIT, DEN
from src.a2_catalog import AllCatalog
from src.a2_numeric import EndpointKernel, CachedStrict
from src.weighted_adapter import Kernel as ExplicitKernel
import long_promotion as lp


def fixture():
    W=np.zeros((6,6),float)
    for i,j,w in [(0,1,-2.),(1,2,1.5),(2,3,-.5),(3,4,3.),(4,5,-1.)]:
        W[i,j]=W[j,i]=w
    return sparse.csr_matrix(W)


class A2NumericTests(unittest.TestCase):
    def setUp(self):
        self.W=fixture();self.catalog=AllCatalog(self.W)
        self.kernel=EndpointKernel(self.catalog)
        self.explicit=ExplicitKernel(self.W,self.catalog.materialize())
        self.A=self.kernel.operator(self.W)
        self.fac=lp.UpdatedFactor(self.A)
        self.s=np.array([40000,-21000,12000,-49000,8000,50000],np.int64)
        self.rows=[]
        self.state=lp.certified_state(self.A,self.fac,self.s,UNIT,certified_rows=self.rows)

    def test_virtual_views_match_existing_weighted_scores(self):
        left=self.kernel.views(Ledger(False),defaultdict(int))
        right=self.explicit.views(Ledger(False),defaultdict(int))
        for e in [1,2,2*6+3,4*6+5]:
            row=int(self.catalog.dense_positions(np.array([e],np.int64))[0])
            self.assertEqual(left.score(e,0,self.A,self.fac,*self.state[1:3]),
                             right.score(row,0,self.A,self.fac,*self.state[1:3]))
            self.assertEqual((self.kernel.child(self.A,e)!=self.explicit.child(self.A,row)).nnz,0)

    def test_virtual_columns_refuse_complete_materialization(self):
        for column in (*self.kernel.pool,self.kernel.d):
            with self.assertRaises((TypeError,ValueError)):
                np.asarray(column)
            with self.assertRaises((TypeError,ValueError)):
                column[:]

    def test_strict_shared_envelope_is_safe_against_exact_small_gains(self):
        import sympy as sy
        strict=CachedStrict(self.kernel,self.A,self.s,self.state[1],self.rows[0])
        batches=list(self.catalog.iter_blocks(block_size=4))
        s=sy.Matrix([sy.Rational(int(v),DEN) for v in self.s])
        base=(s.T*sy.Matrix(self.A.toarray()).inv()*s)[0]
        for batch in batches:
            ids,uppers=strict.screen(batch,0)
            lookup=dict(zip(map(int,ids),map(int,uppers)))
            for e in batch.ids:
                child=self.kernel.child(self.A,int(e))
                gain=Fraction(base-(s.T*sy.Matrix(child.toarray()).inv()*s)[0])
                if int(e) in lookup:
                    self.assertLessEqual(gain,Fraction(lookup[int(e)],UNIT**2))
                else:
                    self.assertLessEqual(gain,0)

    def test_singleton_screens_use_bound_state_cache(self):
        strict=CachedStrict(self.kernel,self.A,self.s,self.state[1],self.rows[0])
        before=strict.version_validation_scans
        for _ in range(7):strict.screen(np.array([2],np.int64),0)
        self.assertEqual(strict.version_validation_scans,before)
        self.assertEqual(before,1)

    def test_transport_is_existing_numerical_operation(self):
        views=self.kernel.views(Ledger(False),defaultdict(int))
        for e in [1,2]:views.score(e,0,self.A,self.fac,*self.state[1:3])
        z=views.advance(2,0,self.A,self.fac,self.state[0])
        child=self.kernel.child(self.A,2)
        next_state,_=lp.certify_child(child,self.fac,self.s,z,UNIT,self.rows)
        self.assertEqual(views.entries[1][0],1)
        self.assertEqual(views.transfers,1)
        lo,hi=views.score(1,1,child,self.fac,*next_state[1:3])
        self.assertLessEqual(lo,hi)
        self.assertEqual(views.hits,1)


if __name__=='__main__':unittest.main()
