"""Check the sparse search stack's shared structure through nested edits and undo."""
import unittest

import numpy as np

from engine import Edit
from policy_kernel import PolicyKernel
from sparse_search import SearchKernel, PreparedSearch, query


class SearchStackTests(unittest.TestCase):
    def test_prepared_consumer_isolation_and_reset(self):
        from scipy import sparse
        import os
        from pathlib import Path
        if not os.environ.get("JCERT_CHOLMOD_LIB") and not any((Path(__file__).parent/name).exists() for name in ("jcert_cholmod.dll","libjcert_cholmod.so","libjcert_cholmod.dylib")):
            self.skipTest("requires the compiled CHOLMOD wrapper")
        W = sparse.csr_matrix((4,4))
        case = dict(edits=[[0,1,1.,False],[2,3,1.,False]],q=[0,1,2,3],z=[.2,-.2,.1,-.1])
        prepared = PreparedSearch(W,case)
        fresh,_,_ = prepared.consumer(False)
        reuse,_,_ = prepared.consumer(True)
        try:
            self.assertTrue(np.shares_memory(fresh.A.indices,reuse.A.indices))
            self.assertFalse(np.shares_memory(fresh.A.data,reuse.A.data))
            self.assertIsNot(fresh.factor,reuse.factor)
            for node,cached in ((fresh,False),(reuse,True),(fresh,False)):
                path,c,_ = query(prepared,node,"dfs",cached)
                self.assertEqual(len(path),2)
                self.assertEqual(c["updates"],c["undos"])
                np.testing.assert_array_equal(node.A.data,prepared.prototype.A.data)
                np.testing.assert_allclose(node.factor.solve(prepared.s),prepared.prototype.factor.solve(prepared.s),atol=1e-14)
        finally:
            fresh.close()
            reuse.close()
            prepared.close()

    def test_nested_edits_and_sibling_reuse(self):
        A = np.array([[3.,2.,0.,0.],[2.,4.,1.,0.],[0.,1.,2.,0.],[0.,0.,0.,1.]])
        edits = [Edit(0,1,2,True),Edit(1,2,1,True),Edit(0,3,.5)]
        node = SearchKernel(A,edits)
        node.prepare_stack()
        structure = node.A.indices.copy(),node.A.indptr.copy()
        trial = np.array([-.75,.5,1.,-1.])
        rhs = np.array([1.,-.25,.5,-1.])

        def check(prefix):
            fresh = PolicyKernel(A,edits)
            for e in prefix:
                fresh.commit(e)
            np.testing.assert_array_equal(node.A.toarray(),fresh.A.toarray())
            np.testing.assert_array_equal(node.absA.toarray(),fresh.absA.toarray())
            np.testing.assert_array_equal(node.diag,fresh.diag)
            np.testing.assert_array_equal(node.invdiag,fresh.invdiag)
            np.testing.assert_array_equal(node.A.indices,structure[0])
            np.testing.assert_array_equal(node.A.indptr,structure[1])
            self.assertTrue(np.all(node.gamma>=fresh.gamma))
            self.assertGreaterEqual(node.residual_radius(rhs,trial),
                                    np.linalg.norm(rhs-fresh.A@trial))

        prefix = []
        for e,direction in ((2,1),(0,1),(1,1),(1,-1),(0,-1),(1,1),(1,-1),(2,-1)):
            node.change_entries(e,direction)
            if direction==1:
                prefix.append(e)
            else:
                self.assertEqual(prefix.pop(),e)
            check(prefix)


if __name__=="__main__":
    unittest.main()


