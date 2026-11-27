import json
import unittest
import numpy as np
from scipy import sparse
from src.a2_catalog import AllCatalog
from src.a2_session import A2Session


class V2Admission(unittest.TestCase):
    def test_compatible_direct_service_admits_10001(self):
        session = A2Session(sparse.csr_matrix((10001, 10001), dtype=np.int64), 'H')
        self.assertEqual(session.n, 10001)
        self.assertEqual(session.factor.base.shape, (10001, 10001))
        self.assertEqual(session.factor.steps, [])

    def test_high_endpoint_identity_and_small_bounded_batches(self):
        n = 82140
        i, j = n-2, n-1
        graph = sparse.csr_matrix(([-1,-1], ([i,j],[j,i])), shape=(n,n), dtype=np.int64)
        catalog = AllCatalog(graph)
        eid = i*n+j
        self.assertEqual(eid, 6746897459)
        self.assertGreater(eid, 2**32)
        self.assertGreater(catalog.m, 2**31)
        self.assertEqual(tuple(map(int, catalog.scalar(eid))), (i,j,1,16))
        batch = catalog.batch([eid, 1, 2])
        self.assertEqual(batch.ids.tolist(), [eid,1,2])
        self.assertEqual(json.loads(json.dumps({'ids':batch.ids.tolist()}))['ids'][0], eid)
        self.assertEqual(len(next(catalog.iter_blocks(block_size=4)).ids),4)

    def test_explicit_large_directory_remains_rejected(self):
        with self.assertRaisesRegex(ValueError, 'M<2\\^31'):
            A2Session(sparse.csr_matrix((82140,82140),dtype=np.int64),'D')


if __name__ == '__main__':
    unittest.main()
