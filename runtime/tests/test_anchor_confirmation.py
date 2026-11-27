import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.anchor_confirmation import config,dev,load_fixture_record,matrix_check


class ConfirmationManifestTests(unittest.TestCase):
    def test_toy_checksums_and_order(self):
        x=np.array([[-65536,4000],[5,65536]],dtype=np.int64)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=root/'toy.npz'
            np.savez(path,opinions=x)
            h,rows=matrix_check(x,2,2)
            rec=dict(file=path.name,shape=[2,2],npz_sha256=dev.sha(path),matrix_sha256=h,opinion_hashes=rows)
            np.testing.assert_array_equal(load_fixture_record(rec,root),x)
            for bad in (dict(npz_sha256='wrong'),dict(matrix_sha256='wrong'),dict(opinion_hashes=rows[::-1])):
                with self.assertRaises(ValueError):load_fixture_record(rec|bad,root)

    def test_range_shape_and_dtype(self):
        for x in (np.array([[65537]],dtype=np.int64),np.array([[0]],dtype=np.int32),np.array([0],dtype=np.int64)):
            with self.assertRaises(ValueError):matrix_check(x,1,1)

    def test_protocol_schedule_no_service_execution(self):
        c=config()
        self.assertEqual(c['prefixes'],[1,2,4,8,16])
        self.assertEqual(len(list(dev.jobs(c))),54)
        self.assertEqual(c['seed'],196613)


if __name__=='__main__':unittest.main()
