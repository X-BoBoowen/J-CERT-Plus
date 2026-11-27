import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from src.data_projection import project


class ProjectionTests(unittest.TestCase):
    def test_timestamp_row_order_and_observed_directions(self):
        with TemporaryDirectory() as folder:
            src = Path(folder)/"fixture.csv.gz"
            with gzip.open(src, "wt") as f:
                f.writelines(["9,2,4,1.01\n","2,9,-3,2\n","9,2,2,1.02\n","9,2,8,1.02\n",
                              "3,9,2,2\n","9,3,-2,3\n","3,2,-7,2\n","90,90,1,5\n"])
            nodes, unit, weighted, info = project(src)
        self.assertEqual(nodes, [2,3,9,90])
        np.testing.assert_equal(weighted.toarray(), [[0,-7,2.5,0],[-7,0,0,0],[2.5,0,0,0],[0,0,0,0]])
        np.testing.assert_equal(np.sign(weighted.toarray()), unit.toarray())
        self.assertEqual((info["dropped_net_zero_pairs"],info["cutoff_timestamp"]), (1,"5"))
        self.assertEqual((info["superseded_rows"],info["tie_break_rows"]), (2,1))

    def test_reject_out_of_range(self):
        with TemporaryDirectory() as folder:
            src = Path(folder)/"invalid.csv.gz"
            with gzip.open(src, "wt") as f:
                f.write("1,2,11,1\n")
            with self.assertRaisesRegex(ValueError, "rating"):
                project(src)
