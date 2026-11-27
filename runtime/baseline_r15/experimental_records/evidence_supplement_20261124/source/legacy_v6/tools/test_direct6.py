"""Exercise native residual bounds with exact rational residuals."""
from fractions import Fraction
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from direct6 import DirectPlanner, compile_residual
from direct6_slots import SlotPlanner
from engine import Edit
from policy_kernel import CertificateError


class DirectResidualTest(unittest.TestCase):
    def test_fixed_slot_parent_restore(self):
        # A reversible flip and insertion share endpoint zero.
        A = csr_matrix([[2., 1., 0.], [1., 2., 0.], [0., 0., 1.]])
        edits = [Edit(0, 1, 1., True), Edit(0, 2, .5, False)]
        with patch("policy_kernel.CholmodFactor"):
            kernel = SlotPlanner(A, edits, backend="cholmod", residual=None)
            try:
                for e in range(2):
                    parent = kernel.A.toarray()
                    kernel.commit(e)
                    child = kernel.A.toarray()
                    self.assertFalse(np.array_equal(parent, child))
                    def fail_at_parent(_self, _pending):
                        np.testing.assert_array_equal(kernel.A.toarray(), parent)
                        raise ArithmeticError("simulated failed recovery")
                    with patch.object(DirectPlanner, "_historical", fail_at_parent):
                        with self.assertRaises(ArithmeticError):
                            kernel._historical(SimpleNamespace(winner=e))
                    np.testing.assert_array_equal(kernel.A.toarray(), child)
                    np.testing.assert_array_equal(kernel.absA.toarray(), abs(child))
                    np.testing.assert_array_equal(kernel.diag, np.diag(child))
                    self.assertEqual(kernel.logical_nnz, np.count_nonzero(child))
            finally:
                kernel.close()

    def test_exact_error_and_nonfinite(self):
        rng = np.random.default_rng(17)
        A = csr_matrix([[4., -.5, 2.], [-.5, 3., 1.], [2., 1., 5.]])
        kernel = DirectPlanner.__new__(DirectPlanner)
        kernel.A = A
        kernel.cap_work = np.empty(3)
        with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            kernel.native_residual = compile_residual(directory)
            for exponent in (-270, -15, 0, 90):
                x = (rng.normal(size=6)*10.**exponent)[::2]
                for rhs in (A@x, rng.normal(size=3)*10.**exponent):
                    exact = [Fraction(b)-sum(Fraction(a)*Fraction(y)
                             for a, y in zip(row, x)) for row, b in zip(A.toarray(), rhs)]
                    radius = kernel.residual_radius(rhs, x)
                    self.assertGreaterEqual(Fraction(radius)**2, sum(v*v for v in exact))
            with self.assertRaises(CertificateError):
                kernel.residual_radius(np.zeros(3), np.full(3, np.inf))


if __name__ == "__main__":
    unittest.main()
