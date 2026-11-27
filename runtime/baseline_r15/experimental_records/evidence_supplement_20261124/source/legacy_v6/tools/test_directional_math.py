"""Exact checks for the directional and deployment examples."""
from fractions import Fraction as F
import unittest


class DirectionalMath(unittest.TestCase):
    def test_long_reduction_cap(self):
        import numpy as np
        from flip_stress import EPS, vec_cap

        n = 500_000
        x = np.zeros(n)
        x[::2] = 2.**900
        x[1::2] = 2.**-900
        bound = float(vec_cap(x))
        self.assertTrue(np.isfinite(bound))
        exact_squared = (n//2)*(F(2)**1800 + F(2)**-1800)
        self.assertGreaterEqual(F(bound)**2, exact_squared)
        # Even when the rounded reduction happens to be exact, retain the
        # stipulated allowance for all possible summation orders.
        x.fill(0.)
        x[0] = 1.
        self.assertGreater(float(vec_cap(x)), 1+n*EPS)

    def test_small_spectral_caps(self):
        import numpy as np
        from flip_stress import norm

        # The sign pattern has two identical singular values sqrt(2)*scale.
        for exponent in (-1060, -900, 0, 900):
            scale = 2.**exponent
            A = scale*np.array([[1., 1.],[1., -1.]])
            self.assertGreaterEqual(F(float(norm(A)))**2, 2*F(scale)**2)
        self.assertEqual(float(norm(np.zeros((2,2)))), 0.)

    def test_path_and_isolates(self):
        # Three-node signed path inverse is conjugated by each endpoint flip.
        inverse8 = ((5, -2, 1), (-2, 4, -2), (1, -2, 5))
        for eps in (F(1, 100000), F(1, 32), F(49, 1000)):
            def objective(left, right):
                x = (left, -F(1, 4) - eps, -right * F(1, 2))
                return sum(x[i] * v * x[j] for i, row in enumerate(inverse8)
                           for j, v in enumerate(row)) / 8
            root = objective(1, 1)
            self.assertEqual(root - objective(-1, 1), eps)
            self.assertEqual(root - objective(1, -1), -F(3, 8) - eps / 2)
            self.assertEqual(objective(-1, 1) - objective(-1, -1),
                             F(1, 8) - eps / 2)
            self.assertEqual(3 * eps * (1 - F(1, 3)), 2 * eps)

    def test_weight_64(self):
        eb = F(1, 10**8)
        p = F(16, 129)
        gain = 16 * p
        self.assertEqual(gain, 2 - F(2, 129))
        self.assertEqual(256 * 129**2 * eb, F(4260096, 10**8))
        radius = p**2 * eb * (1 + 129 * eb)
        self.assertTrue(F(153, 10**12) < radius < F(155, 10**12))

    def test_indefinite_completion(self):
        b = (F(5, 3), -F(3, 7))
        h = (F(11, 5), F(4, 3))
        p = (F(2), F(-1))
        correction = sum(2*x*y - a*y*y for a, x, y in zip(b, h, p))
        residual = sum((x-a*y)**2/a for a, x, y in zip(b, h, p))
        self.assertEqual(sum(x*x/a for a, x in zip(b, h)), correction + residual)


if __name__ == "__main__":
    unittest.main()
