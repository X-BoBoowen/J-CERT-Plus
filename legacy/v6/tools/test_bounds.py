"""Check the enclosure and prior-contract scalar reference with unittest."""
import math
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction as Q

from bounds import coupling_sum, coupling_bound, inverse_error, prior_constants, score_radius, transport_radius


class BoundsTests(unittest.TestCase):
    def test_indefinite_flip(self):
        # Legal two-node negative edge of weight 1/2, flipped to positive.
        # B=diag(2,-1/2). Approximate each diagonal operation independently.
        true_t = (Q(1, 2), Q(-2))
        formed_b = (Q(2) + Q(3, 128), Q(-1, 2) + Q(1, 128))
        t = (Q(63, 128), Q(-63, 32))
        residual = max(abs(1 - b*x) for b, x in zip(formed_b, t))
        e_t = inverse_error(2, residual, Q(1, 64), max(map(abs, t)), Q(1, 128))
        self.assertGreaterEqual(e_t, max(abs(x-y) for x, y in zip(t, true_t)))
        h = (Q(1), Q(1, 2))
        ht = (Q(63, 64), h[1])
        def quadratic(vector, diagonal):
            return sum(x*x*y for x, y in zip(vector, diagonal))
        difference = abs(quadratic(h, true_t) - quadratic(ht, t))
        # L1 norm dominates the Euclidean norm of the represented vector.
        self.assertGreaterEqual(score_radius(2, sum(ht), Q(1, 64), e_t, 0), difference)
        self.assertGreaterEqual(transport_radius(0, 2, sum(ht), Q(1, 64), e_t, 0), difference)

    def test_scalar_rounding_and_assembly_term(self):
        x = Q(1, 3)
        self.assertGreaterEqual(inverse_error(1, 0, 0, 1, x), x)
        self.assertEqual(inverse_error(1, 0, 0, 1, 0), 0)
        self.assertGreaterEqual(score_radius(1, 1, x, 0, 0), 2*x+x*x)
        self.assertGreaterEqual(transport_radius(x, 1, 1, x, 0, x), 4*x+x*x)

    def test_observed_coupling(self):
        cs = [Q(0), Q(1, 8), Q(1, 2)]
        xi, zeta, eta = Q(1, 16), Q(1, 32), Q(1, 1024)
        radius = Q(1, 64)
        for c in cs:
            radius = transport_radius(radius, 3, c, xi, zeta, eta)
        cap = coupling_bound(Q(1, 64), 3, xi, zeta, max(cs), sum(cs), len(cs), len(cs)*eta)
        exact_sum = coupling_sum(Q(1, 64), [(3, c, xi, zeta, eta) for c in cs])
        self.assertEqual(radius, exact_sum)
        self.assertGreaterEqual(cap, exact_sum)
        self.assertEqual(coupling_bound(1, 3, xi, zeta, 0, 0, 0, 0), 1)

    def test_prior_constants(self):
        with localcontext() as ctx:
            ctx.prec = 80
            k, beta, b, delta = Decimal(1), Decimal(2), 3, Decimal(1)/512
            a = 4*k + 4
            c = k*k + a
            t = beta*(1+2*beta*a)
            ell = beta*(2*c*a+a*a) + c*c*t + 1
            H = a + b*ell
            hbar = Decimal(2).sqrt()*k+a
            F = beta*(2*hbar*a+a*a) + hbar*hbar*beta*(1+2*beta*H)+1
            expected = dict(k=k,beta=beta,a=a,c=c,t=t,ell=ell,H=H,hbar=hbar,F=F)
            got = prior_constants(0.5, b, float(delta))
            for key, value in expected.items():
                self.assertGreaterEqual(Decimal(got[key]), value)
                self.assertTrue(math.isclose(got[key], float(value), rel_tol=1e-15))
            eps = Decimal(got["epsilon"])
            limit = min(Decimal(1)/4, 1/(4*beta*H), delta/(8*F))
            self.assertLessEqual(eps, limit)
            self.assertTrue(math.isclose(float(eps), float(limit), rel_tol=1e-15))
            self.assertLess(beta*(1+eps)/(1-beta*H*eps), 2*beta)


if __name__ == "__main__":
    unittest.main()
