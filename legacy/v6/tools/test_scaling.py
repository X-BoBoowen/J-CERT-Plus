import unittest
from fractions import Fraction as F

from bounds import inverse_error, score_radius, transport_radius


class ScalingTest(unittest.TestCase):
    def test_distinct_candidate_and_winner_weights(self):
        qe, qa = F(4), F(1, 2)
        we, wa = qe**2, qa**2
        beta, t = 1 + 2*wa, F(1, 2)
        h, cross, gram = F(3, 4), F(1, 8), F(1, 2)
        eps = F(1, 64)
        scaled = inverse_error(beta, eps, eps, t)
        integral = inverse_error(wa*beta, eps, eps/wa, wa*t)
        self.assertEqual(integral, wa*scaled)
        self.assertEqual(h*h*t, (h/qa)**2*(wa*t))
        self.assertEqual(score_radius(beta, h, eps, scaled, eps),
                         score_radius(wa*beta, h/qa, eps/qa, integral, eps))
        product = qe*qa
        self.assertEqual((gram-cross**2*t)/we,
                         gram/we-(cross/product)**2*(wa*t))
        self.assertEqual(transport_radius(eps, beta, cross, eps, scaled, eps)/we,
                         transport_radius(eps/we, wa*beta, cross/product,
                                          eps/product, integral, eps/we))


if __name__ == "__main__":
    unittest.main()

