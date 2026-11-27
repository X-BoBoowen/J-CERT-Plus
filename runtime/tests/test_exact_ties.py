import unittest
from collections import defaultdict
from fractions import Fraction
from math import ceil, floor

import numpy as np
from scipy import sparse

from src.a2_catalog import AllCatalog
from src.a2_numeric import EndpointKernel
from src.common import DEN, UNIT, Ledger
from src.exact_ties import TieContext, certify_exchange
import long_promotion as lp


def fixture(n=17, edges=(), opinion=None):
    W = np.zeros((n, n), dtype=float)
    for i, j, w in edges:
        W[i, j] = W[j, i] = w
    kernel = EndpointKernel(AllCatalog(sparse.csr_matrix(W)))
    A = kernel.operator(sparse.csr_matrix(W))
    s = np.zeros(n, dtype=np.int64) if opinion is None else np.array(opinion, np.int64)
    if opinion is None:
        s[0], s[1], s[2] = DEN, -DEN, -DEN
    context = TieContext(A, s, 'parent-fixture', frozenset())
    return kernel, A, s, context


def exact_gain(A, s, i, j, d=1, flip=False):
    """Independent rational objective and literal sparse edit formula."""
    import sympy as sy
    parent = sy.Matrix(A.toarray().tolist())
    child = parent.copy()
    if flip:
        child[i, j] -= 2*d
        child[j, i] -= 2*d
    else:
        child[i, i] += d
        child[j, j] += d
        child[i, j] -= d
        child[j, i] -= d
    rhs = sy.Matrix([sy.Rational(int(v), DEN) for v in s])
    return Fraction((rhs.T*(parent.inv()-child.inv())*rhs)[0])


class ExactExchangeTests(unittest.TestCase):
    def test_insert_certificate_proves_exact_gain_without_collapsing_interval(self):
        kernel, A, s, ctx = fixture(n=4)
        gain = exact_gain(A, s, 0, 1)
        self.assertEqual(gain, Fraction(1, 72))
        self.assertEqual(gain, exact_gain(A, s, 0, 2))
        proof = certify_exchange(kernel, A, s/DEN, 1, 2, ctx)
        self.assertEqual(proof['swap'], [1, 2])
        self.assertEqual(proof['source_edit'], dict(i=0, j=1, flip=False, d=1))
        low, high = floor(gain*UNIT**2)-7, ceil(gain*UNIT**2)+9
        evaluated = {2: high+3, 1: high}
        evidence = {}
        winner, lower = kernel.resolve(A, s/DEN, evaluated, 2, low,
                                      tie_context=ctx, evidence=evidence)
        self.assertEqual((winner, lower), (1, low))
        self.assertEqual(evidence['shared_upper'], high)
        self.assertEqual(evidence['shared_lower'], low)
        self.assertGreater(evaluated[1], low)
        self.assertEqual(evidence['unresolved_ids'], [])

    def test_natural_weight_flips_have_exact_conjugate_child_updates(self):
        kernel, A, s, ctx = fixture(n=4, edges=((0, 1, -1.5), (0, 2, -1.5)))
        proof = certify_exchange(kernel, A, s/DEN, 1, 2, ctx)
        self.assertIsNotNone(proof)
        self.assertEqual(proof['source_edit']['d'], 24)
        self.assertTrue(proof['source_edit']['flip'])
        left = exact_gain(A, s, 0, 1, 24, True)
        self.assertGreater(left, 0)
        self.assertEqual(left, exact_gain(A, s, 0, 2, 24, True))

    def test_equal_endpoint_opinions_do_not_excuse_different_flip_weights(self):
        kernel, A, s, ctx = fixture(edges=((0, 1, -1.), (0, 2, -2.)))
        self.assertIsNone(certify_exchange(kernel, A, s/DEN, 1, 2, ctx))

    def test_different_operation_types_cannot_be_certified(self):
        kernel, A, s, ctx = fixture(edges=((0, 1, -1.),))
        self.assertIsNone(certify_exchange(kernel, A, s/DEN, 1, 2, ctx))

    def test_one_opinion_numerator_breaks_equality(self):
        kernel, A, s, ctx = fixture()
        s[2] += 1
        self.assertIsNone(certify_exchange(kernel, A, s/DEN, 1, 2, ctx))

    def test_rounded_float_opinion_is_not_the_exact_bound_source(self):
        kernel, A, s, ctx = fixture()
        other = s/DEN
        other[1] = np.nextafter(other[1], 0.)
        self.assertIsNone(certify_exchange(kernel, A, other, 1, 2, ctx))

    def test_commit_breaks_symmetry_and_stale_parent_context(self):
        kernel, A, s, ctx = fixture()
        child = kernel.child(A, 1*17+3)
        self.assertIsNone(certify_exchange(kernel, child, s/DEN, 1, 2, ctx))
        new = TieContext(child, s, 'child-fixture', frozenset({20}))
        self.assertIsNone(certify_exchange(kernel, child, s/DEN, 1, 2, new))

    def test_chosen_candidate_and_current_nonedge_legality_are_checked(self):
        kernel, A, s, _ = fixture()
        chosen = TieContext(A, s, 'parent-fixture', frozenset({1}))
        self.assertIsNone(certify_exchange(kernel, A, s/DEN, 1, 2, chosen))
        child = kernel.child(A, 1)
        childctx = TieContext(child, s, 'child-fixture', frozenset())
        self.assertIsNone(certify_exchange(kernel, child, s/DEN, 1, 2, childctx))

    def test_equal_rows_are_insufficient_for_asymmetric_or_noninteger_operator(self):
        kernel, A, s, _ = fixture()
        bad = A.tolil(); bad[4, 1] = 1; bad = bad.tocsr()
        ctx = TieContext(bad, s, 'bad-parent', frozenset())
        self.assertIsNone(certify_exchange(kernel, bad, s/DEN, 1, 2, ctx))
        real = A.astype(float)
        ctx = TieContext(real, s, 'float-parent', frozenset())
        self.assertIsNone(certify_exchange(kernel, real, s/DEN, 1, 2, ctx))

    def test_single_transposition_does_not_claim_disjoint_pair_equivalence(self):
        kernel, A, s, ctx = fixture(n=4, opinion=[DEN, -DEN, DEN, -DEN])
        self.assertEqual(exact_gain(A, s, 0, 1), exact_gain(A, s, 2, 3))
        self.assertIsNone(certify_exchange(kernel, A, s/DEN, 1, 11, ctx))

    def test_unproved_third_overlap_remains_unresolved_on_large_graph(self):
        kernel, A, s, ctx = fixture()
        gain = Fraction(1, 72)
        low, high = floor(gain*UNIT**2)-7, ceil(gain*UNIT**2)+9
        evidence = {}
        with self.assertRaisesRegex(ArithmeticError, 'unresolved'):
            kernel.resolve(A, s/DEN, {1: high, 2: high+1, 3: high+2}, 1, low,
                           tie_context=ctx, evidence=evidence)
        self.assertEqual(evidence['overlap_ids'], [2, 3])
        self.assertEqual(evidence['unresolved_ids'], [3])
        self.assertEqual(len(evidence['witnesses']), 1)
        self.assertNotIn('canonical_winner_id', evidence)

    def test_legacy_large_overlap_without_context_stays_unresolved(self):
        kernel, A, s, _ = fixture()
        with self.assertRaisesRegex(ArithmeticError, 'unresolved'):
            kernel.resolve(A, s/DEN, {1: 11, 2: 11}, 1, 10)

    def test_zero_unknown_sign_does_not_become_positive_by_equality(self):
        kernel, A, s, ctx = fixture(opinion=np.zeros(17, np.int64))
        evidence = {}
        with self.assertRaisesRegex(ArithmeticError, 'unresolved'):
            kernel.resolve(A, s/DEN, {1: 1, 2: 1}, None, 0,
                           tie_context=ctx, evidence=evidence)
        self.assertEqual(evidence, {})
        self.assertEqual(kernel.resolve(A, s/DEN, {1: 0, 2: -1}, None, 0,
                                        tie_context=ctx), (None, 0))

    def test_negative_gain_equality_never_accepts_an_edit(self):
        kernel, A, s, ctx = fixture(n=3, edges=((0, 1, -1.), (0, 2, -1.)),
                                  opinion=[DEN, DEN, DEN])
        g = exact_gain(A, s, 0, 1, 16, True)
        self.assertLess(g, 0)
        self.assertIsNotNone(certify_exchange(kernel, A, s/DEN, 1, 2, ctx))
        hi = ceil(g*UNIT**2)
        self.assertEqual(kernel.resolve(A, s/DEN, {1: hi, 2: hi}, None, 0,
                                        tie_context=ctx), (None, 0))

    def test_canonical_switch_commits_its_own_response_and_pivot(self):
        kernel, A, s, ctx = fixture()
        fac = lp.UpdatedFactor(A)
        state = lp.certified_state(A, fac, s, UNIT)
        views = kernel.views(Ledger(False), defaultdict(int))
        left = views.score(1, 0, A, fac, state[1], state[2])
        right = views.score(2, 0, A, fac, state[1], state[2])
        views.retained = views.scored_response
        self.assertEqual(views.retained[0], 2)
        winner, _ = kernel.resolve(A, s/DEN, {1: left[1], 2: right[1]}, 2,
                                  min(left[0], right[0]), tie_context=ctx)
        self.assertEqual(winner, 1)
        actual = views.advance(winner, 0, A, fac, state[0])
        child = kernel.child(A, 1)
        expected = np.linalg.solve(child.toarray().astype(float), s/DEN)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(fac.solve(np.eye(17)[:, 1]),
                                   np.linalg.solve(child.toarray(), np.eye(17)[:, 1]),
                                   rtol=1e-13, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
