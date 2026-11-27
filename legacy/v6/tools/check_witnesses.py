#!/usr/bin/env python3
"""Check the manuscript's small witnesses with exact SymPy arithmetic.

Run from the project root: python tools/check_witnesses.py
Requires SymPy. This checks exact synthetic witnesses only; it does not
validate empirical datasets, execution logs, timing, or benchmark results.

The graph operators are assembled independently from their signed edges.
All arithmetic and expected values below are rational or symbolic.
"""

from itertools import product

import sympy as sp


Q = sp.Rational


def signed_operator(n, edges, alpha=sp.Integer(1)):
    """Edges are (zero-based endpoint i, endpoint j, sign, weight)."""
    operator = sp.eye(n)
    for i, j, sign, weight in edges:
        incidence = sp.zeros(n, 1)
        incidence[i] = 1
        incidence[j] = -sign
        operator += alpha * weight * incidence * incidence.T
    return operator


def positive_addition(n, i, j):
    incidence = sp.zeros(n, 1)
    incidence[i] = 1
    incidence[j] = -1
    return incidence * incidence.T


def negative_to_positive_flip(n, i, j):
    negative_incidence = sp.zeros(n, 1)
    negative_incidence[i] = negative_incidence[j] = 1
    return positive_addition(n, i, j) - negative_incidence * negative_incidence.T


def equal(actual, expected, label):
    difference = actual - expected
    if isinstance(difference, sp.MatrixBase):
        valid = all(sp.simplify(value) == 0 for value in difference)
    else:
        valid = sp.simplify(difference) == 0
    if not valid:
        raise AssertionError(f"{label}: got {actual}; expected {expected}")


def energy(inverse, opinions):
    return sp.factor((opinions.T * inverse * opinions)[0])


def check_path_witnesses():
    # Theorem 1: alpha = 1; budget b = 2 allows all four subsets.
    operator = signed_operator(3, [(0, 1, -1, 1), (1, 2, -1, 1)])
    left = negative_to_positive_flip(3, 0, 1)
    right = negative_to_positive_flip(3, 1, 2)
    inverses = {
        (a, e): (operator + a * left + e * right).inv()
        for a, e in product([0, 1], repeat=2)
    }
    equal(inverses[0, 0], sp.Matrix([[5, -2, 1], [-2, 4, -2], [1, -2, 5]]) / 8,
          "Theorem 1 initial inverse")
    p, q, r = sp.symbols("p q r", real=True)
    opinions = sp.Matrix([p, q, r])
    gains = {key: sp.factor(energy(inverses[0, 0], opinions) - energy(inv, opinions))
             for key, inv in inverses.items()}
    equal(gains[1, 0], p * (r - 2 * q) / 2, "symbolic left-flip gain")
    equal(gains[0, 1], r * (p - 2 * q) / 2, "symbolic right-flip gain")
    equal(gains[1, 1], -q * (p + r), "symbolic joint gain")

    ordered_sets = [(0, 0), (1, 0), (0, 1), (1, 1)]
    batch_opinions = sp.Matrix([1, Q(1, 4), 1])
    batch_gain = [energy(inverses[0, 0], batch_opinions) - energy(inverses[k], batch_opinions)
                  for k in ordered_sets]
    equal(sp.Matrix(batch_gain), sp.Matrix([0, Q(1, 4), Q(1, 4), -Q(1, 2)]),
          "beneficial singletons, harmful batch")

    lockout_opinions = sp.Matrix([1, -Q(1, 10), -Q(1, 2)])
    lockout_energies = [energy(inverses[k], lockout_opinions) for k in ordered_sets]
    equal(sp.Matrix(lockout_energies), sp.Matrix([549, 669, 789, 509]) / 800,
          "Theorem 1 four lockout energies")
    lockout_gains = [lockout_energies[0] - value for value in lockout_energies]
    equal(sp.Matrix(lockout_gains), sp.Matrix([0, -Q(3, 20), -Q(3, 10), Q(1, 20)]),
          "Theorem 1 lockout gains")
    assert all(value < 0 for value in lockout_gains[1:3])
    equal(max(lockout_gains), Q(1, 20), "offline optimum at budget two")
    equal(max(lockout_gains[:3]), 0, "budget-one countercheck")

    stale_opinions = sp.Matrix([1, -Q(3, 10), -Q(1, 2)])
    stale_energies = [energy(inverses[k], stale_opinions) for k in ordered_sets]
    stale_gains = [stale_energies[0] - value for value in stale_energies]
    equal(sp.Matrix(stale_gains), sp.Matrix([0, Q(1, 20), -Q(2, 5), Q(3, 20)]),
          "introduction stale-score gains")
    equal(stale_energies[1] - stale_energies[3], Q(1, 10), "revived marginal")
    print("PASS: Theorem 1 formulas, both deployment witnesses, and stale-score witness.")


def check_finite_gap():
    # K_2,2 has parts {1,2} and {3,4}; all edges are positive, alpha = 1.
    edges = [(i, j, 1, Q(1, 2)) for i in [0, 1] for j in [2, 3]]
    operator = signed_operator(4, edges)
    first = positive_addition(4, 0, 1)
    second = positive_addition(4, 2, 3)
    opinions = sp.Matrix([1, -1, 1, -1]) / 2
    x, y = sp.symbols("x y", nonnegative=True)
    phi = energy((operator + x * first + y * second).inv(), opinions)
    equal(phi, 1 / (4 * (1 + x)) + 1 / (4 * (1 + y)), "finite-gap energy")
    initial = phi.subs({x: 0, y: 0})
    discrete = initial - phi.subs({x: 1, y: 0})
    fractional = initial - phi.subs({x: Q(1, 2), y: Q(1, 2)})
    equal(discrete, Q(1, 8), "finite discrete gain")
    equal(fractional, Q(1, 6), "finite fractional gain")
    equal(fractional / discrete, Q(4, 3), "finite gap ratio")
    # Both partial derivatives are negative. On x+y=1, this derivative
    # changes from negative to positive exactly at x=1/2.
    t = sp.symbols("t", real=True)
    boundary = phi.subs({x: t, y: 1 - t})
    equal(sp.diff(boundary, t),
          3 * (2 * t - 1) / (4 * (t + 1)**2 * (t - 2)**2),
          "fractional optimum on the budget boundary")
    print("PASS: Archived finite relaxation example, gains 1/8 and 1/6, ratio 4/3.")


def check_unbounded_gap():
    operator = signed_operator(3, [(0, 1, -1, 1), (1, 2, 1, 1)])
    flip = negative_to_positive_flip(3, 0, 1)
    addition = positive_addition(3, 0, 2)
    initial_inverse = operator.inv()
    quarter_inverse = (operator + Q(1, 4) * flip).inv()
    equal(initial_inverse, sp.Matrix([[5, -2, -1], [-2, 4, 2], [-1, 2, 5]]) / 8,
          "mixed initial inverse")
    equal(quarter_inverse, sp.Matrix([[20, -4, -2], [-4, 16, 8], [-2, 8, 23]]) / 38,
          "mixed fractional inverse")
    eta = sp.symbols("eta", positive=True)
    opinions = sp.Matrix([Q(6, 7), -Q(3, 14), 1 - eta])
    initial = energy(initial_inverse, opinions)
    flip_gain = sp.factor(initial - energy((operator + flip).inv(), opinions))
    addition_gain = sp.factor(initial - energy((operator + addition).inv(), opinions))
    fractional_gain = sp.factor(initial - energy(quarter_inverse, opinions))
    equal(flip_gain, 3 * (7 * eta - 4) / 49, "mixed flip gain")
    equal(addition_gain, 9 * eta**2 / 40, "mixed addition gain")
    # For 0<eta<4/7 the flip is negative, the addition positive, and the
    # empty-set gain is zero: addition is the budget-one discrete optimum.
    b13 = sp.Matrix([1, 0, -1])
    equal((b13.T * initial_inverse * b13)[0], Q(3, 2), "addition inverse form")
    equal((b13.T * initial_inverse * opinions)[0], 3 * eta / 4, "addition response")
    equal(fractional_gain, 3 * (49 * eta**2 + 252 * eta + 20) / 7448,
          "mixed fractional gain")
    equal(sp.limit(fractional_gain, eta, 0, dir="+"), Q(15, 1862),
          "positive fractional-gain limit")
    ratio = sp.factor(fractional_gain / addition_gain)
    equal(ratio.subs(eta, Q(1, 100)), Q(375415, 931), "eta=1/100 ratio")
    assert Q(375415, 931) > 403
    assert sp.limit(ratio, eta, 0, dir="+") == sp.oo
    print("PASS: Archived mixed relaxation family, all rational formulas, ratio 375415/931 > 403,")
    print("      and divergent feasible-fractional/discrete ratio as eta approaches zero.")


if __name__ == "__main__":
    print(f"Exact witness verification with SymPy {sp.__version__}")
    check_path_witnesses()
    check_finite_gap()
    check_unbounded_gap()
    print("All exact witness checks passed. No empirical results were checked.")
