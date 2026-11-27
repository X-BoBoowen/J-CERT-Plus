"""Scalar reference implementation of (14)--(16) and (25)--(27).

Inputs are certified nonnegative caps. Exact rational operations preserve the
real values of binary64 inputs. Only returned values and sqrt(2w) are rounded.
"""
import math
from fractions import Fraction


def _q(value):
    value = Fraction(value)
    if value < 0:
        raise ValueError("bounds must be nonnegative")
    return value


def _up(value):
    result = float(value)
    if Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _down(value):
    result = float(value)
    if Fraction(result) > value:
        result = math.nextafter(result, 0.0)
    return result


def inverse_error(beta, eta_t, e_g, t_norm, eta_b=0):
    beta, eta_t, e_g, t_norm = map(_q, (beta, eta_t, e_g, t_norm))
    return _up(beta * (eta_t + (e_g + _q(eta_b)) * t_norm))


def score_radius(beta, h_norm, e_h, e_t, eta_score):
    beta, h, e_h, e_t, eta = map(_q, (beta, h_norm, e_h, e_t, eta_score))
    return _up(beta * (2 * h * e_h + e_h**2) + h**2 * e_t + eta)


def transport_radius(e_g, beta, c_norm, e_c, e_t, eta_upd):
    base, beta, c, e_c, e_t, eta = map(_q, (e_g, beta, c_norm, e_c, e_t, eta_upd))
    return _up(base + beta * (2 * c * e_c + e_c**2) + c**2 * e_t + eta)


def coupling_sum(base, steps):
    """Equation (26) with steps (beta, c_norm, e_c, e_t, eta_upd)."""
    total = _q(base)
    for step in steps:
        beta, c, xi, zeta, eta = map(_q, step)
        total += beta * (2*c*xi + xi**2) + c**2*zeta + eta
    return _up(total)


def coupling_bound(base, beta, xi, zeta, cmax, sigma, j, eta_sum):
    """Bound (27); sigma includes all j certified cross-norm upper bounds."""
    if not isinstance(j, int) or j < 0:
        raise ValueError("j must be a nonnegative integer")
    base, beta, xi, zeta, cmax, sigma, eta_sum = map(
        _q, (base, beta, xi, zeta, cmax, sigma, eta_sum))
    return _up(base + (2 * beta * xi + zeta * cmax) * sigma
               + j * beta * xi**2 + eta_sum)


def prior_constants(w, b, delta):
    """Equation (25); epsilon is rounded down. Overflow raises OverflowError."""
    if not isinstance(b, int) or b < 0:
        raise ValueError("b must be a nonnegative integer")
    w, delta = map(_q, (w, delta))
    root = math.sqrt(_up(2 * w))
    if Fraction(root)**2 < 2 * w:
        root = math.nextafter(root, math.inf)
    k = Fraction(root)
    beta = 1 + 2 * w
    a = 4 * k + 4
    c = k**2 + a
    t = beta * (1 + 2 * beta * a)
    ell = beta * (2 * c * a + a**2) + c**2 * t + 1
    H = a + b * ell
    sqrt_two = math.sqrt(2.0)
    if Fraction(sqrt_two)**2 < 2:
        sqrt_two = math.nextafter(sqrt_two, math.inf)
    hbar = Fraction(sqrt_two) * k + a
    F = beta * (2 * hbar * a + a**2) + hbar**2 * beta * (1 + 2 * beta * H) + 1
    constants = dict(k=k, beta=beta, a=a, c=c, t=t, ell=ell, H=H, hbar=hbar, F=F)
    result = {name: _up(value) for name, value in constants.items()}
    public_beta = Fraction(result["beta"])
    public_H = Fraction(result["H"])
    public_F = Fraction(result["F"])
    result["epsilon"] = _down(min(Fraction(1, 4), 1/(4*public_beta*public_H),
                                  delta/(8*public_F)))
    return result
