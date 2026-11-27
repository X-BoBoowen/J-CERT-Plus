"""Certified local Gram views for exactly assembled dyadic signed operators.

The local basis V has entries 0,+1,-1; B=J/w+V.T@inv(A)@V. Exact
rational small-block arithmetic complements a binary64 sparse residual bound.
"""
from fractions import Fraction as Q
import math

import numpy as np

from bounds import inverse_error, score_radius, transport_radius
from engine import ViewManager


class Unresolved(ArithmeticError):
    """Available enclosures do not decide the requested comparison."""


def qarray(a):
    return np.asarray([Q(float(v)) for v in np.ravel(a)], dtype=object).reshape(np.shape(a))


def norm_cap(a):
    """Entrywise one-norm upper bounds both spectral and Euclidean norms."""
    return sum(abs(v) for v in np.ravel(a))


def inverse(B):
    if B.shape == (1, 1):
        if not B[0, 0]:
            raise Unresolved("singular approximate pivot")
        return np.array([[1 / B[0, 0]]], dtype=object)
    a, b, c, d = B.flat
    det = a*d-b*c
    if not det:
        raise Unresolved("singular approximate pivot")
    return np.array([[d, -b], [-c, a]], dtype=object) / det


class CertifiedViewManager(ViewManager):
    """Forward edits, residual-certified positivity and delta-greedy choice."""

    def __init__(self, A, edits):
        super().__init__(A, edits)
        if self.base.shape[0] != self.base.shape[1] or (self.base != self.base.T).nnz:
            raise ValueError("symmetric square base required")
        if not self.base.has_canonical_format or not np.all(np.isfinite(self.base.data)):
            raise ValueError("finite canonical base required")
        n = self.base.shape[0]
        # Work on a common dyadic grid. A conservative global absolute-sum
        # bound implies all partial sums in assembly are exactly representable.
        vals = [Q(float(a)) for a in self.base.data]
        weights = [Q(float(e.w)) for e in self.edits]
        grid = max([1, *(v.denominator for v in vals+weights)])
        if (sum(map(abs, vals)) + 4*sum(map(abs, weights))) * grid >= 2**53:
            raise ValueError("operator exceeds exact assembly grid")
        base = self.base.tocsr()
        self.max_row = Q(0)
        for i in range(n):
            row = base.data[base.indptr[i]:base.indptr[i+1]]
            rowsum = sum(abs(Q(float(v))) for v in row)
            if 2*Q(float(base[i,i]))-rowsum < 1:
                raise ValueError("base lacks signed diagonal dominance margin one")
            self.max_row = max(self.max_row, rowsum)
        seen = set()
        for e, w in zip(self.edits, weights):
            pair = tuple(sorted((e.i,e.j)))
            if not (0 <= pair[0] < pair[1] < n) or pair in seen or w <= 0:
                raise ValueError("invalid catalog")
            if Q(float(base[e.i,e.j])) != (w if e.flip else 0):
                raise ValueError("illegal flip or addition")
            seen.add(pair)
        self.max_row += 2*sum(weights)
        self.local = tuple((np.array([[1.,1.],[-1.,1.]])[:, :len(signs)], signs)
                           for _, signs in self.local)
        self.errors = {}
        self._nroot = math.isqrt(n-1)+1
        self.last_intervals = {}

    def _activate(self, state):
        if any(e < 0 or e >= len(self.edits) for e in state) or len(set(state)) != len(state):
            raise ValueError("invalid state")
        super()._activate(state)

    def _residual(self, x, rhs):
        r = rhs - self.A @ x
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(x)):
            raise Unresolved("nonfinite sparse residual")
        if not np.any(x) and not np.any(rhs):
            return Q(0)
        # Symmetry makes the maximum CSC column length a row-length cap.
        k = 2*int(np.max(np.diff(self.A.indptr)))+2
        u = Q(1,2**53)
        if k*u >= 1:
            raise Unresolved("residual roundoff bound undefined")
        g = k*u/(1-k*u)
        mag = self.max_row*Q(float(np.max(abs(x)))) + Q(float(np.max(abs(rhs))))
        # One absolute underflow term per operation, propagated through sums.
        return self._nroot*(Q(float(np.max(abs(r)))) + g*mag + k*Q(1,2**1074)/(1-k*u))

    def _projection(self, e, x):
        edit = self.edits[e]
        rows = qarray(x[[edit.i,edit.j]])
        return self.local[e][0].astype(int).T @ rows

    def _fresh(self, key, e, kind):
        Y = self._columns(key,e,kind)
        edit = self.edits[e]
        rhs = np.zeros_like(Y)
        rhs[[edit.i,edit.j]] = self.local[e][0]
        # Sum column 2-norm bounds to bound the block spectral norm.
        eta = sum(self._residual(Y[:,k],rhs[:,k]) for k in range(Y.shape[1]))
        return Y, self._projection(e,Y), 2*eta

    def gram(self, state, e):
        key = tuple(sorted(state))
        cache = self.views.setdefault(key,{})
        if e not in cache:
            _, G, error = self._fresh(key,e,"refine")
            cache[e] = self._store(key,e,G,error)
        else:
            self.ledger["gram_hits"] += 1
        return cache[e]

    def _store(self,key,e,G,error):
        center = np.asarray(G,dtype=float)
        radius = error + norm_cap(qarray(center)-G)
        rounded = float(radius)
        self.errors[key,e] = math.nextafter(rounded,math.inf) if Q(rounded)<radius else rounded
        return center

    def _pivot(self,e,G,error):
        w = Q(float(self.edits[e].w))
        B = G + np.diag(self.local[e][1].astype(int))/w
        T = inverse(B)
        beta = w*(1+2*w)
        return T,beta,Q(inverse_error(beta,0,error,norm_cap(T)))

    def _interval(self,key,e,z,rho):
        h = self._projection(e,z)
        G = qarray(self.gram(key,e))
        T,beta,eT = self._pivot(e,G,self.errors[key,e])
        mid = h @ T @ h
        err = Q(score_radius(beta,norm_cap(h),2*rho,eT,0))
        return mid-err,mid+err

    def select(self,state,s,delta=0.,pool=None):
        key = tuple(sorted(state))
        delta = Q(float(delta))
        if delta < 0:
            raise ValueError("nonnegative delta required")
        rhs = np.asarray(s,dtype=float)
        if rhs.shape != (self.base.shape[0],) or not np.all(np.isfinite(rhs)):
            raise ValueError("finite opinion vector required")
        z = self.solve(key,rhs)
        rho = self._residual(z,rhs)
        ids = sorted(set(range(len(self.edits)) if pool is None else pool)-set(key))
        if any(e<0 or e>=len(self.edits) for e in ids):
            raise ValueError("invalid pool")
        # Exact tangent upper bound drops the nonnegative quadratic remainder.
        caps = {}
        for e in ids:
            h = self._projection(e,z)
            hp = max(Q(0),abs(h[-1])-2*rho) if self.edits[e].flip else 0
            caps[e] = Q(float(self.edits[e].w))*((abs(h[0])+2*rho)**2-hp**2)
        self.ledger["screens"] += len(ids)
        intervals = {e:(None,caps[e]) for e in ids}
        best,lower = None,Q(0)
        for e in sorted(ids,key=lambda j:(-caps[j],j)):
            if caps[e] <= 0 or (best is not None and caps[e] <= lower+delta):
                break
            lo,hi = self._interval(key,e,z,rho)
            intervals[e] = lo,min(hi,caps[e])
            if lo>lower:
                best,lower = e,lo
        self.last_intervals = intervals
        if best is not None and all(j==best or hi<=lower+delta for j,(_,hi) in intervals.items()):
            value = float(lower)
            return best, math.nextafter(value,-math.inf) if Q(value)>lower else value
        if all(hi<=0 for _,hi in intervals.values()):
            return None,0.
        raise Unresolved("gain intervals overlap the decision boundary")

    def score(self,state,e,z):
        raise TypeError("a certified decision requires the opinion RHS; use select")

    def screen(self,state,e,z):
        raise TypeError("a certified screen requires the opinion RHS; use select")

    def commit(self,state,e):
        key = tuple(sorted(state))
        self._activate(key)
        if e in key or not 0<=e<len(self.edits):
            raise ValueError("invalid commit")
        child = tuple(sorted((*key,e)))
        cache = self.views.setdefault(child,{})
        pending = [f for f in self.views.get(key,{}) if f!=e and f not in cache]
        if pending:
            Y,G,eC = self._fresh(key,e,"winner")
            T,beta,eT = self._pivot(e,G,eC)
            for f in pending:
                C = self._projection(f,Y)
                updated = qarray(self.views[key][f])-C@T@C.T
                radius = Q(transport_radius(self.errors[key,f],beta,norm_cap(C),eC,eT,0))
                cache[f] = self._store(child,f,updated,radius)
            self.ledger["transported"] += len(pending)
        return child

    def gram_bytes(self):
        return super().gram_bytes()+8*len(self.errors)
