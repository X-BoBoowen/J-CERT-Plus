"""Float64 eager Gram transport for legal signed-edge edits on a fixed base."""
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class Edit:
    i: int
    j: int
    w: float
    flip: bool = False

    def local(self):
        signs = np.array([1., -1.]) if self.flip else np.array([1.])
        cols = np.array([[1., 1.], [-1., 1.]])[:, :len(signs)]
        return np.sqrt(self.w) * cols, signs


class ViewManager:
    """State IDs and candidate IDs are scoped to this immutable base and catalog."""

    def __init__(self, A, edits):
        self.base = sparse.csc_matrix(A, dtype=float, copy=True)
        self.edits = tuple(edits)
        self.local = tuple(edit.local() for edit in self.edits)
        self.i = np.array([e.i for e in self.edits], dtype=int)
        self.j = np.array([e.j for e in self.edits], dtype=int)
        self.w = np.array([e.w for e in self.edits])
        self.flip = np.array([e.flip for e in self.edits], dtype=bool)
        self.views = {}
        self.ledger = Counter()
        self._active = None

    def _activate(self, state):
        state = tuple(sorted(state))
        if self._active == state:
            return
        rows, cols, vals = [], [], []
        for e in state:
            edit = self.edits[e]
            q = [edit.i, edit.j]
            rows.extend(np.repeat(q, 2))
            cols.extend(np.tile(q, 2))
            w = edit.w
            vals.extend((0., -2*w, -2*w, 0.) if edit.flip else (w, -w, -w, w))
        update = sparse.coo_matrix((vals, (rows, cols)), shape=self.base.shape)
        self.A = (self.base + update).tocsc()
        self.diagonal = self.A.diagonal()
        self.offdiag = np.asarray(self.A[self.i,self.j]).reshape(-1) if len(self.i) else np.array([])
        self.factor = splu(self.A)
        self._active = state
        self.ledger["factorizations"] += 1

    def solve(self, state, s):
        self._activate(state)
        self.ledger["state_rhs"] += 1
        return self.factor.solve(np.asarray(s, dtype=float))

    def _columns(self, state, e, kind):
        self._activate(state)
        edit = self.edits[e]
        U, _ = self.local[e]
        rhs = np.zeros((self.base.shape[0], U.shape[1]))
        rhs[[edit.i, edit.j]] = U
        self.ledger[kind + "_rhs"] += U.shape[1]
        self.ledger[kind + "_blocks"] += 1
        return self.factor.solve(rhs)

    def _project(self, e, Y):
        edit = self.edits[e]
        return self.local[e][0].T @ Y[[edit.i, edit.j]]

    def gram(self, state, e):
        key = tuple(sorted(state))
        cache = self.views.setdefault(key, {})
        if e not in cache:
            cache[e] = self._project(e, self._columns(key, e, "refine"))
        else:
            self.ledger["gram_hits"] += 1
        return cache[e]

    def score(self, state, e, z):
        h = self._project(e, z)
        B = np.diag(self.local[e][1]) + self.gram(state, e)
        return float(h @ np.linalg.solve(B, h))

    def _screens(self, state, ids, z):
        self._activate(state)
        i, j = self.i[ids], self.j[ids]
        w, flip = self.w[ids], self.flip[ids]
        d, p = z[i] - z[j], z[i] + z[j]
        a = w * (d*d - flip*p*p)
        v1, v2 = w*(d - flip*p), w*(-d - flip*p)
        x, y = self.diagonal[i], self.diagonal[j]
        t = self.offdiag[ids]
        xx, yy = x + w*(~flip), y + w*(~flip)
        tt = t - w*(1 + flip.astype(int))
        hi = a - (yy*v1*v1 - 2*tt*v1*v2 + xx*v2*v2)/(xx*yy-tt*tt)
        lo = a - v1*v1 - v2*v2 + w/(1+2*w)*(v1-v2)**2
        lo = np.where(flip, lo, np.maximum(lo, a/(1+2*w)))
        hi = np.where(flip, hi, np.minimum(hi, a/(1+w*(x+y+2*t)/(x*y-t*t))))
        self.ledger["screens"] += len(ids)
        return lo, hi

    def screen(self, state, e, z):
        lo, hi = self._screens(state, np.array([e]), z)
        return float(lo[0]), float(hi[0])

    def select(self, state, s, delta=0., pool=None):
        z = self.solve(state, s)
        allowed = set(range(len(self.edits)) if pool is None else pool).difference(state)
        ids = np.array(sorted(allowed), dtype=int)
        _, upper_bounds = self._screens(state, ids, z)
        bounds = zip(upper_bounds, ids)
        incumbent, best = None, 0.
        for upper, e in sorted(bounds, key=lambda item: (-item[0], item[1])):
            if upper < best - 1e-12 * (1 + abs(best)):
                break
            gain = self.score(state, e, z)
            if gain > best or (gain == best and incumbent is not None and e < incumbent):
                incumbent, best = e, gain
        return (None, best) if best <= delta else (incumbent, best)

    def commit(self, state, e):
        key = tuple(sorted(state))
        if e in key:
            raise ValueError("edit already active")
        child = tuple(sorted((*key, e)))
        cache = self.views.setdefault(child, {})
        pending = [f for f in self.views.get(key,{}) if f != e and f not in cache]
        if pending:
            Y = self._columns(key, e, "winner")
            B = np.diag(self.local[e][1]) + self._project(e, Y)
            T = np.linalg.inv(B)
            for f in pending:
                cross = self._project(f, Y)
                cache[f] = self.views[key][f] - cross @ T @ cross.T
            self.ledger["transported"] += len(pending)
        return child

    def gram_bytes(self):
        return sum(G.nbytes for cache in self.views.values() for G in cache.values())



