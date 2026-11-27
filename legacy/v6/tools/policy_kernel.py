"""Forward-path certified signed-edit policy shared by the full-graph runs."""
from collections import Counter
from fractions import Fraction
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

from bounds import _up, inverse_error, score_radius
from certified_engine import Unresolved, inverse, norm_cap, qarray
from cholmod_backend import CholmodFactor
from engine import Edit

CertificateError = Unresolved
EPS = np.finfo(float).eps
up = lambda x: np.nextafter(x, np.inf)
down = lambda x: np.nextafter(x, -np.inf)


class PolicyKernel:
    def __init__(self, A, edits, backend="cg", cached=True, root_factor=None, winner_slots=0, directional=True, own_root_factor=False):
        if backend not in ("cg", "cholmod") or cached not in (True, False, "same"):
            raise ValueError("unknown policy or solver")
        self.backend, self.cached = backend, cached
        self.directional = bool(directional)
        self.A = sparse.csr_matrix(A, dtype=float, copy=True)
        self.A.sum_duplicates()
        self.A.sort_indices()
        self.edits = [e if isinstance(e, Edit) else Edit(*e) for e in edits]
        self.i = np.array([e.i for e in self.edits], dtype=int)
        self.j = np.array([e.j for e in self.edits], dtype=int)
        self.w = np.array([e.w for e in self.edits], dtype=float)
        self.flip = np.array([e.flip for e in self.edits], dtype=bool)
        n = self.A.shape[0]
        if self.A.shape != (n,n) or not n or (self.A != self.A.T).nnz:
            raise ValueError("expected a symmetric nonempty operator")
        pairs = np.minimum(self.i,self.j).astype(np.int64)*n + np.maximum(self.i,self.j)
        if (np.any(self.i == self.j) or min(self.i.min(initial=0), self.j.min(initial=0)) < 0
                or max(self.i.max(initial=0),self.j.max(initial=0)) >= n
                or np.unique(pairs).size != len(self.edits) or np.any(self.w <= 0)):
            raise ValueError("invalid edit catalog")
        self.off = np.asarray(self.A[self.i,self.j]).ravel() if len(pairs) else np.empty(0)
        if not np.array_equal(self.off,np.where(self.flip,self.w,0)):
            raise ValueError("illegal edit")
        # Ensure all partial row sums and committed entries fit a common dyadic grid.
        for k in range(54):
            grid = 2.**k
            if all(np.all(v*grid == np.rint(v*grid)) for v in (self.A.data,self.w)):
                break
        else:
            raise ValueError("exact assembly grid unavailable")
        maximum = np.max(abs(self.A.data),initial=1.) + 4*np.sum(self.w)
        if not np.isfinite(maximum) or maximum*grid*n >= 2**51:
            raise ValueError("exact row assembly range exceeded")
        self._refresh()
        if np.any(2*self.diag - np.asarray(self.absA.sum(axis=1)).ravel() < 1):
            raise ValueError("base lacks signed dominance margin one")
        if own_root_factor and (root_factor is None or backend != "cholmod"):
            raise ValueError("ownership transfer requires a root CHOLMOD factor")
        self.factor = None
        if backend == "cholmod":
            if root_factor is None:
                self.factor = CholmodFactor(self.A)
            else:
                self.factor = root_factor if own_root_factor else root_factor.clone()
        self.ledger = Counter(factorizations=int(self.factor is not None and root_factor is None),
                              factor_clones=int(root_factor is not None and not own_root_factor))
        self.active = np.zeros(len(pairs),dtype=bool)
        self.seen = np.zeros(len(pairs),dtype=bool)
        self.rows = {}
        self.stale = set()
        if not isinstance(winner_slots,int) or winner_slots < 0:
            raise ValueError("winner_slots must be a nonnegative integer")
        self.winner_slots = winner_slots
        self.clear_winner_columns()
        self.last_upper = 0.

    def _refresh(self):
        self.diag = self.A.diagonal()
        self.absA = abs(self.A)
        self.gamma = (2*np.diff(self.A.indptr)+4)*EPS
        invdiag = 1/self.diag
        self.invdiag = invdiag
        self.prec = LinearOperator(self.A.shape, matvec=lambda x: invdiag*x, dtype=float)

    def solve(self, rhs):
        rhs = np.asarray(rhs,dtype=float)
        if rhs.shape != (self.A.shape[0],) or not np.isfinite(rhs).all():
            raise ValueError("invalid RHS")
        if self.factor is not None:
            x = self.factor.solve(rhs)
        else:
            def count(_):
                self.ledger["cg_iterations"] += 1
            x, status = cg(self.A,rhs,M=self.prec,rtol=2e-12,atol=0.,callback=count)
            if status:
                raise CertificateError("PCG did not converge")
        return x,self.residual_radius(rhs,x)

    def residual_radius(self,rhs,x):
        r = abs(rhs-self.A@x)
        # Standard per-row dot-product error, with factor four covering
        # rounding in the nonnegative magnitude product and upper-cap assembly.
        cap = r + 4*self.gamma*(abs(rhs)+self.absA@abs(x)) + 1e-300
        scale = np.max(cap)
        rho = float(up(scale*np.sqrt(np.sum((cap/scale)**2))*(1+8*len(x)*EPS)))
        if not math.isfinite(rho):
            raise CertificateError("nonfinite residual radius")
        return rho

    def upper(self, ids, z, rho):
        ids = np.asarray(ids,dtype=int)
        i,j,w,f = self.i[ids],self.j[ids],self.w[ids],self.flip[ids]
        x,y = z[i],z[j]
        # a = w(d^2 - flip*p^2); enclose both projections of exact z.
        d,p = x-y,x+y
        err = up(2*rho + up(4*EPS*(abs(x)+abs(y))) + 1e-300)
        dh,pl = up(abs(d)+err),np.maximum(0.,down(abs(p)-err))
        a = up(w*up(up(dh*dh)-np.where(f,down(pl*pl),0.)))
        xx, yy, t = self.diag[i]+w*(~f), self.diag[j]+w*(~f), self.off[ids]-w*(1+f.astype(int))
        v1,v2 = np.where(f,-2*w*y,w*d), np.where(f,-2*w*x,-w*d)
        det = xx*yy-t*t
        b,c = (yy*v1-t*v2)/det,(xx*v2-t*v1)/det
        # Evaluate the variational upper bound a - 2 b^T v + b^T D b
        # with directed operations. b,c are arbitrary binary64 trial entries.
        ve = up(w*up(2*err + 8*EPS*(abs(x)+abs(y)) + 1e-300))
        dot = down(down(b*v1)+down(c*v2))
        dot = down(dot-up(up(abs(b)+abs(c))*ve))
        q = up(up(xx*up(b*b))+up(yy*up(c*c)))
        bc = np.where(t>=0,up(b*c),down(b*c))
        q = up(q+up((2*t)*bc))
        cap = np.minimum(a,up(up(a-down(2*dot))+q))
        if not np.isfinite(cap).all():
            raise CertificateError("nonfinite endpoint bound")
        return cap

    def _project(self,e,Y):
        a,b = qarray(Y[[self.i[e],self.j[e]]])
        return np.array([a-b,a+b] if self.flip[e] else [a-b],dtype=object)

    def _fresh(self,e,kind):
        r = 1+int(self.flip[e])
        self.ledger[kind] += 1
        self.ledger[kind+"_rhs"] += r
        self.ledger["winner_rhs" if kind=="W" else "refine_rhs"] += r
        rhs = np.zeros(self.A.shape[0])
        rhs[self.i[e]] = 1.
        Y = np.empty((len(rhs),r))
        err = Fraction(0)
        for k in range(r):
            rhs[self.j[e]] = -1. if k==0 else 1.
            Y[:,k],rho = self.solve(rhs)
            err += 2*Fraction(rho)
        return Y,self._project(e,Y),err

    def _pivot(self,e,G,error):
        w = Fraction(self.w[e])
        signs = [1,-1] if self.flip[e] else [1]
        T = inverse(G+np.diag([Fraction(s)/w for s in signs]))
        beta = w*(1+2*w)
        return T,beta,inverse_error(beta,0,error,norm_cap(T))

    def _store(self,e,G,err):
        center = np.asarray(G,dtype=float)
        self.rows[e] = center,_up(err+norm_cap(G-qarray(center)))

    def _interval(self,e,z,rho,refresh=False,capture=None):
        if e not in self.rows or refresh:
            kind = "R" if refresh else ("F" if self.seen[e] else "C")
            Y,G,err = self._fresh(e,kind)
            if capture is not None:
                capture.append((Y,G,err))
            self.seen[e] = True
            self.stale.discard(e)
            if self.cached:
                self._store(e,G,err)
        else:
            center,err = self.rows[e]
            G = qarray(center)
            self.ledger["gram_hits"] += 1
        h = self._project(e,z)
        T,beta,eT = self._pivot(e,G,err)
        mid = h@T@h
        radius = Fraction(score_radius(beta,norm_cap(h),2*Fraction(rho),eT,0))
        lo,hi = mid-radius,mid+radius
        if self.directional:
            # Unscaled B=J/w+G has true inverse norm cap w(1+2w).
            B = inverse(T)
            p = qarray(np.asarray(T@h,dtype=float))
            pn = norm_cap(p)
            eh,eg = 2*Fraction(rho),Fraction(err)
            s_cap = norm_cap(h-B@p)+eh+eg*pn
            center = 2*(h@p)-p@B@p
            eta_g = 0  # exact rational functional evaluation
            width = 2*eh*pn + eg*pn**2 + beta*s_cap**2 + eta_g
            lo,hi = max(lo,center-width),min(hi,center+width)
            self.ledger["directional_scores"] += 1
        return down(float(lo)),up(float(hi))

    def select(self,s,pool=None,delta=0.):
        self.ledger["state_rhs"] += 1
        z,rho = self.solve(s)
        return self.select_solved(s,z,rho,pool,delta)

    def select_solved(self,s,z,rho,pool=None,delta=0.):
        """Use z with a certified error radius for this exact state and RHS."""
        if not math.isfinite(delta) or delta<0:
            raise ValueError("invalid delta")
        ids = np.arange(len(self.edits)) if pool is None else np.unique(np.asarray(pool,dtype=int))
        if ids.size and (ids.min()<0 or ids.max()>=len(self.edits)):
            raise ValueError("invalid candidate")
        ids = ids[~self.active[ids]]
        if not np.any(s):
            self.last_upper = 0.
            return None,0.
        cap = self.upper(ids,z,rho)
        self.ledger["screens"] += len(ids)
        order = np.lexsort((ids,-cap))
        refresh = set()
        saved_edge,saved_response = None,None
        while True:
            best,lower = None,0.
            highs = {}
            best_response = None
            for position,k in enumerate(order):
                if cap[k]<=0 or (best is not None and cap[k]<=down(lower+delta)):
                    break
                e = int(ids[k])
                if getattr(self,"batch_transport",False):
                    self._lookahead = (ids,order,cap,position,
                                       0. if best is None else down(lower+delta))
                capture = [] if self.winner_slots else None
                lo,hi = self._interval(e,z,rho,refresh=e in refresh,capture=capture)
                highs[e] = min(cap[k],hi)
                if lo>lower:
                    best,lower = e,lo
                    best_response = capture[0] if capture else (
                        saved_response if e==saved_edge else None)
                del capture
            unresolved = [e for e,hi in highs.items() if e!=best and hi>(
                down(lower+delta) if best is not None else 0)]
            if not unresolved:
                break
            refresh = self.stale.intersection([*unresolved,best])
            if not refresh:
                break
            # Reanchor only on an actual comparison, so its response can
            # follow the incumbent. Carry at most one earlier incumbent
            # across the retry; ordinary Gram hits allocate no columns.
            saved_edge,saved_response = best,best_response
        if unresolved:
            raise CertificateError("overlapping candidate or positivity intervals")
        if best_response is not None:
            self._retain_winner(best,best_response)
        self.last_upper = highs.get(best,0.)
        return best,float(lower)

    def clear_winner_columns(self):
        self._winner_columns = {}
        self._winner_operator = None

    def winner_column_bytes(self):
        return sum(response[0].nbytes for response in self._winner_columns.values())

    def _retain_winner(self,e,response):
        if self._winner_operator is not self.A:
            self.clear_winner_columns()
            self._winner_operator = self.A
        self._winner_columns.pop(e,None)
        if len(self._winner_columns) >= self.winner_slots:
            self._winner_columns.pop(next(iter(self._winner_columns)))
        self._winner_columns[e] = response
        self.ledger["peak_winner_column_bytes"] = max(
            self.ledger["peak_winner_column_bytes"],self.winner_column_bytes())

    def transition_columns(self,e):
        if self._winner_operator is self.A and e in self._winner_columns:
            self.ledger["winner_column_hits"] += 1
            return self._winner_columns.pop(e)
        return self._fresh(e,"W")

    def _transport(self,Y,T,beta,eT,eC):
        """Batched equation (16), padding addition Grams with a zero row/column.

        Let c,g,t be entrywise norm caps of each block. Computing C T C.T
        with two length-at-most-two dot products costs at most
        (2 gamma_2 + gamma_2^2) c^2 t in entrywise norm, followed by
        subtraction at cost u(g + (1+gamma_2)^2 c^2 t). The factor 128 eps
        below covers these operations and the norms used to bound them.
        """
        ids = np.fromiter(self.rows,dtype=int,count=len(self.rows))
        flip = self.flip[ids]
        grams = np.zeros((len(ids),2,2))
        radii = np.empty(len(ids))
        for k,f in enumerate(ids):
            center,radii[k] = self.rows[f]
            r = center.shape[0]
            grams[k,:r,:r] = center
        a,b = Y[self.i[ids]],Y[self.j[ids]]
        cross = np.stack([a-b,(a+b)*flip[:,None]],axis=1)
        inv = np.asarray(T,dtype=float)
        inv_error = _up(Fraction(eT)+norm_cap(T-qarray(inv)))
        # Entrywise one-norm bounds the spectral norm. Four projected
        # entries each cost <= u*(|a|+|b|) plus an underflow term.
        cross_error = (_up(eC) + 8*EPS*(abs(a)+abs(b)).max(axis=1) + 1e-300)
        cn = abs(cross).sum(axis=(1,2))
        tn = abs(inv).sum()
        gn = abs(grams).sum(axis=(1,2))
        updated = grams - (cross@inv)@cross.transpose(0,2,1)
        rounding = 128*EPS*(gn + cn*cn*tn) + 1e-295*(1+cn)*(1+tn)
        # All radius terms are nonnegative. Inflate for rounding in the
        # short sums/products (including norm caps and cross_error).
        radius = up((radii + _up(beta)*cross_error*(2*cn+cross_error)
                     + cn*cn*inv_error + rounding)*(1+64*EPS))
        if self.directional:
            # The true remainder after CP' + PC' - PBP' is
            # (C-PB)B^-1(C-PB)'. Symmetrization contracts spectral error.
            B = inverse(T)
            B = (B+B.T)/2
            bf = np.asarray(B,dtype=float)
            be = _up(Fraction(eC)+norm_cap(B-qarray(bf)))
            bn = abs(bf).sum()
            P = cross@inv
            pn = abs(P).sum(axis=(1,2))
            pb = P@bf
            rn = abs(cross-pb).sum(axis=(1,2))
            # 128 eps covers length-two dot products, subsequent
            # subtractions, and reductions over at most four entries.
            eta_res = 128*EPS*(cn+pn*bn) + 1e-295*(1+bn)*(1+pn)
            s_cap = rn + cross_error + pn*be + eta_res
            cp = cross@P.transpose(0,2,1)
            pbp = pb@P.transpose(0,2,1)
            raw = grams - (cp+cp.transpose(0,2,1)-pbp)
            trial = .5*(raw+raw.transpose(0,2,1))
            scale = gn+2*cn*pn+pn*pn*bn
            eta_g = 128*EPS*scale + 1e-295*(1+pn)**2*(1+bn)
            direct = up((radii + 2*cross_error*pn + be*pn*pn
                         + _up(beta)*s_cap*s_cap + eta_g)*(1+128*EPS))
            # Preserve the center of whichever enclosing ball is tighter.
            use = np.isfinite(direct) & (direct < radius)
            updated[use],radius[use] = trial[use],direct[use]
            self.ledger["directional_blocks"] += len(ids)
            self.ledger["directional_blocks_tighter"] += int(use.sum())
        if not np.isfinite(updated).all() or not np.isfinite(radius).all():
            raise CertificateError("nonfinite transported enclosure")
        for k,f in enumerate(ids):
            r = 1+int(flip[k])
            self.rows[f] = updated[k,:r,:r].copy(),float(radius[k])
        self.stale.update(ids)

    def commit(self,e):
        if not 0<=e<len(self.edits) or self.active[e]:
            raise ValueError("invalid commitment")
        self.rows.pop(e,None)
        self.stale.discard(e)
        if self.cached=="same":
            self.rows.clear()
        if self.rows:
            Y,G,eC = self.transition_columns(e)
            T,beta,eT = self._pivot(e,G,eC)
            self._transport(Y,T,beta,eT,eC)
            self.ledger["transported"] += len(self.rows)
        self.clear_winner_columns()
        edit = self.edits[e]
        if self.factor is not None:
            self.factor.change(edit)
            self.ledger["factor_up"] += 1
            self.ledger["factor_down"] += int(edit.flip)
        self._apply_sparse_edit(e)
        self.active[e] = True

    def _apply_sparse_edit(self,e):
        edit = self.edits[e]
        i,j,w = edit.i,edit.j,edit.w
        v = (0.,-2*w,-2*w,0.) if edit.flip else (w,-w,-w,w)
        self.A = self.A + sparse.csr_matrix((v,([i,i,j,j],[i,j,i,j])),shape=self.A.shape)
        self._refresh()

    def memory_bytes(self):
        arrays = [self.i,self.j,self.w,self.flip,self.off,self.active,self.seen,self.diag,self.invdiag,self.gamma]
        for matrix in (self.A,self.absA):
            arrays.extend([matrix.data,matrix.indptr,matrix.indices])
        return self.winner_column_bytes() + sum(a.nbytes for a in arrays) + sum(v.nbytes+8 for v,_ in self.rows.values()) + (self.factor.bytes if self.factor else 0)

    def close(self):
        self.clear_winner_columns()
        if self.factor is not None:
            self.factor.close()







