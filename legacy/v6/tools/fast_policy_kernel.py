"""Optional binary64 local certificates for the scalar/2-by-2 policy hot path.

All norms below are entrywise sums, which dominate spectral norms. IEEE
round-to-nearest is assumed. FL covers 128 eps of accumulated forward error,
including the at-most-four-term norm reductions and scalar cap assembly.
"""
import math
from fractions import Fraction

import numpy as np

from bounds import _up
from certified_engine import norm_cap, qarray
from cost_planner import CostPlanner
from policy_kernel import CertificateError, PolicyKernel

FL = 128 * 2.**-52
TINY = 1e-295


def score_interval(x, y, w, flip, G, eg, rho):
    # Work with a symmetric binary64 pivot; symmetrizing the input Gram
    # contracts spectral error with respect to the symmetric exact Gram.
    iw = 1. / w
    a = float(G[0,0])+iw
    b = d = 0.
    h = x-y
    k = 0.
    if flip:
        b = .5*(float(G[0,1])+float(G[1,0]))
        d = float(G[1,1])-iw
        k = x+y
    gb = abs(float(G[0,0]))+iw
    if flip:
        gb += abs(float(G[1,1]))+2*iw+abs(float(G[0,1]))+abs(float(G[1,0]))
    eb = (eg+FL*gb+TINY)*(1+FL)
    eh = (2*rho+FL*(abs(x)+abs(y))+TINY)*(1+FL)
    bn,hn = abs(a)+2*abs(b)+abs(d),abs(h)+abs(k)
    beta = w*(1+2*w)*(1+FL)
    # Inversion only proposes trial entries. Its conditioning plays no role
    # in either residual proof. Refuse an unusable proposal.
    det = a*d-b*b if flip else a
    if not det or not math.isfinite(det):
        raise CertificateError("unusable floating pivot trial")
    t = d/det if flip else 1/a
    u = -b/det if flip else 0.
    v = a/det if flip else 0.
    p,q = t*h+u*k,u*h+v*k
    pn,tn = abs(p)+abs(q),abs(t)+2*abs(u)+abs(v)
    z1,z2 = a*p+b*q,b*p+d*q
    r = abs(h-z1)+abs(k-z2)
    # The exact h-pivot product residual includes both input uncertainties.
    sr = (r+eh+eb*pn+FL*(hn+bn*pn)+TINY*(1+bn+pn))*(1+FL)
    hp = h*p+k*q
    g = 2*hp-(p*z1+q*z2)
    eta = FL*(2*hn*pn+bn*pn*pn)+TINY*(1+pn)**2*(1+bn)
    rd = (2*eh*pn+eb*pn*pn+beta*sr*sr+eta)*(1+FL)
    # Independent isotropic certificate centered at the evaluated h'Th.
    it = abs(1-(a*t+b*u))
    if flip:
        it += abs(a*u+b*v)+abs(b*t+d*u)+abs(1-(b*u+d*v))
    zr = beta*(it+eb*tn+FL*(2+bn*tn)+TINY*(1+bn+tn))*(1+FL)
    ri = (beta*(2*hn*eh+eh*eh)+hn*hn*zr+FL*hn*hn*tn
          +TINY*(1+hn)**2*(1+tn))*(1+FL)
    lo = math.nextafter(max(hp-ri,g-rd),-math.inf)
    hi = math.nextafter(min(hp+ri,g+rd),math.inf)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo<=hi):
        raise CertificateError("nonfinite fast score enclosure")
    return lo,hi


class FastPolicyKernel(PolicyKernel):
    """Same decision protocol, binary64 local certificate at every refinement."""
    def _fresh(self,e,kind):
        rank = 1+int(self.flip[e])
        self.ledger[kind] += 1
        self.ledger[kind+"_rhs"] += rank
        self.ledger["winner_rhs" if kind=="W" else "refine_rhs"] += rank
        rhs = np.zeros((self.A.shape[0],rank),order="F")
        rhs[self.i[e]] = 1.
        rhs[self.j[e]] = [-1.,1.] if rank == 2 else [-1.]
        if self.factor is not None:
            Y = self.factor.solve(rhs)
            err = 2*sum(self.residual_radius(u,y) for u,y in zip(rhs.T,Y.T))
        else:
            Y = np.empty_like(rhs)
            err = 0.
            for k in range(rank):
                Y[:,k],rho = self.solve(rhs[:,k])
                err += 2*rho
        return self._response(e,Y,err)

    def _response(self,e,Y,err):
        rank = Y.shape[1]
        x,y = Y[self.i[e]],Y[self.j[e]]
        G = np.array([x-y,x+y] if rank==2 else [x-y])
        err = (err+FL*float(np.sum(abs(x)+abs(y)))+TINY)*(1+FL)
        return Y,G,err

    def _pivot(self,e,G,error):
        # Prepare both trial and original pivot once per parent. The one-row
        # demand path never inverts the tiny inverse to recover B.
        G = qarray(G)
        T,beta,et = super()._pivot(e,G,error)
        B = (G+G.T)/2
        iw = 1/Fraction(float(self.w[e]))
        B[0,0] += iw
        if self.flip[e]:
            B[1,1] -= iw
        bf,tf = np.asarray(B,dtype=float),np.asarray(T,dtype=float)
        self._transport_trial = (T,tf,bf,_up(Fraction(et)+norm_cap(T-qarray(tf))),
                                 _up(Fraction(error)+norm_cap(B-qarray(bf))),
                                 _up(beta),float(abs(tf).sum()),float(abs(bf).sum()))
        return T,beta,et

    def _transport(self,Y,T,beta,eT,eC):
        if len(self.rows)!=1 or self._transport_trial[0] is not T:
            return super()._transport(Y,T,beta,eT,eC)
        e,(gram,radius) = next(iter(self.rows.items()))
        _,tf,bf,et,eb,beta,tn,bn = self._transport_trial
        x,y = Y[self.i[e]],Y[self.j[e]]
        C = np.array([x-y,x+y] if self.flip[e] else [x-y])
        ec = (float(eC)+FL*float((abs(x)+abs(y)).sum())+TINY)*(1+FL)
        cn,gn = float(abs(C).sum()),float(abs(gram).sum())
        P = C@tf
        pn = float(abs(P).sum())
        old = gram-P@C.T
        old_rad = (radius+beta*ec*(2*cn+ec)+cn*cn*et
                   + FL*(gn+cn*cn*tn)+TINY*(1+cn)*(1+tn))*(1+FL)
        pb = P@bf
        sr = (float(abs(C-pb).sum()) + ec + eb*pn
              + FL*(cn+pn*bn)+TINY*(1+bn)*(1+pn))*(1+FL)
        cp = C@P.T
        new = gram-(cp+cp.T-pb@P.T)
        new = .5*(new+new.T)
        new_rad = (radius+2*ec*pn+eb*pn*pn+beta*sr*sr
                   +FL*(gn+2*cn*pn+pn*pn*bn)+TINY*(1+pn)**2*(1+bn))*(1+FL)
        use = new_rad<old_rad
        center,rad = (new,new_rad) if use else (old,old_rad)
        if not math.isfinite(rad) or not np.isfinite(center).all():
            raise CertificateError("nonfinite single-row certificate")
        self.rows[e] = center,math.nextafter(rad,math.inf)
        self.stale.add(e)
        self.ledger["directional_blocks"] += 1
        self.ledger["directional_blocks_tighter"] += int(use)

    def _interval(self,e,z,rho,refresh=False,capture=None):
        if not self.directional:
            raise ValueError("fast kernel requires directional=True")
        if e not in self.rows or refresh:
            kind = "R" if refresh else ("F" if self.seen[e] else "C")
            Y,G,err = self._fresh(e,kind)
            if capture is not None:
                capture.append((Y,G,err))
            self.seen[e] = True
            self.stale.discard(e)
            if self.cached:
                self.rows[e] = G,err
        else:
            G,err = self.rows[e]
            self.ledger["gram_hits"] += 1
        self.ledger["directional_scores"] += 1
        return score_interval(float(z[self.i[e]]),float(z[self.j[e]]),
                              float(self.w[e]),self.flip[e],G,err,rho)


class FastCostPlanner(CostPlanner,FastPolicyKernel):
    """One-hop demand planner with residual-checked inverse updates."""

    def __init__(self,*args,batch_transport=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.batch_transport = batch_transport
        self._lookahead = None
        self._batch = None

    def _discard_batch(self):
        if self._batch is not None:
            self.ledger["batch_unused"] += len(self._batch[0])
            self._batch = None

    def select_solved(self,*args,**kwargs):
        try:
            return super().select_solved(*args,**kwargs)
        finally:
            self._discard_batch()
            self._lookahead = None

    def _materialize(self,e,t):
        if not self.batch_transport or self._lookahead is None:
            return super()._materialize(e,t)
        if self._batch is None or e not in self._batch[0]:
            self._discard_batch()
            ids,order,cap,pos,threshold = self._lookahead
            ahead = order[pos:pos+32]
            eligible = ids[ahead[cap[ahead]>threshold]]
            # The caller has already opened the gate for e. Extra rows incur
            # only local contractions. The current version records each Gram
            # only if and when its own gate opens on an actual request.
            edges = [int(k) for k in eligible if k in t.rows]
            if len(edges)<2:
                return super()._materialize(e,t)
            self._make_batch(t,edges)
        offsets,centers,radii = self._batch
        idx = offsets.pop(e)
        r = 1+int(self.flip[e])
        self.rows[e] = centers[idx,:r,:r].copy(),float(radii[idx])
        self.stale.add(e)
        self.ledger["batch_consumed"] += 1

    def _make_batch(self,t,edges):
        _,tf,bf,et,eb,beta,tn,bn = self._transport_trial
        Y,_,column_error = t.response
        ids = np.asarray(edges)
        flip = self.flip[ids]
        rank = 1+int(np.any(flip))
        grams = np.zeros((len(ids),rank,rank))
        radius = np.empty(len(ids))
        for k,e in enumerate(edges):
            g,radius[k] = t.rows[e]
            r = g.shape[0]
            grams[k,:r,:r] = g
        x,y = Y[self.i[ids]],Y[self.j[ids]]
        C = (x-y)[:,None,:]
        if rank==2:
            C = np.stack((x-y,(x+y)*flip[:,None]),axis=1)
        ec = (float(column_error)+FL*(abs(x)+abs(y)).sum(axis=1)+TINY)*(1+FL)
        axes = (1,2)
        cn,gn = abs(C).sum(axis=axes),abs(grams).sum(axis=axes)
        P = C@tf
        pn = abs(P).sum(axis=axes)
        old = grams-P@C.swapaxes(1,2)
        old_rad = (radius+beta*ec*(2*cn+ec)+cn*cn*et
                   +FL*(gn+cn*cn*tn)+TINY*(1+cn)*(1+tn))*(1+FL)
        pb = P@bf
        sr = (abs(C-pb).sum(axis=axes)+ec+eb*pn
              +FL*(cn+pn*bn)+TINY*(1+bn)*(1+pn))*(1+FL)
        cp = C@P.swapaxes(1,2)
        new = grams-(cp+cp.swapaxes(1,2)-pb@P.swapaxes(1,2))
        new = .5*(new+new.swapaxes(1,2))
        new_rad = (radius+2*ec*pn+eb*pn*pn+beta*sr*sr
                   +FL*(gn+2*cn*pn+pn*pn*bn)+TINY*(1+pn)**2*(1+bn))*(1+FL)
        use = new_rad<old_rad
        centers = np.where(use[:,None,None],new,old)
        radii = np.nextafter(np.where(use,new_rad,old_rad),np.inf)
        if not np.isfinite(centers).all() or not np.isfinite(radii).all():
            raise CertificateError("nonfinite batched transport")
        self._batch = ({e:k for k,e in enumerate(edges)},centers,radii)
        self.ledger["batch_calls"] += 1
        self.ledger["batch_computed"] += len(ids)
        self.ledger["directional_blocks"] += len(ids)
        self.ledger["directional_blocks_tighter"] += int(use.sum())

    def _pivot(self,e,G,error):
        # The inverse trial is certified by I-BT; no exact-arithmetic inversion
        # is needed by the one-row transport kernel.
        w = self.w[e]
        invw = 1./w
        B = (G+G.T)*.5
        B[0,0] += invw
        if self.flip[e]:
            B[1,1] -= invw
        if self.flip[e]:
            a,b,d = B[0,0],B[0,1],B[1,1]
            det = a*d-b*b
            if not det or not np.isfinite(det):
                return super()._pivot(e,G,error)
            T = np.array([[d,-b],[-b,a]])/det
        else:
            T = 1./B
        bn,tn = float(abs(B).sum()),float(abs(T).sum())
        beta = (w*(1+2*w)+TINY)*(1+FL)
        eb = (float(error)+FL*(float(abs(G).sum())+2*invw)+TINY)*(1+FL)
        norm_res = float(abs(np.eye(len(B))-B@T).sum())
        et = (beta*(norm_res+eb*tn+FL*(2+bn*tn)+TINY*(1+bn)*(1+tn)) + TINY)*(1+FL)
        if not math.isfinite(et):
            return super()._pivot(e,G,error)
        self._transport_trial = (T,T,B,et,eb,beta,tn,bn)
        return T,beta,et

    def commit(self,e):
        if not 0 <= e < len(self.edits) or self.active[e]:
            raise ValueError("invalid commitment")
        self._state_response = (self._winner_columns.get(e)
                                if self._winner_operator is self.A else None)
        self._state_edge = e
        super().commit(e)
        if self.factor is not None and self.pending is not None:
            # Recovery uses child triangular solves and parent residuals.
            self.pending.replay_work = 4*self.pending.rank*self.A.nnz

    def solve_committed_state(self,rhs,parent_z):
        response = getattr(self,"_state_response",None)
        if response is not None:
            e = self._state_edge
            Y,G,_ = response
            j = np.diag([1.,-1.] if self.flip[e] else [1.])
            h = np.asarray([parent_z[self.i[e]]-parent_z[self.j[e]]] +
                           ([parent_z[self.i[e]]+parent_z[self.j[e]]] if self.flip[e] else []))
            try:
                z = parent_z-Y@np.linalg.solve(G+j/self.w[e],h)
                rho = self.residual_radius(rhs,z)
                if rho <= 1e-10*max(1.,np.linalg.norm(rhs)):
                    self.ledger["state_warm_hits"] += 1
                    return z,rho
            except (np.linalg.LinAlgError,CertificateError):
                pass
        self.ledger["state_rhs"] += 1
        return self.solve(rhs)

    def _historical(self,t):
        if self.factor is None:
            return super()._historical(t)
        from time import perf_counter_ns
        start = perf_counter_ns()
        child = self._operator()
        e,rank = t.winner,t.rank
        U = np.zeros((self.A.shape[0],rank))
        U[self.i[e]] = 1.
        U[self.j[e]] = [-1.,1.] if rank == 2 else [-1.]
        self.ledger["W"] += 1
        self.ledger["W_rhs"] += rank
        self.ledger["winner_rhs"] += rank
        try:
            X = self.factor.solve(U)
            H = X[self.i[e]]-X[self.j[e]]
            if rank == 2:
                H = np.stack([H,X[self.i[e]]+X[self.j[e]]])
            J = np.diag([1.,-1.] if rank == 2 else [1.])
            small = np.eye(rank)-self.w[e]*J@H.reshape(rank,rank)
            try:
                Y = X@np.linalg.solve(small,np.eye(rank))
            except np.linalg.LinAlgError:
                Y = X
            self._bind_operator(t.operator)
            err = 2*sum(self.residual_radius(u,y) for u,y in zip(U.T,Y.T))
            self.ledger["parent_recoveries"] += 1
            return self._response(e,Y,err)
        finally:
            self._bind_operator(child)
            self.ledger["historical_ns"] += perf_counter_ns()-start

    def memory_bytes(self):
        size = super().memory_bytes()
        response = getattr(self,"_state_response",None)
        if response is not None and (self.pending is None or self.pending.response is not response):
            size += response[0].nbytes
        return size

    def close(self):
        self._state_response = None
        super().close()
