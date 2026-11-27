"""Certified global inverse deltas, specialized from LINVIEW's factored update.

Each block is an n-by-r pair L=-Y T, H=Y; its contribution to the inverse is
L H.T. Only requested local entries of the base inverse are cached. Candidate
queries project previously unseen blocks at their two endpoints. Full winner
actions apply the entire global history to a base action. With --registered,
the declared view also contains every catalog Gram. A commit projects its new
block onto all these rows before the next choice. Candidate scores use the
current table and all eligible gains are evaluated at every decision. Each
registered CLI trajectory includes its own factorization and table population.
"""
import argparse
import csv
import json
import os
from time import perf_counter

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy import sparse
from fast_policy_kernel import FL, TINY, score_interval
from policy_kernel import EPS, CertificateError, PolicyKernel
from planned_matrix import OUTPUT, matrix_inputs


def project(matrix, i, j, flip):
    a, b = matrix[i], matrix[j]
    return np.stack((a-b, a+b)) if flip else (a-b)[None, :]


def residual(A, absA, gamma, rhs, x):
    # Exact legal signed systems have row dominance margin one, so
    # ||A^-1||_infinity <= 1. The componentwise residual bound also bounds
    # every exact endpoint error, which is all the screen/score needs.
    cap = abs(rhs-A@x) + 4*gamma*(abs(rhs)+absA@abs(x)) + 1e-300
    return float(np.nextafter(np.max(cap), np.inf))


class GlobalDelta(PolicyKernel):
    def __init__(self, A, edits, backend, s):
        super().__init__(A, edits, backend=backend, winner_slots=1)
        self.current, self.current_abs, self.current_gamma = self.A, self.absA, self.gamma
        self.blocks = []
        self.local = {}
        self.anchor_winner_peak = 0
        self.z, _ = self.solve(s)
        self.ledger["state_rhs"] += 1

    def base_action(self, e):
        i, j, f = self.i[e], self.j[e], self.flip[e]
        rhs = np.zeros((len(self.z), 1+int(f)), order="F")
        rhs[i] = 1.
        rhs[j] = [-1., 1.] if f else -1.
        if self.factor:
            Y = self.factor.solve(rhs)
        else:
            from scipy.sparse.linalg import cg
            Y = np.empty_like(rhs)
            for k in range(rhs.shape[1]):
                Y[:, k], status = cg(self.A, rhs[:, k], M=self.prec, rtol=2e-12, atol=0.)
                if status:
                    raise ArithmeticError("base CG failed")
        self.ledger["base_action_rhs"] += rhs.shape[1]
        return rhs, Y

    def _interval(self, e, z, rho, refresh=False, capture=None):
        i, j, f = self.i[e], self.j[e], self.flip[e]
        if e in self.local:
            G, err, start = self.local[e]
        else:
            rhs, Y = self.base_action(e)
            G = project(Y, i, j, f)
            errors = [residual(self.A, self.absA, self.gamma, rhs[:,k], Y[:,k])
                      for k in range(rhs.shape[1])]
            err = (4*sum(errors)+FL*np.sum(abs(Y[[i,j]]))+TINY)*(1+FL)
            start = 0
            if capture is not None:
                capture.append((Y, None, None))
        pending = len(self.blocks) - start
        self.ledger["local_block_projections"] += pending
        for block_index in range(start, len(self.blocks)):
            left, right, block_error = self.blocks[block_index]
            l, h = project(left, i, j, f), project(right, i, j, f)
            # Entrywise sum bounds the spectral norm, including rounding in
            # the endpoint subtractions, rank-two product and accumulation.
            magnitude = np.sum(abs(left[[i,j]]))*np.sum(abs(right[[i,j]]))
            err = (err+block_error+FL*(np.sum(abs(G))+magnitude)+TINY)*(1+FL)
            G = G + l@h.T
        self.local[e] = G, err, len(self.blocks)
        self.ledger["refinements"] += 1
        return score_interval(float(z[i]), float(z[j]), self.w[e], f, G, err, rho)

    def choose(self, s):
        rho = residual(self.current, self.current_abs, self.current_gamma, s, self.z)
        return self.select_solved(s, self.z, rho, delta=1e-9)[0]

    def commit(self, e):
        if not 0 <= e < len(self.edits) or self.active[e]:
            raise ValueError("invalid commitment")
        i, j, f, w = self.i[e], self.j[e], self.flip[e], self.w[e]
        if e in self._winner_columns:
            right = self._winner_columns[e][0]
            self.anchor_winner_peak = max(self.anchor_winner_peak,right.nbytes)
            rhs = np.zeros_like(right)
            rhs[i] = 1.
            rhs[j] = [-1., 1.] if f else -1.
        else:
            rhs, right = self.base_action(e)
        self.clear_winner_columns()
        self.ledger["winner_block_applications"] += len(self.blocks)
        for L, H, _ in self.blocks:
            right += L@project(H,i,j,f).T
        G = project(right,i,j,f)
        B = (G+G.T)*.5 + np.diag([1/w,-1/w] if f else [1/w])
        T = np.linalg.inv(B)
        left = -(right@T)
        # Certify the inverse difference against the actual current operator,
        # independently of every previous block's error. Thus block radii add.
        er = (1+FL)*sum(residual(self.current,self.current_abs,self.current_gamma,
                                  rhs[:,k],right[:,k]) for k in range(right.shape[1]))
        yr = (1+FL)*np.max(np.sum(abs(right),axis=1))
        tn = (1+FL)*np.sum(abs(T))
        beta = w*(1+2*w)*(1+FL)
        eb = 4*er+FL*(np.sum(abs(right[[i,j]]))+2/w)+TINY
        et = beta*(np.sum(abs(np.eye(len(T))-B@T))+eb*tn
                   +FL*(2+np.sum(abs(B))*tn)+TINY)*(1+FL)
        # Per-entry outer product error <= beta*er*(2yr+er)+yr^2*et.
        # A signed two-endpoint projection sums four entries; rank <= 2.
        error = 8*(beta*er*(2*yr+er)+yr*yr*et+FL*yr*yr*tn+TINY)*(1+FL)
        self.z += left@project(self.z[:,None],i,j,f).ravel()
        self.blocks.append((left,right,error))
        self.ledger["committed_global_rank"] += right.shape[1]
        self.local.pop(e, None)
        self.active[e] = True
        v = (0.,-2*w,-2*w,0.) if f else (w,-w,-w,w)
        self.current += sparse.csr_matrix((v,([i,i,j,j],[i,j,i,j])),shape=self.A.shape)
        self.current_abs = abs(self.current)
        self.current_gamma = (2*np.diff(self.current.indptr)+4)*EPS
        if not f:
            self.diag[[i,j]] += w


class RegisteredDelta(GlobalDelta):
    """Maintain every declared local Gram after every committed inverse delta."""
    def __init__(self, A, edits, backend, s, prepared=None):
        super().__init__(A, edits, backend, s)
        self.update_seconds = 0.
        start = perf_counter()
        if prepared is None:
            self.grams, self.errors = self.populate()
            self.population_seconds = perf_counter()-start
        else:
            grams, errors, self.population_seconds, rhs_count = prepared
            self.grams, self.errors = grams.copy(), errors.copy()
            self.ledger["population_rhs"] = rhs_count
        self.ledger["registered_rows_populated"] = len(edits)

    def populate(self):
        m = len(self.edits)
        grams = np.zeros((m,2,2))
        errors = np.zeros(m)
        vertices, inv = np.unique(np.r_[self.i,self.j],return_inverse=True)
        if len(vertices) > m+sum(self.flip) or self.factor is None:
            for e,(i,j,f) in enumerate(zip(self.i,self.j,self.flip)):
                rhs,Y = self.base_action(e)
                d = Y.shape[1]
                grams[e,:d,:d] = project(Y,i,j,f)
                err = sum(residual(self.A,self.absA,self.gamma,rhs[:,k],Y[:,k])
                          for k in range(d))
                errors[e] = (4*err+FL*np.sum(abs(Y[[i,j]]))+TINY)*(1+FL)
                self.ledger["population_rhs"] += d
            return grams,errors
        # Only the diagonals and one symmetric off-diagonal per row are needed.
        # Solve distinct endpoint unit columns in bounded-size blocks.
        vi,vj = inv[:m],inv[m:]
        diag = np.empty(len(vertices))
        radii = np.empty_like(diag)
        cross = np.empty(m)
        batch = min(16,max(1,2**18//len(self.z)))
        for first in range(0,len(vertices),batch):
            v = vertices[first:first+batch]
            col = np.arange(len(v))
            rhs = np.zeros((len(self.z),len(v)),order="F")
            rhs[v,col] = 1.
            y = self.factor.solve(rhs)
            diag[first:first+len(v)] = y[v,col]
            hit = (vi>=first)&(vi<first+len(v))
            cross[hit] = y[self.j[hit],vi[hit]-first]
            radius = abs(rhs-self.A@y)+4*self.gamma[:,None]*(abs(rhs)+self.absA@abs(y))+1e-300
            radii[first:first+len(v)] = np.nextafter(radius.max(axis=0),np.inf)
        di,dj = diag[vi],diag[vj]
        grams[:,0,0] = di+dj-2*cross
        grams[:,1,1] = di+dj+2*cross
        grams[:,0,1] = grams[:,1,0] = di-dj
        errors[:] = (8*np.maximum(radii[vi],radii[vj])+FL*(abs(di)+abs(dj)+2*abs(cross))+TINY)*(1+FL)
        self.ledger["population_rhs"] = len(vertices)
        return grams,errors

    def _interval(self,e,z,rho,refresh=False,capture=None):
        d = 1+int(self.flip[e])
        self.ledger["refinements"] += 1
        return score_interval(float(z[self.i[e]]),float(z[self.j[e]]),self.w[e],
                              self.flip[e],self.grams[e,:d,:d],self.errors[e],rho)

    def score_all(self,ids,rho):
        """Directional residual certificate from score_interval, lane by lane."""
        f,w = self.flip[ids],self.w[ids]
        x,y = self.z[self.i[ids]],self.z[self.j[ids]]
        G = self.grams[ids]
        iw = 1/w
        a = G[:,0,0]+iw
        b = np.where(f,.5*(G[:,0,1]+G[:,1,0]),0.)
        d = np.where(f,G[:,1,1]-iw,0.)
        h,k = x-y,np.where(f,x+y,0.)
        det = np.where(f,a*d-b*b,a)
        if np.any(det==0) or not np.isfinite(det).all():
            raise CertificateError("unusable registered pivot")
        # The proposed response p,q may be arbitrary. Certify the objective
        # with 2 h'p - p'Bp and remainder r'B^-1 r for r=h-Bp. beta bounds
        # the exact inverse norm for either legal edit, including signed flips.
        t = np.where(f,d,1.)/det
        u,v = -b/det,np.where(f,a,0.)/det
        p,q = t*h+u*k,u*h+v*k
        z1,z2 = a*p+b*q,b*p+d*q
        gb = abs(G[:,0,0])+iw+np.where(f,abs(G[:,1,1])+2*iw+abs(G[:,0,1])+abs(G[:,1,0]),0.)
        eb = (self.errors[ids]+FL*gb+TINY)*(1+FL)
        eh = (2*rho+FL*(abs(x)+abs(y))+TINY)*(1+FL)
        bn,hn,pn = abs(a)+2*abs(b)+abs(d),abs(h)+abs(k),abs(p)+abs(q)
        r = abs(h-z1)+abs(k-z2)
        sr = (r+eh+eb*pn+FL*(hn+bn*pn)+TINY*(1+bn+pn))*(1+FL)
        g = 2*(h*p+k*q)-(p*z1+q*z2)
        eta = FL*(2*hn*pn+bn*pn*pn)+TINY*(1+pn)**2*(1+bn)
        beta = w*(1+2*w)*(1+FL)
        rd = (2*eh*pn+eb*pn*pn+beta*sr*sr+eta)*(1+FL)
        lo,hi = np.nextafter(g-rd,-np.inf),np.nextafter(g+rd,np.inf)
        if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
            raise CertificateError("nonfinite registered score enclosure")
        return lo,hi

    def choose(self,s):
        ids = np.flatnonzero(~self.active)
        rho = residual(self.current,self.current_abs,self.current_gamma,s,self.z)
        lo,hi = self.score_all(ids,rho)
        self.ledger["registered_scores"] += len(ids)
        if not len(ids) or not np.any(s):
            self.last_upper = 0.
            return None
        for attempt in range(2):
            k = int(np.argmax(lo))
            positive = lo[k]>0.
            uncertain = hi>(np.nextafter(lo[k]+1e-9,-np.inf) if positive else 0.)
            if positive:
                uncertain[k] = False
            if not np.any(uncertain):
                self.last_upper = float(hi[k]) if positive else 0.
                return int(ids[k]) if positive else None
            if attempt == 0:
                uncertain[k] = True
                for j in np.flatnonzero(uncertain):
                    lo[j],hi[j] = self._interval(ids[j],self.z,rho)
        raise CertificateError("overlapping registered candidate or positivity intervals")

    def commit(self,e):
        super().commit(e)
        start = perf_counter()
        left,right,block_error = self.blocks[-1]
        li,lj,ri,rj = left[self.i],left[self.j],right[self.i],right[self.j]
        magnitude = (abs(li)+abs(lj)).sum(axis=1)*(abs(ri)+abs(rj)).sum(axis=1)
        self.errors = (self.errors+block_error+FL*(abs(self.grams).sum(axis=(1,2))+magnitude)+TINY)*(1+FL)
        ldiff,rdiff = li-lj,ri-rj
        self.grams[:,0,0] += np.einsum("ij,ij->i",ldiff,rdiff)
        f = self.flip
        lsum,rsum = li[f]+lj[f],ri[f]+rj[f]
        self.grams[f,0,1] += np.einsum("ij,ij->i",ldiff[f],rsum)
        self.grams[f,1,0] += np.einsum("ij,ij->i",lsum,rdiff[f])
        self.grams[f,1,1] += np.einsum("ij,ij->i",lsum,rsum)
        self.ledger["registered_rows_updated"] += len(self.edits)
        self.update_seconds += perf_counter()-start


class RegisteredLookup(RegisteredDelta):
    """Supplemental materialized table with an endpoint-screened reader."""
    choose = GlobalDelta.choose


def trial(A, edits, s, budgets=(20,40), backend="cholmod", registered=False,
          prepared=None, lookup=False):
    start = perf_counter()
    registered = registered or lookup
    table_type = RegisteredLookup if lookup else RegisteredDelta
    kernel = (table_type(A, edits, backend, s, prepared) if registered else
              GlobalDelta(A, edits, backend, s))
    charge = kernel.population_seconds if prepared is not None else 0.
    setup = charge+perf_counter()-start
    seq, result = [], []
    stopped = False
    try:
        for step in range(1,max(budgets)+1):
            if not stopped:
                e = kernel.choose(s)
                if e is None:
                    stopped = True
                else:
                    kernel.commit(e)
                    seq.append(int(e))
            if step in budgets:
                seconds = charge+perf_counter()-start
                candidate = sum(G.nbytes+16 for G,_,_ in kernel.local.values())
                if registered:
                    candidate = kernel.grams.nbytes+kernel.errors.nbytes
                history = sum(L.nbytes+H.nbytes for L,H,_ in kernel.blocks)
                extra = sum(v.nbytes for m in (kernel.current,kernel.current_abs)
                            for v in (m.indices,m.data,m.indptr)) if kernel.blocks else 0
                total = (kernel.memory_bytes()+candidate+history+8*len(kernel.blocks)
                         +kernel.z.nbytes+extra+(kernel.current_gamma.nbytes if kernel.blocks else 0))
                result.append(dict(b=step,seconds=seconds,setup_seconds=setup,commits=len(seq),
                                   sequence=json.dumps(seq),candidate_bytes=candidate,
                                   history_bytes=history,winner_bytes=kernel.winner_column_bytes(),
                                   peak_winner_bytes=kernel.anchor_winner_peak,memory_bytes=total,
                                   factor_peak_bytes=kernel.factor.peak_bytes if kernel.factor else 0,
                                   counters=json.dumps(dict(kernel.ledger),sort_keys=True)))
                if registered:
                    result[-1].update(population_seconds=kernel.population_seconds,
                                      population_reused=prepared is not None,
                                      registered_update_seconds=kernel.update_seconds,
                                      online_seconds=seconds-setup)
    finally:
        kernel.close()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph",required=True)
    p.add_argument("--m",type=int,required=True)
    p.add_argument("--reps",type=int,default=10)
    p.add_argument("--catalog", choices=("clustered", "persistent"), default="clustered")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--registered",action="store_true")
    mode.add_argument("--lookup",action="store_true",help="supplemental screened reader with fixed population cost")
    args = p.parse_args()
    A,s,edits = matrix_inputs(args.graph, args.m, clustered=args.catalog=="clustered",
                              persistent=args.catalog=="persistent")
    prefix = "linview_registered" if args.registered else "linview"
    if args.lookup:
        prefix = "linview_registered_lookup"
    path = OUTPUT/f"{prefix}_{args.catalog}_{args.graph}_{args.m}.csv"
    with path.open() if path.exists() else open(os.devnull) as f:
        done = {int(row["rep"]) for row in csv.DictReader(f) if row["b"]=="40"}
    prepared = None
    if args.lookup and any(rep not in done for rep in range(args.reps)):
        base = RegisteredLookup(A,edits,"cholmod",s)
        prepared = (base.grams,base.errors,base.population_seconds,base.ledger["population_rhs"])
        base.close()
        del base
    for rep in range(args.reps):
        if rep in done:
            continue
        rows = [dict(graph=args.graph,n=A.shape[0],M=args.m,rep=rep,backend="cholmod",
                     mode=prefix,**row) for row in trial(A,edits,s,registered=args.registered,
                                                         lookup=args.lookup,prepared=prepared)]
        with path.open("a",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=rows[0])
            if not stream.tell():
                writer.writeheader()
            writer.writerows(rows)
        print(args.graph,args.m,rep,round(rows[-1]["seconds"],3),flush=True)


if __name__ == "__main__":
    main()
