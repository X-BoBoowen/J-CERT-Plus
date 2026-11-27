"""Natural-weight research kernels. A_int=16*A, z_int=A_int^-1*s.

Use unscaled integer endpoint columns U, and a separate integer edit
coefficient d=16*w. No square roots or nonintegral operator truncation.
"""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction as Q
from math import ceil, floor
from time import perf_counter
import numpy as np
from scipy import sparse

from .common import Ledger, UNIT, SCALE
from long_promotion import SCREEN
from promotion import VerifiedGramViews, transpose
from promotion_directional import directional_block


def scaled(interval,c):
    return c*interval[0],c*interval[1]


def correction(g,h,weight):
    """Enclose H[(J/d)+G]^-1 H'. The directional residual uses its original norm bound.

    Let C=1+2*d/16. The signed Woodbury identity gives
    ||(J+dG)^-1|| <= 1+d||U||²/16 <= C. Thus
    ||[C(J+dG)]^-1|| <= 1. Feed that pivot to the legacy directional
    primitive (which uses the looser bound 3), then multiply by d*C.
    """
    c=1+Q(2*weight,SCALE)
    transformed=[]
    for a,row in enumerate(g):
        line=[]
        for b,iv in enumerate(row):
            lo,hi=scaled(iv,c*weight)
            if a==b:
                shift=(c-1)*(1 if a==0 else -1)
                lo,hi=lo+shift,hi+shift
            line.append((lo,hi))
        transformed.append(line)
    boxed=directional_block(transformed,h,UNIT)
    return [[scaled(el,c*weight) for el in row] for row in boxed]


class WeightedViews(VerifiedGramViews):
    def __init__(self,kernel,ledger,counts):
        super().__init__(kernel.pool,UNIT)
        self.kernel,self.ledger,self.counts=kernel,ledger,counts
        self.scored_response=self.retained=None
        self.role="candidate"

    def response(self,e,A,fac):
        if self.retained is not None and self.retained[0]==e:
            return self.retained[1]
        with self.ledger.phase("response_acquisition"):
            i,j,f=(a[e].item() for a in self.pool)
            U=np.zeros((A.shape[0],1+int(f)),np.int64)
            U[i,0],U[j,0]=1,-1
            if f:
                U[i,1]=U[j,1]=1
            Y=fac.solve(U.astype(np.float64))
            if not np.isfinite(Y).all() or abs(Y).max()>2:
                raise ArithmeticError("unresolved weighted response")
            X=np.rint(Y*UNIT).astype(np.int64)
            if 2*int(A.diagonal().max())*int(abs(X).max())+UNIT >= 2**63:
                raise OverflowError("weighted response residual range")
            residual=int(abs(A@X-UNIT*U).max())
            radius=(residual+SCALE-1)//SCALE
            result=(U,Y,X,radius)
            self.counts[self.role+"_rhs"]+=U.shape[1]
            self.counts["response_blocks"]+=1
            self.counts["response_residual_matvec_columns"]+=U.shape[1]
            self.scored_response=e,result
            return result

    def score(self,e,t,A,fac,x,r):
        if e in self.entries:
            version,g=self.entries[e]
            if version!=t:
                raise ArithmeticError("stale weighted Gram")
            self.hits+=1
        else:
            _,_,X,rr=self.response(e,A,fac)
            self.fresh+=1
            g=self.project(e,X,rr)
            self.entries[e]=t,g
        h=self.project(e,x[:,None],r)
        lo,hi=correction(g,transpose(h),int(self.kernel.d[e]))[0][0]
        return floor(lo*UNIT**2),ceil(hi*UNIT**2)

    def advance(self,e,t,A,fac,z):
        U,Y,X,r=self.response(e,A,fac)
        self.winner_blocks+=1
        d=int(self.kernel.d[e])
        g=self.project(e,X,r)
        self.entries.pop(e,None)
        for j,(version,old) in self.entries.items():
            if version!=t:
                raise ArithmeticError("stale weighted transport")
            change=correction(g,self.project(j,X,r),d)
            boxed=[[(Q(floor((a[0]-b[1])*UNIT),UNIT),Q(ceil((a[1]-b[0])*UNIT),UNIT))
                    for a,b in zip(x,y)] for x,y in zip(old,change)]
            self.entries[j]=t+1,boxed
            self.transfers+=1
        signs=[1.,-1.] if U.shape[1]==2 else [1.]
        inverse=np.linalg.inv(np.diag(signs)/d+U.T@Y)
        fac.steps.append((Y,inverse))
        self.scored_response=self.retained=None
        return z-Y@(inverse@(U.T@z))


def tangent(pool,ids,x,r,weights):
    i,j,f=(a[ids] for a in pool)
    a,b=x[i],x[j]
    radius=int(r)
    if 4*(int(abs(x).max())+radius)**2*int(weights.max(initial=1))>=2**62:
        raise OverflowError("weighted tangent range")
    return weights[ids]*np.where(f,-4*a*b+4*radius*(abs(a)+abs(b)+radius),
                                 (abs(a-b)+2*radius)**2)


class Strict:
    """Weighted two-endpoint upper certificates, with an integer coarse prefilter."""
    def __init__(self,kernel,A,s,x,rows):
        a,source,state,unit,self.diag,residual=rows
        if a is not A or source is not s or state is not x or unit!=UNIT:
            raise ValueError("weighted strict version or source binding")
        self.kernel,self.x=kernel,x
        self.rows=0
        global_r=(int(residual.max())+SCALE-1)//SCALE
        # Compute endpoint error boxes from strict row diagonal dominance.
        if int(self.diag.max())*global_r+int(residual.max())+int(self.diag.max())>=2**62:
            raise OverflowError("weighted strict radius range")
        self.radius=(residual+(self.diag-SCALE)*global_r+self.diag-1)//self.diag
        self.factor=UNIT//SCREEN
        self.xc=x//self.factor
        self.rc=(int(abs(x-self.xc*self.factor).max())+global_r+self.factor-1)//self.factor

    def screen(self,ids,best):
        ids=np.asarray(ids,dtype=np.int32)
        self.rows+=len(ids)
        caps=tangent(self.kernel.pool,ids,self.xc,self.rc,self.kernel.d)
        threshold=-(-int(best)//(self.factor**2))
        ids=ids[caps>=threshold]
        survivors=[]
        uppers=[]
        for e in ids:
            i,j,f=(a[e].item() for a in self.kernel.pool)
            a,b,ra,rb=map(int,(self.x[i],self.x[j],self.radius[i],self.radius[j]))
            d=int(self.kernel.d[e])
            if f:
                broad=d*(-4*a*b+4*(abs(a)*rb+abs(b)*ra+ra*rb))
                pa=2*d*max(0,abs(b)-rb)
                pb=2*d*max(0,abs(a)-ra)
                shift=d
            else:
                diff,err=abs(a-b),ra+rb
                broad=d*(diff+err)**2
                pa=pb=d*max(0,diff-err)
                shift=2*d
            # Child endpoint principal block <= diag(Di+shift,Dj+shift).
            # Variational correction is at least this restricted trial value.
            high=broad-pa**2//(int(self.diag[i])+shift)-pb**2//(int(self.diag[j])+shift)
            if high>=best:
                survivors.append(e)
                uppers.append(high)
        return np.asarray(survivors,dtype=np.int32),np.asarray(uppers,dtype=object)

    def scan(self,a,b,best,excluded):
        ids=np.arange(a,b,dtype=np.int32)
        if len(excluded):
            left,right=np.searchsorted(excluded,[a,b])
            ids=np.setdiff1d(ids,excluded[left:right],assume_unique=True)
        return self.screen(ids,best)


@dataclass(frozen=True, eq=False)
class WeightGroups:
    """A canonical partition bound to the immutable indexed catalog and weights."""
    pool: tuple
    weights: np.ndarray
    _member_buffers: tuple
    _layout: tuple

    @classmethod
    def build(cls,pool,weights):
        # Immutable backing also prevents callers from re-enabling NumPy writes.
        pool=tuple(np.frombuffer(a.tobytes(),dtype=a.dtype) for a in pool)
        weights=np.frombuffer(weights.tobytes(),dtype=weights.dtype)
        members=tuple((int(d),np.flatnonzero(weights==d).astype(np.int32).tobytes())
                      for d in np.unique(weights))
        layout=tuple((a.shape,a.dtype,a.strides) for a in (*pool,weights))
        return cls(pool,weights,members,layout)

    @property
    def members(self):
        # Fresh views cannot alter the shape or dtype of any saved ID vector.
        return tuple((d,np.frombuffer(buf,dtype=np.int32)) for d,buf in self._member_buffers)

    @property
    def nbytes(self):
        return sum(len(buf) for _,buf in self._member_buffers)

    def check_binding(self,pool,weights):
        if (pool is not self.pool or weights is not self.weights or
                any((a.shape,a.dtype,a.strides)!=layout for a,layout in zip((*pool,weights),self._layout))):
            raise ValueError('weight group binding mismatch')


class Index:
    """Exact key ordering per scaled edit weight, with outward cold drift shifts."""
    def __init__(self,kernel,x,r,inc,degree,*,caps=None,timers=True):
        self.inc,self.degree=inc,degree
        self.root_norm=int(abs(x).max())+int(r)
        members=kernel.prepare_weight_groups().members
        raw=kernel.root_caps(x,r) if caps is None else caps
        if raw.shape!=kernel.d.shape:
            raise ValueError('root caps do not cover the weighted catalog')
        clock=perf_counter if timers else lambda: 0.
        self.build_phases=dict(key_gather=0.,stable_sort=0.)
        self.groups=[]
        for d,ids in members:
            start=clock()
            keys=raw[ids]
            gathered=clock()
            order=np.argsort(keys,kind="stable")
            self.groups.append((d,keys[order],ids[order]))
            finished=clock()
            self.build_phases['key_gather']+=gathered-start
            self.build_phases['stable_sort']+=finished-gathered
        self.nbytes=sum(vals.nbytes+ids.nbytes for _,vals,ids in self.groups)
        self.nbytes+=sum(a.nbytes for a in (degree,inc.left,inc.right,inc.ids))

    def query(self,drift,target,excluded):
        if (drift<0).any() or int(drift.max())>4*SCREEN:
            raise ValueError("unsupported weighted index drift")
        tauchoices=[1<<p for p in range((max(1,int(drift.max()))-1).bit_length()+1)]
        costs=[]; cuts=[]
        for tau in tauchoices:
            shift=min(8*(SCREEN//SCALE)*tau,8*self.root_norm*tau+4*tau**2)
            starts=[int(np.searchsorted(keys,int(target)-d*shift)) for d,keys,_ in self.groups]
            costs.append(int(self.degree[drift>tau].sum())+sum(len(g[2])-cut for g,cut in zip(self.groups,starts)))
            cuts.append(starts)
        choice=int(np.argmin(costs))
        tau=tauchoices[choice]
        parts=[self.inc.incident(np.flatnonzero(drift>tau))]
        parts.extend(ids[cut:] for (_,_,ids),cut in zip(self.groups,cuts[choice]))
        ids=np.unique(np.concatenate(parts))
        if excluded:
            ids=np.setdiff1d(ids,np.array(sorted(excluded),np.int32),assume_unique=True)
        return ids,np.array(tauchoices),np.array(costs),tau


class Kernel:
    salt="weighted-normalized:scale16:proof48:screen20"
    @staticmethod
    def validate(W):
        if W.dtype.kind not in "iuf" or W.dtype.itemsize>8:
            raise ValueError("expected real integer or binary64-compatible numeric ratings")
        values=W.data.astype(np.float64,copy=False)
        if (W.shape[0]!=W.shape[1] or (W!=W.T).nnz or np.any(W.diagonal()) or
                not np.isfinite(values).all() or np.any(values==0) or
                np.any(abs(values)>10) or np.any(np.rint(2*values)!=2*values)):
            raise ValueError("expected the natural half-integer signed projection")

    def __init__(self,W,pool):
        self.validate(W)
        self.pool=pool
        i,j,f=pool
        self.d=np.ones(len(i),dtype=np.int64)
        if np.any(f):
            signed=np.asarray(W[i[f],j[f]],dtype=np.float64).ravel()
            if np.any(signed>=0):
                raise ValueError("invalid flip rows")
            self.d[f]=np.rint(-SCALE*signed).astype(np.int64)
        self.d.flags.writeable=False
        self._weight_groups=None

    def prepare_weight_groups(self):
        if self._weight_groups is None:
            groups=WeightGroups.build(self.pool,self.d)
            self.pool,self.d=groups.pool,groups.weights
            self._weight_groups=groups
        self._weight_groups.check_binding(self.pool,self.d)
        return self._weight_groups

    @staticmethod
    def operator(W):
        scaled=(SCALE*W.astype(np.float64,copy=False)).astype(np.int64)
        diag=SCALE+np.asarray(abs(scaled).sum(axis=1)).ravel()
        return (sparse.diags(diag,dtype=np.int64)-scaled).tocsr()

    def child(self,A,e):
        i,j,f=(a[e].item() for a in self.pool)
        d=int(self.d[e])
        vals=(0,-2*d,-2*d,0) if f else (d,-d,-d,d)
        return (A+sparse.coo_matrix((vals,([i,i,j,j],[i,j,i,j])),shape=A.shape)).tocsr()

    def views(self,ledger,counts):
        return WeightedViews(self,ledger,counts)

    def audit_views(self):
        return self.views(Ledger(False),defaultdict(int))

    def strict(self,A,s,x,certified_rows):
        return Strict(self,A,s,x,certified_rows)

    def root_caps(self,x,r):
        return tangent(self.pool,slice(None),x,r,self.d)

    def index(self,x,r,inc,degree,*,caps=None,timers=True):
        return Index(self,x,r,inc,degree,caps=caps,timers=timers)

    def index_bytes(self,index):
        return index.nbytes

    def resolve(self,A,s,evaluated,winner,best,*,tie_context=None,evidence=None):
        overlap=sorted(e for e,hi in evaluated.items() if e!=winner and hi>=best)
        if not overlap or (winner is None and all(hi<=0 for hi in evaluated.values())):
            return winner,best
        if winner is not None and best>0 and tie_context is not None:
            from .exact_ties import bind_exchange
            bound=bind_exchange(self,A,s,tie_context)
            if bound is not None:
                proofs=[];unresolved=[]
                for rival in overlap:
                    proof=bound.certify(winner,rival)
                    if proof is None:unresolved.append(int(rival))
                    else:proofs.append(proof)
                detail=dict(method='structural_node_transposition',
                    parent_version_id=tie_context.parent_version_id,
                    provisional_winner_id=int(winner),overlap_ids=list(map(int,overlap)),
                    witnesses=proofs,unresolved_ids=unresolved,
                    source_lower=int(best),source_upper=int(evaluated[winner]))
                if evidence is not None:evidence.update(detail)
                if not unresolved:
                    equal=[winner,*overlap]
                    upper=min(int(evaluated[e]) for e in equal)
                    if upper<best:
                        raise ArithmeticError('unresolved inconsistent exact-tie intervals')
                    selected=min(equal)
                    # Each original upper bounds the same true gain. Intersect
                    # their uppers, retaining the independently positive lower.
                    for e in equal:evaluated[e]=upper
                    if evidence is not None:
                        evidence.update(canonical_winner_id=int(selected),
                                        shared_lower=int(best),shared_upper=upper)
                    return selected,best
        if A.shape[0]>16:
            raise ArithmeticError("unresolved weighted overlap on large graph")
        # The existing rational fallback remains the decision basis on small
        # graphs when a single-transposition star does not resolve every rival.
        if evidence is not None:evidence.clear()
        import sympy as sy
        rhs=sy.Matrix([sy.Rational(float(v)) for v in s])
        parent=(rhs.T*sy.Matrix(A.toarray()).inv()*rhs)[0]
        ids=sorted(overlap+([winner] if winner is not None else []))
        gains=[Q(parent-(rhs.T*sy.Matrix(self.child(A,e).toarray()).inv()*rhs)[0]) for e in ids]
        k=max(range(len(ids)),key=gains.__getitem__)
        if gains[k]<=0:
            return None,0
        winner=ids[k]
        evaluated[winner]=ceil(gains[k]*UNIT**2)
        return winner,floor(gains[k]*UNIT**2)
