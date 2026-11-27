"""Endpoint-addressed access to the unchanged weighted numerical primitives.

The pool/d objects are scalar or bounded-batch adapters, never catalog arrays.
Only the strict screen changes: its version checks are cached and a standard
current-nonedge resistance bound is shared by D, S and H.
"""
import numpy as np

from .a2_catalog import CandidateBatch
from .weighted_adapter import Kernel, Strict


class EndpointColumn:
    def __init__(self,catalog,field):
        self.catalog,self.field=catalog,field

    def __getitem__(self,key):
        if isinstance(key,slice):
            raise TypeError('virtual endpoint columns do not materialize the catalog')
        if np.isscalar(key):
            return self.catalog.scalar(int(key))[self.field]
        batch=self.catalog.batch(np.asarray(key,dtype=np.int64))
        return (batch.i,batch.j,batch.flip,batch.weights)[self.field]

    def __array__(self,dtype=None,copy=None):
        raise TypeError('virtual endpoint columns have no full-catalog array')

    def __iter__(self):
        raise TypeError('use bounded candidate batches')


class EndpointKernel(Kernel):
    salt='weighted-normalized:scale16:proof48:screen20:allnonedges-v1'

    def __init__(self,catalog):
        self.validate(catalog.W)
        self.catalog=catalog
        self.pool=tuple(EndpointColumn(catalog,k) for k in range(3))
        self.d=EndpointColumn(catalog,3)
        self.max_weight=max(1,int(catalog.flip_weights.max(initial=1)))

    def strict(self,A,s,x,certified_rows):
        return CachedStrict(self,A,s,x,certified_rows)

    def root_caps(self,x,r):
        raise TypeError('endpoint kernel does not create all candidate caps')

    def prepare_weight_groups(self):
        raise TypeError('endpoint kernel does not partition an implicit catalog')


class CachedStrict(Strict):
    """Current-version endpoint envelope with no singleton global scans.

State/residual/diagonal binding and endpoint radii are the existing Strict
constructor. Coarse arithmetic range is checked once at that same boundary.
Input batches are trusted products of the bound catalog access layer.
"""
    def __init__(self,kernel,A,s,x,rows):
        super().__init__(kernel,A,s,x,rows)
        self.version_validation_scans=1
        if 4*(int(np.abs(self.xc).max(initial=0))+self.rc)**2*kernel.max_weight>=2**62:
            raise OverflowError('weighted tangent range')

    def screen(self,ids,best):
        batch=(ids if isinstance(ids,CandidateBatch) else
               self.kernel.catalog.batch(np.asarray(ids,dtype=np.int64)))
        self.rows+=len(batch.ids)
        a,b=self.xc[batch.i],self.xc[batch.j]
        rc=self.rc
        caps=batch.weights*np.where(batch.flip,
            -4*a*b+4*rc*(np.abs(a)+np.abs(b)+rc),
            (np.abs(a-b)+2*rc)**2)
        threshold=-(-int(best)//(self.factor**2))
        keep=np.flatnonzero(caps>=threshold)
        survivors=[];uppers=[]
        needed=max(1,int(best))  # A nonpositive upper cannot produce an edit.
        for k in keep:
            e,i,j,d=map(int,(batch.ids[k],batch.i[k],batch.j[k],batch.weights[k]))
            flip=bool(batch.flip[k])
            a,b,ra,rb=map(int,(self.x[i],self.x[j],self.radius[i],self.radius[j]))
            di,dj=int(self.diag[i]),int(self.diag[j])
            if flip:
                broad=d*(-4*a*b+4*(abs(a)*rb+abs(b)*ra+ra*rb))
                pa=2*d*max(0,abs(b)-rb)
                pb=2*d*max(0,abs(a)-ra)
                shift=d
            else:
                diff,err=abs(a-b),ra+rb
                broad=d*(diff+err)**2
                pa=pb=d*max(0,diff-err)
                shift=2*d
            high=broad-pa**2//(di+shift)-pb**2//(dj+shift)
            if not flip:
                # This candidate is a current nonedge: A[i,j]=0. Restricting
                # b^T A^-1 b's variational maximum to these two coordinates
                # gives R>=1/di+1/dj. Positive integer ceiling is outward.
                numerator=broad*di*dj
                denominator=di*dj+d*(di+dj)
                high=min(high,(numerator+denominator-1)//denominator)
            if high>=needed:
                survivors.append(e);uppers.append(high)
        return np.asarray(survivors,dtype=np.int64),np.asarray(uppers,dtype=object)
