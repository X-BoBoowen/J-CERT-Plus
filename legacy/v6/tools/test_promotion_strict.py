from fractions import Fraction as F
from itertools import product

from tempfile import TemporaryDirectory
import random

import numpy as np
from scipy.sparse import csr_matrix

from promotion_strict import StrictScreen
from promotion import OPINION_DEN
from signed_promotion import build_screen


def oracle(u,v,a,b,flip,s):
    """Exact evaluations of (10) and (11) at one state."""
    if not flip:
        d=(u-v)**2
        return d*F(s,s+2),d*F(a*b,a*b+a+b)
    energy=-4*s*u*v
    qtop=4*s*s*(b*v*v+2*s*u*v+a*u*u)/F(a*b-s*s)
    qbottom=2*s*((u+v)**2+(u-v)**2/F(3))
    return energy-qbottom,energy-qtop


def selftest(library):
    rng=random.Random(619)
    count=0
    for scale in (1,16,256):
        # Two disconnected pairs, one prospective addition, one unit flip.
        pool=(np.array([0,2],np.int32),np.array([1,3],np.int32),np.array([0,1],bool))
        A=csr_matrix(np.diag([scale,scale,2*scale,2*scale]).astype(np.int64))
        A=A+csr_matrix(([scale,scale],([2,3],[3,2])),shape=A.shape)
        for it in range(84):
            src=np.array([rng.randint(-OPINION_DEN,OPINION_DEN) for _ in range(4)],np.int64)
            unit=2**48
            rhs=[int(k)*unit//OPINION_DEN for k in src]
            truth=[F(rhs[0],scale),F(rhs[1],scale),F(2*rhs[2]-rhs[3],3*scale),
                   F(2*rhs[3]-rhs[2],3*scale)]
            x=np.array([round(q)+rng.randint(-7,7) for q in truth],dtype=np.int64)
            st=StrictScreen(A,src,x,unit,scale,pool,library=library)
            residual=abs(A@x-src*(unit//OPINION_DEN))
            cached=(A,src,x,unit,A.diagonal(),residual)
            accelerated=StrictScreen(A,src,x,unit,scale,pool,library=library,certified_rows=cached)
            R=max(-(-int(q)//scale) for q in residual)
            for i,(d,q) in enumerate(zip(A.diagonal(),residual)):
                exact=F(int(q)+(int(d)-scale)*R,int(d))
                assert int(accelerated.radius[i])==int(st.radius[i])==-(-exact.numerator//exact.denominator)
            if not it:
                try:
                    StrictScreen(A,src,x.copy(),unit,scale,pool,library=library,certified_rows=cached)
                except ValueError:
                    pass
                else:
                    raise AssertionError("accepted residual of another state")
            if it in (0,1):
                # Exercise maximum significands, unequal diagonals, zero and
                # cancellation independently of the residual construction.
                st.x[:]=[2**49,2**49-it,-2**49,(-1)**it*2**49]
                st.radius[:]=[0,it,0,it]
                st.diag[:]=[2**24,scale,2*scale,2**24]
            else:
                assert all(abs(q-int(v))<=int(r) for q,v,r in zip(truth,x,st.radius))
            if it>=42:
                # Principal entries span the supported exact determinant range.
                st.diag[:]=[rng.randint(scale,2**25-1) for _ in range(2)]+[
                    rng.randint(2*scale,2**25-1) for _ in range(2)]
                st.radius[:]=[rng.randrange(0,2**46) for _ in range(4)]
            ids,upper=st.scan(0,2,-2**150,[])
            lower=st.last_lower.copy()
            assert ids.tolist()==[0,1] and st.rows==2
            for e,(i,j,flip) in enumerate(zip(*pool)):
                for h,k in product((-1,F(-1,3),0,F(1,3),1),repeat=2):
                    u=int(st.x[i])+h*int(st.radius[i])
                    v=int(st.x[j])+k*int(st.radius[j])
                    lo,hi=oracle(u,v,int(st.diag[i]),int(st.diag[j]),flip,scale)
                    assert lower[e]<=lo<=hi<=upper[e]
                    count+=1
                for offset in (-1,0,1):
                    target=upper[e]+offset
                    kept,caps=st.screen(np.array([e],np.int32),target)
                    assert kept.tolist()==([e] if offset<=0 else [])
                    assert caps.tolist()==([upper[e]] if offset<=0 else [])
            ids,caps=st.scan(0,2,-2**150,[0])
            assert ids.tolist()==[1] and caps[0]==upper[1]
    print(f"{count} exact-rational endpoint enclosures and grid-boundary comparisons passed")


if __name__=="__main__":
    # Windows keeps a loaded DLL locked until the process terminates.
    import os
    import _ctypes
    with TemporaryDirectory() as temp:
        lib=build_screen(temp).library
        try:
            selftest(lib)
        finally:
            if os.name=="nt":
                _ctypes.FreeLibrary(lib._handle)



