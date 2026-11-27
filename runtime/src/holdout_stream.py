"""Frozen signed-neighbor confirmation input generator; no generation on import."""
import numpy as np


def signed_neighbor_opinions(unit_root,count,seed=196613):
    if type(count) is not int or count<1:
        raise ValueError("positive integral request count required")
    W=unit_root
    if (W.format!="csr" or not W.has_canonical_format or W.shape[0]!=W.shape[1] or
        W.shape[0]<1 or (W!=W.T).nnz or np.any(W.diagonal()) or
        not np.all((W.data==1)|(W.data==-1))):
        raise ValueError("expected the canonical simple undirected unit signed root graph")
    degree=np.diff(W.indptr)
    rng=np.random.Generator(np.random.PCG64(seed))
    stream=[]
    for _ in range(count):
        u=rng.uniform(-1.,1.,W.shape[0])
        neighbor=(W@u)/np.maximum(1,degree)
        floating=.5*(u+neighbor)
        if not np.isfinite(floating).all() or np.any(abs(floating)>1.):
            raise ArithmeticError("opinion range")
        codes=np.rint(65536*floating).astype(np.int64)
        if np.any(abs(codes)>65536):
            raise ArithmeticError("opinion quantization range")
        stream.append(codes)
    return stream
