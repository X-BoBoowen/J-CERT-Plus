"""Protocol helpers; no query or certificate algorithm is implemented here."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'source/legacy_v6/tools'
sys.dont_write_bytecode = True
sys.path.insert(0, str(TOOLS))

def protocol():
    raw = (ROOT/'protocol/PROTOCOL.json').read_bytes()
    expected = (ROOT/'protocol/PROTOCOL.sha256').read_text().split()[0]
    assert hashlib.sha256(raw).hexdigest() == expected
    return json.loads(raw)

def digest_array(a):
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()

def json_default(x):
    import numpy as np
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,(set,frozenset)): return sorted(x)
    if isinstance(x,Path): return str(x)
    raise TypeError(type(x).__name__)

def dump(path,obj):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=json_default)+'\n')
    temp.replace(path)

def opinion(W,family,seed):
    import numpy as np
    u=np.random.Generator(np.random.PCG64(seed)).uniform(-1.,1.,W.shape[0])
    v=u.copy()
    if family=='signed_diffusion3':
        d=np.maximum(np.asarray(abs(W).sum(axis=1)).ravel(),1)
        for _ in range(3): v=0.5*(v+(W@v)/d)
    elif family!='iid_uniform': raise ValueError(family)
    s=np.rint(65536*v).astype(np.int64)
    assert np.isfinite(v).all() and np.max(np.abs(s))<=65536
    return u,v,s

def case_name(graph,family,seed): return f'{graph}_{family}_{seed}'

def native():
    """Bind the unchanged legacy native kernels, already compiled before timing."""
    import ctypes as ct
    from types import SimpleNamespace
    screen=SimpleNamespace(library=ct.CDLL(str(ROOT/'diagnostic/build/screen.so')))
    lib=ct.CDLL(str(ROOT/'diagnostic/build/index.so'))
    ptr=ct.c_void_p
    lib.index_pass.argtypes=[ct.c_int32,ptr,ptr,ptr,ptr,ct.c_int64,ct.c_int64,ct.c_int64,ct.c_int32,ptr,ptr]
    lib.index_pass.restype=None
    lib.gather.argtypes=[ct.c_int32,ptr,ptr,ptr,ptr,ptr,ptr,ct.c_uint8,ptr,ptr]
    lib.gather.restype=ct.c_int32
    lib.dyadic_plan.argtypes=([ct.c_int32,ptr,ptr]+[ct.c_int64]*6+[ct.c_int32,ptr,ct.c_int64]+[ptr]*4)
    lib.dyadic_plan.restype=ct.c_int32
    lib.gather_at.argtypes=([ct.c_int32,ptr,ct.c_int64]+[ptr]*5+[ct.c_uint8,ptr,ptr])
    lib.gather_at.restype=ct.c_int32
    return screen,lib
