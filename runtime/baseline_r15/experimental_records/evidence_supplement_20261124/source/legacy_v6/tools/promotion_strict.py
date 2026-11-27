"""Full endpoint sandwich for the integer signed-promotion operator and catalog."""
import ctypes as ct
import math

import numpy as np

from promotion import OPINION_DEN


def ceil_divide(a, b):
    return a // b + (a % b != 0)


class StrictScreen:
    """A committed matrix version; the caller supplies its live lexical rows.

    Live additions have zero prior off-diagonal and live flips have +scale.
    The signed unit catalog and single-use commits establish these invariants.
    """
    def __init__(self, A, s_int, x, proof_unit, operator_scale, pool, *, library, certified_rows=None):
        self.pool = pool
        self.n, self.m = len(x), len(pool[0])
        self.scale = int(operator_scale)
        if (not 0 < self.n < 2**31 or not 0 <= self.m < 2**31 or
                not 1 <= self.scale <= 2**20 or A.shape != (self.n,self.n)):
            raise ValueError("unsupported strict screen dimensions or scale")
        for array, dtype in zip(pool, (np.int32,np.int32,np.bool_), strict=True):
            if array.shape != (self.m,) or array.dtype!=dtype or not array.flags.c_contiguous:
                raise TypeError("expected contiguous typed catalog arrays")
        if (x.dtype != np.int64 or A.dtype!=np.int64 or not A.has_canonical_format
                or not np.issubdtype(s_int.dtype, np.signedinteger)
                or s_int.shape != x.shape or x.ndim != 1
                or proof_unit % OPINION_DEN or not 0 < proof_unit < 2**62):
            raise ValueError("expected an integer CSR operator and dyadic state")
        if int(s_int.min()) < -OPINION_DEN or int(s_int.max()) > OPINION_DEN:
            raise ValueError("opinion outside signed unit box")
        max_x = max(abs(int(x.min())), abs(int(x.max())))
        if certified_rows is None:
            diag = A.diagonal()
            if (int(A.data.min()) <= -2**25 or int(A.data.max()) >= 2**25):
                raise ValueError("principal entries outside exact significand range")
            norm = np.asarray(abs(A).sum(axis=1)).ravel()
            off = norm-diag
            if np.any(off > diag-self.scale):
                raise ValueError("signed diagonal dominance required")
            if int(norm.max())*max_x + proof_unit >= 2**63:
                raise OverflowError("integer residual out of range")
            residual = abs(A @ x - s_int.astype(np.int64)*(proof_unit//OPINION_DEN))
        else:
            # Internal, cursor-local products from certified_state. Its int64
            # overflow check covers the SpMV producing residual. bind_root
            # established a scaled signed Laplacian: the single-use legal
            # edits preserve offsum_i = Aii-scale on each committed version.
            matrix, source, state, unit, diag, residual = certified_rows
            if (matrix is not A or source is not s_int or state is not x
                    or unit!=proof_unit or diag.dtype!=np.int64
                    or residual.dtype!=np.int64 or diag.shape!=x.shape
                    or residual.shape!=x.shape or int(residual.min())<0):
                raise ValueError("certified rows belong to another state or version")
            off = diag-self.scale
        if int(diag.min()) < self.scale or int(diag.max()) >= 2**25:
            raise ValueError("principal entries outside exact significand range")
        # Exact residual eliminates the floating arithmetic allowance in (24).
        # One row of A*error=res and the global infinity contraction give r_i.
        global_r = ceil_divide(int(residual.max()), self.scale)
        if int(off.max())*global_r+int(residual.max()) >= 2**63:
            radius = np.array([ceil_divide(int(q)+int(h)*global_r,int(d))
                               for q,h,d in zip(residual,off,diag)],dtype=np.int64)
        else:
            radius = ceil_divide(residual+off*global_r,diag)
        if max_x + int(radius.max()) >= 2**50:
            raise OverflowError("state enclosure outside exact significand range")
        self.x, self.radius, self.diag = (np.ascontiguousarray(v,dtype=np.float64)
                                          for v in (x,radius,diag))
        self.native = library.endpoint_pass
        self.native.argtypes = ([ct.c_int32]*4+[ct.c_void_p]*2+[ct.c_int32]
                                +[ct.c_void_p]*6+[ct.c_double]*2+[ct.c_void_p]*5)
        self.native.restype = ct.c_int32
        self.rows = 0
        self.lower_max = None
        self.last_lower = np.empty(0,dtype=object)
        self.buffers = None

    def screen(self, ids, best):
        ids = np.ascontiguousarray(ids,dtype=np.int32)
        return self._run(0,len(ids),ids,None,best)

    def scan(self, begin, end, best, excluded):
        if not 0<=begin<=end<=self.m:
            raise ValueError("invalid catalog interval")
        excluded = np.ascontiguousarray(excluded,dtype=np.int32)
        lo,hi = excluded.searchsorted([begin,end])
        return self._run(begin,end,None,excluded[lo:hi],best)

    def _run(self, begin, end, ids, excluded, best):
        size = end-begin
        if self.buffers is None or len(self.buffers[0]) < size:
            self.buffers = (np.empty(size,np.int32), np.empty(size), np.empty(size))
        out,high,low = self.buffers
        maxlow,visited = ct.c_double(),ct.c_int32()
        best = int(best)
        threshold = math.nextafter(float(best-1),-math.inf)
        count = self.native(begin,end,self.n,self.m,
                None if ids is None else ids.ctypes.data,
                None if excluded is None else excluded.ctypes.data,
                0 if excluded is None else len(excluded),
                *(v.ctypes.data for v in (*self.pool,self.x,self.radius,self.diag)),
                self.scale,threshold,out.ctypes.data,high.ctypes.data,low.ctypes.data,
                ct.byref(maxlow),ct.byref(visited))
        if count<0:
            raise ValueError("invalid live catalog row")
        self.rows += visited.value
        if visited.value:
            value = math.floor(maxlow.value)
            self.lower_max = value if self.lower_max is None else max(self.lower_max,value)
        upper = np.fromiter(map(math.ceil,high[:count]),dtype=object,count=count)
        mask = upper >= best
        self.last_lower = np.fromiter(map(math.floor,low[:count][mask]),dtype=object)
        return out[:count][mask],upper[mask]

