"""Sparse CHOLMOD update/downdate. Build instructions are in README.md."""
import ctypes as ct
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy import sparse


def _library():
    name = {"win32": "jcert_cholmod.dll", "darwin": "libjcert_cholmod.dylib"}.get(
        sys.platform, "libjcert_cholmod.so")
    path = Path(os.environ.get("JCERT_CHOLMOD_LIB", Path(__file__).with_name(name)))
    if os.name == "nt":
        # Dependent SuiteSparse and BLAS DLLs must be in this directory or PATH.
        for directory in dict.fromkeys([str(path.resolve().parent), *os.get_exec_path()]):
            if directory and os.path.isdir(directory):
                _dll_directories.append(os.add_dll_directory(os.path.abspath(directory)))
    lib = ct.CDLL(str(path.resolve()))
    vector = np.ctypeslib.ndpointer(dtype=np.float64, flags="ALIGNED")
    index = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    h = ct.c_void_p
    lib.jc_new.argtypes = [ct.c_int, index, index, vector]
    lib.jc_new.restype = h
    lib.jc_clone.argtypes = [h]
    lib.jc_clone.restype = h
    lib.jc_free.argtypes = [h]
    lib.jc_free.restype = None
    lib.jc_modify.argtypes = [h, ct.c_int, ct.c_int, ct.c_double, ct.c_double, ct.c_int]
    lib.jc_modify.restype = ct.c_int
    lib.jc_solve.argtypes = [h, ct.c_int, vector, vector]
    lib.jc_solve.restype = ct.c_int
    lib.jc_memory.argtypes = [h, ct.c_int]
    lib.jc_memory.restype = ct.c_size_t
    return lib


_dll_directories = []
_lib = None


class CholmodFactor:
    def __init__(self, A):
        global _lib
        if _lib is None:
            _lib = _library()
        self._lib = _lib
        lower = sparse.tril(A, format="csc")
        lower.sum_duplicates()
        lower.sort_indices()
        self.n = lower.shape[0]
        if lower.shape != (self.n, self.n) or max(self.n, lower.nnz) >= 2**31:
            raise ValueError("requires square CSC with 32-bit indices")
        self._h = self._lib.jc_new(self.n, lower.indptr.astype(np.int32, copy=False),
                                   lower.indices.astype(np.int32, copy=False),
                                   lower.data.astype(float, copy=False))
        if not self._h:
            raise ArithmeticError("CHOLMOD failed to factor SPD input")

    def _check(self):
        if not self._h:
            raise RuntimeError("closed factor")

    def clone(self):
        """Independent mutable numeric factor at the same operator version."""
        self._check()
        other = type(self).__new__(type(self))
        other.n = self.n
        other._lib = self._lib
        other._h = self._lib.jc_clone(self._h)
        if not other._h:
            raise MemoryError("CHOLMOD could not copy the current factor")
        return other

    def solve(self, rhs):
        self._check()
        rhs = np.asfortranarray(rhs, dtype=float)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.n:
            raise ValueError("RHS dimensions do not match the factor")
        count = 1 if rhs.ndim == 1 else rhs.shape[1]
        result = np.empty_like(rhs, order="F")
        if not self._lib.jc_solve(self._h, count, rhs, result):
            raise ArithmeticError("CHOLMOD solve failed")
        return result

    def change(self, edit):
        self._check()
        a, b = edit.i, edit.j
        if a == b or min(a, b) < 0 or max(a, b) >= self.n or not math.isfinite(edit.w) or edit.w <= 0:
            raise ValueError("invalid edge")
        x = math.sqrt(edit.w)
        # Removing the present negative edge leaves I + a signed Laplacian.
        if edit.flip and not self._lib.jc_modify(self._h, a, b, x, x, 0):
            self.close()
            raise ArithmeticError("CHOLMOD downdate failed")
        if not self._lib.jc_modify(self._h, a, b, x, -x, 1):
            self.close()
            raise ArithmeticError("CHOLMOD update failed")

    def undo(self, edit):
        """Reverse a committed edit on this private copy of its child factor."""
        self._check()
        a, b = edit.i, edit.j
        if a == b or min(a,b) < 0 or max(a,b) >= self.n or not math.isfinite(edit.w) or edit.w <= 0:
            raise ValueError("invalid edge")
        x = math.sqrt(edit.w)
        # The child contains this positive edge. Removing it leaves the
        # identity plus the signed Laplacian of the other edges, hence SPD.
        if not self._lib.jc_modify(self._h, a, b, x, -x, 0):
            self.close()
            raise ArithmeticError("CHOLMOD reverse downdate failed")
        if edit.flip and not self._lib.jc_modify(self._h, a, b, x, x, 1):
            self.close()
            raise ArithmeticError("CHOLMOD reverse update failed")

    @property
    def bytes(self):
        self._check()
        return self._lib.jc_memory(self._h, 0)

    @property
    def peak_bytes(self):
        self._check()
        return self._lib.jc_memory(self._h, 1)

    def stored_entries(self):
        """Occupied simplicial factor entries; exclude allocation slack."""
        self._check()
        count = self._lib.jc_factor_entries
        count.argtypes = [ct.c_void_p]
        count.restype = ct.c_size_t
        value = count(self._h)
        if value == 0:
            raise ArithmeticError("expected a numeric simplicial factor")
        return value

    def close(self):
        if getattr(self, "_h", None):
            self._lib.jc_free(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
