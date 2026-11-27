"""Two independent Jacobi-PCG solves sharing one CSR traversal per iteration."""
import ctypes as ct
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


def compile_paired_pcg(directory):
    """Compile once, outside the measurement window, like the native index."""
    return load_paired_pcg(build_paired_pcg(directory))


def build_paired_pcg(directory):
    path = Path(directory)/("paired_pcg.dll" if os.name=="nt" else "paired_pcg.so")
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        raise RuntimeError("A C compiler is required for paired PCG")
    subprocess.run([cc,"-O3","-march=native","-fPIC","-shared",
                    str(Path(__file__).with_suffix(".c")),"-o",str(path),"-lm"],
                   check=True,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    return path


def load_paired_pcg(path):
    library = ct.CDLL(str(path))
    fn = library.paired_pcg
    fn.library = library
    fn.argtypes = [ct.c_int32]+[ct.c_void_p]*8+[ct.c_double,ct.c_double,ct.c_int32]
    fn.restype = ct.c_int
    scalar = library.scalar_pcg
    scalar.argtypes = fn.argtypes
    scalar.restype = ct.c_int
    fn.scalar = scalar
    return fn


def solve_pair(fn,A,inverse_diag,rhs):
    """Return a two-column response and total independent PCG iterations."""
    if rhs.shape != (A.shape[0],2):
        raise ValueError('Paired PCG requires two columns')
    x,counts = solve_columns(fn,A,inverse_diag,rhs,2e-13,1e-15)
    return x,int(counts.sum())


def solve_columns(fn,A,inverse_diag,rhs,rtol,atol):
    n = A.shape[0]
    if rhs.ndim != 2:
        raise ValueError('Expected an n-by-rank RHS')
    rank = rhs.shape[1]
    if (rank not in (1,2) or rhs.shape!=(n,rank) or A.shape!=(n,n) or A.format!="csr"
            or A.indptr.dtype!=np.int32 or A.indices.dtype!=np.int32
            or A.data.dtype!=np.float64 or inverse_diag.dtype!=np.float64
            or inverse_diag.shape!=(n,) or n>=2**31//10):
        raise ValueError("Paired PCG needs square float64 CSR with int32 indexing")
    rhs = np.ascontiguousarray(rhs,dtype=np.float64)
    x = np.empty_like(rhs)
    work = np.empty((3,n,rank),dtype=np.float64)
    counts = np.empty(rank,dtype=np.int32)
    arrays = (A.indptr,A.indices,A.data,inverse_diag,rhs,x,work,counts)
    if not all(a.flags.c_contiguous for a in arrays):
        raise ValueError("Paired PCG expects contiguous CSR and dense buffers")
    routine = fn if rank == 2 else fn.scalar
    status = routine(n,*(a.ctypes.data for a in arrays),rtol,atol,10*n)
    if status:
        raise ArithmeticError(f"Paired PCG did not converge ({status})")
    return x,counts
