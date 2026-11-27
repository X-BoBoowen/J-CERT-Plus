"""Direct planner with a single CSR pass for both residual products."""
import ctypes as ct
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from fast_policy_kernel import FastCostPlanner
from policy_kernel import CertificateError


def compile_residual(directory):
    path = Path(directory)/("direct6.dll" if os.name == "nt" else "direct6.so")
    subprocess.run([os.environ.get("CC", "gcc"), "-O3", "-fno-fast-math",
                    "-ffp-contract=off", "-fPIC", "-shared",
                    str(Path(__file__).with_suffix(".c")), "-o", str(path), "-lm"],
                   check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    library = ct.CDLL(str(path))
    routine = library.residual6
    routine.argtypes = [ct.c_int] + [ct.c_void_p]*6
    routine.restype = ct.c_double
    return routine


class DirectPlanner(FastCostPlanner):
    def __init__(self, *args, residual, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_residual = residual
        self.cap_work = np.empty(self.A.shape[0])

    def memory_bytes(self):
        return super().memory_bytes() + self.cap_work.nbytes

    def residual_radius(self, rhs, x):
        A = self.A
        if (A.indices.dtype != np.int32 or A.indptr.dtype != np.int32
                or A.data.dtype != np.float64 or not A.indices.flags.c_contiguous
                or not A.indptr.flags.c_contiguous or not A.data.flags.c_contiguous):
            return super().residual_radius(rhs, x)
        rhs = np.ascontiguousarray(rhs, dtype=np.float64)
        x = np.ascontiguousarray(x, dtype=np.float64)
        if rhs.shape != self.cap_work.shape or x.shape != rhs.shape:
            raise ValueError("invalid residual dimensions")
        arrays = (A.indptr, A.indices, A.data, rhs, x, self.cap_work)
        rho = self.native_residual(len(x), *(v.ctypes.data for v in arrays))
        if not math.isfinite(rho):
            raise CertificateError("nonfinite residual radius")
        return rho
