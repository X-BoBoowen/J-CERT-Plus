"""Opt-in single-thread fused Jacobi-PCG planner on the committed CSR operator."""
import numpy as np

from fast_policy_kernel import FastCostPlanner
from paired_pcg import solve_columns


class NativeCostPlanner(FastCostPlanner):
    def __init__(self, *args, native_pcg, **kwargs):
        super().__init__(*args, **kwargs)
        if self.factor is not None:
            raise ValueError("native PCG requires the iterative backend")
        self.native_pcg = native_pcg

    def _columns(self, rhs):
        Y, counts = solve_columns(self.native_pcg, self.A, self.invdiag, rhs, 2e-12, 0.)
        self.ledger["cg_iterations"] += int(counts.sum())
        # The planner prices the last individual RHS, as on the scalar path.
        self.last_iterations = max(1, int(counts[-1]))
        return Y

    def solve(self, rhs):
        rhs = np.asarray(rhs, dtype=float)
        if rhs.shape != (self.A.shape[0],) or not np.isfinite(rhs).all():
            raise ValueError("invalid RHS")
        x = self._columns(rhs[:, None])[:, 0]
        return x, self.residual_radius(rhs, x)

    def _fresh(self, e, kind):
        if not self.flip[e]:
            return super()._fresh(e, kind)
        self.ledger[kind] += 1
        self.ledger[kind + "_rhs"] += 2
        self.ledger["winner_rhs" if kind == "W" else "refine_rhs"] += 2
        rhs = np.zeros((self.A.shape[0], 2))
        rhs[self.i[e]] = 1.
        rhs[self.j[e]] = [-1., 1.]
        Y = self._columns(rhs)
        err = 2*sum(self.residual_radius(u, y) for u, y in zip(rhs.T, Y.T))
        return self._response(e, Y, err)
