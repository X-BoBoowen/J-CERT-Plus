"""Fixed-slot CSR ablation on the same certified direct planner."""
import numpy as np
from scipy import sparse

from direct6 import DirectPlanner


class SlotPlanner(DirectPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The numeric factor was built on the real root sparsity pattern.
        base = self.A.tocoo()
        additions = ~self.flip
        self.logical_nnz = self.A.nnz
        i, j = self.i[additions], self.j[additions]
        self.A = sparse.csr_matrix((np.r_[base.data, np.zeros(2*len(i))],
                                    (np.r_[base.row, i, j], np.r_[base.col, j, i])),
                                   shape=base.shape)
        self._refresh()
        # Sorted flattened coordinates allow four batched searches at setup.
        n = self.A.shape[0]
        row = np.repeat(np.arange(n, dtype=np.int64), np.diff(self.A.indptr))
        keys = row*n + self.A.indices
        i, j = self.i.astype(np.int64), self.j.astype(np.int64)
        wanted = np.column_stack((i*n+i, i*n+j, j*n+i, j*n+j))
        self.slots = np.searchsorted(keys, wanted).astype(self.A.indices.dtype)
        if not np.array_equal(keys[self.slots], wanted):
            raise ValueError("catalog slot missing")
        diagonal = self.w * additions
        off = -self.w * (1 + self.flip.astype(int))
        self.deltas = np.column_stack((diagonal, off, off, diagonal))

    def _patch(self, e, sign):
        slots = self.slots[e]
        values = self.A.data[slots] + sign*self.deltas[e]
        self.A.data[slots] = values
        self.absA.data[slots] = abs(values)
        self.diag[self.i[e]], self.diag[self.j[e]] = values[0], values[3]
        self.invdiag[self.i[e]], self.invdiag[self.j[e]] = 1/values[0], 1/values[3]

    def _apply_sparse_edit(self, e):
        self._patch(e, 1)
        self.logical_nnz += 2 * int(not self.flip[e])

    def _historical(self, t):
        # The factor stays at the child. Only the CSR used for the subsequent
        # parent residual is rolled back. Its entries lie on the checked grid.
        self._patch(t.winner, -1)
        try:
            return super()._historical(t)
        finally:
            self._patch(t.winner, 1)

    def _rhs_units(self):
        if self._factor_entries is None:
            self._factor_entries = self.factor.stored_entries()
        return 4*(self._factor_entries+self.logical_nnz)

    def commit(self, e):
        super().commit(e)
        if self.pending is not None:
            self.pending.replay_work = 4*self.pending.rank*self.logical_nnz

    def memory_bytes(self):
        size = super().memory_bytes()+self.slots.nbytes+self.deltas.nbytes
        if self.pending is not None and self.pending.operator is not None:
            # The saved parent tuple aliases the mutable CSR and vector storage.
            size -= self.gamma.nbytes+self.invdiag.nbytes
            for matrix in (self.A, self.absA):
                size -= matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
        return size
