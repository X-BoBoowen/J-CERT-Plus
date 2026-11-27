"""Enumerate upper-triangular candidate pairs with bounded sparse products."""
import numpy as np
from scipy.sparse import csr_matrix


def build_universe(adjacency, block_rows=256):
    weights = csr_matrix(adjacency, copy=True)
    weights.sum_duplicates()
    weights.eliminate_zeros()
    support = weights.astype(bool)
    negative = weights < 0
    n = weights.shape[0]

    def blocks():
        for start in range(0, n, block_rows):
            end = min(n, start + block_rows)
            edges = support[start:end]
            pairs = (edges @ support).astype(np.int8)
            pairs -= pairs.multiply(edges)
            pairs += negative[start:end].astype(np.int8) * 2
            pairs.eliminate_zeros()
            pairs.sort_indices()
            coo = pairs.tocoo()
            left = coo.row + start
            keep = left < coo.col
            yield left[keep], coo.col[keep], coo.data[keep] == 2

    count = sum(len(left) for left, _, _ in blocks())
    left = np.empty(count, np.int32)
    right = np.empty_like(left)
    flip = np.empty(count, bool)
    offset = 0
    for a, b, f in blocks():
        end = offset + len(a)
        left[offset:end], right[offset:end], flip[offset:end] = a, b, f
        offset = end
    return left, right, flip


class build_incidence:
    """Index for lexicographically ordered undirected endpoints."""

    def __init__(self, i, j, n):
        self.left = np.searchsorted(i, np.arange(n + 1))
        self.right = np.r_[0, np.cumsum(np.bincount(j, minlength=n))]
        dtype = np.int32 if len(j) < 2**31 else np.int64
        self.ids = np.argsort(j, kind="stable").astype(dtype, copy=False)

    def incident(self, vertices):
        parts = []
        for v in vertices:
            parts.append(np.arange(self.left[v], self.left[v+1], dtype=self.ids.dtype))
            parts.append(self.ids[self.right[v]:self.right[v+1]])
        return np.unique(np.concatenate(parts)) if parts else self.ids[:0]
