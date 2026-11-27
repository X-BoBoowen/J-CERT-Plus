"""Exact, current-parent node-transposition witnesses for weighted edits.

This module proves equality of two whole quadratic objectives. Numerical score
closeness is never a premise. Work is performed only after a real score overlap;
no graph-wide validation or hash is added to ordinary separated comparisons.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .common import DEN, SCALE, UNIT, digest


@dataclass(frozen=True, eq=False)
class TieContext:
    A: object
    opinion: np.ndarray
    parent_version_id: str
    chosen: frozenset


def _edit(kernel, A, e, chosen):
    if isinstance(e, (bool, np.bool_)) or not isinstance(e, (int, np.integer)) or e < 0:
        return None
    if e in chosen:
        return None
    try:
        i, j, flip = (column[e] for column in kernel.pool)
        d = kernel.d[e]
    except (IndexError, KeyError, ValueError, TypeError):
        return None
    if (not all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
                for v in (i, j, d)) or
            not isinstance(flip, (bool, np.bool_))):
        return None
    i, j, d, flip = int(i), int(j), int(d), bool(flip)
    if not (0 <= i < j < A.shape[0]) or d <= 0:
        return None
    # The bound catalog supplies the operation. The parent supplies its current
    # legality: nonedge insertion or the original, still negative, weighted edge.
    if int(A[i, j]) != (d if flip else 0):
        return None
    return dict(i=i, j=j, flip=flip, d=d)


def _update(edit):
    i, j, d = edit['i'], edit['j'], edit['d']
    if edit['flip']:
        return {(i, j): -2*d, (j, i): -2*d}
    return {(i, i): d, (j, j): d, (i, j): -d, (j, i): -d}


class _BoundExchange:
    def __init__(self, kernel, A, context):
        self.kernel, self.A, self.context = kernel, A, context
        self.operator_hash = digest(A.indptr, A.indices, A.data)
        self.opinion_hash = digest(context.opinion)

    def certify(self, source, target):
        if source == target:
            return None
        A, ctx = self.A, self.context
        left = _edit(self.kernel, A, source, ctx.chosen)
        right = _edit(self.kernel, A, target, ctx.chosen)
        if left is None or right is None:
            return None
        if left['flip'] != right['flip'] or left['d'] != right['d']:
            return None
        pair_left, pair_right = {left['i'], left['j']}, {right['i'], right['j']}
        moved = sorted(pair_left ^ pair_right)
        if len(moved) != 2:
            return None
        u, v = moved
        if ctx.opinion[u] != ctx.opinion[v] or A[u, u] != A[v, v]:
            return None

        # Symmetry was checked once by bind_exchange. Equal outside row entries
        # and equal diagonals are then exactly equivalent to PAP^T=A.
        ru, rv = A.getrow(u), A.getrow(v)
        ku = (ru.indices != u) & (ru.indices != v) & (ru.data != 0)
        kv = (rv.indices != u) & (rv.indices != v) & (rv.data != 0)
        if (not np.array_equal(ru.indices[ku], rv.indices[kv]) or
                not np.array_equal(ru.data[ku], rv.data[kv])):
            return None

        def permute(node):
            return v if node == u else u if node == v else node

        conjugate = {(permute(i), permute(j)): value
                     for (i, j), value in _update(left).items()}
        if conjugate != _update(right):
            return None
        return dict(kind='weighted_node_transposition_v1',
            parent_version_id=ctx.parent_version_id,
            operator_sha256=self.operator_hash, opinion_sha256=self.opinion_hash,
            opinion_denominator=DEN, operator_scale=SCALE, score_unit=UNIT,
            source_candidate_id=int(source), target_candidate_id=int(target),
            swap=[u, v], source_edit=left, target_edit=right)


def bind_exchange(kernel, A, s_float, context):
    """Bind one actual overlapping comparison, or return no proof context.

    The current operator is immutable for the duration of resolve. A context is
    not reusable across a commit, even if its caller accidentally kept it alive.
    The exact integer opinion, not a rounded real approximation, is authoritative.
    """
    if not isinstance(context, TieContext) or context.A is not A:
        return None
    if (not sparse.isspmatrix_csr(A) or A.shape[0] != A.shape[1] or
            A.dtype != np.int64 or not A.has_canonical_format):
        return None
    s = context.opinion
    if (not isinstance(s, np.ndarray) or s.dtype != np.int64 or s.shape != (A.shape[0],) or
            np.any(s < -DEN) or np.any(s > DEN) or
            not isinstance(context.parent_version_id, str) or not context.parent_version_id or
            not isinstance(context.chosen, frozenset)):
        return None
    if not np.array_equal(np.asarray(s_float), s/DEN):
        return None
    if (A != A.T).nnz:
        return None
    return _BoundExchange(kernel, A, context)


def certify_exchange(kernel, A, s_float, source, target, context):
    """Standalone pair checker; resolve shares its bound context across rivals."""
    bound = bind_exchange(kernel, A, s_float, context)
    return None if bound is None else bound.certify(source, target)
