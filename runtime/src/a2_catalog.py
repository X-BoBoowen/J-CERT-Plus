"""Sparse root identity and bounded candidate access for the A2 experiment.

The public identity is ``i*n+j`` for ``0 <= i < j < n``.  Positive root
edges are absent; negative root edges are flips and every root nonedge is
an insertion.  The catalog never stores an array indexed by that domain.
"""
from dataclasses import dataclass
import heapq
from operator import index
import sys
from time import perf_counter

import numpy as np
from scipy import sparse


_INT64_MAX = np.iinfo(np.int64).max


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a Boolean")
    try:
        return index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _integers(values, name):
    """Check before conversion: canonical IDs must never silently truncate."""
    if not isinstance(values, (np.ndarray, list, tuple)):
        try:
            values = tuple(values)
        except TypeError as error:
            raise ValueError(f"{name} must be a one-dimensional iterable") from error
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not arr.size:
        return np.empty(0, dtype=np.int64)
    if arr.dtype.kind not in "iu":
        if arr.dtype.kind != "O":
            raise ValueError(f"{name} must contain integers")
        checked = [_integer(v, name) for v in arr]
        if any(v < -_INT64_MAX - 1 or v > _INT64_MAX for v in checked):
            raise OverflowError(f"{name} does not fit signed int64")
        return np.asarray(checked, dtype=np.int64)
    if arr.dtype.kind == "u" and np.any(arr > _INT64_MAX):
        raise OverflowError(f"{name} does not fit signed int64")
    return arr.astype(np.int64, copy=False)


def _membership(sorted_values, values):
    found = np.zeros(len(values), dtype=bool)
    if len(sorted_values) and len(values):
        positions = np.searchsorted(sorted_values, values)
        valid = positions < len(sorted_values)
        found[valid] = sorted_values[positions[valid]] == values[valid]
    return found


@dataclass(frozen=True)
class CandidateBatch:
    ids: np.ndarray
    i: np.ndarray
    j: np.ndarray
    flip: np.ndarray
    weights: np.ndarray

    def __len__(self):
        return len(self.ids)


class AllCatalog:
    """A fixed sparse root binding, with no complete candidate representation."""

    def __init__(self, W):
        if not sparse.issparse(W) or len(W.shape) != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("expected a square sparse root graph")
        self.n = int(W.shape[0])
        if self.n * self.n > _INT64_MAX:
            raise OverflowError("canonical endpoint identity requires n*n <= int64 max")
        if W.dtype.kind not in "iuf" or W.dtype.itemsize > 8:
            raise ValueError("expected real half-integer signed root weights")
        root = sparse.csr_matrix(W, copy=True)
        root.sum_duplicates()
        root.sort_indices()
        values = root.data.astype(np.float64, copy=False)
        if ((root != root.T).nnz or np.any(root.diagonal()) or
                not np.isfinite(values).all() or np.any(values == 0) or
                np.any(abs(values) > 10) or np.any(2 * values != np.rint(2 * values))):
            raise ValueError("expected the natural half-integer signed projection")
        self.W = root
        upper = sparse.triu(root, k=1, format="coo")
        keys = upper.row.astype(np.int64) * self.n + upper.col.astype(np.int64)
        order = np.argsort(keys, kind="stable")
        self.root_edge_ids = keys[order]
        signed = upper.data[order].astype(np.float64, copy=False)
        negative = signed < 0
        self.positive_ids = self.root_edge_ids[~negative]
        self.flip_ids = self.root_edge_ids[negative]
        self.flip_weights = np.rint(-16 * signed[negative]).astype(np.int64)
        self.root_edge_keys = frozenset(map(int, self.root_edge_ids))
        pairs = self.n * (self.n - 1) // 2
        self.insertions = pairs - len(self.root_edge_ids)
        self.m = self.insertions + len(self.flip_ids)
        self.max_weight = int(self.flip_weights.max(initial=1))
        for arr in (root.data, root.indices, root.indptr, self.root_edge_ids,
                    self.positive_ids, self.flip_ids, self.flip_weights):
            arr.flags.writeable = False

    def _checked(self, ids):
        ids = _integers(ids, "candidate IDs")
        if not len(ids):
            return ids
        if self.n == 0 or np.any(ids < 0) or np.any(ids >= self.n * self.n):
            raise ValueError("candidate identity is outside the root vertex domain")
        if np.any(ids // self.n >= ids % self.n):
            raise ValueError("candidate identity requires 0 <= i < j < n")
        if _membership(self.positive_ids, ids).any():
            raise ValueError("a positive root edge is not a legal candidate")
        return ids

    def _batch_valid(self, ids):
        """Decode IDs already established legal by the bounded enumerator."""
        i = ids // self.n if self.n else np.empty(0, dtype=np.int64)
        j = ids % self.n if self.n else np.empty(0, dtype=np.int64)
        flip = np.zeros(len(ids), dtype=bool)
        weights = np.ones(len(ids), dtype=np.int64)
        if len(self.flip_ids) and len(ids):
            positions = np.searchsorted(self.flip_ids, ids)
            valid = positions < len(self.flip_ids)
            flip[valid] = self.flip_ids[positions[valid]] == ids[valid]
            weights[flip] = self.flip_weights[positions[flip]]
        return CandidateBatch(ids, i, j, flip, weights)

    def scalar(self, e):
        e = _integer(e, "candidate ID")
        if e < 0 or e > _INT64_MAX or self.n == 0 or e >= self.n * self.n:
            raise ValueError("candidate identity is outside the root vertex domain")
        i, j = divmod(e, self.n)
        if i >= j:
            raise ValueError("candidate identity requires 0 <= i < j < n")
        flip, weight = False, 1
        if e in self.root_edge_keys:
            position = int(np.searchsorted(self.flip_ids, e))
            if position == len(self.flip_ids) or int(self.flip_ids[position]) != e:
                raise ValueError("a positive root edge is not a legal candidate")
            flip, weight = True, int(self.flip_weights[position])
        # Existing verified virtual columns call .item() on scalar lookup.
        return np.int64(i), np.int64(j), np.bool_(flip), np.int64(weight)

    def batch(self, ids):
        return self._batch_valid(self._checked(ids))

    def iter_blocks(self, block_size=65536, excluded=(), counters=None):
        """Yield lexicographic legal rows with O(block_size) temporary arrays.

        Fill a bounded ID buffer across triangular rows, then remove positive
        edges and exclusions with vectorized sorted lookups.  No Python list of
        pairs, inverse candidate map, or all-candidate visited array is built.
        """
        block_size = _integer(block_size, "block size")
        if block_size <= 0:
            raise ValueError("block size must be positive")
        excluded = np.unique(self._checked(excluded))
        if counters is not None:
            for key in ("stream_pairs_examined", "stream_positive_rejects",
                        "stream_processed_skips", "stream_insertions_emitted",
                        "stream_flips_emitted"):
                counters.setdefault(key, 0)
        i, j = 0, 1
        while i < self.n - 1:
            capacity = min(block_size, self.n * (self.n - 1) // 2)
            buffer = np.empty(capacity, dtype=np.int64)
            used = 0
            while used < capacity and i < self.n - 1:
                count = min(capacity - used, self.n - j)
                buffer[used:used + count] = i * self.n + np.arange(j, j + count, dtype=np.int64)
                used += count
                j += count
                if j == self.n:
                    i += 1
                    j = i + 1
            ids = buffer[:used]
            positive = _membership(self.positive_ids, ids)
            keep = ~positive
            if len(excluded):
                skipped = _membership(excluded, ids)
                keep &= ~skipped
            else:
                skipped = np.zeros(len(ids), dtype=bool)
            ids = ids[keep]
            batch = self._batch_valid(ids)
            if counters is not None:
                # These are paid chunks, including work before the current
                # yield. A consumer stopping here has already paid this work.
                counters["stream_pairs_examined"] += used
                counters["stream_positive_rejects"] += int(positive.sum())
                counters["stream_processed_skips"] += int(skipped.sum())
                flips = int(batch.flip.sum())
                counters["stream_insertions_emitted"] += len(batch) - flips
                counters["stream_flips_emitted"] += flips
            if len(ids):
                yield batch

    def materialize(self):
        """Build D's private explicit arrays in bounded vectorized chunks."""
        if self.n > 2**31:
            raise OverflowError("explicit endpoint arrays require signed int32 vertices")
        i = np.empty(self.m, dtype=np.int32)
        j = np.empty(self.m, dtype=np.int32)
        flip = np.empty(self.m, dtype=bool)
        position = 0
        for batch in self.iter_blocks():
            stop = position + len(batch)
            i[position:stop] = batch.i
            j[position:stop] = batch.j
            flip[position:stop] = batch.flip
            position = stop
        if position != self.m:
            raise ArithmeticError("explicit catalog cardinality mismatch")
        return i, j, flip

    def dense_positions(self, ids):
        """Compute D's ordinal as triangle rank minus preceding positive edges."""
        ids = self._checked(ids)
        if not len(ids):
            return np.empty(0, dtype=np.int32 if self.m < 2**31 else np.int64)
        i, j = ids // self.n, ids % self.n
        ranks = i * (2 * self.n - i - 1) // 2 + j - i - 1
        ranks -= np.searchsorted(self.positive_ids, ids)
        return ranks.astype(np.int32 if self.m < 2**31 else np.int64, copy=False)

    def memory(self):
        arrays = (self.W.data, self.W.indices, self.W.indptr, self.root_edge_ids,
                  self.positive_ids, self.flip_ids, self.flip_weights)
        array_bytes = sum(int(arr.nbytes) for arr in arrays)
        # Array payload is never mixed into Python-container accounting.
        seen = set()
        python_bytes = (sys.getsizeof(self) + _python_size(self.__dict__, seen) +
                        _python_size(self.W.__dict__, seen))
        return dict(array_payload_bytes=array_bytes,
                    python_structure_bytes=int(python_bytes),
                    total_owned_bytes=array_bytes + int(python_bytes),
                    sparse_root_edges=len(self.root_edge_ids),
                    negative_flips=len(self.flip_ids), candidate_count=self.m,
                    candidate_array_rows=0, root_csr_shared=False,
                    scope="Owned sparse-root payload and Python structures; no caller "
                          "batches, materialized outputs, sorting workspace, or RSS")


def _python_size(value, seen):
    """Owned Python container/object bytes, with shared references counted once."""
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, np.ndarray):
        return sys.getsizeof(value) - (value.nbytes if value.flags.owndata else 0)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_python_size(k, seen) + _python_size(v, seen)
                    for k, v in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(_python_size(v, seen) for v in value)
    return size


class MixedFrontier:
    """Exact coarse-cap order for one certified version, without emitted history.

    Sorted center positions a<b form triangular insertion rows. Each row starts
    at b=n-1 and walks inward; its integer width and squared cap never increase.
    A heap of their current heads is merged with the sparse weighted flip order.
    Root edges, chosen edits, and caller-supplied processed IDs are rejected
    during advancement. Equal caps are always eligible at the threshold.

    ``excluded`` sets are caller-owned collections of already validated IDs;
    they are consulted directly, never copied into an all-emitted history.
    Other iterables are checked and converted to a temporary membership set.
    """

    def __init__(self, catalog, xc, rc, chosen=()):
        self.catalog = catalog
        self.xc = _integers(xc, "integer centers").copy()
        self.rc = _integer(rc, "integer radius")
        if len(self.xc) != catalog.n or self.rc < 0:
            raise ValueError("one integer center per vertex and nonnegative radius required")
        self.xc.flags.writeable = False
        self.chosen = frozenset(map(int, catalog._checked(chosen)))
        self.counters = dict(raw_heap_pops=0, flip_cursor_pops=0,
                             total_source_pops=0, graph_rejects=0,
                             committed_rejects=0, processed_skips=0,
                             processed_insertion_skips=0, processed_flip_skips=0,
                             legal_insertion_pops=0,
                             insertion_emissions=0, flips_emitted=0,
                             insertion_caps_built=0,
                             flip_caps_built=len(catalog.flip_ids),
                             node_sort_items=catalog.n,
                             flip_sort_items=len(catalog.flip_ids),
                             heap_build_items=max(0, catalog.n - 1),
                             frontier_resets=1)
        start = perf_counter()
        self.order = np.argsort(self.xc, kind="stable")
        self.order.flags.writeable = False
        node_sorted = perf_counter()
        flips = []
        for e, weight in zip(catalog.flip_ids, catalog.flip_weights):
            e, weight = int(e), int(weight)
            i, j = divmod(e, catalog.n)
            a, b = int(self.xc[i]), int(self.xc[j])
            cap = weight * (-4 * a * b + 4 * self.rc * (abs(a) + abs(b) + self.rc))
            flips.append((-cap, e))
        self.flips = sorted(flips)
        self.flip_position = 0
        flip_sorted = perf_counter()
        self.heap = [self._entry(left, catalog.n - 1)
                     for left in range(max(0, catalog.n - 1))]
        heapq.heapify(self.heap)
        heap_built = perf_counter()
        self._prune_heads(())
        filtered = perf_counter()
        self.build_phases = dict(node_sort=node_sorted - start,
                                 flip_order=flip_sorted - node_sorted,
                                 heap_build=heap_built - flip_sorted,
                                 initial_head_filter=filtered - heap_built)

    def _entry(self, left, right):
        i, j = int(self.order[left]), int(self.order[right])
        # Convert each operand before arithmetic: exact caps can exceed int64.
        width = int(self.xc[j]) - int(self.xc[i]) + 2 * self.rc
        cap = width * width
        e = min(i, j) * self.catalog.n + max(i, j)
        self.counters["insertion_caps_built"] += 1
        return -cap, e, left, right

    def _take_insertion(self):
        entry = heapq.heappop(self.heap)
        self.counters["raw_heap_pops"] += 1
        self.counters["total_source_pops"] += 1
        _, _, left, right = entry
        if right > left + 1:
            heapq.heappush(self.heap, self._entry(left, right - 1))
        return entry[1]

    def _take_flip(self):
        e = self.flips[self.flip_position][1]
        self.flip_position += 1
        self.counters["flip_cursor_pops"] += 1
        self.counters["total_source_pops"] += 1
        return e

    def _prune_heads(self, excluded):
        while self.heap:
            e = self.heap[0][1]
            if e in self.catalog.root_edge_keys:
                counter = "graph_rejects"
            elif e in self.chosen:
                counter = "committed_rejects"
            elif e in excluded:
                counter = "processed_skips"
            else:
                break
            self._take_insertion()
            self.counters[counter] += 1
            if counter == "processed_skips":
                self.counters["processed_insertion_skips"] += 1
                self.counters["legal_insertion_pops"] += 1
        while self.flip_position < len(self.flips):
            e = self.flips[self.flip_position][1]
            if e in self.chosen:
                counter = "committed_rejects"
            elif e in excluded:
                counter = "processed_skips"
            else:
                break
            self._take_flip()
            self.counters[counter] += 1
            if counter == "processed_skips":
                self.counters["processed_flip_skips"] += 1

    @property
    def upper(self):
        """An exact integer dominating every remaining legal coarse cap."""
        insertion = -self.heap[0][0] if self.heap else None
        flip = (-self.flips[self.flip_position][0]
                if self.flip_position < len(self.flips) else None)
        if insertion is None:
            return flip
        if flip is None:
            return insertion
        return max(insertion, flip)

    def pop_batch(self, limit, threshold=None, excluded=()):
        limit = _integer(limit, "batch limit")
        if limit <= 0:
            raise ValueError("batch limit must be positive")
        if threshold is not None:
            threshold = _integer(threshold, "coarse threshold")
        if not isinstance(excluded, (set, frozenset, dict)):
            excluded = frozenset(map(int, self.catalog._checked(excluded)))
        ids = []
        self._prune_heads(excluded)
        while len(ids) < limit:
            upper = self.upper
            if upper is None or (threshold is not None and upper < threshold):
                break
            flip_left = self.flip_position < len(self.flips)
            if self.heap and (not flip_left or
                              self.heap[0][:2] <= self.flips[self.flip_position]):
                ids.append(self._take_insertion())
                self.counters["insertion_emissions"] += 1
                self.counters["legal_insertion_pops"] += 1
            else:
                ids.append(self._take_flip())
                self.counters["flips_emitted"] += 1
            self._prune_heads(excluded)
        return self.catalog._batch_valid(np.asarray(ids, dtype=np.int64))

    def memory(self):
        arrays = (self.xc, self.order)
        array_bytes = sum(int(arr.nbytes) for arr in arrays)
        seen = set()
        python_bytes = sys.getsizeof(self) + sys.getsizeof(self.__dict__)
        for value in (self.xc, self.order, self.rc, self.chosen):
            python_bytes += _python_size(value, seen)
        heap_bytes = _python_size(self.heap, seen)
        flip_bytes = _python_size(self.flips, seen)
        python_bytes += heap_bytes + flip_bytes
        for value in (self.flip_position, self.counters, self.build_phases):
            python_bytes += _python_size(value, seen)
        return dict(array_payload_bytes=array_bytes,
                    python_structure_bytes=int(python_bytes),
                    total_owned_bytes=array_bytes + int(python_bytes),
                    insertion_heap_python_bytes=int(heap_bytes),
                    sparse_flip_order_python_bytes=int(flip_bytes),
                    insertion_heap_rows=len(self.heap),
                    sparse_flip_order_rows=len(self.flips),
                    catalog_shared=True, excluded_membership_shared=True,
                    emitted_history_rows=0,
                    scope="Owned current-version arrays, heap/list tuples, Python "
                          "integer keys and metadata; excludes shared catalog, "
                          "caller exclusions, returned batches and temporary workspace")
