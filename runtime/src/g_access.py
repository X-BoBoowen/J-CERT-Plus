"""V3 strong simple access: flips first, then independent diagonal pair blocks.

No H row frontier is imported or used.  The common service supplies certified
current centers of A_int^-1 s; A_int=16*A_phys.  A coarse upper c represents
physical gain <=16*c/SCREEN**2.  For legal insertions d=1 and A_int[i,j]=0,
R_int >= 1/A_int[i,i]+1/A_int[j,j].  Every block bound is rounded UP.
"""
import heapq
import sys
from time import perf_counter

import numpy as np

from .a2_access import Access, COUNTERS, RATIO, batch_bytes
from .a2_catalog import _integers, _integer, _python_size
from .common import SCALE
from long_promotion import SCREEN


def global_insertion_upper(xc, rc):
    """min(4 omega, omega range(z_phys)^2), in the coarse score units."""
    width = (int(xc.max()) - int(xc.min()) if len(xc) else 0) + 2 * int(rc)
    return min((2 * SCREEN // SCALE)**2, width**2)


class PairBlocks:
    """Binary triangular pair partition, created only after the global test fails.

    A block is either one triangular interval or a rectangle between disjoint
    intervals in sorted-center order.  Triangles split into LL/LR/RR; rectangles
    split their longer side.  Singleton pairs are checked for current legality.
    The pending heap covers every ungenerated pair. Equal upper bounds are kept.
    """
    def __init__(self, catalog, xc, rc, diagonal, chosen=(), *, audit=False):
        self.catalog, self.xc, self.rc = catalog, xc, int(rc)
        self.diagonal = diagonal
        self.chosen = frozenset(chosen)
        start = perf_counter()
        self.order = np.argsort(xc, kind='stable')
        sorted_at = perf_counter()
        self.nodes, self.heap, self.serial = {}, [], 0
        self.counters = dict(g_pair_blocks_tested=0, g_pair_blocks_expanded=0,
                             g_pair_leaves_generated=0, g_pair_graph_rejects=0,
                             g_pair_chosen_rejects=0, g_pair_processed_rejects=0,
                             g_block_insertions_emitted=0)
        self.global_upper = global_insertion_upper(xc, rc)
        self.audit_bounds = [] if audit else None
        if catalog.n >= 2:
            self._push(0, catalog.n, 0, catalog.n)
        self.build_phases = dict(node_sort=sorted_at-start, block_root=perf_counter()-sorted_at)

    def _node_maxdiag(self, a, b):
        key = (a, b)
        if key not in self.nodes:
            self.nodes[key] = int(self.diagonal[self.order[a:b]].max())
        return self.nodes[key]

    def _push(self, a, b, c, d):
        if b <= a or d <= c or ((a, b) == (c, d) and b-a < 2):
            return
        if not ((a, b) == (c, d) or b <= c):
            raise ArithmeticError('pair partition lost disjoint ordered ranges')
        left_min, left_max = int(self.xc[self.order[a]]), int(self.xc[self.order[b-1]])
        right_min, right_max = int(self.xc[self.order[c]]), int(self.xc[self.order[d-1]])
        width = max(abs(left_max-right_min), abs(right_max-left_min)) + 2*self.rc
        di, dj = self._node_maxdiag(a, b), self._node_maxdiag(c, d)
        numerator = width**2 * di * dj
        denominator = di*dj + di + dj
        upper = min(self.global_upper, (numerator + denominator - 1)//denominator)
        self.counters['g_pair_blocks_tested'] += 1
        if self.audit_bounds is not None:
            self.audit_bounds.append(dict(ranges=[a, b, c, d], upper_coarse=upper))
        heapq.heappush(self.heap, (-upper, self.serial, a, b, c, d))
        self.serial += 1

    @property
    def upper(self):
        return -self.heap[0][0] if self.heap else None

    def pop(self, threshold, excluded):
        while self.heap:
            if self.upper < threshold:
                return None
            _, _, a, b, c, d = heapq.heappop(self.heap)
            if b-a == d-c == 1:
                i, j = int(self.order[a]), int(self.order[c])
                if i == j:
                    raise ArithmeticError('diagonal leaf in pair partition')
                e = min(i, j)*self.catalog.n + max(i, j)
                self.counters['g_pair_leaves_generated'] += 1
                if e in self.catalog.root_edge_keys:
                    self.counters['g_pair_graph_rejects'] += 1
                elif e in self.chosen:
                    self.counters['g_pair_chosen_rejects'] += 1
                elif e in excluded:
                    self.counters['g_pair_processed_rejects'] += 1
                else:
                    self.counters['g_block_insertions_emitted'] += 1
                    return e
                continue
            self.counters['g_pair_blocks_expanded'] += 1
            if (a, b) == (c, d):
                middle = (a+b)//2
                self._push(a, middle, a, middle)
                self._push(a, middle, middle, b)
                self._push(middle, b, middle, b)
            elif b-a >= d-c:
                middle = (a+b)//2
                self._push(a, middle, c, d)
                self._push(middle, b, c, d)
            else:
                middle = (c+d)//2
                self._push(a, b, c, middle)
                self._push(a, b, middle, d)
        return None


class GAccess(Access):
    """Access-only replacement; the session's scoring/resolver/transport is shared."""
    def __init__(self, catalog, method, ledger, *, timers=True, measurement_policy='legacy_deep'):
        if method != 'G':
            raise ValueError('GAccess expects G')
        super().__init__(catalog, method, ledger, timers=timers, measurement_policy=measurement_policy)
        with ledger.phase('catalog_build'):
            self.root_diagonal = np.rint(SCALE*(1+np.abs(catalog.W).sum(axis=1).A1)).astype(np.int64)
            if np.any(self.root_diagonal < SCALE):
                raise ArithmeticError('invalid physical diagonal')
        self.blocks = None
        self.flip_order = []

    def _flips(self, xc, rc):
        start = perf_counter()
        entries = []
        for e, weight in zip(self.catalog.flip_ids, self.catalog.flip_weights):
            e, weight = int(e), int(weight)
            i, j = divmod(e, self.catalog.n)
            a, b = int(xc[i]), int(xc[j])
            cap = weight*(-4*a*b + 4*int(rc)*(abs(a)+abs(b)+int(rc)))
            entries.append((-cap, e))
        entries.sort()
        return entries, perf_counter()-start

    def begin(self, xc, rc, hot_size, ledger):
        self.index = self.root_frontier = self.frontier = self.iterator = None
        self.max_batch_bytes = self.max_frontier_bytes = 0
        self.frontier_memory = {}
        self.root_caps_created = 0
        self.stats = None
        self.blocks = None
        self.scan_counts = {}
        self.current_xc, self.current_rc = xc, int(rc)
        self.diagonal = self.root_diagonal.copy()
        with ledger.phase('frontier_prepare'):
            self.root_flips, elapsed = self._flips(xc, rc)
        with ledger.phase('seed_access'):
            seeds = np.asarray([e for _, e in self.root_flips[:hot_size]], dtype=np.int64)
        self.seed_counts = dict(seed_candidates=len(seeds), seed_screen_rows=len(seeds),
                                seed_insertions=0, seed_flips=len(seeds))
        self.seed_build_phases = dict(flip_order=elapsed, node_sort=0., block_root=0.)
        self.root_seed_rule = 'paid-sparse-flip-upper-first64/canonical-ties; no insertion seeds'
        self.root_seed_stats = {}
        return seeds

    def warm_candidates(self, hot):
        """Defer warm insertions until after flips; retain their maintained Grams.

        Removing an ID from hot also removes it from the cold exclusion set.
        The block proof must therefore cover it again before it can be omitted.
        """
        flips = {e for e in hot if bool(self.catalog.scalar(e)[2])}
        self.deferred_warm_insertions = len(hot)-len(flips)
        return flips

    def open_version(self, xc, rc, chosen, excluded, epoch, parent, ledger):
        self.current_xc = _integers(xc, 'integer centers')
        self.current_rc = _integer(rc, 'integer radius')
        if len(self.current_xc) != self.catalog.n or self.current_rc < 0:
            raise ValueError('invalid current state enclosure')
        self.chosen = frozenset(map(int, self.catalog._checked(chosen)))
        self.diagonal = self.root_diagonal.copy()
        for e in self.chosen:
            i, j, flip, weight = self.catalog.scalar(e)
            if not flip:
                self.diagonal[i] += int(weight)
                self.diagonal[j] += int(weight)
        self.ledger = ledger
        self.insertion_upper = global_insertion_upper(self.current_xc, self.current_rc)
        self.done, self.flips_done, self.flip_position = False, False, 0
        self.blocks, self.pruned_upper = None, None
        self.stats = dict.fromkeys(COUNTERS, 0)
        self.stats.update(epoch=epoch, parent_version_id=parent, source='G',
                          remaining_upper_coarse=None, source_complete=False,
                          g_whole_insertion_excluded=False, g_global_insertion_upper_coarse=None,
                          g_pair_blocks_tested=0, g_pair_blocks_expanded=0,
                          g_pair_leaves_generated=0, g_pair_graph_rejects=0,
                          g_pair_chosen_rejects=0, g_pair_processed_rejects=0,
                          g_block_insertions_emitted=0,
                          g_deferred_warm_insertions=getattr(self, 'deferred_warm_insertions', 0))
        self.stats.update({key: 0 for key in self.seed_counts})
        self.version_build_phases = dict(flip_order=0., node_sort=0., block_root=0.)
        if epoch == 1:
            self.flip_order = self.root_flips
            self.root_flips = []
            self.stats.update(self.seed_counts)
            self.stats['flips_emitted'] = self.seed_counts['seed_flips']
            self.stats['seed_build_phases'] = dict(self.seed_build_phases)
            self.version_build_phases.update(self.seed_build_phases)
        else:
            with ledger.phase('frontier_prepare'):
                self.flip_order, elapsed = self._flips(xc, rc)
            self.version_build_phases['flip_order'] = elapsed
        self.stats['flip_sort_items'] = len(self.flip_order)
        self.stats['frontier_rebuilds'] = 1

    def _skipped(self, upper):
        if upper is not None:
            self.pruned_upper = int(upper) if self.pruned_upper is None else max(self.pruned_upper, int(upper))

    def _emit(self, ids):
        batch = self.catalog.batch(ids)
        self.stats['emitted_for_screen'] += len(batch)
        self.stats['insertion_emissions'] += int((~batch.flip).sum())
        self.stats['flips_emitted'] += int(batch.flip.sum())
        self.max_batch_bytes = max(self.max_batch_bytes, batch_bytes(batch))
        return batch

    def next_batch(self, best, excluded):
        if self.done:
            return None
        target = -(-int(best)//RATIO)
        if not self.flips_done:
            ids = []
            while self.flip_position < len(self.flip_order) and len(ids) < 256:
                negcap, e = self.flip_order[self.flip_position]
                cap = -negcap
                if cap < target or (best == 0 and cap <= 0):
                    self._skipped(cap)
                    self.flips_done = True
                    break
                self.flip_position += 1
                self.stats['flip_cursor_pops'] += 1
                self.stats['total_source_pops'] += 1
                if e in self.chosen:
                    self.stats['committed_rejects'] += 1
                elif e in excluded:
                    self.stats['processed_flip_skips'] += 1
                else:
                    ids.append(e)
            if self.flip_position == len(self.flip_order):
                self.flips_done = True
            if ids:
                return self._emit(np.asarray(ids, np.int64))
        # Every flip is now scored, warm/committed, or dominated by a current cap.
        upper = self.insertion_upper
        self.stats['g_global_insertion_upper_coarse'] = upper
        if upper < target or (best == 0 and upper <= 0):
            self.stats['g_whole_insertion_excluded'] = True
            self._skipped(upper)
            self.done = True
            return None
        if self.blocks is None:
            with self.ledger.phase('frontier_prepare'):
                self.blocks = PairBlocks(self.catalog, self.current_xc, self.current_rc,
                                         self.diagonal, self.chosen)
            self.version_build_phases.update(self.blocks.build_phases)
            self.stats['node_sort_items'] = self.catalog.n
        # One insertion at a time gives the common scorer immediate feedback.
        # This fixed policy is frozen before timing, with its Python cost paid.
        e = self.blocks.pop(max(1, target), excluded)
        if e is None:
            self._skipped(self.blocks.upper)
            self.done = True
            return None
        return self._emit(np.asarray([e], np.int64))

    def finish_version(self):
        if self.blocks is not None:
            self.stats.update(self.blocks.counters)
        self.stats.update(source_complete=self.done, remaining_upper_coarse=self.pruned_upper,
                          cold_emitted_for_screen=self.stats['emitted_for_screen'],
                          frontier_build_phases=dict(self.version_build_phases))
        if self.measurement_policy=='legacy_deep':self._observe_memory()
        return dict(self.stats)

    def _observe_memory(self):
        arrays = [self.root_diagonal]
        if hasattr(self, 'diagonal'):
            arrays.extend([self.diagonal, self.current_xc])
        containers = [self.flip_order]
        if self.blocks is not None:
            arrays.append(self.blocks.order)
            containers.extend([self.blocks.heap, self.blocks.nodes])
        payload = sum(a.nbytes for a in arrays)
        seen = set()
        python_bytes = sum(_python_size(v, seen) for v in containers)
        self.frontier_memory = dict(array_payload_bytes=payload, python_structure_bytes=python_bytes,
                                   total_owned_bytes=payload+python_bytes,
                                   g_pair_heap_rows=0 if self.blocks is None else len(self.blocks.heap),
                                   g_node_summary_rows=0 if self.blocks is None else len(self.blocks.nodes),
                                   emitted_history_rows=0,
                                   scope='array/heap/node-summary payload; current center may be shared; excludes process RSS')
        self.max_frontier_bytes = max(self.max_frontier_bytes, payload+python_bytes)

    def memory(self):
        if self.measurement_policy=='legacy_deep':self._observe_memory()
        answer = super().memory()
        answer['access_proof'] = 'flip caps + universal/range insertion bound + independent binary pair-block partition'
        return answer

    def close(self):
        super().close()
        self.blocks = None
        self.flip_order = []
        self.root_flips = []
        self.diagonal = None
        self.chosen = frozenset()
