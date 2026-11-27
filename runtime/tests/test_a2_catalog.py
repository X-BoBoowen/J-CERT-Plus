"""Small independent whole-domain checks for the A2 candidate sources."""
from importlib import import_module
from itertools import combinations
import unittest

import numpy as np
from scipy import sparse


def graph(n, edges=()):
    row, col, data = [], [], []
    for i, j, w in edges:
        row.extend((i, j))
        col.extend((j, i))
        data.extend((w, w))
    return sparse.csr_matrix((data, (row, col)), shape=(n, n))


def independent_rows(W):
    """Deliberately inspect every pair, without execution-catalog helpers."""
    rows = []
    for i, j in combinations(range(W.shape[0]), 2):
        w = float(W[i, j])
        if w <= 0:
            rows.append((i * W.shape[0] + j, i, j, w < 0,
                         int(-16 * w) if w < 0 else 1))
    return rows


class CatalogTests(unittest.TestCase):
    def module(self):
        try:
            module = import_module("src.a2_catalog")
        except ModuleNotFoundError as error:
            if error.name != "src.a2_catalog":
                raise
            module = None
        self.assertIsNotNone(module, "the A2 sparse catalog is not implemented")
        return module

    def fixture(self):
        return graph(5, ((0, 1, -1.5), (0, 3, .5),
                         (1, 4, -2.5), (2, 3, 1.)))

    def test_mixed_domain_batch_scalar_and_dense_positions(self):
        # Catches treating a negative edge as an insertion or losing its weight.
        W = self.fixture()
        cat = self.module().AllCatalog(W)
        want = independent_rows(W)
        self.assertEqual((cat.n, cat.m, cat.insertions), (5, 8, 6))
        self.assertEqual(cat.flip_ids.tolist(), [1, 9])
        self.assertEqual(cat.flip_weights.tolist(), [24, 40])
        self.assertEqual(cat.max_weight, 40)
        ids = [row[0] for row in want]
        batch = cat.batch(ids)
        self.assertIsInstance(batch, self.module().CandidateBatch)
        self.assertEqual(batch.ids.dtype, np.dtype("int64"))
        self.assertEqual(list(zip(batch.ids, batch.i, batch.j, batch.flip,
                                  batch.weights)), want)
        for e, i, j, flip, weight in want:
            scalar = cat.scalar(e)
            self.assertEqual(tuple(v.item() for v in scalar),
                             (i, j, flip, weight))
        reverse_repeated = ids[::-1] + ids[:2]
        self.assertEqual(cat.dense_positions(reverse_repeated).tolist(),
                         list(range(7, -1, -1)) + [0, 1])
        dense_i, dense_j, dense_f = cat.materialize()
        self.assertEqual(dense_i.dtype, np.dtype("int32"))
        self.assertEqual(dense_j.dtype, np.dtype("int32"))
        self.assertEqual(dense_f.dtype, np.dtype("bool"))
        self.assertEqual(list(zip(dense_i, dense_j, dense_f)),
                         [(i, j, f) for _, i, j, f, _ in want])

    def test_bounded_scan_is_exhaustive_with_insert_and_flip_exclusions(self):
        # Catches skipped row boundaries, duplicate pairs, and excluded flips.
        W = self.fixture()
        cat = self.module().AllCatalog(W)
        excluded = {1, 7}
        want = [r for r in independent_rows(W) if r[0] not in excluded]
        for block_size in (1, 2, 3, 7, 65536):
            with self.subTest(block_size=block_size):
                got = []
                for batch in cat.iter_blocks(block_size, excluded=excluded):
                    self.assertLessEqual(len(batch.ids), block_size)
                    self.assertGreater(len(batch.ids), 0)
                    got.extend(zip(batch.ids, batch.i, batch.j,
                                   batch.flip, batch.weights))
                self.assertEqual(got, want)
        self.assertEqual(cat.root_edge_keys, frozenset((1, 3, 9, 13)))

    def test_invalid_identity_is_rejected_without_float_or_int32_truncation(self):
        # Catches accepting ordinal IDs, reversed pairs, positive edges, floats.
        cat = self.module().AllCatalog(self.fixture())
        for bad in (-1, 0, 5, 6, 3, 25, 1.0, True, 2**64):
            with self.subTest(bad=bad):
                with self.assertRaises((ValueError, TypeError, OverflowError)):
                    cat.scalar(bad)
                with self.assertRaises((ValueError, TypeError, OverflowError)):
                    cat.batch([bad])
        self.assertEqual(len(cat.batch([]).ids), 0)
        for bad_size in (0, -1, 1.5, True):
            with self.assertRaises((ValueError, TypeError)):
                list(cat.iter_blocks(bad_size))

    def test_empty_and_dense_complements(self):
        # Catches empty-array indexing and counting root negative edges twice.
        for n, edges in ((0, ()), (1, ()), (4, ()),
                         (4, [(i, j, 1.) for i, j in combinations(range(4), 2)]),
                         (4, [(i, j, -1.5) for i, j in combinations(range(4), 2)])):
            W = graph(n, edges)
            cat = self.module().AllCatalog(W)
            want = independent_rows(W)
            got = [int(e) for batch in cat.iter_blocks(2) for e in batch.ids]
            self.assertEqual(got, [r[0] for r in want])
            self.assertEqual(cat.m, len(want))
            self.assertEqual(len(cat.materialize()[0]), len(want))

    def test_large_identity_and_dense_ordinal_need_int64_without_catalog_storage(self):
        # Only two lookups in a sparse empty graph; no large enumeration/timing.
        n = 65538
        cat = self.module().AllCatalog(graph(n))
        last = (n - 2) * n + n - 1
        batch = cat.batch([1, last])
        self.assertGreater(last, 2**32)
        self.assertEqual(batch.ids.tolist(), [1, last])
        self.assertEqual(batch.i.tolist(), [0, n - 2])
        self.assertEqual(batch.j.tolist(), [1, n - 1])
        self.assertEqual(cat.dense_positions([1, last]).tolist(),
                         [0, n * (n - 1) // 2 - 1])
        self.assertGreater(cat.m, 2**31)
        memory = cat.memory()
        self.assertLess(memory["array_payload_bytes"], 1_000_000)
        self.assertLess(memory["python_structure_bytes"], 100_000)
        self.assertEqual(memory["candidate_array_rows"], 0)

    def test_root_graph_validation_and_snapshot(self):
        # Catches accepting a different graph domain or mutable root bindings.
        module = self.module()
        for W in (sparse.csr_matrix([[0., 1.], [0., 0.]]),
                  sparse.csr_matrix([[1., 0.], [0., 0.]]),
                  graph(2, ((0, 1, .25),)), graph(2, ((0, 1, 11.),)),
                  graph(2, ((0, 1, float("nan")),))):
            with self.assertRaises(ValueError):
                module.AllCatalog(W)
        W = self.fixture()
        cat = module.AllCatalog(W)
        W.data[:] = 1.
        self.assertEqual(cat.scalar(1)[3].item(), 24)
        self.assertEqual(float(cat.W[0, 1]), -1.5)

    def test_stream_counters_cover_actual_work_and_partial_chunks(self):
        # Catches charging a full-domain scan when a consumer stopped early.
        cat = self.module().AllCatalog(self.fixture())
        counts = {}
        blocks = cat.iter_blocks(3, excluded={1, 7}, counters=counts)
        self.assertEqual(next(blocks).ids.tolist(), [2])
        self.assertEqual(counts, dict(stream_pairs_examined=3,
                                     stream_positive_rejects=1,
                                     stream_processed_skips=1,
                                     stream_insertions_emitted=1,
                                     stream_flips_emitted=0))
        list(blocks)
        self.assertEqual(counts, dict(stream_pairs_examined=10,
                                     stream_positive_rejects=2,
                                     stream_processed_skips=2,
                                     stream_insertions_emitted=5,
                                     stream_flips_emitted=1))


class FrontierTests(unittest.TestCase):
    module = CatalogTests.module
    fixture = CatalogTests.fixture

    def frontier(self, cat, x, r, chosen=()):
        cls = getattr(self.module(), "MixedFrontier", None)
        self.assertIsNotNone(cls, "the A2 mixed frontier is not implemented")
        return cls(cat, x, r, chosen=chosen)

    @staticmethod
    def independent_caps(W, x, r):
        caps = {}
        for e, i, j, flip, d in independent_rows(W):
            a, b = int(x[i]), int(x[j])
            caps[e] = d * (-4 * a * b + 4 * r * (abs(a) + abs(b) + r)
                           if flip else (abs(a - b) + 2 * r)**2)
        return caps

    def assert_drain(self, W, x, r, chosen=(), excluded=(), batch_size=1):
        cat = self.module().AllCatalog(W)
        head = self.frontier(cat, np.asarray(x, np.int64), r, chosen)
        caps = self.independent_caps(W, x, r)
        remaining = {e: v for e, v in caps.items()
                     if e not in chosen and e not in excluded}
        seen = set()
        peak_memory = head.memory()["total_owned_bytes"]
        while remaining:
            self.assertIsInstance(head.upper, int)
            self.assertGreaterEqual(head.upper, max(remaining.values()))
            batch = head.pop_batch(batch_size, excluded=excluded)
            self.assertGreater(len(batch.ids), 0)
            self.assertLessEqual(len(batch.ids), batch_size)
            for e in map(int, batch.ids):
                self.assertNotIn(e, seen)
                self.assertIn(e, remaining)
                self.assertEqual(remaining[e], max(remaining.values()))
                remaining.pop(e)
                seen.add(e)
            if remaining:
                self.assertGreaterEqual(head.upper, max(remaining.values()))
            self.assertLessEqual(head.memory()["total_owned_bytes"], peak_memory + 4096)
        self.assertEqual(len(head.pop_batch(3, excluded=excluded).ids), 0)
        self.assertIsNone(head.upper)
        self.assertEqual(seen, set(caps) - set(chosen) - set(excluded))
        self.assertEqual(head.counters["raw_heap_pops"], W.shape[0] * (W.shape[0] - 1) // 2)
        self.assertEqual(head.counters["graph_rejects"], W.nnz // 2)
        self.assertEqual(head.counters["committed_rejects"], len(set(chosen)))
        self.assertEqual(head.counters["processed_skips"], len(set(excluded) - set(chosen)))
        expected_flips = sum(f for e, _, _, f, _ in independent_rows(W) if e in seen)
        skipped_insertions = sum(not f for e, _, _, f, _ in independent_rows(W)
                                 if e in excluded and e not in chosen)
        skipped_flips = len(set(excluded) - set(chosen)) - skipped_insertions
        self.assertEqual(head.counters["flips_emitted"], expected_flips)
        self.assertEqual(head.counters["insertion_emissions"], len(seen) - expected_flips)
        self.assertEqual(head.counters.get("processed_insertion_skips"), skipped_insertions)
        self.assertEqual(head.counters.get("processed_flip_skips"), skipped_flips)
        self.assertEqual(head.counters.get("legal_insertion_pops"),
                         len(seen) - expected_flips + skipped_insertions)
        self.assertEqual(head.counters["node_sort_items"], W.shape[0])
        self.assertEqual(head.counters["frontier_resets"], 1)

    def test_every_remaining_head_covers_mixed_catalog_after_each_pop(self):
        # Catches stale row heads, omitted weighted flips, and cap overflow.
        for x, r in (([-3, 2, 2, 0, 1], 0), ([-3, 2, 2, 0, 1], 3),
                     ([0] * 5, 0), ([2] * 5, 1)):
            for batch_size in (1, 3, 256):
                with self.subTest(x=x, r=r, batch_size=batch_size):
                    self.assert_drain(self.fixture(), x, r, batch_size=batch_size)

    def test_chosen_and_processed_insertions_and_flips_are_skipped_once(self):
        # Catches generating a root negative edge both as insert and as flip.
        self.assert_drain(self.fixture(), [-2, 2, 0, 1, -1], 1,
                          chosen={1, 7}, excluded={2, 9}, batch_size=2)

    def test_repeated_center_ties_stay_competitive_at_threshold(self):
        # Catches '<=' stopping that loses an equal positive canonical rival.
        cat = self.module().AllCatalog(graph(4))
        head = self.frontier(cat, [-1, 1, -1, 1], 0)
        seen = []
        while True:
            batch = head.pop_batch(1, threshold=4)
            if not len(batch.ids):
                break
            seen.extend(map(int, batch.ids))
        self.assertEqual(set(seen), {1, 3, 6, 11})
        self.assertEqual(len(seen), 4)
        self.assertEqual(head.upper, 0)
        self.assertEqual(set(map(int, head.pop_batch(10, threshold=0).ids)), {2, 7})
        self.assertIsNone(head.upper)

    def test_negative_flip_head_preserves_exact_signed_integer_cap(self):
        # Catches zero-clamping flip caps or consuming below-threshold rows.
        cat = self.module().AllCatalog(graph(2, ((0, 1, -1.5),)))
        head = self.frontier(cat, [5, 4], 0)
        self.assertEqual(head.upper, -1920)
        self.assertEqual(len(head.pop_batch(10, threshold=0)), 0)
        self.assertEqual(head.upper, -1920)
        self.assertEqual(head.pop_batch(1, threshold=-1920).ids.tolist(), [1])
        self.assertIsNone(head.upper)

    def test_empty_dense_and_all_chosen_domains(self):
        # Catches advertising a nonempty head after all legal rows are gone.
        for n in (0, 1, 4):
            for weight in (1., -1.5):
                W = graph(n, [(i, j, weight) for i, j in combinations(range(n), 2)])
                self.assert_drain(W, [1] * n, 0)
        cat = self.module().AllCatalog(self.fixture())
        chosen = [row[0] for row in independent_rows(self.fixture())]
        head = self.frontier(cat, [1] * 5, 0, chosen)
        self.assertIsNone(head.upper)
        self.assertEqual(len(head.pop_batch(1)), 0)

    def test_caps_exceeding_int64_and_source_reset_are_exact(self):
        # Catches NumPy arithmetic wrapping and reuse of an old center order.
        cat = self.module().AllCatalog(graph(3))
        x = np.array([2**60, -(2**60), 0], np.int64)
        first = self.frontier(cat, x, 2**59)
        expected = (2**61 + 2**60)**2
        self.assertEqual(first.upper, expected)
        self.assertEqual(first.pop_batch(1).ids.tolist(), [1])
        x[:] = 0
        self.assertGreater(first.upper, 0)
        second = self.frontier(cat, x, 0)
        self.assertEqual(second.upper, 0)
        self.assertEqual(len(second.pop_batch(10)), 3)
        self.assertIsNone(second.upper)

    def test_small_independent_graph_family_and_frontier_input_validation(self):
        # Catches errors at arbitrary sorted row/end-point permutations.
        rng = np.random.default_rng(20261125)
        for n in range(2, 8):
            edges = [(i, j, w) for i, j in combinations(range(n), 2)
                     if (w := rng.choice([0., 0., .5, -1.5, -3.])) != 0]
            self.assert_drain(graph(n, edges), rng.integers(-3, 4, n), 1)
        cat = self.module().AllCatalog(self.fixture())
        for x, r, chosen in (([0] * 4, 0, ()), ([0.] * 5, 0, ()),
                             ([0] * 5, -1, ()), ([0] * 5, 0, (3,))):
            with self.assertRaises((ValueError, TypeError)):
                self.frontier(cat, x, r, chosen)
        head = self.frontier(cat, [0] * 5, 0)
        for bad in (0, -1, 1.5, True):
            with self.assertRaises((ValueError, TypeError)):
                head.pop_batch(bad)


if __name__ == "__main__":
    unittest.main()
