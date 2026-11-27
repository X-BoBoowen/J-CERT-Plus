"""Independent legal-domain and physical-gain checks for the new block access."""
import importlib.util
import os
from pathlib import Path
from fractions import Fraction as F
from math import floor, ceil
import unittest

import numpy as np
from scipy import sparse

from src.a2_catalog import AllCatalog
from src.a2_session import A2Session, make_ledger
from src.common import DEN, SCALE, UNIT
from src.g_access import GAccess, PairBlocks
from long_promotion import SCREEN

oracle_path = Path(os.environ['V3_ORACLE'])
spec = importlib.util.spec_from_file_location('v3_physical_oracle', oracle_path)
oracle = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = oracle
spec.loader.exec_module(oracle)


def graph(n, edges):
    a = np.zeros((n, n), dtype=float)
    for i, j, w in edges:
        a[i, j] = a[j, i] = float(w)
    return sparse.csr_matrix(a)


def centers(n, edges, s):
    z = oracle.solve_rational(oracle.physical_operator(n, edges), s)
    xc = np.array([floor(v * SCREEN / SCALE) for v in z], dtype=np.int64)
    return z, xc, 1


class GChecks(unittest.TestCase):
    edges = ((0, 1, F(-3, 2)), (1, 3, F(2)), (2, 3, F(-1, 2)),
             (3, 5, F(1)), (0, 4, F(5, 2)), (2, 5, F(-2)))
    opinion = (F(1), F(-3, 4), F(1, 2), F(-1), F(3, 16), F(7, 8))

    def test_pair_tree_partitions_irregular_nonedge_catalog_exactly(self):
        cat = AllCatalog(graph(6, self.edges))
        _, xc, rc = centers(6, self.edges, self.opinion)
        diag = np.rint(16 * (1 + np.abs(cat.W).sum(axis=1).A1)).astype(np.int64)
        blocks = PairBlocks(cat, xc, rc, diag, chosen=())
        got = []
        while True:
            item = blocks.pop(-1, ())
            if item is None:
                break
            got.append(item)
        wanted = [e.canonical_id for e in oracle.legal_edits(6, self.edges, F(1, 16))
                  if e.kind == 'insert']
        self.assertEqual(sorted(got), sorted(wanted))
        self.assertEqual(len(got), len(set(got)))
        self.assertEqual(blocks.counters['g_pair_leaves_generated'], 15)

    def test_every_constructed_block_bound_contains_physical_nonedges(self):
        cat = AllCatalog(graph(6, self.edges))
        _, xc, rc = centers(6, self.edges, self.opinion)
        diag = np.rint(16 * (1 + np.abs(cat.W).sum(axis=1).A1)).astype(np.int64)
        blocks = PairBlocks(cat, xc, rc, diag, chosen=(), audit=True)
        while blocks.pop(-1, ()) is not None:
            pass
        exact = {r.edit.canonical_id: r.gain for r in
                 oracle.evaluate_catalog(6, self.edges, self.opinion, F(1, 16))
                 if r.edit.kind == 'insert'}
        tested = 0
        for entry in blocks.audit_bounds:
            a, b, c, d = entry['ranges']
            for i in blocks.order[a:b]:
                for j in blocks.order[c:d]:
                    if i == j:
                        continue
                    e = int(min(i, j)) * 6 + int(max(i, j))
                    if e in exact:
                        self.assertLessEqual(exact[e], F(entry['upper_coarse'] * SCALE, SCREEN**2))
                        tested += 1
        self.assertGreater(tested, len(exact))

    def test_chosen_and_processed_candidates_are_removed(self):
        cat = AllCatalog(graph(6, self.edges))
        _, xc, rc = centers(6, self.edges, self.opinion)
        all_insert = [e.canonical_id for e in oracle.legal_edits(6, self.edges, F(1, 16))
                      if e.kind == 'insert']
        chosen, processed = {all_insert[0]}, {all_insert[1]}
        diag = np.rint(16 * (1 + np.abs(cat.W).sum(axis=1).A1)).astype(np.int64)
        i, j = divmod(all_insert[0], 6)
        diag[i] += 1; diag[j] += 1
        blocks = PairBlocks(cat, xc, rc, diag, chosen=chosen)
        got = []
        while True:
            item = blocks.pop(-1, processed)
            if item is None:
                break
            got.append(item)
        self.assertEqual(set(got), set(all_insert) - chosen - processed)

    def test_current_diagonal_reconstructed_after_insertion_and_flip(self):
        cat = AllCatalog(graph(6, self.edges))
        ledger = make_ledger(True)
        access = GAccess(cat, 'G', ledger)
        _, xc, rc = centers(6, self.edges, self.opinion)
        access.begin(xc, rc, 64, ledger)
        ins = next(e for e in oracle.legal_edits(6, self.edges, F(1, 16)) if e.kind == 'insert')
        flip = next(e for e in oracle.legal_edits(6, self.edges, F(1, 16)) if e.kind == 'flip')
        chosen = {ins.canonical_id, flip.canonical_id}
        child_edges = oracle.edited_edges(6, self.edges, ins)
        child_edges = oracle.edited_edges(6, child_edges, flip)
        expected = oracle.physical_operator(6, child_edges)
        access.open_version(xc, rc, chosen, chosen, 2, 'child', ledger)
        self.assertEqual(access.diagonal.tolist(), [int(16 * expected[i][i]) for i in range(6)])

    def test_zero_state_stops_without_insertion_frontend(self):
        service = A2Session(graph(4, ((0, 1, -1), (1, 2, 1), (2, 3, -1))), 'G')
        result = service.request(np.zeros(4, dtype=np.int64), budget=1)
        self.assertEqual(result['status'], 'tolerance_stop')
        self.assertEqual(result['commits'], 0)
        self.assertEqual(result['access']['epochs'][0]['g_pair_blocks_tested'], 0)

    def test_budget_zero_and_second_request_reset(self):
        service = A2Session(graph(4, ((0, 1, -1), (1, 2, 1), (2, 3, -1))), 'G')
        s = np.array([DEN, -DEN, -DEN, DEN], np.int64)
        empty = service.request(s, budget=0)
        self.assertEqual(empty['commits'], 0)
        first = service.request(s, budget=1)
        second = service.request(s, budget=1)
        self.assertEqual(first['decisions'][0]['winner_id'], second['decisions'][0]['winner_id'])
        self.assertEqual(first['current_version_id'], second['current_version_id'])
        self.assertEqual(first['initial_gram_count'], 0)
        self.assertEqual(second['initial_gram_count'], 0)

    def test_continuous_g_trace_passes_independent_exact_full_catalog_checker(self):
        from src.a2_checker import A2CatalogReplay, worker_request
        W = graph(6, self.edges)
        s = np.array([int(x*DEN) for x in self.opinion], np.int64)
        service = A2Session(W, 'G')
        result = service.request(s, budget=4)
        replay = worker_request(A2CatalogReplay(W), result, s, service.graph_hash)
        self.assertTrue(replay['full_catalog_decisions_certified'])
        self.assertEqual(result['commits'], 4)
        self.assertEqual(replay['accepts'], 4)
        for row in result['access']['epochs']:
            self.assertTrue(row['source_complete'])

    def test_service_winners_match_independent_exact_catalog(self):
        fixtures = [(4, ((0, 1, -1), (1, 2, 1), (2, 3, -1)), (1, -1, -1, 1)),
                    (4, ((0, 1, -1), (1, 2, 1), (2, 3, -1)), (1, -1 + F(1, DEN), -1, 1)),
                    (6, self.edges, self.opinion),
                    (2, (), (1, -1))]
        for n, edges, s in fixtures:
            with self.subTest(n=n, s=s):
                service = A2Session(graph(n, edges), 'G')
                result = service.request(np.array([int(F(x) * DEN) for x in s], np.int64), budget=1)
                records = oracle.evaluate_catalog(n, edges, s, F(1, 16))
                reference = oracle.canonical_reference_decision(records)
                self.assertEqual(result['decisions'][0]['winner_id'], reference['canonical_id'])
                winner = next(r for r in records if r.edit.canonical_id == reference['canonical_id'])
                decision = result['decisions'][0]
                self.assertLessEqual(F(decision['low'], result['gain_denominator']), winner.gain)
                self.assertLessEqual(winner.gain, F(decision['high'], result['gain_denominator']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
