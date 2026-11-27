"""Targeted checks for a new mathematical probe, not V2 certificates.

The root scheduler must run this file under the frozen micro-check budget.
Writing this file does not constitute an executed check.
"""

from collections import Counter
from dataclasses import replace
from fractions import Fraction as F
import unittest

from block_probe import (
    BlockContext,
    BlockValidationError,
    can_prune,
    check_irregular_graph,
    check_unit_coordinates,
    iter_legal_pairs,
    make_blocks,
    run_empty_two_groups,
    validate_block_catalog,
)


class BlockProbeChecks(unittest.TestCase):
    def setUp(self):
        self.groups = ((0, 3), (1, 4), (2, 5))
        self.context = BlockContext("probe-v1", F(1, 16), "probe-input")
        self.blocks = make_blocks(
            self.groups,
            tuple(map(F, (-1, 0, 1, -1, 0, 1))),
            (F(1),) * 6,
            self.context,
        )

    def test_unordered_partition_and_legal_filter(self):
        existing = {(0, 1), (1, 2), (2, 3), (3, 4), (1, 4), (4, 5)}
        excluded = {(0, 5)}
        seen = Counter(
            pair
            for block in self.blocks
            for pair in iter_legal_pairs(block, self.groups, existing, excluded)
        )
        expected = {
            (i, j)
            for i in range(6)
            for j in range(i + 1, 6)
            if (i, j) not in existing and (i, j) not in excluded
        }
        self.assertEqual(set(seen), expected)
        self.assertTrue(all(count == 1 for count in seen.values()))
        self.assertEqual(len(expected), 8)

    def test_catalog_rejects_omission_duplicate_old_version_wrong_weight(self):
        validate_block_catalog(self.blocks, self.groups, self.context)
        variants = {
            "missing_block": self.blocks[:-1],
            "duplicate_block": self.blocks + (self.blocks[0],),
            "old_version": (
                replace(self.blocks[0], context=replace(self.context, version="probe-v0")),
            ) + self.blocks[1:],
            "wrong_weight": (
                replace(self.blocks[0], context=replace(self.context, weight=F(1, 4))),
            ) + self.blocks[1:],
        }
        for name, blocks in variants.items():
            with self.subTest(name=name), self.assertRaises(BlockValidationError):
                validate_block_catalog(blocks, self.groups, self.context)

    def test_vertex_partition_rejects_duplicate_or_omitted_vertex(self):
        for groups in (((0, 1), (1, 2)), ((0, 1), (3,))):
            with self.subTest(groups=groups), self.assertRaises(BlockValidationError):
                make_blocks(groups, (F(0),) * 4, (F(1),) * 4, self.context)

    def test_equal_bound_requires_canonical_order(self):
        self.assertTrue(can_prune(F(1), F(2), 10, 0))
        self.assertFalse(can_prune(F(2), F(2), 10, 9))
        self.assertTrue(can_prune(F(2), F(2), 10, 10))
        self.assertTrue(can_prune(F(2), F(2), 10, 11))
        self.assertFalse(can_prune(F(3), F(2), 10, 11))

    def test_empty_four_value_mechanism(self):
        for n, pairs in ((8, 28), (32, 496), (128, 8128)):
            with self.subTest(n=n):
                result = run_empty_two_groups(n)
                self.assertEqual(result["status"], "passed")
                self.assertEqual(result["winner"]["pair"], [0, n - 1])
                self.assertEqual(result["winner"]["gain"], "2/9")
                count = result["mechanism_counts"]
                self.assertEqual(count["block_entries_checked"], 10)
                self.assertEqual(count["nonempty_block_bounds_computed"], 8)
                self.assertEqual(count["generated_candidates"], 1)
                self.assertEqual(count["block_minimum_endpoint_pairs_computed"], 16)
                self.assertEqual(count["scoring_endpoint_pairs_computed"], 1)
                self.assertEqual(count["strictly_pruned_nonempty_blocks"], 7)
                self.assertEqual(count["structurally_empty_blocks"], 2)
                self.assertEqual(count["linear_solve_rhs"], 0)
                self.assertEqual(result["checker_counts"]["pair_gain_evaluations"], pairs)
                self.assertEqual(result["checker_counts"]["linear_solve_rhs"], 0)

    def test_all_irregular_legal_nonedges_against_direct_oracle(self):
        result = check_irregular_graph()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["legal_nonedge_count"], 9)
        self.assertEqual(result["checker_counts"]["pair_gain_evaluations"], 9)
        self.assertEqual(result["checker_counts"]["linear_solve_rhs"], 28)
        self.assertTrue(all(record["bound_valid"] for record in result["pairs"]))
        self.assertTrue(all(record["resistance_lower_bound_valid"] for record in result["pairs"]))

    def test_declared_woodbury_units_against_direct_physical_gain(self):
        result = check_unit_coordinates()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["B_inverse_spectral_norm"], "129")
        self.assertEqual(result["barB_inverse_spectral_norm"], "8256")
        self.assertEqual(result["physical_oracle_gain"], "256/129")
        self.assertEqual(result["checker_counts"]["linear_solve_rhs"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
