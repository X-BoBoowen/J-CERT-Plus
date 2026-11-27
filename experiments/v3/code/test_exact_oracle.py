"""Hand-specified physical-unit checks; run only through the budget scheduler."""

import unittest
from fractions import Fraction as F

from exact_oracle import (
    Edit,
    canonical_reference_decision,
    edited_edges,
    evaluate_catalog,
    gain,
    legal_edits,
    objective,
    physical_operator,
    solve_rational,
)


PATH = ((0, 1, -1), (1, 2, 1), (2, 3, -1))
OPINIONS = (1, -1, -1, 1)
INSERTION_WEIGHT = F(1, 16)


class PhysicalOracleTests(unittest.TestCase):
    def test_signed_degree_uses_absolute_physical_weights(self):
        self.assertEqual(physical_operator(2, ((0, 1, -64),)),
                         ((F(65), F(64)), (F(64), F(65))))
        self.assertEqual(physical_operator(2, ((1, 0, 64),)),
                         ((F(65), F(-64)), (F(-64), F(65))))
        self.assertEqual(objective(4, PATH, OPINIONS), F(4))

    def test_rational_solver_pivots_without_floating_point(self):
        self.assertEqual(solve_rational(((0, 2), (1, 1)), (1, 1)),
                         (F(1, 2), F(1, 2)))
        with self.assertRaises(ValueError):
            solve_rational(((1, 1), (2, 2)), (1, 2))

    def test_complete_path_catalog_and_exact_tie_choose_smallest_id(self):
        rows = evaluate_catalog(4, PATH, OPINIONS, INSERTION_WEIGHT)
        self.assertEqual([r.edit.canonical_id for r in rows], [1, 2, 3, 7, 11])
        self.assertEqual(
            {(r.edit.kind, r.edit.i, r.edit.j): r.gain for r in rows},
            {("flip", 0, 1): F(32, 21),
             ("insert", 0, 2): F(28, 121),
             ("insert", 0, 3): F(0),
             ("insert", 1, 3): F(28, 121),
             ("flip", 2, 3): F(32, 21)},
        )
        decision = canonical_reference_decision(rows)
        self.assertEqual(decision["status"], "accept")
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["canonical_id"], 1)
        self.assertEqual(decision["gain"], F(32, 21))

    def test_perturbed_opinion_is_a_strict_gap_across_all_five_edits(self):
        opinions = (1, -1 + F(1, 65536), -1, 1)
        rows = evaluate_catalog(4, PATH, opinions, INSERTION_WEIGHT)
        self.assertEqual(len(rows), 5)
        values = {r.edit.canonical_id: r.gain for r in rows}
        self.assertEqual(values[11] - values[1], F(1, 114688))
        self.assertEqual(canonical_reference_decision(rows)["canonical_id"], 11)
        self.assertTrue(all(values[11] > values[i] for i in (1, 2, 3, 7)))

    def test_zero_opinions_stop_without_accepting_any_edit(self):
        rows = evaluate_catalog(4, PATH, (0, 0, 0, 0), INSERTION_WEIGHT)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r.gain == F(0) for r in rows))
        decision = canonical_reference_decision(rows)
        self.assertEqual(decision["status"], "certified_stop")
        self.assertFalse(decision["accepted"])
        self.assertIsNone(decision["canonical_id"])

    def test_weight_64_flip_has_physical_gain_256_over_129(self):
        edges = ((0, 1, -64),)
        edit, = legal_edits(2, edges, INSERTION_WEIGHT)
        self.assertEqual(edit.kind, "flip")
        self.assertEqual(edited_edges(2, edges, edit), ((0, 1, F(64)),))
        self.assertEqual(gain(2, edges, (1, -1), edit), F(256, 129))

    def test_small_insertion_weight_has_physical_gain_two_ninths(self):
        edit, = legal_edits(2, (), INSERTION_WEIGHT)
        self.assertEqual(edit, Edit("insert", 0, 1, F(1, 16), 1))
        self.assertEqual(gain(2, (), (1, -1), edit), F(2, 9))

    def test_invalid_physical_graph_and_opinion_inputs_are_rejected(self):
        invalid_graphs = (
            (0, ()), (True, ()), (2, ((0, 0, 1),)),
            (2, ((0, 2, 1),)), (2, ((0, 1, 0),)),
            (2, ((0, 1, 1), (1, 0, 1))),
            (2, ((0, 1, 0.5),)), (2, ((False, 1, 1),)),
        )
        for n, edges in invalid_graphs:
            with self.subTest(n=n, edges=edges):
                with self.assertRaises((TypeError, ValueError)):
                    physical_operator(n, edges)
        with self.assertRaises(ValueError):
            objective(2, (), (1,))
        with self.assertRaises(TypeError):
            objective(2, (), (1.0, -1))
        for weight in (F(0), F(-1), 0.0625, True):
            with self.subTest(weight=weight):
                with self.assertRaises((TypeError, ValueError)):
                    legal_edits(2, (), weight)
        with self.assertRaises(ValueError):
            solve_rational(((1, 0),), (1,))

    def test_edits_must_be_legal_and_preserve_flip_magnitude(self):
        bad_edits = (
            Edit("insert", 0, 1, F(1, 16), 1),
            Edit("flip", 1, 2, F(1), 6),
            Edit("flip", 0, 1, F(2), 1),
            Edit("flip", 0, 1, F(1), 99),
            Edit("insert", 0, 2, F(-1), 2),
            Edit("delete", 0, 1, F(1), 1),
        )
        for edit in bad_edits:
            with self.subTest(edit=edit):
                with self.assertRaises(ValueError):
                    edited_edges(4, PATH, edit)

    def test_report_preserves_exact_values_and_production_boundary(self):
        from run_micro_checks import build_report

        report = build_report(protocol_path=None)
        self.assertEqual(report["production_path"], "unavailable_missing_v2_code")
        self.assertFalse(report["full_decision_verified"])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["cost"]["rhs_solves"], 22)
        fixtures = {f["name"]: f for f in report["fixtures"]}
        self.assertEqual(len(fixtures), 5)
        self.assertEqual(fixtures["path4_perturbed"]["right_minus_left_flip_gain"],
                         F(1, 114688))
        self.assertEqual(fixtures["empty2_insert_1_16"]["candidates"][0]["gain"],
                         F(2, 9))
        self.assertEqual(fixtures["path4_zero"]["reference_decision"]["status"],
                         "certified_stop")


if __name__ == "__main__":
    unittest.main()
