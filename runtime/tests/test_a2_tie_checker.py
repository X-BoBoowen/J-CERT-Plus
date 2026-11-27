"""Independent structural equality checks, including evidence bypass attacks.

Fixtures and certificates below are built from literal integer edits. They do
not import the execution tie resolver or its certificate builder.
"""
import copy
from fractions import Fraction
import hashlib
import unittest

import numpy as np
from scipy import sparse

from src.a2_checker import A2CatalogReplay, worker_request


ROOT = "a" * 64
UNIT = 1 << 48
DENOMINATOR = UNIT ** 2


def _digest(*arrays):
    value = hashlib.sha256()
    for array in arrays:
        value.update(np.ascontiguousarray(array).tobytes())
    return value.hexdigest()


def _operator(W):
    scaled = (16 * W).astype(np.int64)
    return (sparse.diags(16 + np.asarray(abs(scaled).sum(axis=1)).ravel(),
                         dtype=np.int64) - scaled).tocsr()


def _edit(i, j, *, flip=False, d=1):
    return dict(i=i, j=j, flip=flip, d=d)


def _child(A, edit):
    i, j, d = edit["i"], edit["j"], edit["d"]
    values = (0, -2*d, -2*d, 0) if edit["flip"] else (d, -d, -d, d)
    return (A + sparse.coo_matrix((values, ([i, i, j, j], [i, j, i, j])),
                                 shape=A.shape, dtype=np.int64)).tocsr()


def _witness(A, s, source, target, swap, *, parent=ROOT):
    n = A.shape[0]
    return dict(kind="weighted_node_transposition_v1", parent_version_id=parent,
                operator_sha256=_digest(A.indptr, A.indices, A.data),
                opinion_sha256=_digest(s), opinion_denominator=65536,
                operator_scale=16, score_unit=UNIT,
                source_candidate_id=source["i"]*n + source["j"],
                target_candidate_id=target["i"]*n + target["j"],
                source_edit=source, target_edit=target, swap=list(swap))


def _fixture(n=18, *, twins=2, flip=False):
    W = np.zeros((n, n))
    if flip:
        W[0, 1] = W[1, 0] = W[0, 2] = W[2, 0] = -2.5
    else:
        for i in range(1, twins + 1):
            for j in range(i + 1, twins + 1):
                W[i, j] = W[j, i] = 3
    s = np.zeros(n, np.int64)
    s[0], s[1:twins + 1] = 65536, -65536
    return sparse.csr_matrix(W), s


def _row(W, s, *, source_id=1, target_ids=(2,), winner_id=1,
         gain=Fraction(28, 123), flip=False):
    A, n = _operator(W), W.shape[0]
    source = _edit(*divmod(source_id, n), flip=flip, d=40 if flip else 1)
    witnesses = []
    for target_id in target_ids:
        target = _edit(*divmod(target_id, n), flip=flip, d=40 if flip else 1)
        exchanged = sorted(set((source["i"], source["j"])) ^
                           set((target["i"], target["j"])))
        witnesses.append(_witness(A, s, source, target, exchanged))
    scaled = gain * DENOMINATOR / 16
    lower, upper = scaled.numerator // scaled.denominator, -(-scaled.numerator // scaled.denominator)
    resolution = dict(method="structural_node_transposition", parent_version_id=ROOT,
                      provisional_winner_id=source_id, overlap_ids=list(target_ids),
                      witnesses=witnesses, unresolved_ids=[], source_lower=lower,
                      source_upper=upper, canonical_winner_id=winner_id,
                      shared_lower=lower, shared_upper=upper)
    return dict(epoch=1, root_version_id=ROOT, parent_version_id=ROOT,
                current_version_id=hashlib.sha256(f"{ROOT}/{winner_id}".encode()).hexdigest(),
                status="accepted", winner_id=winner_id, low=16*lower, high=16*upper,
                comparison_resolution=resolution)


def _verify(W, s, row, *, exact_limit=12):
    return A2CatalogReplay(W, block_size=7, exact_limit=exact_limit).verify(
        s, [row], 1, gain_denominator=DENOMINATOR, expected_root=ROOT)


def _rational_gain(W, s, i, j, *, flip=False, d=1):
    """Separate dense rational objective for these small, literal fixtures."""
    import sympy as sy
    A = sy.Matrix(_operator(W).toarray().tolist())
    C = A.copy()
    if flip:
        C[i, j] -= 2*d
        C[j, i] -= 2*d
    else:
        C[i, i] += d
        C[j, j] += d
        C[i, j] -= d
        C[j, i] -= d
    v = sy.Matrix([sy.Rational(int(value), 65536) for value in s])
    return Fraction(16*(v.T*(A.inv() - C.inv())*v)[0])


class A2TieCheckerTests(unittest.TestCase):
    def test_large_equal_insertions_certify_without_interval_separation(self):
        W, s = _fixture()
        result = _verify(W, s, _row(W, s))
        self.assertEqual(result["plan"], [1])
        self.assertEqual(result["checked_candidates"], 152)
        self.assertEqual(result["structural_tie_witnesses_verified"], 1)
        self.assertEqual(result["structural_tie_rivals"], 1)
        self.assertTrue(result["full_catalog_decisions_certified"])
        self.assertEqual(result["decision_checks"][0]["published_gain_check"],
                         "independent interval intersection")

    def test_canonical_change_is_checked_independently(self):
        W, s = _fixture()
        result = _verify(W, s, _row(W, s, source_id=2, target_ids=(1,)))
        self.assertEqual(result["plan"], [1])
        self.assertEqual(result["independent_rhs_by_role"]["winner_child"], 1)
        with self.assertRaisesRegex(ValueError, "tie|canonical"):
            _verify(W, s, _row(W, s, source_id=2, target_ids=(1,), winner_id=2))

    def test_small_exact_rationals_agree_for_insertions_and_weighted_flips(self):
        for flip, gain in ((False, Fraction(28, 123)), (True, Fraction(240, 119))):
            with self.subTest(flip=flip):
                W, s = _fixture(3, flip=flip)
                self.assertEqual(_rational_gain(W, s, 0, 1, flip=flip, d=40 if flip else 1), gain)
                self.assertEqual(_rational_gain(W, s, 0, 2, flip=flip, d=40 if flip else 1), gain)
                row = _row(W, s, gain=gain, flip=flip)
                for exact_limit in (0, 12):
                    result = _verify(W, s, row, exact_limit=exact_limit)
                    self.assertEqual(result["plan"], [1])
                    self.assertEqual(result["structural_tie_witnesses_verified"], 1)

    def test_large_weighted_flip_tie_uses_the_declared_integer_coefficient(self):
        W, s = _fixture(flip=True)
        result = _verify(W, s, _row(W, s, gain=Fraction(240, 119), flip=True))
        self.assertEqual(result["plan"], [1])
        self.assertEqual(result["checked_candidates"], 153)
        self.assertEqual(result["structural_tie_witnesses_verified"], 1)

    def test_every_supplied_witness_is_checked_even_on_exact_small_path(self):
        W, s = _fixture(3)
        good = _row(W, s)
        mutations = [
            ("kind", "unproved_float_tie"), ("parent_version_id", "stale"),
            ("operator_sha256", "0"*64), ("opinion_sha256", "0"*64),
            ("opinion_denominator", 65535), ("operator_scale", 1),
            ("score_unit", UNIT//2), ("swap", [1, 1]), ("swap", [2, 1]),
            ("swap", [0, 2]), ("swap", [1, 3]), ("swap", [True, 2]),
            ("source_candidate_id", True), ("target_candidate_id", 5),
            ("source_edit", _edit(0, 1, d=2)),
            ("source_edit", _edit(0, 1, flip=True)),
            ("target_edit", _edit(0, 2, d=2)),
            ("target_edit", dict(i=0, j=2, d=1, flip=0)),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(good)
                bad["comparison_resolution"]["witnesses"][0][key] = value
                with self.assertRaisesRegex(ValueError, "tie"):
                    _verify(W, s, bad)

    def test_exact_opinion_invariance_is_required_even_with_refreshed_hash(self):
        W, s = _fixture(3)
        row = _row(W, s)
        s[2] += 1
        row["comparison_resolution"]["witnesses"][0]["opinion_sha256"] = _digest(s)
        with self.assertRaisesRegex(ValueError, "tie.*opinion"):
            _verify(W, s, row)

    def test_full_parent_conjugacy_is_required_even_with_refreshed_hash(self):
        W, s = _fixture(5)
        row = _row(W, s)
        W = W.tolil()
        W[1, 3] = W[3, 1] = .5
        W = W.tocsr()
        A = _operator(W)
        row["comparison_resolution"]["witnesses"][0]["operator_sha256"] = _digest(
            A.indptr, A.indices, A.data)
        with self.assertRaisesRegex(ValueError, "tie.*parent"):
            _verify(W, s, row)

    def test_full_child_conjugacy_rejects_an_unrelated_legal_target(self):
        W, s = _fixture(5)
        row = _row(W, s)
        proof = row["comparison_resolution"]
        witness = proof["witnesses"][0]
        proof["overlap_ids"] = [3]
        witness["target_candidate_id"] = 3
        witness["target_edit"] = _edit(0, 3)
        with self.assertRaisesRegex(ValueError, "tie.*child"):
            _verify(W, s, row)

    def test_partial_star_does_not_hide_a_stronger_unsupplied_rival(self):
        W, s = _fixture()
        W = W.tolil()
        W[3, 4] = W[4, 3] = 4
        W = W.tocsr()
        s[3:5] = -65536
        small_W, small_s = W[:5, :5].tocsr(), s[:5]
        self.assertEqual(_rational_gain(small_W, small_s, 0, 3), Fraction(18, 79))
        self.assertGreater(Fraction(18, 79), Fraction(28, 123))
        with self.assertRaisesRegex(ValueError, "unresolved rival"):
            _verify(W, s, _row(W, s))

    def test_unsupplied_equal_rival_gets_its_own_independent_proof(self):
        W, s = _fixture(twins=3)
        result = _verify(W, s, _row(W, s, gain=Fraction(20, 87)))
        self.assertEqual(result["checked_candidates"], 150)
        self.assertEqual(result["structural_tie_witnesses_verified"], 1)
        self.assertEqual(result["structural_tie_relations_discovered"], 1)
        self.assertEqual(result["structural_tie_rivals"], 2)
        self.assertEqual(result["decision_checks"][0]["independently_discovered_ties"][0]
                         ["target_candidate_id"], 3)

    def test_canonical_rule_also_covers_unsupplied_equal_rivals(self):
        W, s = _fixture(twins=3)
        row = _row(W, s, source_id=2, target_ids=(3,), winner_id=2, gain=Fraction(20, 87))
        with self.assertRaisesRegex(ValueError, "tie|canonical"):
            _verify(W, s, row)

    def test_transfer_metadata_cannot_invent_a_point_or_ignore_unresolved_ids(self):
        W, s = _fixture(3)
        for key, value in (("shared_upper", 0), ("shared_lower", 0),
                           ("unresolved_ids", [2]), ("canonical_winner_id", 2),
                           ("source_lower", 0), ("parent_version_id", "stale")):
            with self.subTest(key=key):
                row = _row(W, s)
                row["comparison_resolution"][key] = value
                with self.assertRaisesRegex(ValueError, "tie"):
                    _verify(W, s, row)

    def test_transferred_lower_is_not_accepted_as_an_exact_upper(self):
        W, s = _fixture(3)
        row = _row(W, s)
        proof = row["comparison_resolution"]
        proof["shared_upper"] = proof["shared_lower"]
        with self.assertRaisesRegex(ValueError, "tie.*interval"):
            _verify(W, s, row)

    def test_malformed_or_unbound_resolution_is_rejected(self):
        W, s = _fixture(3)
        for resolution in (True, {}, None):
            with self.subTest(resolution=resolution):
                row = _row(W, s)
                row["comparison_resolution"] = resolution
                with self.assertRaisesRegex(ValueError, "tie"):
                    _verify(W, s, row)
        row = _row(W, s)
        for field in ("epoch", "root_version_id", "parent_version_id", "current_version_id"):
            row.pop(field)
        with self.assertRaisesRegex(ValueError, "tie.*parent"):
            A2CatalogReplay(W).verify(s, [row], 1, gain_denominator=DENOMINATOR)

    def test_unresolved_partial_evidence_is_checked_without_certifying_termination(self):
        W, s = _fixture()
        row = _row(W, s)
        row.update(status="unresolved", winner_id=None, current_version_id=ROOT)
        proof = row["comparison_resolution"]
        proof["overlap_ids"].append(3)
        proof["unresolved_ids"] = [3]
        for field in ("canonical_winner_id", "shared_lower", "shared_upper"):
            proof.pop(field)
        result = _verify(W, s, row)
        self.assertFalse(result["termination_certified"])
        self.assertEqual(result["checked_candidates"], 0)
        self.assertEqual(result["structural_tie_witnesses_verified"], 1)
        proof["witnesses"][0]["swap"] = [1, 3]
        with self.assertRaisesRegex(ValueError, "tie"):
            _verify(W, s, row)

    def test_committing_one_edit_invalidates_the_old_parent_symmetry(self):
        W, s = _fixture()
        first = _row(W, s)
        current = first["current_version_id"]
        A = _child(_operator(W), _edit(0, 1))
        source, target = _edit(1, 3), _edit(2, 3)
        source_id, target_id = 21, 39
        witness = _witness(A, s, source, target, (1, 2), parent=current)
        evidence = dict(method="structural_node_transposition", parent_version_id=current,
                        provisional_winner_id=source_id, overlap_ids=[target_id],
                        witnesses=[witness], unresolved_ids=[], source_lower=1, source_upper=2)
        second = dict(epoch=2, root_version_id=ROOT, parent_version_id=current,
                      current_version_id=current, status="unresolved", winner_id=None,
                      comparison_resolution=evidence)
        with self.assertRaisesRegex(ValueError, "tie.*parent"):
            A2CatalogReplay(W).verify(s, [first, second], 2,
                                     gain_denominator=DENOMINATOR, expected_root=ROOT)

    def test_old_witness_version_and_already_chosen_candidate_are_rejected(self):
        W, s = _fixture()
        first = _row(W, s)
        current = first["current_version_id"]
        for refresh_parent in (False, True):
            with self.subTest(refresh_parent=refresh_parent):
                proof = copy.deepcopy(first["comparison_resolution"])
                for field in ("canonical_winner_id", "shared_lower", "shared_upper"):
                    proof.pop(field)
                if refresh_parent:
                    proof["parent_version_id"] = current
                    witness = proof["witnesses"][0]
                    witness["parent_version_id"] = current
                    A = _child(_operator(W), _edit(0, 1))
                    witness["operator_sha256"] = _digest(A.indptr, A.indices, A.data)
                second = dict(epoch=2, root_version_id=ROOT, parent_version_id=current,
                              current_version_id=current, status="unresolved", winner_id=None,
                              comparison_resolution=proof)
                with self.assertRaisesRegex(ValueError, "tie"):
                    A2CatalogReplay(W).verify(s, [first, second], 2,
                                             gain_denominator=DENOMINATOR, expected_root=ROOT)

    def test_exact_equality_never_authorizes_a_zero_gain_acceptance(self):
        W, s = _fixture()
        s[:] = 0
        row = _row(W, s, gain=Fraction(0))
        with self.assertRaisesRegex(ValueError, "positive|tie"):
            _verify(W, s, row)

    def test_actual_large_tie_service_records_replay_in_all_three_arms(self):
        from src.a2_session import A2Session
        W, s = _fixture()
        replay = A2CatalogReplay(W, block_size=7)
        for method in ("D", "S", "H"):
            with self.subTest(method=method):
                service = A2Session(W, method, timers=False)
                request = service.request(s, budget=1)
                self.assertEqual(request["status"], "budget_stop")
                self.assertIn("comparison_resolution", request["decisions"][0])
                result = worker_request(replay, request, s, service.graph_hash)
                self.assertEqual(result["plan"], [1])
                self.assertEqual(result["checked_candidates"], 152)
                self.assertEqual(result["structural_tie_witnesses_verified"], 1)


if __name__ == "__main__":
    unittest.main()
