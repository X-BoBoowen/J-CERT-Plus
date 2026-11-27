"""Exact, bounded mathematical probes; this is not the production baseline G.

There is no V2 caller, H comparison, floating-point enclosure, or service
performance claim.  The empty-graph mechanism uses the analytic A=I shortcut.
Its exhaustive checker has a separate cost and candidate count.  The irregular
graph checker imports only the independently implemented physical oracle.

Run only through the root scheduler under v3/PROTOCOL.json.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable, Sequence


if not __debug__:
    raise RuntimeError("This mathematical checker requires assertions; do not run Python with -O or -OO")


F = Fraction


class BlockValidationError(ValueError):
    """A mathematical probe's partition or context is inconsistent."""


def _exact(value: int | Fraction) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("Only exact int/Fraction values are accepted")
    return F(value)


@dataclass(frozen=True)
class BlockContext:
    version: str
    weight: Fraction
    input_sha256: str

    def __post_init__(self):
        if not self.version or not self.input_sha256 or _exact(self.weight) <= 0:
            raise BlockValidationError("A bound needs a version, input binding, and positive weight")


@dataclass(frozen=True)
class Block:
    left: int
    right: int
    context: BlockContext
    pair_count: int
    min_canonical_id: int | None
    state_difference_upper: Fraction
    resistance_lower: Fraction
    upper: Fraction


def _validate_groups(groups: Sequence[Sequence[int]], n: int) -> None:
    if n < 1 or not groups or any(not group for group in groups):
        raise BlockValidationError("Vertex groups must be nonempty")
    flat = [vertex for group in groups for vertex in group]
    if any(isinstance(vertex, bool) or not isinstance(vertex, int) for vertex in flat):
        raise BlockValidationError("Vertex IDs must be integers")
    if len(flat) != n or set(flat) != set(range(n)):
        raise BlockValidationError("Groups must contain every vertex exactly once")
    if any(tuple(group) != tuple(sorted(group)) for group in groups):
        raise BlockValidationError("Each group must use ascending vertex IDs")


def partition_by_equal_state(q: Sequence[Fraction]) -> tuple[tuple[int, ...], ...]:
    """One logical pass over state entries; deterministic increasing-value order."""
    by_value: dict[Fraction, list[int]] = {}
    for i, value in enumerate(q):
        by_value.setdefault(_exact(value), []).append(i)
    return tuple(tuple(by_value[value]) for value in sorted(by_value))


def _first_pair(groups: Sequence[Sequence[int]], left: int, right: int):
    a, b = groups[left], groups[right]
    if left == right:
        return None if len(a) < 2 else (a[0], a[1])
    i, j = a[0], b[0]
    return (min(i, j), max(i, j))


def _pair_count(groups: Sequence[Sequence[int]], left: int, right: int) -> int:
    a, b = len(groups[left]), len(groups[right])
    return a * (a - 1) // 2 if left == right else a * b


def make_blocks(
    groups: Sequence[Sequence[int]],
    q: Sequence[Fraction],
    diagonal: Sequence[Fraction],
    context: BlockContext,
    radius: Fraction = F(0),
) -> tuple[Block, ...]:
    """Construct triangular group blocks without expanding their pair domains.

    Caller supplies a valid uniform enclosure |z-q|<=radius and the physical
    A diagonal.  This probe uses exact q and radius=0.  The denominator bound
    applies only after filtering to current legal nonedges.
    """
    n = len(q)
    _validate_groups(groups, n)
    if len(diagonal) != n or _exact(radius) < 0:
        raise BlockValidationError("Invalid diagonal length or enclosure radius")
    stats: list[tuple[Fraction, Fraction, Fraction]] = []
    for group in groups:
        values: list[Fraction] = []
        reciprocal_diagonals: list[Fraction] = []
        for vertex in group:
            value = _exact(q[vertex])
            aii = _exact(diagonal[vertex])
            if aii <= 0:
                raise BlockValidationError("The physical diagonal must be positive")
            values.append(value)
            reciprocal_diagonals.append(1 / aii)
        stats.append((min(values), max(values), min(reciprocal_diagonals)))
    blocks = []
    for left in range(len(groups)):
        for right in range(left, len(groups)):
            count = _pair_count(groups, left, right)
            first = _first_pair(groups, left, right)
            lo_i, hi_i, d_i = stats[left]
            lo_j, hi_j, d_j = stats[right]
            delta = max(hi_i - lo_j, hi_j - lo_i) + 2 * radius
            lower = d_i + d_j
            # Singleton diagonal blocks contain no unordered distinct pair.
            upper = context.weight * delta * delta / (1 + context.weight * lower) if count else F(0)
            blocks.append(Block(
                left, right, context, count,
                None if first is None else first[0] * n + first[1],
                delta, lower, upper,
            ))
    return tuple(blocks)


def validate_block_catalog(
    blocks: Sequence[Block],
    groups: Sequence[Sequence[int]],
    expected_context: BlockContext,
) -> None:
    """Validate this probe's structural coverage and context, not a V2 certificate.

    Bound values are constructed internally by make_blocks.  This routine is
    not a parser or verifier for untrusted numerical bound certificates.
    """
    n = sum(map(len, groups))
    _validate_groups(groups, n)
    expected = {(a, b) for a in range(len(groups)) for b in range(a, len(groups))}
    observed = [(block.left, block.right) for block in blocks]
    if len(observed) != len(expected) or set(observed) != expected:
        raise BlockValidationError("Missing or duplicated triangular block")
    for block in blocks:
        if block.context != expected_context:
            raise BlockValidationError("Old version, wrong weight, or wrong input binding")
        if block.pair_count != _pair_count(groups, block.left, block.right):
            raise BlockValidationError("Incorrect structural block cardinality")
        first = _first_pair(groups, block.left, block.right)
        expected_min = None if first is None else first[0] * n + first[1]
        if block.min_canonical_id != expected_min or block.upper < 0:
            raise BlockValidationError("Incorrect canonical minimum or negative upper bound")


def iter_legal_pairs(
    block: Block,
    groups: Sequence[Sequence[int]],
    existing_pairs: Iterable[tuple[int, int]],
    excluded_pairs: Iterable[tuple[int, int]] = (),
):
    """Checker-only enumeration of current nonedges; self pairs are absent."""
    existing = {tuple(sorted(pair)) for pair in existing_pairs}
    excluded = {tuple(sorted(pair)) for pair in excluded_pairs}
    for i in groups[block.left]:
        for j in groups[block.right]:
            if block.left == block.right and i >= j:
                continue
            pair = (min(i, j), max(i, j))
            if pair not in existing and pair not in excluded:
                yield pair


def can_prune(
    upper: Fraction,
    best_gain: Fraction,
    best_canonical_id: int,
    minimum_competitor_id: int,
) -> bool:
    """Exact probe convention: smaller i*n+j wins an exact tie.

    With interval scores, best_gain would need to be a certified lower bound.
    An equality upper==lower is safe only when every competitor's order is no
    earlier than the incumbent.  This routine does not establish positivity.
    """
    return upper < best_gain or (
        upper == best_gain and minimum_competitor_id >= best_canonical_id
    )


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _input_sha256(n, edges, opinions, weight) -> str:
    payload = {
        "n": n,
        "edges": [[i, j, str(w)] for i, j, w in edges],
        "opinions": [str(s) for s in opinions],
        "insertion_weight": str(weight),
        "operator": "I+D-W; D_ii=sum_j abs(W_ij); symmetric W; no loops",
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _base_record(name: str, input_sha256: str) -> dict:
    own_path = Path(__file__).resolve()
    return {
        "source": {"kind": "new_mathematical_fixture", "name": name},
        "code_digest": _sha256(own_path),
        "oracle_code_digest": _sha256(own_path.with_name("exact_oracle.py")),
        "protocol_digest": _sha256(own_path.parents[1] / "PROTOCOL.json"),
        "input_digest": input_sha256,
        "scope": "mechanism-only; not production G; no H comparison; no service speedup",
        "status": "passed",
        "reason": "All assertions in this exact mathematical probe completed",
    }


def _block_record(block: Block) -> dict:
    return {
        "groups": [block.left, block.right],
        "structural_pairs": block.pair_count,
        "minimum_canonical_id": block.min_canonical_id,
        "state_difference_upper": str(block.state_difference_upper),
        "resistance_lower": str(block.resistance_lower),
        "gain_upper": str(block.upper),
    }


def run_empty_two_groups(n: int) -> dict:
    """One scored candidate on the three predeclared empty-graph fixture sizes.

    The mechanism uses analytic z=s and R=2.  A separate exhaustive checker
    subtracts analytic parent/child physical objectives for every nonedge.
    Neither part performs a linear solve; this fact does not generalize to G.
    """
    if n not in (8, 32, 128):
        raise ValueError("Only the three frozen sizes 8, 32, and 128 are enabled")
    start = time.perf_counter()
    m = n // 2
    opinions = (F(-1),) + (F(-63, 64),) * (m - 1) + (F(63, 64),) * (m - 1) + (F(1),)
    weight = F(1, 16)
    diagonal = (F(1),) * n
    # q aliases the input opinions only because the empty physical A is I.
    q = opinions
    groups = partition_by_equal_state(q)
    digest = _input_sha256(n, (), opinions, weight)
    context = BlockContext(f"empty2groups-n{n}-v1", weight, digest)
    blocks = make_blocks(groups, q, diagonal, context)
    validate_block_catalog(blocks, groups, context)
    best_gain: Fraction | None = None
    best_pair: tuple[int, int] | None = None
    best_id: int | None = None
    actions = []
    generated = strict_prunes = tie_prunes = empty_checks = 0
    ordered = sorted(blocks, key=lambda b: (-b.upper, b.min_canonical_id if b.min_canonical_id is not None else n*n, b.left, b.right))
    for block in ordered:
        if block.pair_count == 0:
            empty_checks += 1
            actions.append({"groups": [block.left, block.right], "action": "structurally_empty"})
            continue
        assert block.min_canonical_id is not None
        if best_gain is not None and can_prune(block.upper, best_gain, best_id, block.min_canonical_id):
            strict_prunes += block.upper < best_gain
            tie_prunes += block.upper == best_gain
            actions.append({"groups": [block.left, block.right], "action": "strict_prune" if block.upper < best_gain else "canonical_tie_prune"})
            continue
        pair = _first_pair(groups, block.left, block.right)
        assert pair is not None
        i, j = pair
        value = weight * (q[i] - q[j]) ** 2 / (1 + 2 * weight)
        generated += 1
        cid = i * n + j
        if best_gain is None or value > best_gain or (value == best_gain and cid < best_id):
            best_gain, best_pair, best_id = value, pair, cid
        # A generic graph might need more work here.  This fixture has constant
        # state and diagonal within each block, so this exact closure is valid.
        assert value <= block.upper
        assert can_prune(block.upper, best_gain, best_id, block.min_canonical_id)
        actions.append({"groups": [block.left, block.right], "action": "one_candidate_then_exact_bound_closure"})
    assert best_gain == F(2, 9) and best_pair == (0, n - 1) and best_gain > 0
    coarse_interior = weight * (F(63, 64) - F(-63, 64)) ** 2
    assert coarse_interior == F(3969, 16384) and coarse_interior > best_gain
    mechanism_wall = time.perf_counter() - start

    checker_start = time.perf_counter()
    parent_objective = sum((s * s for s in opinions), F(0))
    block_of_vertex = {vertex: group_id for group_id, vertices in enumerate(groups) for vertex in vertices}
    block_lookup = {(block.left, block.right): block for block in blocks}
    checker_best: tuple[Fraction, int, tuple[int, int]] | None = None
    count = 0
    checked_by_block: Counter[tuple[int, int]] = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = opinions[i], opinions[j]
            # Direct inverse of the child's physical 2x2 block, independently
            # of the mechanism's rank-one gain formula.
            child_pair_objective = ((1 + weight) * (si * si + sj * sj) + 2 * weight * si * sj) / (1 + 2 * weight)
            child_objective = parent_objective - si * si - sj * sj + child_pair_objective
            exact_gain = parent_objective - child_objective
            key = tuple(sorted((block_of_vertex[i], block_of_vertex[j])))
            assert exact_gain <= block_lookup[key].upper
            checked_by_block[key] += 1
            candidate = (exact_gain, i * n + j, (i, j))
            if checker_best is None or candidate[0] > checker_best[0] or (candidate[0] == checker_best[0] and candidate[1] < checker_best[1]):
                checker_best = candidate
            count += 1
    assert count == n * (n - 1) // 2
    assert checker_best == (best_gain, best_id, best_pair)
    assert all(checked_by_block[(block.left, block.right)] == block.pair_count for block in blocks)
    checker_wall = time.perf_counter() - checker_start
    record = _base_record(f"empty2groups_n{n}", digest)
    record.update({
        "n": n,
        "version": context.version,
        "weight": str(weight),
        "fixture": "s=(-1, repeated -63/64, repeated +63/64, +1); each sign group has n/2 vertices",
        "group_sizes": [len(group) for group in groups],
        "logical_legal_nonedge_count": n * (n - 1) // 2,
        "winner": {"pair": list(best_pair), "canonical_id": best_id, "gain": str(best_gain)},
        "coarse_interior_cross_group_bound": str(coarse_interior),
        "coarse_all_insertion_bound": str(4 * weight),
        "input_read_counts": {
            "definition": "Logical scalar entries consumed by named passes, not CPU memory-load instrumentation; fixture construction and output serialization are not counted",
            "state_entries_for_partition": n,
            "state_entries_for_block_summaries": n,
            "diagonal_entries_for_block_summaries": n,
            "opinion_entries_for_input_digest": n,
            "state_entries_for_candidate_evaluation": 2 * generated,
        },
        "mechanism_counts": {
            "block_entries_built": len(blocks),
            "nonempty_block_bounds_computed": sum(block.pair_count > 0 for block in blocks),
            "block_entries_checked": len(actions),
            "structurally_empty_blocks": empty_checks,
            "strictly_pruned_nonempty_blocks": strict_prunes,
            "canonical_tie_pruned_blocks": tie_prunes,
            "generated_candidates": generated,
            "generated_candidates_definition": "submitted for exact gain scoring",
            "block_minimum_endpoint_pairs_computed": 2 * sum(block.pair_count > 0 for block in blocks),
            "block_minimum_endpoint_pairs_definition": "One nonempty-block endpoint minimum in make_blocks and one in validate_block_catalog; repeated metadata computations counted separately",
            "scoring_endpoint_pairs_computed": generated,
            "exact_gain_evaluations": generated,
            "linear_solve_rhs": 0,
            "linear_solve_reason": "Analytic empty-graph A=I, z=s, R=2 shortcut only",
        },
        "checker_counts": {
            "pair_gain_evaluations": count,
            "analytic_child_objective_evaluations": count,
            "bound_comparisons": count,
            "opinion_entries_read": n + 2 * count,
            "linear_solve_rhs": 0,
            "linear_solve_reason": "Analytic inverse of each physical child 2x2 block",
        },
        "blocks": [_block_record(block) for block in blocks],
        "actions": actions,
        "checker_block_pair_counts": [{"groups": list(key), "pairs": value} for key, value in sorted(checked_by_block.items())],
        "cost": {
            "category": "micro_diagnosis_and_checker",
            "mechanism_wall_seconds": mechanism_wall,
            "checker_wall_seconds": checker_wall,
            "complete_service_seconds": None,
            "note": "Micro timings do not measure G, H, or end-to-end service performance",
        },
    })
    return record


def check_irregular_graph() -> dict:
    """Check all nine legal nonedges against independent physical solves."""
    import exact_oracle as oracle

    start = time.perf_counter()
    n = 6
    edges = ((0, 1, F(-1)), (1, 2, F(2)), (2, 3, F(-1, 2)), (3, 4, F(1)), (1, 4, F(-3, 2)), (4, 5, F(3, 4)))
    opinions = (F(1), F(-2, 3), F(1, 5), F(-1), F(2, 7), F(4, 5))
    weight = F(1, 16)
    digest = _input_sha256(n, edges, opinions, weight)
    context = BlockContext("irregular-six-v1", weight, digest)
    matrix = oracle.physical_operator(n, edges)
    q = oracle.solve_rational(matrix, opinions)
    rhs = 1
    assert max(map(abs, q)) <= 1
    groups = ((0, 3), (1, 4), (2, 5))
    blocks = make_blocks(groups, q, tuple(matrix[i][i] for i in range(n)), context)
    validate_block_catalog(blocks, groups, context)
    existing = {(i, j) for i, j, _ in edges}
    expected = {(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in existing}
    coverage = Counter(pair for block in blocks for pair in iter_legal_pairs(block, groups, existing))
    assert set(coverage) == expected and all(value == 1 for value in coverage.values())
    assert len(expected) == 9
    group_of = {vertex: k for k, vertices in enumerate(groups) for vertex in vertices}
    lookup = {(block.left, block.right): block for block in blocks}
    rows = []
    global_upper = min(4 * weight, weight * (max(q) - min(q)) ** 2)
    for i, j in sorted(expected):
        direction = tuple(F(1 if k == i else -1 if k == j else 0) for k in range(n))
        response = oracle.solve_rational(matrix, direction)
        rhs += 1
        resistance = response[i] - response[j]
        diagonal_lower = 1 / matrix[i][i] + 1 / matrix[j][j]
        edit = oracle.Edit("insert", i, j, weight, i * n + j)
        exact_gain = oracle.gain(n, edges, opinions, edit)
        rhs += 2
        key = tuple(sorted((group_of[i], group_of[j])))
        block = lookup[key]
        formula_gain = weight * (q[i] - q[j]) ** 2 / (1 + weight * resistance)
        assert matrix[i][j] == 0
        assert resistance >= diagonal_lower >= block.resistance_lower
        assert exact_gain == formula_gain
        assert exact_gain <= block.upper and exact_gain <= global_upper
        rows.append({
            "pair": [i, j],
            "block": list(key),
            "gain": str(exact_gain),
            "state_difference": str(q[i] - q[j]),
            "resistance": str(resistance),
            "diagonal_resistance_lower": str(diagonal_lower),
            "block_resistance_lower": str(block.resistance_lower),
            "block_gain_upper": str(block.upper),
            "bound_valid": True,
            "resistance_lower_bound_valid": True,
            "rank_one_formula_matches_direct_physical_oracle": True,
        })
    assert rhs == 28
    record = _base_record("irregular_signed_six_vertex_nonedges", digest)
    record.update({
        "version": context.version,
        "n": n,
        "edges": [[i, j, str(w)] for i, j, w in edges],
        "opinions": list(map(str, opinions)),
        "exact_state": list(map(str, q)),
        "weight": str(weight),
        "groups": [list(group) for group in groups],
        "legal_nonedge_count": len(expected),
        "global_insertion_upper": str(global_upper),
        "blocks": [_block_record(block) for block in blocks],
        "pairs": rows,
        "checker_counts": {
            "pair_gain_evaluations": len(rows),
            "block_bound_computations": len(blocks),
            "coverage_pair_enumerations": sum(coverage.values()),
            "state_rhs": 1,
            "resistance_rhs": len(rows),
            "physical_parent_child_gain_rhs": 2 * len(rows),
            "linear_solve_rhs": rhs,
        },
        "cost": {"category": "checker", "wall_seconds": time.perf_counter() - start, "complete_service_seconds": None},
    })
    return record


def check_unit_coordinates() -> dict:
    """Check a declared Woodbury basis, not the unavailable V2 unit pipeline."""
    import exact_oracle as oracle

    start = time.perf_counter()
    weight = F(64)
    edges = ((0, 1, -weight),)
    opinions = (F(1), F(-1))
    matrix = oracle.physical_operator(2, edges)
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    inverse = ((matrix[1][1] / det, -matrix[0][1] / det), (-matrix[1][0] / det, matrix[0][0] / det))
    a, b = (F(1), F(-1)), (F(1), F(1))
    columns = (a, b)
    gram = tuple(tuple(sum((columns[p][i] * inverse[i][j] * columns[q][j] for i in range(2) for j in range(2)), F(0)) for q in range(2)) for p in range(2))
    B = tuple(tuple((F(1) if i == j == 0 else F(-1) if i == j == 1 else F(0)) + weight * gram[i][j] for j in range(2)) for i in range(2))
    barB = tuple(tuple(value / weight for value in row) for row in B)
    assert B[0][1] == B[1][0] == 0 and B == ((F(129), F(0)), (F(0), F(-1, 129)))
    z = tuple(sum((inverse[i][j] * opinions[j] for j in range(2)), F(0)) for i in range(2))
    y = tuple(sum((column[i] * z[i] for i in range(2)), F(0)) for column in columns)
    gain_B = weight * sum((y[i] * y[i] / B[i][i] for i in range(2)), F(0))
    gain_barB = sum((y[i] * y[i] / barB[i][i] for i in range(2)), F(0))
    norm_B_inverse = max(abs(1 / B[i][i]) for i in range(2))
    norm_barB_inverse = max(abs(1 / barB[i][i]) for i in range(2))
    exact_gain = oracle.gain(2, edges, opinions, oracle.Edit("flip", 0, 1, weight, 1))
    assert exact_gain == gain_B == gain_barB == F(256, 129)
    assert norm_B_inverse == 129 and norm_barB_inverse == 8256
    record = _base_record("declared_woodbury_units_weight64", _input_sha256(2, edges, opinions, weight))
    record.update({
        "scope": "mathematical coordinate check against a direct physical oracle; unavailable production normalization and outward rounding remain unverified",
        "definitions": {
            "a": "e_i-e_j", "b": "e_i+e_j", "U": "[a,b]", "J": "diag(1,-1)",
            "update": "A_child=A_parent+weight*U*J*U^T",
            "B": "J+weight*U^T*A_parent^-1*U",
            "barB": "B/weight",
            "y": "U^T*A_parent^-1*s",
            "physical_gain": "weight*y^T*B^-1*y = y^T*barB^-1*y",
            "norm": "spectral 2-norm; these particular matrices are diagonal",
        },
        "weight": str(weight),
        "B": [list(map(str, row)) for row in B],
        "barB": [list(map(str, row)) for row in barB],
        "B_inverse_spectral_norm": str(norm_B_inverse),
        "barB_inverse_spectral_norm": str(norm_barB_inverse),
        "physical_oracle_gain": str(exact_gain),
        "gain_from_B": str(gain_B),
        "gain_from_barB": str(gain_barB),
        "checker_counts": {"physical_parent_child_gain_rhs": 2, "linear_solve_rhs": 2, "analytic_two_by_two_inverse_evaluations": 1},
        "cost": {"category": "checker", "wall_seconds": time.perf_counter() - start, "complete_service_seconds": None},
    })
    return record


def run_all_probes() -> dict:
    empty = [run_empty_two_groups(n) for n in (8, 32, 128)]
    irregular = check_irregular_graph()
    units = check_unit_coordinates()
    input_digests = [row["input_digest"] for row in empty] + [irregular["input_digest"], units["input_digest"]]
    combined_input = hashlib.sha256(json.dumps(input_digests, separators=(",", ":")).encode()).hexdigest()
    record = _base_record("combined_source_limited_mathematical_probes", combined_input)
    record.update({
        "scope": "NOT production G; NOT H comparison; NOT service speedup; NOT V2 certificate rejection tests",
        "input_digest_definition": "SHA256 of the compact JSON array fixture_input_digests, in its recorded order",
        "fixture_input_digests": input_digests,
        "empty_graph": empty,
        "irregular_graph": irregular,
        "unit_coordinates": units,
        "cost": {"category": "micro_diagnosis_and_checker", "linear_solve_rhs": 30, "complete_service_seconds": None},
    })
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(run_all_probes(), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
