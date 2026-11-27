"""Tiny independent physical-operator reference, not a production certificate API.

All numerical work uses fractions.Fraction.  Gains come from two directly
constructed operators and exact linear solves, never a local gain formula.
"""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Edit:
    kind: str
    i: int
    j: int
    weight: Fraction
    canonical_id: int


@dataclass(frozen=True)
class GainRecord:
    edit: Edit
    parent_objective: Fraction
    child_objective: Fraction
    gain: Fraction


def _exact(value):
    if type(value) is int or isinstance(value, Fraction):
        return Fraction(value)
    raise TypeError("Only int and Fraction values are accepted; floats are forbidden")


def _check_n(n):
    if type(n) is not int:
        raise TypeError("Vertex count must be an integer")
    if n < 1:
        raise ValueError("Vertex count must be positive")


def _pair(n, i, j):
    if type(i) is not int or type(j) is not int:
        raise TypeError("Vertex indices must be integers")
    if not (0 <= i < n and 0 <= j < n) or i == j:
        raise ValueError("Edges require distinct in-range vertices")
    return (min(i, j), max(i, j))


def _normalize_edges(n, edges):
    _check_n(n)
    weights = {}
    for edge in edges:
        if len(edge) != 3:
            raise ValueError("Each edge must contain i, j, weight")
        i, j = _pair(n, edge[0], edge[1])
        weight = _exact(edge[2])
        if weight == 0:
            raise ValueError("Zero-weight edges must be represented as absent")
        if (i, j) in weights:
            raise ValueError("Duplicate unordered edge")
        weights[i, j] = weight
    return tuple((i, j, w) for (i, j), w in sorted(weights.items()))


def _vector(n, values):
    values = tuple(_exact(value) for value in values)
    if len(values) != n:
        raise ValueError("Vector dimension does not match the operator")
    return values


def physical_operator(n, edges):
    """Return A=I+D-W for symmetric W and D_ii=sum_j abs(W_ij)."""
    edges = _normalize_edges(n, edges)
    matrix = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for i, j, weight in edges:
        matrix[i][i] += abs(weight)
        matrix[j][j] += abs(weight)
        matrix[i][j] -= weight
        matrix[j][i] -= weight
    return tuple(tuple(row) for row in matrix)


def solve_rational(matrix, rhs):
    """Solve one square rational system by Gaussian elimination with row pivots."""
    matrix = tuple(tuple(_exact(value) for value in row) for row in matrix)
    n = len(matrix)
    if n < 1 or any(len(row) != n for row in matrix):
        raise ValueError("Operator must be a nonempty square matrix")
    rhs = _vector(n, rhs)
    augmented = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    for k in range(n):
        pivot = next((i for i in range(k, n) if augmented[i][k] != 0), None)
        if pivot is None:
            raise ValueError("Singular rational system")
        augmented[k], augmented[pivot] = augmented[pivot], augmented[k]
        for i in range(k + 1, n):
            factor = augmented[i][k] / augmented[k][k]
            augmented[i][k] = Fraction(0)
            for j in range(k + 1, n + 1):
                augmented[i][j] -= factor * augmented[k][j]
    solution = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        remainder = sum((augmented[i][j] * solution[j]
                         for j in range(i + 1, n)), Fraction(0))
        solution[i] = (augmented[i][n] - remainder) / augmented[i][i]
    return tuple(solution)


def objective(n, edges, opinions):
    """Return the exact physical objective s^T A^-1 s (alpha fixed to one)."""
    matrix = physical_operator(n, edges)
    opinions = _vector(n, opinions)
    state = solve_rational(matrix, opinions)
    return sum((s * z for s, z in zip(opinions, state)), Fraction(0))


def legal_edits(n, edges, insertion_weight):
    """Enumerate every negative-edge flip and positive nonedge insertion."""
    edges = _normalize_edges(n, edges)
    insertion_weight = _exact(insertion_weight)
    if insertion_weight <= 0:
        raise ValueError("Insertion weight must be strictly positive")
    weights = {(i, j): w for i, j, w in edges}
    edits = []
    for i in range(n):
        for j in range(i + 1, n):
            weight = weights.get((i, j))
            if weight is None:
                edits.append(Edit("insert", i, j, insertion_weight, i * n + j))
            elif weight < 0:
                edits.append(Edit("flip", i, j, -weight, i * n + j))
    return tuple(edits)


def edited_edges(n, edges, edit):
    """Validate a physical edit and return a newly materialized child edge list."""
    edges = _normalize_edges(n, edges)
    if not isinstance(edit, Edit):
        raise TypeError("Expected an Edit")
    i, j = _pair(n, edit.i, edit.j)
    if (edit.i, edit.j) != (i, j):
        raise ValueError("An edit must use canonical endpoint order i < j")
    if type(edit.canonical_id) is not int or edit.canonical_id != i * n + j:
        raise ValueError("Canonical ID must equal i*n+j")
    weight = _exact(edit.weight)
    if weight <= 0:
        raise ValueError("The resulting edge weight must be positive")
    weights = {(a, b): w for a, b, w in edges}
    previous = weights.get((i, j))
    if edit.kind == "insert":
        if previous is not None:
            raise ValueError("Insertion requires a current nonedge")
    elif edit.kind == "flip":
        if previous is None or previous >= 0 or weight != -previous:
            raise ValueError("Flip requires a negative edge and preserves magnitude")
    else:
        raise ValueError("Only insertion and negative-edge flip are legal")
    weights[i, j] = weight
    return tuple((a, b, w) for (a, b), w in sorted(weights.items()))


def gain(n, edges, opinions, edit):
    """Parent objective minus child objective; exactly two direct RHS solves."""
    edges = _normalize_edges(n, edges)
    opinions = _vector(n, opinions)
    child = edited_edges(n, edges, edit)
    return objective(n, edges, opinions) - objective(n, child, opinions)


def evaluate_catalog(n, edges, opinions, insertion_weight):
    """Evaluate the complete reference catalog with one parent plus one solve/edit."""
    edges = _normalize_edges(n, edges)
    opinions = _vector(n, opinions)
    edits = legal_edits(n, edges, insertion_weight)
    parent_value = objective(n, edges, opinions)
    records = []
    for edit in edits:
        child_value = objective(n, edited_edges(n, edges, edit), opinions)
        records.append(GainRecord(edit, parent_value, child_value,
                                  parent_value - child_value))
    return tuple(records)


def canonical_reference_decision(complete_catalog):
    """Choose a positive maximizer from evaluate_catalog, breaking ties by ID.

    This reports a finite mathematical reference decision only.  The caller must
    supply the complete catalog returned by evaluate_catalog, not a shortlist.
    No production versions, numerical intervals, or certificates are checked.
    """
    rows = tuple(complete_catalog)
    winner = min(rows, key=lambda r: (-r.gain, r.edit.canonical_id)) if rows else None
    accepted = winner is not None and winner.gain > 0
    return {
        "scope": "complete_exact_reference_catalog_only",
        "status": "accept" if accepted else "certified_stop",
        "accepted": accepted,
        "canonical_id": winner.edit.canonical_id if accepted else None,
        "kind": winner.edit.kind if accepted else None,
        "endpoints": [winner.edit.i, winner.edit.j] if accepted else None,
        "gain": winner.gain if accepted else None,
        "maximum_catalog_gain": winner.gain if winner is not None else None,
        "catalog_size": len(rows),
    }
