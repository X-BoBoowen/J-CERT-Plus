#!/usr/bin/env python3
"""Bounded, independent relation diagnostic for the saved Alpha q1 epoch 4.

This program does not import execution/checker code, run the service, certify a
global winner, or modify V2.  The root scheduler must charge each invocation to
the V3 diagnostic budget.  ``structure`` is the first invocation; ``numeric``
is permitted only after a completed, unclassified structural invocation.

Physical input convention: A = I + D - W, s = opinion_codes / 65536.
All matrix proof work uses Python integers for 16*A.  The optional SciPy solves
produce trial vectors only.  Exact integer energy/residual calculations, not
solver accuracy claims or the production local-gain formula, certify bounds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import permutations
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

SCALE = 16
OPINION_DENOMINATOR = 65536
DOMAIN = "all-nonedges-positive1/16+root-negative-flips:v1"
SALT = "weighted-normalized:scale16:proof48:screen20:allnonedges-v1"
PAIR = (7073654, 12373891)
PRECISION_BITS = (52, 80, 112)
THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


class BindingError(ValueError):
    pass


class CertificateError(ValueError):
    pass


class DiagnosticTimeout(TimeoutError):
    pass


def require(condition, message, error=BindingError):
    if not condition:
        raise error(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(packed(value).encode()).hexdigest()


def rational(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def interval(lo, hi):
    require(lo <= hi, "reversed exact interval", CertificateError)
    return {"lower": rational(lo), "upper": rational(hi)}


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def append_attempt(out, value):
    with (out / "alpha_attempts.jsonl").open("a") as handle:
        handle.write(packed(value) + "\n")


def read_bound_json(workspace, entry):
    path = (workspace / entry["path"]).resolve()
    require(path.is_relative_to(workspace / "v2_received"), "input outside V2 evidence boundary")
    require(sha(path) == entry["sha256"], f"input digest mismatch: {entry['path']}")
    return path, json.loads(path.read_text())


def array_sha(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def load_graph(path, n):
    """Read the original CSR payload, validating exact physical weights."""
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        require(archive["format"].item() == b"csr", "expected original CSR input")
        shape = archive["shape"].tolist()
        require(shape == [n, n], "graph shape")
        indptr = archive["indptr"].copy()
        indices = archive["indices"].copy()
        data = archive["data"].copy()
    require(indptr.dtype.kind in "iu" and indices.dtype.kind in "iu", "CSR integer indices")
    require(len(indptr) == n + 1 and int(indptr[0]) == 0 and
            int(indptr[-1]) == len(data) == len(indices), "CSR pointer dimensions")
    require(all(int(indptr[i]) <= int(indptr[i + 1]) for i in range(n)), "CSR monotonic pointers")
    directed = {}
    for i in range(n):
        for k in range(int(indptr[i]), int(indptr[i + 1])):
            j, raw = int(indices[k]), data[k].item()
            require(0 <= j < n and i != j and (i, j) not in directed, "graph endpoints/duplicates")
            require(type(raw) in (int, float) and math.isfinite(raw), "finite exact graph data")
            value = Fraction(raw) * SCALE
            require(value.denominator == 1 and value != 0, "graph weight not on physical 1/16 lattice")
            directed[i, j] = value.numerator
    require(all(directed.get((j, i)) == weight for (i, j), weight in directed.items()),
            "original graph not exactly symmetric")
    return {(i, j): weight for (i, j), weight in directed.items() if i < j}


def operator(n, edges):
    """Materialize all entries of 16*(I+D-W) directly from physical edges."""
    rows = [{i: SCALE} for i in range(n)]
    for (i, j), weight in edges.items():
        rows[i][i] += abs(weight)
        rows[j][j] += abs(weight)
        rows[i][j] = -weight
        rows[j][i] = -weight
    return rows


def csr_arrays(rows):
    import numpy as np

    indptr, indices, data = [0], [], []
    for row in rows:
        for j, value in sorted(row.items()):
            indices.append(j)
            data.append(value)
        indptr.append(len(data))
    return (np.asarray(indptr, dtype="<i4"), np.asarray(indices, dtype="<i4"),
            np.asarray(data, dtype="<i8"))


def operator_binding(rows):
    indptr, indices, data = csr_arrays(rows)
    hasher = hashlib.sha256()
    for array in (indptr, indices, data):
        hasher.update(array.tobytes())
    return {
        "csr_compatibility_sha256": hasher.hexdigest(),
        "exact_physical_operator_sha256": digest({
            "denominator": SCALE,
            "rows": [[(j, value) for j, value in sorted(row.items())] for row in rows]}),
        "physical_denominator": SCALE,
        "n": len(rows),
        "nnz": len(data),
    }


def candidate(candidate_id, n, root_edges, current_edges, chosen):
    require(type(candidate_id) is int and candidate_id not in chosen, "candidate type/already chosen")
    i, j = divmod(candidate_id, n)
    require(0 <= i < j < n, "canonical candidate ID")
    original, current = root_edges.get((i, j)), current_edges.get((i, j))
    if original is None:
        require(current is None, "insertion no longer a nonedge")
        kind, weight = "insert", 1
    else:
        require(original < 0 and current == original, "flip no longer a current root negative edge")
        kind, weight = "flip", -original
    return {"candidate_id": candidate_id, "i": i, "j": j, "kind": kind,
            "weight_numerator": weight, "weight_denominator": SCALE}


def child_edges(parent_edges, edit):
    result = parent_edges.copy()
    result[edit["i"], edit["j"]] = edit["weight_numerator"]
    return result


def legacy_edit(edit):
    return {"i": edit["i"], "j": edit["j"], "d": edit["weight_numerator"],
            "flip": edit["kind"] == "flip"}


def recover(workspace, protocol, out):
    """Rebuild parent and catalog from graph + accepted edits, never a saved hash."""
    import numpy as np

    _, manifest = read_bound_json(workspace, protocol["input_manifest"])
    graph_meta = manifest["graphs"]["Alpha"]
    fixtures = [x for x in manifest["fixtures"]
                if (x["graph"], x["stream"]) == ("Alpha", "signed-neighbor")]
    require(len(fixtures) == 1, "unique registered fixture")
    fixture = fixtures[0]
    graph_path = (workspace / "v2_received/inputs" / graph_meta["weighted_path"]).resolve()
    opinion_path = (workspace / "v2_received/inputs/v2" / fixture["file"]).resolve()
    for path, expected in ((graph_path, graph_meta["weighted_sha256"]),
                           (opinion_path, fixture["npz_sha256"])):
        require(path.is_relative_to(workspace / "v2_received/inputs"), "registered input path escape")
        require(sha(path) == expected, "registered graph/opinion file digest")
    n = graph_meta["n"]
    require(n == protocol["expected_n"], "registered graph vertex count")
    root_edges = load_graph(graph_path, n)
    positives = sum(weight > 0 for weight in root_edges.values())
    negatives = sum(weight < 0 for weight in root_edges.values())
    require((positives, negatives, n * (n - 1) // 2 - len(root_edges)) ==
            (graph_meta["root_positive_edges"], graph_meta["root_negative_edges"],
             graph_meta["root_nonedges"]), "original physical catalog counts")
    with np.load(opinion_path, allow_pickle=False) as archive:
        matrix = archive["opinions"].copy()
    require(matrix.dtype == np.dtype("<i8") and list(matrix.shape) == fixture["shape"] and
            matrix.flags.c_contiguous, "exact opinion layout")
    require(array_sha(matrix) == fixture["matrix_sha256"] and
            [array_sha(row) for row in matrix] == fixture["opinion_hashes"], "exact opinion matrix/row digest")
    s = [int(value) for value in matrix[0]]
    require(len(s) == n and all(-OPINION_DENOMINATOR <= value <= OPINION_DENOMINATOR for value in s),
            "physical opinion range")
    opinion_hash = array_sha(matrix[0])
    records, contexts = [], []
    for entry in protocol["raw_records"]:
        _, raw = read_bound_json(workspace, entry)
        require((raw["graph"], raw["stream"], raw["n"]) == ("Alpha", "signed-neighbor", n), "raw cell identity")
        require(raw["graph_hash"] == graph_meta["weighted_sha256"] and raw["fixture"] == fixture,
                "raw physical input binding")
        require(raw["input_manifest_sha256"] == protocol["input_manifest"]["sha256"], "raw manifest binding")
        requests = [row for row in raw["requests"] if row["query_position"] == 1]
        require(len(requests) == 1, "one q1 per raw worker")
        request = requests[0]
        require(request["opinion_hash"] == opinion_hash and request["status"] == "unresolved" and
                request["commits"] == 3 and request["delta"] == 0 and request["budget"] == 15 and
                request["method"] == raw["method"], "q1 opinion/status/contract binding")
        decisions = request["decisions"]
        require(len(decisions) == 4 and [x["epoch"] for x in decisions] == [1, 2, 3, 4], "saved prefix length")
        require(all(x["status"] == "accepted" for x in decisions[:3]) and
                decisions[3]["status"] == "unresolved" and decisions[3]["winner_id"] is None,
                "preserve historical unresolved status")
        records.append((raw, request, entry))
        contexts.append({"raw_file": entry, "method": raw["method"], "repeat_id": raw["repeat_id"],
                         "unresolved_decision": decisions[3],
                         "initial_seed_ids": request.get("initial_seed_ids"),
                         "initial_seed_order": request.get("initial_seed_order"),
                         "access_epoch4": [x for x in request.get("access", {}).get("epochs", [])
                                           if x.get("epoch") == 4],
                         "saved_gain_denominator": request.get("gain_denominator")})
    require(len(records) == 9 and {(raw["method"], raw["repeat_id"]) for raw, _, _ in records} ==
            {(method, repeat) for method in ("D", "S", "H") for repeat in range(3)}, "nine raw repetitions")
    write_json(out / "alpha_candidate_context.json", {
        "scope": "All saved terminal comparison/access/seed evidence; no shortlist is assumed complete.",
        "records": contexts})
    root_rows = operator(n, root_edges)
    root_binding = operator_binding(root_rows)
    catalog_hash = hashlib.sha256(f"{DOMAIN}|{graph_meta['weighted_sha256']}".encode()).hexdigest()
    root_version = hashlib.sha256("/".join((graph_meta["weighted_sha256"],
        root_binding["csr_compatibility_sha256"], catalog_hash, SALT)).encode()).hexdigest()
    current_edges, chosen, current_version = root_edges.copy(), set(), root_version
    prefix, reconstructed = records[0][1]["decisions"][:3], []
    for epoch, row in enumerate(prefix, 1):
        edit = candidate(row["winner_id"], n, root_edges, current_edges, chosen)
        require(type(row["flip"]) in (int, bool) and row["flip"] in (0, 1) and
                type(row["d"]) is int and row["i"] == edit["i"] and row["j"] == edit["j"] and
                bool(row["flip"]) == (edit["kind"] == "flip") and row["d"] == edit["weight_numerator"],
                "accepted edit endpoints/physical weight")
        require(row["parent_version_id"] == current_version and row["root_version_id"] == root_version,
                "accepted prefix parent binding")
        current_edges = child_edges(current_edges, edit)
        chosen.add(edit["candidate_id"])
        next_version = hashlib.sha256(f"{current_version}/{edit['candidate_id']}".encode()).hexdigest()
        require(row["current_version_id"] == next_version, "accepted prefix child version")
        reconstructed.append({"epoch": epoch, "edit": edit, "parent_version_id": current_version,
                              "child_version_id": next_version})
        current_version = next_version
    parent_rows = operator(n, current_edges)
    parent_binding = operator_binding(parent_rows)
    observed_ids, declarations = set(), {}
    signature_fields = ("epoch", "winner_id", "i", "j", "flip", "d", "root_version_id", "parent_version_id", "current_version_id")
    prefix_signature = [{key: row[key] for key in signature_fields} for row in prefix]
    for raw, request, entry in records:
        require(raw["operator_hash"] == root_binding["csr_compatibility_sha256"] and
                raw["root_version_id"] == root_version and raw["catalog_hash"] == catalog_hash,
                "independently rebuilt root does not match record")
        require(request["root_version_id"] == root_version and request["current_version_id"] == current_version,
                "q1 parent/root binding")
        require([{key: row[key] for key in signature_fields} for row in request["decisions"][:3]] ==
                prefix_signature, "nine records differ in accepted prefix")
        terminal = request["decisions"][3]
        require(terminal["root_version_id"] == root_version and terminal["parent_version_id"] == current_version and
                terminal["current_version_id"] == current_version,
                "fourth-step physical parent binding")
        comparison = terminal["comparison_resolution"]
        require(comparison["method"] == "structural_node_transposition" and
                comparison["parent_version_id"] == current_version and
                comparison["provisional_winner_id"] == PAIR[0] and PAIR[1] in comparison["unresolved_ids"],
                "observed comparison pair binding")
        require(all(type(e) is int for e in comparison["overlap_ids"] + comparison["unresolved_ids"]) and
                len(set(comparison["overlap_ids"])) == len(comparison["overlap_ids"]) and
                PAIR[0] not in comparison["overlap_ids"] and
                set(comparison["unresolved_ids"]) <= set(comparison["overlap_ids"]),
                "saved observed competitor identifiers")
        observed_ids.update([comparison["provisional_winner_id"], *comparison["overlap_ids"],
                             *comparison["unresolved_ids"]])
        for witness in comparison["witnesses"]:
            key = digest(witness)
            declarations.setdefault(key, {"declaration": witness, "raw_sources": []})["raw_sources"].append(entry["path"])
            observed_ids.update((witness["source_candidate_id"], witness["target_candidate_id"]))
    edits = {e: candidate(e, n, root_edges, current_edges, chosen) for e in sorted(observed_ids)}
    live_count = n * (n - 1) // 2 - positives - len(chosen)
    require(graph_meta["catalog_size"] == live_count + len(chosen) and
            all(raw["catalog_size"] == live_count + len(chosen) for raw, _, _ in records),
            "independently reconstructed legal catalog count")
    binding = {
        "input_manifest": protocol["input_manifest"],
        "graph": {"path": str(graph_path.relative_to(workspace)), "sha256": sha(graph_path)},
        "opinions": {"path": str(opinion_path.relative_to(workspace)), "sha256": sha(opinion_path),
                     "q1_codes_sha256": opinion_hash, "physical_denominator": OPINION_DENOMINATOR},
        "root_operator": root_binding, "parent_operator": parent_binding,
        "root_version_id": root_version, "parent_version_id": current_version,
        "accepted_prefix": reconstructed, "raw_record_count": len(records),
        "parent_reconstruction": "original physical graph plus exactly the three accepted legal edits",
        "catalog": {"representation": "all canonical unordered pairs with exact root/current legality predicate",
                    "root_positive_edges": positives, "root_negative_edges": negatives,
                    "root_nonedges": graph_meta["root_nonedges"], "root_legal_count": live_count + 3,
                    "current_legal_count": live_count, "chosen_ids": sorted(chosen),
                    "insertion_weight": rational(Fraction(1, 16)),
                    "all_candidates_numerically_evaluated": False},
        "observed_candidate_edits": list(edits.values()),
        "observed_candidates_are_a_complete_competitor_set": False,
        "physical_spectral_lower_bound": {"value": 1,
            "proof": "A=I+sum_edges |w|*(e_i-sign(w)*e_j)*(e_i-sign(w)*e_j)^T; original symmetry and legal edits checked"},
    }
    write_json(out / "alpha_parent_binding.json", binding)
    return n, s, parent_rows, current_edges, edits, declarations, binding


def conjugacy(left, right, mapping):
    """Directly compare every exact sparse entry of P left P^T and right."""
    if len(left) != len(right) or sum(map(len, left)) != sum(map(len, right)):
        return False
    for i, row in enumerate(left):
        image_row = right[mapping.get(i, i)]
        for j, value in row.items():
            if image_row.get(mapping.get(j, j), 0) != value:
                return False
    return True


def relation(n, s, parent_rows, current_edges, source, target, mapping):
    require(set(mapping) == set(mapping.values()) and all(0 <= v < n for v in mapping),
            "finite support permutation", CertificateError)
    if (source["kind"], source["weight_numerator"]) != (target["kind"], target["weight_numerator"]):
        return "edit_kind_or_weight"
    if {mapping.get(source[k], source[k]) for k in ("i", "j")} != {target["i"], target["j"]}:
        return "edit_endpoint_image"
    if any(s[i] != s[mapping.get(i, i)] for i in range(n)):
        return "exact_opinion_not_invariant"
    if not conjugacy(parent_rows, parent_rows, mapping):
        return "physical_parent_not_invariant"
    source_rows = operator(n, child_edges(current_edges, source))
    target_rows = operator(n, child_edges(current_edges, target))
    if not conjugacy(source_rows, target_rows, mapping):
        return "physical_children_not_conjugate"
    return None


def verify_saved_witnesses(state, out):
    n, s, parent_rows, current_edges, edits, declarations, binding = state
    results = []
    for declaration_sha, record in sorted(declarations.items()):
        witness = record["declaration"]
        require(witness["kind"] == "weighted_node_transposition_v1" and
                witness["parent_version_id"] == binding["parent_version_id"] and
                witness["operator_sha256"] == binding["parent_operator"]["csr_compatibility_sha256"] and
                witness["opinion_sha256"] == binding["opinions"]["q1_codes_sha256"] and
                witness["operator_scale"] == SCALE and witness["opinion_denominator"] == OPINION_DENOMINATOR and
                witness["score_unit"] == 2 ** 48, "saved witness binding/units", CertificateError)
        source, target = edits[witness["source_candidate_id"]], edits[witness["target_candidate_id"]]
        require(witness["source_edit"] == legacy_edit(source) and witness["target_edit"] == legacy_edit(target),
                "saved witness edit binding", CertificateError)
        swap = witness["swap"]
        require(len(swap) == 2 and all(type(v) is int for v in swap) and 0 <= swap[0] < swap[1] < n,
                "saved swap declaration", CertificateError)
        failure = relation(n, s, parent_rows, current_edges, source, target, {swap[0]: swap[1], swap[1]: swap[0]})
        result = {"declaration_sha256": declaration_sha, **record,
                  "independent_relation_valid": failure is None, "failure": failure}
        results.append(result)
        append_attempt(out, {"stage": "saved_witness_validation", **result})
        require(failure is None, f"invalid saved witness: {failure}", CertificateError)
    write_json(out / "alpha_saved_witness_checks.json", results)
    return results


def structural_search(state, out):
    n, s, parent_rows, current_edges, edits, _, binding = state
    source, target = (edits[e] for e in PAIR)
    support = sorted({source["i"], source["j"], target["i"], target["j"]})
    require(len(support) <= 4, "registered endpoint-only witness support")
    for ordinal, images in enumerate(permutations(support), 1):
        mapping = dict(zip(support, images))
        failure = relation(n, s, parent_rows, current_edges, source, target, mapping)
        record = {"stage": "finite_endpoint_permutation", "ordinal": ordinal,
                  "support": support, "images": list(images), "valid": failure is None,
                  "failure": failure}
        append_attempt(out, record)
        if failure is None:
            proof = {"kind": "exact_physical_parent_opinion_child_permutation_v1",
                     "source_candidate": source, "target_candidate": target,
                     "support": support, "images": list(images),
                     "nonidentity_mapping": {str(k): v for k, v in mapping.items() if k != v},
                     "parent_version_id": binding["parent_version_id"],
                     "parent_operator": binding["parent_operator"],
                     "opinion": binding["opinions"],
                     "source_child_operator": operator_binding(operator(n, child_edges(current_edges, source))),
                     "target_child_operator": operator_binding(operator(n, child_edges(current_edges, target))),
                     "checked_equalities": ["P A_parent P^T = A_parent", "P s = s", "P A_a P^T = A_b"],
                     "proof": "Orthogonal permutation conjugacy and exact opinion invariance give equal child objectives; the common parent objective cancels in g_a-g_b.",
                     "search_ordinal": ordinal, "search_set_size": math.factorial(len(support))}
            write_json(out / "alpha_exact_relation_witness.json", proof)
            return proof
    return None


def quantized_float(value, bits):
    require(math.isfinite(float(value)), "nonfinite trial solve", CertificateError)
    # Python Fraction -> round performs exact ties-to-even quantization.
    return round(Fraction(float(value)) * (1 << bits))


def exact_objective_enclosure(rows, s, trial, bits):
    """For q=trial/2^bits compute F and ||s-Aq||^2 using integers only."""
    unit = 1 << bits
    require(bits >= 16 and len(trial) == len(rows) == len(s), "dyadic trial dimensions")
    aq_numerators = [sum(value * trial[j] for j, value in row.items()) for row in rows]
    residual = [(SCALE * unit // OPINION_DENOMINATOR) * opinion - aq
                for opinion, aq in zip(s, aq_numerators)]
    dot_sq = sum(opinion * q for opinion, q in zip(s, trial))
    q_a_q = sum(q * aq for q, aq in zip(trial, aq_numerators))
    f = Fraction((2 * SCALE * unit // OPINION_DENOMINATOR) * dot_sq - q_a_q,
                 SCALE * unit * unit)
    residual_sq = Fraction(sum(value * value for value in residual), SCALE * SCALE * unit * unit)
    certificate = {"bits": bits, "q_numerators": trial, "q_denominator": unit,
                   "q_sha256": digest({"numerators": trial, "denominator": unit}),
                   "F": rational(f), "squared_residual_norm": rational(residual_sq),
                   "objective_interval": interval(f, f + residual_sq),
                   "certificate": "J=F+r^T A^-1 r; 0<=r^T A^-1 r<=||r||_2^2 from A>=I; all F,r arithmetic exact"}
    return f, f + residual_sq, residual, certificate


def numeric_relation(state, out, report):
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import splu

    n, s, _, current_edges, edits, _, _ = state
    children = {e: operator(n, child_edges(current_edges, edits[e])) for e in PAIR}
    factors, trials, previous_bits, residuals = {}, {}, {}, {}
    for e in PAIR:
        indptr, indices, data = csr_arrays(children[e])
        physical_matrix = sparse.csr_matrix((data.astype(np.float64) / SCALE, indices, indptr), shape=(n, n))
        factors[e] = splu(physical_matrix.tocsc())
        initial = factors[e].solve(np.asarray(s, dtype=np.float64) / OPINION_DENOMINATOR)
        trials[e] = [quantized_float(value, PRECISION_BITS[0]) for value in initial]
        previous_bits[e] = PRECISION_BITS[0]
    last_difference = None
    for bits in PRECISION_BITS:
        bounds = {}
        all_exact = True
        for e in PAIR:
            if bits != previous_bits[e]:
                old_bits = previous_bits[e]
                residual_float = np.asarray([float(Fraction(value, SCALE * (1 << old_bits)))
                                             for value in residuals[e]], dtype=np.float64)
                correction = factors[e].solve(residual_float)
                trials[e] = [(q << (bits - old_bits)) + quantized_float(value, bits)
                             for q, value in zip(trials[e], correction)]
                previous_bits[e] = bits
            lo, hi, residuals[e], certificate = exact_objective_enclosure(children[e], s, trials[e], bits)
            all_exact = all_exact and lo == hi
            bounds[e] = (lo, hi)
            write_json(out / f"alpha_trial_{e}_{bits}.json", {"candidate_id": e, **certificate})
        # g_a-g_b=(J_parent-J_a)-(J_parent-J_b)=J_b-J_a.
        low = bounds[PAIR[1]][0] - bounds[PAIR[0]][1]
        high = bounds[PAIR[1]][1] - bounds[PAIR[0]][0]
        last_difference = interval(low, high)
        report.update(difference_interval=last_difference, last_trial_bits=bits)
        write_json(out / "alpha_classification.json", report)
        append_attempt(out, {"stage": "direct_child_energy_enclosure", "trial_bits": bits,
                             "difference_orientation": "g_7073654-g_12373891", "difference": last_difference,
                             "approximate_solver": "separate child physical binary64 LU; untrusted trial generation",
                             "exact_residuals_recomputed": True})
        if low > 0 or high < 0:
            return {"classification": "separated", "difference_interval": last_difference,
                    "larger_gain_candidate_id": PAIR[0] if low > 0 else PAIR[1],
                    "stop_reason": "guaranteed_difference_strictly_excludes_zero", "last_trial_bits": bits}
        if all_exact and low == high == 0:
            return {"classification": "exact_equal", "difference_interval": last_difference,
                    "stop_reason": "both_direct_child_trials_are_exact_solutions_with_equal_objectives",
                    "last_trial_bits": bits}
    return {"classification": "unclassified", "difference_interval": last_difference,
            "stop_reason": "fixed_52_80_112_bit_dyadic_trial_schedule_exhausted", "last_trial_bits": PRECISION_BITS[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("structure", "numeric"), required=True)
    parser.add_argument("--prior-structure-dir")
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    protocol_path = Path(args.protocol).resolve()
    out = Path(args.output_dir).resolve()
    require(out.is_relative_to(workspace / "v3/resumed"), "output must be new V3 resumed evidence")
    require(not out.exists(), "refuse to overwrite a prior diagnostic")
    protocol = json.loads(protocol_path.read_text())
    require(protocol["frozen_before_first_run"] is True and protocol["code_sha256"] == sha(__file__),
            "code/protocol freeze binding")
    require(tuple(protocol["target_pair"]) == PAIR and tuple(protocol["numeric"]["trial_bits"]) == PRECISION_BITS,
            "implemented finite pair/precision schedule differs from freeze")
    prior = None
    if args.stage == "numeric":
        require(args.prior_structure_dir is not None, "numeric stage requires structural attempt")
        prior_path = Path(args.prior_structure_dir).resolve() / "alpha_classification.json"
        require(prior_path.is_relative_to(workspace / "v3/resumed"), "prior result outside V3 evidence")
        prior = json.loads(prior_path.read_text())
        require(prior["stage"] == "structure" and prior["execution_status"] == "completed" and
                prior["classification"] == "unclassified", "second process not justified by prior result")
        require(prior["protocol_sha256"] == sha(protocol_path) and prior["code_sha256"] == sha(__file__),
                "prior run freeze mismatch")
    for key in THREAD_ENV:
        os.environ[key] = "1"
    sys.dont_write_bytecode = True
    limit = protocol["budget"]["per_process_memory_bytes"]
    require(limit == 4 * 1024 ** 3 and protocol["budget"]["per_process_seconds"] == 600,
            "resource freeze mismatch")
    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
    finite = [value for value in (old_soft, old_hard, limit) if value != resource.RLIM_INFINITY]
    effective_limit = min(finite)
    resource.setrlimit(resource.RLIMIT_AS, (effective_limit, effective_limit))
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    out.mkdir(parents=True)
    started = time.monotonic()
    report = {"classification": "unclassified", "stage": args.stage,
              "execution_status": "running", "difference_interval": None,
              "pair": list(PAIR), "difference_orientation": "g_7073654-g_12373891",
              "code_sha256": sha(__file__), "protocol_sha256": sha(protocol_path),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "full_decision_verified": False,
              "full_decision_obligations": {"strict_positive_gain": "not independently checked",
                  "all_current_legal_competitors": "not independently checked",
                  "global_canonical_choice": "not independently checked"},
              "historical_v2_status": "unresolved; not modified",
              "scope": "relation of two current legal candidates only; no global winner/top-two claim",
              "resource_limits": {"wall_seconds": 600, "address_space_bytes": effective_limit, "numeric_threads": 1}}
    if prior is not None:
        report["prior_structure_result_sha256"] = sha(prior_path)
    write_json(out / "alpha_classification.json", report)

    def timed_out(signum, frame):
        raise DiagnosticTimeout("600 second diagnostic wall limit reached")

    signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, 600)
    returncode = 0
    try:
        state = recover(workspace, protocol, out)
        if prior is not None:
            require(prior["parent_physical_sha256"] == state[-1]["parent_operator"]["exact_physical_operator_sha256"],
                    "reconstructed physical parent differs from first process")
        report["parent_physical_sha256"] = state[-1]["parent_operator"]["exact_physical_operator_sha256"]
        report["saved_witnesses_independently_checked"] = len(verify_saved_witnesses(state, out))
        if args.stage == "structure":
            witness = structural_search(state, out)
            if witness is None:
                report.update(classification="unclassified", stop_reason="finite_endpoint_only_permutation_set_exhausted")
            else:
                report.update(classification="exact_equal", difference_interval=interval(Fraction(0), Fraction(0)),
                              stop_reason="independent_exact_parent_opinion_child_conjugacy",
                              witness_file="alpha_exact_relation_witness.json",
                              floating_point_solves_executed=0)
        else:
            report.update(numeric_relation(state, out, report))
        report["execution_status"] = "completed"
    except DiagnosticTimeout as exc:
        report.update(execution_status="timeout", stop_reason=str(exc))
        returncode = 124
    except MemoryError as exc:
        report.update(execution_status="memory_limit_exceeded", stop_reason=str(exc))
        returncode = 125
    except (BindingError, CertificateError) as exc:
        report.update(execution_status="binding_error" if isinstance(exc, BindingError) else "certificate_error",
                      stop_reason=f"{type(exc).__name__}: {exc}")
        returncode = 2
    except Exception as exc:
        report.update(execution_status="runtime_error", stop_reason=f"{type(exc).__name__}: {exc}")
        returncode = 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report["wall_seconds"] = time.monotonic() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "alpha_classification.json", report)
    print(packed({key: report[key] for key in ("classification", "execution_status", "stop_reason", "full_decision_verified", "wall_seconds")}))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
