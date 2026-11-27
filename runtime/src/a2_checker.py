"""Independent bounded full-catalog replay for A2's mixed edit domain.

This module imports only the separate checker primitives. It does not consume
execution candidates, emitted/omitted lists, caps, heaps, Grams or responses.
Every proved decision regenerates all currently legal root pairs from CSR rows.
Declared equality witnesses are independently checked against full integer
parent and child operators; no execution equality helper is imported.
"""
from fractions import Fraction as F
import hashlib
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

from .checker import (B, COARSE, D, S, changed, digest, dyadic_box,
                      exact_energy, need, operator, opinion_hash, response)


DOMAIN = "all-nonedges-positive1/16+root-negative-flips:v1"
SALT = "weighted-normalized:scale16:proof48:screen20:allnonedges-v1"
SCORE_UNIT = 1 << 48
_TIE_COUNTERS = ("structural_tie_witnesses_verified", "structural_tie_relations_discovered",
                 "structural_tie_attempts", "structural_tie_parent_comparisons",
                 "structural_tie_child_comparisons", "structural_tie_rivals")


def _pool(candidate):
    i, j, d, flip = candidate
    return (np.array([i], np.int64), np.array([j], np.int64),
            np.array([d], np.int64), np.array([flip], np.bool_))


def _change(A, candidate):
    return changed(A, _pool(candidate), 0)


def _csr_bytes(A):
    return A.data.nbytes + A.indices.nbytes + A.indptr.nbytes


def _version(parent, e):
    return hashlib.sha256(f"{parent}/{e}".encode()).hexdigest()


class _TieProofs:
    """Per-decision structural proofs reconstructed solely by the checker.

    A witness is data to validate, never a trusted equality flag. Full sparse
    conjugacy is intentionally separate from the execution row-local proof.
    Only actually declared/discovered ties are retained, not a catalog vector.
    """
    def __init__(self, replay, A, s, chosen, parent):
        self.replay, self.A, self.s = replay, A, s
        self.chosen, self.parent = chosen, parent
        self.equal = set()
        self.resolution = None
        self.success = False
        self.work = dict.fromkeys(_TIE_COUNTERS, 0)
        self.discovered = []
        self.temporary_payload_peak = 0
        self._operator_hash = None

    @property
    def operator_hash(self):
        if self._operator_hash is None:
            self._operator_hash = digest(self.A.indptr, self.A.indices, self.A.data)
        return self._operator_hash

    def candidate(self, e):
        need(type(e) is int and e not in self.chosen, "tie candidate legality/chosen")
        try:
            candidate = self.replay.candidate(e)
        except ValueError as exc:
            raise ValueError("tie candidate legality") from exc
        i, j, d, flip = candidate
        off_diagonal = d if flip else 0
        need(self.A[i, j] == off_diagonal and self.A[j, i] == off_diagonal,
             "tie current candidate legality")
        return candidate

    @staticmethod
    def edit(candidate):
        i, j, d, flip = candidate
        return dict(i=i, j=j, d=d, flip=flip)

    def relation_error(self, source, target, swap):
        """Return a reason, or prove P A P^T=A, P s=s, P A_e P^T=A_f."""
        self.work["structural_tie_attempts"] += 1
        if source[2:] != target[2:]:
            return "edit kind/weight"
        u, v = swap
        if self.s[u] != self.s[v]:
            return "opinion invariance"
        permutation = np.arange(self.replay.n, dtype=np.int64)
        permutation[u], permutation[v] = v, u
        parent_permuted = self.A[permutation, :][:, permutation].tocsr()
        self.work["structural_tie_parent_comparisons"] += 1
        self.temporary_payload_peak = max(self.temporary_payload_peak,
                                          permutation.nbytes + _csr_bytes(parent_permuted))
        if (parent_permuted != self.A).nnz:
            return "parent conjugacy"
        del parent_permuted
        source_child, target_child = _change(self.A, source), _change(self.A, target)
        child_permuted = source_child[permutation, :][:, permutation].tocsr()
        self.work["structural_tie_child_comparisons"] += 1
        self.temporary_payload_peak = max(
            self.temporary_payload_peak, permutation.nbytes + _csr_bytes(source_child) +
            _csr_bytes(target_child) + _csr_bytes(child_permuted))
        if (child_permuted != target_child).nnz:
            return "child conjugacy"
        return None

    def validate_witness(self, witness):
        need(isinstance(witness, dict), "tie witness object")
        need(witness.get("kind") == "weighted_node_transposition_v1", "tie witness kind")
        need(self.parent is not None and witness.get("parent_version_id") == self.parent,
             "tie witness parent binding")
        for field, value in (("opinion_denominator", S), ("operator_scale", B),
                             ("score_unit", SCORE_UNIT)):
            need(type(witness.get(field)) is int and witness[field] == value,
                 "tie witness units")
        need(witness.get("operator_sha256") == self.operator_hash, "tie operator hash")
        need(witness.get("opinion_sha256") == digest(self.s), "tie opinion hash")
        source_id, target_id = witness.get("source_candidate_id"), witness.get("target_candidate_id")
        source, target = self.candidate(source_id), self.candidate(target_id)
        need(source_id != target_id, "tie distinct candidates")
        for field, candidate in (("source_edit", source), ("target_edit", target)):
            declared = witness.get(field)
            need(isinstance(declared, dict) and
                 all(type(declared.get(k)) is int for k in ("i", "j", "d")) and
                 type(declared.get("flip")) is bool and declared == self.edit(candidate),
                 "tie edit type/weight/endpoint binding")
        swap = witness.get("swap")
        need(isinstance(swap, list) and len(swap) == 2 and
             all(type(v) is int for v in swap) and 0 <= swap[0] < swap[1] < self.replay.n,
             "tie canonical node transposition")
        error = self.relation_error(source, target, swap)
        need(error is None, f"tie {error}")
        self.work["structural_tie_witnesses_verified"] += 1
        return source_id, target_id

    def read(self, row):
        if "comparison_resolution" not in row:
            return
        evidence = row["comparison_resolution"]
        need(isinstance(evidence, dict) and
             evidence.get("method") == "structural_node_transposition", "tie resolution schema")
        need(self.parent is not None and evidence.get("parent_version_id") == self.parent,
             "tie resolution parent binding")
        source = evidence.get("provisional_winner_id")
        self.candidate(source)
        overlaps, unresolved = evidence.get("overlap_ids"), evidence.get("unresolved_ids")
        for values in (overlaps, unresolved):
            need(isinstance(values, list) and all(type(e) is int for e in values) and
                 len(values) == len(set(values)), "tie rival identities")
        need(bool(overlaps) and source not in overlaps and
             set(unresolved) <= set(overlaps), "tie overlap set")
        for e in overlaps:
            self.candidate(e)
        need(type(evidence.get("source_lower")) is int and
             type(evidence.get("source_upper")) is int and
             evidence["source_lower"] <= evidence["source_upper"], "tie source interval")
        witnesses = evidence.get("witnesses")
        need(isinstance(witnesses, list), "tie witnesses list")
        targets = set()
        for witness in witnesses:
            # Validate even a redundant declaration; no numeric bypass is allowed.
            declared_source, target = self.validate_witness(witness)
            need(declared_source == source and target in overlaps and target not in targets,
                 "tie witness star binding")
            targets.add(target)
        need(set(overlaps) - targets == set(unresolved), "tie unresolved coverage")
        success_fields = ("canonical_winner_id", "shared_lower", "shared_upper")
        success = any(k in evidence for k in success_fields)
        if success:
            need(all(type(evidence.get(k)) is int for k in success_fields) and not unresolved,
                 "tie successful resolution fields")
            need(evidence["canonical_winner_id"] == min({source} | targets),
                 "tie canonical winner")
            need(evidence["source_lower"] > 0 and
                 evidence["shared_lower"] == evidence["source_lower"] and
                 evidence["shared_lower"] <= evidence["shared_upper"] <= evidence["source_upper"],
                 "tie positive shared interval")
        if row["status"] in ("accept", "accepted"):
            need(success and row.get("winner_id") == evidence["canonical_winner_id"],
                 "tie accepted canonical resolution")
        self.equal, self.resolution, self.success = {source} | targets, evidence, success

    def discover(self, source_id, target_id):
        source, target = self.candidate(source_id), self.candidate(target_id)
        if source[2:] != target[2:]:
            return False
        swap = sorted(set(source[:2]) ^ set(target[:2]))
        if len(swap) != 2 or self.relation_error(source, target, swap) is not None:
            return False
        witness = dict(kind="weighted_node_transposition_v1", parent_version_id=self.parent,
                       operator_sha256=self.operator_hash, opinion_sha256=digest(self.s),
                       opinion_denominator=S, operator_scale=B, score_unit=SCORE_UNIT,
                       source_candidate_id=source_id, target_candidate_id=target_id,
                       swap=swap, source_edit=self.edit(source), target_edit=self.edit(target))
        self.discovered.append(witness)
        self.equal.update((source_id, target_id))
        self.work["structural_tie_relations_discovered"] += 1
        return True

    def gain_consistency(self, lower, upper, row, denominator):
        if not self.success:
            return
        evidence = self.resolution
        for prefix in ("source", "shared"):
            low = F(B * evidence[prefix + "_lower"], SCORE_UNIT ** 2)
            high = F(B * evidence[prefix + "_upper"], SCORE_UNIT ** 2)
            need(max(low, lower) <= min(high, upper), "tie gain interval intersection")
        if "low" in row or "high" in row:
            need(denominator is not None and type(row.get("low")) is int and
                 type(row.get("high")) is int and
                 F(row["low"], denominator) == F(B * evidence["shared_lower"], SCORE_UNIT ** 2) and
                 F(row["high"], denominator) == F(B * evidence["shared_upper"], SCORE_UNIT ** 2),
                 "tie published shared interval binding")


class _EndpointCaps:
    """The existing integer variational endpoint cap, cached per parent.

    All divisions below round the subtracted corrections downward. Unlike
    remaining_by_cap, parent vectors and diagonal are prepared only once, then
    applied to bounded blocks. This is checker work, separate from H/S/D.
    """
    def __init__(self, A, s, z, max_weight):
        self.q, self.r, r2, _, _ = dyadic_box(A, s, z, COARSE)
        self.radius = -(-r2 // B)
        self.diag = A.diagonal()
        vmax = int(abs(self.q).max(initial=0))
        need((2 * max_weight * vmax) ** 2 < 2 ** 62 and
             4 * max_weight * vmax ** 2 + self.radius < 2 ** 62,
             "screen range")

    @property
    def nbytes(self):
        return self.q.nbytes + self.r.nbytes + self.diag.nbytes

    def bounds(self, pool):
        i, j, w, flip = pool
        a, b = self.q[i], self.q[j]
        key = w * np.where(flip, -4 * a * b, (a - b) ** 2)
        change = w * abs(a - b)
        pa = np.maximum(0, np.where(flip, 2 * w * abs(b), change) - abs(self.r[i]))
        pb = np.maximum(0, np.where(flip, 2 * w * abs(a), change) - abs(self.r[j]))
        shift = np.where(flip, w, 2 * w)
        return (key + self.radius - pa ** 2 // (self.diag[i] + shift)
                - pb ** 2 // (self.diag[j] + shift))


class A2CatalogReplay:
    def __init__(self, W, *, block_size=65536, exact_limit=12):
        need(type(block_size) is int and block_size > 0 and
             type(exact_limit) is int and 0 <= exact_limit <= 12,
             "checker block/exact limit")
        self.W = W.tocsr(copy=True)
        self.n = self.W.shape[0]
        need(self.n > 0 and self.n * self.n <= np.iinfo(np.int64).max,
             "canonical identity range")
        need(np.all(2 * self.W.data == np.rint(2 * self.W.data)),
             "natural half-integer root weights")
        self.A = operator(self.W)
        self.A.sort_indices()
        self.block_size, self.exact_limit = block_size, exact_limit
        self.root_edges = self.W.nnz // 2
        self.flips = int(np.count_nonzero(self.W.data < 0)) // 2
        self.insertions = self.n * (self.n - 1) // 2 - self.root_edges
        self.m = self.insertions + self.flips
        self.max_weight = max(1, int(round(float(-self.W.data.min(initial=0)) * B)))
        self.operator_hash = digest(self.A.indptr, self.A.indices, self.A.data)
        for a in (self.W.data, self.W.indices, self.W.indptr,
                  self.A.data, self.A.indices, self.A.indptr):
            a.flags.writeable = False

    def root_version_id(self, graph_hash):
        catalog_hash = hashlib.sha256(f"{DOMAIN}|{graph_hash}".encode()).hexdigest()
        return hashlib.sha256("/".join((graph_hash, self.operator_hash,
                                       catalog_hash, SALT)).encode()).hexdigest()

    def candidate(self, e):
        """Decode the canonical pair independently of the execution catalog."""
        need(type(e) is int and 0 <= e < self.n * self.n, "winner identity")
        i, j = divmod(e, self.n)
        need(i < j, "winner identity")
        left, right = self.W.indptr[i:i + 2]
        indices = self.W.indices[left:right]
        k = int(np.searchsorted(indices, j))
        if k == len(indices) or int(indices[k]) != j:
            return i, j, 1, False
        w = self.W.data[int(left) + k]
        need(w < 0, "positive root edge is outside catalog")
        return i, j, int(round(float(-w) * B)), True

    def iter_blocks(self, *, excluded=()):
        """Generate every legal pair in canonical order, bounded by block_size.

        Sparse row intersections classify original edges. O(block_size) arrays
        cover a segment of one row, plus one output block; no complement table,
        candidate-sized vector, dense adjacency or visited bitmap is retained.
        """
        excluded_by_row = {}
        for e in excluded:
            i, j, _, _ = self.candidate(int(e))
            excluded_by_row.setdefault(i, []).append(j)
        size = min(self.block_size, max(1, self.n * (self.n - 1) // 2))

        def buffers():
            return (np.empty(size, np.int64), np.empty(size, np.int64),
                    np.empty(size, np.int64), np.empty(size, np.int64),
                    np.empty(size, np.bool_))

        ids, out_i, out_j, out_w, out_f = buffers()
        fill = 0
        for i in range(self.n - 1):
            left, right = self.W.indptr[i:i + 2]
            indices = self.W.indices[left:right]
            weights = self.W.data[left:right]
            for first in range(i + 1, self.n, size):
                last = min(self.n, first + size)
                js = np.arange(first, last, dtype=np.int64)
                keep = np.ones(len(js), np.bool_)
                ds = np.ones(len(js), np.int64)
                fs = np.zeros(len(js), np.bool_)
                a, b = np.searchsorted(indices, [first, last])
                cols, vals = indices[a:b], weights[a:b]
                offsets = cols.astype(np.int64) - first
                keep[offsets[vals > 0]] = False
                neg = vals < 0
                fs[offsets[neg]] = True
                ds[offsets[neg]] = np.rint(-B * vals[neg]).astype(np.int64)
                for j in excluded_by_row.get(i, ()):
                    if first <= j < last:
                        keep[j - first] = False
                js, ds, fs = js[keep], ds[keep], fs[keep]
                offset = 0
                while offset < len(js):
                    take = min(size - fill, len(js) - offset)
                    dest, source = slice(fill, fill + take), slice(offset, offset + take)
                    ids[dest] = i * self.n + js[source]
                    out_i[dest] = i
                    out_j[dest], out_w[dest], out_f[dest] = js[source], ds[source], fs[source]
                    fill += take
                    offset += take
                    if fill == size:
                        yield ids, (out_i, out_j, out_w, out_f)
                        ids, out_i, out_j, out_w, out_f = buffers()
                        fill = 0
        if fill:
            yield ids[:fill], (out_i[:fill], out_j[:fill], out_w[:fill], out_f[:fill])

    def verify(self, s, decisions, budget, *, delta=0, expected_opinion=None,
               gain_denominator=None, expected_root=None):
        start = perf_counter()
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        s = np.asarray(s)
        delta = F(delta)
        need(s.shape == (self.n,) and s.dtype.kind == "i" and
             np.all((s >= -S) & (s <= S)) and delta >= 0 and
             type(budget) is int and budget >= 0, "request")
        s = s.astype(np.int64, copy=True)
        if expected_opinion is not None:
            need(opinion_hash(s) == expected_opinion, "opinion binding")
        if gain_denominator is not None:
            need(type(gain_denominator) is int and gain_denominator > 0,
                 "published gain denominator")
        rows = list(decisions)
        bindings = ("epoch", "root_version_id", "parent_version_id", "current_version_id")
        bound = expected_root is not None or any(any(k in r for k in bindings) for r in rows)
        if bound and expected_root is None:
            need(bool(rows) and "root_version_id" in rows[0], "missing root binding")
            expected_root = rows[0]["root_version_id"]
        current = expected_root
        A, parent_z = self.A, None
        chosen, chosen_set = [], set()
        checked = refined = pruned = exact_solves = max_rows = 0
        rhs = dict(parent=0, winner_child=0, rival_child=0)
        tie_work = dict.fromkeys(_TIE_COUNTERS, 0)
        known_payload_peak = _csr_bytes(self.W) + _csr_bytes(self.A) + s.nbytes
        root_payload = known_payload_peak
        ended = None
        decision_checks = []

        def solve(matrix, role):
            rhs[role] += 1
            return response(matrix, s)

        def finish_ties(ties, detail=None):
            for key, value in ties.work.items():
                tie_work[key] += value
            if detail is not None:
                detail.update(ties.work)
                if ties.discovered:
                    detail["independently_discovered_ties"] = ties.discovered

        for epoch, row in enumerate(rows, 1):
            need(ended is None, "decision after terminal status")
            need(isinstance(row, dict) and "status" in row, "decision row")
            status = row["status"]
            e = row.get("winner_id")
            if bound:
                need(all(k in row for k in bindings) and row["epoch"] == epoch and
                     row["root_version_id"] == expected_root and
                     row["parent_version_id"] == current, "parent/epoch binding")
            ties = _TieProofs(self, A, s, chosen_set, current)
            ties.read(row)
            tie_parent_payload = root_payload + (0 if A is self.A else _csr_bytes(A))
            known_payload_peak = max(known_payload_peak,
                                     tie_parent_payload + ties.temporary_payload_peak)
            if status == "budget_stop":
                need(e is None and len(chosen) == budget, "premature budget stop")
                if bound:
                    need(row["current_version_id"] == current, "terminal child binding")
                ended = status
                finish_ties(ties)
                continue
            need(len(chosen) < budget, "decision beyond budget")
            if status == "unresolved":
                need(e is None, "unresolved commit")
                if bound:
                    need(row["current_version_id"] == current, "terminal child binding")
                ended = status
                detail = dict(epoch=epoch, status=status, checked_candidates=0, certified=False)
                finish_ties(ties, detail)
                decision_checks.append(detail)
                continue
            need(status in ("accept", "accepted", "tolerance_stop"), "unknown decision")
            accept = status in ("accept", "accepted")
            if accept:
                candidate = self.candidate(e)
                need(e not in chosen_set, "duplicate commit")
                for key, val in zip(("i", "j", "weight", "flip"), candidate):
                    if key in row:
                        need(row[key] == val, f"{key} endpoint binding")
                if "d" in row:
                    need(row["d"] == candidate[2], "weight endpoint binding")
                child = _change(A, candidate)
            else:
                need(e is None, "stop with winner")
                child = None
            next_version = _version(current, e) if bound and accept else current
            if bound:
                need(row["current_version_id"] == next_version, "child binding")
            live = self.m - len(chosen)
            count = local_pruned = local_refined = 0
            candidate_digest = hashlib.sha256()
            rhs_before = sum(rhs.values())
            exact_before = exact_solves
            lower = upper = None
            child_z = None
            if self.n <= self.exact_limit:
                base = exact_energy(A, s)
                exact_solves += 1
                if accept:
                    lower = upper = base - exact_energy(child, s)
                    exact_solves += 1
                    need(lower > 0, "exact winner/tie: nonpositive gain")
                for ids, pool in self.iter_blocks(excluded=chosen_set):
                    count += len(ids)
                    max_rows = max(max_rows, len(ids))
                    candidate_digest.update(ids.astype("<i8", copy=False).tobytes())
                    unique = {id(a): a for a in (A, child) if a is not None and a is not self.A}
                    payload = root_payload + sum(_csr_bytes(a) for a in unique.values())
                    payload += sum(a.nbytes for a in (ids, *pool))
                    known_payload_peak = max(known_payload_peak, payload)
                    for k, rival_id in enumerate(ids):
                        rival_id = int(rival_id)
                        if rival_id == e:
                            gain = lower
                        else:
                            rival = changed(A, pool, k)
                            gain = base - exact_energy(rival, s)
                            exact_solves += 1
                            known_payload_peak = max(known_payload_peak, payload + _csr_bytes(rival))
                        local_refined += 1
                        if accept:
                            need(gain <= lower + delta and
                                 (delta > 0 or rival_id >= e or gain < lower), "exact winner/tie")
                        else:
                            need(gain <= delta, "false tolerance stop")
            else:
                if parent_z is None:
                    parent_z = solve(A, "parent")
                _, _, _, lo, hi = dyadic_box(A, s, parent_z, D)
                if accept:
                    child_z = solve(child, "winner_child")
                    _, _, _, cl, ch = dyadic_box(child, s, child_z, D)
                    lower, upper = lo - ch, hi - cl
                    need(lower > 0, "positive winner unproved")
                    threshold = lower + delta
                else:
                    threshold = delta
                caps = _EndpointCaps(A, s, parent_z, self.max_weight)
                target = threshold * (B * COARSE * COARSE)
                target_floor = target.numerator // target.denominator
                exact_target = target.denominator == 1
                for ids, pool in self.iter_blocks(excluded=chosen_set):
                    count += len(ids)
                    max_rows = max(max_rows, len(ids))
                    candidate_digest.update(ids.astype("<i8", copy=False).tobytes())
                    cap = caps.bounds(pool)
                    if target_floor > np.iinfo(np.int64).max:
                        survive = np.zeros(len(ids), np.bool_)
                    else:
                        survive = cap > target_floor
                        if accept and delta == 0 and exact_target:
                            survive |= (ids < e) & (cap == target_floor)
                    if accept:
                        survive &= ids != e
                    local_pruned += int(np.count_nonzero(~survive & (ids != e)))
                    # Root storage, current/candidate operators, response/state
                    # vectors and block arrays are scoped separately from RSS.
                    matrices = [A] + ([child] if child is not None else [])
                    unique = {id(a): a for a in matrices if a is not self.A}
                    payload = root_payload + sum(_csr_bytes(a) for a in unique.values())
                    payload += caps.nbytes + parent_z.nbytes + (0 if child_z is None else child_z.nbytes)
                    payload += sum(a.nbytes for a in (ids, *pool, cap, survive))
                    known_payload_peak = max(known_payload_peak, payload)
                    for k in np.flatnonzero(survive):
                        rival_id = int(ids[k])
                        if accept and e in ties.equal and rival_id in ties.equal:
                            need(delta > 0 or e < rival_id, "tie canonical winner")
                            ties.work["structural_tie_rivals"] += 1
                            continue
                        rival = changed(A, pool, int(k))
                        rz = solve(rival, "rival_child")
                        _, _, _, rl, _ = dyadic_box(rival, s, rz, D)
                        rival_upper = hi - rl
                        separated = (rival_upper <= threshold and
                                     (not accept or delta > 0 or rival_id > e or rival_upper < threshold))
                        if not separated and accept and ties.discover(e, rival_id):
                            need(delta > 0 or e < rival_id, "tie canonical winner")
                            ties.work["structural_tie_rivals"] += 1
                            separated = True
                        need(separated,
                             "unresolved rival" if accept else "false tolerance stop or unresolved rival")
                        local_refined += 1
                        known_payload_peak = max(known_payload_peak, payload + _csr_bytes(rival) +
                                                 rz.nbytes + ties.temporary_payload_peak)
            need(count == live, "independent catalog coverage count")
            checked += count
            refined += local_refined
            pruned += local_pruned
            detail = dict(epoch=epoch, status="accepted" if accept else status,
                          checked_candidates=count, legal_catalog_size=live,
                          root_pairs_scanned=self.n * (self.n - 1) // 2,
                          candidate_ids_sha256=candidate_digest.hexdigest(),
                          endpoint_pruned_rows=local_pruned, refined_rows=local_refined,
                          independent_rhs=sum(rhs.values()) - rhs_before,
                          exact_energy_solves=exact_solves - exact_before, certified=True)
            finish_ties(ties, detail)
            if accept:
                ties.gain_consistency(lower, upper, row, gain_denominator)
                if "low" in row or "high" in row:
                    need(gain_denominator is not None and
                         type(row.get("low")) is int and type(row.get("high")) is int and
                         row["low"] <= row["high"], "published gain interval")
                    published_low = F(row["low"], gain_denominator)
                    published_high = F(row["high"], gain_denominator)
                    need(max(published_low, lower) <= min(published_high, upper),
                         "published gain intersection")
                    detail["published_gain_check"] = (
                        "exact containment" if lower == upper else "independent interval intersection")
                detail["independent_gain_interval"] = [
                    [lower.numerator, lower.denominator], [upper.numerator, upper.denominator]]
                chosen.append(e)
                chosen_set.add(e)
                A, parent_z, current = child, child_z, next_version
            else:
                ended = "tolerance_stop"
            decision_checks.append(detail)
        if ended is None:
            need(len(chosen) == budget, "missing terminal decision")
            ended = "budget_stop"
        return dict(catalog_size=self.m, insertions=self.insertions, flips=self.flips,
                    accepts=len(chosen), termination=ended, plan=chosen,
                    checked_candidates=checked, checked_candidate_decision_rows=checked,
                    raw_pair_decision_rows=sum(r.get("root_pairs_scanned", 0) for r in decision_checks),
                    refined_rivals_or_small_states=refined, endpoint_pruned_rows=pruned,
                    independent_rhs=sum(rhs.values()), independent_rhs_by_role=rhs,
                    **tie_work,
                    exact_energy_solves=exact_solves, decision_checks=decision_checks,
                    full_catalog_decisions_certified=ended != "unresolved",
                    termination_certified=ended != "unresolved",
                    current_version_id=current, max_block_rows=max_rows,
                    checker_seconds=perf_counter() - start,
                    memory=dict(root_sparse_payload_bytes=_csr_bytes(self.W) + _csr_bytes(self.A),
                                configured_block_rows=self.block_size,
                                known_array_payload_peak_bytes=known_payload_peak,
                                process_peak_rss_before_bytes=rss_before,
                                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                scope="RSS is process-lifetime peak; counted arrays exclude Python, Sympy, "
                                      "CG workspace and temporary ufunc/generator arrays. Structural permutation "
                                      "arrays/operators are counted; proof records scale with actual witnesses. "
                                      "No complete candidate table or M-sized numerical vector is retained."))


def reference_gains(W, s, chosen=()):
    """Independent exact full-catalog gains for small-graph correctness tests."""
    rep = A2CatalogReplay(W)
    need(rep.n <= 12, "exact reference is limited to small graphs")
    s = np.asarray(s, dtype=np.int64)
    A, seen = rep.A, set()
    for e in chosen:
        need(type(e) is int and e not in seen, "duplicate reference commit")
        A = _change(A, rep.candidate(e))
        seen.add(e)
    base = exact_energy(A, s)
    return {int(e): base - exact_energy(changed(A, pool, k), s)
            for ids, pool in rep.iter_blocks(excluded=seen) for k, e in enumerate(ids)}


def worker_request(rep, request, s, graph_hash):
    """Bind the actual recorded request to its graph, opinion, versions and totals."""
    root = rep.root_version_id(graph_hash)
    need(request["root_version_id"] == root and request["method"] in ("D", "S", "H", "G"),
         "scope/root identity")
    need(request["delta"] == 0 and request["gain_denominator"] == 2 ** 96,
         "request numerical contract")
    lo = hi = 0
    for row in request["decisions"]:
        if row["status"] in ("accept", "accepted"):
            need(type(row.get("low")) is int and type(row.get("high")) is int,
                 "published gain totals")
            lo += row["low"]
            hi += row["high"]
    need(lo == request["gain_low"] and hi == request["gain_high"], "total gain binding")
    result = rep.verify(s, request["decisions"], request["budget"], delta=request["delta"],
                        expected_opinion=request["opinion_hash"],
                        gain_denominator=request["gain_denominator"], expected_root=root)
    need(result["accepts"] == request["commits"] and
         result["termination"] == request["status"], "terminal status/count")
    need(result["current_version_id"] == request["current_version_id"], "final version binding")
    return result


def replay_file(trace_path, graph_path, opinions_path, *, request=1):
    """Replay one selected original trace row against explicitly supplied files."""
    import json
    from scipy import sparse
    trace_path, graph_path, opinions_path = map(Path, (trace_path, graph_path, opinions_path))
    trace = json.loads(trace_path.read_text())
    graph_hash = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    if "graph_hash" in trace:
        need(trace["graph_hash"] == graph_hash, "trace graph binding")
    with np.load(opinions_path, allow_pickle=False) as fixture:
        need(fixture.files == ["opinions"], "fixture members")
        matrix = fixture["opinions"]
    need(type(request) is int and 1 <= request <= len(trace["requests"]) and
         request <= len(matrix), "selected request")
    rep = A2CatalogReplay(sparse.load_npz(graph_path))
    result = worker_request(rep, trace["requests"][request - 1], matrix[request - 1], graph_hash)
    return dict(trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                graph_sha256=graph_hash,
                opinions_file_sha256=hashlib.sha256(opinions_path.read_bytes()).hexdigest(),
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                request=request, opinion_hash=opinion_hash(matrix[request - 1]), proof=result)


if __name__ == "__main__":
    import argparse
    import json
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--opinions", type=Path, required=True)
    parser.add_argument("--request", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        report = replay_file(args.trace, args.graph, args.opinions, request=args.request)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as destination:
            destination.write(payload)
    else:
        print(payload, end="")
