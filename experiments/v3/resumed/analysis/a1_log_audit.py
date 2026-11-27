#!/usr/bin/env python3
"""Read-only V2 log audit; no solve, score kernel, replay, or benchmark.

Run only under the root agent's shared V3 scheduler.  NumPy is used to read and
check the retained input arrays, never to solve a numerical system.  All gain
threshold comparisons use exact integer/Fraction arithmetic.  Existing raw
logs supply the certified lower bounds; this script does not independently
re-prove the numerical validity of every saved interval.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import csv


BASE = Path(__file__).resolve().parents[3]
SCALE = 16
UNIT = 1 << 48
DEN = 65536
OMEGA = Fraction(1, 16)
CELL_ORDER = [(g, s) for g in ("Alpha", "OTC")
              for s in ("slow", "independent", "signed-neighbor")]
COMPLETE_REQUEST = {"budget_stop", "tolerance_stop"}
COMPLETE_WORKER = {"completed_budget", "certified_stop"}
WORK_FIELDS = ("live_catalog", "endpoint_checked", "endpoint_excluded", "scored",
               "seed_visits", "cold_endpoint_visits")
ACCESS_FIELDS = ("frontier_rebuilds", "raw_heap_pops", "graph_rejects",
                 "committed_rejects", "processed_skips", "insertion_emissions",
                 "flips_emitted", "stream_pairs_examined", "stream_positive_rejects",
                 "stream_processed_skips", "stream_insertions_emitted",
                 "stream_flips_emitted", "explicit_raw_visits", "emitted_for_screen",
                 "node_sort_items", "flip_sort_items", "flip_cursor_pops",
                 "total_source_pops", "legal_insertion_pops", "processed_insertion_skips",
                 "processed_flip_skips", "insertion_caps_built", "flip_caps_built",
                 "heap_build_items", "cold_emitted_for_screen", "warm_screen_rows")
FRONTIER_SUBPHASES = ("node_sort", "flip_order", "heap_build", "initial_head_filter")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def frac(value):
    return None if value is None else str(Fraction(value))


def cell_label(cell):
    return "/".join(map(str, cell))


def known_sum(values):
    values = list(values)
    return None if any(x is None for x in values) else math.fsum(values)


def ratio(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def timing_valid(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def relative(path, root):
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


class Audit:
    def __init__(self, args):
        self.args = args
        self.sources = {}
        self.issues = []
        self.graphs = {}
        self.fixtures = {}
        self.decisions = []
        self.requests = []
        self.workers = []
        self.stages = []
        self.start = time.monotonic()
        self.created = datetime.now(timezone.utc).isoformat()

    def pin(self, path, expected=None, kind="source"):
        path = Path(path)
        record = {"path": str(path), "sha256": None, "bytes": None,
                  "expected_sha256": expected, "kind": kind, "status": "missing"}
        if path.is_file():
            actual = sha(path)
            record.update(sha256=actual, bytes=path.stat().st_size,
                          status="hashed" if expected is None else "matched" if actual == expected else "mismatch")
        self.sources[str(path)] = record
        if record["status"] in {"missing", "mismatch"}:
            self.issue("source_binding", str(path), record["status"])
        return record

    def read(self, path, expected=None, kind="source"):
        self.pin(path, expected, kind)
        return json.loads(Path(path).read_text())

    def issue(self, category, source, reason):
        self.issues.append({"category": category, "source": source, "reason": reason})

    def check(self, condition, category, source, reason):
        if not condition:
            self.issue(category, source, reason)
        return bool(condition)

    def bind_sources(self):
        e, c = self.args.evidence_root, self.args.code_root
        self.protocol = self.read(e / "v2/PROTOCOL.json", kind="protocol")
        self.manifest = self.read(e / "v2/INPUTS.json", kind="input_manifest")
        self.freeze = self.read(e / "v2/CODE_FREEZE.json", kind="code_freeze")
        self.roster = self.read(e / "v2/A_JOBS.json", kind="roster")
        self.binding = self.read(c / "reproduction/SOURCE_BINDING.json", kind="source_binding")
        self.selection = self.read(self.args.selection, kind="frozen_parent_selection")
        self.pin(Path(__file__), kind="audit_script")
        self.pin(e / "v2/SEED_DRAW.json", self.protocol.get("seed_draw_sha256"), "seed_binding")
        self.protocol_digest = sha(e / "v2/PROTOCOL.json")
        self.input_digest = sha(e / "v2/INPUTS.json")
        self.code_digest = sha(c / "reproduction/SOURCE_BINDING.json")
        self.check(self.manifest.get("protocol_sha256") == self.protocol_digest,
                   "source_binding", "INPUTS.json", "protocol digest mismatch")
        self.check(sha(e / "v2/CODE_FREEZE.json") == self.protocol.get("code_freeze_sha256"),
                   "source_binding", "CODE_FREEZE.json", "freeze digest mismatch")
        self.check(sha(e / "v2/A_JOBS.json") == self.protocol.get("a_jobs_sha256"),
                   "source_binding", "A_JOBS.json", "roster digest mismatch")
        self.check(self.freeze == self.manifest.get("code"), "source_binding", "INPUTS.json",
                   "embedded code freeze differs")
        for name, expected in self.freeze["source_files_sha256"].items():
            self.pin(c / name, expected, "bound_runtime_asset")
            self.check(self.binding["source_files_sha256"].get(name) == expected,
                       "source_binding", name, "reproduction binding differs from frozen runtime")
        constants = self.protocol.get("constants", {})
        required = {"delta": 0, "operator_scale": SCALE, "proof_unit": UNIT,
                    "screen_unit": 1 << 20, "opinion_denominator": DEN,
                    "insertion_weight": "1/16", "budget": 15}
        self.unit_contract_valid = all(constants.get(k) == v for k, v in required.items())
        self.check(self.unit_contract_valid, "unit_contract", "PROTOCOL.json",
                   "actual unit or service constants differ from audited source contract")
        # These are independently read source locations, not production imports.
        self.unit_sources = {
            "constants": "src/common.py:SCALE,UNIT,DEN",
            "operator": "src/weighted_adapter.py:Kernel.operator and Kernel.child",
            "raw_score": "src/weighted_adapter.py:WeightedViews.score",
            "logged_interval": "src/a2_session.py:decision low/high and gain_denominator",
            "admission": "src/a2_session.py:request bounded opinion validation",
            "catalog": "src/a2_catalog.py:AllCatalog._batch_valid/scalar",
        }
        self.source_contract_valid = self.unit_contract_valid and not any(
            r["category"] == "source_binding" for r in self.issues)

    def bind_inputs(self):
        import numpy as np
        self.audit_environment = {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "numpy": np.__version__, "numerical_threads": 1,
            "purpose": "read-only input/log diagnosis; no performance comparison with historical environment"}
        inp = self.args.inputs_root
        for graph in ("Alpha", "OTC"):
            spec = self.manifest["graphs"][graph]
            path = inp / spec["weighted_path"]
            before = len(self.issues)
            self.pin(path, spec["weighted_sha256"], "graph")
            with np.load(path, allow_pickle=False) as raw:
                shape = tuple(map(int, raw["shape"]))
                data, indices, indptr = raw["data"], raw["indices"], raw["indptr"]
                fmt = raw["format"].item()
                if isinstance(fmt, bytes):
                    fmt = fmt.decode()
                csr_valid = (fmt == "csr" and shape == (spec["n"], spec["n"])
                             and len(indptr) == shape[0] + 1 and int(indptr[0]) == 0
                             and int(indptr[-1]) == len(data) == len(indices)
                             and bool(np.all(indptr[1:] >= indptr[:-1]))
                             and bool(np.all((indices >= 0) & (indices < shape[0]))))
                self.check(csr_valid, "model_binding", str(path), "invalid retained CSR structure")
                value_valid = (data.dtype.kind in "iuf" and data.dtype.itemsize <= 8
                               and bool(np.isfinite(data).all()) and bool(np.all(data != 0))
                               and bool(np.all(np.abs(data) <= 10))
                               and bool(np.all(2 * data == np.rint(2 * data))))
                self.check(value_valid, "model_binding", str(path), "weights fail exact natural-weight admission")
                edges = {}
                duplicate = False
                if csr_valid and value_valid:
                    for i in range(shape[0]):
                        for pos in range(int(indptr[i]), int(indptr[i + 1])):
                            key = (i, int(indices[pos]))
                            duplicate |= key in edges
                            edges[key] = int(round(SCALE * float(data[pos])))
                self.check(not duplicate, "model_binding", str(path), "duplicate CSR edge slots")
                self.check(bool(edges) and all(i != j and edges.get((j, i)) == d
                                               for (i, j), d in edges.items()),
                           "model_binding", str(path), "graph not exactly symmetric and zero-diagonal")
                upper = {i * shape[0] + j: d for (i, j), d in edges.items() if i < j}
                positive = sum(d > 0 for d in upper.values())
                negative = sum(d < 0 for d in upper.values())
                nonedges = shape[0] * (shape[0] - 1) // 2 - len(upper)
                self.check((positive, negative, nonedges, nonedges + negative) ==
                           (spec["root_positive_edges"], spec["root_negative_edges"],
                            spec["root_nonedges"], spec["catalog_size"]),
                           "model_binding", str(path), "graph count/contract differs from frozen manifest")
                self.graphs[graph] = {"n": shape[0], "root_edges_scaled": upper,
                    "valid": len(self.issues) == before, "source": str(path),
                    "sha256": sha(path), "root_positive_edges": positive,
                    "root_negative_edges": negative, "root_nonedges": nonedges,
                    "catalog_size": nonedges + negative}
        for fixture in self.manifest["fixtures"]:
            cell = fixture["graph"], fixture["stream"]
            if cell not in CELL_ORDER:
                continue
            path = inp / "v2" / fixture["file"]
            before = len(self.issues)
            self.pin(path, fixture["npz_sha256"], "opinion_fixture")
            with np.load(path, allow_pickle=False) as raw:
                matrix = raw["opinions"]
                valid = (matrix.dtype == np.dtype("int64") and list(matrix.shape) == fixture["shape"]
                         and bool(np.all((-DEN <= matrix) & (matrix <= DEN))))
                self.check(valid, "opinion_binding", str(path), "opinion dtype, shape or |s|<=1 contract failed")
                matrix_hash = hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
                row_hashes = [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in matrix]
                self.check(matrix_hash == fixture["matrix_sha256"] and row_hashes == fixture["opinion_hashes"],
                           "opinion_binding", str(path), "exact opinion array hashes differ")
                self.check((int(matrix.min()), int(matrix.max())) == (fixture["min_code"], fixture["max_code"]),
                           "opinion_binding", str(path), "retained opinion bounds differ from frozen metadata")
                self.fixtures[cell] = {"spec": fixture, "source": str(path), "sha256": sha(path),
                    "valid": len(self.issues) == before, "row_hashes": row_hashes,
                    "shape": list(matrix.shape), "min_code": int(matrix.min()),
                    "max_code": int(matrix.max()), "s_infinity_bound_exact": frac(Fraction(
                        max(abs(int(matrix.min())), abs(int(matrix.max()))), DEN))}

    def common(self, worker, source):
        return {"source": str(source), "source_sha256": self.sources[str(source)]["sha256"],
                "code_digest": self.code_digest, "protocol_digest": self.protocol_digest,
                "input_digest": worker.get("fixture", {}).get("npz_sha256"),
                "input_manifest_digest": self.input_digest, "graph": worker["graph"],
                "stream": worker["stream"], "method": worker["method"],
                "repeat_id": worker["repeat_id"]}

    def stage_rows(self, context, phases, scope, category, parent=None):
        for name, seconds in phases.items():
            self.stages.append({**context, "scope": scope, "status": "measured" if timing_valid(seconds) else "unknown",
                "reason": "retained V2 clock measurement; no work-count to time conversion",
                "stage": name, "seconds": seconds if timing_valid(seconds) else None,
                "cost_category": category, "parent_timing_scope": parent,
                "additivity": "exclusive only within this scope; never add nested subphases to their parent"})

    def accepted_edit_valid(self, decision, graph, chosen):
        e = decision.get("winner_id")
        if type(e) is not int or e in chosen:
            return False
        i, j = divmod(e, graph["n"])
        root_weight = graph["root_edges_scaled"].get(e)
        if not 0 <= i < j < graph["n"] or (root_weight is not None and root_weight > 0):
            return False
        flip = root_weight is not None and root_weight < 0
        weight = -root_weight if flip else 1
        return (decision.get("i") == i and decision.get("j") == j
                and decision.get("flip") in (0, 1, False, True)
                and bool(decision["flip"]) == flip and decision.get("d") == weight)

    def process_worker(self, path, roster_entry, position):
        worker = self.read(path, kind="raw_worker")
        job_path = path.with_suffix(".job.json")
        job = self.read(job_path, kind="raw_job")
        source = str(path)
        start_issues = len(self.issues)
        cell = worker["graph"], worker["stream"]
        graph, fixture = self.graphs[cell[0]], self.fixtures[cell]
        context = self.common(worker, path)
        self.check(all(job.get(k) == v for k, v in roster_entry.items()),
                   "worker_binding", source, "job differs from independent frozen roster")
        self.check(all(worker.get(k) == v for k, v in roster_entry.items() if k != "requests"),
                   "worker_binding", source, "worker identity differs from frozen roster")
        self.check(job.get("frozen_manifest_sha256") == worker.get("frozen_manifest_sha256") == self.input_digest
                   and worker.get("input_manifest_sha256") == self.input_digest,
                   "worker_binding", source, "worker input manifest hash mismatch")
        self.check(worker.get("protocol_sha256") == self.protocol_digest
                   and worker.get("source_files_sha256") == self.freeze["source_files_sha256"],
                   "worker_binding", source, "worker protocol or runtime hash map mismatch")
        self.check(worker.get("fixture") == fixture["spec"] and worker.get("graph_hash") == graph["sha256"]
                   and worker.get("n") == graph["n"] and worker.get("catalog_size") == graph["catalog_size"],
                   "worker_binding", source, "worker graph or fixture binding mismatch")
        worker_binding_valid = self.source_contract_valid and len(self.issues) == start_issues
        requests = worker.get("requests", [])
        self.check(len({q.get("query_position") for q in requests}) == len(requests),
                   "request_binding", source, "duplicate query positions")
        request_rows = []
        self.stage_rows(context, worker.get("structural_phases", {}), "worker_structural_exclusive", "complete_service_or_partial_work")
        for q in sorted(requests, key=lambda r: r["query_position"]):
            qpos = q["query_position"]
            qsource = f"{source}#requests/query_position={qpos}"
            before = len(self.issues)
            self.check(type(qpos) is int and 1 <= qpos <= roster_entry["requests"],
                       "request_binding", qsource, "unregistered query position")
            self.check(q.get("opinion_hash") == fixture["row_hashes"][qpos - 1],
                       "request_binding", qsource, "opinion row hash mismatch")
            self.check(q.get("gain_denominator") == UNIT ** 2 and q.get("delta") == 0
                       and q.get("budget") == 15 and q.get("method") == worker["method"],
                       "request_binding", qsource, "gain units or request contract differ")
            self.check(q.get("root_version_id") == worker.get("root_version_id"),
                       "request_binding", qsource, "request root differs from worker root")
            phases = q.get("phases", {})
            phase_sum = known_sum(v if timing_valid(v) else None for v in phases.values())
            self.check(phase_sum is not None and timing_valid(q.get("request_seconds"))
                       and abs(phase_sum - q["request_seconds"]) < 1e-7,
                       "timing", qsource, "exclusive request phases do not sum to retained wall")
            req_valid = worker_binding_valid and fixture["valid"] and graph["valid"] and len(self.issues) == before
            qctx = {**context, "query_position": qpos, "opinion_hash": q.get("opinion_hash")}
            self.stage_rows(qctx, phases, "request_exclusive", "complete_request" if q.get("status") in COMPLETE_REQUEST else "partial_work")
            # Online phases overlap with full request phases: retained in JSON only.
            accesses = {a.get("epoch"): a for a in q.get("access", {}).get("epochs", [])}
            self.check(len(accesses) == len(q.get("access", {}).get("epochs", [])),
                       "decision_binding", qsource, "duplicate access epochs")
            current = q["root_version_id"]
            chosen = []
            accepted = []
            epoch_rows = []
            prefix_valid = req_valid
            for ordinal, d in enumerate(q.get("decisions", []), 1):
                epoch = d.get("epoch")
                dsource = f"{qsource}/decisions/epoch={epoch}"
                access = accesses.get(epoch, {})
                accepted_status = d.get("status") == "accepted"
                chain_valid = (epoch == ordinal and d.get("root_version_id") == q["root_version_id"]
                               and d.get("parent_version_id") == current)
                self.check(chain_valid, "version_chain", dsource, "parent/root/epoch chain mismatch")
                resolution = d.get("comparison_resolution")
                witnesses = [] if resolution is None else resolution.get("witnesses", [])
                resolution_bound = resolution is None or (
                    resolution.get("parent_version_id") == current
                    and all(w.get("parent_version_id") == current
                            and w.get("opinion_sha256") == q.get("opinion_hash")
                            and w.get("opinion_denominator") == DEN and w.get("operator_scale") == SCALE
                            and w.get("score_unit") == UNIT for w in witnesses))
                self.check(resolution_bound, "witness_binding", dsource, "saved witness source/unit/version mismatch")
                access_valid = (access.get("epoch") == epoch and access.get("parent_version_id") == current
                                and access.get("source") == worker["method"])
                self.check(access_valid, "decision_binding", dsource, "access row not bound to decision parent")
                edit_valid = self.accepted_edit_valid(d, graph, chosen) if accepted_status else None
                if accepted_status:
                    self.check(edit_valid, "legality", dsource, "accepted edit differs from current legal root-bound catalog")
                low, high = d.get("low"), d.get("high")
                interval_valid = (type(low) is int and type(high) is int and 0 < low <= high)
                if accepted_status:
                    self.check(interval_valid, "interval_morphology", dsource, "accepted interval is absent/nonpositive/reversed")
                resolution_complete = (resolution is None or (
                    not resolution.get("unresolved_ids") and resolution.get("canonical_winner_id") == d.get("winner_id")))
                coverage_record = (accepted_status and d.get("proof_scope") == "full_catalog"
                    and d.get("live_catalog") == graph["catalog_size"] - len(chosen)
                    and access_valid and access.get("source_complete") is True and resolution_complete
                    and (not d.get("exact_overlap_resolution") or resolution is not None))
                flip = bool(d["flip"]) if accepted_status and d.get("flip") in (0, 1, False, True) else None
                global_eligible = bool(prefix_valid and chain_valid and edit_valid and interval_valid
                                       and resolution_bound and coverage_record)
                sufficient = (Fraction(low, UNIT ** 2) > 4 * OMEGA) if global_eligible and flip else None
                if not accepted_status:
                    exclusion_status = "not_applicable_no_accepted_winner"
                elif flip is False:
                    exclusion_status = "not_applicable_insertion_winner"
                elif not global_eligible:
                    exclusion_status = "unknown_invalid_or_missing_binding"
                elif sufficient:
                    exclusion_status = "certified_from_logged_lower_and_global_4omega"
                else:
                    exclusion_status = "not_certified_by_available_global_bound"
                # A2 audit=False records contain no parent q/r.  A child residual
                # is not a substitute for the missing parent center/radius pair.
                parent_state_snapshot = next((s for s in q.get("snapshots", []) if s.get("epoch") == epoch), None)
                range_reason = ("parent_q_r_not_retained" if parent_state_snapshot is None else
                                "snapshot_present_but_not_independently_enclosure_validated_by_this_log_audit")
                frontier = access.get("frontier_build_phases")
                seed = access.get("seed_build_phases")
                if worker["method"] == "H" and epoch == 1 and frontier is not None and seed is not None:
                    self.check(frontier == seed, "timing", dsource, "root H seed/frontier duplicate timers disagree")
                # Root H reuses the seed frontier.  Count frontier_build_phases
                # once; do not also add its seed_build_phases alias.
                selected_subphases = frontier if worker["method"] == "H" else seed if epoch == 1 else None
                insert_prep = None
                if selected_subphases is not None:
                    values = [selected_subphases.get(k) for k in ("node_sort", "heap_build")]
                    if all(timing_valid(v) for v in values):
                        insert_prep = math.fsum(values)
                    self.stage_rows({**qctx, "epoch": epoch, "parent_version_id": current},
                        selected_subphases, "frontier_build_subphases", "nested_historical_subphase",
                        "request frontier_prepare; root seed_build_phases is an alias for H")
                removable = insert_prep if worker["method"] == "H" and sufficient is True else None
                source_lower = None if resolution is None else resolution.get("source_lower")
                source_upper = None if resolution is None else resolution.get("source_upper")
                row = {**qctx, "scope": "one saved decision observation", "status": d.get("status"),
                    "reason": d.get("failure_reason"), "epoch": epoch,
                    "root_version_id": q["root_version_id"], "parent_version_id": current,
                    "recorded_parent_version_id": d.get("parent_version_id"),
                    "current_version_id": d.get("current_version_id"), "chosen_prefix": list(chosen),
                    "winner_id": d.get("winner_id"), "edit_type": "flip" if flip else "insert" if flip is False else None,
                    "i": d.get("i"), "j": d.get("j"), "edit_weight_scaled": d.get("d"),
                    "edit_weight_physical": frac(Fraction(d["d"], SCALE)) if d.get("d") is not None else None,
                    "gain_low_integer": low, "gain_high_integer": high, "gain_denominator": q.get("gain_denominator"),
                    "gain_low_physical": frac(Fraction(low, UNIT ** 2)) if type(low) is int else None,
                    "gain_high_physical": frac(Fraction(high, UNIT ** 2)) if type(high) is int else None,
                    "provisional_gain_low_physical": frac(Fraction(source_lower * SCALE, UNIT ** 2)) if type(source_lower) is int else None,
                    "provisional_gain_high_physical": frac(Fraction(source_upper * SCALE, UNIT ** 2)) if type(source_upper) is int else None,
                    "provisional_interval_unit_note": "resolution source endpoints are raw score units and need SCALE; decision endpoints already include SCALE",
                    "proof_scope": d.get("proof_scope"), "recorded_full_catalog_coverage": coverage_record,
                    "independent_full_decision_numerical_replay": None,
                    "log_certificate_scope": "bound and structurally checked saved certificate, not a new full numerical replay",
                    "version_chain_valid": chain_valid, "edit_legality_valid": edit_valid,
                    "source_model_opinion_units_valid": req_valid,
                    "saved_witness_binding_valid": resolution_bound,
                    "comparison_resolution": resolution, "witness_count": len(witnesses),
                    "exact_overlap_resolution": d.get("exact_overlap_resolution"),
                    "scored_rival_max_high": d.get("scored_rival_max_high"), "cold_upper": d.get("cold_upper"),
                    "competitor_max_upper": d.get("competitor_max_upper"),
                    "work": {k: d.get(k) for k in WORK_FIELDS},
                    "access_work": {k: access.get(k) for k in ACCESS_FIELDS},
                    "access_source_complete": access.get("source_complete"),
                    "cold_start_lower_fine": access.get("cold_start_lower_fine"),
                    "cold_end_lower_fine": access.get("cold_end_lower_fine"),
                    "remaining_upper_coarse": access.get("remaining_upper_coarse"),
                    "insertion_weight_physical": str(OMEGA),
                    "global_insert_upper_physical": str(4 * OMEGA) if prefix_valid and chain_valid else None,
                    "range_insert_upper_physical": None, "range_bound_status": "unknown", "range_bound_reason": range_reason,
                    "available_insert_bound_kind": "global_4omega" if prefix_valid and chain_valid else None,
                    "global_exclusion_eligibility": global_eligible if flip else None,
                    "certified_flip_lower_exceeds_available_insert_upper": sufficient,
                    "whole_insert_exclusion_status": exclusion_status,
                    "both_families_competitive": False if sufficient else None,
                    "competition_classification": "insertions_excluded" if sufficient else "cannot_classify",
                    "competition_reason": "strict global bound exclusion" if sufficient else
                        "absence of global exclusion is not evidence that both families compete; separate certified family maxima and parent q/r are absent",
                    "cost": {"category": "historical_decision_subphases", "decision_total_seconds": None,
                             "frontier_build_subphases": selected_subphases,
                             "insert_only_node_sort_plus_heap_build_seconds": insert_prep,
                             "eligible_removable_measured_component_seconds": removable,
                             "insertion_enumeration_seconds": None,
                             "note": "node_sort and heap_build are insertion-only; initial_head_filter, candidate_access, seed_access and endpoint_screen mix families"}}
                self.decisions.append(row)
                epoch_rows.append(row)
                if accepted_status:
                    accepted.append(d)
                    chosen.append(d["winner_id"])
                    expected_current = hashlib.sha256(f"{current}/{d['winner_id']}".encode()).hexdigest()
                    self.check(d.get("current_version_id") == expected_current,
                               "version_chain", dsource, "child version hash differs from parent/winner chain")
                    prefix_valid = bool(prefix_valid and chain_valid and edit_valid
                                        and d.get("current_version_id") == expected_current)
                    current = expected_current
                else:
                    self.check(d.get("current_version_id") == current, "version_chain", dsource,
                               "nonaccepted decision changed current version")
            self.check(q.get("current_version_id") == current and q.get("commits") == len(accepted),
                       "version_chain", qsource, "final version or commit count differs from decision chain")
            self.check(q.get("gain_low") == sum(d["low"] for d in accepted)
                       and q.get("gain_high") == sum(d["high"] for d in accepted),
                       "interval_morphology", qsource, "request gain totals differ from saved accepted intervals")
            complete = q.get("status") in COMPLETE_REQUEST
            if q.get("status") == "budget_stop":
                self.check(len(accepted) == q.get("budget"), "completion", qsource,
                           "budget completion does not contain the registered accepted count")
            expected_result = {"budget_stop": "completed_budget", "tolerance_stop": "certified_stop",
                               "unresolved": "unresolved"}.get(q.get("status"))
            if expected_result is not None:
                self.check(q.get("result_status") == expected_result, "completion", qsource,
                           "wrapper result status differs from request status")
            rrow = {**qctx, "scope": "one saved request execution", "status": q.get("status"),
                "result_status": q.get("result_status"), "reason": q.get("failure_reason"),
                "complete": complete, "accepted_decisions": len(accepted), "decision_records": len(epoch_rows),
                "root_version_id": q["root_version_id"], "current_version_id": q.get("current_version_id"),
                "gain_low_integer": q.get("gain_low"), "gain_high_integer": q.get("gain_high"),
                "gain_denominator": q.get("gain_denominator"), "work": q.get("counters"),
                "retained_snapshots": len(q.get("snapshots", [])),
                "cost": {"category": "complete_request" if complete else "partial_work",
                    "request_seconds": q.get("request_seconds"), "phases_exclusive": phases,
                    "online_seconds": q.get("online_seconds"), "online_phases_overlapping_subset": q.get("online_phases"),
                    "reset_opinion_seconds": q.get("reset_opinion_seconds"),
                    "wrapper_seconds": q.get("wrapper_seconds"),
                    "index_build_phases_nested": q.get("index_build_phases")},
                "decision_sequence": [(d["epoch"], d["winner_id"], d["parent_version_id"],
                                       d["current_version_id"], d.get("flip"), d.get("d")) for d in accepted]}
            self.requests.append(rrow)
            request_rows.append(rrow)
        complete_q4 = (len(request_rows) == roster_entry["requests"] and
                       {q["query_position"] for q in request_rows} == set(range(1, roster_entry["requests"] + 1))
                       and all(q["complete"] for q in request_rows))
        self.check((worker.get("status") in COMPLETE_WORKER) == complete_q4,
                   "completion", source, "worker status differs from independent per-request completion")
        structural = worker.get("structural_seconds")
        phase_structural = known_sum(v if timing_valid(v) else None for v in worker.get("structural_phases", {}).values())
        self.check(timing_valid(structural) and phase_structural is not None and abs(structural - phase_structural) < 1e-7,
                   "timing", source, "exclusive structural phases do not sum to structural time")
        full_or_partial = known_sum([structural, *(q["cost"]["request_seconds"] for q in request_rows)])
        self.check(full_or_partial is not None and abs(full_or_partial - worker["service_accounted_seconds"]) < 1e-7,
                   "timing", source, "service accounting differs from structural plus request cost")
        self.check(worker.get("complete_service_seconds") == (worker.get("service_accounted_seconds") if complete_q4 else None),
                   "timing", source, "incomplete work assigned a complete Q4 time")
        wrow = {**context, "scope": "one frozen worker observation", "status": worker.get("status"),
            "reason": worker.get("failure_reason"), "roster_position": position,
            "planned_requests": roster_entry["requests"], "recorded_requests": len(request_rows),
            "completed_requests_recomputed": sum(q["complete"] for q in request_rows),
            "complete_q4_recomputed": complete_q4, "accepted_decisions": sum(q["accepted_decisions"] for q in request_rows),
            "binding_valid": worker_binding_valid, "root_version_id": worker.get("root_version_id"),
            "cost": {"category": "complete_service" if complete_q4 else "partial_work",
                "complete_service_seconds": full_or_partial if complete_q4 else None,
                "partial_work_seconds": full_or_partial if not complete_q4 else None,
                "service_accounted_seconds": worker.get("service_accounted_seconds"),
                "process_wall_seconds": worker.get("process_wall_seconds"),
                "structural_seconds": structural, "structural_phases_exclusive": worker.get("structural_phases"),
                "overlapping_metering_not_additive": worker.get("metering")},
            "environment_scope": "original V2 machine only; not a current-environment timing comparator"}
        self.workers.append(wrow)

    def process_all(self):
        raw = self.args.evidence_root / "v2/raw_a"
        # The directory also contains one aggregate supervisor document.  It
        # must be bound as metadata, not counted as a 55th worker observation.
        schedule_path = raw / "SCHEDULE_RESULTS.json"
        schedule = self.read(schedule_path, kind="raw_supervisor_schedule")
        self.check(schedule.get("registered_jobs") == len(self.roster)
                   and len(schedule.get("jobs", [])) == len(self.roster),
                   "roster", str(schedule_path), "supervisor roster size differs from frozen jobs")
        self.schedule_metadata = {"source": str(schedule_path), "sha256": sha(schedule_path),
            "scope": "supervisor metadata, excluded from worker denominator",
            "registered_jobs": schedule.get("registered_jobs"),
            "completed_requests_field_used_as_completion_count": False,
            "started_utc_used_as_true_batch_start": False}
        paths = sorted(p for p in raw.glob("*.json")
                       if not p.name.endswith(".job.json") and p.name != "SCHEDULE_RESULTS.json")
        self.check(len(paths) == len(self.roster), "roster", str(raw), "raw worker count differs from planned roster")
        expected_paths = set()
        for position, job in enumerate(self.roster):
            name = f"{position:03d}_{job['graph']}_{job['stream']}_{job['method']}_r{job['repeat_id']}.json"
            path = raw / name
            expected_paths.add(path)
            scheduled = schedule.get("jobs", [])
            self.check(position < len(scheduled) and scheduled[position].get("file") == f"v2/raw_a/{name}",
                       "roster", str(schedule_path), f"supervisor slot {position} differs from frozen worker path")
            if not path.exists():
                self.issue("roster", str(path), "planned worker missing")
                continue
            self.process_worker(path, job, position)
        self.check(set(paths) == expected_paths, "roster", str(raw), "raw worker filenames differ from registered jobs")

    def winner_stats(self, rows):
        accepted = [r for r in rows if r["status"] == "accepted"]
        flips = [r for r in accepted if r["edit_type"] == "flip"]
        eligible = [r for r in flips if r["global_exclusion_eligibility"] is True]
        sufficient = [r for r in flips if r["certified_flip_lower_exceeds_available_insert_upper"] is True]
        return {"decision_observations": len(rows), "accepted_decisions": len(accepted),
            "flip_winners": len(flips), "insertion_winners": sum(r["edit_type"] == "insert" for r in accepted),
            "flip_winner_ratio_among_accepted": ratio(len(flips), len(accepted)),
            "flip_bound_eligible": len(eligible), "flip_bound_unknown": len(flips) - len(eligible),
            "flip_lower_strictly_exceeds_global_insert_bound": len(sufficient),
            "flip_lower_not_above_available_global_bound": sum(r["certified_flip_lower_exceeds_available_insert_upper"] is False for r in flips),
            "sufficient_global_exclusion_ratio_among_eligible_flip_winners": ratio(len(sufficient), len(eligible)),
            "sufficient_global_exclusion_ratio_among_accepted": ratio(len(sufficient), len(accepted)),
            "unknown_range_bound_decisions": sum(r["range_insert_upper_physical"] is None for r in rows),
            "whole_insert_exclusion_ratio_is_not_flip_winner_ratio": True,
            "scope_note": "certified means the logged lower bound and full-domain certificate were source/contract bound; no new all-decisions numerical reproof"}

    def time_summary(self, workers):
        names = {Path(w["source"]).name for w in workers}
        decisions = [d for d in self.decisions if Path(d["source"]).name in names]
        requests = [q for q in self.requests if Path(q["source"]).name in names]
        total = known_sum(w["cost"]["service_accounted_seconds"] for w in workers)
        phases = defaultdict(list)
        for w in workers:
            for k, v in w["cost"]["structural_phases_exclusive"].items():
                phases[k].append(v)
        for q in requests:
            for k, v in q["cost"]["phases_exclusive"].items():
                phases[k].append(v)
        eligible = [d for d in decisions if d["method"] == "H"
                    and d["certified_flip_lower_exceeds_available_insert_upper"] is True]
        h_scope = bool(workers) and all(w["method"] == "H" for w in workers)
        component = known_sum(d["cost"]["eligible_removable_measured_component_seconds"] for d in eligible) if h_scope else None
        p = component / total if component is not None and total else None
        return {"workers": len(workers), "historical_service_or_partial_seconds": total,
            "exclusive_phase_seconds": {k: known_sum(v) for k, v in sorted(phases.items())},
            "global_excludable_H_epochs": len(eligible),
            "insertion_only_measured_component_seconds": component,
            "component_definition": "sum node_sort+heap_build only for H epochs with a valid strict global exclusion; root seed/frontier duplicates counted once",
            "component_fraction_p": p,
            "ideal_speedup_if_only_this_measured_component_vanished": 1 / (1 - p) if p is not None and p < 1 else None,
            "additional_insertion_enumeration_or_score_seconds": None,
            "additional_time_unknown_reason": "request seed_access/candidate_access/endpoint_screen/response buckets mix insertions and flips; counts cannot identify time",
            "speedup_scope": "ideal bound for deleting the specified measured component alone, with every other cost unchanged; not G performance or a bound on all possible optimizations",
            "environment_scope": "original V2 only"}

    def selection_check(self):
        rows = self.selection.get("rows", [])
        checks = []
        for selected in rows:
            matches = [d for d in self.decisions if all(d.get(k) == selected.get(k)
                for k in ("graph", "stream", "method", "repeat_id", "query_position", "epoch"))]
            ok = len(matches) == 1
            record = matches[0] if ok else None
            if record is not None:
                ok = (record["parent_version_id"] == selected["parent_version_id"]
                      and record["opinion_hash"] == selected["opinion_hash"]
                      and record["chosen_prefix"] == selected["chosen_prefix"]
                      and record["source_sha256"] == selected["source_sha256"])
            self.check(ok, "frozen_selection", selected["selection_id"], "selected parent no longer matches its raw source")
            checks.append({"selection_id": selected["selection_id"], "binding_passed": ok,
                "frozen_selection_status": selected["status"],
                "saved_parent_q_r_available": False if record is None else record["range_bound_reason"] != "parent_q_r_not_retained",
                "available_global_bound_result": None if record is None else record["whole_insert_exclusion_status"],
                "both_families_competitive": None if record is None else record["both_families_competitive"],
                "selection_unchanged_after_audit": True})
        self.check(len(rows) <= 7 and len(rows) == 7, "frozen_selection", str(self.args.selection),
                   "expected the six cell fallbacks plus actual unresolved parent")
        return {"source": str(self.args.selection), "source_sha256": sha(self.args.selection),
                "code_digest": self.code_digest, "protocol_digest": self.protocol_digest,
                "input_digest": self.input_digest, "scope": "verification of a pre-existing frozen seven-parent selection",
                "status": "matched" if all(r["binding_passed"] for r in checks) else "mismatch",
                "reason": "No selection is changed using newly calculated outcomes.",
                "cost": {"category": "read_only_diagnosis", "wall_seconds": None}, "rows": checks}

    def summarize(self):
        group_requests = defaultdict(list)
        for q in self.requests:
            group_requests[(q["graph"], q["stream"], q["query_position"])].append(q)
        sequence_conflicts = []
        for key, rows in sorted(group_requests.items()):
            sequences = {canonical_sha([r["status"], r["decision_sequence"]]) for r in rows}
            if len(sequences) != 1 or len(rows) != 9:
                sequence_conflicts.append({"cell_query": list(key), "records": len(rows), "distinct_sequences": len(sequences)})
        pair_count = interval_pairs = different_intervals = differing_endpoints = 0
        for key, rows in sorted(group_requests.items()):
            for left, right in itertools.combinations(rows, 2):
                pair_count += 1
                ld = [d for d in self.decisions if d["source"] == left["source"]
                      and d["query_position"] == left["query_position"] and d["status"] == "accepted"]
                rd = [d for d in self.decisions if d["source"] == right["source"]
                      and d["query_position"] == right["query_position"] and d["status"] == "accepted"]
                for a, b in zip(ld, rd):
                    if (a["winner_id"], a["parent_version_id"], a["current_version_id"]) != (b["winner_id"], b["parent_version_id"], b["current_version_id"]):
                        continue
                    interval_pairs += 1
                    diffs = int(a["gain_low_integer"] != b["gain_low_integer"]) + int(a["gain_high_integer"] != b["gain_high_integer"])
                    differing_endpoints += diffs
                    different_intervals += bool(diffs)
        unresolved = [d for d in self.decisions if d["status"] == "unresolved"]
        ugroups = defaultdict(list)
        for d in unresolved:
            resolution = d["comparison_resolution"] or {}
            key = (d["graph"], d["stream"], d["query_position"], d["epoch"], d["opinion_hash"],
                   d["parent_version_id"], tuple(d["chosen_prefix"]), resolution.get("provisional_winner_id"),
                   tuple(resolution.get("unresolved_ids", [])))
            ugroups[key].append(d)
        unresolved_groups = []
        for key, rows in sorted(ugroups.items()):
            d = rows[0]
            unresolved_groups.append({"graph": key[0], "stream": key[1], "query_position": key[2],
                "epoch": key[3], "opinion_hash": key[4], "parent_version_id": key[5],
                "chosen_prefix": list(key[6]), "provisional_winner_id": key[7], "unresolved_ids": list(key[8]),
                "observations": len(rows), "methods": sorted({r["method"] for r in rows}),
                "repeat_ids": sorted({r["repeat_id"] for r in rows}),
                "saved_comparison_resolution": d["comparison_resolution"],
                "sources": [{"path": r["source"], "sha256": r["source_sha256"]} for r in rows],
                "relation_classification": "unclassified_by_log_audit", "full_decision_verified": False,
                "reason": "overlap or saved partial structural witnesses are not proof of a full tie"})
        complete = [w for w in self.workers if w["complete_q4_recomputed"]]
        paired = defaultdict(dict)
        for w in complete:
            paired[(w["graph"], w["stream"], w["repeat_id"])][w["method"]] = w
        paired = {k: v for k, v in paired.items() if set(v) == {"D", "S", "H"}}
        complete_cells = {k[:2] for k in paired}
        counts = {"independent_streams": len(self.fixtures),
            "distinct_opinion_rows": len({h for f in self.fixtures.values() for h in f["row_hashes"]}),
            "planned_workers": len(self.roster), "raw_workers": len(self.workers),
            "complete_Q4_workers": len(complete), "planned_request_executions": sum(j["requests"] for j in self.roster),
            "recorded_request_executions": len(self.requests),
            "completed_request_executions": sum(q["complete"] for q in self.requests),
            "accepted_decisions": sum(d["status"] == "accepted" for d in self.decisions),
            "decisions_including_terminal": len(self.decisions), "unresolved_observations": len(unresolved),
            "distinct_unresolved_parent_states": len(ugroups), "complete_paired_repeats": len(paired),
            "complete_paired_cells": len(complete_cells)}
        expected = {"independent_streams": 6, "distinct_opinion_rows": 24, "planned_workers": 54,
                    "raw_workers": 54, "complete_Q4_workers": 45, "planned_request_executions": 216,
                    "recorded_request_executions": 216, "completed_request_executions": 207,
                    "accepted_decisions": 3132, "decisions_including_terminal": 3141,
                    "unresolved_observations": 9, "distinct_unresolved_parent_states": 1,
                    "complete_paired_repeats": 15, "complete_paired_cells": 5}
        for name, value in expected.items():
            self.check(counts[name] == value, "denominator", name, f"independently recomputed {counts[name]} differs from retained stated fact {value}")
        method_stats = {method: self.winner_stats([d for d in self.decisions if d["method"] == method]) for method in ("D", "S", "H")}
        canonical = [d for d in self.decisions if d["method"] == "H" and d["repeat_id"] == 0]
        cells = []
        for cell in CELL_ORDER:
            rows = [d for d in canonical if (d["graph"], d["stream"]) == cell]
            cells.append({"graph": cell[0], "stream": cell[1], "canonical_H_repeat0": self.winner_stats(rows),
                          "graph_catalog": {k: v for k, v in self.graphs[cell[0]].items() if k != "root_edges_scaled"}})
        time_groups = {method: self.time_summary([w for v in paired.values() for m, w in v.items() if m == method]) for method in ("D", "S", "H")}
        hpartial = self.time_summary([w for w in self.workers if w["method"] == "H" and not w["complete_q4_recomputed"]])
        selection = self.selection_check()
        summary = {"schema_version": 1, "created_utc": self.created,
            "source": str(self.args.evidence_root / "v2/raw_a"), "code_digest": self.code_digest,
            "protocol_digest": self.protocol_digest, "input_digest": self.input_digest,
            "scope": "read-only audit of all retained V2 A workers, with no new service execution",
            "status": "passed" if not self.issues else "issues_found", "reason": None if not self.issues else "See issues; do not use ineligible bound conclusions.",
            "cost": {"category": "read_only_diagnosis", "wall_seconds": time.monotonic() - self.start,
                     "service_seconds": None, "independent_numerical_checker_seconds": None,
                     "root_scheduler_ledger_is_authoritative": True},
            "counts": counts, "expected_retained_facts": expected,
            "supervisor_metadata": self.schedule_metadata,
            "audit_environment": self.audit_environment,
            "historical_environment": self.freeze.get("environment"),
            "worker_status_counts": dict(Counter(w["status"] for w in self.workers)),
            "request_status_counts": dict(Counter(q["status"] for q in self.requests)),
            "all_decision_observations": self.winner_stats(self.decisions), "by_method": method_stats,
            "canonical_H_repeat0_distinct_development_trajectory": self.winner_stats(canonical), "cells": cells,
            "sequence_consistency": {"distinct_requests": len(group_requests), "conflicts": sequence_conflicts,
                "request_pairs": pair_count, "accepted_interval_comparisons": interval_pairs,
                "interval_differences": different_intervals, "endpoint_field_differences": differing_endpoints,
                "numerical_interval_reproof_performed": False},
            "unresolved_parent_groups": unresolved_groups,
            "historical_cost_complete_paired_groups": time_groups,
            "historical_cost_H_partial_workers": hpartial,
            "unit_mapping": {"A_integer": "16 * A_physical", "state_integer_operator_solution": "A_integer^-1 * s_physical",
                "internal_score_to_physical": "SCALE * score_fine / UNIT^2",
                "logged_decision_to_physical": "decision.low/high / request.gain_denominator",
                "gain_denominator": UNIT ** 2, "operator_scale": SCALE, "insertion_weight": str(OMEGA),
                "global_insert_upper": str(4 * OMEGA), "sources": self.unit_sources,
                "global_bound_justification": "For A=I+diag(sum|w|)-W, strict row diagonal dominance gives ||A^-1||_infinity<=1; bounded s gives |z_i|<=1; legal positive insertions and sign flips preserve this model.",
                "range_bound": None, "range_bound_reason": "parent centers/radii not retained by audit=False"},
            "H_frontier_finding": {"source": "src/a2_catalog.py:MixedFrontier; src/a2_access.py:Access.next_batch",
                "implicit_range_bound": "Insertion rows start at their largest certified endpoint difference and frontier.upper is the max remaining insertion/flip cap; strict threshold retains equal caps.",
                "global_4omega_clamp_present": False,
                "preparation_cost_precedes_exclusion": True,
                "diagonal_denominator_bound_already_present": "src/a2_numeric.py:CachedStrict.screen uses the current-nonedge diagonal denominator after candidate generation, shared by D/S/H.",
                "bound_at_block_level": False,
                "G_comparison_implication": "A simple baseline can change when insertions are generated and bounded; the diagonal inequality itself is already in the shared candidate screen."},
            "frozen_parent_selection": selection,
            "unknowns": ["parent q/r range bound for all audit=False records", "per-decision total wall cost",
                "insertion-only enumeration/response cost within mixed request buckets",
                "whether both edit families are competitive at unclassified parents",
                "independent numerical reproof of every saved lower and upper endpoint"],
            "issues": self.issues}
        return summary, selection


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(path, rows):
    with Path(path).open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "null" if v is None else json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                              if isinstance(v, (dict, list, tuple, bool)) else v for k, v in row.items()})


def report(summary):
    counts = summary["counts"]
    canonical = summary["canonical_H_repeat0_distinct_development_trajectory"]
    h = summary["historical_cost_complete_paired_groups"]["H"]
    text = ["# V3 A1: retained V2 evidence audit", "", f"Status: **{summary['status']}**.", "",
        "This pass reads original records and exact retained inputs. It performs no service solve, scoring replay or historical benchmark. Saved gain intervals retain their original numerical evidence scope.", "",
        "## Independent denominators", "", "| Item | Recomputed |", "|---|---:|"]
    text += [f"| {key} | {value} |" for key, value in counts.items()]
    text += ["", "## Winners and whole-insert exclusion", "",
        f"On the six canonical H/repeat0 trajectories: {canonical['flip_winners']} flip winners and {canonical['insertion_winners']} insertion winners among {canonical['accepted_decisions']} accepts. Of {canonical['flip_bound_eligible']} eligible flip winners, {canonical['flip_lower_strictly_exceeds_global_insert_bound']} have a logged certified physical lower bound strictly above 1/4. These are distinct denominators.", "",
        "The source-bound contract gives omega=1/16 and |z_i|<=1, hence U_insert<=4*omega=1/4 even without saved parent q/r. Decision low/high are already physical gain numerators: divide only by gain_denominator=UNIT^2. Provisional resolution source_lower/source_upper are internal score units and require SCALE before that division.", "",
        "If a flip lower bound is not above 1/4, this audit does not conclude that insertion exclusion is impossible or that both families compete. The finer range bound is unknown because parent centers/radii were not retained. Successful exclusions rely on saved current-version full-catalog certificates; they are not new independent numerical reproofs of every endpoint.", "",
        "## Existing H mechanism and measured cost", "",
        "H already orders insertion row heads by certified endpoint range caps and merges them with sparse flip caps. It can stop before emitting further insertions, but it first pays for sorting node centers and constructing the insertion heap. Equality is retained at the strict delta=0 cutoff. A global 4*omega clamp is not present. The shared candidate screen already uses the current-nonedge diagonal denominator inequality after candidates have been generated.", "",
        f"Across the complete historical paired H workers, total service cost is {h['historical_service_or_partial_seconds']} s. The separately measured node_sort+heap_build cost at globally excludable flip epochs is {h['insertion_only_measured_component_seconds']} s (p={h['component_fraction_p']}); ideal speedup if only that measured component vanished is {h['ideal_speedup_if_only_this_measured_component_vanished']}.", "",
        "This ideal deletion calculation applies only to the named measured component. It excludes all new baseline overhead. initial_head_filter, seed_access, candidate_access, endpoint_screen and response buckets mix edit families, so their insertion-only removable time remains null. Candidate/RHS/Gram counts have not been converted into time. Root H seed/frontier timer aliases are counted once. Partial Alpha workers are reported separately, and no old/new environment speed ratio is formed.", "",
        "## Frozen parent selection", "",
        "The six H/repeat0 q1 initial parents plus the genuine Alpha/signed-neighbor/q1 fourth parent were frozen before new solves. Raw audit=False records do not supply separate certified family maxima or parent centers/radii; therefore the prescribed q1 initial fallback is used with competitiveness marked cannot_classify. The selection is not changed using this audit's outcomes.", "",
        "## Scope limits", "",
        "Nine unresolved observations are grouped by exact opinion, accepted prefix and parent version. Partial structural witnesses or overlapping intervals do not classify the remaining relation. Full decision verification remains a separate task. Complete-service status is recomputed from individual requests, never from the scheduler's completed_requests field. Existing factor/solve metering overlaps service phases and is not added to them.", "",
        "Machine-readable details: a1_summary.json, a1_decision_ledger.jsonl/.csv, a1_request_ledger.jsonl, a1_worker_ledger.jsonl, a1_stage_costs.csv, a1_binding.json and a1_state_selection_check.json.", ""]
    if summary["issues"]:
        text += ["## Issues", ""] + [f"- {i['category']}: {i['source']}: {i['reason']}" for i in summary["issues"]]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=BASE / "v2_received/code")
    parser.add_argument("--evidence-root", type=Path, default=BASE / "v2_received/evidence")
    parser.add_argument("--inputs-root", type=Path, default=BASE / "v2_received/inputs")
    parser.add_argument("--out-dir", type=Path, default=BASE / "v3/resumed/reports")
    parser.add_argument("--selection", type=Path, default=BASE / "v3/resumed/reports/state_selection_proposal.json")
    parser.add_argument("--max-seconds", type=int, default=120)
    parser.add_argument("--memory-mib", type=int, default=2048)
    args = parser.parse_args()
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    if not 0 < args.max_seconds <= 120 or not 0 < args.memory_mib <= 2048:
        parser.error("at most 120 seconds and 2048 MiB are required")
    signal.signal(signal.SIGALRM, lambda signum, frame: (_ for _ in ()).throw(TimeoutError("A1 read-only audit wall limit")))
    signal.alarm(args.max_seconds)
    cap = args.memory_mib * 1024 ** 2
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = ["a1_summary.json", "a1_decision_ledger.jsonl", "a1_decision_ledger.csv",
             "a1_request_ledger.jsonl", "a1_worker_ledger.jsonl", "a1_stage_costs.csv",
             "a1_binding.json", "a1_state_selection_check.json", "a1_report.md", "a1_output_manifest.json"]
    if any((args.out_dir / name).exists() for name in names):
        raise FileExistsError("Refusing to overwrite an A1 result; use a fresh --out-dir")
    audit = Audit(args)
    audit.bind_sources()
    audit.bind_inputs()
    audit.process_all()
    summary, selection = audit.summarize()
    write_json(args.out_dir / "a1_summary.json", summary)
    write_jsonl(args.out_dir / "a1_decision_ledger.jsonl", audit.decisions)
    write_csv(args.out_dir / "a1_decision_ledger.csv", audit.decisions)
    write_jsonl(args.out_dir / "a1_request_ledger.jsonl", audit.requests)
    write_jsonl(args.out_dir / "a1_worker_ledger.jsonl", audit.workers)
    write_csv(args.out_dir / "a1_stage_costs.csv", audit.stages)
    write_json(args.out_dir / "a1_binding.json", {
        "source": "retained Code/Evidence/Inputs packages", "code_digest": audit.code_digest,
        "protocol_digest": audit.protocol_digest, "input_digest": audit.input_digest,
        "scope": "all frozen runtime assets, all 54 raw records/jobs, six opinion fixtures and two graphs",
        "status": "passed" if not audit.issues else "issues_found", "reason": None,
        "cost": {"category": "included_in_read_only_audit", "wall_seconds": None},
        "files": list(audit.sources.values()),
        "graphs": {g: {k: v for k, v in rec.items() if k != "root_edges_scaled"} for g, rec in audit.graphs.items()},
        "fixtures": {cell_label(c): r for c, r in audit.fixtures.items()}})
    write_json(args.out_dir / "a1_state_selection_check.json", selection)
    with (args.out_dir / "a1_report.md").open("x") as handle:
        handle.write(report(summary))
    write_json(args.out_dir / "a1_output_manifest.json", {
        "source": str(Path(__file__)), "code_digest": sha(__file__),
        "protocol_digest": audit.protocol_digest, "input_digest": audit.input_digest,
        "scope": "A1 output bundle, manifest excludes itself", "status": summary["status"], "reason": None,
        "cost": {"category": "read_only_diagnosis", "wall_seconds": time.monotonic() - audit.start},
        "members": [{"path": name, "sha256": sha(args.out_dir / name), "bytes": (args.out_dir / name).stat().st_size}
                    for name in names if name != "a1_output_manifest.json"]})
    signal.alarm(0)
    print(json.dumps({"status": summary["status"], "counts": summary["counts"],
                      "canonical_H": summary["canonical_H_repeat0_distinct_development_trajectory"],
                      "issue_count": len(audit.issues), "output": str(args.out_dir / "a1_summary.json")}))
    return 0 if not audit.issues else 2


if __name__ == "__main__":
    sys.exit(main())
