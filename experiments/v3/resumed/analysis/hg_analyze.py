#!/usr/bin/env python3
"""Audit frozen H/G result metadata without importing numerical production code.

Run only through the root-owned budget scheduler. This program validates bindings,
version/edit trajectories and accounting, not the numerical truth of gain bounds.
Historical agreement is not an independent full-catalog or endpoint reproof.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[3]
COMPLETE = {"completed_budget", "certified_stop"}
STATUS_MAP = {"budget_stop": "completed_budget", "tolerance_stop": "certified_stop",
              "unresolved": "unresolved", "memory_error": "memory_limit_exceeded",
              "error": "runtime/certificate_error"}
EDIT_FIELDS = ("winner_id", "parent_version_id", "current_version_id", "i", "j", "flip", "d")
TOP_BINDINGS = ("root_version_id", "operator_hash", "catalog_hash", "graph_hash", "n", "catalog_size")
FIXTURE_BINDINGS = ("graph", "stream", "fixture_id", "npz_sha256", "matrix_sha256", "opinion_hashes", "shape")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def map_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (WORKSPACE / path).resolve()


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(WORKSPACE))
    except ValueError:
        return str(Path(path).resolve())


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def near(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-5)


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def identity(decision):
    return tuple(decision.get(key) for key in EDIT_FIELDS)


def accepted(request):
    return [d for d in request.get("decisions", []) if d.get("status") == "accepted"]


def request_position(request, index):
    return request.get("query_position", index)


def key_of(value):
    return value.get("graph"), value.get("stream"), value.get("repeat_id")


def command_argument(command, name):
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


class Audit:
    def __init__(self):
        self.checks = 0
        self.issues = []
        self.inputs = {}

    def check(self, condition, scope, code, detail=None):
        self.checks += 1
        if not condition:
            self.issues.append(dict(scope=scope, code=code, detail=detail))
        return bool(condition)

    def read(self, path, scope, required=True):
        path = Path(path)
        if not path.is_file():
            if required:
                self.check(False, scope, "missing_file", relative(path))
            return None
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            self.check(False, scope, "unreadable_json", repr(exc))
            return None
        self.inputs[relative(path)] = sha(path)
        return value

    def phases(self, values, total, scope):
        if not self.check(isinstance(values, dict), scope, "phase_mapping"):
            return
        valid = all(finite(v) and v >= -1e-8 for v in values.values())
        self.check(valid, scope, "finite_nonnegative_phases", values if not valid else None)
        self.check(valid and near(sum(values.values()), total), scope, "exclusive_phase_sum",
                   dict(sum=sum(values.values()) if valid else None, total=total))


def validate_request(audit, trace, request, index, fixture, contracts, scope):
    q = request_position(request, index)
    audit.check(q == index, scope, "ordered_query_position", q)
    expected_opinion = fixture["opinion_hashes"][index - 1] if 1 <= index <= len(fixture["opinion_hashes"]) else None
    audit.check(request.get("opinion_hash") == expected_opinion, scope, "opinion_binding")
    audit.check(request.get("root_version_id") == trace.get("root_version_id"), scope, "root_reset")
    audit.check(request.get("method") == trace.get("method"), scope, "authentic_method")
    audit.check(request.get("budget") == contracts["max_commits"], scope, "request_budget")
    audit.check(request.get("delta") == contracts["delta"], scope, "request_delta")
    audit.check(request.get("gain_denominator") == contracts["proof_unit"] ** 2, scope, "physical_gain_denominator")
    audit.check(request.get("initial_gram_count") == 0, scope, "empty_initial_gram_cache")
    audit.check(request.get("result_status") == STATUS_MAP.get(request.get("status")), scope, "request_status_mapping")
    rows = request.get("decisions")
    if not audit.check(isinstance(rows, list), scope, "decision_list"):
        return
    current = trace.get("root_version_id")
    chosen = set()
    gain_low = gain_high = 0
    terminated = False
    n, m = trace.get("n"), trace.get("catalog_size")
    for epoch, row in enumerate(rows, 1):
        here = f"{scope}/epoch{epoch}"
        audit.check(isinstance(row, dict), here, "decision_object")
        if not isinstance(row, dict):
            continue
        audit.check(not terminated, here, "no_decision_after_terminal")
        audit.check(row.get("epoch") == epoch, here, "epoch_order")
        audit.check(row.get("root_version_id") == trace.get("root_version_id"), here, "decision_root")
        audit.check(row.get("parent_version_id") == current, here, "parent_version_chain")
        if "live_catalog" in row:
            audit.check(row["live_catalog"] == m - len(chosen), here, "live_catalog_count")
        if row.get("status") == "accepted":
            e, i, j, d = (row.get(k) for k in ("winner_id", "i", "j", "d"))
            integral = all(is_int(v) for v in (e, i, j, d)) and is_int(n)
            audit.check(integral and 0 <= i < j < n and e == i * n + j and d > 0,
                        here, "canonical_edit_encoding", {k: row.get(k) for k in ("winner_id", "i", "j", "d")})
            audit.check(row.get("flip") in (0, 1), here, "edit_type")
            audit.check(e not in chosen, here, "single_use_edit")
            audit.check(row.get("proof_scope") == "full_catalog", here, "declared_catalog_scope")
            lo, hi = row.get("low"), row.get("high")
            good_gain = is_int(lo) and is_int(hi) and 0 < lo <= hi
            audit.check(good_gain, here, "positive_ordered_published_gain")
            if good_gain:
                gain_low += lo
                gain_high += hi
            child = hashlib.sha256(f"{current}/{e}".encode()).hexdigest()
            audit.check(row.get("current_version_id") == child, here, "child_version_hash")
            current = child
            chosen.add(e)
        else:
            terminated = True
            audit.check(row.get("status") == request.get("status"), here, "terminal_status_consistency")
            audit.check(row.get("current_version_id") == current, here, "terminal_preserves_parent")
            audit.check(row.get("winner_id") is None, here, "terminal_no_accepted_winner")
    audit.check(request.get("commits") == len(chosen), scope, "accepted_commit_count")
    audit.check(request.get("current_version_id") == current, scope, "final_version")
    audit.check(request.get("gain_low") == gain_low and request.get("gain_high") == gain_high,
                scope, "published_gain_sum")
    if request.get("status") == "budget_stop":
        audit.check(len(chosen) == contracts["max_commits"] and not terminated, scope, "budget_exhaustion")
    else:
        audit.check(terminated, scope, "terminal_decision_present")
    audit.phases(request.get("phases"), request.get("request_seconds"), scope + "/phases")
    audit.phases(request.get("online_phases"), request.get("online_seconds"), scope + "/online_phases")
    audit.check(near(request.get("request_seconds"),
                     request.get("online_seconds", 0) + request.get("reset_opinion_seconds", 0)),
                scope, "online_reset_sum")


def validate_trace(audit, trace, fixture, graph, contracts, scope):
    for field in FIXTURE_BINDINGS:
        audit.check(trace.get("fixture", {}).get(field) == fixture.get(field), scope, "fixture_" + field)
    audit.check(trace.get("graph_hash") == graph["weighted_sha256"], scope, "graph_hash")
    audit.check((trace.get("n"), trace.get("catalog_size")) == (graph["n"], graph["catalog_size"]), scope, "graph_dimensions")
    requests = trace.get("requests")
    if not audit.check(isinstance(requests, list), scope, "request_list"):
        return
    audit.check(len(requests) <= 4, scope, "at_most_four_requests")
    for index, request in enumerate(requests, 1):
        validate_request(audit, trace, request, index, fixture, contracts, f"{scope}/q{index}")


def load_history(audit, frozen, fixtures, contracts):
    history, excluded = [], []
    directory = resolve(frozen["old_raw_dir"])
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith((".job.json", "_job.json")):
            excluded.append(dict(file=relative(path), reason="job specification"))
            continue
        trace = audit.read(path, "V2/" + path.name)
        if not isinstance(trace, dict):
            continue
        if not isinstance(trace.get("requests"), list):
            excluded.append(dict(file=relative(path), reason="requests is not a result list"))
            continue
        cell = trace.get("graph"), trace.get("stream")
        if cell not in fixtures:
            excluded.append(dict(file=relative(path), reason="outside frozen cells"))
            continue
        scope = "V2/" + path.name
        audit.check(trace.get("input_manifest_sha256") == frozen["v2_input_manifest_sha256"], scope, "input_manifest_binding")
        validate_trace(audit, trace, fixtures[cell], frozen["input_graphs"][cell[0]], contracts, scope)
        history.append(dict(path=relative(path), sha256=sha(path), trace=trace))
    roster = Counter((h["trace"].get("graph"), h["trace"].get("stream"), h["trace"].get("method"), h["trace"].get("repeat_id")) for h in history)
    expected = {(graph, stream, method, repeat): 1 for graph, stream in fixtures
                for method in ("D", "S", "H") for repeat in range(3)}
    audit.check(dict(roster) == expected, "V2", "complete_retained_54_worker_roster",
                dict(observed=len(history), expected=len(expected)))
    return history, excluded


def get_supervisor(audit, job, raw_path):
    path = WORKSPACE / "v3/runs" / ("hg_" + Path(job["output_file"]).stem) / "record.json"
    record = audit.read(path, "supervisor/" + str(job["position"]), required=False)
    if record is not None:
        return record, relative(path)
    ledger = WORKSPACE / "v3/budget_ledger.jsonl"
    matches = []
    if ledger.is_file():
        with ledger.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                target = command_argument(record.get("command", []), "--output")
                if target is not None:
                    absolute = Path(target) if Path(target).is_absolute() else Path(record["cwd"]) / target
                    if absolute.resolve() == raw_path.resolve():
                        matches.append(record)
    audit.check(len(matches) <= 1, "supervisor/" + str(job["position"]), "unique_execution", len(matches))
    return (matches[0], relative(ledger)) if len(matches) == 1 else (None, None)


def validate_supervisor(audit, record, job, path, freeze_path, frozen, scope):
    command = record.get("command", [])
    output_arg = command_argument(command, "--output")
    output = Path(output_arg) if output_arg else None
    if output is not None and not output.is_absolute():
        output = Path(record.get("cwd", "")) / output
    audit.check(output is not None and output.resolve() == path.resolve(), scope, "supervisor_output_binding")
    audit.check(command_argument(command, "--job") == str(job["position"]), scope, "supervisor_job_binding")
    declared_freeze = command_argument(command, "--freeze")
    audit.check(declared_freeze is not None and resolve(declared_freeze) == freeze_path.resolve(), scope, "supervisor_freeze_path")
    audit.check(record.get("runtime_git_commit") == frozen["runtime_commit"], scope, "supervisor_commit")
    audit.check(record.get("cpu_affinity") == [frozen["resources"]["cpu"]], scope, "supervisor_cpu")
    audit.check(record.get("memory_limit_bytes") == frozen["resources"]["memory_bytes"], scope, "supervisor_memory_limit")
    audit.check(record.get("numeric_threads") == frozen["resources"]["numerical_threads"], scope, "supervisor_threads")
    audit.check(record.get("timeout_seconds") == frozen["resources"]["seconds"], scope, "supervisor_timeout")
    files = record.get("runtime_files", {})
    audit.check(bool(files) and map_sha(files) == record.get("runtime_source_sha256"), scope, "supervisor_source_digest")
    mismatches = [name for name, digest in files.items() if frozen["source_files_sha256"].get(name) != digest]
    audit.check(not mismatches, scope, "supervisor_source_members", mismatches)
    bound = record.get("input_and_protocol_bindings", {})
    audit.check(bound.get(str(freeze_path.resolve())) == sha(freeze_path), scope, "supervisor_freeze_hash")
    audit.check(finite(record.get("wall_seconds")) and record["wall_seconds"] >= 0, scope, "supervisor_wall")


def effective_status(trace, supervisor):
    if supervisor and supervisor.get("status") == "timeout":
        return "timeout"
    if trace is None:
        return "missing_result" if supervisor else "not_started"
    status = trace.get("status", "unknown")
    if supervisor and supervisor.get("status") == "nonzero_exit" and status in ("starting", "in_progress"):
        return "nonzero_exit_inflight_unknown_cause"
    return status


def current_worker(audit, job, path, frozen, freeze_path, fixtures, contracts):
    scope = "current/" + path.name
    before = len(audit.issues)
    trace = audit.read(path, scope, required=False)
    supervisor, supervisor_path = get_supervisor(audit, job, path)
    if supervisor:
        validate_supervisor(audit, supervisor, job, path, freeze_path, frozen, scope)
    row = {key: job.get(key) for key in ("position", "cell_index", "graph", "stream", "method", "repeat_id")}
    row.update(file=relative(path), file_sha256=sha(path) if path.is_file() else None,
               supervisor_file=supervisor_path, supervisor_status=None if supervisor is None else supervisor.get("status"),
               supervisor_exit_code=None if supervisor is None else supervisor.get("exit_code"),
               supervisor_wall_seconds=None if supervisor is None else supervisor.get("wall_seconds"),
               sampled_peak_rss_bytes=None if supervisor is None else supervisor.get("sampled_peak_rss_bytes"),
               sampled_peak_vms_bytes=None if supervisor is None else supervisor.get("sampled_peak_vms_bytes"),
               status=effective_status(trace, supervisor), recorded_requests=0, accepted_decisions=0,
               computed_complete_Q4=False, binding_and_accounting_valid=False, new_accepted_decisions=0)
    if trace is None:
        return dict(job=job, trace=None, row=row, requests={}, supervisor=supervisor)
    if not audit.check(isinstance(trace, dict) and isinstance(trace.get("requests"), list), scope, "worker_result_object"):
        return dict(job=job, trace=None, row=row, requests={}, supervisor=supervisor)
    audit.check(trace.get("job") == job, scope, "frozen_job_identity")
    for field in ("graph", "stream", "method", "repeat_id"):
        audit.check(trace.get(field) == job[field], scope, "worker_" + field)
    for field, expected in (("freeze_sha256", sha(freeze_path)), ("actual_runtime_commit", frozen["runtime_commit"]),
                            ("declared_runtime_commit", frozen["runtime_commit"]), ("source_digest", frozen["source_digest"]),
                            ("input_manifest_sha256", frozen["v2_input_manifest_sha256"]), ("environment", frozen["environment"])):
        audit.check(trace.get(field) == expected, scope, field)
    pools = trace.get("enforced_threadpools")
    audit.check(isinstance(pools, list) and bool(pools) and all(p.get("num_threads") == 1 for p in pools), scope, "enforced_threadpool_state")
    fixture = fixtures[(job["graph"], job["stream"])]
    validate_trace(audit, trace, fixture, frozen["input_graphs"][job["graph"]], contracts, scope)
    requests = {request_position(q, i): q for i, q in enumerate(trace["requests"], 1)}
    audit.check(len(requests) == len(trace["requests"]), scope, "unique_request_positions")
    complete = set(requests) == {1, 2, 3, 4} and all(q.get("result_status") in COMPLETE for q in requests.values())
    terminal = trace.get("status") not in ("starting", "in_progress")
    if terminal and trace.get("status") in COMPLETE | {"unresolved"}:
        audit.check(trace.get("complete_Q4") == complete, scope, "complete_Q4_recomputed")
    if complete and trace.get("status") in COMPLETE:
        audit.check(near(trace.get("complete_service_seconds"), trace.get("service_accounted_seconds")), scope, "complete_service_cost")
    elif trace.get("complete_service_seconds") is not None:
        audit.check(False, scope, "partial_worker_has_complete_cost")
    structural = trace.get("structural_seconds")
    if structural is not None:
        audit.phases(trace.get("structural_phases"), structural, scope + "/structural_phases")
        expected_cost = structural + sum(q.get("request_seconds", 0) for q in requests.values())
        audit.check(near(trace.get("service_accounted_seconds"), expected_cost), scope, "recorded_service_cost")
        running = structural
        for i, q in sorted(requests.items()):
            running += q.get("request_seconds", 0)
            audit.check(near(trace.get("prefixes", {}).get(str(i)), running), scope + f"/q{i}", "paid_prefix_cost")
        if terminal and not complete and trace.get("partial_work_seconds") is not None:
            audit.check(near(trace.get("partial_work_seconds"), expected_cost), scope, "partial_recorded_prefix_cost")
    if trace.get("service_plus_input_binding_seconds") is not None:
        audit.check(near(trace["service_plus_input_binding_seconds"],
                         trace.get("service_accounted_seconds", 0) + trace.get("opinion_and_graph_binding_seconds", 0)),
                    scope, "service_plus_binding_cost")
    for name in ("process_wall_seconds", "binding_and_import_seconds", "opinion_and_graph_binding_seconds"):
        if name in trace:
            audit.check(finite(trace[name]) and trace[name] >= 0, scope, "nonnegative_" + name)
    for name in ("structural_seconds", "service_accounted_seconds", "complete_service_seconds", "partial_work_seconds",
                 "binding_and_import_seconds", "opinion_and_graph_binding_seconds", "service_plus_input_binding_seconds",
                 "process_wall_seconds", "process_peak_rss_bytes", "root_factor_nnz", "active_phase"):
        row[name] = trace.get(name)
    row.update(raw_status=trace.get("status"), recorded_requests=len(requests),
               accepted_decisions=sum(len(accepted(q)) for q in requests.values()), computed_complete_Q4=complete,
               binding_and_accounting_valid=len(audit.issues) == before,
               partial_cost_scope=trace.get("partial_cost_scope", "all recorded request/setup work; no completed-Q4 claim" if not complete else None))
    for name, value in trace.get("metering", {}).items():
        row["metered_" + name] = value
    return dict(job=job, trace=trace, row=row, requests=requests, supervisor=supervisor)


def compare_requests(audit, left, right, scope, left_label, right_label, differences):
    for key in ("root_version_id", "opinion_hash", "budget", "delta", "gain_denominator"):
        audit.check(left.get(key) == right.get(key), scope, "request_contract_" + key)
    a, b = accepted(left), accepted(right)
    same = True
    for index, (x, y) in enumerate(zip(a, b), 1):
        equal = identity(x) == identity(y)
        same = same and equal
        audit.check(equal, scope + f"/epoch{index}", "accepted_edit_version_agreement",
                    None if equal else dict(left=dict(zip(EDIT_FIELDS, identity(x))), right=dict(zip(EDIT_FIELDS, identity(y)))))
        if equal:
            for field in ("low", "high"):
                if x.get(field) != y.get(field):
                    differences.append(dict(comparison=scope.split("/")[0], left=left_label, right=right_label,
                                            query_position=left.get("query_position"), epoch=index,
                                            parent_version_id=x.get("parent_version_id"), winner_id=x.get("winner_id"),
                                            endpoint=field, left_value=x.get(field), right_value=y.get(field),
                                            difference=x[field] - y[field], denominator=left.get("gain_denominator"),
                                            implication="endpoint difference only; numerical validity not re-proved"))
    full_equal = same and len(a) == len(b)
    if left.get("result_status") in COMPLETE and right.get("result_status") in COMPLETE:
        audit.check(full_equal, scope, "complete_accepted_trajectory_agreement")
        audit.check(left.get("status") == right.get("status"), scope, "complete_termination_agreement")
    if len(a) < len(b) and left.get("status") == "tolerance_stop":
        audit.check(False, scope, "stop_before_other_recorded_positive_acceptance")
    if len(b) < len(a) and right.get("status") == "tolerance_stop":
        audit.check(False, scope, "stop_before_other_recorded_positive_acceptance")
    return dict(shared_prefix_equal=same, full_accepted_sequence_equal=full_equal,
                left_accepts=len(a), right_accepts=len(b), left_status=left.get("status"), right_status=right.get("status"))


def decision_audit(audit, workers, history):
    history_by_cell = defaultdict(list)
    contexts = defaultdict(list)
    for item in history:
        trace = item["trace"]
        history_by_cell[(trace["graph"], trace["stream"])].append(item)
        for q in trace["requests"]:
            for d in accepted(q):
                context = trace["graph"], trace["stream"], q["opinion_hash"], d["parent_version_id"]
                contexts[context].append((identity(d), item["path"]))
    comparisons, decisions, new, differences = [], [], [], []
    for worker in workers:
        trace, job = worker["trace"], worker["job"]
        if trace is None:
            continue
        references = history_by_cell[(job["graph"], job["stream"])]
        for reference in references:
            for field in TOP_BINDINGS:
                audit.check(trace.get(field) == reference["trace"].get(field), "current_vs_V2/" + worker["row"]["file"], "top_binding_" + field)
            ref_requests = {request_position(q, i): q for i, q in enumerate(reference["trace"]["requests"], 1)}
            for position, request in worker["requests"].items():
                ref = ref_requests.get(position)
                if ref is None:
                    continue
                scope = f"current_vs_V2/{job['position']}/{Path(reference['path']).name}/q{position}"
                result = compare_requests(audit, request, ref, scope, worker["row"]["file"], reference["path"], differences)
                comparisons.append(dict(comparison="current_vs_V2", left=worker["row"]["file"], right=reference["path"], query_position=position, **result))
        for position, request in worker["requests"].items():
            for d in accepted(request):
                context = job["graph"], job["stream"], request["opinion_hash"], d["parent_version_id"]
                old = contexts.get(context, [])
                matching = sorted({path for edit, path in old if edit == identity(d)})
                classification = "recorded_V2_acceptance" if matching else "conflicting_V2_acceptance" if old else "new_acceptance_needs_independent_replay"
                if old and not matching:
                    audit.check(False, f"decision/{job['position']}/q{position}/epoch{d['epoch']}", "accepted_edit_conflicts_at_known_parent")
                record = dict(position=job["position"], graph=job["graph"], stream=job["stream"], method=job["method"], repeat_id=job["repeat_id"],
                              file=worker["row"]["file"], query_position=position, opinion_hash=request["opinion_hash"], epoch=d["epoch"],
                              classification=classification, matching_historical_sources=matching,
                              **{k: d.get(k) for k in EDIT_FIELDS}, low=d.get("low"), high=d.get("high"), denominator=request.get("gain_denominator"))
                decisions.append(record)
                if not matching:
                    new.append(record)
                    worker["row"]["new_accepted_decisions"] += 1
    groups = defaultdict(dict)
    for worker in workers:
        group = groups[key_of(worker["job"])]
        audit.check(worker["job"]["method"] not in group, "pair_roster", "unique_method_per_cell_repeat")
        group[worker["job"]["method"]] = worker
    pairs = []
    for (graph, stream, repeat), arms in sorted(groups.items()):
        row = dict(graph=graph, stream=stream, repeat_id=repeat, paired_methods_present=set(arms) == {"H", "G"},
                   trajectory_match=None, eligible_complete_Q4_pair=False, H_over_G=None, G_minus_H_seconds=None, G_time_change_percent=None)
        same = True
        if set(arms) == {"H", "G"}:
            h, g = arms["H"], arms["G"]
            if h["trace"] is not None and g["trace"] is not None:
                for field in TOP_BINDINGS:
                    ok = h["trace"].get(field) == g["trace"].get(field)
                    same = same and ok
                    audit.check(ok, f"H_vs_G/{graph}/{stream}/r{repeat}", "top_binding_" + field)
                common = sorted(set(h["requests"]) & set(g["requests"]))
                for q in common:
                    scope = f"H_vs_G/{graph}/{stream}/r{repeat}/q{q}"
                    result = compare_requests(audit, h["requests"][q], g["requests"][q], scope, h["row"]["file"], g["row"]["file"], differences)
                    same = same and result["full_accepted_sequence_equal"]
                    comparisons.append(dict(comparison="H_vs_G", left=h["row"]["file"], right=g["row"]["file"], query_position=q, **result))
                same = same and set(h["requests"]) == set(g["requests"])
                row["trajectory_match"] = same
            for method, worker in arms.items():
                for field in ("status", "computed_complete_Q4", "service_accounted_seconds", "complete_service_seconds", "partial_work_seconds",
                              "process_wall_seconds", "supervisor_wall_seconds", "process_peak_rss_bytes", "accepted_decisions", "new_accepted_decisions"):
                    row[method + "_" + field] = worker["row"].get(field)
            good = same and all(w["row"]["computed_complete_Q4"] and w["row"]["status"] in COMPLETE
                                and w["row"]["binding_and_accounting_valid"]
                                and w["supervisor"] is not None and w["supervisor"].get("status") == "completed"
                                and w["supervisor"].get("exit_code") == 0
                                and finite(w["row"].get("complete_service_seconds")) and w["row"]["complete_service_seconds"] > 0
                                for w in arms.values())
            row["complete_pair_before_global_gate"] = good
        pairs.append(row)
    return comparisons, decisions, new, differences, pairs


def long_tables(workers):
    requests, phases, counters, access, memory = [], [], [], [], []
    for worker in workers:
        common = {k: worker["job"][k] for k in ("position", "graph", "stream", "method", "repeat_id")}
        trace = worker["trace"]
        if trace is None:
            continue
        for name, value in trace.get("structural_phases", {}).items():
            phases.append(dict(**common, query_position=0, phase=name, seconds=value, scope="exclusive_structural"))
        for q, request in sorted(worker["requests"].items()):
            base = dict(**common, query_position=q)
            requests.append(dict(**base, status=request.get("status"), result_status=request.get("result_status"),
                                 commits=request.get("commits"), request_seconds=request.get("request_seconds"),
                                 online_seconds=request.get("online_seconds"), reset_opinion_seconds=request.get("reset_opinion_seconds"),
                                 wrapper_seconds=request.get("wrapper_seconds"), opinion_hash=request.get("opinion_hash"),
                                 root_version_id=request.get("root_version_id"), current_version_id=request.get("current_version_id"),
                                 failure_reason=request.get("failure_reason"), **{k: request.get("counters", {}).get(k) for k in
                                     ("candidate_rhs", "winner_rhs", "state_rhs", "response_blocks", "score_calls", "gram_materializations", "gram_hits", "transfers")}))
            for name, value in request.get("phases", {}).items():
                phases.append(dict(**base, phase=name, seconds=value, scope="exclusive_request"))
            for name, value in request.get("counters", {}).items():
                counters.append(dict(**base, counter=name, value=value, scope="request_counter"))
            for epoch in request.get("access", {}).get("epochs", []):
                for name, value in epoch.items():
                    if isinstance(value, (int, float, bool)) and name != "epoch":
                        access.append(dict(**base, epoch=epoch.get("epoch"), parent_version_id=epoch.get("parent_version_id"),
                                           counter=name, value=value, scope="access_event_observation"))
                for phase_set in ("frontier_build_phases", "seed_build_phases"):
                    for name, value in epoch.get(phase_set, {}).items():
                        access.append(dict(**base, epoch=epoch.get("epoch"), parent_version_id=epoch.get("parent_version_id"),
                                           counter=phase_set + "/" + name, value=value,
                                           scope="nested_timing_overlaps_exclusive_phases_do_not_add"))
            for name, value in request.get("memory", {}).items():
                if isinstance(value, (int, float, bool)) or value is None:
                    memory.append(dict(**base, metric=name, value=value, scope="observed_payload_not_process_peak"))
            for name, value in request.get("memory", {}).get("frontier_snapshot", {}).items():
                if isinstance(value, (int, float, bool)):
                    memory.append(dict(**base, metric="frontier_snapshot/" + name, value=value, scope="observed_payload_not_process_peak"))
    return requests, phases, counters, access, memory


def selected_parents(audit, frozen, workers):
    path = resolve(frozen["source_selection_file"])
    selection = audit.read(path, "selected_parents")
    audit.check(path.is_file() and sha(path) == frozen["source_selection_sha256"], "selected_parents", "frozen_selection_hash")
    if selection is None:
        return []
    rows = []
    audit.check(len(selection.get("rows", [])) == 7, "selected_parents", "prespecified_seven_rows")
    for item in selection.get("rows", []):
        row = dict(selection_id=item["selection_id"], graph=item["graph"], stream=item["stream"],
                   query_position=item["query_position"], epoch=item["epoch"], parent_version_id=item["parent_version_id"],
                   opinion_hash=item["opinion_hash"], historical_source=item["source"],
                   historical_selection_status=item["status"], observation_status="not_reached_or_unavailable",
                   competition_classification="not_established_by_a_failed_global_bound", observations=[])
        source = resolve(item["source"])
        audit.check(source.is_file() and sha(source) == item["source_sha256"], item["selection_id"], "selected_historical_source_hash")
        for worker in workers:
            job = worker["job"]
            if job["method"] != "G" or (job["graph"], job["stream"], job["repeat_id"]) != (item["graph"], item["stream"], item["repeat_id"]):
                continue
            request = worker["requests"].get(item["query_position"])
            if request is None:
                continue
            prefix = [d["winner_id"] for d in accepted(request) if d["epoch"] < item["epoch"]]
            if request.get("opinion_hash") != item["opinion_hash"] or prefix != item["chosen_prefix"]:
                audit.check(False, item["selection_id"], "selected_parent_input_prefix_binding")
                continue
            events = [e for e in request.get("access", {}).get("epochs", [])
                      if e.get("epoch") == item["epoch"] and e.get("parent_version_id") == item["parent_version_id"]]
            audit.check(len(events) <= 1, item["selection_id"], "unique_selected_event")
            for event in events:
                cap = event.get("g_global_insertion_upper_coarse")
                physical = None if cap is None else Fraction(frozen["contracts"]["operator_scale"] * cap, frozen["contracts"]["screen_unit"] ** 2)
                observation = dict(file=worker["row"]["file"], file_sha256=worker["row"]["file_sha256"],
                                   source_complete=event.get("source_complete"),
                                   global_insertion_upper_coarse=cap,
                                   global_insertion_upper_physical=None if physical is None else dict(numerator=physical.numerator, denominator=physical.denominator, value=float(physical)),
                                   whole_insertion_excluded=event.get("g_whole_insertion_excluded"),
                                   cold_start_lower_fine=event.get("cold_start_lower_fine"), cold_end_lower_fine=event.get("cold_end_lower_fine"),
                                   remaining_upper_coarse=event.get("remaining_upper_coarse"),
                                   recorded_event=event,
                                   scope="observed G certificate event under source binding; parent center/radius not independently reconstructed here")
                if cap is not None:
                    audit.check(is_int(cap) and cap >= 0, item["selection_id"], "nonnegative_global_cap")
                    if event.get("g_whole_insertion_excluded"):
                        lower = event.get("cold_end_lower_fine")
                        ratio = (frozen["contracts"]["proof_unit"] // frozen["contracts"]["screen_unit"]) ** 2
                        audit.check(is_int(lower) and (cap * ratio < lower or lower == 0 and cap <= 0),
                                    item["selection_id"], "recorded_global_exclusion_common_units")
                row["observations"].append(observation)
                row["observation_status"] = "observed_bound" if cap is not None else "bound_not_reached"
        rows.append(row)
    return rows


def write_csv(path, rows, default_fields=()):
    fields = list(default_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def grouped_totals(rows, name, value):
    totals = defaultdict(lambda: dict(sum=0, observations=0))
    for row in rows:
        item = totals[(row["method"], row[name])]
        if finite(row[value]) or isinstance(row[value], bool):
            item["sum"] += row[value]
            item["observations"] += 1
    return [dict(method=method, metric=metric, **data) for (method, metric), data in sorted(totals.items())]


def repeat_differences(workers):
    groups = defaultdict(list)
    for worker in workers:
        row = worker["row"]
        groups[(row["graph"], row["stream"], row["method"])].append(row)
    rows = []
    for (graph, stream, method), group in sorted(groups.items()):
        complete = [row for row in group if row["eligible_complete_Q4_worker"]]
        values = [row["complete_service_seconds"] for row in complete]
        rows.append(dict(graph=graph, stream=stream, method=method, observed_repeats=len(group), eligible_complete_repeats=len(complete),
                         repeat_ids=[row["repeat_id"] for row in complete], complete_seconds=values,
                         maximum_minus_minimum_seconds=max(values) - min(values) if len(values) >= 2 else None,
                         maximum_over_minimum=max(values) / min(values) if len(values) >= 2 else None,
                         scope="current frozen environment only; one repeat cannot estimate repeat variability"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repeat-differences", type=Path, help="optional current-environment repeat-difference CSV")
    args = parser.parse_args()
    args.freeze = args.freeze.resolve()
    args.raw_dir = args.raw_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit = Audit()
    frozen = audit.read(args.freeze, "freeze")
    if frozen is None:
        (args.out_dir / "hg_summary.json").write_text(json.dumps(dict(status="invalid_freeze", issues=audit.issues), indent=2) + "\n")
        return 2
    contracts = frozen["contracts"]
    audit.check(map_sha(frozen["source_files_sha256"]) == frozen["source_digest"], "freeze", "source_map_digest")
    audit.check([j["position"] for j in frozen["jobs"]] == list(range(len(frozen["jobs"]))), "freeze", "contiguous_job_roster")
    audit.check(all(j["method"] in ("H", "G") and j["requests"] == 4 and j["budget_per_request"] == contracts["max_commits"] for j in frozen["jobs"]), "freeze", "job_contracts")
    for name, digest in frozen["source_files_sha256"].items():
        path = WORKSPACE / "v3_runtime" / name
        audit.check(path.is_file() and sha(path) == digest, "source", "frozen_member", name)
    manifest_path = WORKSPACE / "v3_runtime/v2/INPUTS.json"
    manifest = audit.read(manifest_path, "input_manifest")
    audit.check(manifest_path.is_file() and sha(manifest_path) == frozen["v2_input_manifest_sha256"], "input_manifest", "frozen_hash")
    fixtures = {(f["graph"], f["stream"]): f for f in frozen["input_fixtures"]}
    expected_cells = {(graph, stream) for graph in ("Alpha", "OTC")
                      for stream in ("slow", "independent", "signed-neighbor")}
    audit.check(set(fixtures) == expected_cells and len(frozen["input_fixtures"]) == 6,
                "freeze", "exact_six_old_cell_roster")
    audit.check(set(frozen["input_graphs"]) == {"Alpha", "OTC"}, "freeze", "exact_two_old_graph_roster")
    audit.check({(j["graph"], j["stream"]) for j in frozen["jobs"]} == expected_cells,
                "freeze", "job_cells_equal_frozen_six")
    if manifest is not None:
        actual = {(f["graph"], f["stream"]): f for f in manifest["fixtures"]}
        audit.check(all(actual.get(cell) == fixture for cell, fixture in fixtures.items()),
                    "input_manifest", "exact_selected_frozen_fixtures")
        audit.check(all(manifest["graphs"].get(name) == graph for name, graph in frozen["input_graphs"].items()),
                    "input_manifest", "exact_selected_frozen_graphs")
    history, excluded_history = load_history(audit, frozen, fixtures, contracts)
    workers = [current_worker(audit, job, args.raw_dir / Path(job["output_file"]).name, frozen, args.freeze, fixtures, contracts)
               for job in frozen["jobs"]]
    comparisons, decisions, new, differences, pairs = decision_audit(audit, workers, history)
    selections = selected_parents(audit, frozen, workers)
    requests, phases, counters, access, memory = long_tables(workers)
    gate_clear = not audit.issues and not new
    for worker in workers:
        row = worker["row"]
        row["eligible_complete_Q4_worker"] = bool(gate_clear and row["computed_complete_Q4"] and row["status"] in COMPLETE
            and row["binding_and_accounting_valid"] and worker["supervisor"] is not None
            and worker["supervisor"].get("status") == "completed" and worker["supervisor"].get("exit_code") == 0)
    for pair in pairs:
        pair["eligible_complete_Q4_pair"] = bool(gate_clear and pair.get("complete_pair_before_global_gate"))
        if pair["eligible_complete_Q4_pair"]:
            h, g = pair["H_complete_service_seconds"], pair["G_complete_service_seconds"]
            pair.update(H_over_G=h / g, G_minus_H_seconds=g - h, G_time_change_percent=100 * (g / h - 1))
        pair["ratio_scope"] = "same complete Q4 service; no partial-cost ratio" if pair["eligible_complete_Q4_pair"] else "no eligible complete-Q4 ratio"
    eligible = [p for p in pairs if p["eligible_complete_Q4_pair"]]
    counts = dict(workers_scheduled=len(workers), workers_with_results=sum(w["trace"] is not None for w in workers),
                  complete_Q4_workers=sum(w["row"]["computed_complete_Q4"] for w in workers),
                  partial_or_missing_workers=sum(not w["row"]["computed_complete_Q4"] for w in workers),
                  requests_recorded=len(requests), completed_requests=sum(r["result_status"] in COMPLETE for r in requests),
                  unresolved_requests=sum(r["result_status"] == "unresolved" for r in requests),
                  accepted_decisions=len(decisions), new_or_conflicting_acceptances=len(new),
                  distinct_new_or_conflicting_acceptances=len({(d["graph"], d["stream"], d["opinion_hash"], d["parent_version_id"], d["winner_id"]) for d in new}),
                  historical_workers=len(history), historical_requests=sum(len(h["trace"]["requests"]) for h in history),
                  historical_accepted_decisions=sum(len(accepted(q)) for h in history for q in h["trace"]["requests"]),
                  historical_job_specs_excluded=len(excluded_history), request_trajectory_comparisons=len(comparisons),
                  eligible_complete_Q4_pairs=len(eligible), selected_parents=len(selections),
                  selected_parents_observed=sum(s["observation_status"] == "observed_bound" for s in selections))
    ratio_summary = None
    if eligible:
        h_total = sum(p["H_complete_service_seconds"] for p in eligible)
        g_total = sum(p["G_complete_service_seconds"] for p in eligible)
        ratio_summary = dict(complete_pairs=len(eligible), H_total_seconds=h_total, G_total_seconds=g_total,
                             H_over_G_ratio_of_sums=h_total / g_total,
                             H_over_G_geometric_mean=math.exp(sum(math.log(p["H_over_G"]) for p in eligible) / len(eligible)),
                             H_over_G_minimum=min(p["H_over_G"] for p in eligible), H_over_G_maximum=max(p["H_over_G"] for p in eligible),
                             scope="descriptive current-environment single paired repeat; no uncertainty interval or significance claim")
    interval_counts = dict(Counter(row["comparison"] for row in differences))
    summary = dict(schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(),
                   analysis_source=relative(Path(__file__)), analysis_source_sha256=sha(Path(__file__)),
                   freeze=relative(args.freeze), freeze_sha256=sha(args.freeze), raw_dir=relative(args.raw_dir),
                   runtime_commit=frozen["runtime_commit"], source_digest=frozen["source_digest"],
                   status="metadata_and_trajectory_gate_clear" if gate_clear else "gate_closed",
                   checks=audit.checks, failed_checks=len(audit.issues), issues=audit.issues,
                   performance_gate_clear=gate_clear, counts=counts, worker_status_counts=dict(Counter(w["row"]["status"] for w in workers)),
                   request_status_counts=dict(Counter(r["result_status"] for r in requests)), ratio_summary=ratio_summary,
                   interval_endpoint_difference_counts=interval_counts,
                   new_decisions_statement="No genuinely new accepted decisions relative to retained V2." if not new else "New or conflicting accepted decisions require independent replay before performance/completion credit.",
                   independent_numerical_replay_performed=False, numerical_endpoint_reproof_performed=False,
                   checker_limitation="The production checker's nonpoint gain path checks interval intersection, not containment of every published endpoint; matching trajectories do not close that limitation.",
                   historical_agreement_scope="All returned accepted edit/parent/child and input/contract identities compared with every retained matching V2 D/S/H repeat. This is regression evidence, not a fresh independent proof of every historical decision.",
                   cost_scope="Only both-complete Q4 pairs have service ratios. All setup/request costs, partial work and supervisor wall remain visible. Metered solves and nested access timings overlap exclusive phases and must not be added twice.",
                   memory_scope="Worker ru_maxrss is the reported process peak; supervisor RSS is a sampled lower observation. Access payload snapshots are not total peak memory or a worst-case memory bound.",
                   selected_parent_scope="Seven preselected parents; recorded G caps converted by 16/SCREEN^2. Failure of the global bound does not prove both families competitive. No numerical parent reconstruction in this analysis.",
                   excluded_historical_files=excluded_history, input_hashes=audit.inputs)
    write_csv(args.out_dir / "hg_workers.csv", [w["row"] for w in workers])
    write_csv(args.out_dir / "hg_requests.csv", requests)
    write_csv(args.out_dir / "hg_pairs.csv", pairs)
    write_csv(args.out_dir / "hg_phase_costs.csv", phases)
    write_csv(args.out_dir / "hg_counters.csv", counters)
    write_csv(args.out_dir / "hg_access_events.csv", access)
    write_csv(args.out_dir / "hg_memory_observations.csv", memory)
    write_csv(args.out_dir / "hg_interval_differences.csv", differences, ("comparison", "left", "right", "query_position", "epoch", "endpoint", "left_value", "right_value", "difference", "denominator"))
    write_csv(args.out_dir / "hg_phase_totals_all_recorded.csv", grouped_totals(phases, "phase", "seconds"))
    write_csv(args.out_dir / "hg_counter_totals_all_recorded.csv", grouped_totals(counters, "counter", "value"))
    for name, rows in (("hg_decision_comparisons.jsonl", decisions), ("hg_request_comparisons.jsonl", comparisons)):
        with (args.out_dir / name).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    for name, value in (("hg_new_decisions.json", new), ("hg_selected_parents.json", selections), ("hg_summary.json", summary)):
        (args.out_dir / name).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    if args.repeat_differences:
        args.repeat_differences.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.repeat_differences, repeat_differences(workers))
    lines = ["# Frozen H/G result audit", "", f"Metadata gate: **{summary['status']}**; {audit.checks:,} checks, {len(audit.issues)} failures.", "",
             f"The frozen roster contains {counts['workers_scheduled']} workers: {counts['complete_Q4_workers']} complete Q4 results and {counts['partial_or_missing_workers']} partial or missing results. "
             f"There are {counts['requests_recorded']} returned requests, {counts['accepted_decisions']} accepted decisions and {counts['eligible_complete_Q4_pairs']} eligible complete Q4 pairs.", "",
             summary["new_decisions_statement"], "", summary["historical_agreement_scope"], "", summary["checker_limitation"], "",
             "## Complete Q4 comparisons", "", "| Graph | Stream | H status | G status | H service s | G service s | H/G |", "|---|---|---|---|---:|---:|---:|"]
    for pair in pairs:
        def number(value):
            return "—" if value is None else f"{value:.6f}"
        lines.append(f"| {pair['graph']} | {pair['stream']} | {pair.get('H_status', 'missing')} | {pair.get('G_status', 'missing')} | "
                     f"{number(pair.get('H_complete_service_seconds'))} | {number(pair.get('G_complete_service_seconds'))} | {number(pair.get('H_over_G'))} |")
    if ratio_summary:
        lines += ["", f"Across the {len(eligible)} eligible pairs, the ratio of summed service times is {ratio_summary['H_over_G_ratio_of_sums']:.6f}; "
                  f"the geometric mean of paired H/G ratios is {ratio_summary['H_over_G_geometric_mean']:.6f}. " + ratio_summary["scope"] + "."]
    lines += ["", "## Costs and limits", "", summary["cost_scope"], "", summary["memory_scope"], "",
              "All scheduled jobs, including unresolved and missing/failed jobs, are retained in `hg_workers.csv`; the supervisor wall is separate from a possibly incomplete service prefix. "
              "The request, phase, counter, access-event and memory tables retain their measurement scopes. Current versus historical timing ratios are not computed.", "",
              "## Prespecified parent observations", "", summary["selected_parent_scope"], "",
              f"Observed caps are available for {counts['selected_parents_observed']} of the {counts['selected_parents']} prespecified rows; full events and exact physical fractions are in `hg_selected_parents.json`.", "",
              "## Interval and trajectory evidence", "", f"Differing endpoint counts by comparison: `{json.dumps(interval_counts, sort_keys=True)}`. "
              "Each difference is retained in `hg_interval_differences.csv`; a difference is not itself an invalid interval. "
              "The decision ledger lists every accepted edit and its matching historical sources, while the request ledger compares complete accepted sequences and terminal outcomes.", "",
              "No numerical solver, gain computation, fresh independent decision checker or numerical import is run by this metadata analysis.", ""]
    if audit.issues:
        lines += ["## Failed metadata checks", "", "See `hg_summary.json` for every failed scope, code and detail. All paired ratios remain withheld.", ""]
    (args.out_dir / "hg_report.md").write_text("\n".join(lines))
    print(json.dumps(dict(status=summary["status"], failed_checks=len(audit.issues), checks=audit.checks, counts=counts, out_dir=str(args.out_dir)), sort_keys=True))
    return 0 if gate_clear else 2


if __name__ == "__main__":
    sys.exit(main())
