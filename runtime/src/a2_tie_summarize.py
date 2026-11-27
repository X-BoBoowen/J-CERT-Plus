"""Describe a finished A2 fixed-input revision retest without running a new gate.

This standard-library-only postprocessor is outside the frozen numerical
runtime. Run it AFTER the batch and the existing analyzer finish. It never
launches workers, imports an execution/checker kernel, reads graph matrices,
repeats a request, or changes the original failed confirmation. Both outputs
are exclusive creates. Missing/invalid workers stay in every planned denominator.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("D", "S", "H")
LABELS = {"D": "D-all", "S": "Stream-all", "H": "Implicit-all"}
TERMINAL_REQUESTS = {"budget_stop", "tolerance_stop"}
TIE_FIELDS = ("structural_tie_attempts", "structural_tie_witnesses",
              "structural_tie_decisions", "structural_tie_canonical_switches")
NUMERIC_FIELDS = (
    "endpoint_checked", "endpoint_excluded", "live_catalog_denominator",
    "gram_materializations", "gram_first", "gram_rematerializations", "gram_hits",
    "response_blocks", "response_residual_matvec_columns", "state_rhs",
    "candidate_rhs", "winner_rhs", "policy_refresh_rhs", "numerical_reanchor_rhs",
    "score_calls", "comparison_calls", "overlap_comparisons", *TIE_FIELDS)
ACCESS_FIELDS = (
    "insertion_emissions", "raw_heap_pops", "legal_insertion_pops", "flips_emitted",
    "flip_cursor_pops", "frontier_rebuilds", "emitted_for_screen",
    "stream_pairs_examined", "stream_insertions_emitted", "stream_flips_emitted",
    "explicit_raw_visits", "processed_insertion_skips", "processed_flip_skips",
    "seed_candidates", "seed_insertions", "seed_flips")
RHS_FIELDS = ("state_rhs", "candidate_rhs", "winner_rhs", "policy_refresh_rhs",
              "numerical_reanchor_rhs")
BYTE_FIELDS = {
    "explicit_catalog_bytes", "structural_weight_grouping", "incidence_bytes",
    "index_private_bytes", "peak_frontier_owned_bytes", "peak_emitted_batch_array_bytes",
    "graph_csr", "root_operator_csr", "gram_python_bytes",
    "actual_candidate_cache_python_bytes", "decision_and_access_log_python_bytes",
    "shared_response_bytes", "root_factor_resident_bytes", "low_rank_steps",
    "peak_low_rank_steps"}
COUNT_FIELDS = {"candidate_array_rows", "full_root_caps_created_rows", "gram_count",
                "peak_gram_count", "root_factor_reported_nnz"}


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def integer(value):
    return type(value) is int and value >= 0


def mapping(value):
    return value if isinstance(value, dict) else {}


def objects(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def status_text(value, missing="missing_status"):
    return value if isinstance(value, str) else missing if value is None else "malformed_status"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: " + key)
        result[key] = value
    return result


def bad_constant(value):
    raise ValueError("nonfinite JSON value: " + value)


class Reader:
    """Keep exact bytes of every successfully read input in the provenance map."""
    def __init__(self):
        self.hashes = {}

    def read(self, path, issues=None):
        path = Path(path).resolve()
        try:
            data = path.read_bytes()
            self.hashes[str(path)] = hashlib.sha256(data).hexdigest()
            value = json.loads(data, object_pairs_hook=unique_object, parse_constant=bad_constant)
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            return value
        except (OSError, UnicodeError, ValueError, TypeError) as error:
            if issues is None:
                raise ValueError(f"cannot read {path}: {error}") from error
            issues.append(f"{path}: {error}")
            return {}

    def unchanged(self):
        for path, expected in self.hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError("input changed while summarizing: " + path)


def stats(values, denominator):
    usable = [value for value in values if number(value)]
    return dict(samples=len(usable), planned_samples=denominator,
                median=statistics.median(usable) if usable else None,
                minimum=min(usable) if usable else None,
                maximum=max(usable) if usable else None)


def job_key(job):
    return tuple(job.get(key) for key in ("graph", "stream", "seed_label", "repeat_id", "method"))


def cell_key(job):
    return tuple(job[key] for key in ("graph", "stream", "seed_label"))


def file_name(position, job):
    return (f"{position:03d}_{job['graph']}_{job['stream']}_"
            f"{job['seed_label']}_{job['label']}.json")


def registered_jobs(cfg):
    """Reconstruct the fixed roster without importing benchmark/runtime modules."""
    if (cfg.get("id") != "A2-exact-tie-retest-v1" or
            cfg.get("stage") != "revision_retest" or cfg.get("unseen_confirmation") is not False or
            cfg.get("graphs") != ["Alpha", "OTC"] or
            cfg.get("streams") != ["slow", "independent", "signed-neighbor"] or
            cfg.get("seed_labels") != ["confirmation-1", "confirmation-2"] or
            cfg.get("repeat_count") != 3 or cfg.get("requests") != 4 or cfg.get("budget") != 15 or
            cfg.get("delta") != 0 or cfg.get("date") != "2026-11-25" or
            cfg.get("methods") != [dict(method=m, label=LABELS[m]) for m in METHODS] or
            mapping(cfg.get("original_confirmation")).get("status") != "failed"):
        raise ValueError("expected the registered 12-cell, 108-worker fixed-input revision protocol")
    cells = list(product(cfg["graphs"], cfg["streams"], cfg["seed_labels"]))
    result = []
    for repeat in range(3):
        for position, (graph, stream, label) in enumerate(cells):
            pivot = (repeat + position) % 3
            order = METHODS[pivot:] + METHODS[:pivot]
            seed_text = f"A2|{label}|2026-11-25|{graph}|{stream}"
            seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")
            for method in order:
                result.append(dict(graph=graph, stream=stream, seed_label=label, seed=seed,
                                   repeat_id=repeat, method=method, label=LABELS[method]))
    return cells, result


def indexed_rows(value, position_key, issues, label):
    rows, duplicates = {}, set()
    if not isinstance(value, list):
        issues.append(label + " is not a list")
        return rows
    for row in value:
        position = mapping(row).get(position_key)
        if not integer(position) or not 0 <= position < 108:
            issues.append(label + " has a malformed/out-of-roster position")
            continue
        if position in rows:
            duplicates.add(position)
        rows[position] = row
    for position in sorted(duplicates):
        issues.append(f"{label} duplicates position {position}")
        del rows[position]
    if len(value) != 108:
        issues.append(f"{label} has {len(value)} entries, expected 108")
    return rows


def record_metric(records, key, record_kind):
    values = [row[key] for row in records if integer(row.get(key))]
    return dict(observed_total=sum(values), records_with_value=len(values),
                recorded_denominator=len(records), record_kind=record_kind,
                complete_record_coverage=bool(records) and len(values) == len(records))


def flatten_numeric(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + str(key)
        if isinstance(item, dict):
            result.update(flatten_numeric(item, name + "."))
        elif number(item):
            result[name] = item
    return result


def worker_work(trace):
    requests = objects(trace.get("requests"))
    counters = [mapping(request.get("counters")) for request in requests]
    events = [event for request in requests
              for event in objects(mapping(request.get("access")).get("epochs"))]
    result = {"numeric." + key: record_metric(counters, key, "recorded requests")
              for key in NUMERIC_FIELDS}
    result.update({"access." + key: record_metric(events, key, "recorded access epochs")
                   for key in ACCESS_FIELDS})
    rhs = [{"rhs": sum(row[key] for key in RHS_FIELDS)} if
           all(integer(row.get(key)) for key in RHS_FIELDS) else {} for row in counters]
    result["numeric.total_scalar_rhs"] = record_metric(rhs, "rhs", "recorded requests")
    return result


def worker_memory(trace):
    samples = [dict(query_position=row.get("query_position"), memory=mapping(row.get("memory")))
               for row in objects(trace.get("requests"))]
    fields = defaultdict(list)
    for row in samples:
        for key, value in flatten_numeric(row["memory"]).items():
            fields[key].append(value)
    return dict(request_snapshots=samples,
                maximum_recorded_numeric_fields={key: max(values) for key, values in fields.items()},
                numeric_field_snapshot_counts={key: len(values) for key, values in fields.items()},
                note="Per-worker maxima across recorded request snapshots; overlapping fields are not summed.")


def describe_worker(position, job, scheduled, analyzed, trace, global_valid, raw_path, reader):
    issues = []
    if not scheduled or scheduled.get("position") != position or scheduled.get("job") != job:
        issues.append("missing or mismatched scheduler entry")
    if scheduled.get("result") != file_name(position, job):
        issues.append("scheduler result filename differs from registered roster")
    if not analyzed or any(analyzed.get(key) != value for key, value in job.items()) or analyzed.get("position") != position:
        issues.append("missing or mismatched analyzer worker entry")
    if (not trace or trace.get("schedule_position") != position or
            any(trace.get(key) != value for key, value in job.items())):
        issues.append("missing or mismatched raw worker identity")
    if analyzed.get("valid") is not True:
        issues.append("existing analyzer did not validate this worker")
    for issue in analyzed.get("issues", []) if isinstance(analyzed.get("issues"), list) else []:
        issues.append("analyzer: " + str(issue))
    requests = objects(trace.get("requests"))
    request_shape_ok = (isinstance(trace.get("requests"), list) and
                        len(requests) == len(trace["requests"]) == 4 and
                        [r.get("query_position") for r in requests] == [1, 2, 3, 4])
    terminal = sum(status_text(r.get("status")) in TERMINAL_REQUESTS for r in requests)
    cost = mapping(trace.get("prefixes")).get("4")
    cost_matches = number(cost) and cost > 0 and number(analyzed.get("Q4_seconds"))
    if cost_matches:
        cost_matches = math.isclose(cost, analyzed["Q4_seconds"], rel_tol=0., abs_tol=1e-7)
    components = [trace.get("structural_seconds"), *[r.get("request_seconds") for r in requests]]
    reconciled = (request_shape_ok and all(number(v) for v in components) and number(cost) and
                  math.isclose(sum(components), cost, rel_tol=0., abs_tol=1e-7))
    if analyzed.get("complete") is True and (not request_shape_ok or terminal != 4 or
                                            not cost_matches or not reconciled):
        issues.append("analyzer complete flag disagrees with raw request/cost accounting")
    complete = (global_valid and not issues and analyzed.get("complete") is True and
                trace.get("status") == scheduled.get("status") == "completed" and
                scheduled.get("returncode") == 0 and request_shape_ok and terminal == 4 and
                cost_matches and reconciled)
    if not global_valid:
        issues.append("global provenance/roster/analyzer binding has an issue; complete cost unavailable")
    decisions = [d for request in requests for d in objects(request.get("decisions"))]
    events = [event for request in requests
              for event in objects(mapping(request.get("access")).get("epochs"))]
    return dict(position=position, **job, raw_file=str(raw_path),
                raw_sha256=reader.hashes.get(str(raw_path.resolve())),
                scheduler_status=status_text(scheduled.get("status"), "missing_result"),
                trace_status=status_text(trace.get("status"), "missing_result"),
                analyzer_valid=analyzed.get("valid") is True, complete=complete,
                issues=issues, failure_reason=trace.get("failure_reason") or scheduled.get("failure_reason"),
                request_statuses=[status_text(r.get("status")) for r in requests],
                planned_requests=4, recorded_requests=len(requests), terminal_requests=terminal,
                recorded_decisions=len(decisions), accepted_decisions=sum(d.get("status") == "accepted" for d in decisions),
                recorded_access_epochs=len(events),
                full_setup_plus_Q4_seconds=cost if complete else None,
                structural_seconds=trace.get("structural_seconds") if number(trace.get("structural_seconds")) else None,
                process_wall_seconds=scheduled.get("process_wall_seconds") if number(scheduled.get("process_wall_seconds")) else None,
                process_peak_rss_bytes=trace.get("process_peak_rss_bytes") if number(trace.get("process_peak_rss_bytes")) else None,
                external_sampled_peak_rss_bytes=scheduled.get("external_sampled_peak_rss_bytes") if number(scheduled.get("external_sampled_peak_rss_bytes")) else None,
                work=worker_work(trace), memory=worker_memory(trace), _trace=trace)


def cell_costs(cell, lookup):
    methods = {}
    for method in METHODS:
        rows = [lookup[(*cell, repeat, method)] for repeat in range(3)]
        values = [row["full_setup_plus_Q4_seconds"] for row in rows]
        full = all(row["complete"] for row in rows)
        methods[method] = dict(planned_repeats=3, complete_repeats=sum(row["complete"] for row in rows),
            seconds_by_repeat=values, median_seconds=statistics.median(values) if full else None,
            minimum_seconds=min(values) if full else None, maximum_seconds=max(values) if full else None,
            available_completed_repeat_statistics=stats(values, 3),
            status_by_repeat=[row["scheduler_status"] for row in rows])
    return dict(zip(("graph", "stream", "seed_label"), cell), methods=methods)


def paired_comparison(cells, lookup, reference):
    """A block is one exact input cell with all THREE matched repeats present.

    H/better requires both references in all three repeats. Never substitute a
    surviving reference when the other failed; record the excluded block.
    """
    blocks, excluded = [], []
    for cell in cells:
        required = ("H", "D", "S") if reference == "better" else ("H", reference)
        missing = [dict(repeat_id=r, method=m) for r in range(3) for m in required
                   if not lookup[(*cell, r, m)]["complete"]]
        label = dict(zip(("graph", "stream", "seed_label"), cell))
        if missing:
            excluded.append(dict(**label, missing_complete_workers=missing))
            continue
        pairs = []
        for repeat in range(3):
            h = lookup[(*cell, repeat, "H")]["full_setup_plus_Q4_seconds"]
            candidates = ("D", "S") if reference == "better" else (reference,)
            method, cost = min(((m, lookup[(*cell, repeat, m)]["full_setup_plus_Q4_seconds"])
                                for m in candidates), key=lambda item: (item[1], item[0]))
            pairs.append(dict(repeat_id=repeat, H_seconds=h, reference_method=method,
                              reference_seconds=cost))
        blocks.append(dict(**label, pairs=pairs))
    pairs = [pair for block in blocks for pair in block["pairs"]]
    hsum = sum(pair["H_seconds"] for pair in pairs) if pairs else None
    rsum = sum(pair["reference_seconds"] for pair in pairs) if pairs else None
    return dict(reference=reference, planned_input_cells=len(cells), matched_whole_input_cells=len(blocks),
                planned_repeat_pairs=3*len(cells), matched_repeat_pairs=len(pairs),
                every_input_cell_matched=len(blocks) == len(cells),
                H_seconds_on_matched_whole_blocks=hsum, reference_seconds_on_matched_whole_blocks=rsum,
                ratio_H_over_reference=hsum/rsum if pairs else None,
                inverse_speedup_reference_over_H=rsum/hsum if pairs else None,
                selected_reference_counts=dict(Counter(p["reference_method"] for p in pairs)),
                matched_blocks=blocks, excluded_blocks=excluded,
                scope="Ratio of sums on whole three-repeat input blocks; reduced coverage is explicit.")


def aggregate_group(name, cells, workers, lookup):
    rows = [row for row in workers if cell_key(row) in set(cells)]
    all_complete = all(row["complete"] for row in rows)
    by_method = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        complete = sum(row["complete"] for row in selected)
        memory_samples = [sample["memory"] for row in selected
                          for sample in row["memory"]["request_snapshots"]]
        work = {}
        names = sorted({key for row in selected for key in row["work"]})
        for key in names:
            measurements = [row["work"][key] for row in selected]
            coverage = all(item["complete_record_coverage"] for item in measurements)
            total = sum(item["observed_total"] for item in measurements)
            work[key] = dict(observed_total=total,
                complete_workload_total=total if complete == len(selected) and coverage else None,
                records_with_value=sum(item["records_with_value"] for item in measurements),
                recorded_denominator=sum(item["recorded_denominator"] for item in measurements),
                record_kind=measurements[0]["record_kind"], complete_record_coverage=coverage)
        fields = sorted({key for row in selected for key in row["memory"]["maximum_recorded_numeric_fields"]})
        memory = {}
        for key in fields:
            values = [row["memory"]["maximum_recorded_numeric_fields"].get(key) for row in selected]
            unit = "bytes" if key in BYTE_FIELDS or key.endswith("_bytes") else "count" if key in COUNT_FIELDS else "as recorded"
            memory[key] = dict(stats(values, len(selected)), unit=unit,
                              statistic="max over recorded request snapshots per worker; distribution across workers")
        by_method[method] = dict(planned_workers=len(selected), complete_workers=complete,
            planned_requests=4*len(selected), recorded_requests=sum(row["recorded_requests"] for row in selected),
            terminal_requests=sum(row["terminal_requests"] for row in selected),
            recorded_decisions=sum(row["recorded_decisions"] for row in selected),
            accepted_decisions=sum(row["accepted_decisions"] for row in selected),
            recorded_access_epochs=sum(row["recorded_access_epochs"] for row in selected),
            process_peak_rss_bytes=stats([row["process_peak_rss_bytes"] for row in selected], len(selected)),
            external_sampled_peak_rss_bytes=stats([row["external_sampled_peak_rss_bytes"] for row in selected], len(selected)),
            work=work, memory_fields=memory,
            memory_declarations=dict(recorded_request_snapshots=len(memory_samples),
                catalog_materialized_true=sum(item.get("catalog_materialized") is True for item in memory_samples),
                catalog_materialized_false=sum(item.get("catalog_materialized") is False for item in memory_samples),
                m_sized_structure_declarations=[json.loads(value) for value in sorted({
                    json.dumps(item.get("m_sized_structures"), sort_keys=True) for item in memory_samples})],
                root_factor_resident_bytes_unknown=sum(item.get("root_factor_resident_bytes") is None
                                                      for item in memory_samples)))
    total_service = {method: sum(row["full_setup_plus_Q4_seconds"] for row in rows if row["method"] == method)
                     if all_complete else None for method in METHODS}
    total_process = {method: sum(row["process_wall_seconds"] for row in rows if row["method"] == method)
                     if all_complete and all(number(row["process_wall_seconds"]) for row in rows) else None
                     for method in METHODS}
    return dict(name=name, planned_input_cells=len(cells), planned_workers=len(rows),
                complete_workers=sum(row["complete"] for row in rows), all_inputs_complete=all_complete,
                full_scope_service_totals_seconds=total_service,
                full_scope_process_totals_seconds=total_process, methods=by_method,
                comparisons={"H_over_" + ref: paired_comparison(cells, lookup, ref)
                             for ref in ("D", "S", "better")})


def tie_summary(workers):
    states, discrepancies, counter_totals = {}, [], dict.fromkeys(TIE_FIELDS, 0)
    evidence_totals = dict.fromkeys(TIE_FIELDS, 0)
    known_counter_requests = dict.fromkeys(TIE_FIELDS, 0)
    for row in workers:
        trace = row["_trace"]
        for request in objects(trace.get("requests")):
            counts = mapping(request.get("counters"))
            derived = dict.fromkeys(TIE_FIELDS, 0)
            for field in TIE_FIELDS:
                if integer(counts.get(field)):
                    counter_totals[field] += counts[field]
                    known_counter_requests[field] += 1
            for decision_ordinal, decision in enumerate(objects(request.get("decisions")), 1):
                key = (row["graph"], row["stream"], row["seed_label"], trace.get("graph_hash"),
                       trace.get("fixture_npz_sha256"), request.get("opinion_hash"),
                       request.get("query_position"), decision.get("epoch"))
                # Missing identity fields are retained as separate observations,
                # not merged into an apparent common exact-input state.
                if (not all(isinstance(v, str) and v for v in key[:6]) or
                        not all(integer(v) and v > 0 for v in key[6:])):
                    key = (*key, "unbound_worker", row["position"], decision_ordinal)
                key = json.dumps(key, sort_keys=True)
                state = states.setdefault(key, dict(graph=row["graph"], stream=row["stream"],
                    seed_label=row["seed_label"], graph_sha256=trace.get("graph_hash"),
                    fixture_npz_sha256=trace.get("fixture_npz_sha256"), opinion_sha256=request.get("opinion_hash"),
                    query_position=request.get("query_position"), epoch=decision.get("epoch"),
                    expected_method_repeat_exposures=9, decision_exposures=[], observations=[]))
                exposure = dict(position=row["position"], method=row["method"], repeat_id=row["repeat_id"],
                                parent_version_id=decision.get("parent_version_id"),
                                status=decision.get("status"), winner_id=decision.get("winner_id"))
                state["decision_exposures"].append(exposure)
                if "comparison_resolution" not in decision:
                    continue
                proof = mapping(decision.get("comparison_resolution"))
                overlaps = proof.get("overlap_ids") if isinstance(proof.get("overlap_ids"), list) else []
                witnesses = proof.get("witnesses") if isinstance(proof.get("witnesses"), list) else []
                success = integer(proof.get("canonical_winner_id"))
                switch = success and proof.get("canonical_winner_id") != proof.get("provisional_winner_id")
                derived["structural_tie_attempts"] += len(overlaps)
                derived["structural_tie_witnesses"] += len(witnesses)
                derived["structural_tie_decisions"] += int(success)
                derived["structural_tie_canonical_switches"] += int(switch)
                state["observations"].append(dict(**exposure, comparison_resolution=proof,
                                                   canonical_switch=switch))
            for field in TIE_FIELDS:
                evidence_totals[field] += derived[field]
                if not integer(counts.get(field)) or counts[field] != derived[field]:
                    discrepancies.append(dict(position=row["position"], query_position=request.get("query_position"),
                                              field=field, counter=counts.get(field), evidence=derived[field]))
    kept = []
    totals = dict(input_epoch_states=0, states_with_success=0, states_with_unsupported_overlap=0,
                  states_with_canonical_switch=0, unique_attempted_candidate_pairs=0,
                  unique_certified_relations=0, unique_switch_transitions=0)
    for state in states.values():
        if not state["observations"]:
            continue
        attempts, relations, switches = set(), set(), set()
        unsupported = success = False
        for observation in state["observations"]:
            proof = observation["comparison_resolution"]
            source = proof.get("provisional_winner_id")
            for target in proof.get("overlap_ids", []) if isinstance(proof.get("overlap_ids"), list) else []:
                if integer(source) and integer(target):
                    attempts.add(tuple(sorted((source, target))))
            for witness in objects(proof.get("witnesses")):
                left, right = witness.get("source_candidate_id"), witness.get("target_candidate_id")
                swap = witness.get("swap")
                if (integer(left) and integer(right) and isinstance(swap, list) and len(swap) == 2 and
                        all(integer(v) for v in swap)):
                    relations.add(json.dumps(dict(candidate_pair=sorted((left, right)), swap=sorted(swap),
                        operator_sha256=witness.get("operator_sha256"), opinion_sha256=witness.get("opinion_sha256"),
                        parent_version_id=witness.get("parent_version_id"), kind=witness.get("kind")), sort_keys=True))
            success |= integer(proof.get("canonical_winner_id"))
            unsupported |= bool(proof.get("unresolved_ids"))
            if observation["canonical_switch"] and integer(source):
                switches.add((source, proof["canonical_winner_id"]))
        state.update(unique_attempted_candidate_pairs=[list(pair) for pair in sorted(attempts)],
                     unique_certified_relations=[json.loads(item) for item in sorted(relations)],
                     unique_switch_transitions=[dict(provisional_winner_id=a, canonical_winner_id=b)
                                                for a, b in sorted(switches)],
                     parent_version_ids=sorted({str(item["parent_version_id"]) for item in state["decision_exposures"]}),
                     recorded_method_repeat_exposures=len(state["decision_exposures"]),
                     recorded_tie_exposures=len(state["observations"]))
        totals["input_epoch_states"] += 1
        totals["states_with_success"] += int(success)
        totals["states_with_unsupported_overlap"] += int(unsupported)
        totals["states_with_canonical_switch"] += int(bool(switches))
        totals["unique_attempted_candidate_pairs"] += len(attempts)
        totals["unique_certified_relations"] += len(relations)
        totals["unique_switch_transitions"] += len(switches)
        kept.append(state)
    kept.sort(key=lambda s: (s["graph"], s["stream"], s["seed_label"],
                            s["query_position"] if integer(s["query_position"]) else 10**9,
                            s["epoch"] if integer(s["epoch"]) else 10**9))
    return dict(planned_workers=108, planned_worker_requests=432, planned_unique_opinion_requests=48,
                counter_totals_across_workers=counter_totals, requests_with_counter=known_counter_requests,
                evidence_totals_across_workers=evidence_totals,
                counter_evidence_discrepancies=discrepancies, deduplicated=totals, states=kept,
                scope="Observed declarations, not a new witness verification. Unique states use exact graph/fixture/opinion hashes and input/epoch; method/repeat exposures are not independent cases.")


def original_prefix_comparison(workers, original_raw, reader):
    details, read_issues = [], []

    def requests_by_query(trace, label):
        result, repeated = {}, set()
        values = trace.get("requests")
        if not isinstance(values, list):
            read_issues.append(label + ": requests is missing or is not a list")
            return result
        for request in values:
            query = mapping(request).get("query_position")
            if not integer(query) or not 1 <= query <= 4:
                read_issues.append(label + ": malformed request/query position")
                continue
            if query in result:
                repeated.add(query)
            result[query] = request
        for query in sorted(repeated):
            read_issues.append(f"{label}: duplicate query position {query}")
            del result[query]
        return result

    for row in workers:
        path = original_raw / file_name(row["position"], row)
        old = reader.read(path, read_issues)
        new = row["_trace"]
        old_requests = requests_by_query(old, str(path))
        new_requests = requests_by_query(new, row["raw_file"])
        old_identity_ok = bool(old) and old.get("schedule_position") == row["position"] and all(
            old.get(k) == row[k] for k in ("graph", "stream", "seed_label", "repeat_id", "method", "seed"))
        for query in range(1, 5):
            before, after = old_requests.get(query), new_requests.get(query)
            entry = dict(position=row["position"], graph=row["graph"], stream=row["stream"],
                         seed_label=row["seed_label"], method=row["method"], repeat_id=row["repeat_id"],
                         query_position=query, original_trace=str(path), comparisons=[])
            if before is None or after is None or not old_identity_ok:
                entry.update(status="not_comparable", reason="missing request or original worker identity mismatch",
                             original_request_present=before is not None, revised_request_present=after is not None)
                details.append(entry)
                continue
            if any(not isinstance(request.get("decisions"), list) or
                   any(not isinstance(decision, dict) for decision in request["decisions"])
                   for request in (before, after)):
                entry.update(status="not_comparable", reason="malformed original or revised decision list")
                details.append(entry)
                continue
            left = [d for d in objects(before.get("decisions")) if d.get("status") == "accepted"]
            right = [d for d in objects(after.get("decisions")) if d.get("status") == "accepted"]
            differences = []
            for field in ("opinion_hash", "root_version_id"):
                if not before.get(field) or before.get(field) != after.get(field):
                    differences.append(dict(field=field, original=before.get(field), revised=after.get(field)))
            for index, old_decision in enumerate(left):
                new_decision = right[index] if index < len(right) else {}
                for field in ("epoch", "winner_id", "parent_version_id", "current_version_id"):
                    if old_decision.get(field) != new_decision.get(field):
                        differences.append(dict(prefix_position=index+1, field=field,
                                                original=old_decision.get(field), revised=new_decision.get(field)))
            entry.update(status="matched" if not differences else "mismatch",
                         original_status=before.get("status"), revised_status=after.get("status"),
                         original_accepted_prefix_count=len(left), revised_accepted_count=len(right),
                         original_winner_ids=[d.get("winner_id") for d in left],
                         revised_winner_ids=[d.get("winner_id") for d in right],
                         comparisons=differences)
            details.append(entry)
    return dict(planned_worker_requests=432, status_counts=dict(Counter(r["status"] for r in details)),
                nonempty_original_prefixes=sum(r.get("original_accepted_prefix_count", 0) > 0 for r in details),
                read_issues=read_issues, requests=details,
                scope="Compare original accepted prefixes, IDs and version/opinion binding only. No original timing, missing Q4 cost, or rewritten old outcome is used.")


def build_summary(args):
    reader = Reader()
    cfg = reader.read(args.protocol)
    cells, jobs = registered_jobs(cfg)
    manifest = reader.read(args.manifest)
    analysis = reader.read(args.analysis)
    frozen = reader.read(args.raw / "FROZEN_JOBS.json")
    schedule = reader.read(args.raw / "SCHEDULE_RESULTS.json")
    if schedule.get("status") not in ("completed", "completed_with_failures"):
        raise ValueError("batch is not finished; this postprocessor must not inspect an active run")
    old_roster = reader.read(args.original_raw / "FROZEN_JOBS.json")
    original_refs = cfg["original_confirmation"]
    judgment_path = ROOT / original_refs["judgment"]["path"]
    judgment = reader.read(judgment_path)
    if judgment.get("status") != "failed" or reader.hashes[str(judgment_path.resolve())] != original_refs["judgment"]["sha256"]:
        raise ValueError("original confirmation judgment is not the pinned failed record")
    issues = []
    protocol_sha = reader.hashes[str(args.protocol.resolve())]
    manifest_sha = reader.hashes[str(args.manifest.resolve())]
    if (manifest.get("stage") != "revision_retest" or manifest.get("unseen_confirmation") is not False or
            manifest.get("protocol_sha256") != protocol_sha or manifest.get("original_confirmation_status") != "failed"):
        issues.append("revision manifest labels or protocol binding mismatch")
    if (analysis.get("stage") != "revision_retest" or analysis.get("unseen_confirmation") is not False or
            analysis.get("original_confirmation_status") != "failed" or
            analysis.get("protocol_sha256") != protocol_sha or analysis.get("input_manifest_sha256") != manifest_sha or
            analysis.get("planned_workers") != 108):
        issues.append("analysis provenance/denominator/label mismatch")
    if frozen.get("jobs") != jobs or old_roster.get("jobs") != jobs:
        issues.append("new or original frozen roster differs from the registered 108 jobs")
    if reader.hashes[str((args.original_raw / "FROZEN_JOBS.json").resolve())] != original_refs["frozen_jobs"]["sha256"]:
        issues.append("original frozen roster bytes differ from pinned historical provenance")
    for key, value in (("protocol_sha256", protocol_sha), ("input_manifest_sha256", manifest_sha),
                       ("source_files_sha256", manifest.get("source_files_sha256")),
                       ("input_source_files_sha256", manifest.get("input_source_files_sha256")),
                       ("environment", manifest.get("environment")), ("limits", cfg.get("limits"))):
        if frozen.get(key) != value:
            issues.append("frozen schedule provenance mismatch: " + key)
    # A bad individual result must not erase other analyzer-validated complete
    # costs. Binding failures are global; missing/duplicate indexed rows are
    # retained as failures only for their affected registered positions.
    global_valid = not issues
    scheduled = indexed_rows(schedule.get("jobs"), "position", issues, "scheduler jobs")
    analyzed = indexed_rows(analysis.get("workers"), "position", issues, "analyzer workers")
    workers, raw_issues = [], []
    for position, job in enumerate(jobs):
        path = args.raw / file_name(position, job)
        trace = reader.read(path, raw_issues)
        workers.append(describe_worker(position, job, scheduled.get(position, {}), analyzed.get(position, {}),
                                       trace, global_valid, path, reader))
    lookup = {job_key(row): row for row in workers}
    cell_rows = [cell_costs(cell, lookup) for cell in cells]
    for cell, result in zip(cells, cell_rows):
        result["comparisons"] = {"H_over_" + ref: paired_comparison([cell], lookup, ref)
                                 for ref in ("D", "S", "better")}
    graph_rows = [aggregate_group(graph, [cell for cell in cells if cell[0] == graph], workers, lookup)
                  for graph in cfg["graphs"]]
    overall = aggregate_group("all original inputs", cells, workers, lookup)
    ties = tie_summary(workers)
    prefixes = original_prefix_comparison(workers, args.original_raw, reader)
    all_complete = all(row["complete"] for row in workers)
    result = dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
        stage="revision_retest", study_label="revision retest on diagnosed fixed inputs",
        unseen_confirmation=False, original_confirmation_status="failed", new_acceptance_gate=False,
        denominators=dict(input_cells=12, methods=3, repeats=3, planned_workers=108,
                          planned_worker_requests=432, distinct_fixed_opinion_requests=48),
        completed_batch_status=schedule["status"], all_inputs_complete=all_complete,
        complete_workers=sum(row["complete"] for row in workers),
        complete_workers_by_method={m: sum(row["complete"] for row in workers if row["method"] == m) for m in METHODS},
        scheduler_status_counts=dict(Counter(row["scheduler_status"] for row in workers)),
        recorded_request_status_counts=dict(Counter(status for row in workers for status in row["request_statuses"])),
        provenance_and_roster_issues=issues, raw_read_issues=raw_issues,
        existing_analysis_correctness_accounting_pass=analysis.get("correctness_accounting_pass"),
        existing_analysis_validation_issues=analysis.get("validation_issues", []),
        failures=[{key: value for key, value in row.items() if key not in ("_trace", "memory", "work")}
                  for row in workers if not row["complete"]],
        cells=cell_rows, graphs=graph_rows, overall=overall, structural_ties=ties,
        original_prefix_comparison=prefixes,
        workers=[{key: value for key, value in row.items() if key != "_trace"} for row in workers],
        measurement_definitions=dict(
            cost="Full graph load/setup plus all four complete request walls. Three-repeat cell medians/ranges require all three complete repeats.",
            paired_ratio="sum(H seconds)/sum(reference seconds), and its inverse, on whole matched three-repeat input cells only. Better selects min(D,S) within each repeat only when both references completed.",
            full_scope_totals="Graph/global service and process totals are populated only when every D/S/H worker in that scope completed. Matched-block subset ratios retain their explicit reduced denominators.",
            process_rss="Process-lifetime high-water RSS from each trace; separately, maximum externally sampled RSS from each scheduler row. Medians/ranges are across workers, never sums of RSS.",
            memory="Maxima of each separately recorded request-memory field within a worker, then distributions across workers. Fields may overlap and are not added; factor resident bytes remain unknown when not recorded.",
            work="Separate integer sums with request/access-epoch coverage. Repeated methods and repeats count as executed work. Complete workload totals require complete workers and complete field coverage.",
            rhs="Total scalar RHS = state_rhs + candidate_rhs + winner_rhs + policy_refresh_rhs + numerical_reanchor_rhs. Response blocks and response-only residual matvec columns remain separate.",
            ties="Counter sums across worker requests plus evidence-derived counts. Deduplication uses exact input hashes and epoch; it is not a count of unseen or independent replications.",
            independent_proof="This postprocessor does no numerical replay or witness verification. Consult the separately recorded independent checker evidence."),
        provenance=dict(protocol=str(args.protocol.resolve()), manifest=str(args.manifest.resolve()),
                        analysis=str(args.analysis.resolve()), raw_directory=str(args.raw.resolve()),
                        original_raw_directory=str(args.original_raw.resolve()),
                        postprocessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        input_files_sha256=dict(sorted(reader.hashes.items()))))
    return result, reader


def fmt(value, digits=4):
    return "unavailable" if value is None else f"{value:,.{digits}f}"


def distribution_text(item, scale=1., digits=4):
    return ("unavailable" if item["median"] is None else
            f"{fmt(item['median']/scale, digits)} [{fmt(item['minimum']/scale, digits)}, "
            f"{fmt(item['maximum']/scale, digits)}]") + f"; samples {item['samples']}/{item['planned_samples']}"


def render_text(result):
    lines = ["A2 EXACT-TIE REVISION: DESCRIPTIVE RESULTS", "",
        "Revision retest on diagnosed fixed inputs; not an unseen confirmation.",
        "The original A2 confirmation remains failed. No new acceptance gate is applied.", "",
        f"Complete workers: {result['complete_workers']}/108 across all 12 original input cells.",
        "By method: " + ", ".join(f"{m} {result['complete_workers_by_method'][m]}/36" for m in METHODS),
        "Scheduler statuses: " + json.dumps(result["scheduler_status_counts"], sort_keys=True),
        "Recorded request statuses: " + json.dumps(result["recorded_request_status_counts"], sort_keys=True), "",
        "COST BY INPUT CELL", "Seconds: full setup plus Q4; median [minimum, maximum] across three repeats.",
        "A missing repeat leaves the full three-repeat statistic unavailable; available costs remain in JSON."]
    for cell in result["cells"]:
        lines.extend(["", f"{cell['graph']} / {cell['stream']} / {cell['seed_label']}"])
        for method in METHODS:
            cost = cell["methods"][method]
            lines.append(f"  {method}: {fmt(cost['median_seconds'])} [{fmt(cost['minimum_seconds'])}, "
                         f"{fmt(cost['maximum_seconds'])}]; complete {cost['complete_repeats']}/3; "
                         "repeats=" + json.dumps(cost["seconds_by_repeat"]))
        for name, comparison in cell["comparisons"].items():
            lines.append(f"  {name}: {fmt(comparison['ratio_H_over_reference'])}; inverse speedup "
                         f"{fmt(comparison['inverse_speedup_reference_over_H'])}x; "
                         f"matched whole cells {comparison['matched_whole_input_cells']}/1")
    lines.extend(["", "GRAPH AND GLOBAL COST SCOPE",
                  "Full totals are available only when every D/S/H input worker in the scope completed.",
                  "Paired ratios use only whole three-repeat matched cells; reduced coverage is stated explicitly."])
    for group in [*result["graphs"], result["overall"]]:
        lines.extend(["", f"{group['name']}: complete {group['complete_workers']}/{group['planned_workers']}",
                      "  Full service totals (seconds): " + json.dumps(group["full_scope_service_totals_seconds"]),
                      "  Full process totals (seconds): " + json.dumps(group["full_scope_process_totals_seconds"])])
        for name, comparison in group["comparisons"].items():
            lines.append(f"  {name}: {fmt(comparison['ratio_H_over_reference'])}; inverse "
                         f"{fmt(comparison['inverse_speedup_reference_over_H'])}x; whole matched cells "
                         f"{comparison['matched_whole_input_cells']}/{comparison['planned_input_cells']}; "
                         f"repeat pairs {comparison['matched_repeat_pairs']}/{comparison['planned_repeat_pairs']}")
    lines.extend(["", "MEMORY BY GRAPH AND METHOD", "MiB: median [minimum, maximum] across workers.",
                  "Trace process-lifetime RSS and external sampled RSS are separate measurements.",
                  "Other memory fields are recorded payload snapshots/maxima, not an additive process-memory breakdown."])
    show_memory = ("explicit_catalog_bytes", "incidence_bytes", "index_private_bytes",
                   "peak_frontier_owned_bytes", "peak_emitted_batch_array_bytes",
                   "actual_candidate_cache_python_bytes", "decision_and_access_log_python_bytes",
                   "peak_low_rank_steps")
    for group in result["graphs"]:
        for method, values in group["methods"].items():
            lines.append(f"  {group['name']} {method}: trace RSS " +
                         distribution_text(values["process_peak_rss_bytes"], 1024**2, 2))
            lines.append("    external sampled RSS " + distribution_text(values["external_sampled_peak_rss_bytes"], 1024**2, 2))
            lines.append("    structure declarations " + json.dumps(values["memory_declarations"], sort_keys=True))
            for field in show_memory:
                if field in values["memory_fields"]:
                    lines.append("    " + field + " " + distribution_text(values["memory_fields"][field], 1024**2, 3))
    lines.extend(["", "WORK BY GRAPH AND METHOD",
                  "Counts are separate executed-work metrics. Full totals require complete field coverage and workers.",
                  "No insertion/heap/flip/endpoint/Gram/response/RHS counts are combined into a synthetic workload."])
    show_work = ("access.insertion_emissions", "access.raw_heap_pops", "access.legal_insertion_pops",
                 "access.flips_emitted", "access.flip_cursor_pops", "access.frontier_rebuilds",
                 "numeric.endpoint_checked", "numeric.gram_materializations", "numeric.response_blocks",
                 "numeric.total_scalar_rhs", "numeric.state_rhs", "numeric.candidate_rhs",
                 "numeric.winner_rhs", "numeric.policy_refresh_rhs", "numeric.numerical_reanchor_rhs")
    for group in result["graphs"]:
        for method, values in group["methods"].items():
            lines.extend(["", f"{group['name']} {method}: complete workers {values['complete_workers']}/{values['planned_workers']}; "
                          f"recorded requests {values['recorded_requests']}/{values['planned_requests']}; "
                          f"terminal requests {values['terminal_requests']}; decisions {values['recorded_decisions']}; "
                          f"accepted {values['accepted_decisions']}; access epochs {values['recorded_access_epochs']}"])
            for metric in show_work:
                values_metric = values["work"][metric]
                lines.append(f"  {metric}: full={fmt(values_metric['complete_workload_total'], 0)}; "
                             f"observed={values_metric['observed_total']:,}; records "
                             f"{values_metric['records_with_value']}/{values_metric['recorded_denominator']} "
                             f"{values_metric['record_kind']}")
    ties = result["structural_ties"]
    lines.extend(["", "STRUCTURAL TIE COUNTS",
                  "All 108 workers: " + json.dumps(ties["counter_totals_across_workers"], sort_keys=True),
                  "Deduplicated exact-input/epoch states: " + json.dumps(ties["deduplicated"], sort_keys=True),
                  f"Counter/evidence discrepancies: {len(ties['counter_evidence_discrepancies'])}",
                  "Method/repeat exposures are repeated executions, not independent cases."])
    for state in ties["states"]:
        if state["unique_switch_transitions"]:
            lines.append(f"  Canonical switch: {state['graph']}/{state['stream']}/{state['seed_label']} "
                         f"q{state['query_position']} epoch{state['epoch']}: " +
                         json.dumps(state["unique_switch_transitions"], sort_keys=True))
    prefixes = result["original_prefix_comparison"]
    lines.extend(["", "ORIGINAL ACCEPTED PREFIXES", prefixes["scope"],
                  "Request comparisons: " + json.dumps(prefixes["status_counts"], sort_keys=True),
                  f"Nonempty original prefixes: {prefixes['nonempty_original_prefixes']}"])
    for row in prefixes["requests"]:
        if row["status"] != "matched":
            lines.append("  " + json.dumps(row, sort_keys=True))
    lines.extend(["", "FAILURES AND INTEGRITY OBSERVATIONS"])
    if not result["failures"]:
        lines.append("No worker lacks a validated complete setup+Q4 cost in the existing analysis and recorded raw data.")
    for row in result["failures"]:
        lines.append("  " + json.dumps(row, sort_keys=True))
    for label, items in (("provenance", result["provenance_and_roster_issues"]),
                         ("raw read", result["raw_read_issues"]),
                         ("existing analysis", result["existing_analysis_validation_issues"]),
                         ("original read", prefixes["read_issues"]),
                         ("tie reconciliation", ties["counter_evidence_discrepancies"])):
        for item in items:
            lines.append(f"  {label}: " + (item if isinstance(item, str) else json.dumps(item, sort_keys=True)))
    lines.extend(["", "INTERPRETATION LIMITS"])
    lines.extend(f"{key}: {value}" for key, value in result["measurement_definitions"].items())
    lines.extend(["", "The original failed confirmation and this revision retest remain separate historical results.",
                  "All individual worker values, memory snapshots, tie observations, prefix comparisons and input hashes are in the JSON summary.", ""])
    return "\n".join(lines)


def write_exclusive_pair(summary_path, text_path, result, text):
    paths = (summary_path.resolve(), text_path.resolve())
    if paths[0] == paths[1] or any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite an existing summary or text report")
    payloads = (json.dumps(result, indent=2, allow_nan=False) + "\n", text)
    handles, created = [], []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            handles.append(os.fdopen(descriptor, "w", encoding="utf-8"))
            created.append(path)
        for handle, payload in zip(handles, payloads):
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        for handle in handles:
            handle.close()
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        for handle in handles:
            if not handle.closed:
                handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT/"protocol/A2_TIE_RETEST.json")
    parser.add_argument("--manifest", type=Path, default=ROOT/"protocol/A2_TIE_RETEST_INPUTS.json")
    parser.add_argument("--raw", type=Path, default=ROOT/"raw/a2_tie_retest")
    parser.add_argument("--analysis", type=Path, default=ROOT/"reports/TIE_RETEST_ANALYSIS.json")
    parser.add_argument("--original-raw", type=Path, default=ROOT/"raw/a2_confirmation")
    parser.add_argument("--summary", type=Path, default=ROOT/"reports/A2_TIE_RETEST_SUMMARY.json")
    parser.add_argument("--text", type=Path, default=ROOT/"reports/A2_TIE_RETEST_RESULTS.txt")
    args = parser.parse_args()
    if args.summary.exists() or args.text.exists():
        parser.error("refusing to overwrite existing output")
    result, reader = build_summary(args)
    text = render_text(result)
    reader.unchanged()
    write_exclusive_pair(args.summary, args.text, result, text)
    print(f"Wrote descriptive revision summary for {result['complete_workers']}/108 complete workers; "
          "the original confirmation remains failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
