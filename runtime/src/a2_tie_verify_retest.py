"""Post-batch independent verification of the actual A2 timed revision records.

This module is outside the frozen execution runtime. It audits exact parent
chains and every declared structural witness across the 108-job/432-request
roster, then runs the complete catalog checker only for the protocol-selected
OTC H repeat-0 request 4 (raw position 031). It does not rerun the service,
compare methods, change the original failed judgment, or certify every recorded
gain interval. Run only after measurement finishes; both reports are exclusive
creates, including reports that preserve verification gaps.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import io
from itertools import product
import json
import os
from pathlib import Path
import resource
import tempfile
from time import perf_counter, process_time


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("D", "S", "H")
LABELS = {"D": "D-all", "S": "Stream-all", "H": "Implicit-all"}
SELECTION = dict(graph="OTC", stream="signed-neighbor", seed_label="confirmation-1",
                 method="H", repeat_id=0, query_position=4)
SELECTED_POSITION = 31
CHECKER_SOURCES = (
    "src/a2_tie_verify_retest.py", "src/a2_checker.py", "src/checker.py",
    "baseline_r15/experimental_records/legacy_v6/tools/main_witness_verify.py",
)
ORIGINAL_PATHS = {
    "protocol": "protocol/A2_CONFIRMATION.json",
    "inputs": "protocol/A2_CONFIRMATION_INPUTS.json",
    "admission": "protocol/A2_CONFIRMATION_ADMISSION.json",
    "judgment": "reports/A2_CONFIRMATION_JUDGMENT.json",
    "frozen_jobs": "raw/a2_confirmation/FROZEN_JOBS.json",
}
TERMINAL_DECISIONS = {"budget_stop", "tolerance_stop", "unresolved", "error", "memory_error"}
RSS_SCOPE = ("Linux ru_maxrss * 1024: process-lifetime peak, including earlier verification "
             "phases, Python, sparse operators, proof records and numerical workspaces; "
             "not a per-request incremental memory measurement or an algorithm timing result.")
WITNESS_SCOPE = (
    "Exact input/root/opinion/version reconstruction, current candidate legality, and every "
    "declared structural node-transposition relation/canonical metadata across the fixed roster. "
    "No complete-catalog optimality or independent positive-gain proof is claimed for all 432 "
    "requests. Recorded gain totals and transfer arithmetic are bindings only. Full objective "
    "certification is confined to the separately selected timed request; its numerical published "
    "gain check remains independent interval intersection, not full published-interval enclosure."
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "duplicate JSON object key: " + key)
        result[key] = value
    return result


def bad_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def inside_root(path):
    path = Path(path).resolve()
    need(path.is_relative_to(ROOT), "input path leaves the revision repository")
    return path


def error_text(error):
    return f"{type(error).__name__}: {error}"


def peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


class Reader:
    """Hash the exact bytes read; reject a changed file on subsequent reads."""
    def __init__(self):
        self.files = {}

    def read_bytes(self, path, kind):
        path = inside_root(path)
        data = path.read_bytes()
        key = str(path.relative_to(ROOT))
        observed = hashlib.sha256(data).hexdigest()
        if key in self.files:
            need(self.files[key]["sha256"] == observed, "input changed during verification: " + key)
            self.files[key]["kinds"] = sorted(set(self.files[key]["kinds"]) | {kind})
        else:
            self.files[key] = dict(sha256=observed, bytes=len(data), kinds=[kind])
        return data

    def json(self, path, kind):
        value = json.loads(self.read_bytes(path, kind), object_pairs_hook=unique_object,
                           parse_constant=bad_constant)
        need(isinstance(value, dict), "expected JSON object: " + str(path))
        return value

    def bound_bytes(self, path, expected, kind):
        data = self.read_bytes(path, kind)
        need(hashlib.sha256(data).hexdigest() == expected, "frozen digest mismatch: " + str(path))
        return data

    def unchanged(self):
        issues = []
        for relative, record in self.files.items():
            try:
                observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                need(observed == record["sha256"], "digest changed")
            except Exception as error:
                issues.append(dict(file=relative, error=error_text(error)))
        return issues


def new_json(path, value):
    """Atomic exclusive publication; os.link refuses a concurrent replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=path.name + ".",
                                     suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def registered_jobs(cfg):
    need(cfg.get("id") == "A2-exact-tie-retest-v1" and cfg.get("stage") == "revision_retest" and
         cfg.get("unseen_confirmation") is False and cfg.get("date") == "2026-11-25" and
         cfg.get("graphs") == ["Alpha", "OTC"] and
         cfg.get("streams") == ["slow", "independent", "signed-neighbor"] and
         cfg.get("seed_labels") == ["confirmation-1", "confirmation-2"] and
         cfg.get("methods") == [dict(method=m, label=LABELS[m]) for m in METHODS],
         "unexpected fixed-input revision protocol")
    for field, value in (("repeat_count", 3), ("requests", 4), ("budget", 15), ("delta", 0)):
        need(type(cfg.get(field)) is int and cfg[field] == value, "fixed policy mismatch: " + field)
    selection = cfg.get("independent_tie_replay_selection")
    need(isinstance(selection, dict) and all(selection.get(k) == v for k, v in SELECTION.items()),
         "preselected full-catalog replay changed")
    result = []
    cells = list(product(cfg["graphs"], cfg["streams"], cfg["seed_labels"]))
    for repeat in range(3):
        for position, (graph, stream, label) in enumerate(cells):
            pivot = (repeat + position) % 3
            seed_text = f"A2|{label}|2026-11-25|{graph}|{stream}"
            seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")
            for method in METHODS[pivot:] + METHODS[:pivot]:
                result.append(dict(graph=graph, stream=stream, seed_label=label, seed=seed,
                                   repeat_id=repeat, method=method, label=LABELS[method]))
    need(len(result) == 108 and
         all(result[SELECTED_POSITION].get(k) == v for k, v in SELECTION.items() if k != "query_position"),
         "selected request is not original raw position 031")
    return result


def file_name(position, job):
    return (f"{position:03d}_{job['graph']}_{job['stream']}_"
            f"{job['seed_label']}_{job['label']}.json")


def freeze(reader, protocol_path, manifest_path, raw_directory, schedule):
    cfg = reader.json(protocol_path, "protocol")
    jobs = registered_jobs(cfg)
    manifest = reader.json(manifest_path, "input_manifest")
    protocol_hash = reader.files[str(inside_root(protocol_path).relative_to(ROOT))]["sha256"]
    manifest_hash = reader.files[str(inside_root(manifest_path).relative_to(ROOT))]["sha256"]
    need(manifest.get("schema_version") == 2 and manifest.get("verified") is True and
         manifest.get("protocol_id") == cfg["id"] and manifest.get("stage") == cfg["stage"] and
         manifest.get("protocol_sha256") == protocol_hash and manifest.get("domain") == cfg.get("domain") and
         manifest.get("original_confirmation_status") == "failed" and
         manifest.get("unseen_confirmation") is False and
         manifest.get("inputs_reused_without_regeneration") is True,
         "revision manifest binding mismatch")
    references = cfg.get("original_confirmation")
    need(isinstance(references, dict) and references.get("status") == "failed",
         "original confirmation status must remain failed")
    originals = {}
    for name, relative in ORIGINAL_PATHS.items():
        record = references.get(name)
        need(isinstance(record, dict) and record.get("path") == relative,
             "original archive path mismatch: " + name)
        data = reader.bound_bytes(ROOT / relative, record.get("sha256"), "original_archive")
        originals[name] = json.loads(data, object_pairs_hook=unique_object, parse_constant=bad_constant)
    old = originals["inputs"]
    need(originals["judgment"].get("status") == "failed" and
         originals["frozen_jobs"].get("jobs") == jobs, "original judgment or 108-job roster changed")
    for field in ("graphs", "fixtures", "domain", "independent_opinion_rows", "input_source_files_sha256"):
        need(manifest.get(field) == old.get(field), "original input identity changed: " + field)
    need(manifest.get("independent_opinion_rows") == 48, "expected 48 exact opinion rows")
    historical = manifest.get("original_source_binding")
    binding_fields = {"protocol_id", "stage", "protocol_sha256", "source_files_sha256",
                      "input_source_files_sha256", "environment"}
    need(isinstance(historical, dict) and set(historical) == binding_fields and
         all(historical[k] == old.get(k) for k in binding_fields),
         "historical source binding mismatch")
    current_sources, old_sources = manifest.get("source_files_sha256"), old.get("source_files_sha256")
    need(isinstance(current_sources, dict) and isinstance(old_sources, dict) and
         set(current_sources) == set(old_sources) | {"src/a2_tie_retest.py", "src/exact_ties.py"},
         "unexpected revision runtime source set")
    need({p for p in old_sources if old_sources[p] != current_sources[p]} <=
         {"src/a2_session.py", "src/weighted_adapter.py"}, "revision changes exceed common tie scope")
    for field, kind in (("source_files_sha256", "frozen_runtime_source"),
                        ("input_source_files_sha256", "frozen_input_source")):
        mapping = manifest.get(field)
        need(isinstance(mapping, dict) and bool(mapping), "missing source map: " + field)
        for relative, expected in mapping.items():
            reader.bound_bytes(ROOT / relative, expected, kind)
    frozen = reader.json(raw_directory / "FROZEN_JOBS.json", "timed_roster")
    need(frozen.get("jobs") == jobs, "timed roster differs from original rotated roster")
    for key, value in dict(protocol_sha256=protocol_hash, input_manifest_sha256=manifest_hash,
                           source_files_sha256=current_sources,
                           input_source_files_sha256=manifest["input_source_files_sha256"],
                           environment=manifest["environment"], domain=manifest["domain"],
                           cpu=0, supervisor_cpu=8, supervisor_affinity=[8], limits=cfg["limits"]).items():
        need(frozen.get(key) == value, "timed freeze mismatch: " + key)
    schedule_rows = schedule.get("jobs")
    need(isinstance(schedule_rows, list) and len(schedule_rows) == 108, "expected 108 scheduler records")
    for position, (row, job) in enumerate(zip(schedule_rows, jobs)):
        need(isinstance(row, dict) and row.get("position") == position and row.get("job") == job and
             row.get("result") == file_name(position, job), "scheduler roster binding mismatch")
    records = manifest.get("fixtures")
    need(isinstance(records, list) and len(records) == 12, "expected 12 exact archived fixtures")
    fixtures = {(r["graph"], r["stream"], r["seed_label"]): r for r in records}
    need(set(fixtures) == set(product(cfg["graphs"], cfg["streams"], cfg["seed_labels"])),
         "fixture grid mismatch")
    return cfg, manifest, manifest_hash, jobs, fixtures


def load_inputs(reader, manifest, manifest_path, fixtures):
    import numpy as np
    from scipy import sparse
    from .a2_checker import A2CatalogReplay
    graphs, opinions, failures = {}, {}, []
    for graph, spec in manifest["graphs"].items():
        try:
            blob = reader.bound_bytes(ROOT / spec["weighted_path"], spec["weighted_sha256"], "exact_graph")
            replay = A2CatalogReplay(sparse.load_npz(io.BytesIO(blob)))
            need(replay.n == spec["n"] and replay.m == spec["catalog_size"], "graph dimensions/catalog")
            graphs[graph] = replay
        except Exception as error:
            failures.append(dict(graph=graph, error=error_text(error)))
    for key, record in fixtures.items():
        try:
            blob = reader.bound_bytes(manifest_path.parent / record["file"], record["npz_sha256"],
                                      "exact_opinion_fixture")
            with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
                need(archive.files == ["opinions"], "opinion fixture members")
                matrix = archive["opinions"]
            need(matrix.dtype == np.dtype("<i8") and matrix.shape == tuple(record["shape"]) and
                 matrix.shape == (4, manifest["graphs"][key[0]]["n"]) and matrix.flags.c_contiguous and
                 np.all((matrix >= -65536) & (matrix <= 65536)), "exact opinion matrix layout/range")
            need(hashlib.sha256(matrix.tobytes()).hexdigest() == record["matrix_sha256"] and
                 [hashlib.sha256(s.tobytes()).hexdigest() for s in matrix] == record["opinion_hashes"],
                 "exact opinion matrix/row hashes")
            opinions[key] = matrix
        except Exception as error:
            failures.append(dict(graph=key[0], stream=key[1], seed_label=key[2], error=error_text(error)))
    return graphs, opinions, failures


def trace_issues(trace, position, job, cfg, manifest, manifest_hash, record, replay=None):
    expected = dict(job, schema_version=2, schedule_position=position, protocol_id=cfg["id"],
                    protocol_sha256=manifest["protocol_sha256"], input_manifest_sha256=manifest_hash,
                    source_files_sha256=manifest["source_files_sha256"],
                    input_source_files_sha256=manifest["input_source_files_sha256"],
                    domain=manifest["domain"], environment=manifest["environment"],
                    requests_planned=4, budget=15, delta=0, hot_size=64, audit=False,
                    cpu=0, weight_mode="weighted", graph_hash=manifest["graphs"][job["graph"]]["weighted_sha256"],
                    fixture_id=record["fixture_id"], fixture_npz_sha256=record["npz_sha256"],
                    fixture_matrix_sha256=record["matrix_sha256"])
    if replay is not None:
        from .a2_checker import DOMAIN
        expected.update(n=replay.n, catalog_size=replay.m, operator_hash=replay.operator_hash,
                        root_version_id=replay.root_version_id(expected["graph_hash"]),
                        catalog_hash=hashlib.sha256(f"{DOMAIN}|{expected['graph_hash']}".encode()).hexdigest())
    return ["raw binding mismatch: " + key for key, value in expected.items() if trace.get(key) != value]


def relation_identity(witness):
    candidates = sorted(((witness["source_candidate_id"], witness["source_edit"]),
                         (witness["target_candidate_id"], witness["target_edit"])), key=lambda item: item[0])
    return dict(kind=witness["kind"], parent_version_id=witness["parent_version_id"],
                operator_sha256=witness["operator_sha256"], opinion_sha256=witness["opinion_sha256"],
                opinion_denominator=witness["opinion_denominator"], operator_scale=witness["operator_scale"],
                score_unit=witness["score_unit"], swap=witness["swap"],
                candidates=[dict(candidate_id=e, edit=edit) for e, edit in candidates])


def inspect_resolution(replay, A, s, chosen, current, row, location, aggregates):
    from .a2_checker import _TieProofs
    evidence = row["comparison_resolution"]
    result = dict(comparison_resolution=evidence, metadata_status="invalid", witness_results=[])
    witnesses = evidence.get("witnesses") if isinstance(evidence, dict) else None
    declared = witnesses if isinstance(witnesses, list) else []
    aggregates["counts"]["comparison_resolution_records_processed"] += 1
    aggregates["counts"]["declared_witness_observations_processed"] += len(declared)
    proofs = _TieProofs(replay, A, s, chosen, current) if A is not None else None
    failure = None
    if proofs is None:
        failure = "exact parent unavailable after an invalid/unreadable transition"
    else:
        try:
            proofs.read(row)
            if row.get("status") in ("accept", "accepted"):
                need(type(row.get("low")) is int and type(row.get("high")) is int and
                     row["low"] == 16 * evidence["shared_lower"] and
                     row["high"] == 16 * evidence["shared_upper"], "recorded shared gain arithmetic")
            result["metadata_status"] = "verified"
            aggregates["counts"]["comparison_resolution_metadata_verified"] += 1
        except Exception as error:
            failure = error_text(error)
        finally:
            aggregates["proof_check_work"].update(proofs.work)
    if failure:
        result["error"] = failure
        aggregates["counts"]["comparison_resolution_metadata_invalid"] += 1
    # Successful read checked every witness once. If metadata failed, attempt
    # every declaration separately so a bad early declaration cannot hide later
    # valid/invalid relations. These extra proof operations are counted.
    for index, witness in enumerate(declared):
        declaration_hash = hashlib.sha256(canonical(witness).encode()).hexdigest()
        observation = dict(index=index, declaration_sha256=declaration_hash,
                           relation_status="verified" if failure is None else "unverified")
        if failure is not None and proofs is not None:
            checker = _TieProofs(replay, A, s, chosen, current)
            try:
                checker.validate_witness(witness)
                observation["relation_status"] = "verified"
            except Exception as error:
                observation.update(relation_status="invalid", error=error_text(error))
            finally:
                aggregates["proof_check_work"].update(checker.work)
        elif proofs is None:
            observation["error"] = failure
        status = observation["relation_status"]
        aggregates["counts"]["witness_observations_" + status] += 1
        if status == "verified":
            identity = relation_identity(witness)
            key = hashlib.sha256(canonical(identity).encode()).hexdigest()
            observation["exact_relation_sha256"] = key
            record = aggregates["relations"].setdefault(key, dict(identity=identity, observations=[]))
            record["observations"].append(dict(location, witness_index=index))
        else:
            aggregates["bad_witnesses"].append(dict(location, **observation, witness=witness))
        result["witness_results"].append(observation)
    if isinstance(evidence, dict) and evidence.get("unresolved_ids"):
        aggregates["unsupported_cases"].append(dict(location, reason="recorded unresolved structural overlaps",
                                                     unresolved_ids=evidence["unresolved_ids"],
                                                     comparison_resolution=evidence))
    return result


def inspect_request(replay, s, request, job, query, graph_hash, aggregates):
    from .a2_checker import _TieProofs, _change, _version
    from .checker import digest, opinion_hash
    root = replay.root_version_id(graph_hash)
    current, A, chosen = root, replay.A, set()
    issues, decisions, ended = [], [], None
    result = dict(query_position=query, recorded_status=request.get("status"),
                  expected_root_version_id=root, exact_opinion_sha256=opinion_hash(s),
                  chain_verified=False, issues=issues, decisions=decisions)
    expected = dict(query_position=query, root_version_id=root, method=job["method"],
                    budget=15, delta=0, gain_denominator=2**96, opinion_hash=opinion_hash(s))
    for key, value in expected.items():
        if request.get(key) != value or (type(value) is int and type(request.get(key)) is not int):
            issues.append("request binding mismatch: " + key)
    rows = request.get("decisions")
    if not isinstance(rows, list):
        issues.append("request decisions is not a list")
        return result
    total_low = total_high = 0
    for epoch, row in enumerate(rows, 1):
        location = dict(schedule_position=job["schedule_position"], query_position=query, epoch=epoch,
                        graph=job["graph"], stream=job["stream"], seed_label=job["seed_label"],
                        method=job["method"], repeat_id=job["repeat_id"])
        detail = dict(epoch=epoch, issues=[], exact_parent_reconstructed=A is not None)
        decisions.append(detail)
        aggregates["counts"]["recorded_decisions"] += 1
        if not isinstance(row, dict):
            detail["issues"].append("decision is not an object")
            A = None
            continue
        status, e = row.get("status"), row.get("winner_id")
        detail.update(status=status, winner_id=e, recorded_parent_version_id=row.get("parent_version_id"),
                      reconstructed_parent_version_id=current)
        if A is not None:
            detail["reconstructed_operator_sha256"] = digest(A.indptr, A.indices, A.data)
            aggregates["counts"]["exact_parents_reconstructed"] += 1
        if ended is not None:
            detail["issues"].append("decision after terminal status")
        if type(row.get("epoch")) is not int or row["epoch"] != epoch:
            detail["issues"].append("epoch binding mismatch")
        if row.get("root_version_id") != root or row.get("parent_version_id") != current:
            detail["issues"].append("root/parent version binding mismatch")
        if "comparison_resolution" in row:
            detail["structural_verification"] = inspect_resolution(
                replay, A, s, chosen, current, row, location, aggregates)
        if status in ("accept", "accepted"):
            aggregates["counts"]["recorded_accepts"] += 1
            if len(chosen) >= 15:
                detail["issues"].append("decision beyond edit budget")
            if A is None:
                detail["issues"].append("accepted child cannot be reconstructed without its parent")
            else:
                try:
                    candidate = _TieProofs(replay, A, s, chosen, current).candidate(e)
                    for key, value in zip(("i", "j", "d", "flip"), candidate):
                        need(row.get(key) == value, "accepted edit endpoint/type/weight binding: " + key)
                    need(all(type(row.get(key)) is int for key in ("i", "j", "d")) and
                         (type(row.get("flip")) is bool or
                          (type(row.get("flip")) is int and row["flip"] in (0, 1))),
                         "accepted edit integer/boolean representation")
                    A = _change(A, candidate)
                    chosen.add(e)
                    current = _version(current, e)
                    detail["accepted_edit_legal"] = True
                except Exception as error:
                    detail["issues"].append(error_text(error))
                    A = None
            if (type(row.get("low")) is int and type(row.get("high")) is int and
                    0 < row["low"] <= row["high"]):
                total_low += row["low"]
                total_high += row["high"]
            else:
                detail["issues"].append("recorded accepted gain interval is malformed/nonpositive")
        elif isinstance(status, str) and status in TERMINAL_DECISIONS:
            if e is not None:
                detail["issues"].append("terminal decision has a winner")
            if status == "budget_stop" and len(chosen) != 15:
                detail["issues"].append("premature explicit budget stop")
            ended = status
            if status in ("unresolved", "error", "memory_error"):
                aggregates["unsupported_cases"].append(dict(location, reason="recorded terminal failure",
                                                             status=status, failure_reason=row.get("failure_reason")))
        else:
            detail["issues"].append("unsupported decision status: " + repr(status))
            A = None
        if row.get("current_version_id") != current:
            detail["issues"].append("child version binding mismatch")
    if A is None:
        issues.append("one or more exact parents/children could not be reconstructed")
    if request.get("current_version_id") != current:
        issues.append("request final version binding mismatch")
    if type(request.get("commits")) is not int or request["commits"] != len(chosen):
        issues.append("request commit count mismatch")
    if request.get("gain_low") != total_low or request.get("gain_high") != total_high:
        issues.append("request recorded gain total mismatch")
    terminal = ended or ("budget_stop" if len(chosen) == 15 else None)
    if request.get("status") != terminal:
        issues.append("request terminal status/count mismatch")
    result.update(reconstructed_final_version_id=current, reconstructed_commits=len(chosen),
                  chain_verified=A is not None and not issues and all(not d["issues"] for d in decisions))
    return result


def witness_audit(reader, cfg, manifest, manifest_hash, jobs, fixtures, graphs, opinions, raw_directory, schedule):
    aggregates = dict(counts=Counter(), declarations=Counter(), relations={}, bad_witnesses=[],
                      unsupported_cases=[], proof_check_work=Counter())
    workers, raw_failures = [], []
    for position, job in enumerate(jobs):
        path = raw_directory / file_name(position, job)
        key = (job["graph"], job["stream"], job["seed_label"])
        record = fixtures[key]
        worker = dict(position=position, **job, raw_file=str(path.relative_to(ROOT)),
                      scheduler_status=schedule["jobs"][position].get("status"), issues=[], requests=[])
        workers.append(worker)
        try:
            trace = reader.json(path, "timed_raw")
            worker["raw_sha256"] = reader.files[str(path.relative_to(ROOT))]["sha256"]
            worker["trace_status"] = trace.get("status")
            worker["failure_reason"] = trace.get("failure_reason")
            worker["issues"].extend(trace_issues(trace, position, job, cfg, manifest, manifest_hash, record,
                                                graphs.get(job["graph"])))
            aggregates["counts"]["raw_workers_read"] += 1
        except Exception as error:
            trace = {}
            worker["issues"].append(error_text(error))
            raw_failures.append(dict(position=position, raw_file=worker["raw_file"], error=error_text(error)))
        rows = trace.get("requests")
        indexed = {}
        if not isinstance(rows, list):
            worker["issues"].append("recorded requests is not a list")
            rows = []
        aggregates["counts"]["all_recorded_request_objects"] += sum(isinstance(r, dict) for r in rows)
        for request in rows:
            decisions = request.get("decisions") if isinstance(request, dict) else None
            if not isinstance(decisions, list):
                continue
            aggregates["counts"]["raw_recorded_decisions"] += len(decisions)
            for decision in decisions:
                if not isinstance(decision, dict) or "comparison_resolution" not in decision:
                    continue
                aggregates["counts"]["comparison_resolution_records"] += 1
                evidence = decision["comparison_resolution"]
                witnesses = evidence.get("witnesses") if isinstance(evidence, dict) else None
                if isinstance(witnesses, list):
                    aggregates["counts"]["declared_witness_observations"] += len(witnesses)
                    for witness in witnesses:
                        aggregates["declarations"][hashlib.sha256(canonical(witness).encode()).hexdigest()] += 1
        for ordinal, row in enumerate(rows, 1):
            query = row.get("query_position") if isinstance(row, dict) else None
            if type(query) is not int or not 1 <= query <= 4:
                worker["issues"].append(f"request ordinal {ordinal} has an unsupported query position")
                worker.setdefault("unassigned_requests", []).append(dict(ordinal=ordinal, request=row))
            else:
                indexed.setdefault(query, []).append(row)
        if len(rows) != 4:
            worker["issues"].append(f"recorded request count {len(rows)} differs from planned 4")
        if [r.get("query_position") if isinstance(r, dict) else None for r in rows] != [1, 2, 3, 4]:
            worker["issues"].append("request ordering differs from the frozen Q1 through Q4 sequence")
        for query in range(1, 5):
            values = indexed.get(query, [])
            if len(values) != 1:
                worker["requests"].append(dict(query_position=query, chain_verified=False,
                    status="missing" if not values else "duplicate", matching_records=len(values),
                    issues=["expected exactly one recorded request"], duplicate_requests=values))
                continue
            aggregates["counts"]["planned_request_traces_present"] += 1
            if job["graph"] not in graphs or key not in opinions:
                worker["requests"].append(dict(query_position=query, chain_verified=False,
                    status="input_unavailable", issues=["exact graph/opinion input unavailable"],
                    unverified_request=values[0]))
                continue
            try:
                proof = inspect_request(graphs[job["graph"]], opinions[key][query - 1], values[0],
                    dict(job, schedule_position=position), query,
                    manifest["graphs"][job["graph"]]["weighted_sha256"], aggregates)
                worker["requests"].append(proof)
                aggregates["counts"]["request_parent_chains_verified"] += int(proof["chain_verified"])
            except Exception as error:
                worker["requests"].append(dict(query_position=query, chain_verified=False,
                                              status="verification_error", issues=[error_text(error)]))
        print(canonical(dict(phase="parent_and_witness_audit", position=position,
                             request_parent_chains_verified=sum(r["chain_verified"] for r in worker["requests"]))),
              flush=True)
    verified_chains = aggregates["counts"]["request_parent_chains_verified"] == 432
    parents_reconstructed = (not raw_failures and
                             aggregates["counts"]["exact_parents_reconstructed"] ==
                             aggregates["counts"]["raw_recorded_decisions"])
    # Keep this expression explicit: relation validity alone is not metadata validity.
    valid = (verified_chains and parents_reconstructed and all(not w["issues"] for w in workers) and not raw_failures and
             not aggregates["bad_witnesses"] and
             aggregates["counts"]["comparison_resolution_metadata_invalid"] == 0 and
             aggregates["counts"]["comparison_resolution_metadata_verified"] ==
             aggregates["counts"]["comparison_resolution_records"] and
             aggregates["counts"]["witness_observations_verified"] ==
             aggregates["counts"]["declared_witness_observations"])
    aggregates["counts"]["declared_witness_observations_not_processed"] = (
        aggregates["counts"]["declared_witness_observations"] -
        aggregates["counts"]["declared_witness_observations_processed"])
    counts = dict(aggregates["counts"], unique_witness_declarations=len(aggregates["declarations"]),
                  deduplicated_verified_exact_relations=len(aggregates["relations"]))
    return dict(status=("passed_with_preserved_unsupported_cases" if aggregates["unsupported_cases"] else "passed")
                if valid else "verification_gap", all_planned_parent_chains_verified=verified_chains,
                all_recorded_parents_reconstructed=parents_reconstructed,
                all_declared_comparison_resolutions_verified=valid,
                counts=counts, proof_check_work=dict(aggregates["proof_check_work"]), workers=workers,
                deduplicated_verified_relations=[dict(exact_relation_sha256=k, **v,
                    observation_count=len(v["observations"])) for k, v in sorted(aggregates["relations"].items())],
                declaration_observation_counts=dict(sorted(aggregates["declarations"].items())),
                bad_witnesses=aggregates["bad_witnesses"], unsupported_cases=aggregates["unsupported_cases"],
                raw_failures=raw_failures)


def full_replay(reader, cfg, manifest, manifest_hash, jobs, fixtures, graphs, opinions, raw_directory):
    from .a2_checker import worker_request
    job = jobs[SELECTED_POSITION]
    key = (job["graph"], job["stream"], job["seed_label"])
    path = raw_directory / file_name(SELECTED_POSITION, job)
    report = dict(selection=dict(SELECTION, schedule_position=SELECTED_POSITION),
                  raw_file=str(path.relative_to(ROOT)), status="verification_gap")
    started, cpu_started, rss_before = perf_counter(), process_time(), peak_rss()
    try:
        trace = reader.json(path, "selected_timed_raw")
        report["raw_sha256"] = reader.files[str(path.relative_to(ROOT))]["sha256"]
        issues = trace_issues(trace, SELECTED_POSITION, job, cfg, manifest, manifest_hash, fixtures[key],
                              graphs.get(job["graph"]))
        need(not issues, "; ".join(issues))
        requests = trace.get("requests")
        need(isinstance(requests, list) and len(requests) == 4 and
             all(isinstance(r, dict) for r in requests) and
             [r.get("query_position") for r in requests] == [1, 2, 3, 4],
             "selected actual timed request roster mismatch")
        request = requests[3]
        report.update(recorded_request_status=request.get("status"),
                      request_sha256=hashlib.sha256(canonical(request).encode()).hexdigest(),
                      graph_sha256=manifest["graphs"][job["graph"]]["weighted_sha256"],
                      fixture_npz_sha256=fixtures[key]["npz_sha256"],
                      exact_opinion_sha256=fixtures[key]["opinion_hashes"][3])
        proof = worker_request(graphs[job["graph"]], request, opinions[key][3], report["graph_sha256"])
        report["proof"] = proof
        report["status"] = "passed" if (proof["full_catalog_decisions_certified"] and
                                             proof["termination_certified"]) else "verification_gap"
    except Exception as error:
        report["error"] = error_text(error)
    report.update(wall_seconds=perf_counter() - started, process_cpu_seconds=process_time() - cpu_started,
                  process_peak_rss_before_bytes=rss_before, process_peak_rss_bytes=peak_rss(), rss_scope=RSS_SCOPE)
    return report


def run(protocol_path, manifest_path, raw_directory, witness_output, full_output):
    protocol_path, manifest_path, raw_directory = map(inside_root, (protocol_path, manifest_path, raw_directory))
    witness_output, full_output = Path(witness_output).resolve(), Path(full_output).resolve()
    need(witness_output != full_output, "verification outputs must be distinct")
    for path in (witness_output, full_output):
        if path.exists():
            raise FileExistsError("refusing to replace verification output: " + str(path))
    # The batch holds this lock throughout measurement. Refuse an early launch
    # before importing numerical libraries or reading any actual timed outputs.
    with (ROOT / "logs/.measurement.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("measurement is active; launch verification only after the batch") from error
        reader = Reader()
        schedule = reader.json(raw_directory / "SCHEDULE_RESULTS.json", "timed_schedule")
        need(schedule.get("status") in ("completed", "completed_with_failures"),
             "timed scheduler has not finished; no verification reports created")
        started, cpu_started, rss_before = perf_counter(), process_time(), peak_rss()
        base = dict(schema_version=1, study="independent verification of actual timed A2 revision outputs",
                    algorithm_timing=False, original_confirmation_status="failed",
                    planned_workers=108, planned_requests=432, planned_unique_opinion_rows=48,
                    started_utc=datetime.now(timezone.utc).isoformat(), scope=WITNESS_SCOPE,
                    actual_affinity=sorted(os.sched_getaffinity(0)), numerical_threads=1,
                    process_peak_rss_before_bytes=rss_before, rss_scope=RSS_SCOPE)
        for relative in CHECKER_SOURCES:
            reader.read_bytes(ROOT / relative, "independent_checker_source")
        reader.read_bytes(ROOT / "docs/TIE_CERTIFICATE_INTERFACE.txt", "witness_schema")
        checker_hashes = {p: reader.files[p]["sha256"] for p in CHECKER_SOURCES}
        base.update(checker_source_files_sha256=checker_hashes,
                    checker_sha256=checker_hashes["src/a2_checker.py"],
                    numerical_checker_sha256=checker_hashes["src/checker.py"],
                    verifier_sha256=checker_hashes["src/a2_tie_verify_retest.py"])
        full = dict(base, selection=dict(SELECTION, schedule_position=SELECTED_POSITION),
                    status="blocked_by_input_binding",
                    scope="Full legal-catalog replay of the preselected actual timed request only. "
                          "Positive winner evidence and all rival comparisons are independent. Published numerical "
                          "gain intervals are checked for intersection consistency, not full enclosure.")
        try:
            cfg, manifest, manifest_hash, jobs, fixtures = freeze(
                reader, protocol_path, manifest_path, raw_directory, schedule)
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1):
                graphs, opinions, input_failures = load_inputs(reader, manifest, manifest_path, fixtures)
                audit = witness_audit(reader, cfg, manifest, manifest_hash, jobs, fixtures, graphs, opinions,
                                      raw_directory, schedule)
                audit["input_failures"] = input_failures
                audit["input_changed_during_verification"] = reader.unchanged()
                if input_failures or audit["input_changed_during_verification"]:
                    audit["status"] = "verification_gap"
                    audit["all_declared_comparison_resolutions_verified"] = False
                audit.update(base, wall_seconds=perf_counter() - started,
                             process_cpu_seconds=process_time() - cpu_started,
                             process_peak_rss_bytes=peak_rss(), input_and_raw_files=reader.files.copy())
                new_json(witness_output, audit)
                # An unrelated failed worker cannot silently change the frozen
                # selection. Attempt this one request, preserving any proof gap.
                full.update(full_replay(reader, cfg, manifest, manifest_hash, jobs, fixtures,
                                        graphs, opinions, raw_directory))
        except FileExistsError:
            raise
        except Exception as error:
            if not witness_output.exists():
                audit = dict(base, status="input_binding_or_verification_failure", error=error_text(error),
                             all_planned_parent_chains_verified=False,
                             all_declared_comparison_resolutions_verified=False,
                             input_and_raw_files=reader.files.copy(), wall_seconds=perf_counter() - started,
                             process_cpu_seconds=process_time() - cpu_started, process_peak_rss_bytes=peak_rss())
                new_json(witness_output, audit)
            full.update(status="verification_gap", error=error_text(error))
        full["input_changed_during_verification"] = reader.unchanged()
        if full["input_changed_during_verification"]:
            full["status"] = "verification_gap"
        full.update(input_and_raw_files=reader.files.copy(),
                    witness_audit_report=str(witness_output),
                    witness_audit_sha256=hashlib.sha256(witness_output.read_bytes()).hexdigest(),
                    total_verifier_wall_seconds=perf_counter() - started,
                    total_verifier_process_cpu_seconds=process_time() - cpu_started,
                    process_peak_rss_bytes=peak_rss())
        new_json(full_output, full)
        return audit, full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol/A2_TIE_RETEST.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "protocol/A2_TIE_RETEST_INPUTS.json")
    parser.add_argument("--raw", type=Path, default=ROOT / "raw/a2_tie_retest")
    parser.add_argument("--witness-output", type=Path, default=ROOT / "reports/TIE_TIMED_WITNESS_VERIFICATION.json")
    parser.add_argument("--full-output", type=Path, default=ROOT / "reports/TIE_TIMED_OTC_FULL_REPLAY.json")
    args = parser.parse_args()
    audit, full = run(args.protocol, args.manifest, args.raw, args.witness_output, args.full_output)
    print(canonical(dict(witness_audit_status=audit["status"], selected_full_replay_status=full["status"])), flush=True)
    return 0 if audit["status"] in ("passed", "passed_with_preserved_unsupported_cases") and full["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
