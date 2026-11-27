"""Run each archived fixed H failure request once with diagnostic recording.

The score/resolve wrappers call the real revised methods exactly once, preserve
their return values and evidence, and record both pre- and post-resolution
intervals. There is no score override, retry, new opinion, numerical refinement,
or independent-checker invocation here. Existing request timing fields are kept
only as part of the full trace; instrumentation makes them non-performance data.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS"), "1")
COMPLETE = {"budget_stop", "tolerance_stop"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def json_default(value):
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_inventory(path):
    inventory = json.loads(Path(path).read_text())
    states = inventory["states"]
    assert len(states) == 10 and len({s["state_id"] for s in states}) == 10
    assert sum(s["stage"] == "confirmation" for s in states) == 7
    assert sum(s["stage"] == "development" for s in states) == 3
    for state in states:
        assert state["method"] == "H" and state["repeat_id"] == 0
        assert state["budget"] == 15 and state["delta"] == 0
        assert state["expected_failing_epoch"] == state["accepted_prefix_count"] + 1
    return inventory


def verify_inputs(original, state):
    pairs = (("source_file", "source_sha256"),
             ("fixture_file", "fixture_sha256"),
             ("graph_file", "graph_sha256"),
             ("input_manifest_file", "input_manifest_sha256"),
             ("frozen_jobs_file", "frozen_jobs_sha256"))
    for path_key, hash_key in pairs:
        assert sha(original / state[path_key]) == state[hash_key], path_key
    archived = json.loads((original / state["source_file"]).read_text())
    request = archived["requests"][state["opinion_row_zero_based"]]
    assert request["query_position"] == state["query_position"]
    assert request["opinion_hash"] == state["opinion_hash"]
    assert request["status"] == "unresolved"
    assert request["current_version_id"] == state["parent_version_id"]
    accepted = [row for row in request["decisions"] if row["status"] == "accepted"]
    assert len(accepted) == state["accepted_prefix_count"]
    for actual, expected in zip(accepted, state["accepted_prefix"]):
        assert all(actual[key] == value for key, value in expected.items())
    return archived, request


def runtime_hashes(inventory):
    # These are the original runtime sources, plus this recorder and the only
    # new production module. Checker/analysis files are not executed here.
    original = Path(inventory["original_root"])
    source_files = set()
    for item in inventory["manifests"]:
        manifest = json.loads((original / item["input_manifest_file"]).read_text())
        source_files.update(manifest["source_files_sha256"])
    source_files.update(("src/exact_ties.py", "src/a2_tie_fixed_replay.py"))
    return {name: sha(ROOT / name) for name in sorted(source_files)}


def case_filename(index, state):
    return (f"{index:02d}_{state['stage']}_{state['graph']}_"
            f"{state['seed_label']}_q{state['query_position']}.json")


def run_worker(args):
    os.sched_setaffinity(0, {3})
    assert set(os.sched_getaffinity(0)) == {3}
    assert all(os.environ.get(key) == value for key, value in THREAD_ENV.items())
    inventory = read_inventory(args.inventory)
    state = inventory["states"][args.worker_index]
    original = Path(inventory["original_root"])
    frozen = json.loads(args.summary.read_text())["runtime_sources_sha256"]
    assert runtime_hashes(inventory) == frozen, "runtime changed before fixed replay"
    archived, expected = verify_inputs(original, state)
    output = args.raw_dir / case_filename(args.worker_index, state)
    assert not output.exists(), "a fixed request may only be invoked once"
    raw = dict(schema_version=1, state_id=state["state_id"], state=state,
        purpose="one fixed functional replay with passive score/resolve recording",
        performance_evidence=False, independent_checker_run=False,
        started_at=stamp(), status="running", source_file=state["source_file"],
        original_root=str(original), revision_root=str(ROOT),
        inventory_sha256=sha(args.inventory), runtime_sources_sha256=frozen,
        resources=dict(cpu=3, affinity=sorted(os.sched_getaffinity(0)),
                       numerical_threads=1, thread_environment=THREAD_ENV,
                       timeout_seconds=args.timeout_seconds, rss_limit_mib=2048),
        comparisons=[], score_events=[], prefix_checks=[], request=None)
    save(output, raw)
    score_records = {}
    current_parent = state["root_version_id"]
    last_views = None
    patched = False
    try:
        sys.path.insert(0, str(ROOT))
        import numpy as np
        from scipy.sparse import load_npz
        from threadpoolctl import threadpool_info, threadpool_limits
        from src.a2_session import A2Session
        from src.weighted_adapter import Kernel, WeightedViews
        from src.common import DEN, SCALE, UNIT, digest
        from src.session import version_id

        with np.load(original / state["fixture_file"]) as fixture:
            opinions = fixture["opinions"][state["opinion_row_zero_based"]].copy()
        assert opinions.dtype == np.int64 and digest(opinions) == state["opinion_hash"]
        original_score, original_resolve = WeightedViews.score, Kernel.resolve

        def record_score(self, e, t, A, fac, x, r):
            nonlocal last_views
            last_views = self
            epoch, candidate = int(t) + 1, int(e)
            cached = candidate in self.entries
            event = dict(epoch=epoch, candidate_id=candidate,
                         gram_cached_before_score=cached,
                         cached_gram_epoch_before_score=(
                             int(self.entries[candidate][0]) if cached else None),
                         state_residual_radius=int(r))
            raw["score_events"].append(event)
            try:
                answer = original_score(self, e, t, A, fac, x, r)
            except BaseException as error:
                event["exception"] = f"{type(error).__name__}: {error}"
                event["parent_version_id"] = current_parent
                event["operator_sha256"] = digest(A.indptr, A.indices, A.data)
                event["opinion_sha256"] = state["opinion_hash"]
                save(output, raw)
                raise
            event.update(low=int(answer[0]), high=int(answer[1]))
            score_records[(epoch, candidate)] = event
            return answer

        def record_resolve(self, A, s, evaluated, winner, best, **kwargs):
            nonlocal current_parent
            epoch = len(raw["comparisons"]) + 1
            context = kwargs.get("tie_context")
            parent = context.parent_version_id if context is not None else current_parent
            assert parent == current_parent, "diagnostic parent tracking mismatch"
            assert np.array_equal(np.asarray(s), opinions / DEN), "current opinion mismatch"
            overlap = sorted(int(e) for e, hi in evaluated.items()
                             if e != winner and hi >= best)
            selected = sorted(set(overlap + ([] if winner is None else [int(winner)])))

            def candidate_row(e):
                score = score_records[(epoch, e)]
                assert score["high"] == int(evaluated[e]), "raw score upper changed before resolve"
                return dict(candidate_id=e,
                    endpoints=[int(self.pool[0][e]), int(self.pool[1][e])],
                    flip=bool(self.pool[2][e]), d=int(self.d[e]),
                    low=score["low"], high=score["high"],
                    gram_cached_before_score=score["gram_cached_before_score"],
                    cached_gram_epoch_before_score=score["cached_gram_epoch_before_score"],
                    state_residual_radius=score["state_residual_radius"])

            record = dict(epoch=epoch, parent_version_id=parent,
                operator_sha256=digest(A.indptr, A.indices, A.data),
                opinion_sha256=digest(opinions), opinion_denominator=DEN,
                proof_unit=UNIT, published_gain_multiplier=SCALE,
                tie_context_present=context is not None,
                chosen_ids=sorted(map(int, context.chosen)) if context is not None else None,
                provisional_winner_id=None if winner is None else int(winner),
                incumbent_lower=int(best), strict_positive_incumbent=bool(best > 0),
                overlap_ids=overlap,
                evaluated_before=[candidate_row(int(e)) for e in sorted(evaluated)],
                contenders=[candidate_row(e) for e in selected],
                retained_response_candidate_before_resolve=(None if last_views is None or
                    last_views.retained is None else int(last_views.retained[0])),
                original_failure_boundary=epoch == state["expected_failing_epoch"])
            raw["comparisons"].append(record)
            if epoch <= state["accepted_prefix_count"]:
                assert parent == state["accepted_prefix"][epoch-1]["parent_version_id"]
            if record["original_failure_boundary"]:
                assert len(raw["prefix_checks"]) == state["accepted_prefix_count"]
                assert parent == state["parent_version_id"] and overlap
                if state["actual_overlap_evidence_available"]:
                    assert record["operator_sha256"] == state["actual_overlap_evidence"]["operator_sha256"]
                record["original_prefix_and_parent_matched"] = True
            save(output, raw)
            try:
                answer = original_resolve(self, A, s, evaluated, winner, best, **kwargs)
            except BaseException as error:
                record["exception"] = f"{type(error).__name__}: {error}"
                raise
            else:
                chosen_winner, chosen_lower = answer
                record.update(resolved_winner_id=None if chosen_winner is None else int(chosen_winner),
                              resolved_lower=int(chosen_lower),
                              canonical_switch=chosen_winner != winner)
                if epoch <= state["accepted_prefix_count"]:
                    decision = state["accepted_prefix"][epoch-1]
                    assert chosen_winner == decision["winner_id"], "archived prefix winner mismatch"
                    assert int(chosen_lower)*SCALE == decision["low"], "archived prefix lower mismatch"
                    assert version_id(parent, int(chosen_winner)) == decision["current_version_id"]
                    raw["prefix_checks"].append(dict(epoch=epoch,
                        winner_id=int(chosen_winner), lower=int(chosen_lower)*SCALE,
                        parent_version_id=parent, current_version_id=decision["current_version_id"],
                        winner_lower_and_version_matched=True))
                if chosen_winner is not None:
                    current_parent = version_id(parent, int(chosen_winner))
                return answer
            finally:
                evidence = kwargs.get("evidence")
                record["resolution_evidence"] = copy.deepcopy(evidence) if evidence is not None else None
                record["evaluated_after_upper"] = {str(int(e)): int(hi)
                                                   for e, hi in sorted(evaluated.items())}
                if evidence and "unresolved_ids" in evidence:
                    record["unsupported_overlap_ids"] = list(evidence["unresolved_ids"])
                else:
                    record["unsupported_overlap_ids"] = overlap if "exception" in record else []
                save(output, raw)

        WeightedViews.score, Kernel.resolve = record_score, record_resolve
        patched = True
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            assert all(pool["num_threads"] == 1 for pool in pools)
            raw["resources"]["threadpools"] = pools
            session = A2Session(load_npz(original / state["graph_file"]), "H",
                hot_size=state["hot_size"], timers=False,
                graph_hash=state["graph_sha256"], graph_name=state["graph"])
            assert session.root_id == state["root_version_id"]
            assert session.operator_hash == state["root_operator_hash"]
            assert session.catalog_hash == state["catalog_hash"]
            assert session.m == state["catalog_size"]
            request = session.request(opinions, budget=state["budget"], delta=state["delta"], audit=True)
        raw["request"] = request
        accepted = [row for row in request["decisions"] if row["status"] == "accepted"]
        assert len(accepted) >= state["accepted_prefix_count"]
        for got, old in zip(accepted, state["accepted_prefix"]):
            for key in ("winner_id", "low", "parent_version_id", "current_version_id"):
                assert got[key] == old[key], f"final prefix mismatch: {key}"
        boundary = next((c for c in raw["comparisons"] if c["original_failure_boundary"]), None)
        assert boundary is not None and boundary["original_prefix_and_parent_matched"]
        raw.update(status="recorded", prefix_matched=True,
            original_boundary_comparison=copy.deepcopy(boundary),
            revised_request_completed=request["status"] in COMPLETE,
            terminal_status=request["status"], terminal_failure_reason=request["failure_reason"],
            canonical_switches=[c["epoch"] for c in raw["comparisons"] if c.get("canonical_switch")],
            unsupported_comparisons=[dict(epoch=c["epoch"], parent_version_id=c["parent_version_id"],
                unsupported_overlap_ids=c["unsupported_overlap_ids"],
                exception=c.get("exception")) for c in raw["comparisons"] if c["unsupported_overlap_ids"]])
        assert runtime_hashes(inventory) == frozen, "runtime changed during fixed replay"
        verify_inputs(original, state)
        raw["runtime_and_input_hashes_unchanged"] = True
    except BaseException as error:
        raw.update(status="recorder_error", recorder_exception=f"{type(error).__name__}: {error}",
                   recorder_traceback=traceback.format_exc())
    finally:
        if patched:
            WeightedViews.score, Kernel.resolve = original_score, original_resolve
        raw["finished_at"] = stamp()
        save(output, raw)
    print(json.dumps(dict(state_id=state["state_id"], output=str(output), status=raw["status"],
                          terminal_status=raw.get("terminal_status")), sort_keys=True), flush=True)
    return 0 if raw["status"] == "recorded" else 1


def run_all(args):
    import psutil
    inventory = read_inventory(args.inventory)
    original = Path(inventory["original_root"])
    assert not args.summary.exists(), "refusing to replace or rerun the fixed batch"
    assert not args.raw_dir.exists(), "refusing to reuse a fixed replay directory"
    for state in inventory["states"]:
        verify_inputs(original, state)
    args.raw_dir.mkdir(parents=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, purpose="ten original fixed H failure requests, one revised replay each",
        performance_evidence=False, independent_checker_run=False, started_at=stamp(), status="running",
        inventory_file=str(args.inventory), inventory_sha256=sha(args.inventory),
        original_root=str(original), revision_root=str(ROOT),
        runtime_sources_sha256=runtime_hashes(inventory),
        resources=dict(cpu=3, numerical_threads=1, timeout_seconds_per_request=args.timeout_seconds,
                       rss_limit_mib=2048, retries=0), cases=[])
    save(args.summary, report)
    for index, state in enumerate(inventory["states"]):
        command = [sys.executable, "-B", str(Path(__file__).resolve()),
                   "--inventory", str(args.inventory), "--summary", str(args.summary),
                   "--raw-dir", str(args.raw_dir), "--timeout-seconds", str(args.timeout_seconds),
                   "--worker-index", str(index)]
        env = os.environ.copy()
        env.update(THREAD_ENV, PYTHONDONTWRITEBYTECODE="1")
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, start_new_session=True)
        process = psutil.Process(child.pid)
        start = time.monotonic()
        resource_failure = None
        while child.poll() is None:
            if time.monotonic() - start >= args.timeout_seconds:
                resource_failure = "timeout"
            try:
                if process.memory_info().rss > 2048 * 1024 * 1024:
                    resource_failure = "rss_limit"
            except psutil.NoSuchProcess:
                pass
            if resource_failure:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(.025)
        stdout, stderr = child.communicate()
        output = args.raw_dir / case_filename(index, state)
        raw = json.loads(output.read_text()) if output.exists() else {}
        case = dict(state_id=state["state_id"], stage=state["stage"],
            graph=state["graph"], seed_label=state["seed_label"], query_position=state["query_position"],
            raw_file=str(output.relative_to(ROOT)), raw_sha256=sha(output) if output.exists() else None,
            command=command, exit_code=child.returncode, resource_failure=resource_failure,
            worker_stdout=stdout, worker_stderr=stderr, recorder_status=raw.get("status", "missing"),
            prefix_matched=raw.get("prefix_matched", False),
            expected_failing_epoch=state["expected_failing_epoch"],
            accepted_original_prefix_count=state["accepted_prefix_count"],
            original_boundary_comparison=raw.get("original_boundary_comparison"),
            terminal_status=raw.get("terminal_status"),
            terminal_failure_reason=raw.get("terminal_failure_reason"),
            revised_request_completed=raw.get("revised_request_completed", False),
            commits=(raw.get("request") or {}).get("commits"),
            canonical_switches=raw.get("canonical_switches", []),
            unsupported_comparisons=raw.get("unsupported_comparisons", []),
            runtime_and_input_hashes_unchanged=raw.get("runtime_and_input_hashes_unchanged", False))
        report["cases"].append(case)
        save(args.summary, report)
        print(json.dumps({k: case[k] for k in ("state_id", "exit_code", "resource_failure",
                         "prefix_matched", "terminal_status", "commits")}, sort_keys=True), flush=True)
    report.update(finished_at=stamp(), status="recorded",
        fixed_requests_invoked=len(report["cases"]),
        original_prefixes_matched=sum(c["prefix_matched"] for c in report["cases"]),
        completed_requests=sum(c["revised_request_completed"] for c in report["cases"]),
        remaining_unresolved_requests=sum(c["terminal_status"] == "unresolved" for c in report["cases"]),
        resource_failures=sum(c["resource_failure"] is not None for c in report["cases"]),
        recorder_failures=sum(c["exit_code"] != 0 or c["recorder_status"] != "recorded"
                              for c in report["cases"]))
    report["fixed_state_gate_passed"] = all(c["prefix_matched"] and c["revised_request_completed"]
        and c["runtime_and_input_hashes_unchanged"] and c["exit_code"] == 0 for c in report["cases"])
    save(args.summary, report)
    print(json.dumps({k: report[k] for k in ("fixed_requests_invoked", "original_prefixes_matched",
        "completed_requests", "remaining_unresolved_requests", "resource_failures", "recorder_failures",
        "fixed_state_gate_passed")}, sort_keys=True), flush=True)
    return int(report["recorder_failures"] != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=ROOT/"reports/TIE_FIXED_STATE_INVENTORY.json")
    parser.add_argument("--summary", type=Path, default=ROOT/"reports/TIE_FIXED_STATE_REPLAY.json")
    parser.add_argument("--raw-dir", type=Path, default=ROOT/"raw/a2_tie_fixed_replay")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    args.inventory, args.summary, args.raw_dir = (p.resolve() for p in
                                               (args.inventory, args.summary, args.raw_dir))
    if not 0 < args.timeout_seconds <= 180:
        parser.error("each request must have a wall limit of at most 180 seconds")
    if args.worker_index is not None and not 0 <= args.worker_index < 10:
        parser.error("worker index must name one of the ten inventory states")
    return run_all(args) if args.worker_index is None else run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
