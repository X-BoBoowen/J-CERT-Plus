"""Offline A2 access comparison at fixed, reproduced H decision contexts.

This is a mechanism diagnostic, not an end-to-end service comparison or an
oracle policy. It pays for D's request-root explicit index and retains that
root index across the selected root/last contexts. All arms receive the same
certified current state, H warm/committed exclusions and cold-start incumbent.
Only access and strict endpoint screening run after context reproduction;
there are no new candidate responses, scores, Grams, commits or threshold rises.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

from .a2_access import Access, RATIO, batch_bytes
from .a2_catalog import AllCatalog, CandidateBatch, MixedFrontier
from .a2_numeric import EndpointKernel
from .a2_session import A2Session, make_ledger
from .common import DEN, SCALE, UNIT, digest
from .checker import need, opinion_hash
from long_promotion import SCREEN


ROOT = Path(__file__).resolve().parents[1]
REQUEST_FIELDS = (
    "root_version_id", "current_version_id", "opinion_hash", "method", "budget", "delta",
    "status", "failure_reason", "initial_seed_ids", "initial_seed_order", "seed_rule",
    "initial_gram_count", "decisions", "commits", "gain_low", "gain_high",
    "gain_denominator", "root_energy", "final_energy", "counters", "cache_policy",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _without_times(value):
    if isinstance(value, dict):
        return {k: _without_times(v) for k, v in value.items()
                if not k.endswith("_seconds") and not k.endswith("_phases")}
    if isinstance(value, list):
        return [_without_times(v) for v in value]
    return value


def match_request(measured, audited):
    """Require the actual recorded output and untimed witness to agree."""
    for key in REQUEST_FIELDS:
        need(key in measured and key in audited and measured[key] == audited[key],
             "reproduced output mismatch: " + key)
    need(_without_times(measured["access"]) == _without_times(audited["access"]),
         "reproduced access mismatch")
    need(measured["method"] == "H" and measured["delta"] == 0,
         "diagnostic requires the selected zero-tolerance H output")


def reproduce_request(W, s, measured, graph_hash, *, hot_size=64, graph_name="fixture"):
    start = perf_counter()
    session = A2Session(W, "H", hot_size=hot_size, timers=True,
                        graph_hash=graph_hash, graph_name=graph_name)
    setup = perf_counter() - start
    request_start = perf_counter()
    audited = session.request(s, budget=measured["budget"], delta=0, audit=True)
    request_seconds = perf_counter() - request_start
    match_request(measured, audited)
    counters = audited["counters"]
    metadata = dict(
        outputs_match_recorded=True, setup_seconds=setup, request_seconds=request_seconds,
        total_seconds=perf_counter() - start,
        state_rhs=counters["state_rhs"], candidate_rhs=counters["candidate_rhs"],
        winner_rhs=counters["winner_rhs"],
        response_blocks=counters["response_blocks"],
        process_peak_rss_bytes=_rss(),
        scope="Witness reproduction includes ordinary H numerical work; none is counted "
              "as access-diagnostic work or credited as an end-to-end speedup.")
    del session
    gc.collect()
    return audited, metadata


def _make_context(catalog, kernel, root_A, s, measured, snapshot, epoch):
    start = perf_counter()
    row = measured["decisions"][epoch - 1]
    event = measured["access"]["epochs"][epoch - 1]
    need(row["epoch"] == epoch and event["epoch"] == epoch and
         snapshot["epoch"] == epoch and
         snapshot["parent_version_id"] == row["parent_version_id"],
         "snapshot version binding")
    incumbent = event.get("cold_start_lower_fine")
    need(type(incumbent) is int and incumbent >= 0 and
         snapshot["cold_start_lower_fine"] == incumbent,
         "snapshot cold-start incumbent binding")
    chosen = [int(r["winner_id"]) for r in measured["decisions"][:epoch - 1]
              if r["status"] == "accepted"]
    need(snapshot["chosen"] == sorted(chosen) and len(set(chosen)) == len(chosen),
         "snapshot chosen binding")
    warm = [int(e) for e in snapshot["warm_ids"]]
    need(warm == row["seed_ids"] and len(set(warm)) == len(warm) and
         not set(warm).intersection(chosen), "snapshot processed warm binding")
    catalog.batch(np.asarray(chosen + warm, np.int64))
    A = root_A
    for e in chosen:
        A = kernel.child(A, e)
    x = np.asarray(snapshot["x"], dtype=np.int64)
    xc = np.asarray(snapshot["xc"], dtype=np.int64)
    r, rc = snapshot["r"], snapshot["rc"]
    need(x.shape == xc.shape == (catalog.n,) and type(r) is int and r >= 0 and
         type(rc) is int and rc >= 0, "snapshot state shape/radius")
    diag = A.diagonal()
    need((2 * int(diag.max()) + 1) * int(abs(x).max()) + UNIT < 2 ** 63,
         "snapshot residual integer range")
    residual = abs(A @ x - s * (UNIT // DEN))
    divisor = UNIT // SCREEN
    need(int(residual.max()) == r and
         (int(abs(x - xc * divisor).max()) + r + divisor - 1) // divisor == rc and
         int(abs(xc).max()) + rc <= 2 * SCREEN, "snapshot residual/coarse certificate")
    excluded = set(chosen).union(warm)
    excluded_array = np.asarray(sorted(excluded), dtype=np.int64)
    context_id = hashlib.sha256("/".join((
        digest(A.indptr, A.indices, A.data, x, xc), str(r), str(rc), str(incumbent),
        digest(excluded_array))).encode()).hexdigest()
    metadata = dict(
        epoch=epoch, parent_version_id=row["parent_version_id"],
        current_state_sha256=digest(x, xc, np.asarray([r, rc], np.int64)),
        context_id=context_id, incumbent_lower_fine=incumbent,
        incumbent_physical_denominator=UNIT ** 2,
        incumbent_physical_numerator=SCALE * incumbent,
        coarse_threshold=-(-incumbent // RATIO),
        chosen_ids=sorted(chosen), processed_warm_ids=warm,
        excluded_ids_sha256=digest(excluded_array), excluded_count=len(excluded),
        remaining_legal_candidates=catalog.m - len(chosen),
        cold_unprocessed_candidates=catalog.m - len(excluded),
        materialization_and_certificate_seconds=perf_counter() - start,
        additional_state_rhs=0,
        scope="Reconstructed current operator and checked existing integer residual; "
              "the H cold-start incumbent and processed IDs are fixed for all arms.")
    return dict(A=A, s=s, x=x, xc=xc, r=r, rc=rc, diag=diag, residual=residual,
                rows=(A, s, x, UNIT, diag, residual), chosen=set(chosen), warm=set(warm),
                excluded=excluded, incumbent=incumbent, metadata=metadata)


def _run_arm(method, catalog, kernel, context, root_xc, root_rc, d_access):
    start = perf_counter()
    rss_before = _rss()
    prepare_start = perf_counter()
    strict = kernel.strict(context["A"], context["s"], context["x"], context["rows"])
    incumbent = context["incumbent"]
    threshold = -(-incumbent // RATIO)
    excluded = context["excluded"]
    scan_counts = {}
    frontier = iterator = None
    if method == "D":
        need(d_access is not None and d_access.index is not None, "paid root index unavailable")
        # The source is always request-root, including the late current context.
        need(np.array_equal(d_access.root_xc, root_xc) and d_access.root_rc == root_rc,
             "D root index source moved")
        drift = abs(context["xc"] - d_access.root_xc) + context["rc"] + d_access.root_rc
        local = set(map(int, catalog.dense_positions(np.asarray(sorted(excluded), np.int64))))
    elif method == "S":
        iterator = catalog.iter_blocks(block_size=65536, excluded=excluded, counters=scan_counts)
    else:
        frontier = MixedFrontier(catalog, context["xc"], context["rc"], chosen=context["chosen"])
    preparation = perf_counter() - prepare_start
    enumeration = screening = 0.
    emitted = insertions = flips = survivors = batches = max_batch = 0
    survivor_sum = survivor_xor = 0
    enumeration_done = False
    detail = {}
    while True:
        tick = perf_counter()
        if method == "D":
            if enumeration_done:
                batch = None
            else:
                ids, choices, costs, tau = d_access.index.query(drift, threshold, local)
                i, j, f = d_access.explicit.pool
                ci, cj = i[ids], j[ids]
                canonical = ci.astype(np.int64) * catalog.n + cj
                batch = CandidateBatch(canonical, ci, cj, f[ids], d_access.explicit.d[ids])
                shift = min(8 * (SCREEN // SCALE) * int(tau),
                            8 * d_access.index.root_norm * int(tau) + 4 * int(tau) ** 2)
                posting = int(d_access.degree[drift > tau].sum())
                tails = [len(keys) - int(np.searchsorted(keys, threshold - d * shift))
                         for d, keys, _ in d_access.index.groups]
                raw = int(costs[int(np.flatnonzero(choices == tau)[0])])
                need(raw == posting + sum(tails), "D raw access accounting")
                detail = dict(explicit_raw_visits=raw, posting_raw_visits=posting,
                              root_tail_raw_visits=sum(tails), root_group_tail_rows=tails,
                              tau=int(tau), max_current_to_root_drift=int(drift.max()),
                              root_index_epoch=1, root_center_sha256=digest(root_xc),
                              root_radius=int(root_rc), root_index_retained=True,
                              remaining_upper_coarse=threshold - 1)
                enumeration_done = True
        elif method == "S":
            batch = next(iterator, None)
        else:
            upper = frontier.upper
            if upper is None or (incumbent == 0 and upper <= 0) or upper < threshold:
                batch = None
            else:
                batch = frontier.pop_batch(256, threshold=threshold, excluded=excluded)
                if not len(batch):
                    batch = None
        enumeration += perf_counter() - tick
        if batch is None:
            break
        batches += 1
        emitted += len(batch)
        insertions += int((~batch.flip).sum())
        flips += int(batch.flip.sum())
        max_batch = max(max_batch, batch_bytes(batch))
        tick = perf_counter()
        ids, uppers = strict.screen(batch, incumbent)
        screening += perf_counter() - tick
        survivors += len(ids)
        # Commutative diagnostics require no full survivor-ID list. They are
        # descriptive checksums, not collision-free equality certificates.
        survivor_sum += sum(map(int, ids))
        for e in ids:
            survivor_xor ^= int(e)
        del batch, ids, uppers
    if method == "H":
        detail.update(frontier.counters)
        detail["remaining_upper_coarse"] = frontier.upper
        detail["frontier_build_phases"] = dict(frontier.build_phases)
        memory = frontier.memory()
    elif method == "S":
        detail.update(scan_counts)
        detail["remaining_upper_coarse"] = None
        memory = dict(bounded_scan_rows=65536, no_full_candidate_arrays=True)
        need(emitted == catalog.m - len(excluded), "S complete cold-domain count")
    else:
        memory = d_access.memory()
    elapsed = perf_counter() - start
    return dict(status="completed", method=method, context_id=context["metadata"]["context_id"],
                incumbent_lower_fine=incumbent, excluded_ids_sha256=context["metadata"]["excluded_ids_sha256"],
                current_preparation_seconds=preparation, enumeration_seconds=enumeration,
                endpoint_screen_seconds=screening,
                other_accounted_seconds=elapsed - preparation - enumeration - screening,
                context_total_seconds=elapsed, enumeration_batches=batches,
                legal_emits=emitted, insertion_emissions=insertions, flip_emissions=flips,
                endpoint_rows=strict.rows, strict_survivors=survivors,
                strict_survivor_id_sum=survivor_sum, strict_survivor_id_xor=survivor_xor,
                numerical_response_rhs=0, score_calls=0, threshold_updates=0,
                raw_work=detail, memory=memory,
                max_emitted_batch_payload_bytes=max_batch,
                process_peak_rss_before_bytes=rss_before, process_peak_rss_bytes=_rss(),
                rss_scope="Cumulative process-lifetime peak, including witness reproduction and "
                          "the retained D root index; not an isolated per-arm memory comparison.")


def compare_snapshots(W, s, measured, audited, graph_hash, *, progress=None):
    """Compare only fixed-context access; no response-producing API is invoked."""
    match_request(measured, audited)
    start = perf_counter()
    s = np.asarray(s)
    need(s.dtype == np.int64 and opinion_hash(s) == measured["opinion_hash"], "diagnostic opinion binding")
    report = dict(status="preparing_contexts", contexts=[], arms_order=["D", "S", "H"],
                  selected_epochs=[], additional_numerical_response_rhs=0,
                  scope="Fixed-context access and endpoint screening only. Incumbent/state/warm "
                        "evidence comes from a paid, separately reported H reproduction. No "
                        "end-to-end speedup, alternative decision trajectory or oracle policy is inferred.")
    common_start = perf_counter()
    catalog = AllCatalog(W)
    kernel = EndpointKernel(catalog)
    root_A = kernel.operator(catalog.W)
    root_A.sort_indices()
    catalog_hash = hashlib.sha256(("all-nonedges-positive1/16+root-negative-flips:v1|" + graph_hash).encode()).hexdigest()
    root = hashlib.sha256("/".join((graph_hash, digest(root_A.indptr, root_A.indices, root_A.data),
                                    catalog_hash, kernel.salt)).encode()).hexdigest()
    need(root == measured["root_version_id"], "diagnostic graph/domain root binding")
    report["common_sparse_setup_seconds"] = perf_counter() - common_start
    report["catalog_size"] = catalog.m
    report["common_sparse_catalog_memory"] = catalog.memory()
    decisions = measured["decisions"]
    if not decisions:
        report.update(status="unavailable", reason="Selected request reached no decision.",
                      diagnostic_seconds=perf_counter() - start, process_peak_rss_bytes=_rss())
        return report
    last = decisions[-1]["epoch"]
    need([r["epoch"] for r in decisions] == list(range(1, last + 1)), "recorded decision sequence")
    epochs = sorted({1, last})
    report["selected_epochs"] = epochs
    snapshots = {row["epoch"]: row for row in audited["snapshots"]}
    need(len(snapshots) == len(audited["snapshots"]), "duplicate audit snapshot")
    contexts = []
    for epoch in epochs:
        record = dict(epoch=epoch, roles=(["root", "last_reached"] if epoch == 1 == last
                                         else ["root"] if epoch == 1 else ["last_reached"]), arms=[])
        report["contexts"].append(record)
        if epoch not in snapshots:
            record.update(status="unavailable", reason="The actual reached decision has no audit state "
                          "snapshot (for example, unresolved before snapshot construction); no earlier "
                          "decision was substituted.")
            continue
        event = measured["access"]["epochs"][epoch - 1]
        if type(event.get("cold_start_lower_fine")) is not int:
            record.update(status="unavailable", reason="The actual decision never established a cold-start incumbent.")
            continue
        context = _make_context(catalog, kernel, root_A, s, measured, snapshots[epoch], epoch)
        record.update(status="prepared", **context["metadata"])
        contexts.append((context, record))
    if 1 not in snapshots:
        report.update(status="unavailable", reason="Root coordinates are unavailable; a paid request-root "
                      "D index cannot be reproduced without a replacement context.",
                      diagnostic_seconds=perf_counter() - start, process_peak_rss_bytes=_rss())
        return report
    root_xc = np.asarray(snapshots[1]["xc"], np.int64)
    root_rc = int(snapshots[1]["rc"])
    if progress:
        progress(report)
    d_access = None
    d_start = perf_counter()
    d_ledger = make_ledger(True)
    try:
        d_access = Access(catalog, "D", d_ledger, timers=True)
        # The common H incumbent/warm set is held fixed. No D-specific seed
        # selection or seed responses are needed for this conditional query.
        d_access.begin(root_xc, root_rc, 0, d_ledger)
        d_setup, d_phases = d_ledger.finish()
        report["paid_D_root_setup"] = dict(
            status="completed", seconds=perf_counter() - d_start,
            ledger_seconds=d_setup, phases=d_phases, root_caps_created_rows=d_access.root_caps_created,
            root_center_sha256=digest(root_xc), root_radius=root_rc,
            source_epoch=1, retained_across_selected_contexts=True,
            index_build_phases=dict(d_access.index.build_phases), memory=d_access.memory(),
            numerical_response_rhs=0,
            scope="Explicit full catalog, natural weight groups, incidence, all root caps and sorted "
                  "root index are paid once. Fixed H warm evidence is supplied; no D seed "
                  "responses or alternative incumbent are computed.")
    except Exception as error:
        report["paid_D_root_setup"] = dict(status="error", seconds=perf_counter() - d_start,
                                           failure_reason=f"{type(error).__name__}: {error}")
        d_access = None
        gc.collect()
    if progress:
        progress(report)
    for context, record in contexts:
        for method in ("D", "S", "H"):
            if method == "D" and d_access is None:
                result = dict(status="unavailable", method=method, reason="Paid root-index preparation failed.")
            else:
                try:
                    result = _run_arm(method, catalog, kernel, context, root_xc, root_rc, d_access)
                except Exception as error:
                    result = dict(status="error", method=method,
                                  failure_reason=f"{type(error).__name__}: {error}",
                                  process_peak_rss_bytes=_rss())
            record["arms"].append(result)
            if progress:
                progress(report)
        record["status"] = ("completed" if all(r["status"] == "completed" for r in record["arms"])
                            else "partial")
    if d_access is not None:
        d_access.close()
    report.update(status="completed" if all(r["status"] == "completed" for r in report["contexts"]) else "partial",
                  diagnostic_seconds=perf_counter() - start, process_peak_rss_bytes=_rss(),
                  total_memory_scope="Cumulative process peak includes reproduction, all selected context "
                                     "snapshots and the retained D index. Array/container payloads have "
                                     "their own scopes; no per-arm RSS dominance is claimed.")
    return report


def diagnose_request(W, s, measured, graph_hash, *, hot_size=64, graph_name="fixture", progress=None):
    start = perf_counter()
    audited, reproduction = reproduce_request(W, s, measured, graph_hash,
                                              hot_size=hot_size, graph_name=graph_name)

    def update(data):
        if progress:
            progress(dict(reproduction=reproduction, comparison=data))

    comparison = compare_snapshots(W, s, measured, audited, graph_hash, progress=update)
    return dict(status=comparison["status"], reproduction=reproduction, comparison=comparison,
                complete_offline_seconds=perf_counter() - start, process_peak_rss_bytes=_rss())


def run(protocol_path, manifest_path, directory, *, graph=None, checkpoint=None):
    """Use only the frozen H/slow/Q1/repeat0 selection; retain unavailable cells."""
    from scipy import sparse
    from .a2_bench import jobs, load_fixture, load_freeze, result_filename
    protocol_path, manifest_path, directory = map(Path, (protocol_path, manifest_path, directory))
    cfg, manifest, fixtures, manifest_hash = load_freeze(protocol_path, manifest_path)
    selected = manifest["independent_replay_selection"]
    if graph is not None:
        need(graph in {r["graph"] for r in selected}, "unregistered diagnostic graph")
        selected = [r for r in selected if r["graph"] == graph]
    report = dict(schema_version=1, status="running", results=[],
                  protocol_sha256=_sha(protocol_path), input_manifest_sha256=manifest_hash,
                  diagnostic_source_sha256=_sha(__file__),
                  frozen_runtime_sources=manifest["source_files_sha256"],
                  selection=selected,
                  context_rule="root decision and last actually reached decision of the preselected "
                               "H/slow/Q1/repeat0 request; equal epochs deduplicate with both roles",
                  scope="Offline conditional access diagnostic; not end-to-end performance or an oracle.")
    if checkpoint:
        checkpoint(report)
    roster = list(jobs(cfg))
    for selection in selected:
        record = dict(selection=selection, status="starting")
        report["results"].append(record)
        if checkpoint:
            checkpoint(report)
        try:
            matching = [(position, job) for position, job in enumerate(roster)
                        if all(job[k] == selection[k] for k in
                               ("graph", "stream", "seed_label", "seed", "method", "repeat_id"))]
            need(len(matching) == 1 and selection["query_position"] == 1, "diagnostic selection binding")
            position, job = matching[0]
            path = directory / result_filename(position, job)
            trace = json.loads(path.read_text())
            need(trace["schedule_position"] == position and
                 all(trace[k] == job[k] for k in job) and
                 trace["protocol_sha256"] == manifest["protocol_sha256"] and
                 trace["input_manifest_sha256"] == manifest_hash and
                 trace["source_files_sha256"] == manifest["source_files_sha256"] and
                 trace["input_source_files_sha256"] == manifest["input_source_files_sha256"],
                 "selected trace freeze binding")
            spec = manifest["graphs"][job["graph"]]
            graph_path = ROOT / spec["weighted_path"]
            graph_hash = _sha(graph_path)
            need(graph_hash == spec["weighted_sha256"] == trace["graph_hash"], "selected graph binding")
            fixture = fixtures[job["graph"], job["stream"], job["seed_label"]]
            matrix = load_fixture(fixture, manifest_path.parent)
            need(trace["fixture_id"] == fixture["fixture_id"] and
                 trace["fixture_npz_sha256"] == fixture["npz_sha256"] and
                 trace["fixture_matrix_sha256"] == fixture["matrix_sha256"], "selected fixture binding")
            need(bool(trace["requests"]), "selected Q1 request is unavailable")
            measured = trace["requests"][0]
            need(measured["query_position"] == 1 and measured["opinion_hash"] == fixture["opinion_hashes"][0] and
                 measured["stream_id"] == trace["stream_id"] and measured["budget"] == cfg["budget"] and
                 trace["hot_size"] == cfg["hot_size"], "selected request binding")
            record.update(status="reproducing", trace_file=path.name, trace_sha256=_sha(path),
                          graph_path=str(graph_path), graph_sha256=graph_hash,
                          opinion_hash=measured["opinion_hash"], fixture_npz_sha256=fixture["npz_sha256"],
                          fixture_matrix_sha256=fixture["matrix_sha256"], source_position=position)
            if checkpoint:
                checkpoint(report)

            def update(data):
                record["diagnostic"] = data
                if checkpoint:
                    checkpoint(report)

            result = diagnose_request(sparse.load_npz(graph_path), matrix[0], measured, graph_hash,
                                      hot_size=cfg["hot_size"], graph_name=job["graph"], progress=update)
            record.update(status=result["status"], diagnostic=result)
        except Exception as error:
            record.update(status="error", failure_reason=f"{type(error).__name__}: {error}",
                          process_peak_rss_bytes=_rss())
        if checkpoint:
            checkpoint(report)
    report["status"] = "completed" if all(r["status"] == "completed" for r in report["results"]) else "partial"
    report["process_peak_rss_bytes"] = _rss()
    if checkpoint:
        checkpoint(report)
    return report


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    from .anchor_service_bench import save
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--timed-dir", type=Path, required=True)
    parser.add_argument("--graph", choices=("Alpha", "OTC"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(limits=1):
        run(args.protocol, args.manifest, args.timed_dir, graph=args.graph,
            checkpoint=lambda report: save(args.output, report))
