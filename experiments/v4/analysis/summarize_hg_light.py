"""Summarize the frozen V4 batch; never execute a service or manufacture pairs."""
import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics


COMPLETE = {'completed_budget', 'certified_stop'}
CELLS = [(g, s) for g in ('Alpha', 'OTC')
         for s in ('slow', 'independent', 'signed-neighbor')]
COMPLETE_CELLS = (0, 1, 3, 4, 5)
EXPECTED_PHASES = (
    'graph_load', 'catalog_build', 'operator_assembly', 'factor_or_precond_init',
    'state_solve_and_cert', 'caps_build', 'caps_sort', 'posting/index_build',
    'reset_restore', 'opinion_refresh', 'discovery', 'response_acquisition',
    'transport_local', 'comparison/refinement', 'cache_management', 'other',
    'frontier_prepare', 'seed_access', 'candidate_access', 'endpoint_screen')
ACCESS_COUNTERS = (
    'frontier_rebuilds', 'raw_heap_pops', 'graph_rejects', 'committed_rejects',
    'processed_skips', 'insertion_emissions', 'flips_emitted',
    'stream_pairs_examined', 'stream_positive_rejects', 'stream_processed_skips',
    'stream_insertions_emitted', 'stream_flips_emitted', 'explicit_raw_visits',
    'emitted_for_screen', 'node_sort_items', 'flip_sort_items', 'flip_cursor_pops',
    'total_source_pops', 'legal_insertion_pops', 'processed_insertion_skips',
    'processed_flip_skips', 'seed_candidates', 'seed_screen_rows',
    'seed_insertions', 'seed_flips', 'cold_emitted_for_screen', 'warm_screen_rows',
    'insertion_caps_built', 'flip_caps_built', 'heap_build_items', 'frontier_resets',
    'g_pair_blocks_tested', 'g_pair_blocks_expanded', 'g_pair_leaves_generated',
    'g_pair_graph_rejects', 'g_pair_chosen_rejects', 'g_pair_processed_rejects',
    'g_block_insertions_emitted', 'g_deferred_warm_insertions')


def valid_number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def select_implementation(R, medians, repeat_aggregates, gate):
    """Apply the preregistered engineering rule, including its stability guard."""
    medians, repeat_aggregates = list(medians), list(repeat_aggregates)
    adequate = (gate is True and len(medians) == 5 and len(repeat_aggregates) == 3
                and valid_number(R, positive=True)
                and all(valid_number(v, positive=True) for v in medians + repeat_aggregates))
    result = dict(decision='blocked', recommended_method=None, current_default='H',
                  gate_passed=adequate, engineering_threshold=0.10,
                  significance_claim=False, additional_experiment_authorized=False)
    if not adequate:
        result['reason'] = 'Required source/semantic/resource gate or complete five-cell/three-repeat data is missing.'
        return result
    g_base = R >= 1.10 and sum(v > 1 for v in medians) >= 4 and min(medians) >= 0.90
    h_base = R <= 10 / 11 and sum(v < 1 for v in medians) >= 4 and max(medians) <= 10 / 9
    g_stable, h_stable = min(repeat_aggregates) >= 1.10, max(repeat_aggregates) <= 10 / 11
    result['conditions'] = dict(G_base=g_base, H_base=h_base,
                                G_repeat_stability=g_stable, H_repeat_stability=h_stable)
    if g_base and g_stable:
        result.update(decision='recommend_G', recommended_method='G',
                      reason='G meets the aggregate, cell-median and preregistered repeat-stability rules.')
    elif h_base and h_stable:
        result.update(decision='recommend_H_engineering_advantage', recommended_method='H',
                      reason='H meets the reciprocal aggregate, cell-median and repeat-stability rules.')
    else:
        result.update(decision='retain_H_insufficient_or_mixed', recommended_method='H',
                      reason='The frozen rule does not establish sufficient stable evidence to replace the H default.')
    return result


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def close_seconds(left, right):
    return (valid_number(left) and valid_number(right)
            and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-8))


def phase_issues(phases, wall, label):
    """Require every frozen exclusive bucket; missing is not a measured zero."""
    if not isinstance(phases, dict):
        return [label + ': missing phase dictionary']
    issues = []
    if set(phases) != set(EXPECTED_PHASES):
        issues.append(label + ': phase field set differs from the frozen H/G ledger')
    if not all(valid_number(v) for v in phases.values()):
        issues.append(label + ': phase value is not finite and nonnegative')
    elif not close_seconds(sum(phases.values()), wall):
        issues.append(label + ': exclusive phase sum differs from its wall time')
    return issues


def write_csv(path, rows, fields):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (dict, list)) else v
                             for k, v in row.items() if k in fields})


def summarize(workspace):
    from validate_worker import binding_issues
    from verify_formal_repeat import light_obligation_issues

    w = Path(workspace).resolve()
    root = w / 'v4'
    freeze_path = root / 'protocols/HG_LIGHT_FROZEN.json'
    if not freeze_path.is_file():
        raise FileNotFoundError('No formal freeze exists; no future-data summary is generated.')
    frozen, freeze_hash = load(freeze_path), sha(freeze_path)
    issues, bindings, workers, request_rows = [], {}, [], []
    jobs = frozen.get('jobs', [])
    expected_order = [(r, c, m) for r in range(3) for c in range(6)
                      for m in (('H', 'G') if (c + r) % 2 == 0 else ('G', 'H'))]
    observed_order = [(j.get('repeat_id'), j.get('cell_index'), j.get('method')) for j in jobs]
    if observed_order != expected_order or len(jobs) != 36:
        issues.append('Formal job order/count differs from the fixed 36-worker schedule.')
    if frozen.get('run_kind') != 'formal' or frozen.get('measurement_policy') != 'light_v1':
        issues.append('Wrong formal run kind or policy.')
    selection_rule = frozen.get('selection_rule', {})
    if selection_rule.get('G') != dict(aggregate_min=1.10, positive_cell_medians_min=4,
                                      cell_median_min=0.90, repeat_aggregate_min=1.10):
        issues.append('Frozen G selection thresholds differ from the implemented preregistered rule.')
    if selection_rule.get('H') != dict(aggregate_max=10 / 11, negative_cell_medians_min=4,
                                      cell_median_max=10 / 9, repeat_aggregate_max=10 / 11):
        issues.append('Frozen H selection thresholds differ from the implemented preregistered rule.')
    for rel, expected in frozen.get('analysis_files_sha256', {}).items():
        path = w / rel
        if not path.is_file() or sha(path) != expected:
            issues.append('Changed/missing frozen analysis source: ' + rel)
    own_rel = 'v4/analysis/summarize_hg_light.py'
    if frozen.get('analysis_files_sha256', {}).get(own_rel) != sha(Path(__file__)):
        issues.append('Summary implementation was not bound by the formal freeze.')
    invariant_path = root / 'reports/semantic_invariance.json'
    invariant = load(invariant_path) if invariant_path.is_file() else None
    if (not invariant or invariant.get('status') != 'passed'
            or invariant.get('runtime_commit') != frozen.get('runtime_commit')
            or sha(invariant_path) != frozen.get('gate_report_sha256')):
        issues.append('Same-build invariance gate is absent, blocked or not bound to this freeze.')
    registered_files = {j.get('output_file') for j in jobs}
    extras = sorted(str(p.relative_to(w)) for p in (w / frozen['raw_dir']).glob('*.json')
                    if p.name not in registered_files)
    if extras:
        issues.append('Unregistered raw files exist; they are not silently treated as additional repeats.')

    for job in jobs:
        file = w / frozen['raw_dir'] / job['output_file']
        record_path = root / 'runs' / ('hg_light_' + file.stem) / 'record.json'
        trace = record = None
        local_issues = []
        for label, path in (('raw', file), ('supervisor', record_path)):
            if not path.is_file():
                local_issues.append('missing ' + label)
                continue
            try:
                value = load(path)
                if not isinstance(value, dict):
                    raise TypeError('expected JSON object')
                if label == 'raw':
                    trace = value
                else:
                    record = value
                bindings[str(path.relative_to(w))] = sha(path)
            except (ValueError, TypeError, OSError) as exc:
                local_issues.append('invalid ' + label + ': ' + str(exc))
        if trace is not None and record is not None:
            try:
                local_issues.extend(binding_issues(trace, frozen, job, record, freeze_path))
                local_issues.extend(light_obligation_issues(trace))
            except (KeyError, TypeError, ValueError, StopIteration) as exc:
                local_issues.append('malformed worker/record schema: ' + repr(exc))
        requests = trace.get('requests', []) if trace else []
        if not isinstance(requests, list) or not all(isinstance(q, dict) for q in requests):
            local_issues.append('invalid request list')
            requests = []
        complete = len(requests) == 4 and all(q.get('result_status') in COMPLETE for q in requests)
        cost = trace.get('service_accounted_seconds') if trace else None
        structural = trace.get('structural_seconds') if trace else None
        request_times = [q.get('request_seconds') for q in requests]
        if trace is not None:
            local_issues.extend(phase_issues(trace.get('structural_phases'), structural, 'structural'))
            for q in requests:
                local_issues.extend(phase_issues(q.get('phases'), q.get('request_seconds'),
                                                'request ' + str(q.get('query_position'))))
            if not (valid_number(structural) and all(valid_number(t) for t in request_times)
                    and close_seconds(cost, structural + sum(request_times))):
                local_issues.append('service time does not match structural plus actual request times')
            if complete and not valid_number(cost, positive=True):
                local_issues.append('invalid complete service time')
        raw_status = trace.get('status') if trace else None
        effective_status = raw_status or 'not_recorded'
        if record and record.get('status') == 'timeout':
            effective_status = 'timeout'
        elif record and record.get('status') != 'completed':
            effective_status = raw_status if raw_status in ('memory_limit_exceeded', 'runtime/certificate_error') else 'runtime/certificate_error'
        elif not record:
            effective_status = 'not_started' if not trace else 'missing_supervisor_record'
        accepted = sum(sum(d.get('status') == 'accepted' for d in q.get('decisions', [])) for q in requests)
        row = dict(position=job.get('position'), cell_index=job.get('cell_index'), graph=job.get('graph'),
                   stream=job.get('stream'), method=job.get('method'), repeat_id=job.get('repeat_id'),
                   file=str(file.relative_to(w)), record_file=str(record_path.relative_to(w)),
                   raw_sha256=bindings.get(str(file.relative_to(w))), record_sha256=bindings.get(str(record_path.relative_to(w))),
                   recorded=trace is not None, status=effective_status, raw_status=raw_status,
                   supervisor_status=record.get('status') if record else None,
                   exit_code=record.get('exit_code') if record else None,
                   complete_Q4_observed=complete, eligible_complete_Q4=complete and not local_issues,
                   requests_recorded=len(requests), completed_requests=sum(q.get('result_status') in COMPLETE for q in requests),
                   accepted_records=accepted, structural_seconds=structural, service_accounted_seconds=cost,
                   complete_service_seconds=cost if complete else None,
                   partial_work_seconds=cost if not complete else None,
                   worker_checkpoint_wall_seconds=trace.get('process_wall_seconds') if trace else None,
                   supervisor_process_wall_seconds=record.get('wall_seconds') if record else None,
                   worker_checkpoint_ru_maxrss_bytes=trace.get('process_peak_rss_bytes') if trace else None,
                   supervisor_sampled_max_rss_bytes=record.get('sampled_peak_rss_bytes') if record else None,
                   supervisor_sampled_max_vms_bytes=record.get('sampled_peak_vms_bytes') if record else None,
                   worker_solve_calls=trace.get('metering', {}).get('solve_calls') if trace else None,
                   worker_scalar_rhs=trace.get('metering', {}).get('scalar_rhs') if trace else None,
                   issues=local_issues)
        workers.append(dict(row=row, trace=trace, record=record, requests=requests))
        issues.extend(job['output_file'] + ': ' + problem for problem in local_issues)
        for q in requests:
            request_rows.append(dict(position=row['position'], cell_index=row['cell_index'], graph=row['graph'],
                stream=row['stream'], method=row['method'], repeat_id=row['repeat_id'], query_position=q.get('query_position'),
                status=q.get('status'), result_status=q.get('result_status'), failure_reason=q.get('failure_reason'),
                accepted_records=sum(d.get('status') == 'accepted' for d in q.get('decisions', [])),
                decision_records=len(q.get('decisions', [])), request_seconds=q.get('request_seconds'),
                initial_gram_count=q.get('initial_gram_count'),
                final_gram_count=q.get('memory', {}).get('gram_count'),
                peak_gram_count=q.get('memory', {}).get('peak_gram_count'),
                low_rank_step_count=q.get('memory', {}).get('low_rank_step_count'),
                root_version_id=q.get('root_version_id'), current_version_id=q.get('current_version_id'),
                opinion_hash=q.get('opinion_hash'), counters=q.get('counters', {})))

    gates = []
    by_file = {entry['row']['file']: entry['row'] for entry in workers}
    for repeat in range(3):
        gate_path = root / f'reports/formal_repeat_{repeat}.json'
        errors = []
        report = None
        if not gate_path.is_file():
            errors.append('missing repeat gate')
        else:
            report = load(gate_path)
            bindings[str(gate_path.relative_to(w))] = sha(gate_path)
            expected = dict(status='passed', repeat_id=repeat, runtime_commit=frozen['runtime_commit'], freeze_sha256=freeze_hash)
            for key, value in expected.items():
                if report.get(key) != value:
                    errors.append('gate binding/status ' + key)
            if report.get('issues') != []:
                errors.append('gate has issues or lacks explicit empty issue list')
            entries = report.get('workers', [])
            if not isinstance(entries, list) or len(entries) != 12:
                errors.append('gate does not cover twelve workers')
                entries = entries if isinstance(entries, list) else []
            covered = []
            for item in entries:
                observed = by_file.get(item.get('file'))
                if (not observed or observed['repeat_id'] != repeat
                        or item.get('sha256') != observed['raw_sha256']
                        or item.get('record_sha256') != observed['record_sha256']
                        or item.get('issues') != [] or item.get('comparison', {}).get('equal') is not True):
                    errors.append('gate worker hash/comparison mismatch: ' + str(item.get('file')))
                covered.append(item.get('file'))
            expected_files = {entry['row']['file'] for entry in workers if entry['row']['repeat_id'] == repeat}
            if set(covered) != expected_files or len(covered) != len(set(covered)):
                errors.append('gate coverage mismatch or duplicate')
        gates.append(dict(repeat_id=repeat, file=str(gate_path.relative_to(w)),
                          observed_status=report.get('status') if report else 'missing',
                          valid=not errors, issues=errors))
        issues.extend(f'repeat {repeat}: ' + error for error in errors)
        if errors:
            # A returned full request is still recorded as full work, but a
            # missing/failed semantic gate cannot admit it to a time ratio.
            for item in workers:
                if item['row']['repeat_id'] == repeat:
                    item['row']['eligible_complete_Q4'] = False

    lookup = {(entry['row']['cell_index'], entry['row']['repeat_id'], entry['row']['method']): entry for entry in workers}
    pairs = []
    for repeat in range(3):
        for cell in range(6):
            h, g = lookup.get((cell, repeat, 'H')), lookup.get((cell, repeat, 'G'))
            hr, gr = h['row'] if h else {}, g['row'] if g else {}
            valid_pair = bool(hr.get('eligible_complete_Q4') and gr.get('eligible_complete_Q4'))
            eligible = valid_pair and cell in COMPLETE_CELLS
            if valid_pair and cell not in COMPLETE_CELLS:
                issues.append('Unexpected completion of the retained unresolved cell; no new success is assumed.')
            pairs.append(dict(cell_index=cell, graph=CELLS[cell][0], stream=CELLS[cell][1], repeat_id=repeat,
                first_method='H' if (cell + repeat) % 2 == 0 else 'G',
                H_status=hr.get('status', 'not_scheduled'), G_status=gr.get('status', 'not_scheduled'),
                H_complete_Q4=hr.get('complete_Q4_observed', False), G_complete_Q4=gr.get('complete_Q4_observed', False),
                complete_pair=valid_pair, eligible_original_complete_pair=eligible,
                H_service_seconds=hr.get('complete_service_seconds') if eligible else None,
                G_service_seconds=gr.get('complete_service_seconds') if eligible else None,
                H_over_G=hr['complete_service_seconds'] / gr['complete_service_seconds'] if eligible else None,
                H_partial_work_seconds=hr.get('service_accounted_seconds') if not eligible else None,
                G_partial_work_seconds=gr.get('service_accounted_seconds') if not eligible else None,
                H_requests_recorded=hr.get('requests_recorded', 0), G_requests_recorded=gr.get('requests_recorded', 0),
                H_completed_requests=hr.get('completed_requests', 0), G_completed_requests=gr.get('completed_requests', 0),
                H_accepted_records=hr.get('accepted_records', 0), G_accepted_records=gr.get('accepted_records', 0)))
    complete_pairs = [p for p in pairs if p['eligible_original_complete_pair']]
    partial_pairs = [p for p in pairs if not p['eligible_original_complete_pair']]
    cells, repeat_rows = [], []
    for cell in range(6):
        members = [p for p in pairs if p['cell_index'] == cell]
        ratios = [p['H_over_G'] for p in members if p['H_over_G'] is not None]
        ht = [p['H_service_seconds'] for p in members if p['H_service_seconds'] is not None]
        gt = [p['G_service_seconds'] for p in members if p['G_service_seconds'] is not None]
        cells.append(dict(cell_index=cell, graph=CELLS[cell][0], stream=CELLS[cell][1],
            complete_pair_count=len(ratios), complete_for_selection=cell in COMPLETE_CELLS and len(ratios) == 3,
            ratios_by_repeat=[p['H_over_G'] for p in members],
            H_service_seconds_by_repeat=[p['H_service_seconds'] for p in members],
            G_service_seconds_by_repeat=[p['G_service_seconds'] for p in members],
            H_service_min=min(ht) if ht else None, H_service_max=max(ht) if ht else None,
            G_service_min=min(gt) if gt else None, G_service_max=max(gt) if gt else None,
            ratio_median=statistics.median(ratios) if ratios else None,
            ratio_min=min(ratios) if ratios else None, ratio_max=max(ratios) if ratios else None,
            ratio_range_is_confidence_interval=False))
    for repeat in range(3):
        members = [p for p in complete_pairs if p['repeat_id'] == repeat]
        hs = sum(p['H_service_seconds'] for p in members) if members else None
        gs = sum(p['G_service_seconds'] for p in members) if members else None
        repeat_rows.append(dict(repeat_id=repeat, complete_pairs=len(members), H_total_seconds=hs, G_total_seconds=gs,
                                H_over_G_ratio_of_sums=hs / gs if hs is not None and gs else None))
    hs = sum(p['H_service_seconds'] for p in complete_pairs) if complete_pairs else None
    gs = sum(p['G_service_seconds'] for p in complete_pairs) if complete_pairs else None
    R = hs / gs if hs is not None and gs else None
    count_gate = (len(complete_pairs) == 15 and len(workers) == 36
                  and all(cells[c]['complete_for_selection'] for c in COMPLETE_CELLS)
                  and all(row['complete_pairs'] == 5 for row in repeat_rows))
    if not count_gate:
        issues.append('Five original complete cells do not all have three valid full Q4 pairs.')
    gate = not issues and count_gate and all(g['valid'] for g in gates)
    decision = select_implementation(R, [cells[c]['ratio_median'] for c in COMPLETE_CELLS],
                                      [r['H_over_G_ratio_of_sums'] for r in repeat_rows], gate)

    complete_keys = {(p['cell_index'], p['repeat_id']) for p in complete_pairs}
    aggregates, counter_rows, phase_rows, access_rows, metering_rows = {}, [], [], [], []
    for subset in ('all_recorded_work', 'complete_pairs_only'):
        aggregates[subset] = {}
        for method in ('H', 'G'):
            selected = [item for item in workers if item['trace'] is not None and item['row']['method'] == method
                        and (subset == 'all_recorded_work' or (item['row']['cell_index'], item['row']['repeat_id']) in complete_keys)]
            requests = [q for item in selected for q in item['requests']]
            names = sorted({key for q in requests for key in q.get('counters', {})})
            totals = {}
            for name in names:
                values = [q.get('counters', {}).get(name) for q in requests]
                observed = [v for v in values if type(v) is int and v >= 0]
                total = sum(observed) if len(observed) == len(values) else None
                totals[name] = total
                counter_rows.append(dict(subset=subset, method=method, counter=name, total=total,
                    observed_total=sum(observed), observations=len(observed), missing_or_invalid=len(values)-len(observed),
                    requests_recorded=len(requests), workers_recorded=len(selected)))
            phase_total = {}
            phase_objects = dict(structural_exclusive=[item['trace'].get('structural_phases') for item in selected],
                                 request_exclusive=[q.get('phases') for q in requests])
            for scope, objects in phase_objects.items():
                if not objects:
                    continue
                names = set(EXPECTED_PHASES) | {k for obj in objects if isinstance(obj, dict) for k in obj}
                for name in sorted(names):
                    values = [obj.get(name) if isinstance(obj, dict) else None for obj in objects]
                    observed = [value for value in values if valid_number(value)]
                    total = sum(observed) if len(observed) == len(values) else None
                    phase_total[(scope, name)] = total
                    phase_rows.append(dict(subset=subset, method=method, phase_scope=scope, phase=name,
                        seconds=total, observed_seconds_only=sum(observed), observations=len(observed),
                        missing_or_invalid=len(values)-len(observed), registered_phase=name in EXPECTED_PHASES,
                        workers_recorded=len(selected), requests_recorded=len(requests)))
            events = [e for q in requests for e in q.get('access', {}).get('epochs', [])]
            for name in ACCESS_COUNTERS:
                values = [e[name] for e in events if name in e]
                if values:
                    valid = all(type(v) is int and v >= 0 for v in values)
                    access_rows.append(dict(subset=subset, method=method, counter=name,
                        total=sum(values) if valid else None, events_with_field=len(values), all_events=len(events),
                        cross_method_unit_equivalence_claim=False))
            measured = {}
            for name in ('solve_calls', 'scalar_rhs', 'solve_seconds', 'factor_attempt_seconds'):
                values = [item['trace'].get('metering', {}).get(name) for item in selected]
                measured[name] = sum(values) if values and all(valid_number(v) for v in values) else None
                metering_rows.append(dict(subset=subset, method=method, metric=name, value=measured[name],
                    workers_recorded=len(selected), timing_overlaps_service=name.endswith('_seconds'),
                    scope='worker metering through saved checkpoint; scalar_rhs includes all solve roles'))
            aggregates[subset][method] = dict(workers_recorded=len(selected), requests_recorded=len(requests),
                accepted_records=sum(item['row']['accepted_records'] for item in selected),
                request_counters=totals, worker_metering=measured,
                phase_totals={scope: {name: value for (s, name), value in phase_total.items() if s == scope}
                              for scope in ('structural_exclusive', 'request_exclusive')})

    oldpath = w / 'v3/resumed/reports/hg_verified/hg_summary.json'
    old = load(oldpath) if oldpath.is_file() else None
    summary = dict(schema='jcert-v4-hg-light-summary-1',
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='passed' if gate else 'blocked', performance_selection_gate_passed=gate,
        runtime_commit=frozen.get('runtime_commit'), freeze_sha256=freeze_hash,
        analysis_sha256=sha(Path(__file__)), measurement_policy='light_v1',
        issues=issues, selection=decision, repeat_gates=gates,
        unregistered_raw_files=extras,
        counts=dict(workers_scheduled=len(jobs), worker_records=sum(x['row']['recorded'] for x in workers),
            complete_Q4_workers=sum(x['row']['complete_Q4_observed'] for x in workers),
            eligible_complete_Q4_workers=sum(x['row']['eligible_complete_Q4'] for x in workers),
            requests_recorded=len(request_rows), completed_requests=sum(r['result_status'] in COMPLETE for r in request_rows),
            accepted_record_observations=sum(r['accepted_records'] for r in request_rows),
            original_complete_pairs=len(complete_pairs), partial_or_ineligible_pairs=len(partial_pairs),
            independent_input_streams=6, new_input_streams=0,
            worker_status_counts=dict(Counter(x['row']['status'] for x in workers)),
            request_status_counts=dict(Counter(r['result_status'] for r in request_rows))),
        ratio_summary=dict(H_total_seconds=hs, G_total_seconds=gs, H_over_G_ratio_of_sums=R,
            complete_pairs=len(complete_pairs), selection_eligible=gate,
            scope='Descriptive sum ratio on the recorded eligible original-complete pairs; the full frozen gate is required for implementation selection.'),
        cells=cells, repeat_aggregates=repeat_rows, complete_pairs=complete_pairs,
        partial_or_ineligible_pairs=partial_pairs, workers=[item['row'] for item in workers],
        work_aggregates=aggregates,
        historical_V3=dict(file=str(oldpath.relative_to(w)), sha256=sha(oldpath) if old else None,
            legacy_policy_ratio_summary=old.get('ratio_summary') if old else None,
            role='Separate historical implementation observation; no policy-removal causal ratio or D/G/Stream/G inference.'),
        evidence_bindings=bindings,
        interpretation=dict(complete_cost='graph load and structural preparation + all actual full request times',
            process_wall='separate; never added to service',
            solver_timers='overlap exclusive service phases; never added to those phases',
            partial_cost='saved work/progress only; no complete Q4 ratio',
            cold_tail_ids='cold batch rows sent to the common endpoint classifier, not total generated or unique candidates',
            candidate_rhs='request role counter, distinct from total worker scalar_rhs',
            access_counters='H row pops and G block/leaf/cursor counters retain separate meanings',
            memory='worker ru_maxrss through final pre-serialization checkpoint; supervisor 100ms RSS/VMS sampled maxima; neither is exact object memory',
            statistics='three prespecified repetitions of six previously observed streams; no confidence interval, significance or new-input generalization',
            numerical_verification='exact logged same-method semantic checks; no new independent full-catalog numerical replay',
            stop='one frozen batch; no adaptive repeats, inputs, precision, resolver or backend changes'))
    out = root / 'reports'
    files = {'hg_light_summary.json': None,
        'hg_light_workers.csv': ([item['row'] for item in workers], list(workers[0]['row']) if workers else ['position']),
        'hg_light_pairs.csv': (pairs, list(pairs[0])),
        'hg_light_cells.csv': (cells, list(cells[0])),
        'hg_light_requests.csv': (request_rows, list(request_rows[0]) if request_rows else ['position']),
        'hg_light_counters.csv': (counter_rows, ['subset','method','counter','total','observed_total','observations','missing_or_invalid','requests_recorded','workers_recorded']),
        'hg_light_phases.csv': (phase_rows, ['subset','method','phase_scope','phase','seconds','observed_seconds_only','observations','missing_or_invalid','registered_phase','workers_recorded','requests_recorded']),
        'hg_light_access_counts.csv': (access_rows, ['subset','method','counter','total','events_with_field','all_events','cross_method_unit_equivalence_claim']),
        'hg_light_worker_metering.csv': (metering_rows, ['subset','method','metric','value','workers_recorded','timing_overlaps_service','scope'])}
    if any((out / name).exists() for name in files):
        raise FileExistsError('Refusing to overwrite an existing final summary or CSV; retain analysis attempts separately.')
    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        if content is not None:write_csv(out / name, *content)
    with (out / 'hg_light_summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    args = p.parse_args()
    report = summarize(args.workspace)
    print(json.dumps(dict(status=report['status'], selection=report['selection'],
                          ratios=report['ratio_summary'], counts=report['counts'], issues=report['issues'])))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
