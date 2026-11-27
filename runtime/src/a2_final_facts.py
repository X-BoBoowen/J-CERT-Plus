"""Extract report-ready A2 facts from official analysis, judgment and raw outputs.

Stdlib only. This is presentation-data extraction, not another validator or gate:
completion eligibility and the claim come from the supplied official reports.
No numerical experiment, proof, acceptance test or plot is run. Execute only
after root declares the timing phase idle.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


METHODS = ('D', 'S', 'H')
ACCESS_COUNTS = (
    'frontier_rebuilds', 'raw_heap_pops', 'graph_rejects', 'committed_rejects',
    'processed_skips', 'insertion_emissions', 'flips_emitted',
    'stream_pairs_examined', 'explicit_raw_visits', 'emitted_for_screen',
    'node_sort_items', 'flip_sort_items', 'legal_insertion_pops',
    'processed_insertion_skips', 'processed_flip_skips', 'flip_cursor_pops',
    'total_source_pops', 'stream_positive_rejects', 'stream_processed_skips',
    'stream_insertions_emitted', 'stream_flips_emitted',
)


def read_json(path):
    contents = path.read_bytes()
    return json.loads(contents), dict(path=str(path.resolve()), sha256=hashlib.sha256(contents).hexdigest(),
                                      bytes=len(contents))


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def stats(values):
    values = [value for value in values if number(value)]
    return dict(values=values, count=len(values), median=statistics.median(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None,
                mean=statistics.mean(values) if values else None)


def sum_maps(rows):
    result = {}
    for row in rows:
        for key, value in row.items():
            if number(value):
                result[key] = result.get(key, 0) + value
    return result


def maximum(values):
    present = [value for value in values if number(value)]
    return max(present) if present else None


def cell_key(row):
    return tuple(row[key] for key in ('graph', 'stream', 'seed_label'))


def worker_key(row):
    return cell_key(row) + (row['repeat_id'], row['method'])


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def extract_work(trace):
    if trace is None:
        return None
    requests, signatures, unresolved, failures = [], [], [], []
    for q, request in enumerate(trace.get('requests', []), 1):
        events = request.get('access', {}).get('epochs', [])
        event_counts = [{key: event[key] for key in ACCESS_COUNTS if key in event} for event in events]
        accepted = [decision for decision in request.get('decisions', []) if decision.get('status') == 'accepted']
        kinds = Counter('flip' if decision.get('flip') == 1 else
                        'insertion' if decision.get('flip') == 0 else 'unknown' for decision in accepted)
        failure_decisions = [decision for decision in request.get('decisions', [])
                             if decision.get('status') in ('unresolved', 'memory_error', 'error')]
        for decision in failure_decisions:
            position = dict(query_position=q, epoch=decision.get('epoch'), status=decision['status'],
                            accepted_prefix_commits=request.get('commits'),
                            failure_reason=decision.get('failure_reason', request.get('failure_reason')))
            failures.append(position)
            if decision['status'] == 'unresolved':
                unresolved.append(position)
        if request.get('status') in ('unresolved', 'memory_error', 'error') and not failure_decisions:
            position = dict(query_position=q, epoch=None, status=request['status'],
                            accepted_prefix_commits=request.get('commits'), failure_reason=request.get('failure_reason'))
            failures.append(position)
            if request['status'] == 'unresolved':
                unresolved.append(position)
        signature = dict(
            query_position=q, status=request.get('status'), commits=request.get('commits'),
            numeric_counters=request.get('counters', {}),
            access_versions=[counts | {key: event.get(key) for key in ('epoch', 'source', 'source_complete')}
                             for counts, event in zip(event_counts, events, strict=True)],
            decisions=[{key: decision.get(key) for key in ('epoch', 'status', 'winner_id', 'flip')}
                       for decision in request.get('decisions', [])])
        signatures.append(signature)
        requests.append(dict(query_position=q, status=request.get('status'), declared_commits=request.get('commits'),
            numeric_counters=request.get('counters', {}), access_counts=sum_maps(event_counts),
            accepted_insertions=kinds['insertion'], accepted_flips=kinds['flip'],
            accepted_unknown_kind=kinds['unknown'], accepted_total=len(accepted),
            requests_reporting_root_index_build_phases=int(bool(request.get('index_build_phases'))),
            work_signature_sha256=fingerprint(signature)))
    numeric = sum_maps(request['numeric_counters'] for request in requests)
    access = sum_maps(request['access_counts'] for request in requests)
    return dict(
        worker_status=trace.get('status'), returned_requests=len(requests),
        counts_available=bool(requests), numeric_counter_totals=numeric if requests else None,
        access_counter_totals=access if requests else None,
        report_counts=(dict(
            legal_insertion_emissions_K=access.get('insertion_emissions'),
            explicit_flip_emissions=access.get('flips_emitted'),
            raw_heap_pops=access.get('raw_heap_pops'), total_source_pops=access.get('total_source_pops'),
            endpoint_rows=numeric.get('endpoint_checked'), Gram_materializations=numeric.get('gram_materializations'),
            response_blocks=numeric.get('response_blocks'),
            scalar_response_RHS=numeric.get('response_residual_matvec_columns'),
            candidate_RHS=numeric.get('candidate_rhs'), state_RHS=numeric.get('state_rhs'),
            frontier_rebuilds=access.get('frontier_rebuilds'), node_sort_items=access.get('node_sort_items'),
            flip_sort_items=access.get('flip_sort_items'),
            requests_reporting_root_index_build_phases=sum(
                request['requests_reporting_root_index_build_phases'] for request in requests)) if requests else None),
        accepted_edits=dict(
            insertions=sum(request['accepted_insertions'] for request in requests),
            flips=sum(request['accepted_flips'] for request in requests),
            unknown_kind=sum(request['accepted_unknown_kind'] for request in requests),
            total=sum(request['accepted_total'] for request in requests)) if requests else None,
        unresolved_positions=unresolved, failure_positions=failures, per_request=requests,
        work_signature_sha256=fingerprint(dict(worker_status=trace.get('status'), requests=signatures)),
        scope='Returned request work, including accepted prefixes before failures; no uncheckpointed work is inferred.')


def compare_work(first, other, repeat):
    if first is None or other is None:
        return dict(repeat_id=repeat, same_as_repeat0=None, differing_request_positions=None,
                    reason='one of the raw repeat records is unavailable')
    if not first['counts_available'] or not other['counts_available']:
        return dict(repeat_id=repeat, same_as_repeat0=None, differing_request_positions=None,
                    reason='one repeat has no returned request work to compare')
    a = {row['query_position']: row['work_signature_sha256'] for row in first['per_request']}
    b = {row['query_position']: row['work_signature_sha256'] for row in other['per_request']}
    return dict(repeat_id=repeat, same_as_repeat0=first['work_signature_sha256'] == other['work_signature_sha256'],
                differing_request_positions=[q for q in sorted(set(a) | set(b)) if a.get(q) != b.get(q)],
                work_signature_sha256=other['work_signature_sha256'])


def process_ratios(method_rows):
    result = {}
    for reference in ('D', 'S', 'better'):
        pairs = []
        for repeat in range(3):
            h = method_rows['H'][repeat]
            hwall = h.get('process_wall_seconds') if h.get('complete') else None
            candidates = []
            for method in (('D', 'S') if reference == 'better' else (reference,)):
                row = method_rows[method][repeat]
                wall = row.get('process_wall_seconds')
                if row.get('complete') and number(wall) and wall > 0:
                    candidates.append((wall, method))
            wall, method = min(candidates) if candidates else (None, None)
            available = number(hwall) and hwall > 0 and wall is not None
            pairs.append(dict(repeat_id=repeat, H_process_seconds=hwall, reference_method=method,
                              reference_process_seconds=wall, ratio=hwall / wall if available else None))
        full = all(pair['ratio'] is not None for pair in pairs)
        result['H_over_' + reference] = dict(
            planned_pairs=3, completed_pairs=sum(pair['ratio'] is not None for pair in pairs), pairs=pairs,
            full_cell_ratio=(sum(pair['H_process_seconds'] for pair in pairs) /
                             sum(pair['reference_process_seconds'] for pair in pairs)) if full else None)
    return result


def phase_totals(metrics):
    metrics = list(metrics)
    return dict(
        worker_records=len(metrics),
        workers_with_measured_structural_time=sum(number(row.get('structural_seconds')) for row in metrics),
        structural_seconds=sum(row['structural_seconds'] for row in metrics if number(row.get('structural_seconds'))),
        structural_phase_seconds=sum_maps(row.get('structural_phases') or {} for row in metrics),
        request_exclusive_phase_seconds=sum_maps(row.get('request_phases_seconds', {}) for row in metrics),
        nested_frontier_build_seconds=sum_maps(row.get('frontier_build_phases_seconds', {}) for row in metrics),
        nested_index_build_seconds=sum_maps(row.get('index_build_phases_seconds', {}) for row in metrics))


def numeric_memory(memory, prefix=''):
    result = {}
    for key, value in memory.items():
        name = prefix + key
        if number(value):
            result[name] = value
        elif isinstance(value, dict):
            result.update(numeric_memory(value, name + '.'))
    return result


def primary_facts(judgment, workers, metrics):
    selected = judgment.get('primary_results', [])
    selection = [{key: row[key] for key in ('graph', 'stream', 'seed_label')} for row in selected]
    available = bool(selected) and all(row.get('complete') is True for row in selected)
    result = dict(
        selection=selection, selected_row_count=len(selected),
        complete_selected_row_count=sum(row.get('complete') is True for row in selected),
        selected_matched_Q4_blocks=3 * len(selected), available=available,
        scope='Exactly the frozen judgment primary_results rows, including their seed labels; not the full scheduled roster.',
        metric='sum(all H whole-Q4 service costs) / sum(min(D_r, S_r) for each matched whole-Q4 block)',
        service_totals_seconds=None, service_ratios=None, process_wall_totals_seconds=None,
        process_wall_ratios=None, exclusive_phase_totals_by_method=None,
        unavailable_rows=[{key: row[key] for key in ('graph', 'stream', 'seed_label')}
                          for row in selected if row.get('complete') is not True])
    if not available:
        result['unavailable_reason'] = 'No primary aggregate is formed unless every explicitly selected primary row completed.'
        return result
    costs = {method: [value for row in selected for value in row[method + '_seconds']] for method in METHODS}
    reference = [min(d, s) for d, s in zip(costs['D'], costs['S'], strict=True)]
    totals = {method: sum(values) for method, values in costs.items()}
    totals['better_completed_reference'] = sum(reference)
    result['service_totals_seconds'] = totals
    result['service_ratios'] = {f'H_over_{reference}': totals['H'] / totals[reference]
                               for reference in ('D', 'S', 'better_completed_reference')}
    result['selected_reference_methods'] = [method for row in selected for method in row['reference_methods']]
    selected_keys = [cell_key(row) + (repeat,) for row in selected for repeat in range(3)]
    walls = {method: [workers[key + (method,)].get('process_wall_seconds') for key in selected_keys]
             for method in METHODS}
    if all(number(value) and value > 0 for values in walls.values() for value in values):
        wall_totals = {method: sum(values) for method, values in walls.items()}
        wall_totals['better_completed_reference'] = sum(min(d, s) for d, s in zip(walls['D'], walls['S'], strict=True))
        result['process_wall_totals_seconds'] = wall_totals
        result['process_wall_ratios'] = {f'H_over_{reference}': wall_totals['H'] / wall_totals[reference]
                                        for reference in ('D', 'S', 'better_completed_reference')}
    result['exclusive_phase_totals_by_method'] = {
        method: phase_totals(metrics[key + (method,)] for key in selected_keys) for method in METHODS}
    return result


def extract(analysis, judgment, raw_directory, sources):
    workers = {worker_key(row): row for row in analysis['workers']}
    metrics = {cell_key(cell) + (row['repeat_id'], method): row for cell in analysis['cells']
               for method in METHODS for row in cell['work_and_memory'][method]}
    raw, work = {}, {}
    sources['raw_results'] = {}
    for key, row in workers.items():
        filename = (f"{row['position']:03d}_{row['graph']}_{row['stream']}_"
                    f"{row['seed_label']}_{row['label']}.json")
        path = raw_directory / filename
        if path.is_file():
            raw[key], sources['raw_results'][filename] = read_json(path)
        else:
            raw[key] = None
            sources['raw_results'][filename] = dict(path=str(path.resolve()), sha256=None, unavailable=True)
        work[key] = extract_work(raw[key])
    cells = []
    for cell in analysis['cells']:
        key = cell_key(cell)
        result = dict(zip(('graph', 'stream', 'seed_label'), key, strict=True))
        method_rows = {method: [workers[key + (repeat, method)] for repeat in range(3)] for method in METHODS}
        result['methods'] = {}
        result['exclusive_phases'] = {}
        for method in METHODS:
            official = cell['costs'][method]
            repeats = [work[key + (repeat, method)] for repeat in range(3)]
            comparisons = [compare_work(repeats[0], repeats[repeat], repeat) for repeat in (1, 2)]
            same = [row['same_as_repeat0'] for row in comparisons]
            result['methods'][method] = dict(
                planned_repeats=official['planned_workers'], complete_repeats=official['complete_workers'],
                complete_Q4_seconds=official['Q4_seconds'],
                complete_Q4_statistics=stats(official['Q4_seconds']), statuses=official['statuses'],
                complete_process_wall_statistics=stats(row['process_wall_seconds'] for row in method_rows[method]
                                                       if row.get('complete')),
                process_wall_seconds_by_repeat=[row.get('process_wall_seconds') for row in method_rows[method]],
                first_repeat_id=0, first_repeat_work=repeats[0], repeat_work_comparisons=comparisons,
                other_repeats_share_work=False if False in same else True if all(value is True for value in same) else None)
            selected_metrics = [metrics[key + (repeat, method)] for repeat in range(3)]
            result['exclusive_phases'][method] = dict(
                all_observed_including_failed_prefixes=phase_totals(selected_metrics),
                completed_workers_only=phase_totals(row for row in selected_metrics if row.get('complete')))
        result['full_cell_service_ratios'] = {
            name: dict(full_cell_ratio=record['full_cell_ratio'], planned_pairs=record['planned_pairs'],
                       completed_pairs=record['paired_count'], paired_ratios=record['ratios'], pairs=record['pairs'])
            for name, record in cell['comparisons']['Q4'].items()}
        result['full_cell_process_wall_ratios'] = process_ratios(method_rows)
        cells.append(result)
    memory_rows = []
    for graph in sorted({row['graph'] for row in analysis['workers']}):
        for method in METHODS:
            selected = [(key, row) for key, row in workers.items() if row['graph'] == graph and row['method'] == method]
            rss, memory_maxima = [], {}
            for key, _ in selected:
                metric = metrics[key]
                peak = maximum((metric.get('process_peak_rss_bytes'), metric.get('external_sampled_peak_rss_bytes')))
                if number(peak) and peak > 0:
                    rss.append(peak)
                for sample in metric.get('memory_samples', []):
                    for field, value in numeric_memory(sample).items():
                        memory_maxima[field] = max(memory_maxima.get(field, value), value)
            memory_rows.append(dict(
                graph=graph, method=method, registered_workers=len(selected),
                started_workers=sum(number(row.get('process_wall_seconds')) for _, row in selected),
                process_peak_RSS_bytes=stats(rss), process_peak_RSS_MiB=stats(value / 2**20 for value in rss),
                max_observed_Gram_entries=maximum(memory_maxima.get(field) for field in ('gram_count', 'peak_gram_count')),
                max_observed_Gram_python_bytes=memory_maxima.get('gram_python_bytes'),
                max_response_boundary_snapshot_bytes=memory_maxima.get('shared_response_bytes'),
                max_frontier_owned_bytes=memory_maxima.get('peak_frontier_owned_bytes'),
                max_emitted_batch_array_bytes=memory_maxima.get('peak_emitted_batch_array_bytes'),
                max_low_rank_cache_payload_bytes=maximum(memory_maxima.get(field) for field in ('low_rank_steps', 'peak_low_rank_steps')),
                all_numeric_memory_field_maxima=memory_maxima))
    return dict(
        schema_version=1, source_files=sources, stage=analysis.get('stage'),
        protocol_sha256=analysis.get('protocol_sha256'), input_manifest_sha256=analysis.get('input_manifest_sha256'),
        scope='Report-ready extraction from official results; no new numerical verification, gate or claim.',
        frozen_judgment=judgment,
        full_schedule=dict(
            planned_workers=analysis['planned_workers'], complete_workers=analysis['complete_workers'],
            status_counts=analysis['status_counts'], complete_workers_by_method=analysis.get('complete_workers_by_method'),
            full_Q4_totals_seconds=analysis['full_schedule_totals_seconds'],
            full_process_totals_seconds=analysis.get('full_schedule_process_totals_seconds'),
            aggregate_comparisons=analysis.get('aggregate_comparisons'),
            observed_service_seconds_including_partial=analysis.get('observed_service_seconds_including_partial'),
            observed_process_seconds_including_partial=analysis.get('observed_process_seconds_including_partial'),
            reference_resource_failures=analysis.get('reference_resource_failures')),
        restricted_primary=primary_facts(judgment, workers, metrics), label_specific_cells=cells,
        graph_method_memory=memory_rows,
        official_exclusive_phase_totals_by_method={method: dict(
            all_observed_including_failed_prefixes=phase_totals(row for key, row in metrics.items() if key[-1] == method),
            completed_workers_only=phase_totals(row for key, row in metrics.items() if key[-1] == method and row.get('complete')))
            for method in METHODS},
        interpretation_notes={
            'ranges': 'Exact minimum/maximum of completed repeated observations, not confidence intervals; completion counts remain explicit.',
            'primary': 'Selection is exactly frozen_judgment.primary_results. No partial-primary total is substituted when any selected row is incomplete.',
            'full_schedule': 'Official unavailable full-roster totals stay null even when the restricted primary total is available.',
            'process_better': 'Secondary process-wall ratios choose the smaller complete D/S process wall for each matched block; this is not a new acceptance gate.',
            'work': 'Repeat0 is always the representative, including its failures. Work equality compares reported counters, per-version access, accepted IDs and outcomes; timings/RSS are excluded.',
            'counts': 'K, heap/source pops, endpoint rows, Grams, response blocks and scalar response RHS are distinct. State RHS is separate.',
            'memory': 'RSS covers all started workers with positive measurements and uses max(reported peak, sampled peak). Owned payload snapshots are separate; unknown cache bytes remain null.',
            'phases': 'Structural and request ledgers are exclusive. Nested frontier/index build timings are separate and must not be added twice. Killed in-flight work absent from checkpoints is not inferred.',
            'claim': 'Use only the supplied frozen judgment claim, status, gates and scope; this extractor does not generalize or amend them.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('analysis', 'raw-directory', 'judgment', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    arguments = parser.parse_args()
    analysis, analysis_source = read_json(arguments.analysis)
    judgment, judgment_source = read_json(arguments.judgment)
    sources = dict(analysis=analysis_source, judgment=judgment_source,
                   extractor=dict(path=str(Path(__file__).resolve()),
                                  sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    result = extract(analysis, judgment, arguments.raw_directory, sources)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(arguments.output)
    print(json.dumps(dict(output=str(arguments.output.resolve()), judgment_status=judgment.get('status'),
                          primary_available=result['restricted_primary']['available'],
                          label_specific_cells=len(result['label_specific_cells'])), indent=2))


if __name__ == '__main__':
    main()
