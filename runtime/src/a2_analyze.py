"""Offline A2 full-roster accounting and descriptive research-value comparison.

Resource failures stay in the denominator. A complete paired comparison is
available only where both costs really completed; failure never becomes infinity.
This validator is not the independent full-catalog numerical replay.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import statistics

from .a2_bench import (DOMAIN, ROOT, THREAD_ENV, TERMINAL_REQUESTS, _check_seconds,
                       _graph_record, job_key, jobs, load_fixture, load_freeze,
                       read_protocol, result_filename, save, sha)

ACCESS_COUNTS = ('frontier_rebuilds', 'raw_heap_pops', 'graph_rejects',
                 'committed_rejects', 'processed_skips', 'insertion_emissions',
                 'flips_emitted', 'stream_pairs_examined', 'explicit_raw_visits',
                 'emitted_for_screen', 'node_sort_items', 'flip_sort_items',
                 'legal_insertion_pops', 'processed_insertion_skips',
                 'processed_flip_skips', 'flip_cursor_pops', 'total_source_pops',
                 'stream_positive_rejects', 'stream_processed_skips',
                 'stream_insertions_emitted', 'stream_flips_emitted')
RESOURCE_FAILURES = {'timeout', 'rss_cap', 'memory_error', 'batch_cap', 'batch_cap_not_started'}
WORKER_TERMINAL = {'completed', 'completed_with_failures', 'error', 'memory_error',
                   'timeout', 'rss_cap', 'batch_cap', 'batch_cap_not_started',
                   'process_launch_error', 'worker_exit', 'exit_before_checkpoint',
                   'invalid_checkpoint'}


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _integer(value):
    return type(value) is int and value >= 0


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _objects(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _requests(trace):
    return _objects(trace.get('requests', []))


def _observed_service(trace):
    structural = trace.get('structural_seconds')
    return (structural if _number(structural) else 0) + sum(
        r['request_seconds'] for r in _requests(trace) if _number(r.get('request_seconds')))


def _resource_failure(row):
    """Classify once per worker, including MemoryError caught by the session."""
    trace, scheduled = row.get('trace') or {}, row.get('scheduler') or {}
    return (scheduled.get('status') in RESOURCE_FAILURES or trace.get('status') == 'memory_error' or
            any(request.get('status') == 'memory_error' for request in _requests(trace)))


def _events(request):
    access = request.get('access', {})
    return _objects(access.get('epochs', [])) if isinstance(access, dict) else []


def _prefix(row, q):
    if not row.get('valid', False):
        return None
    trace = row.get('trace') or {}
    requests = trace.get('requests', [])
    cost = trace.get('prefixes', {}).get(str(q))
    if (not row.get('valid', False) or not _number(cost) or cost <= 0 or
            len(requests) < q or any(r.get('status') not in TERMINAL_REQUESTS for r in requests[:q])):
        return None
    return cost


def _complete(row, cfg):
    if not row.get('valid', False):
        return False
    trace, scheduled = row.get('trace') or {}, row.get('scheduler') or {}
    return (trace.get('status') == scheduled.get('status') == 'completed' and
            scheduled.get('returncode') == 0 and len(trace.get('requests', [])) == cfg['requests'] and
            _prefix(row, cfg['requests']) is not None)


def _work(trace):
    counters, access, phases, memory = defaultdict(int), defaultdict(int), defaultdict(float), {}
    frontier_phases, index_phases, versions, memory_samples = defaultdict(float), defaultdict(float), [], []
    for q, request in enumerate(_requests(trace), 1):
        for key, value in _mapping(request.get('counters')).items():
            if _integer(value):
                counters[key] += value
        for event in _events(request):
            for key in ACCESS_COUNTS:
                value = event.get(key)
                if _integer(value):
                    access[key] += value
            for key, value in _mapping(event.get('frontier_build_phases')).items():
                if _number(value):
                    frontier_phases[key] += value
            decision = next((d for d in _objects(request.get('decisions')) if d.get('epoch') == event.get('epoch')), {})
            live = decision.get('live_catalog')
            emitted = event.get('insertion_emissions')
            versions.append(dict(query_position=q, **event, registered_catalog_size=trace.get('catalog_size'),
                                 current_legal_candidates=live,
                                 insertion_emissions_over_current_legal_catalog=(emitted / live)
                                 if _integer(emitted) and _integer(live) and live > 0 else None))
        for key, value in _mapping(request.get('index_build_phases')).items():
            if _number(value):
                index_phases[key] += value
        for key, value in _mapping(request.get('phases')).items():
            if _number(value):
                phases[key] += value
        for key, value in _mapping(request.get('memory')).items():
            if _number(value):
                memory[key] = max(memory.get(key, 0), value)
        memory_samples.append(dict(query_position=q, **_mapping(request.get('memory'))))
    return dict(counters=dict(counters), access=dict(access),
                request_phases_seconds=dict(phases), memory_sample_maxima=memory,
                frontier_build_phases_seconds=dict(frontier_phases),
                index_build_phases_seconds=dict(index_phases), versions=versions,
                memory_samples=memory_samples)


def _pairs(rows_by_method, q, reference, cfg):
    def cost(row):
        # Q1 is explicitly a prefix measurement. Q4 is the complete registered
        # service and cannot come from a killed/error process with a late checkpoint.
        return _prefix(row, q) if q != cfg['requests'] or _complete(row, cfg) else None

    pairs = []
    for repeat in range(cfg['repeat_count']):
        h = rows_by_method['H'][repeat]
        hcost = cost(h)
        references = ('D', 'S') if reference == 'better_reference' else (reference,)
        available = [(method, cost(rows_by_method[method][repeat])) for method in references]
        available = [(method, cost) for method, cost in available if cost is not None]
        chosen, baseline = min(available, key=lambda x: (x[1], x[0])) if available else (None, None)
        ratio = hcost / baseline if hcost is not None and baseline is not None and baseline > 0 else None
        pairs.append(dict(repeat_id=repeat, H_seconds=hcost, reference_seconds=baseline,
                          reference_method=chosen, ratio=ratio,
                          H_status=(h.get('scheduler') or {}).get('status', 'missing_result'),
                          reference_statuses={method: (rows_by_method[method][repeat].get('scheduler') or {}).get(
                              'status', 'missing_result') for method in references}))
    ratios = [p['ratio'] for p in pairs if p['ratio'] is not None]
    complete = len(ratios) == cfg['repeat_count']
    return dict(planned_pairs=cfg['repeat_count'], paired_count=len(ratios), pairs=pairs,
                ratios=ratios, median_ratio=statistics.median(ratios) if ratios else None,
                full_cell_ratio=(sum(p['H_seconds'] for p in pairs) /
                                 sum(p['reference_seconds'] for p in pairs)) if complete else None,
                complete_pair_set=complete)


def summarize(cfg, observations):
    """Pure descriptive aggregation; no research gate and no dropped failures."""
    lookup = {job_key(row['job']): row for row in observations}
    normalized = []
    for position, job in enumerate(jobs(cfg)):
        normalized.append(lookup.get(job_key(job), dict(position=position, job=job, trace=None,
                          scheduler={'status': 'missing_result', 'returncode': None,
                                     'process_wall_seconds': None}, valid=False,
                          issues=['registered worker has no result'])))
    statuses = Counter((row.get('scheduler') or {}).get('status', 'missing_result') for row in normalized)
    bykey = {job_key(row['job']): row for row in normalized}
    methods = [m['method'] for m in cfg['methods']]
    total_service, total_process, observed_service, observed_process, complete_counts = {}, {}, {}, {}, {}
    for method in methods:
        selected = [r for r in normalized if r['job']['method'] == method]
        complete = [r for r in selected if _complete(r, cfg)]
        complete_counts[method] = len(complete)
        total_service[method] = (sum(_prefix(r, 4) for r in complete) if len(complete) == len(selected) else None)
        process = [(r.get('scheduler') or {}).get('process_wall_seconds') for r in complete]
        total_process[method] = (sum(process) if len(complete) == len(selected) and all(_number(x) for x in process) else None)
        observed_service[method] = sum(_observed_service(r.get('trace') or {}) for r in selected)
        observed_process[method] = sum((r.get('scheduler') or {})['process_wall_seconds'] for r in selected
            if _number((r.get('scheduler') or {}).get('process_wall_seconds')))
    cells = []
    for graph, stream, label in product(cfg['graphs'], cfg['streams'], cfg['seed_labels']):
        rows = {method: [bykey[graph, stream, label, repeat, method]
                         for repeat in range(cfg['repeat_count'])] for method in methods}
        comparisons = {f'Q{q}': {f'H_over_{reference}': _pairs(rows, q, reference, cfg)
                                 for reference in ('D', 'S', 'better_reference')}
                       for q in cfg['prefixes']}
        costs, metrics = {}, {}
        for method in methods:
            selected = rows[method]
            full = [r for r in selected if _complete(r, cfg)]
            q4 = [_prefix(r, 4) if _complete(r, cfg) else None for r in selected]
            available = [cost for cost in q4 if cost is not None]
            costs[method] = dict(planned_workers=len(selected), complete_workers=len(full),
                Q1_seconds=[_prefix(r, 1) for r in selected], Q4_seconds=q4,
                Q4_complete_median_seconds=statistics.median(available) if available else None,
                process_wall_seconds=[(r.get('scheduler') or {}).get('process_wall_seconds') for r in selected],
                statuses=[(r.get('scheduler') or {}).get('status', 'missing_result') for r in selected])
            metrics[method] = [dict(repeat_id=r['job']['repeat_id'], complete=_complete(r, cfg),
                catalog_size=(r.get('trace') or {}).get('catalog_size'),
                structural_seconds=(r.get('trace') or {}).get('structural_seconds'),
                structural_phases=(r.get('trace') or {}).get('structural_phases'),
                process_peak_rss_bytes=(r.get('trace') or {}).get('process_peak_rss_bytes'),
                external_sampled_peak_rss_bytes=(r.get('scheduler') or {}).get('external_sampled_peak_rss_bytes'),
                **_work(r.get('trace') or {})) for r in selected]
        cells.append(dict(graph=graph, stream=stream, seed_label=label,
                          costs=costs, comparisons=comparisons, work_and_memory=metrics))
    aggregate = {}
    paired_coverage = {}
    for reference in ('D', 'S', 'better_reference'):
        name = f'H_over_{reference}'
        pairs = [p for c in cells for p in c['comparisons']['Q4'][name]['pairs']]
        usable = [p for p in pairs if p['ratio'] is not None]
        full = len(usable) == len(pairs)
        # Fixed-reference full-schedule comparison additionally requires genuinely
        # completed processes. A completed Q4 checkpoint before a kill stays partial.
        processes_complete = total_service['H'] is not None and (
            reference == 'better_reference' or total_service[reference] is not None)
        if reference == 'better_reference':
            processes_complete = processes_complete and all(
                any(_complete(bykey[c['graph'], c['stream'], c['seed_label'], repeat, method], cfg)
                    for method in ('D', 'S')) for c in cells for repeat in range(cfg['repeat_count']))
        aggregate[name] = (sum(p['H_seconds'] for p in usable) / sum(p['reference_seconds'] for p in usable)
                           if full and processes_complete else None)
        paired_coverage[name] = dict(planned_pairs=len(pairs), paired_count=len(usable),
            selected_reference_counts=dict(Counter(p['reference_method'] for p in usable)),
            complete_schedule_comparison=aggregate[name] is not None)
    negative = []
    for cell in cells:
        for reference, comparison in cell['comparisons']['Q4'].items():
            if comparison['median_ratio'] is not None and comparison['median_ratio'] > 1:
                negative.append(dict(graph=cell['graph'], stream=cell['stream'],
                    seed_label=cell['seed_label'], comparison=reference,
                    paired_count=comparison['paired_count'], median_ratio=comparison['median_ratio'],
                    full_cell_ratio=comparison['full_cell_ratio']))
    workers, failures = [], []
    for row in normalized:
        trace, scheduled = row.get('trace') or {}, row.get('scheduler') or {}
        item = dict(position=row['position'], **row['job'], status=scheduled.get('status', 'missing_result'),
                    trace_status=trace.get('status'), valid=row.get('valid', False),
                    complete=_complete(row, cfg), returned_requests=len(_requests(trace)),
                    terminal_requests=sum(r.get('status') in TERMINAL_REQUESTS for r in _requests(trace)),
                    Q1_seconds=_prefix(row, 1), Q4_seconds=_prefix(row, 4),
                    structural_seconds=trace.get('structural_seconds'),
                    process_wall_seconds=scheduled.get('process_wall_seconds'),
                    observed_service_seconds=trace.get('service_accounted_seconds'),
                    resource_failure=_resource_failure(row),
                    failure_reason=trace.get('failure_reason') or scheduled.get('failure_reason'),
                    issues=row.get('issues', []))
        workers.append(item)
        if not item['complete']:
            failures.append(item)
    resources = {method: sum(_resource_failure(r)
                             for r in normalized if r['job']['method'] == method) for method in ('D', 'S')}
    advantages = {method: sum(_complete(bykey[c['graph'], c['stream'], c['seed_label'], repeat, 'H'], cfg)
        and _resource_failure(bykey[c['graph'], c['stream'], c['seed_label'], repeat, method])
        for c in cells for repeat in range(cfg['repeat_count'])) for method in ('D', 'S')}
    return dict(planned_workers=len(normalized), observed_result_workers=sum(r.get('trace') is not None for r in normalized),
                complete_workers=sum(complete_counts.values()), complete_workers_by_method=complete_counts,
                status_counts=dict(statuses), cells=cells, workers=workers, failures=failures,
                full_schedule_totals_seconds=total_service,
                full_schedule_process_totals_seconds=total_process,
                observed_service_seconds_including_partial=observed_service,
                observed_process_seconds_including_partial=observed_process,
                aggregate_comparisons=aggregate, paired_coverage=paired_coverage,
                negative_cells=negative, reference_resource_failures=resources,
                H_completed_against_reference_resource_failure=advantages,
                research_value_summary={
                    'decision': 'development evidence only; root must make the bounded research judgment',
                    'comparison_basis': 'complete structural plus Q4 service; Q1 prefix and process walls reported separately',
                    'H_faster_than_both_complete_reference_totals':
                        all(aggregate[f'H_over_{r}'] is not None and aggregate[f'H_over_{r}'] < 1 for r in ('D', 'S')),
                    'H_faster_than_complete_streaming_total':
                        aggregate['H_over_S'] is not None and aggregate['H_over_S'] < 1,
                    'H_faster_than_better_completed_reference':
                        aggregate['H_over_better_reference'] is not None and aggregate['H_over_better_reference'] < 1,
                    'negative_cell_count': len(negative),
                    'no_automatic_percentage_gate': True,
                },
                comparability_note='Every registered worker remains. Partial Q1/Q4 checkpoints are shown as prefixes, never as complete process service. Ratios have finite completed reference costs only; resource failures are separate evidence. Per-cell partial-pair medians are descriptive and label their reduced paired coverage. No A1 gate is reused.')


def validate_request(request, trace, fixture, position, spec, W):
    """Validate emitted semantics against the sparse root, without an M-entry map."""
    _need(request['query_position'] == position and request['opinion_hash'] == fixture['opinion_hashes'][position - 1],
          'request position/opinion digest')
    _need(request['stream_id'] == trace['stream_id'] and request['method'] == trace['method'], 'request method/stream')
    _need(request['budget'] == 15 and request['delta'] == 0 and request['initial_gram_count'] == 0,
          'request budget/delta or nonempty initial Grams')
    _check_seconds(request['phases'], request['request_seconds'], 'request')
    _check_seconds(request['online_phases'], request['online_seconds'], 'online')
    _need(_number(request['reset_opinion_seconds']) and
          abs(request['reset_opinion_seconds'] + request['online_seconds'] - request['request_seconds']) <= 1e-7,
          'reset/online reconciliation')
    _need(request['root_version_id'] == trace['root_version_id'], 'request structural root reset')
    n, m = spec['n'], spec['catalog_size']
    seeds = request['initial_seed_ids']
    _need(isinstance(seeds, list) and len(seeds) <= 64 and len(set(seeds)) == len(seeds), 'seed IDs/order')
    order = request['initial_seed_order']
    _need(isinstance(order, list) and len(order) == len(seeds) and len(set(order)) == len(order)
          and set(order) == set(seeds), 'recorded seed order is not the actual initial seed permutation')
    expected_seed_rule = ('paid-global-root-caps-top64/canonical-ties' if trace['method'] == 'D' else
                          'paid-joint-frontier-first64/heap-order-recorded')
    _need(request['seed_rule'] == expected_seed_rule, 'seed work declaration')
    for e in seeds:
        _need(_integer(e), 'canonical seed ID type')
        i, j = divmod(e, n)
        _need(0 <= i < j < n and W[i, j] <= 0, 'seed outside the root domain')
    counters, memory = request['counters'], request['memory']
    for key in ('response_blocks', 'candidate_rhs', 'gram_materializations', 'endpoint_checked'):
        _need(_integer(counters[key]), 'invalid counter ' + key)
    _need(all(_integer(v) for v in counters.values()), 'all numerical counters must be nonnegative integers')
    roles = ('candidate_rhs', 'winner_rhs', 'policy_refresh_rhs', 'numerical_reanchor_rhs')
    if all(name in counters for name in roles + ('response_residual_matvec_columns',)):
        _need(sum(counters[name] for name in roles) == counters['response_residual_matvec_columns'],
              'response scalar-RHS role split')
        _need(counters['response_blocks'] <= counters['response_residual_matvec_columns'], 'response block/RHS count')
    _need(type(memory['catalog_materialized']) is bool and
          _integer(memory['candidate_array_rows']) and _integer(memory['full_root_caps_created_rows']) and
          isinstance(memory['m_sized_structures'], list), 'memory structure declarations')
    if trace['method'] in ('H', 'S'):
        _need(memory['catalog_materialized'] is False and memory['candidate_array_rows'] == 0 and
              memory['full_root_caps_created_rows'] == 0 and memory['m_sized_structures'] == [],
              'H/S materialized an M-sized candidate structure')
    else:
        _need(memory['catalog_materialized'] is True and memory['candidate_array_rows'] == m and
              memory['full_root_caps_created_rows'] == m, 'D explicit reference structure declaration')
    decisions, epochs = request['decisions'], _events(request)
    _need(isinstance(decisions, list) and len(decisions) <= 15 and len(epochs) == len(decisions),
          'decision/access epoch coverage')
    parent, committed, lows, highs = trace['root_version_id'], set(), 0, 0
    endpoint_total = 0
    for index, (decision, event) in enumerate(zip(decisions, epochs, strict=True), 1):
        _need(decision['epoch'] == index and decision['root_version_id'] == trace['root_version_id'] and
              decision['parent_version_id'] == parent, 'decision root/parent/epoch binding')
        _need(event['epoch'] == index and event['parent_version_id'] == parent and event['source'] == trace['method'],
              'access current-version/source binding')
        for name in ACCESS_COUNTS:
            _need(_integer(event[name]), 'invalid access work counter ' + name)
        status = decision['status']
        expected_rebuild = 1 if trace['method'] == 'H' or (trace['method'] == 'S' and index == 1) else 0
        completed_coverage = status in ('accepted', 'tolerance_stop') or event.get('source_complete') is True
        _need(event['frontier_rebuilds'] == expected_rebuild if completed_coverage else
              event['frontier_rebuilds'] in (0, expected_rebuild),
              'frontier not rebuilt at the required current version')
        _need(type(event['source_complete']) is bool and
              (event['remaining_upper_coarse'] is None or type(event['remaining_upper_coarse']) is int),
              'remaining frontier coverage declaration')
        if status in ('accepted', 'tolerance_stop'):
            _need(event['source_complete'] is True, 'decision published before coverage source completed')
            _need(decision['live_catalog'] == m - len(committed) and decision['proof_scope'] == 'full_catalog',
                  'full current legal catalog denominator/scope')
            _need(_integer(decision['endpoint_checked']) and _integer(decision['endpoint_excluded']) and
                  decision['endpoint_excluded'] <= decision['endpoint_checked'], 'decision endpoint counters')
            endpoint_total += decision['endpoint_checked']
        if status == 'accepted':
            e = decision['winner_id']
            _need(_integer(e) and e not in committed, 'canonical winner type or repeated commit')
            i, j = divmod(e, n)
            _need(0 <= i < j < n and W[i, j] <= 0, 'winner outside root nonedge/negative-flip domain')
            root_weight = float(W[i, j])
            _need((decision['i'], decision['j'], decision['flip']) == (i, j, int(root_weight < 0)),
                  'winner endpoint/kind mismatch')
            _need(type(decision['d']) is int and decision['d'] ==
                  (int(16 * abs(root_weight)) if root_weight < 0 else 1), 'winner scaled physical weight mismatch')
            if 'weight' in decision:
                _need(decision['weight'] == (abs(root_weight) if root_weight < 0 else 1 / 16),
                      'winner physical weight mismatch')
            low, high = decision['low'], decision['high']
            _need(type(low) is int and type(high) is int and 0 < low <= high, 'accepted gain enclosure')
            lows, highs = lows + low, highs + high
            committed.add(e)
            parent = hashlib.sha256(f'{parent}/{e}'.encode()).hexdigest()
        elif status == 'tolerance_stop':
            _need(decision['winner_id'] is None and index == len(decisions) and request['status'] == 'tolerance_stop',
                  'tolerance stop/winner/position')
            _need(decision['low'] == 0 and decision['high'] is None, 'no-positive gain convention')
            for name in ('scored_rival_max_high', 'cold_upper', 'competitor_max_upper'):
                _need(decision.get(name) is None or decision[name] <= 0, 'positive rival at tolerance stop')
        else:
            _need(status in ('unresolved', 'memory_error', 'error') and index == len(decisions) and
                  request['status'] == status and decision['winner_id'] is None, 'failure decision semantics')
        _need(decision['current_version_id'] == parent, 'canonical child version chain')
    _need(request['current_version_id'] == parent and request['commits'] == len(committed),
          'request final version/commit count')
    _need(request['gain_low'] == lows and request['gain_high'] == highs and
          _integer(request['gain_denominator']) and request['gain_denominator'] > 0, 'request cumulative gain enclosure')
    if request['status'] == 'budget_stop':
        _need(len(committed) == 15 and len(decisions) == 15, 'partial budget-stop result')
    elif request['status'] == 'tolerance_stop':
        _need(bool(decisions) and decisions[-1]['status'] == 'tolerance_stop', 'missing terminal decision')
    else:
        _need(request['status'] in ('unresolved', 'memory_error', 'error'), 'unknown request status')
    if request['status'] in TERMINAL_REQUESTS:
        _need(counters['endpoint_checked'] == endpoint_total, 'request/decision endpoint count split')


def _validate_trace(cfg, manifest, manifest_hash, job, position, trace, scheduled, fixture, spec, W):
    _need(all(trace.get(k) == v for k, v in job.items()), 'job identity')
    _need(trace['schema_version'] == 2 and trace['schedule_position'] == position, 'schema/schedule position')
    for key, expected in (('protocol_id', cfg['id']), ('protocol_sha256', manifest['protocol_sha256']),
                          ('source_files_sha256', manifest['source_files_sha256']),
                          ('input_source_files_sha256', manifest['input_source_files_sha256']),
                          ('input_manifest_sha256', manifest_hash), ('environment', manifest['environment']),
                          ('domain', DOMAIN), ('requests_planned', 4), ('budget', 15),
                          ('delta', 0), ('hot_size', 64), ('audit', False), ('cpu', 0),
                          ('weight_mode', 'weighted')):
        _need(trace.get(key) == expected, 'trace provenance/settings ' + key)
    status = scheduled['status']
    _need(status in WORKER_TERMINAL and trace['status'] == status, 'worker/scheduler status')
    requests = trace['requests']
    _need(isinstance(requests, list) and len(requests) <= 4 and scheduled['completed_requests'] == len(requests),
          'scheduler request-count binding')
    if status == 'batch_cap_not_started':
        _need(not requests and scheduled['returncode'] is None and scheduled['process_wall_seconds'] is None,
              'not-started worker has execution cost/output')
    if 'affinity' in trace:
        _need(trace['cpu'] == 0 and trace['affinity'] == [0], 'worker CPU affinity')
    if 'threadpools' in trace:
        _need(trace['threadpools'] and all(p['num_threads'] == 1 for p in trace['threadpools']), 'numerical thread pools')
        _need(trace['thread_environment'] == {name: '1' for name in THREAD_ENV}, 'numerical thread environment')
    if scheduled['process_wall_seconds'] is not None:
        _need(_number(scheduled['process_wall_seconds']) and
              scheduled['process_wall_seconds'] <= cfg['limits']['worker_seconds'] + 2., 'worker process wall cap')
    peak = scheduled.get('external_sampled_peak_rss_bytes')
    _need(peak is None or _integer(peak), 'sampled peak RSS')
    if 'process_peak_rss_bytes' in trace:
        _need(_integer(trace['process_peak_rss_bytes']), 'worker peak RSS')
    if status == 'completed':
        _need(scheduled['returncode'] == 0 and len(requests) == 4 and
              all(r['status'] in TERMINAL_REQUESTS for r in requests), 'completed worker includes missing/failed requests')
        _need(trace['affinity'] == [0] and trace['threadpools'], 'completed worker lacks environment evidence')
        _need(peak is not None and max(peak, trace['process_peak_rss_bytes']) <= cfg['limits']['rss_mib'] * 1024**2,
              'completed worker exceeded RSS cap')
    if 'structural_seconds' not in trace:
        _need(not requests and not trace['prefixes'], 'requests/prefixes exist without structural setup')
        return
    _need(trace['n'] == spec['n'] and trace['catalog_size'] == spec['catalog_size'], 'graph/full-catalog dimensions')
    for key, value in (('graph_hash', spec['weighted_sha256']), ('fixture_id', fixture['fixture_id']),
                        ('fixture_npz_sha256', fixture['npz_sha256']), ('fixture_matrix_sha256', fixture['matrix_sha256'])):
        _need(trace[key] == value, 'graph/fixture binding ' + key)
    for key in ('root_version_id', 'operator_hash', 'catalog_hash'):
        _need(isinstance(trace[key], str) and len(trace[key]) == 64 and
              all(c in '0123456789abcdef' for c in trace[key]), 'invalid ' + key)
    _check_seconds(trace['structural_phases'], trace['structural_seconds'], 'structural')
    cumulative = trace['structural_seconds']
    expected_prefixes = {}
    for q, request in enumerate(requests, 1):
        validate_request(request, trace, fixture, q, spec, W)
        cumulative += request['request_seconds']
        if q in cfg['prefixes']:
            expected_prefixes[str(q)] = cumulative
    _need(set(trace['prefixes']) == set(expected_prefixes) and
          all(abs(trace['prefixes'][key] - value) <= 1e-7 for key, value in expected_prefixes.items()),
          'Q1/Q4 prefix must equal setup plus every preceding request')
    _need(abs(trace['service_accounted_seconds'] - cumulative) <= 1e-7, 'cumulative service reconciliation')


def _compare_outputs(observations):
    issues = []
    groups, bindings = defaultdict(list), defaultdict(set)
    for row in observations:
        if not row.get('valid') or not row.get('trace'):
            continue
        trace = row['trace']
        if 'structural_seconds' in trace:
            bindings[row['job']['graph']].add(tuple(trace[k] for k in ('n', 'catalog_size', 'operator_hash', 'catalog_hash', 'root_version_id')))
        for q, request in enumerate(trace['requests'], 1):
            groups[(row['job']['graph'], row['job']['stream'], row['job']['seed_label'], q)].append((row, request))
    for graph, binding in bindings.items():
        if len(binding) > 1:
            issues.append(f'{graph}: method/repeat structural operator/catalog/root identity mismatch')
    for key, records in groups.items():
        # A failed first worker is not a complete-output reference. Prefer a
        # completed request; if all failed, use the longest certified prefix so
        # an early failure cannot hide a later shared-epoch disagreement.
        completed = [record for record in records if record[1]['status'] in TERMINAL_REQUESTS]
        reference_row, reference = (completed[0] if completed else max(records,
            key=lambda record: sum(d['status'] == 'accepted' for d in record[1]['decisions'])))
        reference_accepted = [d for d in reference['decisions'] if d['status'] == 'accepted']
        for row, request in records:
            if row is reference_row:
                continue
            try:
                _need(request['gain_denominator'] == reference['gain_denominator'], 'request gain-scale mismatch')
                both_complete = request['status'] in TERMINAL_REQUESTS and reference['status'] in TERMINAL_REQUESTS
                if both_complete:
                    _need(all(request[k] == reference[k] for k in ('status', 'current_version_id', 'commits')),
                          'request status/version/commit mismatch')
                    _need(max(request['gain_low'], reference['gain_low']) <= min(request['gain_high'], reference['gain_high']),
                          'cumulative gain intervals do not overlap')
                    _need(len(request['decisions']) == len(reference['decisions']), 'decision-count mismatch')
                    pairs = zip(request['decisions'], reference['decisions'], strict=True)
                else:
                    accepted = [d for d in request['decisions'] if d['status'] == 'accepted']
                    if reference['status'] == 'tolerance_stop':
                        _need(len(accepted) <= len(reference_accepted),
                              'partial positive commit extends past a completed tolerance stop')
                    pairs = zip(accepted, reference_accepted)
                for a, b in pairs:
                    _need(all(a.get(k) == b.get(k) for k in ('status', 'epoch', 'parent_version_id', 'current_version_id', 'winner_id')),
                          'shared decision status/version/canonical-winner mismatch')
                    if a['status'] == 'accepted':
                        _need(max(a['low'], b['low']) <= min(a['high'], b['high']), 'decision gain intervals do not overlap')
            except (KeyError, TypeError, ValueError) as error:
                issues.append(f'{key} {row["job"]["method"]} repeat{row["job"]["repeat_id"]}: {error}')
                row['valid'] = False
                row['issues'].append(str(error))
    return issues


def analyze(protocol_path, directory, output, manifest_path):
    protocol_path, directory, manifest_path = map(Path, (protocol_path, directory, manifest_path))
    cfg, protocol_hash = read_protocol(protocol_path)
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = sha(manifest_path)
    issues = []
    try:
        load_freeze(protocol_path, manifest_path)
    except (OSError, KeyError, TypeError, ValueError) as error:
        issues.append('freeze: ' + str(error))
    fixtures = {(r['graph'], r['stream'], r['seed_label']): r for r in manifest['fixtures']}
    sparse_graphs = {}
    for graph in cfg['graphs']:
        try:
            spec, _ = _graph_record(graph)
            _need(spec == manifest['graphs'][graph], 'graph metadata/catalog cardinality changed')
            from scipy.sparse import load_npz
            sparse_graphs[graph] = load_npz(ROOT / spec['weighted_path'])
        except (OSError, KeyError, TypeError, ValueError) as error:
            issues.append(f'graph {graph}: {error}')
    for key, record in fixtures.items():
        try:
            load_fixture(record, manifest_path.parent)
        except (OSError, KeyError, TypeError, ValueError) as error:
            issues.append(f'fixture {key}: {error}')
    schedule = list(jobs(cfg))
    frozen, scheduler = {}, {}
    try:
        frozen = json.loads((directory / 'FROZEN_JOBS.json').read_text())
        scheduler = json.loads((directory / 'SCHEDULE_RESULTS.json').read_text())
        _need(frozen['schema_version'] == 2 and frozen['jobs'] == schedule, 'frozen complete rotated roster')
        _need(frozen['cpu'] == 0 and frozen['supervisor_cpu'] == 8 and
              frozen['supervisor_affinity'] == [8] and frozen['limits'] == cfg['limits'], 'frozen CPU/resource policy')
        for name, expected in (('protocol_sha256', protocol_hash), ('input_manifest_sha256', manifest_hash),
                                ('source_files_sha256', manifest['source_files_sha256']),
                                ('input_source_files_sha256', manifest['input_source_files_sha256']),
                                ('environment', manifest['environment']), ('domain', DOMAIN)):
            _need(frozen[name] == expected, 'scheduler provenance ' + name)
        _need(scheduler['schema_version'] == 2 and scheduler['limits'] == cfg['limits'] and
              scheduler['supervisor_affinity'] == [8], 'observed scheduler policy')
        _need(len(scheduler['jobs']) == len(schedule), 'scheduler lost registered workers')
        _need(scheduler['status'] in ('completed', 'completed_with_failures'), 'scheduler did not terminate full roster')
        expected_status = 'completed' if all(r['status'] == 'completed' for r in scheduler['jobs']) else 'completed_with_failures'
        _need(scheduler['status'] == expected_status, 'scheduler final status')
        _need(_number(scheduler['batch_wall_seconds']) and
              scheduler['batch_wall_seconds'] <= cfg['limits']['batch_seconds'] + 2., 'batch elapsed cap')
    except (OSError, KeyError, TypeError, ValueError) as error:
        issues.append('schedule: ' + str(error))
    expected_files = {result_filename(i, j) for i, j in enumerate(schedule)}
    extras = {p.name for p in directory.glob('*.json')} - expected_files - {'FROZEN_JOBS.json', 'SCHEDULE_RESULTS.json'}
    if extras:
        issues.append('unregistered result files: ' + ', '.join(sorted(extras)))
    global_integrity_pass = not issues
    observations = []
    for position, job in enumerate(schedule):
        filename = result_filename(position, job)
        path = directory / filename
        row = dict(position=position, job=job, trace=None, scheduler=None, valid=True, issues=[])
        try:
            scheduled = scheduler['jobs'][position]
            row['scheduler'] = scheduled
            _need(scheduled['position'] == position and scheduled['job'] == job and scheduled['result'] == filename,
                  'scheduler file/position/job mapping')
            trace = json.loads(path.read_text())
            _need(isinstance(trace, dict), 'result file must contain a JSON object')
            row['trace'] = trace
            _validate_trace(cfg, manifest, manifest_hash, job, position, trace, scheduled,
                            fixtures[job['graph'], job['stream'], job['seed_label']],
                            manifest['graphs'][job['graph']], sparse_graphs[job['graph']])
        except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
            row['valid'] = False
            row['issues'].append(str(error))
            issues.append(filename + ': ' + str(error))
        if not global_integrity_pass:
            row['valid'] = False
            row['issues'].append('global freeze/input/scheduler integrity failed')
        if not isinstance(row['scheduler'], dict):
            row['scheduler'] = {'status': 'invalid_schedule_entry', 'returncode': None,
                                'process_wall_seconds': None}
        elif not isinstance(row['scheduler'].get('status'), str):
            row['scheduler'] = dict(row['scheduler'], status='invalid_schedule_status')
        observations.append(row)
    issues.extend(_compare_outputs(observations))
    result = summarize(cfg, observations)
    structural_requests = [q for row in observations if row.get('valid') and row['job']['method'] in ('H', 'S')
                           for q in (row.get('trace') or {}).get('requests', [])]
    result.update(schema_version=2, protocol_sha256=protocol_hash, input_manifest_sha256=manifest_hash,
                  stage=cfg['stage'], validation_issues=issues, correctness_accounting_pass=not issues,
                  structural_removal_observed=bool(structural_requests) and all(
                      q['memory']['catalog_materialized'] is False and
                      q['memory']['candidate_array_rows'] == q['memory']['full_root_caps_created_rows'] == 0 and
                      q['memory']['m_sized_structures'] == [] for q in structural_requests),
                  independent_replay_selection=manifest['independent_replay_selection'],
                  independent_full_catalog_proof='separate A2 checker report required; output agreement is not independent proof',
                  offline_analysis_sha256=sha(Path(__file__)),
                  confirmation_standard=cfg.get('confirmation_standard'))
    save(Path(output), result)
    print(json.dumps({key: result[key] for key in ('planned_workers', 'complete_workers', 'status_counts',
                     'correctness_accounting_pass', 'aggregate_comparisons', 'negative_cells', 'validation_issues')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    analyze(arguments.protocol, arguments.directory, arguments.output, arguments.manifest)
