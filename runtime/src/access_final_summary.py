"""Derive the final descriptive audit from the completed, frozen confirmation.

No timing, fixture generation, coefficient selection, or protocol mutation.
Run from the research root with ``python -m src.access_final_summary``.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import statistics


ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 ** 2
RESPONSE_RHS = ('candidate_rhs', 'winner_rhs', 'policy_refresh_rhs',
                'numerical_reanchor_rhs')
WORK_COUNTERS = ('state_rhs', *RESPONSE_RHS, 'response_blocks',
                 'gram_materializations', 'gram_rematerializations',
                 'gram_hits', 'transfers', 'endpoint_checked',
                 'endpoint_excluded', 'score_calls', 'cold_tail_ids')


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    directory = Path('raw/access_confirmation')
    schedule = read(directory / 'SCHEDULE_RESULTS.json')
    results = read('reports/ACCESS_CONFIRMATION_RESULTS.json')
    replay = read('reports/ACCESS_CONFIRMATION_REPLAY.json')
    assert schedule['status'] == 'completed'
    assert results['gate_pass'] and not results['validation_issues']
    assert replay['verification'] == 'passed'
    workers = [read(directory / j['result']) for j in schedule['jobs']]
    assert len(workers) == 108
    assert all(w['status'] == 'completed' and len(w['requests']) == 16
               for w in workers)
    keyed = {(w['graph'], w['stream'], w['seed'], w['repeat_id'], w['label']): w
             for w in workers}
    assert len(keyed) == len(workers)
    by_arm = {}
    for label in ('rebuild', 'retain', 'guard_0.125'):
        selected = [w for w in workers if w['label'] == label]
        totals = Counter()
        access_phases = Counter()
        index_phases = Counter()
        for w in selected:
            for q in w['requests']:
                totals.update({k: q['counters'][k] for k in WORK_COUNTERS})
                access_phases.update(q['discovery']['access_phases'])
                index_phases.update(q['index_build_phases'])
        service = sum(w['structural_seconds'] +
                      sum(q['request_seconds'] for q in w['requests'])
                      for w in selected)
        assert abs(service - results['total_scheduled_service_seconds'][label]) < 1e-8
        by_arm[label] = dict(
            streams=len(selected), requests=sum(len(w['requests']) for w in selected),
            service_seconds=service,
            scheduler_process_wall_seconds=sum(j['process_wall_seconds']
                for j in schedule['jobs'] if j['job']['label'] == label),
            worker_observed_wall_seconds=sum(w['worker_observed_wall_seconds']
                                            for w in selected),
            structural_seconds=sum(w['structural_seconds'] for w in selected),
            response_rhs=sum(totals[k] for k in RESPONSE_RHS),
            counters=dict(totals), access_phases_seconds=dict(access_phases),
            paid_index_phases_seconds=dict(index_phases),
            raw_visits=sum(e['pre_dedup_visits'] for w in selected
                for q in w['requests'] for e in q['discovery']['epochs']),
            process_peak_rss_bytes_max=max(w['process_peak_rss_bytes'] for w in selected),
        )

    compared_differences = {label: {k: dict(different_requests=0, total_delta=0,
                          min_delta=0, max_delta=0) for k in WORK_COUNTERS}
                          for label in ('retain', 'guard_0.125')}
    guard_requests = []
    for w in workers:
        if w['label'] == 'rebuild':
            continue
        b = keyed[w['graph'], w['stream'], w['seed'], w['repeat_id'], 'rebuild']
        for q, reference in zip(w['requests'], b['requests']):
            assert q['query_position'] == reference['query_position']
            if w['label'] == 'guard_0.125':
                guard_requests.append((w, q))
            for k, row in compared_differences[w['label']].items():
                delta = q['counters'][k] - reference['counters'][k]
                row['different_requests'] += int(delta != 0)
                row['total_delta'] += delta
                row['min_delta'] = min(row['min_delta'], delta)
                row['max_delta'] = max(row['max_delta'], delta)

    kinds = Counter()
    fallback_epochs = Counter()
    max_fraction = 0.0
    by_graph = {}
    for w, q in guard_requests:
        d = q['discovery']
        assert d['fallback_count'] <= 1 and d['old_consumed'] <= d['visit_budget']
        max_fraction = max(max_fraction, d['old_consumed'] / d['visit_budget'])
        rejects = [e for e in d['epochs'] if e['rejected_plan'] is not None]
        assert len(rejects) == d['fallback_count']
        if rejects:
            epoch = rejects[0]['epoch']
            kinds['immediate_fallback' if epoch == 1 else 'mid_request_fallback'] += 1
            fallback_epochs[epoch] += 1
        elif q['query_position'] == 1:
            kinds['initial_build'] += 1
        else:
            kinds['retained_whole_request'] += 1
    for graph in ('Alpha', 'OTC'):
        ws = [w for w in workers if w['graph'] == graph]
        m = ws[0]['catalog_size']
        paired_rss = []
        for w in ws:
            if w['label'] != 'guard_0.125':
                continue
            b = keyed[graph, w['stream'], w['seed'], w['repeat_id'], 'rebuild']
            paired_rss.append(w['process_peak_rss_bytes'] - b['process_peak_rss_bytes'])
        by_graph[graph] = dict(
            catalog_size=m, old_visit_budget=m // 8,
            fixed_membership_bytes=4*m, retained_root_caps_bytes=8*m,
            derived_two_sorted_index_payload_bytes=24*m,
            guard_minus_rebuild_peak_rss_mib=dict(
                median=statistics.median(paired_rss)/MIB,
                minimum=min(paired_rss)/MIB, maximum=max(paired_rss)/MIB),
            process_peak_rss_mib_by_arm={label: dict(
                median=statistics.median(w['process_peak_rss_bytes'] / MIB
                    for w in ws if w['label'] == label),
                maximum=max(w['process_peak_rss_bytes'] / MIB
                    for w in ws if w['label'] == label))
                for label in ('rebuild', 'retain', 'guard_0.125')},
        )
    repeat_totals = []
    for repeat in range(3):
        costs = {label: sum(w['service_accounted_seconds'] for w in workers
                           if w['repeat_id'] == repeat and w['label'] == label)
                 for label in ('rebuild', 'retain', 'guard_0.125')}
        repeat_totals.append(dict(repeat_id=repeat, costs=costs,
            guard_over_rebuild=costs['guard_0.125']/costs['rebuild']))
    seed_totals = []
    for seed in sorted({w['seed'] for w in workers}):
        costs = {label: sum(w['service_accounted_seconds'] for w in workers
                           if w['seed'] == seed and w['label'] == label)
                 for label in ('rebuild', 'retain', 'guard_0.125')}
        seed_totals.append(dict(seed=seed, costs=costs,
            guard_over_rebuild=costs['guard_0.125']/costs['rebuild']))
    independent_requests = [dict(timed_file=row['timed_file'],
        request=q['request'], behaviors=q['behaviors'],
        accepts=q['actual_timed_output_proof']['accepts'],
        checked_candidates=q['actual_timed_output_proof']['checked_candidates'])
        for row in replay['verified'] for q in row['requests']]
    summary = dict(
        input_hashes={str(p): digest(p) for p in (
            'protocol/ACCESS_CONFIRMATION.json',
            'reports/ACCESS_CONFIRMATION_RESULTS.json',
            'reports/ACCESS_CONFIRMATION_REPLAY.json',
            directory / 'SCHEDULE_RESULTS.json')},
        workers=len(workers), requests=sum(len(w['requests']) for w in workers),
        accepted_decisions=sum(q['commits'] for w in workers for q in w['requests']),
        batch_wall_seconds=schedule['batch_wall_seconds'], by_arm=by_arm,
        same_request_counter_differences_guard_minus_rebuild=compared_differences['guard_0.125'],
        same_request_counter_differences_retain_minus_rebuild=compared_differences['retain'],
        guard_behavior_counts=dict(kinds), guard_fallback_epochs=dict(fallback_epochs),
        guard_max_budget_fraction=max_fraction,
        by_graph=by_graph, repeat_totals=repeat_totals, seed_totals=seed_totals,
        independent_requests=independent_requests,
        independent_request_count=len(independent_requests),
        independent_decisions=sum(q['accepts'] for q in independent_requests),
        independent_candidate_checks=sum(q['checked_candidates'] for q in independent_requests),
        independent_missing_behaviors=replay['missing_behaviors'],
        independent_checker_seconds=replay['checker_seconds'],
        scope='Descriptive statistics from the one completed frozen confirmation. '
              'No independent-observation or wall-time-bound claim. RSS differences '
              'are process-peak observations, not isolated payload allocation effects. '
              'Nested phase totals must not be added to service totals.',
    )
    output = ROOT / 'analysis/FINAL_CONFIRMATION_AUDIT.json'
    output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), workers=summary['workers'],
        requests=summary['requests'], guard_behavior_counts=dict(kinds),
        independent_requests=len(independent_requests),
        independent_decisions=summary['independent_decisions']), indent=2))


if __name__ == '__main__':
    main()
