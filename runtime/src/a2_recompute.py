"""Portable, stdlib-only recomputation of archived A2 cost/work evidence.

This reads JSON and optionally hashes archived bytes. It imports no experiment
code, opens no numerical arrays, runs no solver, and never checks the host's
platform/packages against the archived environment. It is not a gain proof.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from statistics import median


SUCCESS = {'budget_stop', 'tolerance_stop'}
RESOURCE = {'timeout', 'rss_cap', 'memory_error', 'batch_cap', 'batch_cap_not_started'}
METHODS = [('D', 'D-all'), ('S', 'Stream-all'), ('H', 'Implicit-all')]
NON_WORK = {'epoch', 'tau', 'remaining_upper_coarse', 'cold_start_lower_fine', 'cold_end_lower_fine'}
TOL = 1e-7


def need(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def integer(value):
    return type(value) is int and value >= 0


def objects(value):
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def read(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    need(isinstance(value, dict), f'{path}: expected JSON object')
    return value, hashlib.sha256(raw).hexdigest()


def phases(values, seconds):
    need(number(seconds) and isinstance(values, dict), 'wall/phase types')
    need(all(type(v) in (int, float) and math.isfinite(v) and v >= -1e-8 for v in values.values())
         and abs(sum(values.values()) - seconds) <= TOL, 'wall/phase reconciliation')


def roster(cfg):
    labels = {'development': ['development'], 'confirmation': ['confirmation-1', 'confirmation-2']}
    need(cfg['stage'] in labels and cfg['seed_labels'] == labels[cfg['stage']], 'registered seed labels')
    need(cfg['date'] == '2026-11-25' and cfg['graphs'] == ['Alpha', 'OTC'] and
         cfg['streams'] == ['slow', 'independent', 'signed-neighbor'], 'registered cells/date')
    need(cfg['requests'] == 4 and cfg['budget'] == 15 and cfg['repeat_count'] == 3 and
         cfg['delta'] == 0 and cfg['hot_size'] == 64 and cfg['prefixes'] == [1, 4], 'registered service')
    need(cfg['methods'] == [dict(method=m, label=l) for m, l in METHODS], 'registered methods')
    need(cfg['cpu'] == 0 and cfg['supervisor_cpu'] == 8 and cfg['numerical_threads'] == 1 and
         cfg['limits']['worker_seconds'] == 180 and cfg['limits']['rss_mib'] == 2048 and
         cfg['limits']['batch_seconds'] == (2700 if cfg['stage'] == 'development' else 3600), 'registered resources')
    result = []
    for repeat in range(3):
        for position, (graph, stream, label) in enumerate(product(cfg['graphs'], cfg['streams'], cfg['seed_labels'])):
            seed = int.from_bytes(hashlib.sha256(f'A2|{label}|2026-11-25|{graph}|{stream}'.encode()).digest()[:8], 'big')
            pivot = (repeat + position) % 3
            for method, name in METHODS[pivot:] + METHODS[:pivot]:
                result.append(dict(graph=graph, stream=stream, seed_label=label, seed=seed,
                                   repeat_id=repeat, method=method, label=name))
    return result


def filename(position, job):
    return f"{position:03d}_{job['graph']}_{job['stream']}_{job['seed_label']}_{job['label']}.json"


def measurements(trace, scheduled):
    requests = objects(trace.get('requests'))
    counts, work, phase_totals = Counter(), Counter(), Counter()
    edits = Counter(insertions=0, flips=0)
    for request in requests:
        counts.update({k: v for k, v in request.get('counters', {}).items() if integer(v)})
        phase_totals.update({k: v for k, v in request.get('phases', {}).items() if number(v)})
        access = request.get('access', {})
        for event in objects(access.get('epochs')):
            work.update({k: v for k, v in event.items() if k not in NON_WORK and integer(v)})
        for decision in objects(request.get('decisions')):
            if decision.get('status') == 'accepted' and decision.get('flip') in (0, 1):
                edits['flips' if decision['flip'] else 'insertions'] += 1
    rss = [v for v in (trace.get('process_peak_rss_bytes'), scheduled.get('external_sampled_peak_rss_bytes')) if integer(v)]
    structural = trace.get('structural_seconds')
    observed = (structural if number(structural) else 0) + sum(r['request_seconds'] for r in requests if number(r.get('request_seconds')))
    need(number(observed), 'nonfinite cumulative observed wall')
    statuses = Counter(r.get('status', 'missing_status') for r in requests)
    resource = scheduled.get('status') in RESOURCE or trace.get('status') == 'memory_error' or statuses['memory_error'] > 0
    return dict(returned_requests=len(requests), request_status_counts=dict(statuses),
                resource_failure=resource, unresolved_requests=statuses['unresolved'],
                accepted_edits=dict(edits), request_counters=dict(counts), epoch_work=dict(work),
                request_phases_seconds=dict(phase_totals), observed_service_seconds=observed,
                structural_seconds=structural if number(structural) else None,
                structural_phases_seconds={k: v for k, v in trace.get('structural_phases', {}).items() if number(v)},
                process_wall_seconds=scheduled.get('process_wall_seconds') if number(scheduled.get('process_wall_seconds')) else None,
                peak_rss_bytes=max(rss, default=None))


def validate_worker(trace, scheduled, job, position, cfg, manifest, manifest_hash, fixture):
    need(all(trace.get(k) == v for k, v in job.items()), 'worker job binding')
    need(trace['schedule_position'] == position and trace['status'] == scheduled['status'], 'position/status binding')
    for name, expected in (('protocol_id', cfg['id']), ('protocol_sha256', manifest['protocol_sha256']),
                           ('input_manifest_sha256', manifest_hash), ('requests_planned', 4), ('budget', 15),
                           ('delta', 0), ('hot_size', 64), ('domain', cfg['domain'])):
        need(trace.get(name) == expected, 'worker binding: ' + name)
    for name in ('source_files_sha256', 'input_source_files_sha256', 'environment'):
        need(trace.get(name) == manifest[name], 'archived provenance binding: ' + name)
    requests = trace['requests']
    need(isinstance(requests, list) and len(requests) <= 4 and all(isinstance(r, dict) for r in requests), 'request list')
    need(scheduled['completed_requests'] == len(requests), 'scheduler request count')
    if 'structural_seconds' not in trace:
        need(not requests and not trace.get('prefixes') and scheduled['status'] != 'completed', 'requests/completion before structural setup')
        return {}, []
    graph = manifest['graphs'][job['graph']]
    n = graph['n']
    need(type(n) is int and 1 <= n <= 10000, 'registered vertex range')
    m = n * (n - 1) // 2 - graph['root_positive_edges']
    need(trace['n'] == n and trace['catalog_size'] == graph['catalog_size'] == m, 'whole-catalog denominator')
    need(fixture['seed'] == job['seed'] and fixture['shape'] == [4, n] and len(fixture['opinion_hashes']) == 4, 'fixture seed/shape binding')
    need(trace['graph_hash'] == graph['weighted_sha256'] and trace['fixture_id'] == fixture['fixture_id'] and
         trace['fixture_npz_sha256'] == fixture['npz_sha256'] and trace['fixture_matrix_sha256'] == fixture['matrix_sha256'], 'graph/fixture binding')
    phases(trace['structural_phases'], trace['structural_seconds'])
    cumulative, prefixes, signatures = trace['structural_seconds'], {}, []
    for q, request in enumerate(requests, 1):
        need(request['query_position'] == q and request['opinion_hash'] == fixture['opinion_hashes'][q - 1], 'opinion/order binding')
        need(request['method'] == job['method'] and request['stream_id'] == trace['stream_id'] and
             request['root_version_id'] == trace['root_version_id'], 'method/stream/root reset')
        need(request['budget'] == 15 and request['delta'] == 0 and request['initial_gram_count'] == 0 and
             request['gain_denominator'] == 2**96, 'request contract')
        phases(request['phases'], request['request_seconds'])
        phases(request['online_phases'], request['online_seconds'])
        need(number(request['reset_opinion_seconds']) and abs(request['reset_opinion_seconds'] + request['online_seconds'] - request['request_seconds']) <= TOL, 'online/reset split')
        parent, chosen, accepted, low, high = trace['root_version_id'], set(), [], 0, 0
        decisions = request['decisions']
        need(isinstance(decisions, list) and len(decisions) <= 15, 'decision count')
        for epoch, decision in enumerate(decisions, 1):
            need(decision['epoch'] == epoch and decision['root_version_id'] == trace['root_version_id'] and
                 decision['parent_version_id'] == parent, 'decision parent/epoch')
            if decision['status'] == 'accepted':
                e = decision['winner_id']
                need(integer(e) and e not in chosen and 0 <= e // n < e % n < n, 'canonical/nonrepeated winner')
                need((decision['i'], decision['j']) == divmod(e, n) and type(decision['flip']) is int and decision['flip'] in (0, 1), 'accepted endpoints/kind')
                need(type(decision['low']) is int and type(decision['high']) is int and 0 < decision['low'] <= decision['high'], 'gain interval shape')
                child = hashlib.sha256(f'{parent}/{e}'.encode()).hexdigest()
                accepted.append((e, parent, child, decision['low'], decision['high']))
                chosen.add(e); parent = child
                low += decision['low']; high += decision['high']
            else:
                need(epoch == len(decisions) and decision['status'] == request['status'] and
                     decision['status'] in {'tolerance_stop', 'unresolved', 'memory_error', 'error'} and decision['winner_id'] is None, 'terminal decision')
            need(decision['current_version_id'] == parent, 'computed child version')
        need(request['commits'] == len(chosen) and request['current_version_id'] == parent and
             (request['gain_low'], request['gain_high']) == (low, high), 'request totals/version')
        if request['status'] == 'budget_stop':
            need(len(accepted) == len(decisions) == 15, 'partial budget stop')
        else:
            need(bool(decisions) and decisions[-1]['status'] == request['status'] != 'accepted', 'missing terminal decision')
        signatures.append(dict(query=q, status=request['status'], accepted=accepted, final=parent))
        cumulative += request['request_seconds']
        if q in (1, 4):
            prefixes[str(q)] = cumulative
    need(set(trace['prefixes']) == set(prefixes) and all(number(trace['prefixes'][k]) and abs(trace['prefixes'][k] - v) <= TOL for k, v in prefixes.items()), 'recomputed Q1/Q4 prefixes')
    need(number(trace['service_accounted_seconds']) and abs(trace['service_accounted_seconds'] - cumulative) <= TOL, 'recomputed service total')
    if scheduled['status'] == 'completed':
        need(scheduled['returncode'] == 0 and len(requests) == 4 and all(r['status'] in SUCCESS for r in requests), 'completed eligibility')
        need(trace['affinity'] == [cfg['cpu']] and trace['threadpools'] and all(p['num_threads'] == 1 for p in trace['threadpools']), 'archived CPU/thread evidence')
        peaks = (trace['process_peak_rss_bytes'], scheduled['external_sampled_peak_rss_bytes'])
        need(all(integer(p) for p in peaks) and max(peaks) <= cfg['limits']['rss_mib'] * 1024**2, 'completed RSS cap')
        need(number(scheduled['process_wall_seconds']) and scheduled['process_wall_seconds'] <= cfg['limits']['worker_seconds'] + 2, 'completed process wall')
    return prefixes, signatures


def ratio_summary(pairs):
    ratios = [a / b if a is not None and b is not None and b > 0 else None for a, b in pairs]
    finite = [v for v in ratios if v is not None]
    return dict(planned_pairs=len(pairs), paired_count=len(finite), ratios=ratios,
                minimum=min(finite, default=None), median=median(finite) if finite else None,
                maximum=max(finite, default=None), full_pair_set_ratio=(sum(a for a, _ in pairs) / sum(b for _, b in pairs)) if len(finite) == len(pairs) else None)


def recompute(protocol, manifest_path, directory, *, check_files=False):
    cfg, protocol_hash = read(protocol)
    manifest, manifest_hash = read(manifest_path)
    frozen, frozen_hash = read(directory / 'FROZEN_JOBS.json')
    scheduler, scheduler_hash = read(directory / 'SCHEDULE_RESULTS.json')
    jobs = roster(cfg)
    global_issues, workers, groups = [], [], defaultdict(list)
    bindings = dict(protocol_sha256=protocol_hash, input_manifest_sha256=manifest_hash,
                    source_files_sha256=manifest['source_files_sha256'], input_source_files_sha256=manifest['input_source_files_sha256'], environment=manifest['environment'])
    if (manifest.get('verified') is not True or manifest.get('schema_version') != 2 or
            manifest.get('protocol_sha256') != protocol_hash or manifest.get('protocol_id') != cfg['id'] or
            manifest.get('stage') != cfg['stage'] or manifest.get('domain') != cfg['domain']):
        global_issues.append('manifest/protocol binding')
    if frozen.get('schema_version') != 2 or frozen.get('jobs') != jobs or any(frozen.get(k) != v for k, v in bindings.items()):
        global_issues.append('frozen complete roster/provenance binding')
    scheduled_rows = objects(scheduler.get('jobs'))
    if len(scheduled_rows) != len(jobs) or scheduler.get('status') not in {'completed', 'completed_with_failures'}:
        global_issues.append('scheduler missing jobs or still incomplete')
    expected_status = 'completed' if all(r.get('status') == 'completed' for r in scheduled_rows) else 'completed_with_failures'
    if scheduler.get('schema_version') != 2 or scheduler.get('status') != expected_status:
        global_issues.append('scheduler final status')
    for document in (frozen, scheduler):
        if document.get('limits') != cfg['limits'] or document.get('supervisor_affinity') != [cfg['supervisor_cpu']]:
            global_issues.append('archived supervisor/resource policy binding')
    fixtures = {(f['graph'], f['stream'], f['seed_label']): f for f in manifest['fixtures']}
    expected_fixtures = set(product(cfg['graphs'], cfg['streams'], cfg['seed_labels']))
    if set(fixtures) != expected_fixtures or len(manifest['fixtures']) != len(expected_fixtures):
        global_issues.append('fixture grid binding')
    if check_files:
        root = protocol.resolve().parent.parent
        files = dict(manifest['source_files_sha256']) | dict(manifest['input_source_files_sha256'])
        paths = [(root / name, expected) for name, expected in files.items()]
        paths += [(root / g['weighted_path'], g['weighted_sha256']) for g in manifest['graphs'].values()]
        paths += [(manifest_path.parent / f['file'], f['npz_sha256']) for f in manifest['fixtures']]
        for path, expected in paths:
            try:
                with path.open('rb') as handle:
                    actual = hashlib.file_digest(handle, 'sha256').hexdigest()
                need(actual == expected, 'byte checksum mismatch')
            except (OSError, ValueError) as error:
                global_issues.append(f'{path}: {error}')
    for position, job in enumerate(jobs):
        name = filename(position, job)
        scheduled = scheduled_rows[position] if position < len(scheduled_rows) else {}
        row = dict(position=position, job=job, result=name, status=scheduled.get('status', 'missing_result'), valid=False, issues=[], Q1_seconds=None, Q4_seconds=None)
        trace = {}
        try:
            need(scheduled.get('position') == position and scheduled.get('job') == job and scheduled.get('result') == name, 'schedule/job/result binding')
            trace, row['raw_sha256'] = read(directory / name)
            prefixes, signatures = validate_worker(trace, scheduled, job, position, cfg, manifest, manifest_hash, fixtures[job['graph'], job['stream'], job['seed_label']])
            row['valid'] = True
            requests = objects(trace.get('requests'))
            if prefixes.get('1') and requests[0]['status'] in SUCCESS:
                row['Q1_seconds'] = prefixes['1']
            if scheduled['status'] == 'completed' and len(requests) == 4 and all(r['status'] in SUCCESS for r in requests):
                row['Q4_seconds'] = prefixes['4']
            for signature in signatures:
                groups[(job['graph'], job['stream'], job['seed_label'], signature['query'])].append(signature)
            row['accepted_version_signatures'] = [dict(query=s['query'], status=s['status'],
                commits=len(s['accepted']), final_version_id=s['final'], accepted_ids=[d[0] for d in s['accepted']],
                sha256=hashlib.sha256(json.dumps([d[:3] for d in s['accepted']], separators=(',', ':')).encode()).hexdigest())
                for s in signatures]
        except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
            row['issues'].append(str(error))
        try:
            row.update(measurements(trace, scheduled))
        except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
            row['valid'] = False
            row['issues'].append('malformed work/measurement object')
        workers.append(row)
    for key, records in groups.items():
        accepted_by_epoch, successes = {}, set()
        for record in records:
            if record['status'] in SUCCESS:
                successes.add((record['status'], record['final'], len(record['accepted'])))
            for epoch, decision in enumerate(record['accepted'], 1):
                if epoch in accepted_by_epoch:
                    prior = accepted_by_epoch[epoch]
                    if decision[:3] != prior[:3] or max(decision[3], prior[3]) > min(decision[4], prior[4]):
                        global_issues.append(f'{key}/epoch{epoch}: certified accepted-prefix disagreement')
                    decision = (*decision[:3], max(decision[3], prior[3]), min(decision[4], prior[4]))
                accepted_by_epoch[epoch] = decision
        if len(successes) > 1 or any(status == 'tolerance_stop' and len(accepted_by_epoch) > count for status, _, count in successes):
            global_issues.append(f'{key}: completed termination/prefix disagreement')
    for row in workers:
        row['complete_eligible'] = row['valid'] and not global_issues and row['Q4_seconds'] is not None
        if not row['complete_eligible']:
            row['Q4_seconds'] = None
    totals, cells = {}, []
    comparisons = [('H', 'D'), ('H', 'S'), ('D', 'S')]
    aggregate_pairs = defaultdict(list)
    for method, _ in METHODS:
        selected = [r for r in workers if r['job']['method'] == method]
        complete = [r for r in selected if r['complete_eligible']]
        totals[method] = dict(planned_workers=len(selected), complete_workers=len(complete),
            failure_or_incomplete_workers=len(selected) - len(complete), planned_requests=4 * len(selected),
            returned_requests=sum(r.get('returned_requests', 0) for r in selected),
            worker_status_counts=dict(Counter(r['status'] for r in selected)),
            resource_failure_workers=sum(bool(r.get('resource_failure')) for r in selected),
            unresolved_requests=sum(r.get('unresolved_requests', 0) for r in selected),
            unresolved_workers=sum(r.get('unresolved_requests', 0) > 0 for r in selected),
            full_Q4_seconds=sum(r['Q4_seconds'] for r in complete) if len(complete) == len(selected) else None,
            full_process_seconds=sum(r['process_wall_seconds'] for r in complete) if len(complete) == len(selected) else None,
            observed_structural_seconds=sum(r['structural_seconds'] for r in selected if number(r.get('structural_seconds'))),
            observed_service_seconds=sum(r.get('observed_service_seconds', 0) for r in selected),
            observed_process_seconds=sum(r['process_wall_seconds'] for r in selected if number(r.get('process_wall_seconds'))),
            peak_rss_bytes=max((r['peak_rss_bytes'] for r in selected if integer(r.get('peak_rss_bytes'))), default=None))
        for field in ('accepted_edits', 'request_counters', 'epoch_work', 'request_status_counts', 'request_phases_seconds', 'structural_phases_seconds'):
            combined = Counter()
            for row in selected:
                combined.update(row.get(field, {}))
            totals[method][field] = dict(combined)
    lookup = {(r['job']['graph'], r['job']['stream'], r['job']['seed_label'], r['job']['repeat_id'], r['job']['method']): r for r in workers}
    for graph, stream, label in product(cfg['graphs'], cfg['streams'], cfg['seed_labels']):
        costs = [{m: lookup[graph, stream, label, repeat, m]['Q4_seconds'] for m, _ in METHODS} for repeat in range(3)]
        cell = dict(graph=graph, stream=stream, seed_label=label, Q4_seconds_by_repeat=costs, comparisons={})
        for numerator, denominator in comparisons:
            key = f'{numerator}_over_{denominator}'
            pairs = [(r[numerator], r[denominator]) for r in costs]
            cell['comparisons'][key] = ratio_summary(pairs)
            aggregate_pairs[key].extend(pairs)
        cells.append(cell)
    return dict(schema_version=1, purpose='independent portable JSON cost/work recomputation; no numerical gain proof',
        protocol_sha256=protocol_hash, manifest_sha256=manifest_hash, frozen_jobs_sha256=frozen_hash,
        schedule_results_sha256=scheduler_hash, recompute_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        archived_files_byte_checked=check_files, host_environment_checked=False,
        planned_workers=len(jobs), valid_workers=sum(r['valid'] for r in workers),
        integrity_pass=not global_issues and all(r['valid'] for r in workers), global_issues=global_issues,
        methods=totals, cells=cells, aggregate_comparisons={k: ratio_summary(v) for k, v in aggregate_pairs.items()},
        workers=workers, scope='All registered workers retained. Work and accepted edit totals include returned partial requests and matched repeats. Complete Q4 ratios exclude incomplete/invalid workers, retain planned pair counts, and never substitute infinity for a failed reference. Array contents, gains, root edge legality and omitted-candidate coverage require the separate numerical checker.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'manifest', 'directory', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--check-files', action='store_true', help='Hash only manifest-listed runtime/input files; retain the archive directory layout.')
    args = parser.parse_args()
    need(not args.output.exists(), 'refusing to replace an existing recomputation report')
    result = recompute(args.protocol, args.manifest, args.directory, check_files=args.check_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as destination:
        json.dump(result, destination, indent=2, allow_nan=False)
        destination.write('\n')
    print(json.dumps({k: result[k] for k in ('planned_workers', 'valid_workers', 'integrity_pass', 'global_issues')}))
    return 0 if result['integrity_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
