"""Frozen full-cost gates. No coefficient refitting or result deletion."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as st
import math
import hashlib

from .common import dump, file_hash


def result_rows(directory):
    rows=[]
    for path in sorted(directory.glob('*.json')):
        if path.name in ('FROZEN_JOBS.json','SCHEDULE_RESULTS.json'):
            continue
        try:
            obj=json.loads(path.read_text())
        except (ValueError,OSError) as error:
            obj={'requests':[], 'status':'invalid_result', 'parse_error':str(error)}
        if 'requests' in obj:
            rows.append((path,obj))
    return rows


def analyze(protocol_path,directory,output,manifest_path=None):
    from .access_bench import validate_protocol,jobs,catalog_visits
    cfg=json.loads(protocol_path.read_text())
    cfg=validate_protocol(cfg)
    rows=result_rows(directory)
    expected=len(cfg['graphs'])*len(cfg['streams'])*len(cfg['seeds'])*len(cfg['methods'])*cfg['repeat_count']
    issues=[]
    manifest=json.loads(manifest_path.read_text()) if manifest_path is not None else None
    if manifest is None:
        issues.append('input/source manifest is required for admission')
    elif manifest['protocol_sha256']!=file_hash(protocol_path):
        issues.append('manifest protocol identity mismatch')
    fixture_records={} if manifest is None else {(r['graph'],r['stream'],r['seed']):r for r in manifest['fixtures']}
    expected_jobs={(j['graph'],j['stream'],j['seed'],j['repeat_id'],j['label']):j for j in jobs(cfg)}
    if len(rows)!=expected: issues.append(f'{len(rows)} result files / {expected} planned workers')
    byjob={}
    invalid=set()
    for path,o in rows:
        key=None
        try:
            key=(o['graph'],o['stream'],o['seed'],o['repeat_id'],o['label'])
            if key in byjob: issues.append(f'duplicate job {key}')
            byjob[key]=o
            if type(o['process_peak_rss_bytes']) is not int or o['process_peak_rss_bytes']<0:
                raise ValueError('invalid peak RSS')
            job=expected_jobs.get(key)
            if job is None or any(o.get(k)!=v for k,v in job.items()):
                issues.append(f'{path.name}: unregistered method/job')
                invalid.add(key)
            expected_filename=f"{o['schedule_position']:03d}_{o['graph']}_{o['stream']}_{o['seed']}_{o['label']}.json"
            if path.name!=expected_filename:
                issues.append(f'{path.name}: filename/job binding mismatch')
                invalid.add(key)
            if o['status']!='completed' or len(o['requests'])!=cfg['requests']:
                issues.append(f'{path.name}: {o["status"]}, {len(o["requests"])} requests')
            if o['protocol_sha256']!=file_hash(protocol_path):
                issues.append(f'{path.name}: protocol identity mismatch')
            if manifest is not None:
                fixture=fixture_records[key[:3]]
                if (o['input_manifest_sha256']!=file_hash(manifest_path) or
                    o['source_files_sha256']!=manifest['source_files_sha256'] or
                    o['graph_hash']!=manifest['graphs'][o['graph']]['weighted_sha256'] or
                    o['fixture_id']!=fixture['fixture_id'] or o['fixture_npz_sha256']!=fixture['npz_sha256'] or
                    o['fixture_matrix_sha256']!=fixture['matrix_sha256']):
                    issues.append(f'{path.name}: fixture/source identity mismatch')
            if o['arm']=='budget-guard' and o['visit_budget']!=catalog_visits(o['coefficient'],o['catalog_size']):
                issues.append(f'{path.name}: coefficient/budget mismatch')
            cumulative=o['structural_seconds']
            if abs(sum(o['structural_phases'].values())-cumulative)>1e-7:
                issues.append(f'{path.name}: structural accounting')
            for position,r in enumerate(o['requests'],1):
                for required in ('gain_low','gain_high','initial_seed_ids','current_version_id','commits','decisions','counters','memory'):
                    if required not in r: raise KeyError(required)
                for required in ('response_blocks','candidate_rhs'):
                    if required not in r['counters']: raise KeyError(required)
                if r['query_position']!=position or (manifest is not None and r['opinion_hash']!=fixture['opinion_hashes'][position-1]):
                    issues.append(f'{path.name}: request/opinion order mismatch')
                if abs(sum(r['phases'].values())-r['request_seconds'])>1e-7:
                    issues.append(f'{path.name}: request accounting')
                if not math.isfinite(r['request_seconds']) or r['request_seconds']<0:
                    issues.append(f'{path.name}: invalid elapsed time')
                cumulative+=r['request_seconds']
                if position in cfg['prefixes']:
                    if str(position) not in o['prefixes'] or abs(cumulative-o['prefixes'][str(position)])>1e-7:
                        issues.append(f'{path.name}: prefix {position} does not equal setup plus all requests')
                        invalid.add(key)
                if r['status'] not in ('budget_stop','tolerance_stop'):
                    issues.append(f'{path.name}: request {r["query_position"]} {r["status"]}')
                parent=o['root_version_id']
                if r['root_version_id']!=parent: issues.append(f'{path.name}: wrong request root')
                committed=set()
                for decision in r['decisions']:
                    if decision['parent_version_id']!=parent: issues.append(f'{path.name}: broken version chain')
                    if decision['status']=='accepted':
                        winner=decision['winner_id']
                        if winner in committed: issues.append(f'{path.name}: repeated commit')
                        committed.add(winner)
                        parent=hashlib.sha256(f'{parent}/{winner}'.encode()).hexdigest()
                    if decision['current_version_id']!=parent: issues.append(f'{path.name}: wrong child version')
                if parent!=r['current_version_id']: issues.append(f'{path.name}: final version mismatch')
                d=r['discovery']
                if o['arm']=='budget-guard':
                    if not (0<=d['old_consumed']<=o['visit_budget'] and d['fallback_count']<=1):
                        issues.append(f'{path.name}: budget violation')
                    old=sum(e['pre_dedup_visits'] for e in d['epochs'] if e['action']=='retain')
                    if old!=d['old_consumed']: issues.append(f'{path.name}: consumed does not reconcile')
                    if sum(e['action']=='fallback' for e in d['epochs'])!=d['fallback_count']:
                        issues.append(f'{path.name}: fallback counter mismatch')
                    if any(e['pre_dedup_visits']!=e['posting_raw_visits']+e['tail_raw_visits'] for e in d['epochs']):
                        issues.append(f'{path.name}: raw work split mismatch')
        except (KeyError,TypeError,ValueError,IndexError) as error:
            issues.append(f'{path.name}: missing {error}')
            if key is not None: invalid.add(key)
    if set(byjob)!=set(expected_jobs):
        issues.append('observed job keys differ from the frozen full schedule')
    from .access_validate import validate_evidence
    issues.extend(validate_evidence(cfg,manifest,directory,byjob,
                                  file_hash(manifest_path) if manifest_path is not None else None))
    control=next(m['label'] for m in cfg['methods'] if m['arm']=='rebuild-always')
    for key,o in byjob.items():
        base=byjob.get(key[:-1]+(control,))
        if base is None:
            issues.append(f'no matching baseline {key}')
            continue
        if key in invalid or key[:-1]+(control,) in invalid: continue
        for r,b in zip(o['requests'],base['requests']):
            fields=('status','initial_seed_ids','current_version_id','commits')
            if any(r.get(f)!=b.get(f) for f in fields) or [x.get('winner_id') for x in r.get('decisions',[])] != [x.get('winner_id') for x in b.get('decisions',[])]:
                issues.append(f'{key}: output mismatch q{r["query_position"]}')
            if len(r['decisions'])!=len(b['decisions']):
                issues.append(f'{key}: decision count mismatch')
            for d,reference in zip(r['decisions'],b['decisions']):
                if any(d.get(f)!=reference.get(f) for f in ('status','epoch','parent_version_id','current_version_id')):
                    issues.append(f'{key}: per-decision status/version mismatch')
                if d.get('winner_id') is not None:
                    if any(x.get(f) is None for x in (d,reference) for f in ('low','high')):
                        issues.append(f'{key}: missing decision gain enclosure')
                    elif max(d['low'],reference['low'])>min(d['high'],reference['high']):
                        issues.append(f'{key}: per-decision gain boxes do not overlap')
            if any(f not in x for x in (r,b) for f in ('gain_low','gain_high')):
                issues.append(f'{key}: missing gain enclosure')
                invalid.add(key)
            elif max(r['gain_low'],b['gain_low'])>min(r['gain_high'],b['gain_high']):
                issues.append(f'{key}: gain boxes do not overlap q{r["query_position"]}')
    groups=defaultdict(list)
    for key,o in byjob.items():
        if key in invalid: continue
        groups[key[:3]+(key[-1],)].append(o)
    cells=[]
    for (graph,stream,seed,label),samples in sorted(groups.items()):
        base=groups.get((graph,stream,seed,control),[])
        complete=[o for o in samples if o['status']=='completed' and len(o['requests'])==cfg['requests'] and
                  all(str(q) in o.get('prefixes',{}) for q in cfg['prefixes'])]
        if len(complete)!=cfg['repeat_count']: continue
        ratios=[]
        for sample in complete:
            b=next((o for o in base if o['repeat_id']==sample['repeat_id']),None)
            if b is not None and '16' in b.get('prefixes',{}):
                ratios.append(sample['prefixes']['16']/b['prefixes']['16'])
        if len(ratios)!=cfg['repeat_count']: continue
        totals=[o['prefixes']['16'] for o in complete]
        cells.append(dict(graph=graph,stream=stream,seed=seed,label=label,
            cost_seconds_median=st.median(totals),cost_seconds_all=totals,
            paired_ratio_median=st.median(ratios),paired_ratios=ratios,
            prefixes_seconds_median={str(q):st.median(o['prefixes'][str(q)] for o in complete) for q in cfg['prefixes']},
            peak_rss_bytes_median=st.median(o['process_peak_rss_bytes'] for o in complete),
            response_blocks_median=st.median(sum(r['counters']['response_blocks'] for r in o['requests']) for o in complete),
            candidate_rhs_median=st.median(sum(r['counters']['candidate_rhs'] for r in o['requests']) for o in complete),
            raw_visits_median=st.median(sum(e['pre_dedup_visits'] for r in o['requests'] for e in r['discovery']['epochs']) for o in complete),
            fallbacks_median=st.median(sum(r['discovery'].get('fallback_count',0) for r in o['requests']) for o in complete)))
    totals={}
    observed={}
    for m in cfg['methods']:
        samples=[o for _,o in rows if o.get('label')==m['label']]
        observed[m['label']]=sum(o.get('structural_seconds',0)+sum(r.get('request_seconds',0) for r in o['requests']) for o in samples)
        totals[m['label']]=(sum(o['prefixes']['16'] for o in samples)
            if len(samples)==expected//len(cfg['methods']) and all(o.get('status')=='completed' and '16' in o.get('prefixes',{}) for o in samples)
            and not any(k in invalid and k[-1]==m['label'] for k in byjob) else None)
    base_total=totals[control]
    criteria=cfg.get('acceptance',dict(maximum_cell_ratio=1.10,maximum_slow_ratio=.95,minimum_net_saving=.02))
    candidates=[]
    for m in cfg['methods']:
        if m['arm']!='budget-guard': continue
        label=m['label']
        selected=[c for c in cells if c['label']==label]
        worst=max((c['paired_ratio_median'] for c in selected),default=None)
        worst_slow=max((c['paired_ratio_median'] for c in selected if c['stream']=='slow'),default=None)
        ratio=totals[label]/base_total if base_total and totals[label] is not None else None
        eligible=(not issues and ratio is not None and worst is not None and worst_slow is not None and len(selected)==len(cfg['graphs'])*len(cfg['streams'])*len(cfg['seeds']) and
            worst<=criteria['maximum_cell_ratio'] and worst_slow<=criteria['maximum_slow_ratio'] and
            ratio<=1-criteria['minimum_net_saving'])
        candidates.append(dict(label=label,coefficient=m['coefficient'],aggregate_ratio=ratio,
            worst_cell_ratio=worst,worst_slow_ratio=worst_slow,eligible=eligible))
    feasible=[c for c in candidates if c['eligible']]
    selected=min(feasible,key=lambda c:(c['aggregate_ratio'],c['coefficient'])) if feasible else None
    noises=[(max(c['cost_seconds_all'])-min(c['cost_seconds_all']))/c['cost_seconds_median'] for c in cells if c['label']==control]
    out=dict(protocol_sha256=file_hash(protocol_path),expected_workers=expected,observed_workers=len(rows),
        validation_issues=issues,correctness_accounting_pass=not issues,cells=cells,
        total_scheduled_service_seconds=totals,observed_service_seconds_including_partial=observed,
        candidates=candidates,selected=selected,
        gate_pass=selected is not None,acceptance=criteria,
        baseline_median_repeat_relative_range=st.median(noises) if noises else None,
        note='Aggregate is sum of all complete scheduled P+sum(request) costs, equally many streams/arms; ratio is not the median of unrelated prefixes. Any incomplete worker prevents admission; partial costs remain in raw artifacts.')
    dump(output,out)
    print(json.dumps({k:out[k] for k in ('observed_workers','validation_issues','candidates','selected','gate_pass','baseline_median_repeat_relative_range')},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    a=ap.parse_args()
    analyze(a.protocol,a.directory,a.output,a.manifest)
