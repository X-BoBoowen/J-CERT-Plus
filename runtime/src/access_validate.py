"""Offline consistency checks on every frozen scheduling and budget event record."""
import json
import math


def validate_evidence(cfg, manifest, directory, byjob, manifest_sha256=None):
    from .access_bench import jobs
    errors=[]

    def need(condition, message):
        if not condition: raise ValueError(message)

    schedule=list(jobs(cfg))
    try:
        frozen=json.loads((directory/'FROZEN_JOBS.json').read_text())
        observed=json.loads((directory/'SCHEDULE_RESULTS.json').read_text())
        need(frozen['jobs']==schedule,'frozen job registry')
        need(frozen['cpu']==cfg['cpu'] and frozen['limits']==cfg['limits'],'frozen resource limits')
        if manifest is not None:
            need(frozen['source_files_sha256']==manifest['source_files_sha256'] and
                 frozen['protocol_sha256']==manifest['protocol_sha256'],'schedule provenance')
            need(frozen['input_manifest_sha256']==manifest_sha256,'schedule input manifest digest')
        need(observed['status']=='completed','scheduler did not complete all workers')
        need(observed['limits']==cfg['limits'],'scheduler resource limits')
        need(len(observed['jobs'])==len(schedule),'scheduler complete denominator')
        need(observed['batch_wall_seconds']<=cfg['limits']['batch_seconds']+1.,'batch elapsed cap')
    except (OSError,KeyError,TypeError,ValueError) as error:
        errors.append(f'schedule: {error}')
        observed=None

    for position,job in enumerate(schedule):
        label=f'job {position} {job}'
        try:
            key=tuple(job[k] for k in ('graph','stream','seed','repeat_id','label'))
            trace=byjob[key]
            need(trace['schedule_position']==position,'worker schedule position')
            need(trace['cpu']==cfg['cpu'] and trace['affinity']==[cfg['cpu']],'worker affinity')
            need(trace['threadpools'] and all(p['num_threads']==1 for p in trace['threadpools']),
                 'numerical thread count')
            need(trace['process_peak_rss_bytes']<=cfg['limits']['rss_mib']*1024**2,'worker RSS cap')
            if observed is not None:
                row=observed['jobs'][position]
                expected_file=f"{position:03d}_{job['graph']}_{job['stream']}_{job['seed']}_{job['label']}.json"
                need(row['position']==position and row['job']==job and row['result']==expected_file,'scheduled file mapping')
                need(row['status']=='completed' and row['returncode']==0 and
                     row['completed_requests']==cfg['requests'],'completed worker scheduler status')
                need(row['external_sampled_peak_rss_bytes']<=cfg['limits']['rss_mib']*1024**2,'sampled RSS cap')
                need(0<=row['process_wall_seconds']<=cfg['limits']['worker_seconds']+.25,'worker elapsed cap')
            hashes={i+1:r['opinion_hash'] for i,r in enumerate(trace['requests'])}
            carried=None
            for q,r in enumerate(trace['requests'],1):
                history=r['discovery']
                need(history['initial_anchor_request']==carried,'request inherited source')
                need(history['initial_anchor_opinion_hash']==(hashes[carried] if carried else None),
                     'inherited source opinion')
                active=q if trace['arm']=='rebuild-always' or carried is None else carried
                consumed=switches=0
                events=history['epochs']
                need(len(events)==len(r['decisions']),'one logged cold event per decision')
                guard=trace['arm']=='budget-guard'
                limit=trace['visit_budget'] if guard else None
                need(history['visit_budget']==limit,'request visit budget')
                for t,(event,decision) in enumerate(zip(events,r['decisions'],strict=True),1):
                    need(event['epoch']==t and event['parent_version_id']==decision['parent_version_id'],
                         'current decision epoch/version')
                    need(event['target']>=0 and event['tau']==decision['tau'],'current threshold/cutoff')
                    need(event['budget']==limit,'epoch visit budget')
                    before=consumed if guard else None
                    need(event['consumed_before']==before,'consumption before gather')
                    rejected=event['rejected_plan']
                    if rejected is not None:
                        need(guard and active<q and switches==0,'single old-source fallback')
                        need(rejected['anchor_request']==active and rejected['anchor_opinion_hash']==hashes[active]
                             and rejected['target']==event['target'],'rejected plan source/target')
                        check_plan(rejected)
                        need(consumed+rejected['pre_dedup_visits']>limit,'denied plan must exceed remaining budget')
                        need(event['action']=='fallback' and event['fallback_reason']=='visit_budget_exceeded',
                             'fallback event label')
                        switches+=1
                        active=q
                    else:
                        need(event['fallback_reason'] is None,'unexpected fallback reason')
                        need(event['action']==('fresh' if active==q else 'retain'),'source action label')
                    need(event['anchor_request']==active and event['anchor_opinion_hash']==hashes[active],
                         'gather source request/opinion')
                    check_plan(event)
                    if guard and active<q:
                        need(consumed+event['pre_dedup_visits']<=limit,'old admission exceeded budget')
                        consumed+=event['pre_dedup_visits']
                    need(event['consumed']==(consumed if guard else None),'consumption after gather')
                    need(0<=event['returned_ids']<=event['unique_ids_before_exclusion']<=event['pre_dedup_visits'],
                         'raw/unique/returned count ordering')
                    need(event['returned_ids']==decision['cold_endpoint_visits'],'cold screening count')
                need(history['anchor_request']==active and history['anchor_opinion_hash']==hashes[active],
                     'final published source')
                need(history['old_consumed']==(consumed if guard else None) and
                     history['fallback_count']==switches,'request budget/switch totals')
                carried=active
        except (KeyError,TypeError,ValueError,IndexError) as error:
            errors.append(f'{label}: {error}')
    return errors


def check_plan(event):
    raw,posting,tail=(event[k] for k in ('pre_dedup_visits','posting_raw_visits','tail_raw_visits'))
    if any(type(v) is not int or v<0 for v in (raw,posting,tail)):
        raise ValueError('nonnegative integer raw work')
    if raw!=posting+tail or sum(event['group_tails'])!=tail:
        raise ValueError('posting/group tail counts do not reconcile')
    if any(type(v) is not int or v<0 for v in event['group_tails']):
        raise ValueError('nonnegative integer group tail')
