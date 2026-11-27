"""Analyze a finished held-out A1 batch. Refuse partial schedules; keep stopped worker prefixes."""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[1]
PREFIX=(1,2,4,8,16)
REF='rebuild-always'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write_rows(path,rows):
    keys=dict.fromkeys(k for row in rows for k in row)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def compare(req,ref):
    fields=('root_version_id','current_version_id','opinion_hash','initial_seed_ids','initial_gram_count',
            'budget','delta','gain_denominator','status','commits')
    if any(req[k]!=ref[k] for k in fields):
        raise ArithmeticError('request/stop outcome disagrees with repeat-0 rebuild')
    if len(req['decisions'])!=len(ref['decisions']):
        raise ArithmeticError('decision count disagreement')
    accepted=exact=0
    for a,b in zip(req['decisions'],ref['decisions']):
        if any(a[k]!=b[k] for k in ('epoch','parent_version_id','current_version_id','winner_id','status')):
            raise ArithmeticError('decision/parent/child mismatch')
        if a['status']=='accepted':
            if min(a['low'],b['low'])<=0 or max(a['low'],b['low'])>min(a['high'],b['high']):
                raise ArithmeticError('gain interval disagreement')
            exact+=int((a['low'],a['high'])==(b['low'],b['high']))
            accepted+=1
    lo=max(req['gain_low'],ref['gain_low'])
    hi=min(req['gain_high'],ref['gain_high'])
    e1,e2=req['final_energy'],ref['final_energy']
    if lo>hi or e1['denominator']!=e2['denominator'] or max(e1['low'],e2['low'])>min(e1['high'],e2['high']):
        raise ArithmeticError('cumulative energy/gain disagreement')
    return accepted,exact


def crossings(pairs):
    pts=[q for q in PREFIX if q in pairs]
    wins=[q for q in pts if pairs[q]<=0]
    durable=next((q for q in wins if all(pairs[k]<=0 for k in pts if k>=q)),None)
    return dict(first=next(iter(wins),None),first_after_q1=next((q for q in wins if q>1),None),
                sustained_from=durable,stayed_from_first=bool(wins) and durable==wins[0],
                final_observed_prefix=max(pts,default=0),complete_through_q16=16 in pts)


def summarize(values):
    if not values:
        return None
    return dict(n=len(values),median=median(values),minimum=min(values),maximum=max(values))


def analyze(folder,out):
    schedule=read(folder/'SCHEDULE_RESULTS.json')
    frozen=read(folder/'FROZEN_JOBS.json')
    protocol=read(ROOT/'protocol/A1_CONFIRMATION_PROTOCOL.json')
    manifest=read(ROOT/'protocol/A1_CONFIRMATION_INPUTS.json')
    if len(schedule['jobs'])!=54 or [j['job'] for j in schedule['jobs']]!=frozen['jobs']:
        raise ValueError('batch incomplete or job order differs from frozen schedule')
    if frozen['protocol_sha256']!=sha(ROOT/'protocol/A1_CONFIRMATION_PROTOCOL.json') or (
            frozen['input_manifest_sha256']!=sha(ROOT/'protocol/A1_CONFIRMATION_INPUTS.json')):
        raise ValueError('batch input/protocol differs from freeze')
    data={}
    for job in schedule['jobs']:
        labels=job['job']
        key=tuple(labels[k] for k in ('graph','stream','repeat_id','arm'))
        file=folder/job['result']
        obj=read(file) if file.is_file() else {}
        reqs=obj.get('requests',[])
        if job['status']=='completed' and len(reqs)!=16:
            raise ValueError('completed job missing requests')
        rec=next(x for x in manifest['fixtures'] if (x['graph'],x['stream'])==key[:2])
        if reqs:
            if any(obj[k]!=job['job'][k] for k in ('graph','stream','arm','repeat_id')):
                raise ValueError('scheduled envelope differs from measured worker')
            if (obj['graph_hash']!=rec['graph_sha256'] or obj['fixture_id']!=rec['fixture_id']
                    or obj['seed']!=196613 or obj['weight_mode']!='weighted'):
                raise ValueError('unexpected held-out worker graph or dataset')
            if (obj['fixture_matrix_sha256']!=rec['matrix_sha256'] or obj['fixture_npz_sha256']!=rec['npz_sha256'] or
                    obj['input_manifest_sha256']!=frozen['input_manifest_sha256'] or
                    obj['protocol_sha256']!=frozen['protocol_sha256']):
                raise ValueError('worker fixture/protocol binding mismatch')
            for n,r in enumerate(reqs):
                if (r['opinion_hash']!=rec['opinion_hashes'][n] or r['query_position']!=n+1 or
                        r['initial_gram_count']!=0 or r['budget']!=15 or r['delta']!=0):
                    raise ValueError('unexpected request identity or contract')
                sec=r['request_seconds']
                if abs(sum(r['phases'].values())-sec)>1e-7 or abs(r['reset_opinion_seconds']+r['online_seconds']-sec)>1e-7:
                    raise ArithmeticError('request timing mismatch')
            for q in PREFIX:
                if q<=len(reqs) and abs(obj['prefixes'][str(q)]-obj['structural_seconds']-
                      sum(r['request_seconds'] for r in reqs[:q]))>1e-7:
                    raise ArithmeticError('prefix timing mismatch')
        data[key]=obj,job,file,reqs
    workers=[];prefixes=[];verification=[]
    for (graph,stream,repeat,arm),(obj,job,file,reqs) in data.items():
        ref,ref_job,ref_file,ref_q=data[graph,stream,0,REF]
        base,bjob,bfile,bq=data[graph,stream,repeat,REF]
        counts=Counter()
        if reqs and ref_q:
            for k in ('graph_hash','operator_hash','catalog_hash','root_version_id','source_files_sha256',
                      'kernel_build_id','fixture_npz_sha256','fixture_matrix_sha256','input_manifest_sha256'):
                if obj[k]!=ref[k]:
                    raise ValueError('reference build/input identity differs')
        for r,b in zip(reqs,ref_q):
            accepted,exact=compare(r,b)
            counts.update(requests=1,decisions=len(r['decisions']),accepted=accepted,identical_bounds=exact)
        verification.append(dict(graph=graph,stream=stream,repeat=repeat,arm=arm,trace=file.name,
            trace_sha256=sha(file) if file.exists() else None,reference=ref_file.name,
            reference_sha256=sha(ref_file) if ref_file.exists() else None,matched=dict(counts),
            unmatched_requests=max(0,len(reqs)-len(ref_q))))
        head=dict(graph=graph,stream=stream,repeat=repeat,arm=arm)
        ds={}
        for q in PREFIX:
            if q>min(len(bq),len(reqs)):
                continue
            t=obj['prefixes'][str(q)];bt=base['prefixes'][str(q)]
            ds[q]=t-bt
            prefixes.append(dict(**head,q=q,arm_s=t,rebuild_s=bt,arm_minus_rebuild_s=t-bt,
                rebuild_over_arm=bt/t,arm_worker_status=job['status'],rebuild_worker_status=bjob['status']))
        elig=reqs[1:]
        counters=Counter()
        for r in elig:counters.update(r['counters'])
        row=dict(**head,worker_status=job['status'],completed_requests=len(reqs),
            failure_reason=obj.get('failure_reason'),stops=dict(Counter(r['status'] for r in reqs)),
            process_s=job.get('process_wall_seconds'),rss_mib=(obj.get('process_peak_rss_bytes',0)/2**20 or None),
            parent_sampled_rss_mib=(job.get('external_sampled_peak_rss_bytes',0)/2**20 or None),
            observed_eligible_s=sum(r['request_seconds'] for r in elig),
            eligible_checks=counters['endpoint_checked'],eligible_response_blocks=counters['response_blocks'],
            eligible_response_rhs=sum(counters[k] for k in ('candidate_rhs','winner_rhs','policy_refresh_rhs','numerical_reanchor_rhs')),
            eligible_anchor_actions=[r['discovery']['action'] for r in elig],eligible_state_rhs=counters['state_rhs'],
            eligible_predictor_s=sum(r['discovery']['prediction_s'] for r in elig),
            eligible_index_build_s=sum(r['discovery']['rebuild_s'] for r in elig),
            eligible_discovery_s=sum(r['phases']['discovery'] for r in elig),
            eligible_response_s=sum(r['phases']['response_acquisition'] for r in elig),
            eligible_retain=sum(r['discovery']['action']=='retain' for r in elig),
            eligible_rebuild=sum(r['discovery']['action']=='rebuild' for r in elig),
            **crossings(ds))
        if len(reqs)==len(bq)==16:
            ref_eligible=sum(r['request_seconds'] for r in bq[1:])
            row['eligible_arm_minus_rebuild_s']=row['observed_eligible_s']-ref_eligible
            row['eligible_rebuild_over_arm']=ref_eligible/row['observed_eligible_s']
        if job['status']==bjob['status']=='completed':
            row['process_arm_minus_rebuild_s']=row['process_s']-bjob['process_wall_seconds']
            row['rss_arm_minus_rebuild_mib']=row['rss_mib']-base['process_peak_rss_bytes']/2**20
        workers.append(row)
    groups=[]
    for g in protocol['graphs']:
        for s in protocol['streams']:
            for a in protocol['methods']:
                w=[r for r in workers if (r['graph'],r['stream'],r['arm'])==(g,s,a)]
                pp=[r for r in prefixes if (r['graph'],r['stream'],r['arm'])==(g,s,a)]
                item=dict(graph=g,stream=s,arm=a,statuses=dict(Counter(r['worker_status'] for r in w)),
                    decisions_by_repeat=[r['eligible_anchor_actions'] for r in sorted(w,key=lambda r:r['repeat'])],
                    crossings_by_repeat=[{k:r[k] for k in ('repeat','first','first_after_q1','sustained_from',
                          'stayed_from_first','final_observed_prefix')} for r in sorted(w,key=lambda r:r['repeat'])],
                    prefixes={})
                pdelta={}
                for q in PREFIX:
                    qq=[r for r in pp if r['q']==q]
                    item['prefixes'][q]={k:summarize([r[k] for r in qq]) for k in (
                        'arm_s','rebuild_s','arm_minus_rebuild_s','rebuild_over_arm')}
                    if qq:pdelta[q]=item['prefixes'][q]['arm_minus_rebuild_s']['median']
                item['median_paired_crossing']=crossings(pdelta)
                for k in ('observed_eligible_s','eligible_arm_minus_rebuild_s','eligible_rebuild_over_arm','process_s','rss_mib',
                         'process_arm_minus_rebuild_s','rss_arm_minus_rebuild_mib','eligible_checks','eligible_response_blocks',
                         'eligible_response_rhs','eligible_predictor_s','eligible_index_build_s','eligible_discovery_s',
                         'eligible_response_s','eligible_retain','eligible_rebuild'):
                    item[k]=summarize([r[k] for r in w if r['completed_requests']==16 and r.get(k) is not None])
                groups.append(item)
    totals=Counter()
    for v in verification:totals.update(v['matched'])
    result=dict(protocol=protocol['id'],schedule_sha256=sha(folder/'SCHEDULE_RESULTS.json'),
        manifest_sha256=frozen['input_manifest_sha256'],jobs=len(workers),statuses=dict(Counter(r['worker_status'] for r in workers)),
        observed_requests=sum(r['completed_requests'] for r in workers),repeat0_rebuild_agreement=dict(totals),
        unmatched_requests=sum(r['unmatched_requests'] for r in verification),references=verification,groups=groups,
        scope='Within-batch same-opinion comparison to repeat-0 optimized rebuild, separate from independent proof. Timings pair within repeat. Crossings use observed prefixes only; Q1 is an initial-build control. Full stream comparisons require complete workers.')
    out.mkdir(parents=True,exist_ok=True)
    write_rows(out/'prefix_pairs.csv',prefixes)
    write_rows(out/'workers.csv',workers)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    obj=analyze(a.input,a.output)
    print(obj['statuses'],obj['repeat0_rebuild_agreement'],flush=True)
