"""Read-only same-method regression gate for a complete predeclared repeat."""
import argparse
import json
from pathlib import Path

from check_semantic_invariance import compare_workers, SCHEMA
from validate_worker import binding_issues, sha


def light_obligation_issues(row):
    issues=[]
    if row.get('status') not in ('completed_budget','certified_stop','unresolved'):
        issues.append('unexpected worker outcome')
    if row.get('active_phase')!='complete':issues.append('worker not at final checkpoint')
    requests=row.get('requests',[])
    complete=(len(requests)==4 and all(r.get('result_status') in ('completed_budget','certified_stop') for r in requests))
    if row.get('complete_Q4') is not complete:issues.append('inconsistent complete_Q4')
    if complete:
        if row.get('complete_service_seconds')!=row.get('service_accounted_seconds') or row.get('partial_work_seconds') is not None:
            issues.append('complete cost scope')
    else:
        if row.get('complete_service_seconds') is not None or row.get('partial_work_seconds')!=row.get('service_accounted_seconds'):
            issues.append('partial cost scope')
    for q,r in enumerate(requests,1):
        prefix=f'q{q} '
        if r.get('initial_gram_count')!=0:issues.append(prefix+'nonempty request Gram cache')
        if r.get('root_version_id')!=row.get('root_version_id'):issues.append(prefix+'root reset')
        decisions=r.get('decisions',[])
        accepted=sum(d.get('status')=='accepted' for d in decisions)
        if r.get('commits')!=accepted:issues.append(prefix+'accepted count')
        mem=r.get('memory',{})
        if mem.get('memory_observation_schema')!='light_v1':issues.append(prefix+'memory schema')
        for field in ('explicit_catalog_bytes','structural_weight_grouping','incidence_bytes','index_private_bytes',
                      'peak_frontier_owned_bytes','frontier_snapshot','sparse_catalog','gram_python_bytes',
                      'actual_candidate_cache_python_bytes','decision_and_access_log_python_bytes',
                      'low_rank_steps','peak_low_rank_steps'):
            if field not in mem or mem[field] is not None:issues.append(prefix+'unmeasured field '+field)
        observations=mem.get('light_observations',[])
        expected=[('request_begin',0)]
        for d in decisions:expected.extend([('version_open',d['epoch']),('version_finish',d['epoch'])])
        expected.append(('request_end',len(decisions)))
        actual=[(item.get('point'),item.get('epoch')) for item in observations]
        if actual!=expected:issues.append(prefix+'observation points')
        if mem.get('low_rank_step_count')!=accepted:issues.append(prefix+'factor step count')
        if observations:
            if observations[0].get('gram_items')!=0 or observations[0].get('low_rank_step_count')!=0:
                issues.append(prefix+'observed request cache/factor reset')
            if observations[-1].get('chosen_items')!=accepted:issues.append(prefix+'observed chosen count')
        if 'diagnostic_cache_audit' in r or 'diagnostic_deep_calls' in r:
            issues.append(prefix+'diagnostic instrumentation leaked into formal worker')
    return issues


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--repeat',type=int,choices=(0,1,2),required=True);a=p.parse_args()
    w=a.workspace.resolve();path=w/'v4/protocols/HG_LIGHT_FROZEN.json'
    frozen=json.loads(path.read_text());target=w/f'v4/reports/formal_repeat_{a.repeat}.json'
    if target.exists():raise FileExistsError('repeat gates are retained, never overwritten')
    report=dict(status='running',repeat_id=a.repeat,runtime_commit=frozen['runtime_commit'],
        freeze_sha256=sha(path),schema=SCHEMA,workers=[],issues=[],
        scope='exact same-method logged semantics against retained V3; resource/source/input/policy and positive completion/reset checks; no new independent numerical replay')
    for rel,digest in frozen['analysis_files_sha256'].items():
        if sha(w/rel)!=digest:report['issues'].append('changed frozen analysis '+rel)
    jobs=[j for j in frozen['jobs'] if j['repeat_id']==a.repeat]
    if len(jobs)!=12:report['issues'].append('repeat does not register twelve workers')
    for job in jobs:
        file=w/frozen['raw_dir']/job['output_file'];record_path=w/'v4/runs'/('hg_light_'+file.stem)/'record.json'
        if not file.is_file() or not record_path.is_file():
            report['issues'].append('missing registered worker '+job['output_file']);continue
        row=json.loads(file.read_text());record=json.loads(record_path.read_text())
        issues=binding_issues(row,frozen,job,record,path)+light_obligation_issues(row)
        oldjob=next(j for j in frozen['historical_V3_jobs'] if (j['graph'],j['stream'],j['method'])==(job['graph'],job['stream'],job['method']))
        oldpath=w/frozen['historical_V3_raw_dir']/oldjob['output_file']
        comparison=compare_workers(json.loads(oldpath.read_text()),row,historical=True)
        if not comparison['equal']:issues.append('same-method historical logged semantics differ')
        report['workers'].append(dict(file=str(file.relative_to(w)),sha256=sha(file),record_sha256=sha(record_path),
            V3_file=str(oldpath.relative_to(w)),V3_sha256=sha(oldpath),method=job['method'],graph=job['graph'],
            stream=job['stream'],complete_Q4=row.get('complete_Q4'),status=row.get('status'),
            accepted_records=sum(r['commits'] for r in row['requests']),comparison=comparison,issues=issues))
        report['issues'].extend(job['output_file']+': '+issue for issue in issues)
    report['status']='passed' if not report['issues'] and len(report['workers'])==12 else 'blocked'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],repeat_id=a.repeat,workers=len(report['workers']),issues=report['issues'])))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
