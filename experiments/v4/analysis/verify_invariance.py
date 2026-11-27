"""Accept four diagnostics only with complete bindings and positive audit evidence."""
import argparse
import json
from pathlib import Path

from check_semantic_invariance import compare_workers,SCHEMA
from diagnostic_support import diagnostic_obligation_issues,DEEP_LABELS
from validate_worker import binding_issues,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
    w=a.workspace.resolve();path=w/'v4/protocols/INVARIANCE.json';frozen=json.loads(path.read_text())
    report=dict(status='running',runtime_commit=frozen['runtime_commit'],freeze_sha256=sha(path),
        schema=SCHEMA,workers=[],same_method_policy_pairs=[],historical_same_method=[],issues=[],
        scope='four diagnostic-only full Q4 workers; exact logged semantics plus final-cache/factor digests; no new independent numerical replay')
    rows={}
    for job in frozen['jobs']:
        file=w/frozen['raw_dir']/job['output_file']
        record_path=w/'v4/runs'/('inv_'+file.stem)/'record.json'
        if not file.is_file() or not record_path.is_file():
            report['issues'].append('missing registered diagnostic '+job['output_file']);continue
        row=json.loads(file.read_text());record=json.loads(record_path.read_text())
        issues=binding_issues(row,frozen,job,record,path)+diagnostic_obligation_issues(row)
        for issue in issues:report['issues'].append(job['output_file']+': '+issue)
        rows[(job['method'],job['measurement_policy'])]=row
        calls={label:sum(r.get('diagnostic_deep_calls',{}).get(label,0) for r in row['requests']) for label in DEEP_LABELS}
        report['workers'].append(dict(file=str(file.relative_to(w)),sha256=sha(file),record_sha256=sha(record_path),
            method=job['method'],policy=job['measurement_policy'],complete_Q4=row.get('complete_Q4'),
            accepted_records=sum(r['commits'] for r in row['requests']),deep_calls=calls,issues=issues))
        oldjob=next(j for j in frozen['historical_V3_jobs'] if (j['graph'],j['stream'],j['method'])==(job['graph'],job['stream'],job['method']))
        oldpath=w/frozen['historical_V3_raw_dir']/oldjob['output_file']
        comparison=compare_workers(json.loads(oldpath.read_text()),row,historical=True)
        comparison.update(method=job['method'],policy=job['measurement_policy'],V3_file=str(oldpath.relative_to(w)),V3_sha256=sha(oldpath))
        report['historical_same_method'].append(comparison)
        if not comparison['equal']:report['issues'].append('historical logged semantics differ: '+job['output_file'])
    for method in ('H','G'):
        if (method,'legacy_deep') not in rows or (method,'light_v1') not in rows:
            report['issues'].append('missing same-method pair '+method);continue
        comparison=compare_workers(rows[(method,'legacy_deep')],rows[(method,'light_v1')])
        comparison['method']=method;report['same_method_policy_pairs'].append(comparison)
        if not comparison['equal']:report['issues'].append('same-build policy semantics differ: '+method)
    report['status']='passed' if not report['issues'] and len(report['workers'])==4 else 'blocked'
    target=w/'v4/reports/semantic_invariance.json';target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'workers':len(report['workers']),'issues':report['issues'],
                      'same_method_equal':[r['equal'] for r in report['same_method_policy_pairs']]}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
