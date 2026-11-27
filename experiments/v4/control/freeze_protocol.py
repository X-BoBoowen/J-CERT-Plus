"""Freeze diagnostic or formal V4 jobs, before any worker in that batch."""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--kind',choices=('invariance','formal'),required=True);a=p.parse_args()
    w=a.workspace.resolve();root=w/'v4';runtime=w/'v4_runtime'
    target=root/'protocols'/('INVARIANCE.json' if a.kind=='invariance' else 'HG_LIGHT_FROZEN.json')
    if target.exists():raise FileExistsError('frozen protocols are never overwritten')
    old=json.loads((w/'v3/resumed/protocols/HG_FROZEN.json').read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=runtime,text=True).strip():
        raise ValueError('runtime must be committed before freezing')
    allowed={'src/a2_session.py','src/a2_access.py','src/g_access.py','src/v3_worker.py'}
    files={};changes={}
    for rel,digest in old['source_files_sha256'].items():
        observed=sha(runtime/rel)
        if observed!=digest:
            if rel not in allowed:raise ValueError('nonmeasurement baseline source changed: '+rel)
            changes[rel]={'V3':digest,'V4':observed}
        files[rel]=observed
    for folder in ('src','tests'):
        for file in (runtime/folder).glob('*.py'):files[str(file.relative_to(runtime))]=sha(file)
    if sha(runtime/'v2/INPUTS.json')!=old['v2_input_manifest_sha256']:raise ValueError('input manifest changed')
    environment=dict(python=sys.version,platform=platform.platform(),packages={name:importlib.metadata.version(name)
        for name in ('numpy','scipy','psutil','threadpoolctl','sympy')})
    if environment!=old['environment']:raise ValueError('V4 environment differs from the verified V3 baseline')
    for graph in old['input_graphs'].values():
        if sha(runtime/graph['weighted_path'])!=graph['weighted_sha256']:raise ValueError('graph bytes changed')
    jobs=[]
    cells=old['input_fixtures']
    if [(f['graph'],f['stream']) for f in cells]!=[(g,s) for g in ('Alpha','OTC') for s in ('slow','independent','signed-neighbor')]:
        raise ValueError('unexpected six-cell order')
    def append(cell_index,repeat,method,policy):
        f=cells[cell_index];position=len(jobs)
        suffix=f"{method}_{policy}" if a.kind=='invariance' else method
        jobs.append(dict(position=position,cell_index=cell_index,graph=f['graph'],stream=f['stream'],
            method=method,repeat_id=repeat,requests=4,budget_per_request=15,measurement_policy=policy,
            output_file=f"{position:02d}_{f['graph']}_{f['stream']}_{suffix}_r{repeat}.json"))
    if a.kind=='invariance':
        for method in ('H','G'):
            for policy in ('legacy_deep','light_v1'):append(0,0,method,policy)
    else:
        gate=json.loads((root/'reports/semantic_invariance.json').read_text())
        if gate.get('status')!='passed' or gate.get('runtime_commit')!=commit:raise ValueError('same-build diagnosis gate not passed')
        for repeat in range(3):
            for cell_index in range(6):
                for method in (('H','G') if (cell_index+repeat)%2==0 else ('G','H')):
                    append(cell_index,repeat,method,'light_v1')
    sys.path.insert(0,str(root/'analysis'))
    from check_semantic_invariance import SCHEMA
    result=dict(id='J-CERT+-V4-'+a.kind,run_kind=a.kind,
        frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status='frozen_before_first_worker',
        runtime_commit=commit,v2_baseline_commit=old['v2_baseline_commit'],v3_baseline_commit=old['runtime_commit'],
        measurement_policy='explicit_per_job' if a.kind=='invariance' else 'light_v1',
        source_files_sha256=files,source_digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
        intentional_source_changes=changes,numerical_kernel_change=False,access_strategy_change=False,
        analysis_files_sha256={str(file.relative_to(w)):sha(file) for file in sorted((root/'analysis').glob('*.py'))},
        environment=environment,resources=dict(seconds=180,memory_bytes=2*1024**3,cpu=min(os.sched_getaffinity(0)),numerical_threads=1),
        jobs=jobs,worker_count=len(jobs),new_input_streams=0,independent_input_streams=6,
        input_fixtures=cells,input_graphs=old['input_graphs'],v2_input_manifest_sha256=old['v2_input_manifest_sha256'],
        contracts=old['contracts'],historical_V3_freeze_sha256=sha(w/'v3/resumed/protocols/HG_FROZEN.json'),
        historical_V3_raw_dir='v3/resumed/raw_hg',historical_V3_jobs=old['jobs'],
        raw_dir='v4/raw_invariance' if a.kind=='invariance' else 'v4/raw_hg_light',
        semantic_comparison_schema=SCHEMA,
        policy_observation_points=['request_begin','version_open','version_finish','request_end'],
        observation_scope='fixed-count len/nbytes/existing counters only; all recursive reports and factor-step byte scans disabled for light_v1; named shared payloads are not summed as total memory',
        main_time_definition='graph-load and structural preparation + all actual full-request walls, including necessary light observations and original G lazy import',
        process_wall_definition='separate full child lifetime including imports, bindings, checkpoints, serialization and diagnosis-only scans; never added to service',
        resource_observation_scope='supervisor 100ms RSS/VMS sampled maxima; worker ru_maxrss through last read before final JSON serialization; hard2GiB address-space limit',
        scheduler_sha256=sha(root/'control/budget_run.py'),execution_protocol_sha256=sha(root/'protocols/EXECUTION.json'),
        control_files_sha256={str(file.relative_to(w)):sha(file) for file in sorted((root/'control').glob('*.py'))},
        analysis_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        selection_rule=dict(complete_gate='five original complete cells each have all three paired repeats, plus binding, semantics and resources',
            R='sum H service / sum G service over fifteen complete pairs',r_i='median of three within-cell paired H/G ratios',
            G=dict(aggregate_min=1.10,positive_cell_medians_min=4,cell_median_min=0.90,repeat_aggregate_min=1.10),
            H=dict(aggregate_max=10/11,negative_cell_medians_min=4,cell_median_max=10/9,repeat_aggregate_max=10/11),
            stability_ruling='repeat-specific aggregate guard resolves undefined fluctuation clause before timing; no significance claim',
            otherwise='retain current H default; report mixed/insufficient evidence; no extra repeats'),
        stop='no source changes after formal freeze; no replacement of completed worker; gate failure stops; no optimization loop',
        verification='no new full-domain numerical replay; exact same-method logged semantics, plus diagnostic-only final cache/step digests',
        gate_report_sha256=None if a.kind=='invariance' else sha(root/'reports/semantic_invariance.json'))
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'path':str(target),'sha256':sha(target),'commit':commit,'jobs':len(jobs)}))


if __name__=='__main__':main()
