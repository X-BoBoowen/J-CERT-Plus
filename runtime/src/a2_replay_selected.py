"""Run only preregistered independent complete-catalog replays, serially.

This is offline verification, never included in the algorithm's timing.
It uses the accepted RSS supervisor but the independent A2 catalog checker.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from .a2_bench import ROOT, THREAD_ENV, load_freeze, save, sha
from .access_bench import supervise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('do not overwrite selected independent replay evidence')
    cfg,manifest,_,manifest_hash=load_freeze(args.protocol,args.manifest)
    schedule=json.loads((args.directory/'SCHEDULE_RESULTS.json').read_text())
    selected=manifest['independent_replay_selection']
    limits=dict(worker_seconds=600,rss_mib=2048,batch_seconds=1200,rss_sample_seconds=.025)
    out=dict(scope='Independent full-catalog selected requests; separate from service timing',
        protocol=cfg['id'],manifest_sha256=manifest_hash,limits=limits,
        checker_sha256=sha(ROOT/'src/a2_checker.py'),selection=selected,results=[],status='in_progress')
    started=time.monotonic();deadline=started+limits['batch_seconds']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for key in THREAD_ENV:os.environ[key]='1'
    os.sched_setaffinity(0,{8})
    for selection in selected:
        keys=('graph','stream','seed_label','method','repeat_id')
        job=next(row for row in schedule['jobs'] if all(row['job'][k]==selection[k] for k in keys))
        trace=args.directory/job['result']
        fixture=next(row for row in manifest['fixtures'] if all(row[k]==selection[k] for k in ('graph','stream','seed_label')))
        opinions=(args.manifest.parent/fixture['file']).resolve()
        graph=ROOT/manifest['graphs'][selection['graph']]['weighted_path']
        output=args.output.with_name(args.output.stem+'_'+selection['graph']+'.json')
        if output.exists():raise FileExistsError('selected replay output already exists')
        command=['taskset','-c','0',sys.executable,'-m','src.a2_checker',str(trace),
            '--graph',str(graph),'--opinions',str(opinions),'--request',str(selection['query_position']),
            '--output',str(output)]
        if time.monotonic()<deadline:
            code,wall,peak,stop=supervise(command,output.with_suffix('.log'),
                time.monotonic()+limits['worker_seconds'],deadline,limits)
            row=dict(selection=selection,trace=str(trace),trace_sha256=sha(trace),
                output=str(output),returncode=code,wall_seconds=wall,
                external_sampled_peak_rss_bytes=peak,status=stop or ('passed' if code==0 else 'failed'))
            if output.exists():
                row['proof']=json.loads(output.read_text());row['output_sha256']=sha(output)
                proof=row['proof']['proof']
                if row['status']=='passed' and not (proof['full_catalog_decisions_certified'] and proof['termination_certified']):
                    row['status']='certified_prefix_only'
            elif row['status']=='passed':row['status']='missing_checker_output'
        else:row=dict(selection=selection,status='batch_cap_not_started')
        out['results'].append(row);out['wall_seconds']=time.monotonic()-started;save(args.output,out)
        print(selection['graph'],row['status'],row.get('wall_seconds'),flush=True)
    out['status']='passed' if all(r['status']=='passed' for r in out['results']) else 'incomplete_or_failed'
    save(args.output,out)
    return 0 if out['status']=='passed' else 1


if __name__=='__main__':sys.exit(main())
