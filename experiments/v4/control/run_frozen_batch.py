"""Only the predeclared 36 workers; resume by inspection, never replace a result."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
    w=a.workspace.resolve();root=w/'v4';runtime=w/'v4_runtime';freeze=root/'protocols/HG_LIGHT_FROZEN.json'
    frozen=json.loads(freeze.read_text())
    if frozen['run_kind']!='formal' or len(frozen['jobs'])!=36:raise ValueError('wrong formal batch')
    for rel,digest in {**frozen['analysis_files_sha256'],**frozen['control_files_sha256']}.items():
        with (w/rel).open('rb') as f:observed=hashlib.file_digest(f,'sha256').hexdigest()
        if observed!=digest:raise ValueError('changed frozen analysis/control source: '+rel)
    scheduler=root/'control/budget_run.py'
    for repeat in range(3):
        if repeat:
            previous=json.loads((root/f'reports/formal_repeat_{repeat-1}.json').read_text())
            if previous['status']!='passed':raise SystemExit('preceding repeat failed its behavior gate')
        for job in (j for j in frozen['jobs'] if j['repeat_id']==repeat):
            output=w/frozen['raw_dir']/job['output_file'];task='hg_light_'+output.stem
            task_dir=root/'runs'/task
            if task_dir.exists() or output.exists():
                record_path=task_dir/'record.json'
                if not record_path.is_file() or not output.is_file():raise SystemExit('incomplete existing attempt; no replacement: '+task)
                record=json.loads(record_path.read_text())
                if record['status']!='completed' or record['exit_code']!=0:raise SystemExit('existing failed attempt; no replacement: '+task)
                print(json.dumps({'existing_attempt_preserved':task}),flush=True)
                continue
            command=[sys.executable,str(scheduler),'--root',str(root),'--runtime',str(runtime),
                '--task',task,'--category','formal','--decision',
                'Prespecified paired H/G complete-service cost under the shared light policy; no adaptive repeats.',
                '--timeout','180','--memory-gib','2','--binding',str(freeze),'--cwd',str(runtime),
                '--',sys.executable,'-m','src.v3_worker','--freeze',str(freeze),'--job',str(job['position']),'--output',str(output)]
            completed=subprocess.run(command)
            if completed.returncode:raise SystemExit('worker resource/runtime failure; stop without replacement: '+task)
            row=json.loads(output.read_text())
            if row.get('status') not in ('completed_budget','certified_stop','unresolved'):
                raise SystemExit('unexpected outcome; stop without replacement: '+task)
        gate=root/f'reports/formal_repeat_{repeat}.json'
        if not gate.exists():
            command=[sys.executable,str(scheduler),'--root',str(root),'--runtime',str(runtime),
                '--task',f'formal_gate_{repeat}','--category','analysis','--decision',
                'Admit only the already preplanned next repeat after exact same-method historical behavior and positive resource/binding checks.',
                '--timeout','120','--memory-gib','2','--binding',str(freeze),'--cwd',str(w),
                '--',sys.executable,str(root/'analysis/verify_formal_repeat.py'),'--workspace',str(w),'--repeat',str(repeat)]
            completed=subprocess.run(command)
            if completed.returncode:raise SystemExit('repeat behavior gate failed; later repeats not run')
        result=json.loads(gate.read_text())
        if result['status']!='passed':raise SystemExit('repeat gate blocked; later repeats not run')
        print(json.dumps({'repeat_gate':'passed','repeat_id':repeat}),flush=True)
    print(json.dumps({'batch':'finished_predeclared_36_workers','further_experiments':'not_authorized_by_this_batch'}),flush=True)


if __name__=='__main__':main()
