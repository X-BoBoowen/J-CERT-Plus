"""Serial dispatcher for the twelve already frozen jobs, never a numerical child."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-dispatch', action='store_true')
    options = parser.parse_args()
    workspace = Path(__file__).resolve().parents[2]
    freeze = workspace / 'v3/resumed/protocols/HG_FROZEN.json'
    spec = json.loads(freeze.read_text())
    raw_dir = workspace / spec['current_raw_dir']
    raw_dir.mkdir(parents=True, exist_ok=True)
    old = []
    for path in (workspace / spec['old_raw_dir']).glob('*.json'):
        obj = json.loads(path.read_text())
        if (isinstance(obj, dict) and obj.get('method') == 'H' and obj.get('repeat_id') == 0
                and isinstance(obj.get('requests'), list)):
            old.append(obj)
    if len(old) != 6 or len({(r['graph'], r['stream']) for r in old}) != 6:
        raise ValueError('expected exactly six retained H/repeat0 result records')
    batch = dict(freeze_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest(),
                 launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), jobs=[], status='running')
    report_path = workspace / 'v3/resumed/reports/hg_batch_dispatch.json'
    def save():
        report_path.write_text(json.dumps(batch, indent=2)+'\n')
    save()
    for job in spec['jobs']:
        path = raw_dir / job['output_file']
        task = 'hg_'+path.stem
        run_dir = workspace / 'v3/runs' / task
        existing = path.exists() or run_dir.exists()
        if existing:
            if not options.resume_dispatch or not path.exists() or not (run_dir / 'record.json').exists():
                raise RuntimeError('job exists without completed resumable evidence: '+task)
            kept = json.loads((run_dir / 'record.json').read_text())
            retained = json.loads(path.read_text())
            if (retained.get('freeze_sha256') != batch['freeze_sha256'] or retained.get('job') != job
                    or retained.get('source_digest') != spec['source_digest']):
                raise ValueError('retained job cannot be rebound to this batch')
        cmd = [sys.executable, str(workspace / 'v3/control/resumed_budget_run.py'),
               '--root', str(workspace / 'v3'), '--runtime', str(workspace / 'v3_runtime'),
               '--task', task, '--category', 'small_timing',
               '--decision', 'Paired old-stream H/G full-service cost under the same frozen current numerical core; test whether a strong simple cover leaves net value for H.',
               '--timeout', '180', '--memory-gib', '2', '--binding', str(freeze),
               '--cwd', str(workspace / 'v3_runtime'), '--', sys.executable, '-m', 'src.v3_worker',
               '--freeze', str(freeze), '--job', str(job['position']), '--output', str(path)]
        if existing:
            exit_code = kept['exit_code']
            print('RETAIN WITHOUT RERUN '+task, flush=True)
        else:
            print('START '+task, flush=True)
            exit_code = subprocess.run(cmd, check=False).returncode
        entry = dict(job=job, scheduler_exit_code=exit_code, retained_without_rerun=existing,
                     output=str(path.relative_to(workspace)), old_prefix_check='not_run')
        batch['jobs'].append(entry)
        if path.exists():
            trace = json.loads(path.read_text())
            entry['worker_status'] = trace['status']
            entry['complete_Q4'] = trace.get('complete_Q4', False)
            previous = next(r for r in old if (r['graph'], r['stream']) == (job['graph'], job['stream']))
            for current, former in zip(trace.get('requests', []), previous['requests']):
                ca = [r for r in current['decisions'] if r['status'] == 'accepted']
                fa = [r for r in former['decisions'] if r['status'] == 'accepted']
                keys = ('epoch','winner_id','parent_version_id','current_version_id','i','j','flip','d')
                for c, f in zip(ca, fa):
                    if any(c.get(k) != f.get(k) for k in keys):
                        entry['old_prefix_check'] = 'accepted_trace_conflict'
                        batch['status'] = 'paused_correctness_gate'
                        save()
                        return 2
                if len(ca) > len(fa):
                    entry['old_prefix_check'] = 'new_decision_requires_independent_replay'
                    batch['status'] = 'paused_verification_gate'
                    save()
                    return 2
            entry['old_prefix_check'] = 'all_returned_accepted_prefixes_match_retained_H'
        else:
            entry['worker_status'] = 'no_worker_output'
        if entry['worker_status'] == 'runtime/certificate_error':
            batch['status'] = 'paused_runtime_or_certificate_error'
            save()
            return 2
        save()
        print('END '+task+' '+entry['worker_status'], flush=True)
    batch['status'] = 'initial_batch_ended_all_twelve_retained'
    save()
    return 0


if __name__ == '__main__':
    sys.exit(main())
