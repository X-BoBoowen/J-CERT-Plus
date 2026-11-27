#!/usr/bin/env python3
"""Sole V4 serial scheduler; all numerical attempts share the 1200-second ledger."""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--runtime', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--category', required=True,
                   choices=('micro', 'invariance', 'formal', 'analysis', 'delivery_check'))
    p.add_argument('--decision', required=True)
    p.add_argument('--timeout', type=float, required=True)
    p.add_argument('--memory-gib', type=float, required=True)
    p.add_argument('--binding', action='append', default=[])
    p.add_argument('--cwd', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    root, runtime = Path(a.root).resolve(), Path(a.runtime).resolve()
    protocol_path = root / 'protocols/EXECUTION.json'
    protocol = json.loads(protocol_path.read_text())
    limits_spec = protocol['category_limits'][a.category]
    if not (0 < a.timeout <= limits_spec['seconds'] and
            0 < a.memory_gib * 1024**3 <= limits_spec['memory_bytes']):
        p.error('requested resource limits exceed the frozen category limit')
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        p.error('missing child command')
    ledger_path = root / 'budget_ledger.jsonl'
    with (root / 'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = [json.loads(line) for line in ledger_path.read_text().splitlines()]
        used = sum(row['wall_seconds'] for row in previous)
        count = sum(row.get('category') == a.category for row in previous)
        if limits_spec.get('max_processes') is not None and count >= limits_spec['max_processes']:
            raise SystemExit('category process count exhausted')
        reserve = 0 if a.category == 'delivery_check' else protocol['verification_reserve_seconds']
        if protocol['numeric_budget_seconds'] - used < a.timeout + reserve:
            raise SystemExit('budget or verification reserve would be exceeded')
        task_dir = root / 'runs' / a.task
        task_dir.mkdir(parents=True, exist_ok=False)
        paths = list((runtime / 'src').rglob('*.py'))
        paths += list((runtime / 'baseline_r15/experimental_records/evidence_supplement_20261124/source').rglob('*.py'))
        paths += list((runtime / 'baseline_r15/experimental_records/evidence_supplement_20261124/diagnostic').rglob('*.py'))
        paths += list((runtime / 'baseline_r15').rglob('*.so'))
        runtime_files = {str(x.relative_to(runtime)): sha(x) for x in sorted(set(paths))}
        analysis_files = {str(x.relative_to(root)): sha(x) for folder in
                          (root / 'analysis', root / 'control')
                          for x in sorted(folder.glob('*.py'))}
        bindings = {str(Path(x).resolve()): sha(x) for x in a.binding}
        packages = {}
        for name in ('numpy', 'scipy', 'psutil', 'threadpoolctl', 'sympy', 'mpmath'):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        environment = dict(python=sys.version, executable=sys.executable,
                           executable_sha256=sha(sys.executable), platform=platform.platform(),
                           packages=packages, allowed_affinity=sorted(os.sched_getaffinity(0)))
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime, text=True).strip()
        tracked_changes = subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=runtime, text=True).strip()
        env = os.environ.copy()
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = '1'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONOPTIMIZE'] = '0'
        affinity = min(os.sched_getaffinity(0))
        memory_bytes = int(a.memory_gib * 1024**3)
        def impose():
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            os.sched_setaffinity(0, {affinity})
        row = dict(task=a.task, category=a.category, decision_changed_by_result=a.decision,
                   command=command, cwd=str(Path(a.cwd).resolve()), started_utc=now(),
                   timeout_seconds=a.timeout, memory_limit_bytes=memory_bytes,
                   memory_limit_kind='RLIMIT_AS', cpu_affinity=[affinity], numeric_threads=1,
                   runtime_git_commit=commit, tracked_runtime_changes=tracked_changes,
                   runtime_files=runtime_files, analysis_files=analysis_files,
                   runtime_source_sha256=hashlib.sha256(json.dumps(runtime_files, sort_keys=True).encode()).hexdigest(),
                   protocol_sha256=sha(protocol_path), input_and_protocol_bindings=bindings,
                   environment=environment, child_process_policy='one numerical child; no nested numerical workers',
                   cost_scope='one child process wall including imports, numerical work, checks and serialization',
                   stdout=str((task_dir / 'stdout.txt').relative_to(root)),
                   stderr=str((task_dir / 'stderr.txt').relative_to(root)))
        (task_dir / 'start.json').write_text(json.dumps(row, indent=2) + '\n')
        start, peak_rss, peak_vms, timed_out = time.monotonic(), 0, 0, False
        with (task_dir / 'stdout.txt').open('w') as out, (task_dir / 'stderr.txt').open('w') as err:
            child = subprocess.Popen(command, cwd=a.cwd, env=env, stdout=out, stderr=err,
                                     preexec_fn=impose, start_new_session=True)
            while child.poll() is None:
                try:
                    fields = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                              for line in Path(f'/proc/{child.pid}/status').read_text().splitlines() if ':' in line}
                    peak_rss = max(peak_rss, int(fields.get('VmRSS', '0 kB').split()[0]) * 1024)
                    peak_vms = max(peak_vms, int(fields.get('VmSize', '0 kB').split()[0]) * 1024)
                except (FileNotFoundError, ProcessLookupError):
                    pass
                elapsed = time.monotonic() - start
                if elapsed >= a.timeout:
                    timed_out = True
                    os.killpg(child.pid, signal.SIGKILL)
                    break
                try:
                    child.wait(timeout=min(0.1, a.timeout - elapsed))
                except subprocess.TimeoutExpired:
                    pass
            code = child.wait()
        wall = time.monotonic() - start
        row.update(finished_utc=now(), wall_seconds=wall, exit_code=code,
                   status='timeout' if timed_out else 'completed' if code == 0 else 'nonzero_exit',
                   sampled_peak_rss_bytes=peak_rss, sampled_peak_vms_bytes=peak_vms,
                   rss_scope='100ms sampled /proc child RSS/VMS maxima, may miss peaks; worker ru_maxrss covers its final pre-serialization checkpoint',
                   cumulative_wall_seconds=used + wall,
                   remaining_seconds=protocol['numeric_budget_seconds'] - used - wall)
        (task_dir / 'record.json').write_text(json.dumps(row, indent=2) + '\n')
        with ledger_path.open('a') as f:
            f.write(json.dumps(row, sort_keys=True) + '\n')
        print(json.dumps({k: row[k] for k in ('task', 'status', 'exit_code', 'wall_seconds',
                         'sampled_peak_rss_bytes', 'cumulative_wall_seconds', 'remaining_seconds')}, indent=2))
        return 124 if timed_out else code


if __name__ == '__main__':
    sys.exit(main())
