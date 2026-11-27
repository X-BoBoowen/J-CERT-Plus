#!/usr/bin/env python3
"""Serial bounded launcher for the standalone V3 numerical checks.

This does not run or recreate the unavailable V2 runtime. Every numerical
child, including a failing test or packaging checker, receives one ledger row.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--category', required=True)
    parser.add_argument('--decision', required=True)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--memory-gib', type=float, default=2)
    parser.add_argument('--cwd')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    protocol_path = root / 'PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text())
    ledger_path = root / 'budget_ledger.jsonl'
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('a command after -- is required')
    if not (0 < args.timeout <= 120 and 0 < args.memory_gib <= 2):
        parser.error('this source-limited launcher permits only micro checks: <=120s, <=2GiB')
    lock_path = root / 'scheduler.lock'
    root.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = [json.loads(line) for line in ledger_path.read_text().splitlines()] if ledger_path.exists() else []
        used = sum(row['wall_seconds'] for row in records)
        cap = protocol['numeric_budget_seconds']
        if cap - used < args.timeout + protocol['verification_reserve_seconds']:
            raise SystemExit('refused: verification reserve would be consumed')
        task_dir = root / 'runs' / args.task
        if task_dir.exists():
            raise SystemExit('task name already exists; never overwrite a run')
        task_dir.mkdir(parents=True)
        files = sorted((root / 'code').glob('*.py')) + [Path(__file__).resolve()]
        code_files = {str(p.relative_to(root)): sha(p) for p in files}
        code_digest = hashlib.sha256(json.dumps(code_files, sort_keys=True).encode()).hexdigest()
        env = os.environ.copy()
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        # Mathematical probe assertions must remain active. The probe also
        # rejects an explicit python -O command at import time.
        env['PYTHONOPTIMIZE'] = '0'
        affinity = min(os.sched_getaffinity(0))
        memory_bytes = int(args.memory_gib * 1024**3)
        def limits():
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            os.sched_setaffinity(0, {affinity})
        started = utc()
        start = time.monotonic()
        timed_out = False
        with (task_dir / 'stdout.txt').open('w') as out, (task_dir / 'stderr.txt').open('w') as err:
            child = subprocess.Popen(command, cwd=args.cwd or root, env=env,
                                     stdout=out, stderr=err, preexec_fn=limits,
                                     start_new_session=True)
            try:
                returncode = child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                returncode = child.wait()
        wall = time.monotonic() - start
        status = 'timeout' if timed_out else ('completed' if returncode == 0 else 'nonzero_exit')
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                             text=True, stderr=subprocess.DEVNULL).strip()
            dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                                cwd=root, text=True).strip())
        except (subprocess.SubprocessError, FileNotFoundError):
            commit, dirty = None, None
        row = {
            'task': args.task, 'category': args.category,
            'decision_changed_by_result': args.decision,
            'command': command, 'cwd': str(Path(args.cwd or root).resolve()),
            'started_utc': started, 'finished_utc': utc(),
            'wall_seconds': wall, 'timeout_seconds': args.timeout,
            'memory_limit_bytes': memory_bytes, 'memory_limit_kind': 'RLIMIT_AS',
            'cpu_affinity': [affinity], 'numeric_threads': 1,
            'python_optimize_environment': env['PYTHONOPTIMIZE'],
            'status': status, 'exit_code': returncode,
            'code_sha256': code_digest, 'code_files': code_files,
            'protocol_sha256': sha(protocol_path),
            'protocol_amendments': {p.name: sha(p) for p in sorted(root.glob('PROTOCOL_AMENDMENT_*.json'))},
            'input_sha256': protocol['attached_plan_sha256'],
            'input_sha256_scope': 'attached task specification only; exact fixture input digests are in the numerical reports',
            'git_commit': commit, 'tracked_files_dirty': dirty,
            'cost_scope': 'numerical_child_wall_including_imports_tests_and_checks',
            'child_process_policy': 'single numerical child; no nested workers',
            'cumulative_wall_seconds': used + wall,
            'remaining_seconds': cap - used - wall,
            'stdout': str((task_dir / 'stdout.txt').relative_to(root)),
            'stderr': str((task_dir / 'stderr.txt').relative_to(root)),
        }
        (task_dir / 'record.json').write_text(json.dumps(row, indent=2) + '\n')
        with ledger_path.open('a') as output:
            output.write(json.dumps(row, sort_keys=True) + '\n')
        print(json.dumps(row, indent=2))
        return returncode if not timed_out else 124


if __name__ == '__main__':
    sys.exit(main())
