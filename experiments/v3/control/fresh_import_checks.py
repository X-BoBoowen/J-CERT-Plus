"""Verify delivered V3 code against an independently assembled V2+V3 tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--code-root', type=Path, required=True)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    runtime, code = args.runtime.resolve(), args.code_root.resolve()
    sys.path[:0] = [str(runtime), str(code / 'v3/code')]
    os.environ['V3_ORACLE'] = str(code / 'v3/code/exact_oracle.py')
    frozen = json.loads(args.freeze.read_text())
    observed = subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()
    if observed != frozen['runtime_commit']:
        raise ValueError('fresh repository commit mismatch')
    for relative, expected in frozen['source_files_sha256'].items():
        if hashlib.sha256((runtime/relative).read_bytes()).hexdigest() != expected:
            raise ValueError('fresh source/member mismatch: '+relative)
    from src.a2_session import A2Session
    from src.g_access import GAccess, PairBlocks
    from src.a2_checker import A2CatalogReplay
    suite = unittest.TestSuite()
    # These existing test directories are ordinary directories, not Python
    # packages. Give each its own loader/root instead of requiring __init__.py
    # or carrying the first discovery's cached top-level directory forward.
    suite.addTests(unittest.TestLoader().discover(str(code/'v3/code'),pattern='test_*.py',
                                                 top_level_dir=str(code/'v3/code')))
    suite.addTests(unittest.TestLoader().discover(str(runtime/'tests'),pattern='test_v3_g*.py',
                                                 top_level_dir=str(runtime/'tests')))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(status='passed' if result.wasSuccessful() else 'failed',
                  runtime_commit=observed, source_members_verified=len(frozen['source_files_sha256']),
                  tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), runtime=str(runtime), extracted_code_root=str(code),
                  scope='actual V3 archive + pristine verified V2 packages and git history; imports and relevant mathematical/G tests only; no performance batch replay')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
