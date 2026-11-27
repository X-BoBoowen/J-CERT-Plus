"""Verify the actual unpacked V4 code against its freeze; no performance rerun."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--code-root',type=Path,required=True);p.add_argument('--v3-code-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runtime=a.runtime.resolve();code=a.code_root.resolve();analysis=code/'v4/analysis'
    frozen=json.loads((code/'v4/protocols/HG_LIGHT_FROZEN.json').read_text())
    if a.output.exists():raise FileExistsError(a.output)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()
    if commit!=frozen['runtime_commit']:raise ValueError('fresh reconstructed commit mismatch')
    for rel,digest in frozen['source_files_sha256'].items():
        if sha(runtime/rel)!=digest:raise ValueError('fresh source mismatch '+rel)
    for rel,digest in frozen['analysis_files_sha256'].items():
        if sha(code/rel)!=digest:raise ValueError('unpacked frozen analysis mismatch '+rel)
    sys.path[:0]=[str(runtime),str(analysis)]
    os.environ['V4_ANALYSIS_ROOT']=str(analysis)
    os.environ['V3_ORACLE']=str(a.v3_code_root.resolve()/'v3/code/exact_oracle.py')
    imports={}
    for name in ('src.a2_session','src.a2_access','src.g_access','src.a2_checker','src.measurement_policy','src.v3_worker'):
        module=importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(runtime):raise ValueError('import escaped fresh runtime')
        imports[name]=str(Path(module.__file__).relative_to(runtime))
    suites=[]
    for folder,pattern in ((runtime/'tests','test_measurement_policy.py'),(runtime/'tests','test_v3_g.py'),
                           (analysis,'test_diagnostic_gate.py'),(analysis,'test_selection_rule.py')):
        suite=unittest.TestLoader().discover(str(folder),pattern=pattern)
        suites.append((pattern,suite))
    runner=unittest.TextTestRunner(verbosity=2)
    results=[]
    for pattern,suite in suites:
        result=runner.run(suite)
        results.append(dict(pattern=pattern,test_methods=result.testsRun,failures=len(result.failures),
                            errors=len(result.errors),skipped=len(result.skipped),passed=result.wasSuccessful()))
    report=dict(status='passed' if all(r['passed'] for r in results) else 'failed',
        runtime_commit=commit,runtime_source_members=len(frozen['source_files_sha256']),
        analysis_source_members=len(frozen['analysis_files_sha256']),fresh_imports=imports,
        tests=results,test_methods_total=sum(r['test_methods'] for r in results),
        source='actual V4_Code.zip extraction combined with the exact V2/V3 code archives and Git bundles',
        scope='imports and changed-path micro tests only; no formal worker or historical benchmark rerun; test-method count is not assertion count, independent-input count or real-data numerical replay count')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
