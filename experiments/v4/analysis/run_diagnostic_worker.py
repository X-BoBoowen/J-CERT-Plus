"""Execute one frozen diagnostic worker with cache snapshots and call counts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--job',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    frozen=json.loads(a.freeze.read_text())
    if frozen['run_kind']!='invariance':raise ValueError('diagnostic wrapper prohibited in formal timing')
    for rel,digest in frozen['analysis_files_sha256'].items():
        if hashlib.sha256((a.freeze.resolve().parents[2]/rel).read_bytes()).hexdigest()!=digest:
            raise ValueError('diagnostic/analysis source changed: '+rel)
    sys.path.insert(0,str(a.runtime.resolve()))
    from diagnostic_support import ReportingProbe,diagnostic_session_type
    from src import a2_session,v3_worker
    original=a2_session.A2Session
    with ReportingProbe() as probe:
        a2_session.A2Session=diagnostic_session_type(original,probe)
        sys.argv=['v3_worker','--freeze',str(a.freeze),'--job',str(a.job),'--output',str(a.output)]
        try:return v3_worker.main()
        finally:a2_session.A2Session=original


if __name__=='__main__':raise SystemExit(main())
