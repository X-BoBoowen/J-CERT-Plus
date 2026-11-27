#!/usr/bin/env python3
"""Small existing structural and numerical test cases, not a timing experiment."""
from pathlib import Path
import os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',V3_ORACLE=str(ROOT/'experiments/v3/code/exact_oracle.py'),V4_ANALYSIS_ROOT=str(ROOT/'experiments/v4/analysis'))
for test in ['test_a2_catalog.py','test_v3_g.py','test_measurement_policy.py']:
    subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p',test,'-v'],cwd=ROOT/'runtime',env=env,check=True,timeout=120)
