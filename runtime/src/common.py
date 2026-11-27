"""Shared, deterministic stream inputs and timing boundaries."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'baseline_r15/experimental_records/evidence_supplement_20261124'
spec = importlib.util.spec_from_file_location('r15_diagnostic_common', EVIDENCE/'diagnostic/common.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
PHASES = ('graph_load','catalog_build','operator_assembly','factor_or_precond_init',
          'state_solve_and_cert','caps_build','caps_sort','posting/index_build','reset_restore',
          'opinion_refresh','discovery','response_acquisition','transport_local',
          'comparison/refinement','cache_management','other')
SCALE = 16
UNIT = 1 << 48
DEN = 65536


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(memoryview(np.ascontiguousarray(a)).cast('B'))
    return h.hexdigest()


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,default=legacy.json_default,allow_nan=False)+'\n')
    os.replace(tmp,path)


def opinions(n, stream, seed, count):
    if n<1 or count<1 or stream not in ('same','slow','independent'):
        raise ValueError('invalid stream')
    rng=np.random.Generator(np.random.PCG64(seed))
    result=[]
    for i in range(count):
        if i==0 or stream=='independent':
            s=np.rint(DEN*rng.uniform(-1,1,n)).astype(np.int64)
        elif stream=='slow':
            s=s.copy()
            size=(n+9)//10
            j=rng.choice(n,size=size,replace=False)
            s[j]=np.rint(.9*s[j]+.1*DEN*rng.uniform(-1,1,size)).astype(np.int64)
        result.append(s)
    return result


class Ledger:
    """Exclusive nested phases; inclusive outer wall covers untimed orchestration."""
    def __init__(self, enabled=True):
        self.enabled=enabled
        self.times=dict.fromkeys(PHASES,0.)
        self.child=0.
        self.started=perf_counter()

    @contextmanager
    def phase(self,name):
        if not self.enabled:
            yield
            return
        start=perf_counter()
        outer=self.child
        self.child=0.
        try:
            yield
        finally:
            elapsed=perf_counter()-start
            self.times[name]+=elapsed-self.child
            self.child=outer+elapsed

    def finish(self):
        elapsed=perf_counter()-self.started
        residual=elapsed-sum(self.times.values())
        if residual < -1e-9:
            raise RuntimeError('overlapping timer buckets')
        self.times['other']+=residual
        return elapsed,self.times
