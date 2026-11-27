"""One fixed synthetic resource/integration preflight, never a research result."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_info

from .a2_session import A2Session
from .a2_checker import A2CatalogReplay,worker_request


def main():
    n=128
    label='A2|resource-preflight|2026-11-25|synthetic-128'
    seed=int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],'big')
    rng=np.random.Generator(np.random.PCG64(seed))
    values=np.zeros((n,n),dtype=float)
    for i in range(n):
        for j in range(i+1,n):
            if j==i+1 or rng.random()<.025:
                values[i,j]=values[j,i]=float(rng.choice([-2.,-.5,.5,1.,1.5,3.]))
    W=sparse.csr_matrix(values)
    opinion=np.rint(65536*rng.uniform(-1,1,n)).astype(np.int64)
    rows=[]
    for method in ('D','S','H'):
        start=time.perf_counter()
        session=A2Session(W,method)
        request=session.request(opinion,budget=15)
        elapsed=time.perf_counter()-start
        assert request['status']=='budget_stop' and request['commits']==15,(method,request['status'])
        signature=[d['winner_id'] for d in request['decisions']]
        rows.append(dict(method=method,wall_seconds=elapsed,
            service_seconds=session.setup_seconds+request['request_seconds'],
            commits=request['commits'],winner_ids=signature,m=session.m,
            memory=request['memory'],request_counters=request['counters']))
    assert all(row['winner_ids']==rows[0]['winner_ids'] for row in rows)
    proof=worker_request(A2CatalogReplay(W),request,opinion,session.graph_hash)
    out=dict(scope='Fixed synthetic128-node preflight only; no full-graph outcome inspected or settings tuned',
        label=label,seed=seed,n=n,arms=rows,independent_H_full_request=proof,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_scope='single preflight process across three sequential arms and checker; not per-arm peaks',
        threadpools=threadpool_info())
    Path('raw/A2_SYNTHETIC_PREFLIGHT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(scope=out['scope'],process_peak_rss_bytes=out['process_peak_rss_bytes'],
                         methods=[dict(method=r['method'],commits=r['commits'],m=r['m']) for r in rows]),indent=2))


if __name__=='__main__':main()
