"""Offline fixed-shortlist replay on each distinct E1 D request.

Reuses no measured numerical state. This is a fresh shortlist diagnostic,
separate from the independent full-catalog output checker.
"""
import json
from pathlib import Path
from time import perf_counter

from .analyze import shadow
from .common import ROOT, dump, file_hash


def run():
    folder=ROOT/'raw/guarantee_cost'
    output=ROOT/'raw/audit/e1_shadow'
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/'COVERAGE.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    start=perf_counter()
    schedule=json.loads((folder/'SCHEDULE_RESULTS.json').read_text())
    seen={}; coverage=[]; jobs=[]
    for row in schedule['jobs']:
        job=row['job']
        if job['method']!='D' or job['repeat_id']!=0:
            continue
        if row['status']!='completed':
            raise ValueError('A registered D worker did not complete')
        path=folder/row['result']
        d=json.loads(path.read_text()); positions=[]
        for q,r in enumerate(d['requests'],1):
            key=(d['graph'],d['weight_mode'],r['opinion_hash'])
            identity=(r['initial_seed_ids'],[e['winner_id'] for e in r['decisions']],r['status'])
            if key not in seen:
                target=f"{len(jobs):02d}_{d['graph']}_{d['weight_mode']}_{d['stream']}.json"
                seen[key]=(target,q,identity)
                positions.append(q)
            else:
                target,oldq,oldid=seen[key]
                if oldid!=identity:
                    raise ValueError('Repeated root request changed path or working set')
            target,oldq,_=seen[key]
            coverage.append(dict(source_file=row['result'],source_sha256=file_hash(path),
                query_position=q,opinion_hash=r['opinion_hash'],result=target,result_query_position=oldq))
        if positions:
            jobs.append((path,output/seen[(d['graph'],d['weight_mode'],d['requests'][positions[0]-1]['opinion_hash'])][0],positions))
    results=[]
    for source,target,positions in jobs:
        shadow(source,target,positions)
        d=json.loads(target.read_text())
        events=[e for r in d['requests'] for e in r['events']]
        results.append(dict(file=target.name,sha256=file_hash(target),queries=len(d['requests']),
            events=len(events),matched=sum(e.get('matches') is True for e in events),
            unresolved=sum(e.get('status')=='unresolved' for e in events)))
        dump(manifest,dict(status='running',coverage=coverage,results=results))
    dump(manifest,dict(status='completed',source_schedule_sha256=file_hash(folder/'SCHEDULE_RESULTS.json'),
        audit_source_sha256=file_hash(Path(__file__)),shadow_source_sha256=file_hash(ROOT/'src/analyze.py'),
        scope='Repeat-0 D parents; deduplicated by graph, weight and actual opinion; fixed root top64 freshly re-ranked and scored',
        independent_full_catalog_proof=False,coverage=coverage,results=results,
        unique_queries=len(seen),source_queries=len(coverage),elapsed_seconds=perf_counter()-start))


if __name__=='__main__':
    run()
