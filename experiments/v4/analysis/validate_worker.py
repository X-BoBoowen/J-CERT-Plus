"""Positive binding and resource evidence; equality alone is not admission."""
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def binding_issues(worker,frozen,job,record,freeze_path):
    issues=[]
    expected={'job':job,'graph':job['graph'],'stream':job['stream'],'method':job['method'],
        'repeat_id':job['repeat_id'],'declared_runtime_commit':frozen['runtime_commit'],
        'actual_runtime_commit':frozen['runtime_commit'],'source_digest':frozen['source_digest'],
        'measurement_policy':job['measurement_policy'],'memory_observation_schema':job['measurement_policy'],
        'run_kind':frozen['run_kind'],'environment':frozen['environment'],
        'input_manifest_sha256':frozen['v2_input_manifest_sha256'],'freeze_sha256':sha(freeze_path)}
    for key,value in expected.items():
        if worker.get(key)!=value:issues.append('binding '+key)
    fixture=next(f for f in frozen['input_fixtures'] if (f['graph'],f['stream'])==(job['graph'],job['stream']))
    if worker.get('fixture')!=fixture:issues.append('fixture binding')
    graph=frozen['input_graphs'][job['graph']]
    for key,value in [('graph_hash',graph['weighted_sha256']),('n',graph['n']),('catalog_size',graph['catalog_size'])]:
        if worker.get(key)!=value:issues.append('graph '+key)
    if len(worker.get('requests',[]))!=4:issues.append('missing request outputs')
    for q,r in enumerate(worker.get('requests',[]),1):
        if r.get('query_position')!=q:issues.append(f'q{q} order')
        if q>4 or r.get('opinion_hash')!=fixture['opinion_hashes'][q-1]:issues.append(f'q{q} opinion')
        if r.get('measurement_policy')!=job['measurement_policy'] or r.get('memory_observation_schema')!=job['measurement_policy']:
            issues.append(f'q{q} policy')
        if r.get('budget')!=15 or r.get('delta')!=0:issues.append(f'q{q} contract')
    limits=frozen['resources']
    expected_record={'runtime_git_commit':frozen['runtime_commit'],'tracked_runtime_changes':'',
        'timeout_seconds':limits['seconds'],'memory_limit_bytes':limits['memory_bytes'],
        'memory_limit_kind':'RLIMIT_AS','cpu_affinity':[limits['cpu']],'numeric_threads':1,
        'status':'completed','exit_code':0}
    for key,value in expected_record.items():
        if record.get(key)!=value:issues.append('supervisor '+key)
    for rel,digest in record.get('runtime_files',{}).items():
        if frozen['source_files_sha256'].get(rel)!=digest:issues.append('supervisor source '+rel)
    if not record.get('runtime_files'):issues.append('missing supervisor source map')
    return issues
