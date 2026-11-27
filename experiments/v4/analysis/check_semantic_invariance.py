"""Exact, path-scoped V4 semantic comparison. Unregistered fields are retained."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

REQUEST_EXCLUDED=('request_seconds','wrapper_seconds','online_seconds','reset_opinion_seconds',
    'phases','online_phases','index_build_phases','measurement_policy','memory_observation_schema',
    'diagnostic_deep_calls')
ACCESS_EXCLUDED=('seed_build_phases','frontier_build_phases')
MEMORY_SEMANTIC=('gram_count','peak_gram_count','root_factor_reported_nnz',
    'catalog_materialized','candidate_array_rows','full_root_caps_created_rows','m_sized_structures','access_proof')
MEMORY_RESOURCE_FIELDS=('explicit_catalog_bytes','structural_weight_grouping','incidence_bytes',
    'index_private_bytes','peak_frontier_owned_bytes','peak_emitted_batch_array_bytes','frontier_snapshot',
    'scope','sample','graph_csr','root_operator_csr','sparse_catalog','gram_python_bytes',
    'actual_candidate_cache_python_bytes','decision_and_access_log_python_bytes','shared_response_bytes',
    'root_factor_resident_bytes','low_rank_steps','peak_low_rank_steps','memory_observation_schema',
    'light_observations','low_rank_step_count','catalog_array_payloads','catalog_root_edge_key_items')
WORKER_EXCLUDED=('binding_and_import_seconds','opinion_and_graph_binding_seconds','process_wall_seconds',
    'structural_seconds','structural_phases','service_accounted_seconds','service_plus_input_binding_seconds',
    'complete_service_seconds','partial_work_seconds','prefixes','timing_scope','process_peak_rss_bytes',
    'resource_observation_scope','environment','threadpools','enforced_threadpools','freeze_sha256',
    'source_digest','actual_runtime_commit','declared_runtime_commit','repeat_id','measurement_policy',
    'memory_observation_schema','run_kind','diagnostic_worker_script_sha256')
JOB_EXCLUDED=('position','output_file','repeat_id','measurement_policy','kind')
METERING_EXCLUDED=('solve_seconds','factor_attempt_seconds')
SCHEMA=dict(id='v4-exact-semantics-1',integer_policy='arbitrary precision JSON integers; no floating normalization',
    request_excluded=list(REQUEST_EXCLUDED),access_event_excluded=list(ACCESS_EXCLUDED),
    request_memory_policy='only the listed resource fields are excluded; all other memory fields are preserved',
    memory_resource_fields=list(MEMORY_RESOURCE_FIELDS),
    memory_semantic_fields=list(MEMORY_SEMANTIC),worker_excluded=list(WORKER_EXCLUDED),
    job_excluded=list(JOB_EXCLUDED),metering_excluded=list(METERING_EXCLUDED),
    diagnostic_cache_policy='compare exactly for same-build diagnosis; unavailable in historical V3 and excluded only in historical mode',
    unknown_field_policy='retain and compare',environment_and_source_policy='validated separately against each frozen binding')


def request_semantics(request,*,historical=False):
    result=copy.deepcopy(request)
    memory=result.pop('memory',{})
    result['memory_work_semantics']={key:value for key,value in memory.items() if key not in MEMORY_RESOURCE_FIELDS}
    for key in REQUEST_EXCLUDED:result.pop(key,None)
    if historical:result.pop('diagnostic_cache_audit',None)
    for event in result.get('access',{}).get('epochs',[]):
        for key in ACCESS_EXCLUDED:event.pop(key,None)
    return result


def worker_semantics(worker,*,historical=False):
    result=copy.deepcopy(worker)
    for key in WORKER_EXCLUDED:result.pop(key,None)
    for key in JOB_EXCLUDED:result.get('job',{}).pop(key,None)
    for key in METERING_EXCLUDED:result.get('metering',{}).pop(key,None)
    result['requests']=[request_semantics(row,historical=historical) for row in worker['requests']]
    return result


def differences(left,right,path='$'):
    if type(left) is not type(right):
        return [dict(path=path,kind='type_or_missing',left=left,right=right)]
    if isinstance(left,dict):
        output=[]
        for key in sorted(set(left)|set(right)):
            if key not in left or key not in right:
                output.append(dict(path=path+'.'+key,kind='missing',left_present=key in left,right_present=key in right))
            else:output.extend(differences(left[key],right[key],path+'.'+key))
        return output
    if isinstance(left,list):
        if len(left)!=len(right):return [dict(path=path,kind='length',left=len(left),right=len(right))]
        return [row for i,(a,b) in enumerate(zip(left,right)) for row in differences(a,b,path+f'[{i}]')]
    return [] if left==right else [dict(path=path,kind='value',left=left,right=right)]


def compare_workers(left,right,*,historical=False):
    a=worker_semantics(left,historical=historical);b=worker_semantics(right,historical=historical)
    diff=differences(a,b)
    def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(equal=not diff,difference_count=len(diff),differences=diff,
                left_semantic_sha256=digest(a),right_semantic_sha256=digest(b),historical_mode=historical)


def main():
    p=argparse.ArgumentParser();p.add_argument('--left',type=Path,required=True)
    p.add_argument('--right',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--historical',action='store_true');a=p.parse_args()
    result=compare_workers(json.loads(a.left.read_text()),json.loads(a.right.read_text()),historical=a.historical)
    result.update(left=str(a.left),right=str(a.right),schema=SCHEMA)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'equal':result['equal'],'difference_count':result['difference_count']}))
    return 0 if result['equal'] else 1


if __name__=='__main__':raise SystemExit(main())
