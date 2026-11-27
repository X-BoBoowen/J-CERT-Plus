"""One preregistered unseen batch and one gated scale attempt; no retries.

Numerical decisions remain in A2Session. This module only binds inputs,
records costs, enforces process limits and classifies complete/partial runs.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import secrets
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'v2'
GRAPHS = ('Alpha', 'OTC')
STREAMS = ('slow', 'independent', 'signed-neighbor')
METHODS = ('D', 'S', 'H')
THREAD_ENV = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
STATUS_MAP = {'budget_stop':'completed_budget', 'tolerance_stop':'certified_stop',
              'unresolved':'unresolved', 'memory_error':'memory_limit_exceeded',
              'error':'runtime/certificate_error'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value, *, exclusive=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if exclusive:
        with path.open('x') as handle: handle.write(raw)
    else:
        temp=path.with_suffix(path.suffix+'.tmp')
        temp.write_text(raw); os.replace(temp,path)


def small_jobs():
    result=[]
    for repeat in range(3):
        for cell,(graph,stream) in enumerate((g,s) for g in GRAPHS for s in STREAMS):
            pivot=(repeat+cell)%3
            for method in METHODS[pivot:]+METHODS[:pivot]:
                result.append(dict(graph=graph,stream=stream,repeat_id=repeat,method=method,
                                   stage='A',requests=4,budget=15))
    return result


def derive_seed(master,graph,stream,used):
    counter=0
    while True:
        label=f'J-CERT+|V2|2026-11-26|{master}|{graph}|{stream}|{counter}'
        seed=int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],'big')
        if seed not in used:
            return dict(seed=seed,collision_counter=counter,derivation_label=label)
        counter+=1


def completed(result):
    return result.get('status') in ('completed_budget','certified_stop')


def memory_failure(reason):
    return any(word in str(reason).lower() for word in
       ('out of memory','not enough memory','cannot allocate memory','malloc fails',
        'superlu_malloc','insufficient memory','allocation failed','memoryerror'))


def validate_job(job,protocol):
    stage=job.get('stage');method=job.get('method');repeat=job.get('repeat_id')
    if job.get('budget')!=15 or type(repeat) is not int or not 0<=repeat<3:
        raise ValueError('unregistered budget or repeat')
    if stage=='A':
        if job.get('graph') not in GRAPHS or job.get('stream') not in STREAMS or method not in METHODS:
            raise ValueError('unregistered A cell')
        requests,seconds,memory=4,protocol['small']['worker_seconds'],protocol['small']['memory_bytes']
    elif stage in ('B1','B2_Q1','B3_Q4'):
        if job.get('graph')!='Slashdot' or job.get('stream')!='independent' or method not in ('H','S'):
            raise ValueError('unregistered scale cell')
        if stage!='B3_Q4' and repeat!=0:raise ValueError('unregistered pilot repeat')
        if stage=='B1' and method!='H':raise ValueError('unregistered root profile')
        requests={'B1':0,'B2_Q1':1,'B3_Q4':4}[stage]
        seconds=protocol['scale'][{'B1':'root_profile_seconds','B2_Q1':'q1_seconds','B3_Q4':'q4_seconds'}[stage]]
        memory=protocol['scale']['memory_bytes']
    else:raise ValueError('unregistered stage')
    if any(type(job.get(key)) is not int or job[key]!=value for key,value in
           [('requests',requests),('seconds',seconds),('memory_bytes',memory)]):
        raise ValueError('request count or resource limit differs from freeze')


def explicit_preflight(m,limit_bytes):
    payload=17*m
    return dict(status='not_started_memory_preflight' if payload>limit_bytes else
                'not_run_gate_failed' if m>=2**31 else 'admitted',
                minimum_pool_and_weights_bytes=payload,
                mandatory_arrays={'i_int32':4*m,'j_int32':4*m,'flip_bool':m,'weight_int64':8*m},
                memory_limit_bytes=limit_bytes,ordinal_capacity_admitted=m<2**31,
                required_ordinal_bound='M < 2^31',
                scope='this exact explicit representation, not an information-theoretic bound')


def sources():
    from .a2_bench import source_hashes, input_source_hashes
    result=source_hashes()|input_source_hashes()
    for name in ('v2_runner','exact_ties','a2_checker','a2_tie_verify_retest'):
        result[f'src/{name}.py']=sha(ROOT/f'src/{name}.py')
    return result


def used_seeds():
    found=set(); inputs=[]
    def walk(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('seed','seeds','main_seed'):
                    for val in item if isinstance(item,list) else [item]:
                        if type(val) is int:found.add(val)
                walk(item)
        elif isinstance(value,list):
            for item in value:walk(item)
    paths=list((ROOT/'protocol').glob('*.json'))
    paths+=list((ROOT/'baseline_r15/experimental_records/evidence_supplement_20261124/protocol').glob('*.json'))
    for path in sorted(paths):
        try: walk(json.loads(path.read_text()))
        except (ValueError,UnicodeError):continue
        inputs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
    # Additional explicit fixed seeds used by archived legacy input generators.
    found.update((17,104729,196613))
    return sorted(found),inputs


def prepare():
    if V2.exists():raise FileExistsError('a V2 freeze already exists; no regeneration')
    for key in THREAD_ENV:os.environ[key]='1'
    import psutil
    from .a2_bench import environment, _graph_record
    code=dict(created_utc=now(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              source_files_sha256=sources(),environment=environment(),
              tie_runtime_baseline='531522d09b962286afdcde9038ad1a1b0b5d90fa',
              old_retest_input_binding='9de16a76f5cddcdd4aba0c82d555f2ed42e46fe2')
    save(V2/'CODE_FREEZE.json',code,exclusive=True)
    available=psutil.virtual_memory().available
    if available<8*1024**3:raise RuntimeError('fixed 4 GiB scale cap would exceed half available memory')
    rules=dict(created_utc=now(),cpu=0,supervisor_cpu=8,numerical_threads=1,
       small=dict(worker_seconds=180,memory_bytes=2*1024**3,batch_seconds=10800),
       scale=dict(root_profile_seconds=600,q1_seconds=600,q4_seconds=1800,repeats=3,
                  memory_bytes=4*1024**3,total_seconds=21600,independent_check_seconds=1800),
       memory_policy='hard RLIMIT_AS for virtual address space before numerical imports; RSS measured separately',
       host_available_memory_before_freeze=available,host_total_memory=psutil.virtual_memory().total,
       backend='unchanged scipy.sparse.linalg.splu binary64 direct LU with default existing options, common to D/S/H',
       constants=dict(delta=0,hot_size=64,budget=15,operator_scale=16,proof_unit=2**48,
                      screen_unit=2**20,opinion_denominator=65536,insertion_weight='1/16',
                      rounding='numpy.rint ties-to-even; Fraction floor/ceil outward for certificates',
                      child_refresh_residual_limit='max(2^28, max(diagonal(A_child))+1)',
                      child_woodbury_abs_bound=1.5,response_abs_bound=2,
                      overlap='current-version exact single-transposition witness; otherwise unresolved for n>16',
                      numeric_precision='binary64 solve, int64 checked residuals, exact rational interval primitives',
                      cg_rtol=None,cg_atol=None),
       independent_selection=[dict(graph=g,stream='independent',method='H',repeat_id=0,query_position=1)
                              for g in GRAPHS],
       scale_selection='H/repeat0/q1 first decision and last accepted decision, deduplicated; 1800s total',
       comparison='all 54 workers retained; no speed threshold; no retries; 6 independent streams/24 correlated requests',
       reset='fixed root factor per worker; empty request Gram and answer caches',
       scale_order=['B0','B1 H root/state/frontier','B2 H Q1','B2 S Q1','B3 rotated admitted H/S Q4 x3'],
       scale_stream_profile_blocks=0,
       measurement='full graph load/setup + requests; process wall and checker separate; exclusive internal phases')
    save(V2/'PRE_REGISTRATION.json',rules,exclusive=True)
    used,seed_sources=used_seeds()
    master=secrets.token_hex(32)
    draw=dict(created_utc=now(),master_seed_hex=master,source='one secrets.token_hex(32) draw after CODE_FREEZE',
              prior_seed_integers=used,prior_seed_sources=seed_sources,
              collision_policy='increment per-cell derivation counter only for an already registered integer seed')
    save(V2/'SEED_DRAW.json',draw,exclusive=True)
    import numpy as np
    from scipy import sparse
    from .common import opinions
    from .confirmation_replay import reference_stream
    from .holdout_stream import signed_neighbor_opinions
    from .access_bench import matrix_hashes
    graphs={}; fixtures=[]; units={}
    for graph in GRAPHS:
        graphs[graph],units[graph]=_graph_record(graph)
    slash=ROOT/'baseline_r15/experimental_records/legacy_v6/datasets/Slashdot.npz'
    expected='78d004ff0f3cab226ef2190244a3f59675337adcbd7c445fa98a343964ba86b0'
    if sha(slash)!=expected:raise ValueError('Slashdot archive digest mismatch')
    sw=sparse.load_npz(slash); n=sw.shape[0]
    negative=int(np.count_nonzero(sw.data<0))//2
    positive=int(np.count_nonzero(sw.data>0))//2
    if (n,sw.nnz,negative,positive)!=(82140,997064,117599,380933):raise ValueError('Slashdot dimensions mismatch')
    units['Slashdot']=sw
    graphs['Slashdot']=dict(n=n,weighted_path=str(slash.relative_to(ROOT)),weighted_sha256=expected,
       unit_path=str(slash.relative_to(ROOT)),unit_sha256=expected,root_positive_edges=positive,
       root_negative_edges=negative,root_nonedges=n*(n-1)//2-sw.nnz//2,catalog_size=n*(n-1)//2-positive,
       input_scope='archived undirected signed projection with unit magnitudes')
    for graph,stream in [(g,s) for g in GRAPHS for s in STREAMS]+[('Slashdot','independent')]:
        seed=derive_seed(master,graph,stream,set(used));used.append(seed['seed'])
        mat=np.stack(signed_neighbor_opinions(units[graph],4,seed['seed']) if stream=='signed-neighbor'
                     else opinions(graphs[graph]['n'],stream,seed['seed'],4))
        if not np.array_equal(mat,reference_stream(units[graph],stream,seed['seed'],4)):
            raise ValueError('independent input generator disagrees')
        matrix_hash,row_hashes=matrix_hashes(mat,(4,graphs[graph]['n']))
        path=V2/'inputs'/f'{graph}_{stream}.npz';path.parent.mkdir(exist_ok=True)
        np.savez(path,opinions=mat)
        fixtures.append(dict(graph=graph,stream=stream,**seed,fixture_id=f'{graph}/{stream}/V2/Q4',
            file=str(path.relative_to(V2)),npz_sha256=sha(path),matrix_sha256=matrix_hash,
            opinion_hashes=row_hashes,shape=list(mat.shape),independently_matched_rows=4,
            min_code=int(mat.min()),max_code=int(mat.max())))
    if sources()!=code['source_files_sha256']:raise ValueError('code changed during input freeze')
    save(V2/'A_JOBS.json',small_jobs(),exclusive=True)
    protocol=rules|dict(id='J-CERT+-V2-unseen-and-Slashdot-20261126',stage='unseen_confirmation_and_scale',
                       code_freeze_sha256=sha(V2/'CODE_FREEZE.json'),seed_draw_sha256=sha(V2/'SEED_DRAW.json'),
                       a_jobs_sha256=sha(V2/'A_JOBS.json'))
    save(V2/'PROTOCOL.json',protocol,exclusive=True)
    manifest=dict(created_utc=now(),protocol_sha256=sha(V2/'PROTOCOL.json'),code=code,
                  graphs=graphs,fixtures=fixtures,verified=True,small_streams=6,small_opinions=24,
                  slashdot_streams=1,slashdot_opinions=4)
    save(V2/'INPUTS.json',manifest,exclusive=True)
    return manifest


def load_freeze():
    from .a2_bench import environment
    protocol=json.loads((V2/'PROTOCOL.json').read_text())
    manifest=json.loads((V2/'INPUTS.json').read_text())
    if manifest['protocol_sha256']!=sha(V2/'PROTOCOL.json'):raise ValueError('protocol changed')
    for filename,key in [('CODE_FREEZE.json','code_freeze_sha256'),('SEED_DRAW.json','seed_draw_sha256'),
                         ('A_JOBS.json','a_jobs_sha256')]:
        if sha(V2/filename)!=protocol[key]:raise ValueError(filename+' changed')
    if json.loads((V2/'A_JOBS.json').read_text())!=small_jobs():raise ValueError('registered roster changed')
    if json.loads((V2/'CODE_FREEZE.json').read_text())!=manifest['code']:raise ValueError('code freeze identity changed')
    if manifest['code']['environment']!=environment():raise ValueError('runtime environment changed')
    if manifest['code']['source_files_sha256']!=sources():raise ValueError('runtime changed after freeze')
    expected=[(g,s) for g in GRAPHS for s in STREAMS]+[('Slashdot','independent')]
    if [(r['graph'],r['stream']) for r in manifest['fixtures']]!=expected:raise ValueError('fixture grid changed')
    draw=json.loads((V2/'SEED_DRAW.json').read_text());used=set(draw['prior_seed_integers'])
    for row in manifest['fixtures']:
        seed=derive_seed(draw['master_seed_hex'],row['graph'],row['stream'],used)
        if any(row[key]!=value for key,value in seed.items()):raise ValueError('seed derivation changed')
        used.add(seed['seed'])
        if row['shape']!=[4,manifest['graphs'][row['graph']]['n']]:raise ValueError('fixture shape changed')
    return protocol,manifest


def worker(job_path,output):
    job=json.loads(Path(job_path).read_text());output=Path(output)
    if output.exists():raise FileExistsError(output)
    validate_job(job,json.loads((V2/'PROTOCOL.json').read_text()))
    for key in THREAD_ENV:os.environ[key]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    resource.setrlimit(resource.RLIMIT_AS,(job['memory_bytes'],job['memory_bytes']))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    os.sched_setaffinity(0,{0})
    started=time.monotonic()
    trace=job|dict(status='starting',requests=[],prefixes={},active_phase='bindings',
                   started_utc=now(),hard_limit=resource.getrlimit(resource.RLIMIT_AS)[0],
                   affinity=sorted(os.sched_getaffinity(0)),metering=dict(solve_calls=0,scalar_rhs=0,solve_seconds=0.))
    def checkpoint():
        trace['worker_wall_seconds']=time.monotonic()-started
        trace['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        save(output,trace)
    checkpoint()
    try:
        protocol,manifest=load_freeze()
        if job.get('frozen_manifest_sha256')!=sha(V2/'INPUTS.json'):raise ValueError('worker manifest differs from supervisor freeze')
        trace.update(protocol_sha256=manifest['protocol_sha256'],input_manifest_sha256=sha(V2/'INPUTS.json'),
                     source_files_sha256=manifest['code']['source_files_sha256'])
        from scipy.sparse import load_npz
        from threadpoolctl import threadpool_info, threadpool_limits
        import numpy as np
        from .a2_session import A2Session,make_ledger,lp
        from .common import UNIT
        from .a2_bench import _check_seconds
        from .access_bench import load_fixture
        spec=manifest['graphs'][job['graph']]
        fixture=next(r for r in manifest['fixtures'] if (r['graph'],r['stream'])==(job['graph'],job['stream']))
        matrix=load_fixture(fixture,V2)
        graph_path=ROOT/spec['weighted_path']
        if sha(graph_path)!=spec['weighted_sha256']:raise ValueError('graph digest changed')
        trace.update(fixture=fixture,graph_hash=spec['weighted_sha256'],threadpools=threadpool_info(),
                     n=spec['n'],catalog_size=spec['catalog_size'])
        original_factor=lp.UpdatedFactor
        class MeteredFactor(original_factor):
            def __init__(self,A):
                trace['active_phase']='factor_or_precond_init';checkpoint()
                start=time.perf_counter()
                try:super().__init__(A)
                finally:
                    trace['metering']['factor_attempt_seconds']=time.perf_counter()-start
                    checkpoint()
            def solve(self,rhs):
                trace['metering']['solve_calls']+=1
                trace['metering']['scalar_rhs']+=1 if rhs.ndim==1 else rhs.shape[1]
                start=time.perf_counter()
                try:return super().solve(rhs)
                finally:trace['metering']['solve_seconds']+=time.perf_counter()-start
        lp.UpdatedFactor=MeteredFactor
        with threadpool_limits(limits=1):
            trace['active_phase']='graph_load';checkpoint()
            start=time.perf_counter();W=load_npz(graph_path)
            graph_load=time.perf_counter()-start
            trace['active_phase']='catalog_operator_access_setup';checkpoint()
            service_start=time.perf_counter()
            session=A2Session(W,job['method'],hot_size=64,timers=True,
                              graph_hash=spec['weighted_sha256'],graph_name=job['graph'])
            structural=graph_load+time.perf_counter()-service_start
            phases=dict(session.setup_phases);phases['graph_load']=graph_load
            phases['other']+=structural-sum(phases.values())
            _check_seconds(phases,structural,'V2 structural')
            if session.n!=spec['n'] or session.m!=spec['catalog_size']:raise ValueError('catalog mismatch')
            trace.update(structural_seconds=structural,structural_phases=phases,
               root_version_id=session.root_id,operator_hash=session.operator_hash,catalog_hash=session.catalog_hash,
               service_accounted_seconds=structural,root_factor_nnz=session.factor.base.nnz,
               status='prepared',active_phase='root_ready')
            del W;checkpoint()
            if job['stage']=='B1':
                trace['active_phase']='state_solve_and_cert';checkpoint()
                start=time.perf_counter();rows=[]
                state=lp.certified_state(session.A,session.factor,matrix[0],UNIT,certified_rows=rows)
                trace['profile_state_seconds']=time.perf_counter()-start
                trace['active_phase']='frontier_prepare_and_paid_seeds';checkpoint()
                ledger=make_ledger(True);seeds=session.access.begin(*state[3:],64,ledger)
                frontier_seconds,frontier_phases=ledger.finish()
                trace.update(profile_frontier_seconds=frontier_seconds,
                   profile_frontier_phases=frontier_phases,profile_seed_ids=seeds.tolist(),
                   profile_access_memory=session.access.memory(),
                   service_accounted_seconds=structural+trace['profile_state_seconds']+frontier_seconds,
                   status='profile_completed',active_phase='complete')
                checkpoint();return trace
            cumulative=structural
            for q,opinion in enumerate(matrix[:job['requests']],1):
                trace['active_phase']=f'request_{q}';checkpoint()
                start=time.perf_counter();request=session.request(opinion,budget=15,delta=0,audit=False)
                elapsed=time.perf_counter()-start
                _check_seconds(request['phases'],request['request_seconds'],'V2 internal request')
                overhead=elapsed-request['request_seconds']
                request['phases']['reset_restore']+=overhead;request['reset_opinion_seconds']+=overhead
                request.update(request_seconds=elapsed,query_position=q,wrapper_seconds=overhead,
                               result_status=STATUS_MAP.get(request['status'],'runtime/certificate_error'))
                if request['status']=='error' and memory_failure(request.get('failure_reason')):
                    request['result_status']='memory_limit_exceeded'
                if request['opinion_hash']!=fixture['opinion_hashes'][q-1]:raise ValueError('opinion identity mismatch')
                _check_seconds(request['phases'],elapsed,'V2 full request')
                trace['requests'].append(request);cumulative+=elapsed
                trace['prefixes'][str(q)]=cumulative;trace['service_accounted_seconds']=cumulative
                trace['status']='in_progress';checkpoint()
            states=[r['result_status'] for r in trace['requests']]
            trace['status']=('completed_budget' if all(x=='completed_budget' for x in states) else
                'certified_stop' if all(x in ('completed_budget','certified_stop') for x in states) else
                next(x for x in states if x not in ('completed_budget','certified_stop')))
            trace['active_phase']='complete'
    except Exception as error:
        msg=repr(error)
        memory=isinstance(error,MemoryError) or memory_failure(msg)
        trace.update(status='memory_limit_exceeded' if memory else 'runtime/certificate_error',
                     failure_reason=msg,traceback=traceback.format_exc())
    finally:checkpoint()
    return trace


def run_job(job,directory,position,deadline):
    import psutil
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if time.monotonic()>=deadline:return dict(job=job,status='not_run_gate_failed',reason='fixed total time limit')
    job=job|dict(frozen_manifest_sha256=sha(V2/'INPUTS.json'))
    label=f"{position:03d}_{job['graph']}_{job['stream']}_{job['method']}_r{job.get('repeat_id',0)}"
    jp=directory/(label+'.job.json');out=directory/(label+'.json');log=directory/(label+'.log')
    save(jp,job,exclusive=True)
    if out.exists() or log.exists():raise FileExistsError('refusing a repeat')
    start=time.monotonic();peak=0;peak_vms=0;stop=None
    env=os.environ.copy();env.update({k:'1' for k in THREAD_ENV});env['PYTHONDONTWRITEBYTECODE']='1'
    with log.open('x') as handle:
        child=subprocess.Popen([sys.executable,'-m','src.v2_runner','worker','--job',str(jp),
                                '--output',str(out)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        observed=psutil.Process(child.pid)
        while child.poll() is None:
            try:
                mem=observed.memory_info();peak=max(peak,mem.rss);peak_vms=max(peak_vms,mem.vms)
            except psutil.NoSuchProcess:break
            elapsed=time.monotonic()-start
            if time.monotonic()>=deadline or elapsed>=job['seconds']:stop='timeout'
            if mem.rss>job['memory_bytes']:stop='memory_limit_exceeded'
            if stop:child.kill();break
            time.sleep(.025)
        returncode=child.wait()
    result=json.loads(out.read_text()) if out.exists() else dict(requests=[],status='runtime/certificate_error')
    if stop:result.update(last_worker_status=result['status'],status=stop)
    elif returncode and not completed(result) and result.get('status')!='profile_completed':
        stderr=log.read_text(errors='replace').lower()
        memory=memory_failure(stderr)
        result.update(last_worker_status=result['status'],status='memory_limit_exceeded' if memory else 'runtime/certificate_error')
    result.update(process_wall_seconds=time.monotonic()-start,returncode=returncode,
                  external_sampled_peak_rss_bytes=peak,external_sampled_peak_vms_bytes=peak_vms)
    if result.get('status') not in ('completed_budget','certified_stop','profile_completed'):
        result['complete_service_seconds']=None
    else:result['complete_service_seconds']=result.get('service_accounted_seconds')
    save(out,result)
    print(label,result['status'],f"{result['process_wall_seconds']:.3f}s",flush=True)
    return dict(file=str(out.relative_to(ROOT)),status=result['status'],returncode=returncode,
                process_wall_seconds=result['process_wall_seconds'],completed_requests=len(result.get('requests',[])))


def run_a():
    protocol,manifest=load_freeze();directory=V2/'raw_a'
    if directory.exists():raise FileExistsError('batch has already been started')
    directory.mkdir();start=time.monotonic();deadline=start+protocol['small']['batch_seconds']
    rows=[];jobs=json.loads((V2/'A_JOBS.json').read_text())
    for pos,job in enumerate(jobs):
        job=job|dict(seconds=protocol['small']['worker_seconds'],memory_bytes=protocol['small']['memory_bytes'])
        if time.monotonic()>=deadline:
            rows.append(dict(job=job,status='not_run_gate_failed',reason='predeclared total batch limit'))
        else:rows.append(run_job(job,directory,pos,deadline))
        save(directory/'SCHEDULE_RESULTS.json',dict(jobs=rows,registered_jobs=54,started_utc=now(),
              batch_wall_seconds=time.monotonic()-start))
    return rows


def run_b():
    protocol,manifest=load_freeze();directory=V2/'raw_b'
    if directory.exists():raise FileExistsError('scale branch has already been started')
    directory.mkdir();start=time.monotonic();cfg=protocol['scale'];deadline=start+cfg['total_seconds']
    rows=[];base=dict(graph='Slashdot',stream='independent',repeat_id=0,budget=15,memory_bytes=cfg['memory_bytes'])
    pre=explicit_preflight(manifest['graphs']['Slashdot']['catalog_size'],cfg['memory_bytes'])
    save(directory/'D_PREFLIGHT.json',pre,exclusive=True)
    def record():save(directory/'STAGE_RESULTS.json',dict(stages=rows,D=pre,branch_wall_seconds=time.monotonic()-start))
    b1=run_job(base|dict(method='H',stage='B1',requests=0,seconds=cfg['root_profile_seconds']),directory,0,deadline)
    rows.append(b1);record()
    if b1['status']!='profile_completed':
        rows.extend(dict(stage=stage,method=method,repeat_id=repeat,status='not_run_gate_failed',reason='B1 root preparation did not complete')
                    for stage,repeats in (('B2_Q1',range(1)),('B3_Q4',range(3))) for repeat in repeats for method in ('H','S'))
        record();return rows
    admitted=[]
    for pos,method in enumerate(('H','S'),1):
        row=run_job(base|dict(method=method,stage='B2_Q1',requests=1,seconds=cfg['q1_seconds']),directory,pos,deadline)
        rows.append(row);record()
        if completed(row):admitted.append(method)
        elif method=='H':
            rows.append(dict(stage='B2_Q1',method='S',repeat_id=0,status='not_run_gate_failed',reason='H Q1 did not complete'))
            rows.extend(dict(stage='B3_Q4',method=m,repeat_id=r,status='not_run_gate_failed',reason='H Q1 did not complete')
                        for r in range(3) for m in ('H','S'))
            record();return rows
    for method in ('H','S'):
        if method not in admitted:
            rows.extend(dict(stage='B3_Q4',method=method,repeat_id=r,status='not_run_gate_failed',reason='method Q1 did not complete') for r in range(3))
    pos=3
    for repeat in range(3):
        order=admitted[repeat%len(admitted):]+admitted[:repeat%len(admitted)]
        for method in order:
            row=run_job(base|dict(method=method,stage='B3_Q4',requests=4,repeat_id=repeat,seconds=cfg['q4_seconds']),directory,pos,deadline)
            rows.append(row);pos+=1;record()
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','run-a','run-b','worker'))
    parser.add_argument('--job',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='worker':worker(args.job,args.output);return
    if args.mode=='prepare':
        value=prepare();print('Frozen',len(value['fixtures']),'streams; no performance runs started');return
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/.measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.sched_setaffinity(0,{8})
        (run_a if args.mode=='run-a' else run_b)()


if __name__=='__main__':main()
