"""Pre-registered A1 service worker and serial batch. See protocol/A1_SERVICE_PROTOCOL.json."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'protocol/A1_SERVICE_PROTOCOL.json'
ARMS=('rebuild-always','retain-always','adaptive')
SOURCE_FILES=('anchor_session','anchor_service_bench','session','weighted_adapter','common','checker')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    def native(v):
        return v.item() if hasattr(v,'item') else str(v)
    tmp.write_text(json.dumps(obj,allow_nan=False,default=native)+'\n')
    tmp.replace(path)


def identities():
    return {x:sha(ROOT/f'src/{x}.py') for x in SOURCE_FILES}


def validate(sources,development):
    w=ROOT/'protocol/WEIGHTED_FD_ADMISSION_v1.json'
    core=json.loads(w.read_text())
    frozen=json.loads((ROOT/'protocol/PILOT_SOURCE_FREEZE.json').read_text())
    if (core!=frozen['gate'] or any(sources[k]!=frozen['sha256']['src/'+k+'.py']
            for k in ('session','common','weighted_adapter','checker'))):
        raise ValueError('weighted numerical core differs from the pilot source freeze')
    if not (core.get('admitted') and 'D' in core['methods'] and
            core['session_sha256']==sources['session'] and
            core['weighted_adapter_sha256']==sources['weighted_adapter']):
        raise ValueError('weighted core admission does not match this source')
    if development:
        return {'core':sha(w),'a1':'pre-admission correctness only'}
    f=ROOT/'reports/A1_SERVICE_ADMISSION.json'
    gate=json.loads(f.read_text())
    if (gate.get('admitted') is not True or gate.get('source_files_sha256')!=sources or
            set(gate.get('arms',()))!=set(ARMS) or gate.get('protocol_sha256')!=sha(PROTOCOL)):
        raise ValueError('A1 service gate lacks matching sources, arms or protocol')
    evidence=gate.get('coverage',{})
    if not evidence or any(sha(ROOT/path)!=expected for path,expected in evidence.items()):
        raise ValueError('A1 replay evidence missing or changed')
    return {'core':sha(w),'a1':sha(f)}


def work(a):
    start=time.monotonic()
    if a.output.exists():
        raise FileExistsError(a.output)
    os.sched_setaffinity(0,{a.cpu})
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                'BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key]='1'
    protocol=json.loads(PROTOCOL.read_text())
    sources=identities()
    obj=dict(protocol_id=protocol['id'],protocol_sha256=sha(PROTOCOL),source_files_sha256=sources,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        kernel_build_id=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
        graph=a.graph,weight_mode='weighted',stream=a.stream,seed=a.seed,method='D',arm=a.arm,
        stream_id=f'{a.graph}/weighted/{a.stream}/{a.seed}',repeat_id=a.repeat_id,
        schedule_position=a.schedule_position,requests_planned=a.requests,budget=a.budget,delta=0,
        hot_size=64,requests=[],prefixes={},status='starting',failure_reason=None,
        cpu=a.cpu,affinity=sorted(os.sched_getaffinity(0)),audit=a.correctness,
        scope='pre-admission correctness only' if a.correctness else 'timed A1 service')
    def checkpoint():
        obj['worker_observed_wall_seconds']=time.monotonic()-start
        obj['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        save(a.output,obj)
    try:
        obj['admission']=validate(sources,a.correctness)
        if not a.correctness and (a.requests,a.budget,a.seed)!=(4,15,104729):
            raise ValueError('unregistered timed request')
        from .anchor_session import AnchorSession
        from .common import Ledger,opinions
        from scipy.sparse import load_npz
        from threadpoolctl import threadpool_info,threadpool_limits
        with threadpool_limits(limits=1):
            obj['threadpools']=threadpool_info()
            ledger=Ledger(True)
            path=ROOT/f'data/projected/{a.graph}_weighted.npz'
            with ledger.phase('graph_load'):
                graph_hash=sha(path)
                meta=json.loads((ROOT/f'data/projected/{a.graph}_metadata.json').read_text())
                if graph_hash!=meta['weighted_sha256']:
                    raise ValueError('registered projected graph mismatch')
                W=load_npz(path)
            sess=AnchorSession(W,a.arm,graph_hash=graph_hash,ledger=ledger)
            del W
            obj.update(graph_hash=graph_hash,catalog_rule='root lexical two-hop nonedges + negative flips',
                operator_hash=sess.operator_hash,catalog_hash=sess.catalog_hash,root_version_id=sess.root_id,
                n=sess.W.shape[0],catalog_size=sess.m,structural_seconds=sess.setup_seconds,
                structural_phases=sess.setup_phases,status='prepared')
            t=time.monotonic()
            inputs=opinions(sess.W.shape[0],a.stream,a.seed,a.requests)
            obj['stream_fixture_generation_seconds']=time.monotonic()-t
            cumulative=sess.setup_seconds
            checkpoint()
            for idx,s in enumerate(inputs,1):
                t=time.monotonic()
                req=sess.request(s,budget=a.budget,audit=a.correctness)
                elapsed=time.monotonic()-t
                req['phases']['reset_restore']+=elapsed-req['request_seconds']
                req['reset_opinion_seconds']+=elapsed-req['request_seconds']
                req['request_seconds']=elapsed
                req.update(query_position=idx,stream_id=obj['stream_id'])
                if abs(sum(req['phases'].values())-elapsed)>1e-7 or abs(
                        req['reset_opinion_seconds']+req['online_seconds']-elapsed)>1e-7:
                    raise ValueError('request accounting mismatch')
                cumulative+=elapsed
                obj['requests'].append(req)
                if idx in (1,2,4) or idx==a.requests:
                    obj['prefixes'][str(idx)]=cumulative
                obj['status']='in_progress'
                checkpoint()
            obj['status']=('completed' if all(r['status'] in ('budget_stop','tolerance_stop')
                             for r in obj['requests']) else 'completed_with_failures')
    except Exception as exc:
        obj.update(status='memory_error' if isinstance(exc,MemoryError) else 'error',
                   failure_reason=repr(exc),traceback=traceback.format_exc())
    finally:
        checkpoint()
    return obj


def jobs(protocol):
    for repeat in range(protocol['repeat_count']):
        for g,graph in enumerate(protocol['graphs']):
            for f,stream in enumerate(protocol['streams']):
                pivot=(repeat+3*g+f)%3
                for arm in ARMS[pivot:]+ARMS[:pivot]:
                    yield dict(graph=graph,stream=stream,arm=arm,repeat_id=repeat)


def batch(out,cpu):
    import psutil
    protocol=json.loads(PROTOCOL.read_text())
    gate=validate(identities(),False)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(out)
    lock=(ROOT/'logs/.measurement.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    limits=protocol['limits']
    began=time.monotonic()
    results=[]
    schedule=list(jobs(protocol))
    save(out/'FROZEN_JOBS.json',dict(jobs=schedule,admission=gate,protocol_sha256=sha(PROTOCOL),cpu=cpu))
    for pos,job in enumerate(schedule):
        path=out/f"{pos:03d}_{job['graph']}_{job['stream']}_{job['arm']}.json"
        item=dict(position=pos,job=job,result=path.name,status='batch_cap_not_started')
        if time.monotonic()-began<limits['batch_seconds']:
            cmd=[sys.executable,'-m','src.anchor_service_bench','--cpu',str(cpu),'--output',str(path.resolve()),
                 '--schedule-position',str(pos)]
            for k,v in job.items():
                cmd.extend(('--'+k.replace('_','-'),str(v)))
            with path.with_suffix('.log').open('x') as log:
                started=time.monotonic()
                process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                monitored=psutil.Process(process.pid)
                peak=0
                stop=None
                while process.poll() is None:
                    try:
                        peak=max(peak,monitored.memory_info().rss)
                    except psutil.NoSuchProcess:
                        break
                    now=time.monotonic()
                    if peak>limits['rss_mib']*1024**2:
                        stop='rss_cap'
                    elif now-started>=limits['worker_seconds']:
                        stop='timeout'
                    elif now-began>=limits['batch_seconds']:
                        stop='batch_cap'
                    if stop:
                        process.kill()
                        break
                    time.sleep(limits['rss_sample_seconds'])
                code=process.wait()
                wall=time.monotonic()-started
            partial=json.loads(path.read_text()) if path.exists() else {}
            status=stop or partial.get('status','exit_before_checkpoint')
            item.update(status=status,returncode=code,process_wall_seconds=wall,
                external_sampled_peak_rss_bytes=peak,completed_requests=len(partial.get('requests',[])))
            if stop or code or not path.exists():
                partial.update(status=status,failure_reason=stop or partial.get('failure_reason','worker exited'))
                save(path,partial)
        results.append(item)
        save(out/'SCHEDULE_RESULTS.json',dict(jobs=results,limits=limits,batch_wall_seconds=time.monotonic()-began))
        print(pos,item['status'],path.name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    out=p.add_mutually_exclusive_group(required=True)
    out.add_argument('--batch',type=Path)
    out.add_argument('--output',type=Path)
    p.add_argument('--graph',choices=['Alpha','OTC'])
    p.add_argument('--stream',choices=['same','slow','independent'])
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--seed',type=int,default=104729)
    p.add_argument('--requests',type=int,default=4)
    p.add_argument('--budget',type=int,default=15)
    p.add_argument('--method',choices=['D'],default='D')
    p.add_argument('--weight-mode',choices=['weighted'],default='weighted')
    p.add_argument('--repeat-id',type=int,default=0)
    p.add_argument('--schedule-position',type=int,default=0)
    p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
    p.add_argument('--correctness',action='store_true',help='audit-only output for independent pre-admission replay')
    args=p.parse_args()
    if args.batch:
        batch(args.batch,args.cpu)
    else:
        if not args.graph or not args.stream or not args.arm or min(args.requests,args.budget)<0 or args.requests==0:
            p.error('one worker requires graph, stream, arm and valid request/budget')
        obj=work(args)
        print(obj['status'],obj['failure_reason'],flush=True)
        sys.exit(0 if obj['status']=='completed' else 1)
