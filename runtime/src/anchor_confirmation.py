"""Frozen Q16 held-out fixtures and serial worker for the admitted A1 service."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import traceback

from . import anchor_service_bench as dev

ROOT=dev.ROOT
PROTOCOL=ROOT/'protocol/A1_CONFIRMATION_PROTOCOL.json'
MANIFEST=ROOT/'protocol/A1_CONFIRMATION_INPUTS.json'
FIXTURES=ROOT/'data/confirmation_196613'
CHECK=ROOT/'reports/A1_CONFIRMATION_INPUT_CHECK.json'


def config():
    return json.loads(PROTOCOL.read_text())


def wrapper_sources():
    return {k:dev.sha(ROOT/f'src/{k}.py') for k in ('anchor_confirmation','holdout_stream','common')}


def matrix_check(matrix,n,count):
    from .common import digest
    import numpy as np
    if (matrix.dtype!=np.int64 or matrix.shape!=(count,n) or not matrix.flags.c_contiguous
            or np.any(matrix< -65536) or np.any(matrix>65536)):
        raise ValueError('invalid held-out opinion matrix')
    return digest(matrix),[digest(row) for row in matrix]


def prepare():
    from .common import opinions
    from .holdout_stream import signed_neighbor_opinions
    from scipy.sparse import load_npz
    import numpy as np
    cfg=config()
    if MANIFEST.exists() or FIXTURES.exists():
        raise FileExistsError('refusing to replace held-out fixture freeze')
    records=[]
    with tempfile.TemporaryDirectory(dir=FIXTURES.parent) as temp:
        dest=Path(temp)/'frozen'
        dest.mkdir()
        for graph in cfg['graphs']:
            paths={mode:ROOT/f'data/projected/{graph}_{mode}.npz' for mode in ('unit','weighted')}
            hashes={mode:dev.sha(path) for mode,path in paths.items()}
            meta=json.loads((ROOT/f'data/projected/{graph}_metadata.json').read_text())
            if any(hashes[mode]!=meta[mode+'_sha256'] for mode in paths):
                raise ValueError('projected graph changed')
            W=load_npz(paths['unit'])
            for stream in cfg['streams']:
                vecs=(signed_neighbor_opinions(W,cfg['requests'],cfg['seed']) if stream=='signed-neighbor'
                      else opinions(W.shape[0],stream,cfg['seed'],cfg['requests']))
                mat=np.stack(vecs)
                matrix_hash,rows=matrix_check(mat,W.shape[0],cfg['requests'])
                filename=f'{graph}_{stream}.npz'
                np.savez(dest/filename,opinions=mat)
                records.append(dict(graph=graph,stream=stream,fixture_id=f"{graph}/{stream}/{cfg['seed']}/Q{cfg['requests']}",
                    file=str((FIXTURES/filename).relative_to(ROOT)),graph_sha256=hashes['weighted'],
                    unit_graph_sha256=hashes['unit'],npz_sha256=dev.sha(dest/filename),matrix_sha256=matrix_hash,
                    opinion_hashes=rows,shape=list(mat.shape)))
        dest.rename(FIXTURES)
    dev.save(MANIFEST,dict(protocol_sha256=dev.sha(PROTOCOL),source_files_sha256=wrapper_sources(),seed=cfg['seed'],
                          count=cfg['requests'],fixtures=records))


def load_fixture_record(rec,root=ROOT):
    import numpy as np
    path=root/rec['file']
    if dev.sha(path)!=rec['npz_sha256']:
        raise ValueError('fixture file checksum mismatch')
    with np.load(path,allow_pickle=False) as source:
        if source.files!=['opinions']:
            raise ValueError('unexpected fixture members')
        matrix=source['opinions']
    mat_hash,rows=matrix_check(matrix,rec['shape'][1],rec['shape'][0])
    if (mat_hash,rows)!=(rec['matrix_sha256'],rec['opinion_hashes']):
        raise ValueError('fixture matrix/row checksum mismatch')
    return matrix


def gates():
    a1=dev.validate(dev.identities(),False)
    cfg=config()
    manifest=json.loads(MANIFEST.read_text())
    check=json.loads(CHECK.read_text())
    mh=dev.sha(MANIFEST)
    ph=dev.sha(PROTOCOL)
    if (manifest['protocol_sha256']!=ph or manifest['source_files_sha256']!=wrapper_sources() or
            manifest['seed']!=cfg['seed'] or manifest['count']!=cfg['requests'] or
            check.get('verified') is not True or check.get('input_manifest_sha256')!=mh or
            check.get('protocol_sha256')!=ph or check.get('fixture_count')!=6):
        raise ValueError('confirmation protocol/source/independent fixture check mismatch')
    keys=[(r['graph'],r['stream']) for r in manifest['fixtures']]
    if len(keys)!=6 or set(keys)!={(g,f) for g in cfg['graphs'] for f in cfg['streams']}:
        raise ValueError('incomplete or duplicate held-out fixture')
    return manifest,dict(**a1,independent_inputs=dev.sha(CHECK)),mh


def worker(args):
    from .common import Ledger
    from .anchor_session import AnchorSession
    from scipy.sparse import load_npz
    from threadpoolctl import threadpool_info,threadpool_limits
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg=config()
    manifest,gate,mh=gates()
    record=next(r for r in manifest['fixtures'] if (r['graph'],r['stream'])==(args.graph,args.stream))
    obj=dict(protocol_id=cfg['id'],protocol_sha256=dev.sha(PROTOCOL),source_files_sha256=dev.identities()|wrapper_sources(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_manifest_sha256=mh,fixture_id=record['fixture_id'],fixture_npz_sha256=record['npz_sha256'],
        fixture_matrix_sha256=record['matrix_sha256'],graph=args.graph,stream=args.stream,seed=cfg['seed'],
        weight_mode='weighted',arm=args.arm,method='D',budget=cfg['budget'],delta=0,hot_size=64,
        stream_id=f"{args.graph}/weighted/{args.stream}/{cfg['seed']}",requests_planned=cfg['requests'],
        audit=False,scope='held-out confirmation',admission=gate,repeat_id=args.repeat_id,
        schedule_position=args.schedule_position,cpu=args.cpu,affinity=sorted(os.sched_getaffinity(0)),
        requests=[],prefixes={},status='starting',failure_reason=None)
    obj['kernel_build_id']=hashlib.sha256(json.dumps(obj['source_files_sha256'],sort_keys=True).encode()).hexdigest()
    started=time.monotonic()
    def checkpoint():
        obj['worker_observed_wall_seconds']=time.monotonic()-started
        obj['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        dev.save(args.output,obj)
    try:
        with threadpool_limits(limits=1):
            obj['threadpools']=threadpool_info()
            ledger=Ledger(True)
            gp=ROOT/f'data/projected/{args.graph}_weighted.npz'
            with ledger.phase('graph_load'):
                gh=dev.sha(gp)
                if gh!=record['graph_sha256']:
                    raise ValueError('fixture/graph binding failed')
                W=load_npz(gp)
            sess=AnchorSession(W,args.arm,graph_hash=gh,ledger=ledger)
            del W
            obj.update(graph_hash=gh,catalog_rule='root lexical two-hop nonedges + negative flips',
                operator_hash=sess.operator_hash,catalog_hash=sess.catalog_hash,root_version_id=sess.root_id,
                n=sess.W.shape[0],catalog_size=sess.m,structural_seconds=sess.setup_seconds,
                structural_phases=sess.setup_phases,status='prepared')
            t=time.monotonic()
            matrix=load_fixture_record(record)
            if matrix.shape!=(cfg['requests'],sess.W.shape[0]):
                raise ValueError('fixture shape mismatch')
            obj['stream_fixture_load_validate_seconds']=time.monotonic()-t
            prefix=sess.setup_seconds
            checkpoint()
            for q,s in enumerate(matrix,1):
                t=time.monotonic()
                req=sess.request(s,budget=cfg['budget'])
                elapsed=time.monotonic()-t
                gap=elapsed-req['request_seconds']
                req['phases']['reset_restore']+=gap
                req['reset_opinion_seconds']+=gap
                req.update(request_seconds=elapsed,query_position=q,stream_id=obj['stream_id'])
                if (req['opinion_hash']!=record['opinion_hashes'][q-1] or
                        abs(sum(req['phases'].values())-elapsed)>1e-7 or
                        abs(req['reset_opinion_seconds']+req['online_seconds']-elapsed)>1e-7):
                    raise ValueError('held-out opinion or request accounting mismatch')
                obj['requests'].append(req)
                prefix+=elapsed
                if q in cfg['prefixes']:
                    obj['prefixes'][str(q)]=prefix
                obj['status']='in_progress'
                checkpoint()
            obj['status']=('completed' if all(r['status'] in ('budget_stop','tolerance_stop') for r in obj['requests'])
                           else 'completed_with_failures')
    except Exception as exc:
        obj.update(status='memory_error' if isinstance(exc,MemoryError) else 'error',
                   failure_reason=repr(exc),traceback=traceback.format_exc())
    finally:
        checkpoint()
    return obj['status']


def batch(out,cpu):
    import psutil
    cfg=config()
    _,gate,mh=gates()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(out)
    lock=(ROOT/'logs/.measurement.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    schedule=list(dev.jobs(cfg))
    limits=cfg['limits']
    dev.save(out/'FROZEN_JOBS.json',dict(jobs=schedule,admission=gate,protocol_sha256=dev.sha(PROTOCOL),
             input_manifest_sha256=mh,cpu=cpu))
    jobs=[]
    began=time.monotonic()
    for pos,job in enumerate(schedule):
        path=out/f"{pos:03d}_{job['graph']}_{job['stream']}_{job['arm']}.json"
        row=dict(position=pos,job=job,result=path.name,status='batch_cap_not_started')
        if time.monotonic()-began<limits['batch_seconds']:
            cmd=[sys.executable,'-m','src.anchor_confirmation','--cpu',str(cpu),'--output',str(path.resolve()),
                 '--schedule-position',str(pos)]
            for k,v in job.items():
                cmd.extend(('--'+k.replace('_','-'),str(v)))
            with path.with_suffix('.log').open('x') as log:
                start=time.monotonic()
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                mon=psutil.Process(p.pid)
                peak=0; stop=None
                while p.poll() is None:
                    try:peak=max(peak,mon.memory_info().rss)
                    except psutil.NoSuchProcess:break
                    now=time.monotonic()
                    if peak>limits['rss_mib']*1024**2:stop='rss_cap'
                    elif now-start>=limits['worker_seconds']:stop='timeout'
                    elif now-began>=limits['batch_seconds']:stop='batch_cap'
                    if stop:
                        p.kill()
                        break
                    time.sleep(limits['rss_sample_seconds'])
                code=p.wait()
                wall=time.monotonic()-start
            partial=json.loads(path.read_text()) if path.exists() else {}
            row.update(status=stop or partial.get('status','exit_before_checkpoint'),returncode=code,
                process_wall_seconds=wall,external_sampled_peak_rss_bytes=peak,completed_requests=len(partial.get('requests',[])))
            if stop or code or not path.exists():
                partial.update(status=row['status'],failure_reason=stop or partial.get('failure_reason','worker exited'))
                dev.save(path,partial)
        jobs.append(row)
        dev.save(out/'SCHEDULE_RESULTS.json',dict(jobs=jobs,limits=limits,batch_wall_seconds=time.monotonic()-began))
        print(pos,row['status'],path.name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--batch',type=Path)
    mode.add_argument('--output',type=Path)
    p.add_argument('--graph',choices=['Alpha','OTC'])
    p.add_argument('--stream',choices=['slow','independent','signed-neighbor'])
    p.add_argument('--arm',choices=dev.ARMS)
    p.add_argument('--repeat-id',type=int,default=0)
    p.add_argument('--schedule-position',type=int,default=0)
    p.add_argument('--cpu',type=int,default=0)
    args=p.parse_args()
    if args.cpu!=config()['cpu']:
        p.error('CPU differs from registered confirmation protocol')
    if args.prepare:
        prepare()
    elif args.batch:
        batch(args.batch,args.cpu)
    else:
        for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                  'BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
            os.environ[k]='1'
        if args.graph is None or args.stream is None or args.arm is None:
            p.error('worker requires --graph, --stream, --arm')
        os.sched_setaffinity(0,{args.cpu})
        sys.exit(0 if worker(args)=='completed' else 1)
