"""One isolated persistent stream. Launch via the serial scheduler for measurements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
import traceback


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',required=True,choices=['Alpha','OTC'])
    p.add_argument('--graph-file',type=Path,help='explicit fixture, recorded in output')
    p.add_argument('--weight-mode',default='unit',choices=['unit','weighted'])
    p.add_argument('--stream',required=True,choices=['same','slow','independent'])
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--method',required=True,choices=list('FTIDS'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--requests',type=int,default=4)
    p.add_argument('--budget',type=int,default=15)
    p.add_argument('--hot-size',type=int,default=64)
    p.add_argument('--repeat-id',type=int,default=0)
    p.add_argument('--schedule-position',type=int,default=0)
    p.add_argument('--protocol-id',default='unit-correctness-development')
    p.add_argument('--source-commit')
    p.add_argument('--cpu',type=int)
    p.add_argument('--timers',choices=['on','off'],default='on')
    p.add_argument('--audit',action='store_true')
    p.add_argument('--unadmitted-weighted-correctness',action='store_true')
    return p.parse_args()


def run(args):
    start=perf_counter()
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                 'BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name]='1'
    if args.cpu is not None:
        os.sched_setaffinity(0,{args.cpu})
    from .common import EVIDENCE, ROOT, Ledger, dump, file_hash, opinions
    from .session import Session
    import numpy as np
    import scipy
    from scipy.sparse import load_npz
    from threadpoolctl import threadpool_info,threadpool_limits
    src_commit=args.source_commit or subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    graph_path=args.graph_file or ROOT/f'data/projected/{args.graph}_{args.weight_mode}.npz'
    builds={name:file_hash(EVIDENCE/f'diagnostic/build/{name}.so') for name in ('screen','index')}
    sources={name:file_hash(ROOT/f'src/{name}.py') for name in ('session','worker','common')}
    if args.weight_mode=='weighted' and (ROOT/'src/weighted_adapter.py').exists():
        sources['weighted_adapter']=file_hash(ROOT/'src/weighted_adapter.py')
    kernel_id=hashlib.sha256(json.dumps(dict(builds=builds,sources=sources),sort_keys=True).encode()).hexdigest()
    info=dict(protocol_id=args.protocol_id,source_commit=src_commit,source_files_sha256=sources,
        kernel_build_id=kernel_id,kernel_binary_sha256=builds,graph=args.graph,weight_mode=args.weight_mode,
        graph_path=str(graph_path),catalog_rule='fixed root lexical two-hop nonedges plus negative flips',
        stream=args.stream,stream_id=f'{args.graph}/{args.weight_mode}/{args.stream}/{args.seed}',
        seed=args.seed,method=args.method,repeat_id=args.repeat_id,schedule_position=args.schedule_position,
        budget=args.budget,delta=0,requests_planned=args.requests,hot_size=args.hot_size,
        timers=args.timers,audit=args.audit,cache_policy='structural root only; no cross-request Grams or answers',
        numpy_version=np.__version__,scipy_version=scipy.__version__,affinity=sorted(os.sched_getaffinity(0)),
        reset_policy='new root cursor, share exact root factor, empty low-rank step list',
        requests=[],prefixes={},status='starting',failure_reason=None)
    def checkpoint():
        info['worker_observed_wall_seconds']=perf_counter()-start
        info['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        dump(args.output,info)
    try:
        weighted_dev=args.unadmitted_weighted_correctness and args.audit and args.timers=='off'
        if args.unadmitted_weighted_correctness and not weighted_dev:
            raise ValueError('unadmitted weighted correctness requires --audit --timers off')
        if args.weight_mode!='unit' and not weighted_dev:
            gate_path=ROOT/'reports/WEIGHTED_ADMISSION.json'
            gate=json.loads(gate_path.read_text()) if gate_path.is_file() else {}
            if (gate.get('admitted') is not True or args.method not in gate.get('methods',[]) or
                    gate.get('weighted_adapter_sha256')!=sources.get('weighted_adapter') or
                    gate.get('session_sha256')!=sources['session']):
                info.update(status='not_admitted',failure_reason='weighted method/build lacks matching admission')
                return info
            info['weighted_admission_sha256']=file_hash(gate_path)
        if args.requests<1 or args.budget<0:
            raise ValueError('invalid number of requests or edits')
        with threadpool_limits(limits=1):
            info['threadpools']=threadpool_info()
            ledger=Ledger(args.timers=='on')
            with ledger.phase('graph_load'):
                graph_hash=file_hash(graph_path)
                W=load_npz(graph_path)
            session=Session(W,args.method,graph_name=args.graph,hot_size=args.hot_size,
                            ledger=ledger,timers=args.timers=='on',graph_hash=graph_hash,weight_mode=args.weight_mode)
            info.update(status='prepared',graph_hash=graph_hash,operator_hash=session.operator_hash,
                catalog_hash=session.catalog_hash,root_version_id=session.root_id,n=W.shape[0],
                catalog_size=session.m,structural_seconds=session.setup_seconds,
                structural_phases=session.setup_phases)
            del W
            # Opinions are request payloads supplied to a service. Generation is a
            # harness expense included exactly once in process wall, outside P/R/O.
            tin=perf_counter()
            inputs=opinions(session.W.shape[0],args.stream,args.seed,args.requests)
            info['stream_fixture_generation_seconds']=perf_counter()-tin
            cumulative=session.setup_seconds
            checkpoint()
            for q,s in enumerate(inputs,1):
                call=perf_counter()
                result=session.request(s,budget=args.budget,audit=args.audit)
                call_seconds=perf_counter()-call
                # Include destruction of the request frame and its last cursor references.
                frame_release=call_seconds-result['request_seconds']
                result['phases']['reset_restore']+=frame_release
                result['reset_opinion_seconds']+=frame_release
                result['request_seconds']=call_seconds
                result['query_position']=q
                result['stream_id']=info['stream_id']
                info['requests'].append(result)
                cumulative+=call_seconds
                if q in (1,2,4,8,16) or q==args.requests:
                    info['prefixes'][str(q)]=cumulative
                info['status']='in_progress'
                checkpoint()
            failures=[r for r in info['requests'] if r['status'] not in ('budget_stop','tolerance_stop')]
            info['status']='completed' if not failures else 'completed_with_failures'
    except BaseException as exc:
        info.update(status='memory_error' if isinstance(exc,MemoryError) else 'error',
                    failure_reason=repr(exc),traceback=traceback.format_exc())
    finally:
        checkpoint()
    return info


if __name__=='__main__':
    args=arguments()
    if args.output.exists():
        raise FileExistsError(f'result already exists: {args.output}')
    record=run(args)
    print(json.dumps({k:record[k] for k in ('method','stream_id','status','failure_reason')}),flush=True)
    sys.exit(0 if record['status'] in ('completed','completed_with_failures','not_admitted') else 1)
