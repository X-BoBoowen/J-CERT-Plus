"""Summarize an ordered lifecycle batch without treating prefixes or repeats as new inputs."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

from .common import PHASES,dump


def write_table(path,rows):
    if not rows:
        return
    with path.open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def ratio(a,b):
    return a/b if b else None


def analyze(folder,out):
    out.mkdir(parents=True,exist_ok=True)
    schedule=json.loads((folder/'SCHEDULE_RESULTS.json').read_text())
    measurements,failures,work,ceilings,phase_rows=[],[],[],[],[]
    pairs=defaultdict(dict)
    for item in schedule['jobs']:
        path=folder/item['result']
        obj=json.loads(path.read_text()) if path.exists() else {}
        identity=item['job']
        if item['status']!='completed':
            failures.append(dict(position=item['position'],status=item['status'],failure=obj.get('failure_reason'),
                                requests=item.get('completed_requests',0),**identity))
        if obj.get('timers')!='on' or obj.get('audit'):
            continue
        qelapsed=0.
        reset_total=online_total=0.
        qs=obj.get('requests',[])
        setup=obj.get('structural_phases',{})
        pres=obj.get('structural_seconds',0.)-sum(setup.values())
        if abs(pres)>1e-7:
            raise ValueError(f'unreconciled structure at {path}')
        for q,r in enumerate(qs,1):
            sec=r['request_seconds']
            phases=r['phases']
            if abs(sec-sum(phases.values()))>1e-7:
                raise ValueError(f'unreconciled request {q} at {path}')
            prepost=r['reset_opinion_seconds']
            online=r['online_seconds']
            if abs(sec-prepost-online)>1e-7 or abs(online-sum(r['online_phases'].values()))>1e-7:
                raise ValueError(f'unreconciled reset/refresh/online at {path} request {q}')
            reset_total+=prepost
            online_total+=online
            qelapsed+=sec
            prefix=obj['structural_seconds']+qelapsed
            keys=('graph','weight_mode','stream','seed','method','repeat_id')
            base={k:obj[k] for k in keys}
            base['q']=q
            phase_rows.append(dict(**base,status=r['status'],request_s=sec,reset_opinion_s=prepost,
                online_s=online,**phases,**{'online_'+k:v for k,v in r['online_phases'].items()}))
            for group,names in [('opinion_caps_and_bin_rekey',('caps_build','caps_sort','posting/index_build')),
                                ('full_request_opinion_refresh',('state_solve_and_cert','caps_build','caps_sort',
                                                               'posting/index_build','opinion_refresh')),
                                ('response_acquisition',('response_acquisition',)),
                                ('transport_local',('transport_local',)),('comparison_refinement',('comparison/refinement',))]:
                expense=sum(phases[name] for name in names)
                ceilings.append(dict(**base,target=group,total=sec,target_s=expense,
                    zero_overhead_ceiling=ratio(sec,sec-expense),break_even_new_cost_s=expense))
            work.append(dict(**base,status=r['status'],commits=r['commits'],**r['counters']))
            if str(q) not in obj['prefixes']:
                continue
            if abs(prefix-obj['prefixes'][str(q)])>1e-7:
                raise ValueError(f'unreconciled prefix at {path}')
            row=dict(**base,status=r['status'],structural_s=obj['structural_seconds'],
                request_s=sec,cumulative_s=prefix,cumulative_reset_opinion_s=reset_total,
                cumulative_online_s=online_total,complete_worker_status=item['status'],
                process_wall_s=item.get('process_wall_seconds'),
                process_peak_rss_bytes=obj.get('process_peak_rss_bytes'),uncosted_wall_s=None)
            if q==len(qs):
                row['uncosted_wall_s']=item.get('process_wall_seconds',prefix)-prefix
            measurements.append(row)
            if item['status']=='completed' and all(p['status'] in ('budget_stop','tolerance_stop') for p in qs[:q]):
                pairs[tuple(base[k] for k in ('graph','weight_mode','stream','seed','repeat_id','q'))][obj['method']]=(row,qs[:q])
    comparisons=[]
    for block,methods in pairs.items():
        if 'D' not in methods:
            continue
        d,d_req=methods['D']
        for method,(ref,ref_req) in methods.items():
            if method=='D':
                continue
            if any(l['opinion_hash']!=r['opinion_hash'] for l,r in zip(d_req,ref_req)):
                raise ValueError('unmatched opinions within time pair')
            def ids(r):
                return [e['winner_id'] for e in r['decisions'] if e['status']=='accepted']
            same=all(ids(l)==ids(r) and l['status']==r['status'] for l,r in zip(d_req,ref_req))
            if method!='S' and not same:
                raise ValueError(f'full guarantee winner disagreement {block} {method}')
            ds,rs=d['cumulative_s'],ref['cumulative_s']
            comparisons.append(dict(graph=block[0],weight_mode=block[1],stream=block[2],seed=block[3],
                repeat_id=block[4],q=block[5],reference=method,all_paths_equal=same,
                D_over_reference=ratio(ds,rs),reference_over_D=ratio(rs,ds),
                D_minus_reference_s=ds-rs,reference_minus_D_s=rs-ds,
                signed_gain_difference_low_num=d_req[-1]['gain_low']-ref_req[-1]['gain_high'],
                signed_gain_difference_high_num=d_req[-1]['gain_high']-ref_req[-1]['gain_low'],
                gain_denominator=d_req[-1]['gain_denominator']))
    by_input=defaultdict(list)
    for r in comparisons:
        key=('graph','weight_mode','stream','seed','q','reference')
        by_input[tuple(r[k] for k in key)].append(r)
    reduced=[]
    for group,rows in by_input.items():
        reduced.append(dict(zip(key,group)) | dict(timing_repeats=len(rows),
            median_D_over_reference=median(r['D_over_reference'] for r in rows),
            median_D_minus_reference_s=median(r['D_minus_reference_s'] for r in rows),
            complete_path_matches=all(r['all_paths_equal'] for r in rows)))
    for file,rows in [('prefixes.csv',measurements),('per_request_phases.csv',phase_rows),
              ('per_request_work.csv',work),('ideal_cost_ceilings.csv',ceilings),
              ('paired_repeats.csv',comparisons),('input_blocks.csv',reduced)]:
        write_table(out/file,rows)
    dump(out/'summary.json',dict(scheduled=len(schedule['jobs']),failures=failures,
        timed_prefixes=len(measurements),matched_reference_prefixes=len(comparisons),input_blocks=len(reduced),
        ceiling_interpretation='T/(T-target) removes a whole measured request phase for free. Actual candidate strategies must subtract their own measured overhead. State phase includes child certification.',
        process_boundary='Full process wall includes imports, stream fixture generation, checkpoint serialization, output and scheduling.',
        sampling_unit='One graph × stream family × seed per weight mode. Repeated timings and nested Q prefixes are correlated; do not pool these as independent draws.'))


def shadow(trace_path,out,positions=None):
    """Clean fixed-shortlist replay along the observed D parents, forcing D's commits."""
    from .common import DEN,ROOT,SCALE,UNIT,digest,file_hash,opinions
    from .session import Session
    import numpy as np
    import long_promotion as lp
    from promotion_directional import directional_block
    from scipy import sparse
    from threadpoolctl import threadpool_limits
    obj=json.loads(trace_path.read_text())
    if obj['method']!='D':
        raise ValueError('shadow path must supply D parents')
    path=ROOT/f"data/projected/{obj['graph']}_{obj['weight_mode']}.npz"
    graph_hash=file_hash(path)
    if graph_hash!=obj['graph_hash']:
        raise ValueError('graph identity mismatch')
    inputs=opinions(obj['n'],obj['stream'],obj['seed'],obj['requests_planned'])
    results=[]
    with threadpool_limits(limits=1):
        sess=Session(sparse.load_npz(path),'S',hot_size=obj['hot_size'],weight_mode=obj['weight_mode'],
                     graph_hash=graph_hash,timers=False)
        if sess.root_id!=obj['root_version_id'] or sess.catalog_hash!=obj['catalog_hash']:
            raise ValueError('root version/catalog mismatch')
        for k,req in enumerate(obj['requests'],1):
            if positions and k not in positions:
                continue
            s=inputs[k-1]
            if req['opinion_hash']!=digest(s):
                raise ValueError('opinion identity mismatch')
            A=sess.A
            root_views=sess.kernel.audit_views() if sess.kernel else lp.Views(sess.pool,UNIT,SCALE)
            if not sess.kernel:
                root_views.directional_block=directional_block
            root_state=lp.certified_state(A,sess.factor,s,UNIT)
            caps=(sess.kernel.root_caps(*root_state[3:]) if sess.kernel else
                  lp.cap(sess.pool,slice(None),*root_state[3:],sess.diag,False,SCALE))
            ids=np.argsort(-caps,kind='stable')[:obj['hot_size']]
            initial=set(map(int,ids))
            if initial!=set(req['initial_seed_ids']):
                raise ValueError('initial shortlist mismatch')
            chosen=set()
            events=[]
            for d in req['decisions']:
                if d['status'] not in ('accepted','tolerance_stop'):
                    events.append(dict(epoch=d['epoch'],status='input_'+d['status']))
                    break
                fac=sess.factor if not chosen else lp.UpdatedFactor(A)
                state=root_state if not chosen else lp.certified_state(A,fac,s,UNIT)
                views=sess.kernel.audit_views() if sess.kernel else lp.Views(sess.pool,UNIT,SCALE)
                if not sess.kernel:
                    views.directional_block=directional_block
                evaluated={}
                best,winner=0,None
                for e in sorted(initial-chosen):
                    lo,hi=views.score(e,0,A,fac,state[1],state[2])
                    evaluated[e]=hi
                    if lo>best:
                        best,winner=lo,e
                try:
                    winner,best=(sess.kernel.resolve(A,s/DEN,evaluated,winner,best) if sess.kernel else
                               lp.resolve_weighted(A,s/DEN,sess.pool,evaluated,winner,best,UNIT,SCALE))
                    shadow_status='accepted' if winner is not None else 'tolerance_stop'
                except ArithmeticError as err:
                    shadow_status='unresolved'
                    winner,best=None,None
                forced=d['winner_id']
                event=dict(epoch=d['epoch'],parent_version_id=d['parent_version_id'],
                    status=shadow_status,shadow_winner=winner,shadow_low=None if best is None else best*SCALE,
                    shadow_high=None if winner is None else evaluated[winner]*SCALE,full_winner=forced,
                    matches=winner==forced if shadow_status!='unresolved' else None,
                    outsider=forced is not None and forced not in initial)
                if forced is not None:
                    low,high=root_views.score(forced,0,sess.A,sess.factor,root_state[1],root_state[2])
                    event.update(root_gain_low=low*SCALE,root_gain_high=high*SCALE,
                        root_gain_sign='positive' if low>0 else 'negative' if high<0 else
                                       'zero' if low==high==0 else 'indeterminate')
                    if forced in initial and shadow_status!='unresolved' and winner!=forced:
                        raise ArithmeticError('D winner in shortlist disagrees with clean shortlist')
                    i,j,f=(a[forced].item() for a in sess.pool)
                    w=(0,-2*SCALE,-2*SCALE,0) if f else (1,-1,-1,1)
                    A=(sess.kernel.child(A,forced) if sess.kernel else
                       A+sparse.coo_matrix((w,([i,i,j,j],[i,j,i,j])),shape=A.shape).tocsr())
                    chosen.add(forced)
                events.append(event)
            results.append(dict(query_position=k,opinion_hash=digest(s),events=events))
    dump(out,dict(source_file_sha256=file_hash(trace_path),gain_denominator=UNIT**2,requests=results,
         semantics='Offline fresh refactors on observed D parents; fixed root top-k, every remaining shortlist candidate scored. No access to measured caches.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--input',type=Path,help='scheduled batch folder')
    source.add_argument('--shadow-trace',type=Path,help='a D worker result for independent shortlist comparison')
    p.add_argument('--query-position',action='append',type=int)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.shadow_trace:
        shadow(args.shadow_trace,args.output,args.query_position)
    else:
        analyze(args.input,args.output)
