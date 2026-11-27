"""Offline classification of observed winners at the *unedited root* only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter

from scipy import sparse

from . import checker
from .common import ROOT, dump, file_hash, opinions


def require(condition,label):
    if not condition:
        raise ValueError(label)


def load_root(graph,weight):
    path=ROOT/f'data/projected/{graph}_{weight}.npz'
    graph_hash=file_hash(path)
    info=json.loads((path.parent/f'{graph}_metadata.json').read_text())
    require(info[weight+'_sha256']==graph_hash,'projection manifest digest mismatch')
    w=sparse.load_npz(path)
    A=checker.operator(w)
    A.sort_indices()
    pool=checker.catalog(w)
    op_hash=checker.digest(A.indptr,A.indices,A.data)
    cat_hash=checker.digest(pool[0],pool[1],pool[3])
    salt=('superlu-lowrank:scale16:proof48:screen20' if weight=='unit'
          else 'weighted-normalized:scale16:proof48:screen20')
    rid=hashlib.sha256('/'.join((graph_hash,op_hash,cat_hash,'lexical-twohop-negative:v1',salt)).encode()).hexdigest()
    return A,pool,(graph_hash,op_hash,cat_hash,rid)


def bind_request(trace,r,q,s,pool,ids):
    require(tuple(trace[k] for k in ('graph_hash','operator_hash','catalog_hash','root_version_id'))==ids,
            'external graph/operator/catalog/root binding')
    require(r['root_version_id']==ids[-1] and r['query_position']==q and r['method']=='D'
            and r['budget']==trace['budget'] and r['delta']==trace['delta']==0,'request identity')
    require(r['opinion_hash']==checker.opinion_hash(s),'reconstructed opinion hash mismatch')
    m=len(pool[0]); seeds=r['initial_seed_ids']
    require(type(trace['hot_size']) is int and trace['hot_size']==64 and
            len(seeds)==min(m,trace['hot_size']) and len(set(seeds))==len(seeds) and
            all(type(e) is int and 0<=e<m for e in seeds),'invalid trace initial candidate IDs')
    require(r['initial_gram_count']==0,'unexpected initial Gram count')
    seedset=set(seeds); current=ids[-1]; winners=[]
    for epoch,row in enumerate(r['decisions'],1):
        require(row['epoch']==epoch and row['parent_version_id']==current and row['root_version_id']==ids[-1],
                'decision version/epoch binding')
        if row['status']=='accepted':
            e=row['winner_id']
            require(type(e) is int and 0<=e<m and e not in winners,'illegal or duplicate winner')
            require((row['i'],row['j'],bool(row['flip']))==(int(pool[0][e]),int(pool[1][e]),bool(pool[3][e])),
                    'winner ID/endpoint/type mismatch')
            require(bool(row['outsider'])==(e not in seedset),'trace initial work-set membership mismatch')
            current=hashlib.sha256(f'{current}/{e}'.encode()).hexdigest()
            winners.append(e)
        else:
            require(row['status']=='tolerance_stop' and row['winner_id'] is None and epoch==len(r['decisions']),
                    'unexpected terminal event')
        require(row['current_version_id']==current,'decision child version mismatch')
    require(current==r['current_version_id'] and len(winners)==r['commits'],'final version/count mismatch')
    if r['status']=='budget_stop':
        require(len(winners)==r['budget'] and len(winners)==len(r['decisions']),'invalid budget completion')
    else:
        require(r['status']=='tolerance_stop' and r['decisions'][-1]['status']=='tolerance_stop','invalid stop')
    return tuple(winners),tuple(sorted(seeds))


def energy_box(A,s):
    z=checker.response(A,s)
    _,_,_,lo,hi=checker.dyadic_box(A,s,z,checker.D)
    return lo,hi


def classify(A,pool,s,winners,seeds):
    start=perf_counter()
    pl,pu=energy_box(A,s)
    rows=[]
    for epoch,e in enumerate(winners,1):
        cl,cu=energy_box(checker.changed(A,pool,e),s)
        lower,upper=pl-cu,pu-cl
        sign=('positive' if lower>0 else 'negative' if upper<0 else
              'zero' if lower==upper==0 else 'indeterminate')
        rows.append(dict(epoch=epoch,winner_id=e,i=int(pool[0][e]),j=int(pool[1][e]),
                    flip=bool(pool[3][e]),scaled_weight=int(pool[2][e]),outsider=e not in seeds,
                    root_gain_low=str(lower),root_gain_high=str(upper),root_gain_sign=sign))
    return dict(root_energy_low=str(pl),root_energy_high=str(pu),winners=rows,
                verified_winner_count=len(rows),root_state_solves=1,root_child_state_solves=len(rows),
                root_query_check_seconds=perf_counter()-start)


def run(folder,output):
    if output.exists():
        raise FileExistsError(f'preserve prior audit: {output}')
    begin=perf_counter()
    schedule_path=folder/'SCHEDULE_RESULTS.json'
    schedule=json.loads(schedule_path.read_text())
    roots={}; by_query={}; unique=[]; coverage=[]; failures=[]; omitted=[]
    def emit():
        query_counts=Counter(w['root_gain_sign'] for u in unique for w in u['winners'])
        mapped_counts=Counter(w['root_gain_sign'] for c in coverage for w in unique[c['unique_query_index']]['winners'])
        dump(output,dict(scope='completed scheduled D workers at repetition 0; root-versus-single-edit gain for their actual winners',
            schedule_sha256=file_hash(schedule_path),checker_sha256=file_hash(Path(checker.__file__)),
            audit_source_sha256=file_hash(Path(__file__)),dyadic_scale=checker.D,
            method='independent CG trials and exact variational dyadic energy bounds; root/child physical objective difference',
            initial_workset_source='measured root shortlist IDs (cardinality, range, duplicate, outsider and repeated-input consistency checked); no independent re-ranking',
            cost_excludes_measured_queries=True,unique_queries=len(unique),source_requests=len(coverage),
            unique_winner_sign_counts=dict(query_counts),source_request_winner_sign_counts=dict(mapped_counts),
            root_solves=sum(r['root_state_solves'] for r in unique),child_solves=sum(r['root_child_state_solves'] for r in unique),
            unique_query_results=unique,coverage=coverage,excluded_jobs=omitted,failures=failures,
            total_audit_wall_seconds=perf_counter()-begin,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    for item in schedule['jobs']:
        job=item['job']
        if job['method']!='D' or job['repeat_id']!=0:
            continue
        if item['status']!='completed':
            omitted.append(dict(file=item['result'],schedule_position=item['position'],status=item['status']))
            continue
        path=folder/item['result']; q=None
        try:
            raw=path.read_bytes(); source_hash=hashlib.sha256(raw).hexdigest(); trace=json.loads(raw)
            for key in ('graph','weight_mode','stream','seed','method','repeat_id','protocol_id'):
                require(trace[key]==job[key],f'{key} differs from scheduled job')
            require(trace['status']=='completed' and trace['schedule_position']==item['position'] and
                    not trace['audit'] and trace['timers']=='on','not the completed measured worker')
            gk=(trace['graph'],trace['weight_mode'])
            if gk not in roots: roots[gk]=load_root(*gk)
            A,pool,identities=roots[gk]
            inputs=opinions(A.shape[0],trace['stream'],trace['seed'],trace['requests_planned'])
            require(len(inputs)==len(trace['requests']), 'incomplete worker request count')
            for q,(r,s) in enumerate(zip(trace['requests'],inputs),1):
                winners,seeds=bind_request(trace,r,q,s,pool,identities)
                key=(*gk,r['opinion_hash'],winners)
                if key not in by_query:
                    result=classify(A,pool,s,winners,set(seeds))
                    by_query[key]=len(unique)
                    unique.append(dict(graph=gk[0],weight_mode=gk[1],graph_sha256=identities[0],
                                       root_version_id=identities[-1],opinion_hash=r['opinion_hash'],
                                       initial_seed_ids=list(seeds),**result))
                k=by_query[key]
                require(tuple(unique[k]['initial_seed_ids'])==seeds,'identical root query changed initial work set')
                coverage.append(dict(file=path.name,file_sha256=source_hash,schedule_position=item['position'],
                    q=q,stream=trace['stream'],seed=trace['seed'],repeat_id=trace['repeat_id'],unique_query_index=k))
        except Exception as exc:
            failures.append(dict(file=path.name,q=q,error=repr(exc),position=item['position']))
        emit()
    emit()
    return len(failures)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=ROOT/'raw/pilot')
    parser.add_argument('--output',type=Path,default=ROOT/'analysis/ROOT_SIGNS.json')
    args=parser.parse_args()
    raise SystemExit(bool(run(args.folder,args.output)))
