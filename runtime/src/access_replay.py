"""Independent output proofs and matched untimed budget-event reconstruction."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
from scipy import sparse

from .checker import (CatalogReplay, D, ROOT, changed, digest, dyadic_box, exact_energy,
                      need, opinion_hash, response, worker_request)

UNIT = 2**48
SCREEN = 2**20
DIV = UNIT // SCREEN
SCALE = 16
PLAN_FIELDS = ('anchor_request', 'anchor_opinion_hash', 'target', 'tau', 'pre_dedup_visits',
               'posting_raw_visits', 'tail_raw_visits', 'group_tails', 'max_drift')
EVENT_FIELDS = PLAN_FIELDS + ('epoch', 'parent_version_id', 'returned_ids',
                              'unique_ids_before_exclusion', 'budget', 'consumed_before',
                              'consumed', 'action', 'fallback_reason')
HISTORY_FIELDS = ('initial_anchor_request', 'initial_anchor_opinion_hash', 'anchor_request',
                  'anchor_opinion_hash', 'visit_budget', 'old_consumed', 'fallback_count')


def checksum(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def field_match(actual, reference, fields, description):
    for name in fields:
        need(name in actual and name in reference and actual[name] == reference[name],
             f'{description}: {name}')


def match_timed_event(timed, audit):
    field_match(timed, audit, EVENT_FIELDS, 'timed event mismatch')
    rejected = timed['rejected_plan']
    other = audit['rejected_plan']
    need((rejected is None) == (other is None), 'timed rejected-plan mismatch')
    if rejected is not None:
        field_match(rejected, other, PLAN_FIELDS, 'timed rejected-plan mismatch')


def match_request(timed, audit):
    field_match(timed, audit, ('method','arm','delta','budget','root_version_id','current_version_id',
                               'opinion_hash','status','commits','gain_low','gain_high',
                               'gain_denominator','initial_seed_ids'), 'timed request mismatch')
    need(timed['decisions'] == audit['decisions'] or
         all(all(k in b and b[k] == v for k,v in a.items())
             for a,b in zip(timed['decisions'],audit['decisions'],strict=True)),
         'timed decision (including version/winner/bounds) mismatch')
    h, a = timed['discovery'], audit['discovery']
    field_match(h, a, HISTORY_FIELDS, 'timed history mismatch')
    need(len(h['epochs']) == len(a['epochs']) == len(timed['decisions']), 'timed epoch count')
    for e,f in zip(h['epochs'], a['epochs'], strict=True):
        match_timed_event(e, f)


def codes(seq, n, bound):
    need(isinstance(seq, list) and len(seq) == n and
         all(type(v) is int and -bound <= v <= bound for v in seq),'integer grid vector')
    return np.asarray(seq, dtype=np.int64)


def maximum(v):
    return max(-int(v.min()), int(v.max()))


def state(A, opinion, event):
    """Integer residual in this exact version, both fine and coarse radii."""
    n = A.shape[0]
    x = codes(event['fine_x'], n, 2*UNIT)
    xc = codes(event['current_xc'], n, 2*SCREEN)
    need(2*int(A.diagonal().max()) * maximum(x) + UNIT < 2**63, 'residual arithmetic range')
    raw = maximum(A@x - opinion*(UNIT//65536))
    gap = maximum(x - xc*DIV)
    rc = (raw + gap + DIV-1)//DIV
    need(gap <= DIV//2 and event['raw_residual'] == raw and
         event['fine_radius'] == (raw + SCALE-1)//SCALE and
         event['current_rc'] == rc and maximum(xc) + rc <= 2*SCREEN,
         'uncertified current state/radius')
    return xc, rc


class Source:
    """Independent integer keys on the full registered catalog, from a certified root."""
    def __init__(self, pool, center, radius):
        self.pool, self.center, self.radius = pool, center, radius
        i,j,d,flip = pool
        self.norm = maximum(center) + radius
        need(4*self.norm**2 * int(d.max(initial=1)) < 2**62, 'root tangent arithmetic range')
        a,b = center[i],center[j]
        self.keys = d*np.where(flip, -4*a*b+4*radius*(abs(a)+abs(b)+radius),
                               (abs(a-b)+2*radius)**2)
        self.sorted_keys = [(int(w), np.sort(self.keys[d == w])) for w in np.unique(d)]
        self.degree = (np.bincount(i,minlength=len(center)) + np.bincount(j,minlength=len(center)))

    def top(self, k):
        # Positive or negative tied keys are ordered by canonical candidate ID.
        return sorted(map(int,np.argsort(-self.keys,kind='stable')[:k]))


def verify_plan(source, drift, event):
    """Compute all dyadic costs; scan original IDs directly for the selected union."""
    target = event['target']
    maxdrift = int(drift.max())
    need(type(target) is int and target >= 0 and int(drift.min()) >= 0 and
         maxdrift == event['max_drift'] and maxdrift <= 4*SCREEN, 'target/drift range')
    choices = []
    tau = 1
    while True:
        shift = min(8*(SCREEN//SCALE)*tau, 8*source.norm*tau+4*tau*tau)
        posting = int(source.degree[drift>tau].sum())
        tails = [int(len(keys)-keys.searchsorted(target-w*shift,side='left'))
                 for w,keys in source.sorted_keys]
        choices.append((posting+sum(tails),tau,shift,posting,tails))
        if tau >= maxdrift:
            break
        tau *= 2
    raw,tau,shift,posting,tails = min(choices, key=lambda entry: (entry[0],entry[1]))
    need(event['tau'] == tau, 'minimum-cost dyadic tau or lowest-cost tie')
    need(event['posting_raw_visits'] == posting and event['tail_raw_visits'] == sum(tails)
         and event['pre_dedup_visits'] == raw and event['group_tails'] == tails, 'raw postings/tails')
    i,j,d,_ = source.pool
    moving = drift > tau
    tail = source.keys >= target - d*shift
    # This second, full-catalog incidence scan counts repeated endpoints separately.
    need(int(moving[i].sum()) + int(moving[j].sum()) == posting and
         int(tail.sum()) == sum(tails), 'full-catalog occurrence counts')
    return np.flatnonzero(moving[i] | moving[j] | tail).astype(np.int32)


def checked_origin(event, active, sources, hashes, xc, rc):
    need(event['anchor_request'] == active and event['anchor_opinion_hash'] == hashes[active],
         'anchor source request/opinion')
    source = sources[active]
    need(np.array_equal(event['anchor_xc'],source.center) and event['anchor_rc'] == source.radius,
         'anchor root certificate mismatch')
    exact = abs(xc-source.center) + rc + source.radius
    need(np.array_equal(event['drift'],exact), 'incorrect anchor-to-child drift')
    return verify_plan(source, exact, event)


def warm_witness(rep, A, opinion, event, decision):
    """Prove some legal warm candidate exceeds the largest excluded coarse bin."""
    target = event['target']
    if target == 0:
        return None, 0
    threshold = Fraction(SCALE*(target-1),SCREEN**2)
    seeds = event['seed_ids']
    winner = decision['winner_id']
    if winner in seeds and Fraction(decision['low'],UNIT**2) > threshold:
        # The selected measured winner enclosure is proved by worker_request.
        return winner, 0
    # For a cold winner the final certified gain does not by itself justify
    # the warm threshold. Search actual warm IDs by current tangent and
    # independently prove one gain beats the omitted integer bin.
    xc = np.asarray(event['current_xc'],dtype=np.int64)
    rc = event['current_rc']
    ids = np.asarray(seeds,dtype=np.int32)
    i,j,d,flip = (arr[ids] for arr in rep.pool)
    a,b = xc[i],xc[j]
    caps = d*np.where(flip,-4*a*b+4*rc*(abs(a)+abs(b)+rc),(abs(a-b)+2*rc)**2)
    raw_fine = np.asarray(event['fine_x'],dtype=np.int64)
    z = raw_fine.astype(float)*(SCALE/UNIT)
    lower = dyadic_box(A,opinion,z,D)[3]
    checked = 0
    for k in np.argsort(-caps,kind='stable'):
        if int(caps[k]) < target:
            break
        child = changed(A,rep.pool,int(ids[k]))
        high = dyadic_box(child,opinion,response(child,opinion),D)[4]
        checked += 1
        if lower-high > threshold:
            return int(ids[k]),checked
        if len(opinion)<=12 and exact_energy(A,opinion)-exact_energy(child,opinion)>threshold:
            return int(ids[k]),checked
    raise ValueError('warm threshold has no independently certified legal incumbent')


def verify_audit(rep, opinion, q, measured, audit, initial, witness, hashes, budget):
    """Validate one selected request and its previously certified origin snapshots."""
    match_request(measured,audit)
    h = audit['discovery']
    source = {key: Source(rep.pool,*witness[key]) for key in {q, initial}}
    current = source[q]
    need(h['visit_budget'] == budget and h['source_root_version'] == audit['root_version_id'],
         'budget or structural root mismatch')
    initial_id = h['initial_anchor_request']
    if q==1:
        need(initial_id is None and h['initial_anchor_opinion_hash'] is None,
             'unexpected old anchor on first request')
        active = q
    else:
        need(initial_id == initial and h['initial_anchor_opinion_hash'] == hashes[initial],
             'inherited active source')
        active = initial
    need(audit['initial_seed_ids'] == current.top(64), 'root top-64 from certified keys')
    A, parent = rep.A, audit['root_version_id']
    chosen = set()
    used = switches = returned = scanned = warm_child_proofs = 0
    warm_ids = []
    for t,(event,decision) in enumerate(zip(h['epochs'],audit['decisions'],strict=True),1):
        need(event['epoch']==t and event['parent_version_id']==parent==decision['parent_version_id']
             and event['budget']==budget and event['consumed_before']==used,'decision/old-budget binding')
        xc,rc = state(A,opinion,event)
        need(np.array_equal(event['current_root_xc'],current.center) and
             event['current_root_rc']==current.radius and
             np.array_equal(event['base_drift'],abs(xc-current.center)+rc+current.radius),
             'current request-root drift')
        if t==1:
            need(np.array_equal(current.center,xc) and rc==current.radius,'epoch 1 must be root')
        need(event['tau']==decision['tau'] and 0<=event['target']<=
             -(-decision['low']//(SCALE*DIV**2)),'target incompatible with certified winner')
        reject = event['rejected_plan']
        if reject is not None:
            need(active!=q and switches==0 and reject['target']==event['target'],
                 'duplicate fallback or wrong denied target')
            checked_origin(reject,active,source,hashes,xc,rc)
            scanned += len(rep.pool[0])
            need(used+reject['pre_dedup_visits']>budget and event['action']=='fallback' and
                 event['fallback_reason']=='visit_budget_exceeded','admission denied at wrong budget')
            active = q
            switches += 1
        else:
            expected = 'fresh' if active==q else 'retain'
            need(event['action']==expected and event['fallback_reason'] is None,'incorrect source action')
        ids = checked_origin(event,active,source,hashes,xc,rc)
        scanned += len(rep.pool[0])
        if active!=q:
            used += event['pre_dedup_visits']
        need(used==event['consumed'] and used<=budget,'old visit accounting/overflow')
        seeds,excluded = event['seed_ids'],event['excluded_ids']
        m = len(rep.pool[0])
        need(seeds==sorted(set(seeds)) and all(type(k) is int and 0<=k<m for k in seeds)
             and not chosen.intersection(seeds) and excluded==sorted(chosen.union(seeds))
             and len(seeds)==decision['seed_visits'],'current warm/chosen exclusions')
        if t==1:
            need(seeds==audit['initial_seed_ids'],'root warm set')
        witness_id, checks = warm_witness(rep,A,opinion,event,decision)
        warm_ids.append(witness_id)
        warm_child_proofs += checks
        need(event['unique_ids_before_exclusion']==len(ids),'unique candidate count')
        ids = ids[~np.isin(ids,np.asarray(excluded,np.int32),assume_unique=True)]
        need(event['returned_ids']==len(ids)==decision['cold_endpoint_visits'] and
             digest(ids)==event['returned_ids_sha256'],'full-catalog emitted union/hash')
        returned += len(ids)
        winner = decision['winner_id']
        if winner is not None:
            need(winner not in chosen, 'repeated commit')
            chosen.add(winner)
            A = changed(A,rep.pool,winner)
            parent = hashlib.sha256(f'{parent}/{winner}'.encode()).hexdigest()
    need(h['old_consumed']==used and h['fallback_count']==switches and
         h['anchor_request']==active and h['anchor_opinion_hash']==hashes[active],
         'final source or budget totals')
    return dict(epochs=len(h['epochs']),fallback_count=switches,old_visits=used,
                budget=budget,returned_ids=returned,full_catalog_candidate_plan_rows=scanned,
                warm_threshold_witness_ids=warm_ids,warm_additional_child_proofs=warm_child_proofs)


def behavior(request, q):
    events = request['discovery']['epochs']
    switches = [event['epoch'] for event in events if event['action']=='fallback']
    need(len(switches)<=1 and len(switches)==request['discovery']['fallback_count'],
         'malformed behavior summary')
    if switches:
        return 'immediate' if switches[0]==1 else 'mid'
    if q==16 and events and all(e['action']=='retain' for e in events):
        return 'q16_retained'
    return None


def selected_paths(directory, cfg, coefficient):
    registration=json.loads((directory/'SCHEDULE_RESULTS.json').read_text())
    methods=[x for x in cfg['methods'] if x['arm']=='budget-guard' and
             Fraction(str(x['coefficient']))==coefficient]
    need(len(methods)==1,'guard coefficient not registered')
    label=methods[0]['label']
    entries=[row for row in registration['jobs'] if row['job']['label']==label and
             row['job']['repeat_id']==0]
    from itertools import product
    cells={(r['job']['graph'],r['job']['stream'],r['job']['seed']) for r in entries}
    need(cells==set(product(cfg['graphs'],cfg['streams'],cfg['seeds'])) and
         len(cells)==len(entries),'incomplete repeat-0 selection grid')
    by_graph={g:{} for g in cfg['graphs']}
    selection_grid=[]
    for row in sorted(entries,key=lambda r:(r['job']['seed'],r['job']['stream'])):
        job=row['job']
        path=directory/row['result']
        timed=json.loads(path.read_text())
        need(timed['schedule_position']==row['position'] and all(timed[k]==v for k,v in job.items()),
             'schedule/actual timed trace mismatch')
        selection_grid.append(dict(filename=path.name,status=row['status'],
                                   completed_requests=len(timed['requests']),sha256=checksum(path)))
        for q,req in enumerate(timed['requests'],1):
            branch=behavior(req,q)
            if branch:
                by_graph[job['graph']].setdefault(branch,(path,q))
    chosen={}
    missing={}
    for graph,branches in by_graph.items():
        missing[graph]=sorted({'immediate','mid','q16_retained'}-set(branches))
        for branch,(path,q) in branches.items():
            chosen.setdefault(path,{}).setdefault(q,[]).append(branch)
    return chosen,missing,selection_grid


def measured_inputs(timed, cfg, protocol_path, manifest_path, manifest, coefficient):
    need(timed['arm']=='budget-guard' and timed['repeat_id']==0 and
         timed['method']=='D' and timed['weight_mode']=='weighted' and timed['audit'] is False and
         timed['delta']==0 and timed['hot_size']==64 and timed['budget']==cfg['budget'] and
         timed['requests_planned']==cfg['requests'] and
         Fraction(str(timed['coefficient']))==coefficient,'timed service contract')
    need(manifest['verified'] is True and manifest['protocol_id']==timed['protocol_id']==cfg['id']
         and checksum(protocol_path)==manifest['protocol_sha256']==timed['protocol_sha256'] and
         checksum(manifest_path)==timed['input_manifest_sha256'] and
         timed['source_files_sha256']==manifest['source_files_sha256'],'protocol/source/fixture manifest')
    field=(timed['graph'],timed['stream'],timed['seed'])
    rows=[r for r in manifest['fixtures'] if (r['graph'],r['stream'],r['seed'])==field]
    need(len(rows)==1,'fixture ambiguity')
    row=rows[0]
    datafile=manifest_path.parent/row['file']
    need(row['fixture_id']==timed['fixture_id'] and
         checksum(datafile)==row['npz_sha256']==timed['fixture_npz_sha256'],'fixture file identity')
    with np.load(datafile,allow_pickle=False) as stored:
        need(stored.files==['opinions'],'unexpected fixture members')
        mat=stored['opinions']
    hashes=[opinion_hash(r) for r in mat]
    g=manifest['graphs'][field[0]]
    gp=ROOT/g['weighted_path']
    need(gp.resolve()==(ROOT/f'data/projected/{field[0]}_weighted.npz').resolve() and
         checksum(gp)==g['weighted_sha256']==timed['graph_hash'],'weighted graph binding')
    need(mat.dtype==np.int64 and mat.flags.c_contiguous and
         list(mat.shape)==row['shape']==[cfg['requests'],g['n']] and
         digest(mat)==row['matrix_sha256']==timed['fixture_matrix_sha256'] and
         hashes==row['opinion_hashes'] and np.all((mat>=-65536)&(mat<=65536)),
         'fixture matrix/individual rows')
    return sparse.load_npz(gp),mat,dict(enumerate(hashes,1))


def reproduce_requests(W, timed, matrix, hashes, selections, budget, rep):
    # Execution is solely a witness producer. The proof routines above do not import its index.
    from .anchor_session import AnchorSession
    sess=AnchorSession(W,'budget-guard',graph_hash=timed['graph_hash'])
    sess.bridge.visit_budget=budget
    old_sources={}
    roots_needed=set(selections)
    for q in selections:
        initial=q if q==1 else timed['requests'][q-2]['discovery']['anchor_request']
        need(type(initial) is int and 1<=initial<=q and (q==1 or initial<q),'active source range')
        old_sources[q]=initial
        roots_needed.add(initial)
    witness={}
    results=[]
    for q,op in enumerate(matrix[:max(selections)],1):
        measured=timed['requests'][q-1]
        need(measured['query_position']==q and measured['opinion_hash']==hashes[q] and
             measured['root_version_id']==timed['root_version_id'] and
             measured['budget']==timed['budget'] and measured['delta']==0,'request/fixture binding')
        audit=sess.request(op,budget=timed['budget'],audit=q in roots_needed)
        # Check prefix control flow even on upstream requests with no full proof selected.
        match_request(measured,audit)
        if q in roots_needed:
            first=audit['discovery']['epochs'][0]
            center,radius=state(rep.A,op,first)
            need(np.array_equal(first['current_root_xc'],center) and first['current_root_rc']==radius,
                 'uncertified prior root witness')
            witness[q]=(center,radius)
        if q in selections:
            detail=verify_audit(rep,op,q,measured,audit,old_sources[q],witness,hashes,budget)
            results.append(dict(request=q,opinion_hash=hashes[q],behaviors=selections[q],
                                matched_audit=detail))
    return results


def run(protocol_path, manifest_path, directory, coefficient):
    start=perf_counter()
    cfg=json.loads(protocol_path.read_text())
    manifest=json.loads(manifest_path.read_text())
    # Bind witness-producing code and the independent decision checker to measured source.
    for rel,expected in manifest['source_files_sha256'].items():
        need(checksum(ROOT/rel)==expected,'measured source changed: '+rel)
    chosen,missing,grid=selected_paths(directory,cfg,coefficient)
    output=[]
    for path, selections in sorted(chosen.items()):
        timed=json.loads(path.read_text())
        W,matrix,hashes=measured_inputs(timed,cfg,protocol_path,manifest_path,manifest,coefficient)
        rep=CatalogReplay(W)
        m=len(rep.pool[0])
        budget=int(coefficient*m)
        need(m==timed['catalog_size'] and budget==timed['visit_budget'],'registered raw budget')
        # Independent decision proofs consume the actual, unmodified timed request bytes.
        proofs={q:worker_request(rep,timed['requests'][q-1],matrix[q-1],timed['graph_hash'],
                                 kind='weighted') for q in selections}
        matched=reproduce_requests(W,timed,matrix,hashes,selections,budget,rep)
        for entry in matched:
            entry['actual_timed_output_proof']=proofs[entry['request']]
        output.append(dict(timed_file=path.name,timed_sha256=checksum(path),graph=timed['graph'],
                           graph_sha256=timed['graph_hash'],seed=timed['seed'],stream=timed['stream'],
                           fixture_file_sha256=timed['fixture_npz_sha256'],
                           fixture_matrix_sha256=timed['fixture_matrix_sha256'],requests=matched))
    return dict(verification='passed' if output else 'no selected requests',coefficient=str(coefficient),timed_directory=str(directory),
                protocol_sha256=checksum(protocol_path),manifest_sha256=checksum(manifest_path),
                checker_sha256=checksum(__file__),decision_checker_sha256=checksum(ROOT/'src/checker.py'),
                selection_grid=grid,missing_behaviors=missing,verified=output,
                scope='Full independent decisions on selected original files. Every recorded event field '
                      'matches the untimed prefix. Emitted ID SHA-256 and fine/anchor arrays are audit-only '
                      'and independently checked on that matched reproduction. Timed files contain ID counts.',
                checker_seconds=perf_counter()-start,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--timed-dir',type=Path,required=True)
    parser.add_argument('--coefficient',type=Fraction,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result=run(args.protocol,args.manifest,args.timed_dir,args.coefficient)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as target:
        json.dump(result,target,indent=2)
        target.write('\n')
