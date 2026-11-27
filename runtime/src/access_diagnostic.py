"""Four preregistered same-context discovery diagnostics; never a stitched oracle."""
import argparse
import copy
import json
import os
import resource
from pathlib import Path
from time import perf_counter
from types import MethodType

import numpy as np
from scipy.sparse import load_npz
from threadpoolctl import threadpool_limits

from .anchor_session import AnchorSession, Cursor, plan, gather
from .common import ROOT, dump, file_hash
from .weighted_adapter import Index, Strict

FROZEN = ROOT.parent / 'frozen/J_CERT_Plus_VLDB_Research_20261124'
CASES = (
    ('Alpha', 'independent', 2, '004_Alpha_independent_adaptive.json'),
    ('OTC', 'independent', 9, '013_OTC_independent_adaptive.json'),
    ('Alpha', 'slow', 2, '001_Alpha_slow_retain-always.json'),
    ('OTC', 'slow', 9, '010_OTC_slow_retain-always.json'),
)


def rows(graph, stream):
    with np.load(ROOT/f'data/confirmation_196613/{graph}_{stream}.npz') as z:
        return z['opinions']


def initialize(W, matrix, origin, position, arm, graph_hash):
    sess = AnchorSession(W, arm, graph_hash=graph_hash)
    sess.bridge.position = origin - 1
    sess.request(matrix[origin-1], budget=0)
    sess.bridge.position = position - 1
    return sess


def restore(W, matrix, ref, origin, position, graph_hash):
    """Capture the exact contexts of the retained path, outside measured regions."""
    start = perf_counter()
    sess = initialize(W, matrix, origin, position, 'retain-always', graph_hash)
    contexts = []
    original_query, original_strict = Cursor.query, sess.bridge.strict
    active = {}

    def strict(A, s, x, certified_rows):
        value = original_strict(A, s, x, certified_rows)
        active['strict'] = value
        old = value.screen
        calls = 0

        def screen(ids, best):
            nonlocal calls
            result = old(ids, best)
            if calls == 1:
                contexts[-1]['fine_target'] = int(best)
                contexts[-1]['strict_ids'] = result[0].copy()
                contexts[-1]['strict_upper'] = result[1].copy()
            calls += 1
            return result
        value.screen = screen
        return value

    def query(cursor, base_drift, target, excluded):
        if not contexts:
            active['caps'] = cursor.bridge.current_caps.copy()
            active['fresh'] = Index(cursor.bridge.kernel, cursor.x, cursor.r,
                                    cursor.inc, cursor.degree, caps=active['caps'])
        old = cursor.bridge.anchor.index
        result = original_query(cursor, base_drift, target, excluded)
        contexts.append(dict(old=old, fresh=active['fresh'],
            old_drift=cursor.anchor_drift(base_drift).copy(), fresh_drift=base_drift.copy(),
            target=int(target), excluded=set(excluded), strict=active['strict'],
            returned=result[0].copy(), root_x=cursor.x, root_r=cursor.r))
        return result

    sess.bridge.strict = strict
    Cursor.query = query
    try:
        got = sess.request(matrix[position-1], budget=15, audit=False)
    finally:
        Cursor.query = original_query
    if got['status'] != ref['status'] or got['initial_seed_ids'] != ref['initial_seed_ids']:
        raise AssertionError('restoration root/status mismatch')
    for old, new in zip(ref['decisions'], got['decisions'], strict=True):
        for field in ('winner_id','parent_version_id','current_version_id','endpoint_checked',
                      'seed_visits','cold_endpoint_visits','low','high','tau'):
            if old.get(field) != new.get(field):
                raise AssertionError(('restored decision mismatch', field, old, new))
    for old, new in zip(ref['discovery']['epochs'], got['discovery']['epochs'], strict=True):
        for field in ('target','tau','max_drift','pre_dedup_visits','returned_ids','anchor_request'):
            if old[field] != new[field]:
                raise AssertionError(('restored query mismatch', field, old, new))
    return sess, contexts, active['caps'], perf_counter()-start


def profile(index, drift, target, excluded, screen, fine_target):
    """Execute the same two np.unique operations as frozen gather, separating them."""
    t = perf_counter()
    probe = plan(index, drift, target)
    plan_s = perf_counter()-t
    tau = probe.choices[probe.best]
    t = perf_counter()
    moving = np.flatnonzero(drift > tau)
    pieces = []
    for v in moving:
        pieces.append(np.arange(index.inc.left[v], index.inc.left[v+1], dtype=index.inc.ids.dtype))
        pieces.append(index.inc.ids[index.inc.right[v]:index.inc.right[v+1]])
    post_raw = sum(len(x) for x in pieces)
    post_s = perf_counter()-t
    t = perf_counter()
    posts = np.unique(np.concatenate(pieces)) if pieces else index.inc.ids[:0]
    inner_dedup_s = perf_counter()-t
    t = perf_counter()
    tails = [ids[cut:] for (_,_,ids),cut in zip(index.groups,probe.cuts[probe.best])]
    tail_counts = [len(x) for x in tails]
    tail_s = perf_counter()-t
    t = perf_counter()
    unique = np.unique(np.concatenate([posts]+tails))
    before_exclusion = len(unique)
    ids = np.setdiff1d(unique,np.asarray(sorted(excluded),np.int32),assume_unique=True) if excluded else unique
    outer_dedup_s = perf_counter()-t
    t = perf_counter()
    survivors, upper = Strict.screen(screen, ids, fine_target)
    strict_s = perf_counter()-t
    raw = post_raw+sum(tail_counts)
    if raw != probe.costs[probe.best]:
        raise AssertionError('raw plan disagrees with gathered slices')
    return dict(plan=plan_s, posting_gather=post_s, tail_gather=tail_s,
        concatenate_deduplicate_exclude=inner_dedup_s+outer_dedup_s,
        posting_inner_dedup=inner_dedup_s, strict_endpoint_screen=strict_s,
        raw_visits=raw, posting_raw_visits=post_raw, tail_raw_visits=sum(tail_counts),
        tail_lengths_by_weight=[dict(weight=g[0],length=n) for g,n in zip(index.groups,tail_counts)],
        unique_before_exclusion=before_exclusion, unique_after_exclusion=len(ids),
        survivors=len(survivors), target=int(target), tau=int(tau), max_drift=int(drift.max())), ids, survivors, upper


def build_samples(kernel, x, r, inc, degree, caps, repeats):
    results=[]
    for rep in range(repeats):
        t=perf_counter()
        legacy_members=tuple((int(d),np.flatnonzero(kernel.d==d).astype(np.int32)) for d in np.unique(kernel.d))
        regroup_s=perf_counter()-t
        if any(d!=d0 or not np.array_equal(a,a0) for (d,a),(d0,a0) in
               zip(legacy_members,kernel.prepare_weight_groups().members,strict=True)):
            raise AssertionError('structural membership mismatch')
        t=perf_counter()
        current=Index(kernel,x,r,inc,degree,caps=caps)
        elapsed=perf_counter()-t
        results.append(dict(repeat=rep, repeated_grouping_removed=regroup_s,
                            necessary_build=elapsed, **current.build_phases))
    return results


def run(repeats):
    os.sched_setaffinity(0,{0})
    start=perf_counter()
    out=dict(scope='four fixed same-context discovery diagnostics; not a full-flow oracle',
             repeats=repeats,cases=[],affinity=sorted(os.sched_getaffinity(0)),
             tail_accounting='tail_gather creates slices; tail payload copying is in concatenate/deduplicate/exclude',
             raw_definition='posting degree sum plus all selected group-tail lengths, before both unique operations')
    measured=0.
    for graph,stream,q,filename in CASES:
        original=FROZEN/'raw/a1_confirmation'/filename
        trace=json.loads(original.read_text())
        reference=trace['requests'][q-1]
        origin=reference['discovery']['anchor_request']
        W=load_npz(ROOT/f'data/projected/{graph}_weighted.npz')
        graph_hash=file_hash(ROOT/f'data/projected/{graph}_weighted.npz')
        if graph_hash != trace['graph_hash']:
            raise AssertionError('frozen graph identity mismatch')
        matrix=rows(graph,stream)
        from .common import digest
        if digest(matrix[q-1]) != reference['opinion_hash']:
            raise AssertionError('frozen opinion identity mismatch')
        sess,contexts,caps,recovery_s=restore(W,matrix,reference,origin,q,graph_hash)
        first=contexts[0]
        t=perf_counter()
        builds=build_samples(sess.bridge.kernel,first['root_x'],first['root_r'],
                             sess.inc,sess.degree,caps,repeats)
        stage_rows=[]
        for rep in range(repeats):
            for epoch,c in enumerate(contexts,1):
                modes=('old','fresh') if (rep+epoch)%2 else ('fresh','old')
                for mode in modes:
                    sample,ids,survivors,upper=profile(c[mode],c[mode+'_drift'],c['target'],
                                           c['excluded'],c['strict'],c['fine_target'])
                    if mode=='old':
                        np.testing.assert_array_equal(ids,c['returned'])
                        np.testing.assert_array_equal(survivors,c['strict_ids'])
                        np.testing.assert_array_equal(upper,c['strict_upper'])
                    else:
                        canonical=gather(c[mode],c[mode+'_drift'],plan(c[mode],c[mode+'_drift'],c['target']),c['excluded'])[0]
                        np.testing.assert_array_equal(ids,canonical)
                    stage_rows.append(dict(repeat=rep,epoch=epoch,source=mode,**sample))
        measured+=sum(b['necessary_build']+b['repeated_grouping_removed'] for b in builds)
        measured+=sum(sum(r[k] for k in ('plan','posting_gather','tail_gather',
                     'concatenate_deduplicate_exclude','strict_endpoint_screen')) for r in stage_rows)
        # Full request measurements are real executions. State setup is outside request cost,
        # separately reported; no times from different execution histories are spliced.
        full=[]
        for rep in range(repeats):
            modes=('rebuild-always','retain-always') if rep%2 else ('retain-always','rebuild-always')
            for arm in modes:
                setup_t=perf_counter()
                service=initialize(W,matrix,origin,q,arm,graph_hash)
                setup_s=perf_counter()-setup_t
                t=perf_counter()
                r=service.request(matrix[q-1],budget=15)
                measured+=perf_counter()-t
                if [d['winner_id'] for d in r['decisions']] != [d['winner_id'] for d in reference['decisions']]:
                    raise AssertionError('full comparison output mismatch')
                full.append(dict(repeat=rep,arm=arm,setup_restore_seconds=setup_s,
                    request_seconds=r['request_seconds'],phases=r['phases'],
                    build_phases=r.get('index_build_phases'),discovery=r['discovery'],
                    counters=r['counters'],memory=r['memory']))
                del service
        out['cases'].append(dict(graph=graph,stream=stream,position=q,origin=origin,
            frozen_trace=filename,frozen_trace_sha256=file_hash(original),restoration_verified=True,
            recovery_seconds=recovery_s,catalog_size=sess.m,n=W.shape[0],
            structural_grouping_seconds=sess.setup_phases['structural_weight_grouping'],
            structural_grouping_bytes=sess.weight_group_bytes,root_cap_retention_bytes=caps.nbytes,
            builds=builds,stage_samples=stage_rows,full_requests=full))
        print(graph,stream,q,'complete',flush=True)
        dump(ROOT/'raw/access_diagnostic.json',out)
        if measured>300: raise RuntimeError('registered diagnostic timing budget exceeded')
    out['measured_region_seconds']=measured
    out['elapsed_including_recovery_validation_setup_seconds']=perf_counter()-start
    out['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    out['source_sha256']={f:file_hash(ROOT/'src'/f) for f in
        ('access_diagnostic.py','weighted_adapter.py','anchor_session.py','session.py')}
    dump(ROOT/'raw/access_diagnostic.json',out)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    with threadpool_limits(limits=1): run(args.repeats)
