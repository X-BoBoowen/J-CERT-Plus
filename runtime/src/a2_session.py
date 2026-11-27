"""Same certified continuous-edit service with three candidate access paths.

The weighted Gram score, response residual certificate, comparison resolution,
and transport are inherited unchanged through EndpointKernel. Access alone
chooses the seeds and cold batches. Every accepted edit restarts current-version
coverage; D's request-root index remains valid through certified drift.
"""
import copy
import hashlib
import numpy as np
from time import perf_counter

from .common import DEN,SCALE,UNIT,Ledger,digest
from .session import energy,version_id,freeze
from .a2_catalog import AllCatalog
from .a2_numeric import EndpointKernel
from .a2_access import Access,RATIO
from .exact_ties import TieContext
from .measurement_policy import validate_policy,LightObservations
import long_promotion as lp


NUMERIC_COUNTERS=('state_rhs','candidate_rhs','winner_rhs','policy_refresh_rhs',
    'numerical_reanchor_rhs','response_blocks','response_residual_matvec_columns',
    'estimated_state_residual_matvecs','cg_iterations','gram_materializations',
    'gram_first','gram_rematerializations','gram_hits','transfers','cold_tail_queries',
    'cold_tail_ids','endpoint_checked','endpoint_excluded','live_catalog_denominator',
    'score_calls','comparison_calls','overlap_comparisons','corrected_states',
    'structural_tie_attempts','structural_tie_witnesses','structural_tie_decisions',
    'structural_tie_canonical_switches')


def make_ledger(timers):
    ledger=Ledger(timers)
    for phase in ('frontier_prepare','seed_access','candidate_access','endpoint_screen'):
        ledger.times[phase]=0.
    return ledger


class A2Session:
    def __init__(self,W,method,*,hot_size=64,timers=True,graph_hash=None,graph_name='fixture',
                 measurement_policy='legacy_deep'):
        self.measurement_policy=validate_policy(measurement_policy)
        if self.measurement_policy=='light_v1' and method not in ('H','G'):
            raise ValueError('light_v1 is validated only for the H/G measurement comparison')
        if method not in ('D','S','H','G') or type(hot_size) is not int or hot_size<0:
            raise ValueError('expected D/S/H/G and a nonnegative seed count')
        if not 1<=W.shape[0]<=82140:
            raise ValueError('bounded V2 direct-solver service admits 1..82140 vertices')
        self.method,self.hot_size,self.timers=method,hot_size,timers
        self.graph_name=graph_name
        setup=make_ledger(timers)
        with setup.phase('catalog_build'):
            self.catalog=AllCatalog(W)
            self.W=self.catalog.W
            self.n,self.m=self.catalog.n,self.catalog.m
            self.graph_hash=graph_hash or digest(np.array(self.W.shape,np.int64),
                self.W.data,self.W.indptr,self.W.indices)
            self.catalog_hash=hashlib.sha256(('all-nonedges-positive1/16+root-negative-flips:v1|'+
                                            self.graph_hash).encode()).hexdigest()
        with setup.phase('operator_assembly'):
            self.kernel=EndpointKernel(self.catalog)
            self.A=self.kernel.operator(self.W);self.A.sort_indices()
            self.operator_hash=digest(self.A.indptr,self.A.indices,self.A.data)
        if method == 'G':
            from .g_access import GAccess
            self.access=GAccess(self.catalog,method,setup,timers=timers,measurement_policy=self.measurement_policy)
        else:
            self.access=Access(self.catalog,method,setup,timers=timers,measurement_policy=self.measurement_policy)
        with setup.phase('factor_or_precond_init'):
            self.factor=lp.UpdatedFactor(self.A)
        for array in (self.A.data,self.A.indices,self.A.indptr):freeze(array)
        self.root_id=hashlib.sha256('/'.join((self.graph_hash,self.operator_hash,
                    self.catalog_hash,self.kernel.salt)).encode()).hexdigest()
        self.setup_seconds,self.setup_phases=setup.finish()

    def request(self,s,*,budget=15,delta=0,audit=False):
        if type(budget) is not int or budget<0 or delta!=0:
            raise ValueError('nonnegative edit budget and delta=0 required')
        ledger=make_ledger(self.timers)
        with ledger.phase('reset_restore'):
            if self.factor.steps:raise RuntimeError('root factor contains child updates')
            fac=copy.copy(self.factor);fac.steps=[]
            A=self.A;chosen=set();seen=set();decisions=[];snapshots=[];access_rows=[]
            counts=dict.fromkeys(NUMERIC_COUNTERS,0)
            views=self.kernel.views(ledger,counts)
        with ledger.phase('opinion_refresh'):
            s=np.asarray(s)
            if s.dtype!=np.int64 or s.shape!=(self.n,) or np.any(s < -DEN) or np.any(s>DEN):
                raise ValueError('expected bounded int64 opinion numerators')
            s=s.copy();freeze(s);op_hash=digest(s);s_float=s/DEN
        with ledger.phase('state_solve_and_cert'):
            certified_rows=[]
            state=lp.certified_state(A,fac,s,UNIT,certified_rows=certified_rows)
            counts['state_rhs']=1;counts['estimated_state_residual_matvecs']=1
            root_energy=energy(state,s)
        initial_order=self.access.begin(*state[3:],self.hot_size,ledger)
        hot=set(map(int,initial_order));initial=frozenset(hot)
        current=self.root_id;status='budget_stop';reason=None
        light=LightObservations() if self.measurement_policy=='light_v1' else None
        if light is not None:
            light.observe('request_begin',0,self.access,views,fac,hot,seen,chosen,decisions,access_rows)
        total_lo=total_hi=0
        pre_online=ledger.times.copy();online_start=perf_counter()
        peak_views=peak_steps=0
        for t in range(budget):
            if self.method=='G':
                with ledger.phase('cache_management'):
                    hot=self.access.warm_candidates(hot)
            z,x,r,xc,rc=state
            parent=current;best=0;winner=None;evaluated={};resolution={}
            skipped_upper=None;endpoint_survivors=0;ids_before=0
            overlap=False;seeds=np.asarray(sorted(hot),np.int64)
            strict=None;work_recorded=False;cold_started=False;cold_start_best=None

            def skipped(bound):
                nonlocal skipped_upper
                bound=int(bound)
                skipped_upper=bound if skipped_upper is None else max(skipped_upper,bound)

            def refine(ids,upper):
                nonlocal best,winner,peak_views
                with ledger.phase('comparison/refinement'):
                    for pos in np.argsort(-upper,kind='stable'):
                        if int(upper[pos])<best:
                            skipped(upper[pos]);break
                        e=int(ids[pos]);views.scored_response=None
                        existed=e in views.entries
                        views.role='numerical_reanchor' if existed else 'candidate'
                        try:
                            lo,hi=views.score(e,t,A,fac,x,r)
                        finally:
                            views.role='candidate'
                            peak_views=max(peak_views,len(views.entries))
                        counts['score_calls']+=1
                        if not existed:counts['gram_rematerializations' if e in seen else 'gram_first']+=1
                        seen.add(e);hot.add(e);evaluated[e]=hi
                        if lo>best or (winner is not None and lo==best and e<winner):
                            best,winner=lo,e;views.retained=views.scored_response
                        views.scored_response=None

            try:
                # Establish current-version access state before any warm score
                # can fail. Failed requests must never reuse a prior event.
                with ledger.phase('candidate_access'):
                    self.access.open_version(xc,rc,chosen,hot|chosen,t+1,parent,ledger)
                    if light is not None:
                        light.observe('version_open',t+1,self.access,views,fac,hot,seen,chosen,decisions,access_rows)
                with ledger.phase('endpoint_screen'):
                    strict=self.kernel.strict(A,s,x,certified_rows[0])
                    ids,uppers=strict.screen(seeds,best)
                    endpoint_survivors+=len(ids)
                    if len(ids)<len(seeds):skipped(max(0,best-1))
                refine(ids,uppers)
                cold_start_best=best
                cold_started=True
                while True:
                    with ledger.phase('candidate_access'):
                        batch=self.access.next_batch(best,hot|chosen)
                    if batch is None:break
                    count=len(batch);ids_before+=count
                    before=best
                    with ledger.phase('endpoint_screen'):
                        ids,uppers=strict.screen(batch,best)
                        endpoint_survivors+=len(ids)
                        if len(ids)<count:skipped(max(0,best-1))
                    refine(ids,uppers)
                    del batch
                access=self.access.finish_version()
                if light is not None:
                    light.observe('version_finish',t+1,self.access,views,fac,hot,seen,chosen,decisions,access_rows)
                access['cold_start_lower_fine']=int(cold_start_best)
                access['cold_end_lower_fine']=int(best)
                access['warm_screen_rows']=len(seeds)
                remaining=access['remaining_upper_coarse']
                if remaining is not None:skipped(int(remaining)*RATIO)
                access_rows.append(access)
                counts['endpoint_checked']+=strict.rows
                counts['endpoint_excluded']+=strict.rows-endpoint_survivors
                counts['live_catalog_denominator']+=self.m-len(chosen)
                counts['cold_tail_queries']+=1;counts['cold_tail_ids']+=ids_before
                work_recorded=True
                with ledger.phase('comparison/refinement'):
                    counts['comparison_calls']+=1
                    overlap=any(hi>=best for e,hi in evaluated.items() if e!=winner)
                    counts['overlap_comparisons']+=int(overlap)
                    try:
                        winner,best=self.kernel.resolve(A,s_float,evaluated,winner,best,
                            tie_context=TieContext(A,s,parent,frozenset(chosen)),evidence=resolution)
                    finally:
                        if resolution:
                            counts['structural_tie_attempts']+=len(resolution['overlap_ids'])
                            counts['structural_tie_witnesses']+=len(resolution['witnesses'])
                            done='canonical_winner_id' in resolution
                            counts['structural_tie_decisions']+=int(done)
                            counts['structural_tie_canonical_switches']+=int(done and
                                resolution['canonical_winner_id']!=resolution['provisional_winner_id'])
                rivals=max((hi*SCALE for e,hi in evaluated.items() if e!=winner),default=None)
                cold=None if skipped_upper is None else skipped_upper*SCALE
                bounds=[b for b in (rivals,cold) if b is not None]
                decision=dict(epoch=t+1,root_version_id=self.root_id,parent_version_id=parent,
                    current_version_id=parent,winner_id=winner,low=best*SCALE,
                    high=None if winner is None else evaluated[winner]*SCALE,
                    scored_rival_max_high=rivals,cold_upper=cold,
                    competitor_max_upper=max(bounds,default=None),proof_scope='full_catalog',
                    exact_overlap_resolution=overlap,live_catalog=self.m-len(chosen),
                    endpoint_checked=strict.rows,endpoint_excluded=strict.rows-endpoint_survivors,
                    scored=len(evaluated),seed_visits=len(seeds),seed_ids=seeds.tolist(),
                    cold_endpoint_visits=ids_before)
                if resolution:decision['comparison_resolution']=resolution
                if audit:
                    snapshots.append(dict(epoch=t+1,parent_version_id=parent,x=x.tolist(),r=int(r),
                        xc=xc.tolist(),rc=int(rc),chosen=sorted(chosen),
                        warm_ids=seeds.tolist(),cold_start_lower_fine=int(cold_start_best),
                        winner_id=winner))
                if winner is None:
                    decision['status']='tolerance_stop';decisions.append(decision)
                    status='tolerance_stop';break
                if best<=0:raise ArithmeticError('unresolved positive gain on proof grid')
                i,j,flip,d=self.catalog.scalar(winner)
                decision.update(i=int(i),j=int(j),flip=int(flip),d=int(d),outsider=winner not in initial)
                with ledger.phase('cache_management'):hot.discard(winner)
                with ledger.phase('transport_local'):
                    views.role='winner'
                    zchild=views.advance(winner,t,A,fac,z)
                    views.role='candidate'
                    Achild=self.kernel.child(A,winner)
                with ledger.phase('state_solve_and_cert'):
                    child,source=lp.certify_child(Achild,fac,s,zchild,UNIT,certified_rows)
                    counts['estimated_state_residual_matvecs']+=1+int(source=='corrected')
                    counts['corrected_states']+=int(source=='corrected')
                    counts['state_rhs']+=int(source=='corrected')
                A,state=Achild,child;chosen.add(winner);current=version_id(parent,winner)
                decision.update(status='accepted',current_version_id=current,
                    child_state_source=source,child_state_residual=int(state[2]))
                decisions.append(decision)
                total_lo+=best*SCALE;total_hi+=evaluated[winner]*SCALE
                peak_views=max(peak_views,len(views.entries))
                if self.measurement_policy=='legacy_deep':
                    peak_steps=max(peak_steps,sum(y.nbytes+k.nbytes for y,k in fac.steps))
            except (ArithmeticError,RuntimeError,MemoryError) as exc:
                reason=f'{type(exc).__name__}: {exc}'
                status=('unresolved' if 'unresolved' in str(exc).lower() or 'overlap' in str(exc).lower()
                        else 'memory_error' if isinstance(exc,MemoryError) else 'error')
                if len(access_rows)<t+1:
                    access=self.access.finish_version()
                    if light is not None:
                        light.observe('version_finish',t+1,self.access,views,fac,hot,seen,chosen,decisions,access_rows)
                    access.update(epoch=t+1,parent_version_id=parent,source=self.method,source_complete=False,
                        warm_screen_rows=0 if strict is None else min(strict.rows,len(seeds)),
                        cold_start_lower_fine=cold_start_best,cold_end_lower_fine=int(best))
                    access_rows.append(access)
                if not work_recorded:
                    counts['endpoint_checked']+=0 if strict is None else strict.rows
                    counts['endpoint_excluded']+=0 if strict is None else strict.rows-endpoint_survivors
                    counts['live_catalog_denominator']+=self.m-len(chosen)
                    counts['cold_tail_queries']+=int(cold_started)
                    counts['cold_tail_ids']+=ids_before
                failed=dict(epoch=t+1,root_version_id=self.root_id,parent_version_id=parent,
                    current_version_id=parent,status=status,winner_id=None,failure_reason=reason)
                if resolution:failed['comparison_resolution']=resolution
                decisions.append(failed)
                break
        online_seconds=perf_counter()-online_start
        online_phases={key:value-pre_online[key] for key,value in ledger.times.items()}
        online_phases['other']+=online_seconds-sum(online_phases.values())
        with ledger.phase('opinion_refresh'):final_energy=energy(state,s)
        counts.update(gram_materializations=views.fresh,gram_hits=views.hits,transfers=views.transfers)
        response_bytes=0 if views.retained is None else sum(a.nbytes for a in views.retained[1][:3])
        if light is not None:
            light.observe('request_end',len(decisions),self.access,views,fac,hot,seen,chosen,decisions,access_rows)
        memory=self.access.memory()
        memory.update(sample='last committed state before close plus observed payload maxima',
            graph_csr=sum(a.nbytes for a in (self.W.data,self.W.indices,self.W.indptr)),
            root_operator_csr=sum(a.nbytes for a in (self.A.data,self.A.indices,self.A.indptr)),
            sparse_catalog=self.catalog.memory() if light is None else None,root_factor_reported_nnz=self.factor.base.nnz,
            gram_count=len(views.entries),peak_gram_count=max(peak_views,len(views.entries)),
            gram_python_bytes=lp.store_bytes(views.entries) if light is None else None,
            actual_candidate_cache_python_bytes=lp.store_bytes(dict(grams=views.entries,hot=hot,seen=seen,chosen=chosen)) if light is None else None,
            decision_and_access_log_python_bytes=lp.store_bytes((decisions,access_rows)) if light is None else None,
            shared_response_bytes=response_bytes,root_factor_resident_bytes=None,
            low_rank_steps=sum(y.nbytes+k.nbytes for y,k in fac.steps) if light is None else None,
            peak_low_rank_steps=max(peak_steps,sum(y.nbytes+k.nbytes for y,k in fac.steps)) if light is None else None,
            canonical_id_storage='computed int64 endpoint ID; caches only actual evaluated candidates')
        if light is not None:
            memory.update(sample='fixed-count observations at actual begin/open/finish/end; deep watermarks unobserved',
                memory_observation_schema='light_v1',light_observations=light.rows,
                low_rank_step_count=len(fac.steps),
                catalog_array_payloads=dict(root_edge_ids=int(self.catalog.root_edge_ids.nbytes),
                    positive_ids=int(self.catalog.positive_ids.nbytes),flip_ids=int(self.catalog.flip_ids.nbytes),
                    flip_weights=int(self.catalog.flip_weights.nbytes)),
                catalog_root_edge_key_items=len(self.catalog.root_edge_keys))
        index_phases=self.access.index.build_phases.copy() if self.access.index is not None else {}
        with ledger.phase('reset_restore'):
            self.access.close()
            del views,fac,A,state,certified_rows
        result=dict(root_version_id=self.root_id,current_version_id=current,opinion_hash=op_hash,
            measurement_policy=self.measurement_policy,memory_observation_schema=self.measurement_policy,
            method=self.method,budget=budget,delta=delta,status=status,failure_reason=reason,
            initial_seed_ids=sorted(initial),initial_seed_order=initial_order.tolist(),
            seed_rule=self.access.root_seed_rule,initial_gram_count=0,decisions=decisions,
            commits=len(chosen),gain_low=total_lo,gain_high=total_hi,gain_denominator=UNIT**2,
            root_energy=root_energy,final_energy=final_energy,counters=counts,memory=memory,
            access=dict(epochs=access_rows),index_build_phases=index_phases,
            scope_checker='pending selected independent full-catalog replay',
            cache_policy='fixed structural root; empty request Grams; answers disabled')
        if audit:result.update(snapshots=snapshots,opinion_numerators=s.tolist())
        result['request_seconds'],result['phases']=ledger.finish()
        result['online_seconds']=online_seconds;result['online_phases']=online_phases
        result['reset_opinion_seconds']=result['request_seconds']-online_seconds
        return result
