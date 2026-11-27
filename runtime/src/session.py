"""Persistent unit-weight service on the frozen signed-promotion numerical primitives."""
import copy
import hashlib
from time import perf_counter

from .common import DEN, SCALE, UNIT, Ledger, digest, legacy
import numpy as np
from scipy import sparse
import long_promotion as lp
from promotion_directional import directional_block
from promotion_index_profile import FusedClassIndex
from promotion_strict import StrictScreen
from universe_index import build_incidence, build_universe
from workloads import system


def energy(state,s):
    _,x,r,_,_=state
    # The integer residual radius encloses the scaled state. Physical energy
    # is SCALE*s'z because the retained factor represents SCALE*A.
    mid=sum(int(a)*int(b) for a,b in zip(x,s))*SCALE
    rad=int(np.abs(s).sum())*int(r)*SCALE
    return dict(low=mid-rad,high=mid+rad,denominator=UNIT*DEN)


def version_id(parent,e):
    return hashlib.sha256(f'{parent}/{e}'.encode()).hexdigest()


class PersistentIndex(FusedClassIndex):
    """Frozen native class bins per opinion, immutable incidence per catalog."""
    def __init__(self,lib,pool,x,r,inc,degree):
        self.lib,self.scale=lib,SCALE
        self.root_input=tuple(pool),x,r
        self.fixed_tau=None
        self.n,self.m=len(x),len(pool[0])
        self.root_norm=int(np.abs(x).max())+int(r)
        if self.m>=2**31 or self.root_norm>2*lp.SCREEN:
            raise ValueError('unsupported root or catalog for native index')
        self.offset=4*self.root_norm**2
        self.width=1<<max(0,(2*self.offset).bit_length()-15)
        self.bins=2*self.offset//self.width+1
        args=[self.m,*(a.ctypes.data for a in (*pool,x)),r,self.offset,self.width,self.bins]
        counts=np.zeros(2*self.bins,np.int32)
        lib.index_pass(*args,counts.ctypes.data,None)
        self.offsets=np.zeros(len(counts)+1,np.int32)
        np.cumsum(counts,out=self.offsets[1:])
        if self.offsets[-1]!=self.m:
            raise ArithmeticError('class bins do not cover catalog')
        self.order=np.empty(self.m,np.int32)
        positions=self.offsets[:-1].copy()
        lib.index_pass(*args,positions.ctypes.data,self.order.ctypes.data)
        self.inc,self.degree=inc,degree
        self.stamps=None
        self.epoch=0
        self.nbytes=sum(y.nbytes for y in (self.order,self.offsets,degree,inc.left,inc.right,inc.ids))


class MeasuredViews(lp.Views):
    def __init__(self,pool,ledger,counters):
        super().__init__(pool,UNIT,SCALE)
        self.ledger,self.counts=ledger,counters
        self.directional_block=directional_block
        self.retain_winner=True
        self.role='candidate'
        self.scored_response=self.retained=None

    def response(self,e,A,fac):
        reused=self.retained is not None and self.retained[0]==e
        with self.ledger.phase('response_acquisition'):
            result=super().response(e,A,fac)
        if not reused:
            rhs=1+int(self.pool[2][e])
            self.counts[self.role+'_rhs']+=rhs
            self.counts['response_blocks']+=1
            self.counts['response_residual_matvec_columns']+=rhs
        return result


def freeze(a):
    while isinstance(a,np.ndarray):
        a.flags.writeable=False
        a=a.base


class Session:
    def __init__(self,W,method,*,hot_size=64,timers=True,graph_name='fixture',ledger=None,
                 graph_hash=None,weight_mode='unit'):
        if method not in ('F','T','I','D','S') or hot_size<0:
            raise ValueError('unsupported lifecycle method or shortlist size')
        self.method,self.hot_size,self.timers=method,hot_size,timers
        self.indexed=method in ('I','D')
        self.transport=method in ('T','D','S')
        if weight_mode not in ('unit','weighted'):
            raise ValueError('unrecognized weight mode')
        kernel_type=None
        if weight_mode=='weighted':
            from .weighted_adapter import Kernel
            kernel_type=Kernel
        setup=ledger or Ledger(timers)
        with setup.phase('graph_load'):
            self.W=W=W.tocsr(copy=True)
            W.sum_duplicates()
            W.eliminate_zeros()
            W.sort_indices()
            if W.shape[0]!=W.shape[1] or W.shape[0]>10000 or W.shape[0]<1:
                raise ValueError('only direct-solver graphs with 1..10000 nodes are admitted')
            if kernel_type is not None:
                kernel_type.validate(W)
            elif ((W!=W.T).nnz or np.any(W.diagonal()) or not np.all(abs(W.data)==1)):
                raise ValueError('expected undirected simple signed unit graph')
            self.graph_hash=graph_hash or digest(np.array(W.shape,np.int64),W.data,W.indptr,W.indices)
        with setup.phase('catalog_build'):
            self.pool=build_universe(W)
            self.catalog_hash=digest(*self.pool)
            self.m=len(self.pool[0])
            if self.m>=2**31:
                raise ValueError('catalog exceeds native ID range')
        with setup.phase('operator_assembly'):
            self.kernel=kernel_type(W,self.pool) if kernel_type else None
            self.A=self.kernel.operator(W) if self.kernel else (SCALE*system(W)).astype(np.int64)
            self.A.sort_indices()
            self.diag=self.A.diagonal()
            self.operator_hash=digest(self.A.indptr,self.A.indices,self.A.data)
        self.weight_groups=None
        self.weight_group_bytes=0
        if self.indexed and self.kernel:
            setup.times.setdefault('structural_weight_grouping',0.)
            with setup.phase('structural_weight_grouping'):
                self.weight_groups=self.kernel.prepare_weight_groups()
                self.pool=self.weight_groups.pool
                self.weight_group_bytes=self.weight_groups.nbytes
        with setup.phase('factor_or_precond_init'):
            self.factor=lp.UpdatedFactor(self.A)
        self.broad,self.lib=legacy.native()
        self.inc=self.degree=None
        if self.indexed:
            with setup.phase('posting/index_build'):
                self.inc=build_incidence(*self.pool[:2],W.shape[0])
                self.inc.left=np.ascontiguousarray(self.inc.left,dtype=np.int64)
                self.inc.right=np.ascontiguousarray(self.inc.right,dtype=np.int64)
                self.degree=np.diff(self.inc.left)+np.diff(self.inc.right)
                for a in (self.inc.left,self.inc.right,self.inc.ids,self.degree):
                    freeze(a)
        for a in (*self.pool,self.diag,W.data,W.indices,W.indptr,self.A.data,self.A.indices,self.A.indptr):
            freeze(a)
        self.root_id=hashlib.sha256(('/'.join((self.graph_hash,self.operator_hash,self.catalog_hash,
                                   'lexical-twohop-negative:v1',self.kernel.salt if self.kernel else
                                   'superlu-lowrank:scale16:proof48:screen20'))).encode()).hexdigest()
        self.setup_seconds,self.setup_phases=setup.finish()

    def request(self,s,*,budget=15,delta=0,audit=False):
        if type(budget) is not int or budget<0 or delta!=0:
            raise ValueError('this registered path requires nonnegative budget and delta=0')
        ledger=Ledger(self.timers)
        with ledger.phase('reset_restore'):
            if self.weight_groups is not None:
                if self.pool is not self.kernel.pool:
                    raise ValueError('weight group binding mismatch')
                self.weight_groups.check_binding(self.pool,self.kernel.d)
            if self.factor.steps:
                raise RuntimeError('structural root has low-rank child steps')
            fac=copy.copy(self.factor)
            fac.steps=[]
            A=self.A
            chosen,decisions,seen=set(),[],set()
            counts=dict.fromkeys(('state_rhs','candidate_rhs','winner_rhs','policy_refresh_rhs',
                'numerical_reanchor_rhs','response_blocks','response_residual_matvec_columns',
                'estimated_state_residual_matvecs','cg_iterations','gram_materializations','gram_first',
                'gram_rematerializations','gram_hits','transfers','cold_tail_queries','cold_tail_ids',
                'endpoint_checked','endpoint_excluded','live_catalog_denominator','score_calls',
                'comparison_calls','overlap_comparisons','corrected_states'),0)
            views=self.kernel.views(ledger,counts) if self.kernel else MeasuredViews(self.pool,ledger,counts)
        with ledger.phase('opinion_refresh'):
            s=np.asarray(s)
            if s.dtype!=np.int64 or s.shape!=(A.shape[0],) or np.any(s< -DEN) or np.any(s>DEN):
                raise ValueError('expected bounded int64 opinion numerator')
            s=s.copy()
            freeze(s)
            op_hash=digest(s)
            s_float=s/DEN
        with ledger.phase('state_solve_and_cert'):
            certified_rows=[]
            state=lp.certified_state(A,fac,s,UNIT,certified_rows=certified_rows)
            counts['state_rhs']+=1
            counts['estimated_state_residual_matvecs']+=1
            root_state=state
            root_energy=energy(state,s)
        xc0,rc0=state[3:]
        with ledger.phase('caps_build'):
            caps=(self.kernel.root_caps(xc0,rc0) if self.kernel else
                  lp.cap(self.pool,slice(None),xc0,rc0,self.diag,False,SCALE))
        with ledger.phase('caps_sort'):
            k=min(self.hot_size,self.m)
            if k:
                cutoff=np.partition(caps,self.m-k)[-k]
                above=np.flatnonzero(caps>cutoff)
                first=np.r_[above,np.flatnonzero(caps==cutoff)[:k-len(above)]]
                hot=set(map(int,first))
            else:
                hot=set()
            if self.weight_groups is None:
                del caps
            initial=frozenset(hot)
        index=None
        if self.indexed:
            with ledger.phase('posting/index_build'):
                if self.kernel:
                    index=self.kernel.index(xc0,rc0,self.inc,self.degree,caps=caps,timers=self.timers)
                    del caps
                else:
                    index=PersistentIndex(self.lib,self.pool,xc0,rc0,self.inc,self.degree).fork_cursor()
        current=self.root_id
        status,reason='budget_stop',None
        total_lo=total_hi=0
        snapshots=[]
        pre_online_phases=ledger.times.copy()
        online_start=perf_counter()
        for t in range(budget):
            z,x,r,xc,rc=state
            parent=current
            best,winner=0,None
            evaluated={}
            live=self.m-t
            ids_before=0
            skipped_upper=None
            overlap=False
            def skipped(bound):
                nonlocal skipped_upper
                skipped_upper=bound if skipped_upper is None else max(skipped_upper,bound)
            def refine(ids,upper,cold):
                nonlocal best,winner
                with ledger.phase('comparison/refinement'):
                    for j in np.argsort(-upper,kind='stable'):
                        if int(upper[j])<best:
                            skipped(int(upper[j]))
                            break
                        e=int(ids[j])
                        views.scored_response=None
                        existed=e in views.entries
                        # A fresh acquisition after F/I discarded this Gram is a
                        # policy refresh. A solve on a still-present view can
                        # only be a numerical reanchor in this score call.
                        views.role=('numerical_reanchor' if existed else
                                    'policy_refresh' if e in seen and not self.transport else 'candidate')
                        try:
                            lo,hi=views.score(e,t,A,fac,x,r)
                        finally:
                            views.role='candidate'
                        counts['score_calls']+=1
                        if not existed:
                            counts['gram_rematerializations' if e in seen else 'gram_first']+=1
                        seen.add(e)
                        evaluated[e]=hi
                        if cold and self.indexed:
                            hot.add(e)
                        if lo>best or (winner is not None and lo==best and e<winner):
                            best,winner=lo,e
                            views.retained=views.scored_response
                        views.scored_response=None
            try:
                with ledger.phase('discovery'):
                    strict=(self.kernel.strict(A,s,x,certified_rows[0]) if self.kernel else
                            StrictScreen(A,s,x,UNIT,SCALE,self.pool,library=self.broad.library,
                                         certified_rows=certified_rows[0]))
                    seeds=np.array(sorted(hot),np.int32)
                    ids,upper=strict.screen(seeds,best)
                    endpoint_survivors=len(ids)
                    if len(ids)<len(seeds):
                        skipped(best-1)
                refine(ids,upper,False)
                if self.method!='S':
                    with ledger.phase('discovery'):
                        excluded=hot|chosen
                        target=best
                        if self.indexed:
                            ratio=(UNIT//lp.SCREEN)**2
                            drift=abs(xc-xc0)+rc+rc0
                            todo,choices,costs,tau=index.query(drift,(best+ratio-1)//ratio,excluded)
                            ids_before=len(todo)
                            if ids_before<live-len(seeds):
                                skipped(((target+ratio-1)//ratio-1)*ratio)
                            ids,upper=strict.screen(todo,best)
                            counts['cold_tail_queries']+=1
                            counts['cold_tail_ids']+=ids_before
                        else:
                            blocks=[]
                            exc=np.array(sorted(excluded),np.int32)
                            for a in range(0,self.m,lp.BLOCK):
                                block=strict.scan(a,min(a+lp.BLOCK,self.m),best,exc)
                                if len(block[0]):
                                    blocks.append(block)
                            ids=np.concatenate([b[0] for b in blocks]) if blocks else np.empty(0,np.int32)
                            upper=np.concatenate([b[1] for b in blocks]) if blocks else np.empty(0,dtype=object)
                            ids_before=live-len(seeds)
                        if len(ids)<ids_before:
                            skipped(best-1)
                        endpoint_survivors+=len(ids)
                    refine(ids,upper,True)
                counts['endpoint_checked']+=strict.rows
                counts['endpoint_excluded']+=strict.rows-endpoint_survivors
                counts['live_catalog_denominator']+=live
                with ledger.phase('comparison/refinement'):
                    counts['comparison_calls']+=1
                    overlap=any(hi>=best for e,hi in evaluated.items() if e!=winner)
                    counts['overlap_comparisons']+=int(overlap)
                    winner,best=(self.kernel.resolve(A,s_float,evaluated,winner,best) if self.kernel else
                                 lp.resolve_weighted(A,s_float,self.pool,evaluated,winner,best,UNIT,SCALE))
                scored_rival=max((hi*SCALE for e,hi in evaluated.items() if e!=winner),default=None)
                cold_upper=None if skipped_upper is None else skipped_upper*SCALE
                bounds=[b for b in (scored_rival,cold_upper) if b is not None]
                d=dict(epoch=t+1,root_version_id=self.root_id,parent_version_id=parent,
                    current_version_id=parent,winner_id=winner,low=best*SCALE,
                    high=None if winner is None else evaluated[winner]*SCALE,
                    scored_rival_max_high=scored_rival,cold_upper=cold_upper,
                    competitor_max_upper=max(bounds,default=None),proof_scope='shortlist' if self.method=='S' else 'full_catalog',
                    exact_overlap_resolution=overlap,live_catalog=live,endpoint_checked=strict.rows,
                    endpoint_excluded=strict.rows-endpoint_survivors,scored=len(evaluated),seed_visits=len(seeds),
                    cold_endpoint_visits=ids_before,tau=None if not self.indexed else int(tau))
                if winner is None:
                    d['status']='tolerance_stop'
                    decisions.append(d)
                    status='tolerance_stop'
                    break
                if best<=0:
                    raise ArithmeticError('unresolved positive gain on proof grid')
                i,j,f=(a[winner].item() for a in self.pool)
                d.update(i=i,j=j,flip=int(f),outsider=winner not in initial)
                if audit:
                    snapshots.append(dict(epoch=t+1,x=x.tolist(),r=r,opinion_hash=op_hash,
                        winner_id=winner,winner_gram=[[[str(lo),str(hi)] for lo,hi in row]
                        for row in views.entries[winner][1]]))
                with ledger.phase('cache_management'):
                    if not self.transport:
                        views.entries.clear()
                    hot.discard(winner)
                with ledger.phase('transport_local'):
                    views.role='winner'
                    zchild=views.advance(winner,t,A,fac,z)
                    views.role='candidate'
                    values=(0,-2*SCALE,-2*SCALE,0) if f else (1,-1,-1,1)
                    Achild=(self.kernel.child(A,winner) if self.kernel else
                            A+sparse.coo_matrix((values,([i,i,j,j],[i,j,i,j])),shape=A.shape).tocsr())
                with ledger.phase('state_solve_and_cert'):
                    statechild,source=lp.certify_child(Achild,fac,s,zchild,UNIT,certified_rows)
                    counts['estimated_state_residual_matvecs']+=1+int(source=='corrected')
                    counts['corrected_states']+=int(source=='corrected')
                    counts['state_rhs']+=int(source=='corrected')
                A,state=Achild,statechild
                chosen.add(winner)
                current=version_id(parent,winner)
                d.update(status='accepted',current_version_id=current,child_state_source=source,
                         child_state_residual=int(state[2]))
                decisions.append(d)
                total_lo+=best*SCALE
                total_hi+=evaluated[winner]*SCALE
            except (ArithmeticError,RuntimeError,MemoryError) as exc:
                reason=f'{type(exc).__name__}: {exc}'
                status=('unresolved' if 'overlap' in str(exc).lower() or 'unresolved' in str(exc).lower()
                        else 'memory_error' if isinstance(exc,MemoryError) else 'error')
                decisions.append(dict(epoch=t+1,root_version_id=self.root_id,parent_version_id=parent,
                    current_version_id=parent,status=status,winner_id=None,failure_reason=reason))
                break
        online_seconds=perf_counter()-online_start
        online_phases={k:v-pre_online_phases[k] for k,v in ledger.times.items()}
        online_phases['other']+=online_seconds-sum(online_phases.values())
        with ledger.phase('opinion_refresh'):
            final_energy=energy(state,s)
        counts.update(gram_materializations=views.fresh,gram_hits=views.hits,transfers=views.transfers)
        # Capture selected ownership at the same request boundary, without materializing
        # SciPy's L/U sparse copies to measure backend storage.
        response_bytes=0 if views.retained is None else sum(a.nbytes for a in views.retained[1][:3])
        memory=dict(sample='last committed state before reset',graph_csr=sum(a.nbytes for a in
            (self.W.data,self.W.indices,self.W.indptr)),root_operator_csr=sum(a.nbytes for a in
            (self.A.data,self.A.indices,self.A.indptr)),catalog=sum(a.nbytes for a in self.pool),
            structural_weight_grouping=self.weight_group_bytes,
            root_factor_reported_nnz=self.factor.base.nnz,index=0 if index is None else
            (self.kernel.index_bytes(index) if self.kernel else
             index.nbytes+index.stamps.nbytes+sum(a.nbytes for a in (index._out,index._choices,index._costs,index._bounds))),
            gram_count=len(views.entries),gram_python_bytes=lp.store_bytes(views.entries) if audit else None,
            shared_response_bytes=response_bytes,root_factor_resident_bytes=None,
            low_rank_steps=sum(y.nbytes+k.nbytes for y,k in fac.steps))
        index_build_phases=index.build_phases if self.weight_groups is not None else None
        with ledger.phase('reset_restore'):
            del views,fac,index,A,state,certified_rows
        result=dict(root_version_id=self.root_id,current_version_id=current,opinion_hash=op_hash,
            method=self.method,budget=budget,delta=delta,status=status,failure_reason=reason,
            initial_seed_ids=sorted(initial),initial_gram_count=0,decisions=decisions,commits=len(chosen),
            gain_low=total_lo,gain_high=total_hi,gain_denominator=UNIT**2,root_energy=root_energy,
            final_energy=final_energy,counters=counts,memory=memory,scope_checker='pending independent replay',
            cache_policy='structural root only; empty request Grams; answers disabled')
        if index_build_phases is not None:
            result['index_build_phases']=index_build_phases
        if audit:
            result['snapshots']=snapshots
            result['opinion_numerators']=s.tolist()
            # Clean side cursor, never sees online caches. Root gain signs are an
            # audit-only diagnostic. There is no test against a midpoint estimate.
            av=self.kernel.audit_views() if self.kernel else lp.Views(self.pool,UNIT,SCALE)
            if not self.kernel:
                av.directional_block=directional_block
            for d in decisions:
                e=d.get('winner_id')
                if e is not None:
                    lo,hi=av.score(e,0,self.A,self.factor,root_state[1],root_state[2])
                    d.update(root_gain_low=lo*SCALE,root_gain_high=hi*SCALE,root_gain_sign=(
                        'positive' if lo>0 else 'negative' if hi<0 else 'zero' if lo==hi==0 else 'indeterminate'))
        result['request_seconds'],result['phases']=ledger.finish()
        result['online_seconds']=online_seconds
        result['online_phases']=online_phases
        result['reset_opinion_seconds']=result['request_seconds']-online_seconds
        return result
