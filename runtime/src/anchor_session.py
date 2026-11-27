"""A weighted, index-only cross-opinion adapter for the frozen D request loop.

All five numerical operations (certification, exact current root ranking, screening,
scoring, commit) remain in the original weighted Session. Only cold discovery and
the lifetime of its sorted keys change. A Session instance serves a serial stream.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .common import SCALE, UNIT, digest
from .coverage_budget import Decision, PlanTickets, VisitBudget
from .session import Session, version_id
from .weighted_adapter import Index
import long_promotion as lp

DIVISOR=UNIT//lp.SCREEN
ARMS=('rebuild-always','retain-always','adaptive')
GUARD_ARM='budget-guard'
ACCESS_PHASES=('plan','posting_gather','tail_gather','concatenate_deduplicate_exclude',
               'strict_endpoint_screen','management')


@dataclass(frozen=True)
class Anchor:
    binding: tuple
    request: int
    opinion: str
    xc: np.ndarray
    rc: int
    index: object
    sorted_bytes: int = 0


@dataclass(frozen=True)
class Probe:
    choices: tuple
    costs: tuple
    cuts: tuple
    best: int
    posting: tuple


def plan(index,drift,target):
    """Calculate the frozen weighted index's exact pre-deduplication costs."""
    maximum=int(drift.max())
    if int(drift.min())<0 or maximum>4*lp.SCREEN:
        raise ArithmeticError('drift outside registered screening grid')
    choices=tuple(1<<p for p in range((max(1,maximum)-1).bit_length()+1))
    costs=[]
    cuts=[]
    posting=[]
    for tau in choices:
        shift=min(8*(lp.SCREEN//SCALE)*tau,8*index.root_norm*tau+4*tau**2)
        offsets=tuple(int(np.searchsorted(keys,int(target)-d*shift))
                      for d,keys,_ in index.groups)
        visits=sum(len(group[2])-offset for group,offset in zip(index.groups,offsets))
        posting.append(int(index.degree[drift>tau].sum()))
        costs.append(visits+posting[-1])
        cuts.append(offsets)
    return Probe(choices,tuple(costs),tuple(cuts),int(np.argmin(costs)),tuple(posting))


def gather(index,drift,probe,excluded,*,phases=None,counts=None):
    """Materialize precisely one selected plan, reusing its previously computed cuts."""
    k=probe.best
    tau=probe.choices[k]
    start=perf_counter() if phases is not None else 0.
    inc=index.inc
    posting=[]
    for v in np.flatnonzero(drift>tau):
        posting.append(np.arange(inc.left[v],inc.left[v+1],dtype=inc.ids.dtype))
        posting.append(inc.ids[inc.right[v]:inc.right[v+1]])
    gathered=perf_counter() if phases is not None else 0.
    # Keep both the frozen posting union and the outer posting/tail union.
    posts=np.unique(np.concatenate(posting)) if posting else inc.ids[:0]
    del posting
    deduplicated=perf_counter() if phases is not None else 0.
    tails=[ids[cut:] for (_,_,ids),cut in zip(index.groups,probe.cuts[k])]
    sliced=perf_counter() if phases is not None else 0.
    ids=np.unique(np.concatenate([posts]+tails))
    if counts is not None:
        counts['unique_ids_before_exclusion']=len(ids)
    if excluded:
        ids=np.setdiff1d(ids,np.array(sorted(excluded),np.int32),assume_unique=True)
    if phases is not None:
        finished=perf_counter()
        phases['posting_gather']+=gathered-start
        phases['tail_gather']+=sliced-deduplicated
        phases['concatenate_deduplicate_exclude']+=(deduplicated-gathered)+(finished-sliced)
    return ids,np.asarray(probe.choices),np.asarray(probe.costs),tau


class Cursor:
    def __init__(self,bridge,x,r,inc,degree):
        self.bridge=bridge
        self.x,self.r=x,int(r)
        self.inc,self.degree=inc,degree
        self.action=None
        self.index_build_s=0.
        self.build_phases=dict(key_gather=0.,stable_sort=0.)
        self.prediction_s=0.
        self.prediction=None
        self.events=[]
        self.access_phases=dict.fromkeys(ACCESS_PHASES,0.)
        self.tickets=PlanTickets()
        self.fallback_count=0
        self.sorted_index_overlap_bytes=0
        self.root_caps_retention_peak_bytes=0

    def rebuild(self,*,fallback=False):
        b=self.bridge
        started=perf_counter()
        with b.ledger.phase('posting/index_build'):
            # Session forwards the certified request-root coordinates and the
            # same caps used for top-64. The cursor preserves that provenance.
            caps=b.root_caps_for(self)
            built=Index(b.kernel,self.x,self.r,self.inc,self.degree,caps=caps,timers=b.ledger.enabled)
            del caps
            for stage,seconds in built.build_phases.items():
                self.build_phases[stage]+=seconds
            sorted_bytes=0
            for _,keys,ids in built.groups:
                keys.flags.writeable=False
                ids.flags.writeable=False
                sorted_bytes+=keys.nbytes+ids.nbytes
            built.groups=tuple(built.groups)
            xc=np.frombuffer(self.x.tobytes(),dtype=self.x.dtype)
            if fallback:
                self.sorted_index_overlap_bytes=b.check_anchor().sorted_bytes+sorted_bytes
            b.publish(Anchor(b.binding,b.position,b.opinion,xc,self.r,built,sorted_bytes))
        elapsed=perf_counter()-started
        self.index_build_s+=elapsed
        b.last_build=elapsed
        if fallback:
            self.fallback_count+=1
            self.action='fallback'
        else:
            self.action='initial-build' if b.position==1 else 'rebuild'

    def anchor_drift(self,base_drift):
        b=self.bridge
        anchor=b.anchor
        if anchor is None or anchor.binding!=b.binding:
            raise ValueError('anchor binding mismatch')
        # Recover precisely the coarse certificate used by frozen Session.
        # If rounding z at the fine grid lands on a coarse midpoint, either
        # possible coarse bin has the same fine distance, so its radius rc
        # is unambiguous. The original query supplies the exact unsigned
        # distance from its request root. The fine state fixes the sign
        # whenever that distance is positive.
        nearest=(b.fine_x+DIVISOR//2)//DIVISOR
        rc=(int(abs(b.fine_x-nearest*DIVISOR).max())+b.raw_residual+DIVISOR-1)//DIVISOR
        magnitude=base_drift-rc-self.r
        xc=self.x+np.sign(b.fine_x-self.x*DIVISOR)*magnitude
        if (int(magnitude.min())<0 or int(abs(b.fine_x-xc*DIVISOR).max())>DIVISOR//2 or
                not np.array_equal(base_drift,abs(xc-self.x)+rc+self.r)):
            raise ArithmeticError('could not recover certified child screening coordinates')
        b.coarse_x,b.coarse_radius=xc,rc
        return abs(xc-anchor.xc)+rc+anchor.rc

    def preview(self,decision,anchor,drift,excluded):
        start=perf_counter() if self.bridge.ledger.enabled else 0.
        probe=plan(anchor.index,drift,decision.target)
        if self.bridge.ledger.enabled:
            self.access_phases['plan']+=perf_counter()-start
        return self.tickets.issue(decision,anchor,probe,drift,excluded)

    def take(self,ticket,decision):
        self.bridge.check_decision(decision)
        return self.tickets.take(ticket,decision,self.bridge.check_anchor())

    @staticmethod
    def plan_fields(ticket):
        probe=ticket.probe
        k=probe.best
        anchor=ticket.anchor
        tails=[len(ids)-cut for (_,_,ids),cut in zip(anchor.index.groups,probe.cuts[k])]
        return dict(anchor_request=anchor.request,anchor_opinion_hash=anchor.opinion,
            target=ticket.decision.target,tau=probe.choices[k],
            pre_dedup_visits=probe.costs[k],posting_raw_visits=probe.posting[k],
            tail_raw_visits=probe.costs[k]-probe.posting[k],group_tails=tails,
            max_drift=int(ticket.drift.max()))

    def query(self,base_drift,target,excluded):
        timed=self.bridge.ledger.enabled
        if timed:
            started=perf_counter()
            prior_phases=sum(self.access_phases.values())
            prior_build=self.index_build_s
        try:
            return self._query(base_drift,target,excluded)
        finally:
            if timed:
                stages=sum(self.access_phases.values())-prior_phases
                rebuild=self.index_build_s-prior_build
                self.access_phases['management']+=max(0.,perf_counter()-started-stages-rebuild)

    def _query(self,base_drift,target,excluded):
        b=self.bridge
        anchor=b.check_anchor()
        decision=b.decision(len(self.events)+1,target)
        drift=self.anchor_drift(base_drift)
        old=anchor.request<b.position
        predicting=self.action is None and b.arm=='adaptive'
        if predicting:
            start=perf_counter()
        ticket=self.preview(decision,anchor,drift,excluded)
        visits=ticket.probe.costs[ticket.probe.best]
        consumed_before=b.guard.consumed if b.guard else None
        rejected_plan=fallback_reason=None
        event_action='retain' if old else 'fresh'
        if old and b.guard and not b.guard.reserve(visits):
            rejected_plan=self.plan_fields(ticket)
            if b.audit:
                rejected_plan.update(anchor_xc=anchor.xc.tolist(),anchor_rc=anchor.rc,
                                     drift=drift.tolist())
            fallback_reason='visit_budget_exceeded'
            self.take(ticket,decision)
            del ticket,anchor,drift
            self.rebuild(fallback=True)
            anchor=b.check_anchor()
            drift=self.anchor_drift(base_drift)
            ticket=self.preview(decision,anchor,drift,excluded)
            event_action='fallback'
        elif predicting:
            # Preserve the historical first-query rule for the legacy arm.
            base=int(np.count_nonzero(b.current_caps>=target))
            excess=max(0,visits-base)
            estimated=excess*b.budget*b.price_per_visit
            self.prediction=dict(warm_target=int(target),estimated_old_visits=visits,
                current_cap_hits=base,estimated_extra_s=estimated,previous_build_s=b.last_build,
                previous_price_per_visit=b.price_per_visit,budget=b.budget)
            self.prediction_s=perf_counter()-start
            if estimated>b.last_build:
                self.take(ticket,decision)
                del ticket,anchor,drift
                self.rebuild()
                anchor=b.check_anchor()
                drift=self.anchor_drift(base_drift)
                ticket=self.preview(decision,anchor,drift,excluded)
                event_action='fresh'
        if self.action is None:
            self.action='retain'
        if not b.guard:
            b.release_root_caps()
        ticket=self.take(ticket,decision)
        event=self.plan_fields(ticket)
        result=gather(ticket.index,ticket.drift,ticket.probe,ticket.excluded,
                      phases=self.access_phases if b.ledger.enabled else None,counts=event)
        ids=result[0]
        event.update(epoch=decision.epoch,parent_version_id=decision.parent,returned_ids=len(ids),
            budget=b.guard.limit if b.guard else None,consumed_before=consumed_before,
            consumed=b.guard.consumed if b.guard else None,action=event_action,
            fallback_reason=fallback_reason,rejected_plan=rejected_plan)
        if b.audit:
            event.update(drift=drift.tolist(),base_drift=base_drift.tolist(),
                fine_x=b.fine_x.tolist(),fine_radius=b.fine_radius,raw_residual=b.raw_residual,
                current_xc=b.coarse_x.tolist(),current_rc=b.coarse_radius,seed_ids=b.seed_ids,
                excluded_ids=sorted(excluded),returned_ids_sha256=digest(ids.astype(np.int32,copy=False)),
                anchor_xc=anchor.xc.tolist(),anchor_rc=anchor.rc,
                current_root_xc=self.x.tolist(),current_root_rc=self.r)
        self.events.append(event)
        return result


class DiscoveryBridge:
    """The numerical kernel is unchanged; each query receives its own short-lived cursor."""
    def __init__(self,kernel,binding,arm,visit_budget=None):
        self.kernel,self.binding,self.arm=kernel,binding,arm
        self.visit_budget=visit_budget
        self._structural_binding=binding
        self.position=0
        self.anchor=self._issued_source=None
        self.cursor=None
        self.last_build=self.price_per_visit=0.

    def __getattr__(self,name):
        return getattr(self.kernel,name)

    def begin(self,s,budget,audit):
        if self.cursor is not None:
            raise RuntimeError('serial session already active')
        anchor=self.check_anchor()
        self.guard=VisitBudget(self.visit_budget) if self.arm==GUARD_ARM else None
        self.position+=1
        self.budget=budget
        self.audit=audit
        self.opinion=digest(s)
        self._active_request=(self.position,self.opinion)
        self.parent=self._current_parent=self.binding[0]
        self._root_witness=self.fine_x=self.ledger=self.coarse_x=None
        self.initial_anchor_request=anchor.request if anchor else None
        self.initial_anchor_opinion=anchor.opinion if anchor else None
        self.fine_radius=0

    def check_anchor(self):
        anchor=self.anchor
        if self.binding!=self._structural_binding:
            raise ValueError('anchor binding mismatch')
        if self._issued_source is None:
            if anchor is None:
                return None
        else:
            issued,index,groups,inc,degree,norm,layout=self._issued_source
            if (anchor is issued and anchor.binding==self.binding and anchor.request<=self.position and
                    anchor.index is index and index.groups is groups and index.inc is inc and
                    index.degree is degree and index.root_norm==norm and
                    (anchor.xc.shape,anchor.xc.dtype,anchor.xc.strides)==layout):
                return anchor
        raise ValueError('anchor binding mismatch')

    def decision(self,epoch,target):
        decision=Decision(self.position,self.opinion,self.parent,epoch,int(target))
        self.check_decision(decision)
        return decision

    def check_decision(self,decision):
        if ((self.position,self.opinion)!=self._active_request or
                (decision.request,decision.opinion)!=self._active_request or
                decision.parent!=self._current_parent or self.parent!=self._current_parent):
            raise ValueError('decision binding mismatch')

    @property
    def current_caps(self):
        return self._root_witness[-1] if self._root_witness is not None else None

    def root_caps_for(self,cursor):
        # Only Session.index supplies the root witness, immediately after root
        # certification and ranking. This is a trusted in-process boundary.
        if self._root_witness is None:
            raise ValueError('current root ranking keys unavailable')
        request,root_x,r,caps=self._root_witness
        if (cursor is not self.cursor or request!=self._active_request or
                request!=(self.position,self.opinion) or cursor.x is not root_x or cursor.r!=r):
            raise ValueError('current root ranking keys binding mismatch')
        return caps

    def release_root_caps(self):
        self._root_witness=None

    def publish(self,anchor):
        xc=anchor.xc
        index=anchor.index
        self.anchor=anchor
        self._issued_source=(anchor,index,index.groups,index.inc,index.degree,index.root_norm,
                             (xc.shape,xc.dtype,xc.strides))
        self.release_root_caps()

    def views(self,ledger,counts):
        self.ledger=ledger
        return self.kernel.views(ledger,counts)

    def index(self,x,r,inc,degree,*,caps=None,timers=True):
        if caps is None:
            caps=self.kernel.root_caps(x,r)
        self._root_witness=(self._active_request,x,int(r),caps)
        self.cursor=Cursor(self,x,r,inc,degree)
        if self.anchor is None or self.arm=='rebuild-always':
            self.cursor.rebuild()
        else:
            self.cursor.root_caps_retention_peak_bytes=caps.nbytes
        return self.cursor

    def strict(self,A,s,x,certified_rows):
        result=self.kernel.strict(A,s,x,certified_rows)
        self.fine_x=x
        self.raw_residual=int(certified_rows[5].max())
        self.fine_radius=(self.raw_residual+SCALE-1)//SCALE
        self.seed_ids=None
        original=result.screen
        if self.ledger.enabled or self.audit:
            def screen(ids,best):
                if self.audit and self.seed_ids is None:
                    self.seed_ids=sorted(map(int,ids))
                if not self.ledger.enabled:
                    return original(ids,best)
                start=perf_counter()
                try:
                    return original(ids,best)
                finally:
                    self.cursor.access_phases['strict_endpoint_screen']+=perf_counter()-start
            result.screen=screen
        return result

    def child(self,A,e):
        if self.parent!=self._current_parent:
            raise ValueError('decision binding mismatch')
        child=self.kernel.child(A,e)
        self.parent=self._current_parent=version_id(self.parent,e)
        return child

    def index_bytes(self,cursor):
        return self.anchor.index.nbytes+self.anchor.xc.nbytes if self.anchor else 0

    def close(self):
        self.cursor=self.ledger=self.fine_x=self.coarse_x=self._active_request=None
        self.release_root_caps()
        self.seed_ids=None


class AnchorSession:
    def __init__(self,W,arm,*,visit_budget=None,**kwargs):
        if arm not in (*ARMS,GUARD_ARM):
            raise ValueError('unknown A1 arm')
        self.service=Session(W,'D',weight_mode='weighted',**kwargs)
        s=self.service
        binding=(s.root_id,s.graph_hash,s.operator_hash,s.catalog_hash,s.kernel.salt,
                 'weighted-index-groups-v1',SCALE,UNIT,lp.SCREEN)
        t=perf_counter()
        self.bridge=DiscoveryBridge(s.kernel,binding,arm,visit_budget)
        s.kernel=self.bridge
        addition=perf_counter()-t
        s.setup_seconds+=addition
        s.setup_phases['other']+=addition

    def __getattr__(self,name):
        return getattr(self.service,name)

    def request(self,s,*,budget=15,audit=False):
        started=perf_counter()
        b=self.bridge
        b.begin(s,budget,audit)
        try:
            result=self.service.request(s,budget=budget,audit=audit)
            cursor=b.cursor
            phases=result['phases']
            checked=result['counters']['endpoint_checked']
            observed_cost=sum(phases[k] for k in ('discovery','comparison/refinement','response_acquisition'))
            if checked:
                b.price_per_visit=observed_cost/checked
            retained_bytes=b.anchor.xc.nbytes+b.anchor.sorted_bytes
            caps_at_end_bytes=0 if b.current_caps is None else b.current_caps.nbytes
            result['memory'].update(effective_weights=b.kernel.d.nbytes,
                root_caps_retention_peak_bytes=cursor.root_caps_retention_peak_bytes,
                incremental_root_caps_bytes=cursor.root_caps_retention_peak_bytes if b.guard else 0,
                sorted_index_overlap_bytes=cursor.sorted_index_overlap_bytes)
            result.update(arm=b.arm,discovery=dict(action=cursor.action or 'no-query',
                source_root_version=self.root_id,anchor_request=b.anchor.request,anchor_opinion_hash=b.anchor.opinion,
                anchor_age=b.position-b.anchor.request,anchor_memory_bytes=self.bridge.index_bytes(cursor),
                retained_key_coordinate_bytes=retained_bytes,
                initial_anchor_request=b.initial_anchor_request,initial_anchor_opinion_hash=b.initial_anchor_opinion,
                visit_budget=b.guard.limit if b.guard else None,old_consumed=b.guard.consumed if b.guard else None,
                fallback_count=cursor.fallback_count,access_phases=cursor.access_phases,
                root_caps_retention_peak_bytes=cursor.root_caps_retention_peak_bytes,
                root_caps_retained_at_end_bytes=caps_at_end_bytes,
                sorted_index_overlap_bytes=cursor.sorted_index_overlap_bytes,
                predicted=cursor.prediction,prediction_s=cursor.prediction_s,rebuild_s=cursor.index_build_s,
                next_price_per_visit=b.price_per_visit,epochs=cursor.events),
                cache_policy='immutable structural root + root index anchor; empty request Grams; answers disabled')
        finally:
            b.close()
        elapsed=perf_counter()-started
        overhead=elapsed-result['request_seconds']
        result['request_seconds']=elapsed
        result['reset_opinion_seconds']+=overhead
        phases['cache_management']+=overhead
        return result
