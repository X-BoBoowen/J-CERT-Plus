"""Diagnostic-only observers; never imported by a formal timing worker."""
import hashlib
import json

DEEP_LABELS=('H_frontier','H_frontier_inner','G_frontier','catalog','cache_and_log')


class ReportingProbe:
    def __init__(self):
        self.calls=dict.fromkeys(DEEP_LABELS,0)
        self.originals=[]

    def __enter__(self):
        from src.a2_access import Access
        from src.a2_catalog import AllCatalog,MixedFrontier
        from src.g_access import GAccess
        from src.a2_session import lp
        for owner,name,label in ((Access,'observe_frontier','H_frontier'),
            (MixedFrontier,'memory','H_frontier_inner'),(GAccess,'_observe_memory','G_frontier'),
            (AllCatalog,'memory','catalog'),(lp,'store_bytes','cache_and_log')):
            original=getattr(owner,name)
            def observed(*args,_original=original,_label=label,**kwargs):
                self.calls[_label]+=1
                return _original(*args,**kwargs)
            self.originals.append((owner,name,original))
            setattr(owner,name,observed)
        return self

    def __exit__(self,*exc):
        for owner,name,original in reversed(self.originals):setattr(owner,name,original)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _array(array):
    return dict(shape=list(array.shape),dtype=str(array.dtype),strides=list(array.strides),
                writeable=bool(array.flags.writeable),sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def _response(value):
    if value is None:return None
    e,(U,Y,X,r)=value
    return dict(candidate_id=int(e),U=_array(U),Y=_array(Y),X=_array(X),radius=int(r))


def cache_audit(views,fac,service):
    if views is None:return dict(schema='v4-final-cache-audit-1',captured_views=False,captured_factor=False)
    contents=[]
    for e,(version,gram) in sorted(views.entries.items()):
        intervals=[[[[int(value.numerator),int(value.denominator)] for value in pair]
                     for pair in row] for row in gram]
        contents.append([int(e),int(version),intervals])
    steps=None if fac is None else [[_array(y),_array(k)] for y,k in fac.steps]
    return dict(schema='v4-final-cache-audit-1',captured_views=True,captured_factor=fac is not None,
        entry_count=len(views.entries),entries_sha256=_digest(contents),
        entry_insertion_order=list(map(int,views.entries)),
        entry_versions=[[int(e),int(version)] for e,(version,_) in sorted(views.entries.items())],
        fresh=int(views.fresh),hits=int(views.hits),winner_blocks=int(views.winner_blocks),
        transfers=int(views.transfers),role=views.role,
        retained_response=_response(views.retained),scored_response=_response(views.scored_response),
        retained_is_scored_response=None if views.retained is None or views.scored_response is None
            else views.retained is views.scored_response,
        step_count=None if fac is None else len(fac.steps),steps_sha256=None if steps is None else _digest(steps),
        factor_base_is_shared_root=None if fac is None else fac.base is service.factor.base,
        root_factor_step_count=len(service.factor.steps),
        scope='final cache entries/order/version/exact Gram content, response array bits and factor steps after request; diagnostic-only scans',
        unobserved=['intermediate cache contents','local hot/seen/chosen set contents','absolute object addresses'])


def diagnostic_session_type(base,probe):
    class DiagnosticSession(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            original=self.kernel.views
            def capture(*args,**kwargs):
                views=original(*args,**kwargs)
                self._diagnostic_views=views
                advance=views.advance
                def observe_advance(e,t,A,fac,z):
                    self._diagnostic_factor=fac
                    return advance(e,t,A,fac,z)
                views.advance=observe_advance
                return views
            self.kernel.views=capture

        def request(self,*args,**kwargs):
            before=probe.calls.copy()
            self._diagnostic_views=self._diagnostic_factor=None
            result=super().request(*args,**kwargs)
            result['diagnostic_cache_audit']=cache_audit(self._diagnostic_views,self._diagnostic_factor,self)
            result['diagnostic_deep_calls']={name:probe.calls[name]-before[name] for name in DEEP_LABELS}
            self._diagnostic_views=self._diagnostic_factor=None
            return result
    return DiagnosticSession


def diagnostic_obligation_issues(worker):
    issues=[]
    method,policy=worker.get('method'),worker.get('measurement_policy')
    if method not in ('H','G') or policy not in ('legacy_deep','light_v1'):issues.append('invalid diagnostic identity')
    requests=worker.get('requests',[])
    if len(requests)!=4:issues.append('diagnostic must contain all four Alpha/slow requests')
    if worker.get('status')!='completed_budget' or worker.get('complete_Q4') is not True:issues.append('diagnostic not complete Q4')
    for q,r in enumerate(requests,1):
        prefix=f'q{q}: '
        if r.get('query_position')!=q or r.get('status')!='budget_stop' or r.get('commits')!=15:
            issues.append(prefix+'wrong request order/completion/commit count')
        if r.get('measurement_policy')!=policy:issues.append(prefix+'policy mismatch')
        audit=r.get('diagnostic_cache_audit',{})
        required=('entry_count','entries_sha256','step_count','steps_sha256','factor_base_is_shared_root','root_factor_step_count')
        if (audit.get('schema')!='v4-final-cache-audit-1' or audit.get('captured_views') is not True
            or audit.get('captured_factor') is not True or any(key not in audit for key in required)):
            issues.append(prefix+'missing cache/factor audit evidence')
        elif (audit['step_count']!=15 or audit['root_factor_step_count']!=0
              or audit['factor_base_is_shared_root'] is not True):
            issues.append(prefix+'unexpected factor/root reuse state')
        calls=r.get('diagnostic_deep_calls')
        if not isinstance(calls,dict) or set(calls)!=set(DEEP_LABELS) or any(type(v) is not int or v<0 for v in calls.values()):
            issues.append(prefix+'missing or invalid explicit deep-call counters')
        else:
            expected=dict.fromkeys(DEEP_LABELS,0)
            if policy=='legacy_deep':
                expected.update(catalog=1,cache_and_log=3)
                if method=='H':expected.update(H_frontier=32,H_frontier_inner=32)
                if method=='G':expected.update(G_frontier=16)
            if calls!=expected:issues.append(prefix+'deep-call evidence differs from declared policy')
    return issues
