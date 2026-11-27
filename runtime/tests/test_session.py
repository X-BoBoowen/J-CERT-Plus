import numpy as np
import unittest
from scipy import sparse

from src.common import opinions
from src.session import Session


def graph():
    rows = [(0,1,-1),(0,2,1),(1,3,-1),(2,4,1),(3,5,1),(4,5,-1),(5,6,1)]
    w = np.zeros((7,7))
    for i,j,x in rows:
        w[i,j] = w[j,i] = x
    return sparse.csr_matrix(w)


def test_stream_prefix_and_bounds():
    for family in ('same','independent','slow'):
        a = opinions(21,family,104729,4)
        assert all(np.array_equal(x,y) for x,y in zip(a,opinions(21,family,104729,2)))
        assert all(x.dtype==np.int64 and np.abs(x).max()<=65536 for x in a)
        changed = np.count_nonzero(a[0]!=a[1])
        assert changed==0 if family=='same' else changed>0
        if family=='slow':
            assert changed<=3


def check_reset_and_retained_structure(method):
    sess = Session(graph(),method,hot_size=3)
    a,b = opinions(7,'independent',104729,2)
    base = sess.factor.base
    records = [sess.request(s,budget=3) for s in [a,b,a]]
    assert [r['opinion_hash'] for r in records][0]==records[-1]['opinion_hash']
    for s,r in zip([a,b,a],records):
        assert r['status'] in ('budget_stop','tolerance_stop')
        control = Session(graph(),'S' if method=='S' else 'F',hot_size=3).request(s,budget=3)
        assert [d['winner_id'] for d in r['decisions']]==[d['winner_id'] for d in control['decisions']]
        assert r['initial_gram_count']==0 and not sess.factor.steps
        assert r['counters']['state_rhs']>=1
        assert abs(sum(r['phases'].values())-r['request_seconds'])<1e-8
        assert abs(r['online_seconds']+r['reset_opinion_seconds']-r['request_seconds'])<1e-8
        assert abs(sum(r['online_phases'].values())-r['online_seconds'])<1e-8
        assert r['online_phases']['factor_or_precond_init']==0
        assert r['final_energy']['low']<=r['final_energy']['high']
        if method in ('F','I'):
            assert r['counters']['transfers']==0
            assert r['counters']['policy_refresh_rhs']>=r['counters']['gram_rematerializations']
        c=r['counters']
        assert c['response_residual_matvec_columns']==sum(c[k] for k in (
            'candidate_rhs','winner_rhs','policy_refresh_rhs','numerical_reanchor_rhs'))
    assert sess.factor.base is base
    assert records[0]['decisions'][0]['parent_version_id']==sess.root_id
    assert records[0]['gain_low']==records[-1]['gain_low']


def test_root_immutable_and_weight_guard():
    sess = Session(graph(),'D')
    with unittest.TestCase().assertRaises(ValueError):
        sess.W.data[0] = 2
    g=graph()
    g.data*=1.5
    with unittest.TestCase().assertRaises(ValueError):
        Session(g,'F')


def test_legacy_parity():
    from src.common import UNIT,SCALE,legacy
    import long_promotion as lp
    from promotion_index_profile import FusedClassIndex
    w=graph()
    s=opinions(7,'independent',130363,1)[0]
    broad,lib=legacy.native()
    for method in 'FTIDS':
        session=Session(w,method,hot_size=3)
        indexed=session.indexed
        mode='promoted' if indexed else 'static' if method=='S' else 'full'
        rows,low,high,_=lp.search(w,s,session.pool,mode,budget=4,hot_size=3,adaptive=indexed,
            operator_scale=SCALE,dyadic_tau=True,trace=True,broad_pass=broad,proof_unit=UNIT,
            index_factory=(lambda *a:FusedClassIndex(lib,*a)) if indexed else None,woodbury_state=True,
            retain_winner=True,directional=True,transport=session.transport,matched_screen=True,
            strict_screen=True)
        got=session.request(s,budget=4)
        assert [(r['winner_id'],r['low'],r['high']) for r in rows]==[
            (r['winner_id'],r['low'],r['high']) for r in got['decisions'] if r['status']=='accepted']
        assert (low,high)==(got['gain_low'],got['gain_high'])


def test_zero_stop():
    sess = Session(graph(),'D')
    r=sess.request(np.zeros(7,np.int64),budget=2)
    assert r['status']=='tolerance_stop' and r['commits']==0
    assert r['decisions'][0]['winner_id'] is None and r['gain_high']==0

class SessionTests(unittest.TestCase):
    test_stream = staticmethod(test_stream_prefix_and_bounds)
    test_root = staticmethod(test_root_immutable_and_weight_guard)
    test_stop = staticmethod(test_zero_stop)
    test_parity = staticmethod(test_legacy_parity)
    def test_reset(self):
        for method in 'FTIDS':
            with self.subTest(method=method):
                check_reset_and_retained_structure(method)

if __name__=='__main__':
    unittest.main()
