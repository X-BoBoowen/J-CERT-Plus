import unittest
from unittest.mock import patch
from fractions import Fraction
import numpy as np
from scipy import sparse

from src.a2_session import A2Session
from src.a2_catalog import AllCatalog
from src.common import DEN, UNIT


def mixed_graph(n=6):
    W=np.zeros((n,n),float)
    for i,j,w in [(0,1,-2.),(1,2,1.5),(2,3,-.5),(3,4,3.),(4,5,-1.)]:
        W[i,j]=W[j,i]=w
    return sparse.csr_matrix(W)


def exact_next(W,s,chosen):
    """Independent combinations and rational full-system energy."""
    import sympy as sy
    n=W.shape[0];arr=W.toarray()
    A=16*np.eye(n,dtype=np.int64)
    candidates=[]
    for i in range(n):
        for j in range(i+1,n):
            weight=arr[i,j]
            if weight:
                d=int(16*abs(weight));sign=1 if weight>0 else -1
                A[i,i]+=d;A[j,j]+=d;A[i,j]-=sign*d;A[j,i]-=sign*d
            if weight<=0:candidates.append((i*n+j,i,j,weight<0,int(-16*weight) if weight<0 else 1))
    def change(a,row):
        _,i,j,flip,d=row;out=a.copy()
        if flip:out[i,j]-=2*d;out[j,i]-=2*d
        else:
            out[i,i]+=d;out[j,j]+=d;out[i,j]-=d;out[j,i]-=d
        return out
    byid={e:row for row in candidates for e in [row[0]]}
    for e in chosen:A=change(A,byid[e])
    rhs=sy.Matrix([sy.Rational(int(x),DEN) for x in s])
    def energy(a):return (rhs.T*sy.Matrix(a.tolist()).inv()*rhs)[0]
    base=energy(A)
    gains={e:Fraction(base-energy(change(A,row))) for e,*_ in candidates
           for row in [byid[e]] if e not in chosen}
    if not gains:return None,Fraction(0)
    top=max(gains.values())
    return (min(e for e,g in gains.items() if g==top),top) if top>0 else (None,Fraction(0))


class A2SessionTests(unittest.TestCase):
    def test_three_strategies_full_exact_mixed_trajectory(self):
        W=mixed_graph();s=np.array([40000,-21000,12000,-49000,8000,50000],np.int64)
        rows={method:A2Session(W,method,timers=False).request(s,budget=5,audit=True)
              for method in ('D','S','H')}
        signatures=[]
        for method,result in rows.items():
            chosen=[]
            for decision in result['decisions']:
                e,gain=exact_next(W,s,chosen)
                self.assertEqual(decision['winner_id'],e,(method,decision))
                if e is None:
                    self.assertEqual(decision['status'],'tolerance_stop');break
                self.assertEqual(decision['status'],'accepted')
                self.assertLessEqual(Fraction(decision['low'],UNIT**2),16*gain)
                self.assertGreaterEqual(Fraction(decision['high'],UNIT**2),16*gain)
                chosen.append(e)
            signatures.append(chosen)
            self.assertEqual(result['initial_gram_count'],0)
            if method=='H':
                self.assertEqual(sum(x['frontier_rebuilds'] for x in result['access']['epochs']),len(result['decisions']))
        self.assertTrue(all(x==signatures[0] for x in signatures))

    def test_implicit_and_stream_do_not_materialize_catalog(self):
        W=mixed_graph();s=np.array([40000,-21000,12000,-49000,8000,50000],np.int64)
        for method in ('S','H'):
            with patch.object(AllCatalog,'materialize',side_effect=AssertionError('full catalog forbidden')):
                session=A2Session(W,method,timers=False)
                result=session.request(s,budget=3)
            self.assertFalse(result['memory']['catalog_materialized'])
            self.assertEqual(result['memory']['candidate_array_rows'],0)
            self.assertEqual(result['memory']['full_root_caps_created_rows'],0)
            self.assertEqual(result['memory']['m_sized_structures'],[])

    def test_positive_exact_tie_uses_canonical_id(self):
        W=sparse.csr_matrix((4,4),dtype=float)
        s=np.array([DEN,-DEN,DEN,-DEN],np.int64)
        for method in ('D','S','H'):
            result=A2Session(W,method,hot_size=1,timers=False).request(s,budget=1)
            self.assertEqual(result['decisions'][0]['winner_id'],1,method)

    def test_zero_and_empty_catalog_stop_without_positive_claim(self):
        for W in (sparse.csr_matrix((4,4),dtype=float),
                  sparse.csr_matrix(np.ones((4,4))-np.eye(4))):
            for method in ('D','S','H'):
                result=A2Session(W,method,timers=False).request(np.zeros(4,np.int64),budget=3)
                self.assertEqual(result['status'],'tolerance_stop',(method,result))
                self.assertEqual(result['commits'],0)
                self.assertEqual(result['gain_low'],0)

    def test_request_reset_and_budget_zero(self):
        W=mixed_graph();s=np.array([40000,-21000,12000,-49000,8000,50000],np.int64)
        for method in ('D','S','H'):
            session=A2Session(W,method,timers=False)
            first=session.request(s,budget=2);second=session.request(s,budget=2)
            self.assertEqual([d['winner_id'] for d in first['decisions']],
                             [d['winner_id'] for d in second['decisions']])
            self.assertEqual(len(session.factor.steps),0)
            zero=session.request(s,budget=0)
            self.assertEqual(zero['decisions'],[])
            self.assertEqual(zero['commits'],0)

    def test_large_exact_tie_preserves_unresolved_contract(self):
        W=sparse.csr_matrix((17,17),dtype=float)
        s=np.array([DEN if i%2 else -DEN for i in range(17)],np.int64)
        for method in ('D','S','H'):
            result=A2Session(W,method,hot_size=1,timers=False).request(s,budget=1)
            self.assertEqual(result['status'],'unresolved',method)
            self.assertEqual(result['commits'],0)
            self.assertGreater(result['memory']['gram_count'],0)
            self.assertGreaterEqual(result['memory']['peak_gram_count'],
                                    result['memory']['gram_count'])

    def test_warm_memory_failure_uses_current_request_counters(self):
        from src.weighted_adapter import WeightedViews
        W=mixed_graph();s=np.array([40000,-21000,12000,-49000,8000,50000],np.int64)
        for method in ('D','S','H'):
            session=A2Session(W,method,timers=False)
            for prior_success in (False,True):
                if prior_success:session.request(s,budget=2)
                with patch.object(WeightedViews,'score',side_effect=MemoryError('fixture warm failure')):
                    failed=session.request(s,budget=1)
                self.assertEqual(failed['status'],'memory_error')
                event=failed['access']['epochs'][0]
                self.assertEqual(event['epoch'],1)
                self.assertEqual(event['parent_version_id'],session.root_id)
                self.assertFalse(event['source_complete'])
                self.assertEqual(event['cold_emitted_for_screen'],0)
                self.assertEqual(event['seed_candidates'],len(failed['initial_seed_ids']))
                self.assertEqual(failed['counters']['endpoint_checked'],len(failed['initial_seed_ids']))
                self.assertEqual(event['warm_screen_rows'],len(failed['initial_seed_ids']))

    def test_no_initial_seeds_retains_complete_competition(self):
        W=mixed_graph();s=np.array([40000,-21000,12000,-49000,8000,50000],np.int64)
        for method in ('D','S','H'):
            result=A2Session(W,method,hot_size=0,timers=False).request(s,budget=2)
            self.assertEqual(result['initial_seed_ids'],[])
            selected=[]
            for row in result['decisions']:
                winner,_=exact_next(W,s,selected)
                self.assertEqual(row['winner_id'],winner)
                selected.append(winner)

    def test_partial_commit_memory_keeps_low_rank_payload_peak(self):
        W=sparse.csr_matrix((4,4),dtype=float)
        s=np.array([10000,-20000,30000,-45000],np.int64)
        session=A2Session(W,'H',timers=False)
        with patch('src.a2_session.lp.certify_child',side_effect=MemoryError('fixture child failure')):
            result=session.request(s,budget=1)
        self.assertEqual(result['status'],'memory_error')
        self.assertEqual(result['commits'],0)
        self.assertGreater(result['memory']['low_rank_steps'],0)
        self.assertGreaterEqual(result['memory']['peak_low_rank_steps'],result['memory']['low_rank_steps'])

    def test_seed_and_cold_emission_accounting_share_the_same_scope(self):
        W=sparse.csr_matrix((4,4),dtype=float)
        s=np.array([10000,-20000,30000,-45000],np.int64)
        for method in ('D','S','H'):
            result=A2Session(W,method,timers=True).request(s,budget=1)
            event=result['access']['epochs'][0]
            self.assertEqual(event['seed_candidates'],6)
            self.assertEqual(event['seed_insertions'],6)
            self.assertEqual(event['insertion_emissions'],6)
            self.assertEqual(event['cold_emitted_for_screen'],0)
            self.assertEqual(event['warm_screen_rows'],6)
            self.assertTrue(event['seed_build_phases'])

    def test_fifteen_commits_with_independent_complete_checker(self):
        from src.a2_checker import A2CatalogReplay,worker_request
        W=mixed_graph(8)
        s=np.array([40000,-21000,12000,-49000,8000,50000,-33000,59000],np.int64)
        signatures=[]
        for method in ('D','S','H'):
            session=A2Session(W,method,timers=False)
            result=session.request(s,budget=15)
            self.assertEqual(result['commits'],15,(method,result))
            proof=worker_request(A2CatalogReplay(W),result,s,session.graph_hash)
            self.assertTrue(proof)
            signatures.append([d['winner_id'] for d in result['decisions']])
        self.assertTrue(all(s==signatures[0] for s in signatures))


if __name__=='__main__':unittest.main()
