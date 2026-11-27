import unittest
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from src.anchor_session import ARMS,Anchor,AnchorSession,Cursor,DIVISOR,gather,plan
from src.anchor_service_bench import jobs
from src.checker import CatalogReplay,worker_request
from src.common import opinions
from src.session import Session


def graph():
    w=np.zeros((8,8),float)
    for i,j,v in [(0,1,-1.5),(1,2,-2.5),(2,3,0.5),(3,4,-3),
                  (4,5,1.5),(5,6,-.5),(6,7,2),(0,7,-4),(1,4,1.)]:
        w[i,j]=w[j,i]=v
    return sparse.csr_matrix(w)


class AnchorServiceTests(unittest.TestCase):
    def test_agreement_reset_and_no_cross_request_grams(self):
        W=graph()
        solver=CatalogReplay(W)
        for arm in ARMS:
            base=Session(W,'D',weight_mode='weighted',hot_size=3)
            new=AnchorSession(W,arm,hot_size=3)
            first,second=opinions(8,'independent',104729,2)
            for q,s in enumerate((first,second,first,np.zeros_like(first)),1):
                ref=base.request(s,budget=3)
                got=new.request(s,budget=3,audit=True)
                self.assertEqual(got['initial_seed_ids'],ref['initial_seed_ids'])
                self.assertEqual(got['initial_gram_count'],0)
                self.assertEqual(got['status'],ref['status'])
                self.assertEqual([d['winner_id'] for d in got['decisions']],
                                 [d['winner_id'] for d in ref['decisions']])
                self.assertLessEqual(max(got['gain_low'],ref['gain_low']),min(got['gain_high'],ref['gain_high']))
                worker_request(solver,got,s,new.graph_hash,kind='weighted')
                if arm=='rebuild-always':
                    self.assertEqual((got['gain_low'],got['gain_high']),(ref['gain_low'],ref['gain_high']))
                    self.assertEqual([d['endpoint_checked'] for d in got['decisions']],
                                     [d['endpoint_checked'] for d in ref['decisions']])
                if arm=='retain-always':
                    self.assertEqual(got['discovery']['anchor_request'],1)
                self.assertEqual(new.factor.steps,[])
                self.assertIsNone(new.bridge.cursor)
                self.assertIsNone(new.bridge.fine_x)
                self.assertIsNone(new.bridge.current_caps)
                self.assertFalse(new.bridge.anchor.xc.flags.writeable)
                self.assertAlmostEqual(sum(got['phases'].values()),got['request_seconds'],places=8)

    def test_midpoint_child_coordinate_recovery(self):
        # Fine-grid ties can pick either coarse bin depending on original binary64 z.
        # Even when fine x is identical, original D coarse distances resolve both.
        root=np.asarray([0,2,-3,1],np.int64)
        old=np.asarray([-2,4,2,9],np.int64)
        anchor=Anchor(('test',),1,'first',old,3,None)
        x=np.array([DIVISOR//2,DIVISOR*3//2,-DIVISOR*5//2,DIVISOR//2],np.int64)
        bridge=SimpleNamespace(anchor=anchor,binding=('test',),position=2,fine_x=x,raw_residual=2)
        cursor=Cursor(bridge,root,1,None,None)
        for actual in (np.array([0,2,-3,1]),np.array([1,1,-2,0])):
            base=abs(actual-root)+2
            drift=cursor.anchor_drift(base)
            np.testing.assert_array_equal(bridge.coarse_x,actual)
            np.testing.assert_array_equal(drift,abs(actual-old)+1+3)

    def test_plan_gather_matches_weighted_index(self):
        sess=AnchorSession(graph(),'retain-always',hot_size=2)
        s=opinions(8,'same',130363,1)[0]
        sess.request(s,budget=1)
        index=sess.bridge.anchor.index
        for drift in (np.zeros(8,np.int64),np.arange(8,dtype=np.int64)*51):
            for target in (0,1703,100000001,2**30):
                probe=plan(index,drift,target)
                expected=index.query(drift,target,{1,2})
                got=gather(index,drift,probe,{1,2})
                for a,b in zip(expected,got):
                    np.testing.assert_array_equal(a,b)

    def test_binding_rejection_and_zero_budget(self):
        from dataclasses import replace
        sess=AnchorSession(graph(),'retain-always',hot_size=2)
        s=opinions(8,'same',104729,1)[0]
        req=sess.request(s,budget=0)
        self.assertEqual(req['commits'],0)
        sess.bridge.anchor=replace(sess.bridge.anchor,binding=('wrong',))
        with self.assertRaisesRegex(ValueError,'anchor binding mismatch'):
            sess.request(s,budget=1)

    def test_counterbalanced_schedule(self):
        protocol=dict(repeat_count=3,graphs=['Alpha','OTC'],streams=['same','slow','independent'])
        seq=list(jobs(protocol))
        self.assertEqual(len(seq),54)
        for g in protocol['graphs']:
            for stream in protocol['streams']:
                field=[r['arm'] for r in seq if r['graph']==g and r['stream']==stream]
                self.assertEqual(set(field[::3]),set(ARMS))
                self.assertEqual(sorted(field),sorted(ARMS*3))


if __name__=='__main__':
    unittest.main()
