from fractions import Fraction as F
import unittest
import numpy as np
from scipy import sparse
from src.common import UNIT,SCALE
from src.checker import reference_gains
from src.weighted_adapter import Kernel
from long_promotion import UpdatedFactor, certified_state
from universe_index import build_universe, build_incidence


class WeightedTests(unittest.TestCase):
    def test_normalized_update_transfer_screen_index_and_cursor_reset(self):
        W = sparse.csr_matrix([[0,-1.5,0,0],[-1.5,0,-2.5,0],[0,-2.5,0,.5],[0,0,.5,0]])
        p=build_universe(W)
        k=Kernel(W,p)
        A=k.operator(W)
        n=W.shape[0]
        self.assertEqual(A[0,1],24)
        s=np.array([65536,-30000,-32000,5000],dtype=np.int64)
        f=UpdatedFactor(A)
        root=certified_state(A,f,s,UNIT)
        v=k.audit_views()
        inc=build_incidence(*p[:2],n)
        index=k.index(root[3],root[4],inc,np.diff(inc.left)+np.diff(inc.right))
        chosen=[]
        for t in range(len(p[0])):
            rows=[]
            z,x,r,xc,rc=certified_state(A,f,s,UNIT,certified_rows=rows)
            gains=reference_gains(W,s,chosen)
            remain=[j for j,g in enumerate(gains) if g is not None]
            live=np.array(remain,dtype=np.int32)
            screen=k.strict(A,s,x,rows[0])
            ids,caps=screen.screen(live,-10**50)
            self.assertEqual(ids.tolist(),remain)
            for e,upper in zip(ids,caps):
                e=int(e)
                g=gains[e]
                self.assertGreaterEqual(F(int(upper)*SCALE,UNIT**2),g)
                low,high=v.score(e,t,A,f,x,r)
                self.assertLessEqual(F(low*SCALE,UNIT**2),g)
                self.assertLessEqual(g,F(high*SCALE,UNIT**2))
            ratio=(UNIT//(1<<20))**2
            for j in remain:
                target=int(gains[j]*UNIT**2/SCALE)//ratio
                candidates,*_=index.query(abs(xc-root[3])+rc+root[4],target,set(chosen))
                self.assertIn(j,candidates)
            e=remain[0]
            zchild=v.advance(e,t,A,f,z)
            A=k.child(A,e)
            expected=np.linalg.solve(A.toarray(),s/65536)
            self.assertTrue(np.allclose(zchild,expected,atol=1e-13))
            self.assertTrue(np.allclose(f.solve(np.eye(n)),np.linalg.inv(A.toarray()),atol=1e-13))
            chosen.append(e)
        self.assertGreater(v.transfers,0)
        self.assertEqual(len(UpdatedFactor(k.operator(W)).steps),0)

    def test_switch_reset_all_policy_combinations_and_binding_tampering(self):
        from copy import deepcopy
        from src.session import Session
        from src.checker import CatalogReplay, worker_request
        W=sparse.csr_matrix([[0.,-1.5,0.,0.],[-1.5,0.,-2.5,0.],[0.,-2.5,0.,.5],[0.,0.,.5,0.]])
        rep=CatalogReplay(W)
        s=np.array([65536,-30000,-32000,5000],dtype=np.int64)
        for method in "FDTIS":
            session=Session(W,method,hot_size=100,weight_mode="weighted",timers=False)
            for opinion in (s,s,s[::-1].copy(),np.zeros_like(s)):
                trace=session.request(opinion,budget=2)
                if method=="S":
                    # Deliberately make this toy shortlist the entire catalog.
                    self.assertEqual(len(trace["initial_seed_ids"]),len(rep.pool[0]))
                    decisions=[{**row,"status":"accept" if row["status"]=="accepted" else row["status"]}
                               for row in trace["decisions"]]
                    result=rep.verify(opinion,decisions,trace["budget"],gain_denominator=trace["gain_denominator"])
                else:
                    result=worker_request(rep,trace,opinion,session.graph_hash,kind="weighted")
                self.assertEqual(result["accepts"],trace["commits"])
            trace=session.request(s,budget=1)
            if method=="D":
                for field in ("root_version_id","parent_version_id","current_version_id","low","i"):
                    bad=deepcopy(trace)
                    row=bad["decisions"][0]
                    row[field] = "wrong" if "version" in field else 123456789
                    with self.assertRaises(ValueError):
                        worker_request(rep,bad,s,session.graph_hash,kind="weighted")

    def test_weighted_exact_tie_switch_and_negative_catalog(self):
        from src.session import Session
        from src.checker import CatalogReplay, worker_request
        W=sparse.csr_matrix([[0.,-1.5,0.],[-1.5,0.,-1.5],[0.,-1.5,0.]])
        s=np.array([65536,-65536,65536],np.int64)
        rep=CatalogReplay(W)
        for method in "FD":
            sess=Session(W,method,weight_mode="weighted",timers=False)
            record=sess.request(s,budget=1)
            self.assertEqual(record["decisions"][0]["winner_id"],0)
            worker_request(rep,record,s,sess.graph_hash,kind="weighted")
            negative=np.full_like(s,65536)
            stop=sess.request(negative,budget=1)
            self.assertEqual(stop["status"],"tolerance_stop")
            worker_request(rep,stop,negative,sess.graph_hash,kind="weighted")

    def test_narrow_integer_ratings_widen_before_scaling(self):
        W=sparse.csr_matrix(np.array([[0,-10,0],[-10,0,10],[0,10,0]],dtype=np.int8))
        pool=build_universe(W)
        k=Kernel(W,pool)
        self.assertEqual(k.operator(W)[0,1],160)
        self.assertEqual(k.operator(W)[1,2],-160)
        self.assertEqual(k.d[pool[2]].tolist(),[160])
