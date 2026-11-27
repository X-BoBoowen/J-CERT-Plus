from copy import deepcopy
import unittest
import numpy as np
from scipy import sparse

from src.anchor_session import AnchorSession
from src.anchor_replay import verify_events
from src.checker import CatalogReplay,worker_request
from src.holdout_stream import signed_neighbor_opinions


class CrossOpinionReplayTests(unittest.TestCase):
    def test_independent_index_union_and_tampering(self):
        W=sparse.csr_matrix([[0.,-1.5,0.,0.],[-1.5,0.,-2.5,0.],[0.,-2.5,0.,.5],[0.,0.,.5,0.]])
        rep=CatalogReplay(W)
        service=AnchorSession(W,"retain-always",timers=True,hot_size=1)
        s=np.array([65536,-30000,-32000,5000],dtype=np.int64)
        t=np.array([64536,-29000,-31970,5100],dtype=np.int64)
        first=service.request(s,budget=1,audit=True)
        second=service.request(t,budget=1,audit=True)
        state={}
        verify_events(rep,s,first,1,state)
        verify_events(rep,t,second,2,state)
        worker_request(rep,second,t,service.graph_hash,kind="weighted")
        for field in ("fine_radius","target","anchor_opinion_hash","returned_ids_sha256","excluded_ids","drift"):
            rec=deepcopy(second)
            e=rec["discovery"]["epochs"][0]
            if field=="drift": e[field][0]=-1
            elif field=="excluded_ids": e[field]=[]
            elif field in ("target","fine_radius"): e[field]=-1
            else: e[field]="bad"
            with self.assertRaises(ValueError):
                verify_events(rep,t,rec,2,dict(state))

    def test_signed_average_uses_unit_signs_and_preserves_isolate(self):
        W=sparse.csr_matrix([[0.,-1,0],[-1,0,0],[0,0,0]])
        # Exercising only a toy seed. The confirmation seed is intentionally unused here.
        toy_seed=29
        rng=np.random.Generator(np.random.PCG64(toy_seed))
        out=signed_neighbor_opinions(W,2,seed=toy_seed)
        for actual in out:
            u=rng.uniform(-1,1,3)
            expected=np.rint(32768*np.array([u[0]-u[1],u[1]-u[0],u[2]])).astype(np.int64)
            np.testing.assert_equal(actual,expected)
        with self.assertRaises(ValueError):
            signed_neighbor_opinions(W*2,1,seed=toy_seed)
