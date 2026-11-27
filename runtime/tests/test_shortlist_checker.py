from copy import deepcopy
import unittest
import numpy as np
from scipy import sparse
from src.session import Session
from src.checker import CatalogReplay
from src.shortlist_checker import verify_request


class LimitedScopeTests(unittest.TestCase):
    def test_prove_registered_top_k_and_refuse_tampering(self):
        W=sparse.csr_matrix([[0.,-1.5,0.,0.],[-1.5,0.,-2.5,0.],[0.,-2.5,0.,.5],[0.,0.,.5,0.]])
        session=Session(W,"S",hot_size=2,timers=False,weight_mode="weighted")
        s=np.array([65536,-30000,-32000,5000],dtype=np.int64)
        req=session.request(s,budget=1,audit=True)
        rep=CatalogReplay(W)
        invoke=lambda r:verify_request(rep,s,r,session.graph_hash,"weighted",2)
        checked=invoke(req)
        self.assertEqual(checked["scope"],"registered_root_shortlist_only")
        self.assertEqual(checked["candidate_visits"],2)
        self.assertGreater(checked["catalog_size"],2)
        for key in ("x","r","seed","child","endpoint"):
            broken=deepcopy(req)
            if key=="x": broken["snapshots"][0]["x"][0]+=1024
            if key=="r": broken["snapshots"][0]["r"]+=1
            if key=="seed": broken["initial_seed_ids"].reverse()
            if key=="child": broken["decisions"][0]["current_version_id"]="different"
            if key=="endpoint": broken["decisions"][0]["i"]=-1
            with self.assertRaises(ValueError): invoke(broken)
