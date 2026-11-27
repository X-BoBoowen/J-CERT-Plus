"""V4 contract checks: observation must not change the actual service."""
import copy
import inspect
import os
from pathlib import Path
import sys
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from scipy import sparse

from src.a2_session import A2Session, lp
from src.a2_access import Access
from src.a2_catalog import AllCatalog, MixedFrontier
from src.g_access import GAccess
from src.common import DEN


def path_graph():
    return sparse.csr_matrix(np.array([[0,-1,0,0],[-1,0,1,0],
                                      [0,1,0,-1],[0,0,-1,0]],dtype=float))


class MeasurementPolicyChecks(unittest.TestCase):
    def setUp(self):
        self.assertIn('measurement_policy',inspect.signature(A2Session).parameters,
                      'the explicit policy API is required; an environment switch is insufficient')
        self.s=np.array([DEN,-DEN,-DEN,DEN],np.int64)

    def projection(self, result):
        sys.path.insert(0,os.environ['V4_ANALYSIS_ROOT'])
        from check_semantic_invariance import request_semantics
        return request_semantics(result)

    def test_unknown_measurement_policy_rejected(self):
        for method in ('H','G'):
            with self.subTest(method=method), self.assertRaises(ValueError):
                A2Session(path_graph(),method,measurement_policy='light_typo')

    def test_formal_worker_requires_explicit_matching_light_policy(self):
        from src import measurement_policy as policy
        self.assertTrue(hasattr(policy,'worker_policy'),'worker must reject missing/mismatched frozen policy')
        for frozen,job in (({'run_kind':'formal','measurement_policy':'light_v1'},{}),
                           ({'run_kind':'formal','measurement_policy':'light_v1'},{'measurement_policy':'legacy_deep'}),
                           ({'run_kind':'formal','measurement_policy':'legacy_deep'},{'measurement_policy':'legacy_deep'})):
            with self.assertRaises(ValueError):policy.worker_policy(frozen,job)
        self.assertEqual(policy.worker_policy({'run_kind':'formal','measurement_policy':'light_v1'},
                                              {'measurement_policy':'light_v1'}),'light_v1')

    def test_light_policy_never_calls_deep_walks(self):
        # A missing guard on any named deep-report family must break this test,
        # while the real solver, candidate search and acceptance still run.
        targets=((Access,'observe_frontier'),(MixedFrontier,'memory'),
                 (GAccess,'_observe_memory'),(AllCatalog,'memory'),(lp,'store_bytes'))
        with ExitStack() as stack:
            for owner,name in targets:
                stack.enter_context(patch.object(owner,name,side_effect=AssertionError('deep report called: '+name)))
            for method in ('H','G'):
                service=A2Session(path_graph(),method,measurement_policy='light_v1')
                normal=service.request(self.s,budget=1)
                self.assertEqual(normal['decisions'][0]['winner_id'],1)
                self.assertEqual(normal['commits'],1)
                zero=service.request(np.zeros(4,np.int64),budget=1)
                self.assertEqual(zero['status'],'tolerance_stop')
                self.assertEqual(zero['commits'],0)
                empty=service.request(self.s,budget=0)
                self.assertEqual(empty['commits'],0)

    def test_legacy_policy_retains_reporting(self):
        counts={}
        def wrap(owner,name,label):
            original=getattr(owner,name)
            def observed(*args,**kwargs):
                counts[label]=counts.get(label,0)+1
                return original(*args,**kwargs)
            return patch.object(owner,name,observed)
        with ExitStack() as stack:
            for owner,name,label in ((Access,'observe_frontier','H'),(GAccess,'_observe_memory','G'),
                                     (AllCatalog,'memory','catalog'),(lp,'store_bytes','cache')):
                stack.enter_context(wrap(owner,name,label))
            for method in ('H','G'):
                result=A2Session(path_graph(),method).request(self.s,budget=1)
                self.assertEqual(result['measurement_policy'],'legacy_deep')
                self.assertGreater(result['memory']['peak_frontier_owned_bytes'],0)
        self.assertGreater(counts['H'],0);self.assertGreater(counts['G'],0)
        self.assertEqual(counts['catalog'],2);self.assertEqual(counts['cache'],6)

    def test_measurement_policy_preserves_small_service_semantics(self):
        near=self.s.copy();near[1]+=1
        for method in ('H','G'):
            old=A2Session(path_graph(),method,measurement_policy='legacy_deep')
            new=A2Session(path_graph(),method,measurement_policy='light_v1')
            for opinion,budget,winner in ((self.s,0,None),(self.s,1,1),(near,1,11),(self.s,1,1)):
                before=old.request(opinion,budget=budget)
                after=new.request(opinion,budget=budget)
                self.assertEqual(self.projection(before),self.projection(after))
                self.assertEqual(after['initial_gram_count'],0)
                if winner is not None:self.assertEqual(after['decisions'][0]['winner_id'],winner)

    def test_light_memory_schema_does_not_claim_deep_peak(self):
        for method in ('H','G'):
            service=A2Session(path_graph(),method,measurement_policy='light_v1')
            for budget,points in ((0,['request_begin','request_end']),
                                  (1,['request_begin','version_open','version_finish','request_end'])):
                result=service.request(self.s,budget=budget)
                self.assertEqual(result['memory_observation_schema'],'light_v1')
                memory=result['memory']
                for name in ('peak_frontier_owned_bytes','gram_python_bytes',
                             'actual_candidate_cache_python_bytes','decision_and_access_log_python_bytes',
                             'low_rank_steps','peak_low_rank_steps','sparse_catalog'):
                    self.assertIsNone(memory.get(name),name)
                self.assertEqual([r['point'] for r in memory['light_observations']],points)
                self.assertEqual(memory['low_rank_step_count'],budget)
                self.assertIn('gram_count',memory);self.assertIn('peak_gram_count',memory)

    def test_semantic_projection_detects_endpoint_and_cache_work_changes(self):
        original=A2Session(path_graph(),'H').request(self.s,budget=1)
        changed=copy.deepcopy(original);changed['decisions'][0]['low']+=1
        self.assertNotEqual(self.projection(original),self.projection(changed))
        changed=copy.deepcopy(original);changed['memory']['gram_count']+=1
        self.assertNotEqual(self.projection(original),self.projection(changed))
        changed=copy.deepcopy(original);changed['counters']['candidate_rhs']+=1
        self.assertNotEqual(self.projection(original),self.projection(changed))
        changed=copy.deepcopy(original);changed['new_unregistered_business_field']=1
        self.assertNotEqual(self.projection(original),self.projection(changed))
        changed=copy.deepcopy(original);changed['memory']['new_unregistered_cache_field']=1
        self.assertNotEqual(self.projection(original),self.projection(changed))
        changed=copy.deepcopy(original);changed['request_seconds']+=123
        changed['memory']['gram_python_bytes']=None
        self.assertEqual(self.projection(original),self.projection(changed))


if __name__=='__main__':unittest.main(verbosity=2)
