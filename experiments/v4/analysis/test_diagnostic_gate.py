"""Targeted diagnostic evidence-presence checks, not another research input."""
import copy
import importlib
from pathlib import Path
import unittest


class DiagnosticGateChecks(unittest.TestCase):
    def module(self):
        self.assertTrue((Path(__file__).with_name('diagnostic_support.py')).is_file(),
                        'diagnostic-only cache capture and positive evidence gate are required')
        return importlib.import_module('diagnostic_support')

    def test_diagnosis_rejects_truncation_missing_audit_and_missing_deep_count(self):
        m=self.module()
        request={'query_position':1,'status':'budget_stop','commits':15,
                 'measurement_policy':'light_v1',
                 'diagnostic_cache_audit':{'schema':'v4-final-cache-audit-1','captured_views':True,
                    'captured_factor':True,'entry_count':3,'entries_sha256':'a'*64,
                    'step_count':15,'steps_sha256':'b'*64,'factor_base_is_shared_root':True,
                    'root_factor_step_count':0},
                 'diagnostic_deep_calls':{key:0 for key in m.DEEP_LABELS}}
        worker={'status':'completed_budget','complete_Q4':True,'method':'H','measurement_policy':'light_v1',
                'requests':[dict(copy.deepcopy(request),query_position=q) for q in range(1,5)]}
        self.assertEqual(m.diagnostic_obligation_issues(worker),[])
        changed=copy.deepcopy(worker);changed['requests'].pop()
        self.assertTrue(m.diagnostic_obligation_issues(changed))
        changed=copy.deepcopy(worker);del changed['requests'][0]['diagnostic_cache_audit']
        self.assertTrue(m.diagnostic_obligation_issues(changed))
        changed=copy.deepcopy(worker);changed['requests'][0]['diagnostic_deep_calls'].pop(m.DEEP_LABELS[0])
        self.assertTrue(m.diagnostic_obligation_issues(changed))
        changed=copy.deepcopy(worker);changed['requests'][0]['diagnostic_deep_calls'][m.DEEP_LABELS[0]]=1
        self.assertTrue(m.diagnostic_obligation_issues(changed))

    def test_real_tiny_cache_capture_is_policy_invariant_and_not_empty_evidence(self):
        m=self.module()
        import numpy as np
        from scipy import sparse
        from src.a2_session import A2Session
        from src.common import DEN
        W=sparse.csr_matrix(np.array([[0,-1,0,0],[-1,0,1,0],[0,1,0,-1],[0,0,-1,0]],float))
        s=np.array([DEN,-DEN,-DEN,DEN],np.int64)
        for method in ('H','G'):
            rows=[]
            with m.ReportingProbe() as probe:
                cls=m.diagnostic_session_type(A2Session,probe)
                for policy in ('legacy_deep','light_v1'):
                    service=cls(W,method,measurement_policy=policy)
                    rows.append(service.request(s,budget=1))
            a,b=[r['diagnostic_cache_audit'] for r in rows]
            self.assertTrue(a['captured_views']);self.assertTrue(a['captured_factor'])
            self.assertTrue(a['factor_base_is_shared_root']);self.assertEqual(a['root_factor_step_count'],0)
            self.assertEqual(a['step_count'],1);self.assertEqual(a,b)
            self.assertEqual(a['entry_count'],rows[0]['memory']['gram_count'])
            self.assertTrue(all(v==0 for v in rows[1]['diagnostic_deep_calls'].values()))
            self.assertGreater(rows[0]['diagnostic_deep_calls']['catalog'],0)


if __name__=='__main__':unittest.main(verbosity=2)
