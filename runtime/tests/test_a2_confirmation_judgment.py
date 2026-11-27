import copy
import hashlib
import json
import unittest
from src.a2_confirmation_judgment import evaluate,evaluate_bound


def fixture():
    protocol=dict(id='fixture-confirmation',graphs=['Alpha'],streams=['slow','signed-neighbor'],seed_labels=['confirmation-1'],repeat_count=3,
        confirmation_standard=dict(claim='fixture',rule=dict(primary_cells=[dict(graph='Alpha',stream='slow')],
            maximum_primary_cell_ratio=.6,maximum_primary_paired_ratio=1.)))
    cells=[];workers=[]
    for stream in protocol['streams']:
        costs={m:dict(complete_workers=3,Q4_seconds=[{'D':10.,'S':20.,'H':4.}[m]]*3) for m in ('D','S','H')}
        if stream=='signed-neighbor':
            for c in costs.values():c.update(complete_workers=0,Q4_seconds=[None]*3)
        cells.append(dict(graph='Alpha',stream=stream,seed_label='confirmation-1',costs=costs))
        workers.extend(dict(graph='Alpha',stream=stream,seed_label='confirmation-1',repeat_id=r,method=m,
            complete=stream=='slow',resource_failure=False) for r in range(3) for m in ('D','S','H'))
    analysis=dict(input_manifest_sha256='fixture',planned_workers=18,observed_result_workers=18,complete_workers=9,cells=cells,workers=workers,
        correctness_accounting_pass=True,structural_removal_observed=True,status_counts={'completed':9,'completed_with_failures':9})
    replay=dict(status='passed',protocol='fixture-confirmation',manifest_sha256='fixture',results=[dict(
        selection=dict(graph='Alpha',stream='slow',seed_label='confirmation-1',method='H',repeat_id=0,query_position=1),status='passed',
        proof=dict(proof=dict(full_catalog_decisions_certified=True,termination_certified=True)))])
    return protocol,analysis,replay


class ConfirmationJudgmentTests(unittest.TestCase):
    def test_common_outside_scope_unresolved_is_retained(self):
        result=evaluate(*fixture())
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['planned_workers'],18)
        self.assertEqual(result['complete_workers'],9)

    def test_primary_failure_cannot_be_imputed_or_dropped(self):
        p,a,r=fixture();a['cells'][0]['costs']['D']['Q4_seconds'][1]=None
        a['cells'][0]['costs']['D']['complete_workers']=2
        result=evaluate(p,a,r)
        self.assertEqual(result['status'],'failed')
        self.assertIn('all_primary_Q4_completed',result['failed_gates'])

    def test_mean_gate_does_not_hide_a_slow_repeat(self):
        p,a,r=fixture();a['cells'][0]['costs']['H']['Q4_seconds']=[1.,1.,11.]
        result=evaluate(p,a,r)
        self.assertTrue(result['gates']['primary_cell_cost_reduction'])
        self.assertFalse(result['gates']['every_primary_paired_repeat_not_slower'])

    def test_outside_scope_H_specific_failure_is_rejected(self):
        p,a,r=fixture()
        for row in a['workers']:
            if row['stream']=='signed-neighbor' and row['method']=='S':row['complete']=True
        self.assertFalse(evaluate(p,a,r)['gates']['no_H_specific_completion_failure'])

    def test_incomplete_independent_proof_does_not_pass(self):
        p,a,r=fixture();r['results'][0]['proof']['proof']['termination_certified']=False
        self.assertEqual(evaluate(p,a,r)['status'],'failed')

    def test_other_input_or_request_proof_cannot_replace_selected_replay(self):
        p,a,r=fixture();r['manifest_sha256']='different input'
        self.assertEqual(evaluate(p,a,r)['status'],'failed')
        p,a,r=fixture();r['results'][0]['selection']['query_position']=2
        self.assertEqual(evaluate(p,a,r)['status'],'failed')

    def test_missing_failure_evidence_does_not_pass(self):
        p,a,r=fixture();del a['workers'][0]['resource_failure']
        self.assertEqual(evaluate(p,a,r)['status'],'failed')

    def test_protocol_content_binding_rejects_changed_threshold(self):
        p,a,r=fixture();raw=json.dumps(p).encode();a['protocol_sha256']=hashlib.sha256(raw).hexdigest()
        self.assertEqual(evaluate_bound(raw,a,r)['status'],'passed')
        changed=copy.deepcopy(p);changed['confirmation_standard']['rule']['maximum_primary_cell_ratio']=.8
        with self.assertRaisesRegex(ValueError,'frozen protocol'):
            evaluate_bound(json.dumps(changed).encode(),a,r)


if __name__=='__main__':unittest.main()
