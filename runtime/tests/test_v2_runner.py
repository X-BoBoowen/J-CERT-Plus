import unittest
from src import v2_runner as v2


class V2Runner(unittest.TestCase):
    def test_worker_rejects_zero_requests_and_changed_limit(self):
        protocol={'small':{'worker_seconds':180,'memory_bytes':2*1024**3},
                  'scale':{'root_profile_seconds':600,'q1_seconds':600,'q4_seconds':1800,'memory_bytes':4*1024**3}}
        job=v2.small_jobs()[0]|{'seconds':180,'memory_bytes':2*1024**3}
        v2.validate_job(job,protocol)
        for changed in ({'requests':0},{'memory_bytes':8*1024**3},{'budget':16},{'repeat_id':3}):
            with self.assertRaises(ValueError):v2.validate_job(job|changed,protocol)

    def test_registered_schedule_contains_54_unique_runs(self):
        rows=v2.small_jobs()
        self.assertEqual(len(rows),54)
        self.assertEqual(len({(r['graph'],r['stream'],r['repeat_id'],r['method']) for r in rows}),54)
        for start in range(0,54,3):
            self.assertEqual({r['method'] for r in rows[start:start+3]}, {'D','S','H'})
        self.assertEqual([r['method'] for r in rows[:3]],['D','S','H'])
        self.assertEqual([r['method'] for r in rows[18:21]],['S','H','D'])

    def test_seed_collision_rule_is_deterministic_and_excludes_used(self):
        first=v2.derive_seed('a'*64,'Alpha','independent',set())
        again=v2.derive_seed('a'*64,'Alpha','independent',{first['seed']})
        self.assertNotEqual(first['seed'],again['seed'])
        self.assertEqual(again['collision_counter'],1)
        self.assertEqual(again,v2.derive_seed('a'*64,'Alpha','independent',{first['seed']}))

    def test_resource_and_unresolved_are_not_certified_stops(self):
        for state in ['unresolved','timeout','memory_limit_exceeded','runtime/certificate_error']:
            self.assertFalse(v2.completed({'status':state}))
        self.assertTrue(v2.completed({'status':'completed_budget'}))
        self.assertTrue(v2.completed({'status':'certified_stop'}))
        self.assertTrue(v2.memory_failure('SUPERLU_MALLOC fails for buf in intMalloc()'))

    def test_direct_explicit_memory_preflight_uses_actual_payload(self):
        check=v2.explicit_preflight(3373067797,4*1024**3)
        self.assertEqual(check['minimum_pool_and_weights_bytes'],57342152549)
        self.assertEqual(check['status'],'not_started_memory_preflight')
        self.assertFalse(check['ordinal_capacity_admitted'])


if __name__=='__main__':unittest.main()
