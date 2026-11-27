import unittest

import numpy as np

from src.access_replay import Source, verify_plan, match_timed_event


class AccessReplayTests(unittest.TestCase):
    def test_full_catalog_counts_reject_suboptimal_and_noncanonical_tau(self):
        # At tau 1 both endpoints move, the one candidate occurs twice; tau 2
        # does no work. At zero target all three counts tie when drift is zero.
        pool = (np.array([0]), np.array([1]), np.array([1]), np.array([False]))
        source = Source(pool, np.zeros(2, np.int64), 0)
        event = dict(tau=2, target=1000, posting_raw_visits=0, tail_raw_visits=0,
                     group_tails=[0], max_drift=2, pre_dedup_visits=0)
        ids = verify_plan(source, np.array([2, 2]), event)
        self.assertEqual(ids.tolist(), [])
        with self.assertRaisesRegex(Exception, 'minimum'):
            verify_plan(source, np.array([2, 2]),
                        event | dict(tau=1, posting_raw_visits=2, pre_dedup_visits=2))
        tied = event | dict(target=0, tau=1, tail_raw_visits=1, group_tails=[1],
                            max_drift=2, posting_raw_visits=0, pre_dedup_visits=1)
        # Isolated vertex 2 moves, both real endpoints stay put for both taus.
        source = Source(pool, np.zeros(3, np.int64), 0)
        self.assertEqual(verify_plan(source, np.array([0, 0, 2]), tied).tolist(), [0])
        with self.assertRaisesRegex(Exception, 'minimum'):
            verify_plan(source, np.array([0, 0, 2]), tied | {'tau': 2})

    def test_timed_event_agreement_includes_denied_plan_and_unique_counts(self):
        fields=dict(epoch=1,parent_version_id='parent',anchor_request=2,anchor_opinion_hash='op',
                    target=1,tau=2,pre_dedup_visits=3,returned_ids=1,max_drift=3,
                    posting_raw_visits=1,tail_raw_visits=2,group_tails=[2],
                    unique_ids_before_exclusion=2,budget=0,consumed_before=0,consumed=0,
                    action='fallback',fallback_reason='visit_budget_exceeded')
        rejected={k:fields[k] for k in ('anchor_request','anchor_opinion_hash','target','tau',
                  'pre_dedup_visits','posting_raw_visits','tail_raw_visits','group_tails','max_drift')}
        measured=fields | {'rejected_plan':rejected}
        audit=measured | {'rejected_plan':rejected | {'anchor_xc':[0]},'fine_x':[0]}
        match_timed_event(measured,audit)
        with self.assertRaisesRegex(Exception,'timed'):
            match_timed_event(measured,audit | {'returned_ids':2})
        with self.assertRaisesRegex(Exception,'timed'):
            match_timed_event(measured,audit | {'rejected_plan':rejected | {'tau':4}})

    def test_small_mid_request_switch_and_adversarial_replay(self):
        import copy
        from test_anchor_session import graph
        from src.anchor_session import AnchorSession
        from src.checker import CatalogReplay, opinion_hash, worker_request
        from src.common import opinions
        from src.access_replay import state, verify_audit
        W=graph()
        rows=opinions(8,'independent',104729,2)
        reference=AnchorSession(W,'budget-guard',visit_budget=100000)
        reference.request(rows[0],budget=3)
        old=reference.request(rows[1],budget=3)
        limit=old['discovery']['epochs'][0]['pre_dedup_visits']
        self.assertGreater(old['discovery']['epochs'][1]['pre_dedup_visits'],0)
        samples=[]
        for audit in (False,True):
            service=AnchorSession(W,'budget-guard',visit_budget=limit)
            samples.append([service.request(s,budget=3,audit=audit) for s in rows])
        timed,audited=samples
        rep=CatalogReplay(W)
        hashes={q:opinion_hash(s) for q,s in enumerate(rows,1)}
        witnesses={q:state(rep.A,rows[q-1],r['discovery']['epochs'][0])
                   for q,r in enumerate(audited,1)}
        self.assertEqual(timed[1]['discovery']['epochs'][1]['action'],'fallback')
        self.assertEqual(worker_request(rep,timed[1],rows[1],service.graph_hash,kind='weighted')['accepts'],3)
        result=verify_audit(rep,rows[1],2,timed[1],audited[1],1,witnesses,hashes,limit)
        self.assertEqual(result['old_visits'],limit)
        self.assertEqual(result['fallback_count'],1)
        for field in ('anchor_xc','drift','current_root_xc','fine_x','excluded_ids','returned_ids_sha256'):
            altered=copy.deepcopy(audited[1])
            e=altered['discovery']['epochs'][1]
            if field=='returned_ids_sha256':
                e[field]='a'*64
            elif field=='excluded_ids':
                e[field]=[]
            else:
                e[field][0]+=1
            with self.subTest(field=field), self.assertRaises(Exception):
                verify_audit(rep,rows[1],2,timed[1],altered,1,witnesses,hashes,limit)

    def test_exact_small_timed_file_and_fixture_digest(self):
        import json
        from fractions import Fraction
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from scipy import sparse
        from test_anchor_session import graph
        from src import access_replay as replay
        from src.anchor_session import AnchorSession
        from src.anchor_service_bench import save
        from src.common import opinions
        W=graph()
        mat=np.stack(opinions(8,'independent',104729,2))
        h=replay.checksum
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)
            (dest/'data/projected').mkdir(parents=True)
            (dest/'src').mkdir()
            (dest/'src/checker.py').write_bytes((replay.ROOT/'src/checker.py').read_bytes())
            gp=dest/'data/projected/Alpha_weighted.npz'
            sparse.save_npz(gp,W)
            np.savez(dest/'fixture.npz',opinions=mat)
            job=dict(graph='Alpha',stream='independent',seed=104729,repeat_id=0,arm='budget-guard',
                     label='guard',coefficient=0.)
            cfg=dict(id='tiny',graphs=['Alpha'],streams=['independent'],seeds=[104729],requests=2,
                     budget=3,methods=[dict(arm='budget-guard',label='guard',coefficient=0.)])
            pp,mp,tp=dest/'protocol.json',dest/'manifest.json',dest/'timed.json'
            pp.write_text(json.dumps(cfg))
            fixture=dict(graph='Alpha',stream='independent',seed=104729,fixture_id='tiny',
                         file='fixture.npz',shape=list(mat.shape),npz_sha256=h(dest/'fixture.npz'),
                         matrix_sha256=replay.digest(mat),opinion_hashes=[replay.opinion_hash(s) for s in mat])
            manifest=dict(verified=True,fixtures=[fixture],protocol_id='tiny',protocol_sha256=h(pp),
                          source_files_sha256={},graphs={'Alpha':dict(weighted_path='data/projected/Alpha_weighted.npz',
                                                                      weighted_sha256=h(gp),n=8)})
            mp.write_text(json.dumps(manifest))
            production=AnchorSession(W,'budget-guard',graph_hash=h(gp),visit_budget=0)
            measured=[]
            for q,s in enumerate(mat,1):
                req=production.request(s,budget=3,audit=False)
                req['query_position']=q
                measured.append(req)
            trace=dict(**job,requests=measured,method='D',weight_mode='weighted',audit=False,delta=0,
                       hot_size=64,budget=3,requests_planned=2,schedule_position=0,fixture_id='tiny',
                       protocol_id='tiny',protocol_sha256=h(pp),input_manifest_sha256=h(mp),
                       source_files_sha256={},fixture_npz_sha256=fixture['npz_sha256'],
                       fixture_matrix_sha256=fixture['matrix_sha256'],graph_hash=h(gp),
                       root_version_id=production.root_id,catalog_size=production.m,visit_budget=0)
            save(tp,trace)
            (dest/'SCHEDULE_RESULTS.json').write_text(json.dumps(dict(jobs=[dict(
                job=job,result=tp.name,position=0,status='completed')])))
            with patch.object(replay,'ROOT',dest):
                report=replay.run(pp,mp,dest,Fraction(0))
                self.assertEqual(report['verified'][0]['timed_sha256'],h(tp))
                self.assertEqual(report['verified'][0]['requests'][0]['behaviors'],['immediate'])
                self.assertEqual(report['verified'][0]['requests'][0]['actual_timed_output_proof']['accepts'],3)
                trace['requests'][1]['discovery']['epochs'][0]['rejected_plan']['tau']*=2
                save(tp,trace)
                with self.assertRaisesRegex(Exception,'timed rejected-plan'):
                    replay.run(pp,mp,dest,Fraction(0))

    def test_positive_cold_winner_threshold_needs_independent_warm_witness(self):
        from test_anchor_session import graph
        from src.anchor_session import AnchorSession
        from src.checker import CatalogReplay
        from src.common import opinions
        from src.access_replay import warm_witness
        W=graph()
        opinion=opinions(8,'independent',104729,1)[0]
        s=AnchorSession(W,'budget-guard',visit_budget=0)
        res=s.request(opinion,budget=1,audit=True)
        event=res['discovery']['epochs'][0].copy()
        winner=res['decisions'][0]['winner_id']
        event['seed_ids']=[winner]
        rep=CatalogReplay(W)
        self.assertGreater(event['target'],0)
        self.assertEqual(warm_witness(rep,rep.A,opinion,event,dict(winner_id=None)),(winner,1))
        event['seed_ids']=[]
        with self.assertRaisesRegex(ValueError,'warm threshold'):
            warm_witness(rep,rep.A,opinion,event,dict(winner_id=None))
