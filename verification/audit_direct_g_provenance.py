#!/usr/bin/env python3
"""Read-only archive binding and exact stored-interval comparison.

Repository adaptation: read the seven original scientific archives directly.
Redundant comparisons to private editorial snapshots and reviewer packet are omitted.
No production imports, solver calls, graph reconstruction, new inputs, or replay.
Only standard-library ZIP/JSON/hash/NPY-header/byte and Fraction operations run.
"""
from __future__ import annotations
import argparse
import ast
import io
import json
import struct
from collections import Counter
from contextlib import ExitStack
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

ARCHIVES = {
    'v2_code': ('artifacts/frozen/V2_Code.zip', '4d9d96167fdcaee440aced8b280a36118e0d142bc5d8550707316311d8433dfb'),
    'v2_evidence': ('artifacts/frozen/V2_Evidence.zip', '1e98f70bef0c22fd2f6d8b528490ab42ceedf75c7427f292d9b58965c86b9fcf'),
    'v2_inputs': ('artifacts/frozen/V2_Inputs.zip', 'b643dde2a09cc5c6866a1295d9923480f53b5a6b426beb2b0c557949e4346308'),
    'v3_code': ('artifacts/frozen/V3_Code.zip', 'e88a49aef90acb09d483dfd4c5016d4336c5a8a353b40087ef464cc5b4c4e7c3'),
    'v3_evidence': ('artifacts/frozen/V3_Evidence.zip', '67096d7ddc7d52e3dfd5a1c3ae441a56daf4f273c3fbd76b9fa54e23691fca90'),
    'v4_code': ('artifacts/frozen/V4_Code.zip', 'a8ec381746db4fab185511119c5eab6f1780692a4dbff4d0a156559244d18358'),
    'v4_evidence': ('artifacts/frozen/V4_Evidence.zip', 'cfe27334aee896c4ebb7116976ed979cbc873e978b3c78ed79caee8b2a2db065'),
}
ID_FIELDS = ('epoch','root_version_id','parent_version_id','current_version_id','winner_id','i','j','d','flip','status','live_catalog')
TOP_FIELDS = ('root_version_id','operator_hash','catalog_hash','graph_hash','n','catalog_size')
REQ_FIELDS = ('root_version_id','current_version_id','opinion_hash','query_position','budget','delta','gain_denominator','commits','status')
DOMAIN = 'all-nonedges-positive1/16+root-negative-flips:v1'
SALT = 'weighted-normalized:scale16:proof48:screen20:allnonedges-v1'
V3_FREEZE = 'v3/resumed/protocols/HG_FROZEN.json'
V4_FREEZE = 'v4/protocols/HG_LIGHT_FROZEN.json'
EDITORIAL_ROOT = 'J_CERT_Plus_PVLDB_Editorial_Source_20261126/paper/'


def need(value, context):
    if not value:
        raise AssertionError(context)


def hash_bytes(data):
    return sha256(data).hexdigest()


def json_digest(obj):
    return hash_bytes(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


class Store:
    def __init__(self, root, stack):
        self.root, self.archives, self.sources, self.manifests = root, {}, {}, {}
        for label,(path,expected) in ARCHIVES.items():
            data=(root/path).read_bytes()
            got=hash_bytes(data)
            need(got == expected, f'{label} package hash')
            z=stack.enter_context(ZipFile(io.BytesIO(data)))
            self.archives[label]=z
            self.sources[label]={'path':path,'bytes':len(data),'sha256':got,'members':{}}
            if label.startswith('v2_'):
                manifest_name='MANIFEST_'+label[3:].upper()+'.json'
                manifest=self.json(label,manifest_name)
                self.manifests[label]={row['path']:row for row in manifest['members']}

    def read(self,label,member):
        data=self.archives[label].read(member)  # ZipFile also checks member CRC.
        pin={'bytes':len(data),'sha256':hash_bytes(data)}
        if label in self.manifests:
            declared=self.manifests[label][member]
            need(all(pin[k]==declared[k] for k in pin),f'{label}/{member} manifest')
            pin['internal_manifest_verified']=True
        self.sources[label]['members'][member]=pin
        return data

    def json(self,label,member):
        return json.loads(self.read(label,member))

    def hash(self,label,member):
        return hash_bytes(self.read(label,member))


def opinion_payload(npz_bytes, expected_shape):
    """Hash stored little-endian int64 payload; no NumPy or arithmetic solver."""
    with ZipFile(io.BytesIO(npz_bytes)) as z:
        need(z.namelist()==['opinions.npy'], 'opinion NPZ members')
        data=z.read('opinions.npy')
    need(data[:8]==b'\x93NUMPY\x01\x00','supported NPY header')
    length=struct.unpack('<H',data[8:10])[0]
    header=ast.literal_eval(data[10:10+length].decode('latin1').strip())
    need(header['descr']=='<i8' and header['fortran_order'] is False, 'exact opinion byte encoding')
    need(list(header['shape'])==expected_shape,'exact opinion shape')
    rows,cols=header['shape']; payload=data[10+length:]
    need(len(payload)==rows*cols*8,'opinion payload size')
    return {'shape':list(header['shape']),'dtype':header['descr'],
            'matrix_sha256':hash_bytes(payload),
            'opinion_hashes':[hash_bytes(payload[i*cols*8:(i+1)*cols*8]) for i in range(rows)],
            'operation':'NPY header parsing and byte hashing only'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    with ExitStack() as stack:
        s=Store(args.workspace.resolve(),stack)
        v2=s.json('v2_evidence','v2/CODE_FREEZE.json')
        inputs=s.json('v2_evidence','v2/INPUTS.json')
        protocol=s.json('v2_evidence','v2/PROTOCOL.json')
        pre=s.json('v2_evidence','v2/PRE_REGISTRATION.json')
        v3=s.json('v3_evidence',V3_FREEZE)
        v4=s.json('v4_evidence',V4_FREEZE)
        need(v2['commit']=='4c2c7b7e84c7c03ee56d45453fea6950c94817bd','V2 commit')
        need(v3['runtime_commit']=='c6b752630bd8dbe1f9fd296a7ed02f249f834575','V3 commit')
        need(v4['runtime_commit']=='9d09794b6a6de8a4c97f615abf3f3cae1f458b75','V4 commit')
        need(inputs['code']==v2,'V2 input->code freeze')
        need(s.hash('v2_evidence','v2/PROTOCOL.json')==inputs['protocol_sha256'],'V2 protocol hash')
        need(s.hash('v2_evidence','v2/INPUTS.json')==v4['v2_input_manifest_sha256'],'V4 input manifest hash')
        need(s.hash('v3_evidence',V3_FREEZE)==v4['historical_V3_freeze_sha256'],'V3->V4 freeze hash')
        need(v3['contracts']==v4['contracts'],'V3/V4 contracts')
        need(v3['input_fixtures']==v4['input_fixtures'],'V3/V4 fixtures')
        need(v3['input_graphs']==v4['input_graphs'],'V3/V4 graphs')
        v2_actual_source={}
        for name,expected in v2['source_files_sha256'].items():
            got=s.hash('v2_code',name);need(got==expected,'V2 frozen source '+name)
            v2_actual_source[name]=got
        checker=s.read('v2_code','src/a2_checker.py').decode()
        validator=s.read('v2_evidence','validation/v2_validate.py').decode()
        for literal in [f'DOMAIN = "{DOMAIN}"',f'SALT = "{SALT}"',
                        'need(request["delta"] == 0 and request["gain_denominator"] == 2 ** 96,',
                        'need(delta > 0 or e < rival_id, "tie canonical winner")']:
            need(literal in checker,'V2 checker contract source '+literal)
        need("json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)" in validator,'request hash source')
        need(v3['intentional_baseline_changes']['src/a2_checker.py']['v2']==hash_bytes(checker.encode()),'V3 checker baseline')
        for generation,freeze in [('v3',v3),('v4',v4)]:
            for name in ['src/a2_checker.py','src/a2_session.py','src/g_access.py']:
                origin = generation+'_code'
                member='runtime_files/'+name
                if member not in s.archives[origin].namelist() and generation=='v4':
                    origin='v3_code'  # V4 is an overlay; unchanged checker comes from V3.
                need(s.hash(origin,member)==freeze['source_files_sha256'][name],generation+' runtime source '+name)
        requests=[];states=set();observations=[]
        for graph,index in [('Alpha','004'),('OTC','013')]:
            report_name=f'validation/a_full_catalog_{graph}.json'
            report=s.json('v2_evidence',report_name)
            selection={'graph':graph,'stream':'independent','method':'H','repeat_id':0,'query_position':1}
            need(report['selection']==selection and selection in pre['independent_selection'],'preselected request '+graph)
            need(report['frozen_commit']==v2['commit'] and report['validator_sha256']==hash_bytes(validator.encode()),'report version/source '+graph)
            need(report['full_request_certified'] and report['terminal_status_certified'] and report['accepted_decisions_independently_certified']==15,'archived replay status '+graph)
            need(not report['replacement_used'] and not report['request_trimmed'],'preselected request unmodified '+graph)
            replay=report['proof']; proof=replay['proof']
            name=f'v2/raw_a/{index}_{graph}_independent_H_r0.json'
            worker=s.json('v2_evidence',name);request=worker['requests'][0]
            need(s.hash('v2_evidence',name)==replay['trace_sha256'],'original worker bytes '+graph)
            need(Path(report['raw_file']).name==Path(name).name,'original worker path '+graph)
            need(json_digest(request)==report['selected_request_sha256'],'original request canonical bytes '+graph)
            need(worker['source_files_sha256']==v2_actual_source,'worker source bindings '+graph)
            need(worker['input_manifest_sha256']==s.hash('v2_evidence','v2/INPUTS.json'),'worker manifest '+graph)
            need(worker['protocol_sha256']==inputs['protocol_sha256'],'worker protocol '+graph)
            need(replay['checker_sha256']==s.hash('v2_code','src/a2_checker.py'),'checker source '+graph)
            fixture=worker['fixture'];input_fixture=next(x for x in inputs['fixtures'] if x['graph']==graph and x['stream']=='independent')
            need(fixture==input_fixture,'fixture manifest '+graph)
            graph_member=inputs['graphs'][graph]['weighted_path'];graph_hash=s.hash('v2_inputs',graph_member)
            need(graph_hash==worker['graph_hash']==replay['graph_sha256']==v4['input_graphs'][graph]['weighted_sha256'],'original graph bytes '+graph)
            opinion_member='v2/'+fixture['file'];opinion_bytes=s.read('v2_inputs',opinion_member)
            need(hash_bytes(opinion_bytes)==fixture['npz_sha256']==replay['opinions_file_sha256'],'original opinion archive bytes '+graph)
            opinion=opinion_payload(opinion_bytes,fixture['shape'])
            need(opinion['matrix_sha256']==fixture['matrix_sha256'] and opinion['opinion_hashes']==fixture['opinion_hashes'],'opinion payload digest '+graph)
            need(opinion['opinion_hashes'][0]==request['opinion_hash']==replay['opinion_hash'],'original selected opinion bytes '+graph)
            cat=hash_bytes(f'{DOMAIN}|{graph_hash}'.encode())
            root=hash_bytes('/'.join((graph_hash,worker['operator_hash'],cat,SALT)).encode())
            need(cat==worker['catalog_hash'] and root==worker['root_version_id']==request['root_version_id'],'declared root recomputation '+graph)
            need(request['delta']==0 and request['budget']==15 and request['gain_denominator']==2**96 and request['commits']==15,'V2 original request contract '+graph)
            need(request['status']==proof['termination']=='budget_stop' and request['current_version_id']==proof['current_version_id'],'archived request terminal '+graph)
            need([x['winner_id'] for x in request['decisions']]==proof['plan'],'archived accepted plan '+graph)
            need(proof['full_catalog_decisions_certified'] and proof['termination_certified'] and len(proof['decision_checks'])==15,'archived full decision scope '+graph)
            parent=root
            for row in request['decisions']:
                need(row['root_version_id']==root and row['parent_version_id']==parent,'V2 direct parent chain '+graph)
                parent=hash_bytes(f"{parent}/{row['winner_id']}".encode())
                need(row['current_version_id']==parent,'V2 direct child chain '+graph)
                need(row['winner_id']==row['i']*worker['n']+row['j'] and 0<=row['i']<row['j']<worker['n'],'canonical pair bytes '+graph)
            need(parent==request['current_version_id'],'V2 final chain '+graph)
            job3=next(x for x in v3['jobs'] if (x['graph'],x['stream'],x['method'])==(graph,'independent','G'))
            name3='v3/resumed/raw_hg/'+job3['output_file'];worker3=s.json('v3_evidence',name3);request3=worker3['requests'][0]
            need(worker3['freeze_sha256']==s.hash('v3_evidence',V3_FREEZE) and worker3['actual_runtime_commit']==v3['runtime_commit'],'V3 worker freeze '+graph)
            need(all(worker3[k]==worker[k] for k in TOP_FIELDS),'direct V2/V3 top-level identity '+graph)
            need(all(request3[k]==request[k] for k in REQ_FIELDS),'direct V2/V3 request identity '+graph)
            need([[x[k] for k in ID_FIELDS] for x in request3['decisions']]==[[x[k] for k in ID_FIELDS] for x in request['decisions']],'direct V2/V3 step identity '+graph)
            g_requests=[]
            for job in v4['jobs']:
                if (job['graph'],job['stream'],job['method'])!=(graph,'independent','G'):continue
                name4='v4/raw_hg_light/'+job['output_file'];worker4=s.json('v4_evidence',name4);request4=worker4['requests'][0]
                need(worker4['job']==job and worker4['freeze_sha256']==s.hash('v4_evidence',V4_FREEZE) and worker4['actual_runtime_commit']==v4['runtime_commit'],'V4 worker freeze '+name4)
                need(worker4['measurement_policy']=='light_v1' and worker4['run_kind']=='formal','V4 measurement '+name4)
                need(worker4['fixture']==fixture,'direct V2/V4 fixture identity '+name4)
                need(all(worker4[k]==worker[k] for k in TOP_FIELDS),'direct V2/V4 top-level identity '+name4)
                need(all(request4[k]==request[k] for k in REQ_FIELDS),'direct V2/V4 request identity '+name4)
                need([[x[k] for k in ID_FIELDS] for x in request4['decisions']]==[[x[k] for k in ID_FIELDS] for x in request['decisions']],'direct V2/V4 step identity '+name4)
                g_checks=[]
                need(len(request4['access']['epochs'])==15,'selected G access event count '+name4)
                for oldrow,row,check,event in zip(request['decisions'],request4['decisions'],proof['decision_checks'],request4['access']['epochs']):
                    need(check['certified'] and check['status']==row['status']=='accepted' and check['epoch']==row['epoch'],'archived individual scope '+name4)
                    need(check['legal_catalog_size']==row['live_catalog']==oldrow['live_catalog'],'legal catalog cardinality '+name4)
                    need(check['structural_tie_attempts']==check['structural_tie_rivals']==0,'archived tie absence '+name4)
                    lower,upper=(Fraction(*x) for x in check['independent_gain_interval'])
                    glo,ghi=Fraction(row['low'],request4['gain_denominator']),Fraction(row['high'],request4['gain_denominator'])
                    need(glo<=lower<=upper<=ghi and lower>0 and glo>0,'exact interval containment and strict positivity '+name4)
                    need(event['epoch']==row['epoch'] and event['parent_version_id']==row['parent_version_id'],'selected G access binding '+name4)
                    access_path='whole_insertion_excluded' if event['g_whole_insertion_excluded'] else 'blocks_entered' if event['g_pair_blocks_tested']>0 else 'neither'
                    key=(graph_hash,worker['operator_hash'],cat,request['opinion_hash'],row['parent_version_id'])
                    states.add(key)
                    out={'graph':graph,'repeat_id':job['repeat_id'],'query_position':1,
                         **{k:row[k] for k in ID_FIELDS},
                         'state_key':list(key),
                         'independent_interval':[str(lower),str(upper)],
                         'g_published_interval':[str(glo),str(ghi)],
                         'independent_interval_strictly_positive':True,
                         'g_published_lower_bound_strictly_positive':True,
                         'g_recorded_access_path':access_path,
                         'winner_type':'flip' if row['flip'] else 'insertion',
                         'g_contains_archived_independent_interval':True,
                         'direct_v2_h_to_v4_g_identity_equal':True,
                         'original_checker_legal_candidate_digest':check['candidate_ids_sha256'],
                         'original_checker_full_decision_certified':True,
                         'source_member':name4,'archived_replay_member':report_name,
                         'new_full_catalog_replay':False,'g_block_pruning_independently_audited':False}
                    g_checks.append(out);observations.append(out)
                g_requests.append({'repeat_id':job['repeat_id'],'source_member':name4,'checks':g_checks})
            need(sorted(x['repeat_id'] for x in g_requests)==[0,1,2],'G repetitions '+graph)
            requests.append({'graph':graph,'selection':selection,'original_v2_worker_member':name,
                'original_v2_worker_sha256':replay['trace_sha256'],'original_v2_request_sha256':report['selected_request_sha256'],
                'archived_full_replay_member':report_name,'archived_checker_source_sha256':replay['checker_sha256'],
                'archived_validator_source_sha256':report['validator_sha256'],
                'graph_member':graph_member,'graph_sha256':graph_hash,'opinion_npz_member':opinion_member,
                'opinion_npz_sha256':hash_bytes(opinion_bytes),'opinion_payload_rehash':opinion,
                'selected_opinion_sha256':request['opinion_hash'],'root_operator_hash_declared':worker['operator_hash'],
                'catalog_hash_recomputed':cat,'root_version_id_recomputed':root,
                'direct_parent_child_candidate_chain_equal':True,
                'v3_intermediate_binding_also_matches_direct_v2':True,'v3_worker_member':name3,
                'g_observations':g_requests})
        need(len(states)==30 and len(observations)==90,'archived state and observation counts')
        result={'schema':'jcert-repository-direct-provenance-v1','status':'passed',
            'script_sha256':hash_bytes(Path(__file__).read_bytes()),
            'operations':{'new_production_workers':0,'new_numerical_solves':0,'new_inputs':0,
                'new_full_catalog_replays':0,'production_modules_imported':0,
                'archived_interval_comparisons':len(observations),'distinct_states':len(states),
                'original_v2_source_file_pins_checked':len(v2_actual_source)},
            'conclusions':{'all_90_exact_containments':True,'all_30_original_preselected_states_bound':True,
                'raw_v2_h_to_v4_g_identity_checked_without_v3_transitive_bridge':True,
                'v3_bridge_also_consistent':True,'old_v2_original_bytes_now_available':True,
                'g_has_new_independent_full_catalog_replay':False,
                'independent_state_count_with_h_and_g_combined':30},
            'selected_30_state_descriptive_mix':dict(Counter(x['winner_type']+' / '+x['g_recorded_access_path'] for x in observations if x['repeat_id']==0)),
            'scope_limits':{
                'matrices':'Graph-file bytes and declared root-operator identity are rehashed/bound. No physical parent matrix was reconstructed, solved, or compared anew.',
                'catalog':'Root graph digest, declared legal-domain contract and exact selected-edit chains match. Archived checker candidate-ID hashes are retained. G supplies no independent full-domain ID stream digest; cardinality equality is not a separate set-equality proof.',
                'numerics':'Existing checker enclosures and accepted-decision results are reused. Exact Fraction comparisons prove stored interval inclusion only; their numerical validity is not re-executed.',
                'coverage':'The audit does not independently inspect each G block pruning step, all G competitors, or G tie resolution.',
                'samples':'Thirty mathematical states are shared with the original H replay. Ninety G observations are three timing repetitions of those states. These are not 60 independent states or 90 independent replays.',
                'source':'All 102 V2 worker source/native pins and accessed archive-member manifest entries match. Runtime source text was read and hashed, never imported or executed.'},
            'requests':requests,'sources':s.sources}
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
    print(json.dumps({'status':result['status'],'distinct_states':len(states),'observations':len(observations),
                      'new_solves':0,'new_replays':0,'output':str(args.output)},sort_keys=True))

if __name__=='__main__':
    main()
