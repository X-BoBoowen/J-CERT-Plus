"""Evaluate the one preregistered A2 confirmation claim without retuning it."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def evaluate(protocol, analysis, replay):
    standard=protocol['confirmation_standard']
    rule=standard['rule']
    required={(cell['graph'],cell['stream'],label)
              for cell in rule['primary_cells'] for label in protocol['seed_labels']}
    lookup={(cell['graph'],cell['stream'],cell['seed_label']):cell for cell in analysis['cells']}
    full={(g,s,label) for g in protocol['graphs'] for s in protocol['streams'] for label in protocol['seed_labels']}
    selected_label=min(protocol['seed_labels'])
    expected_replay={(g,'slow',selected_label,'H',0,1) for g in protocol['graphs']}
    replay_keys=('graph','stream','seed_label','method','repeat_id','query_position')
    observed_replay={tuple(r['selection'].get(k) for k in replay_keys) for r in replay['results']}
    roster=analysis['workers'];planned=len(full)*3*protocol['repeat_count']
    expected_roster={(g,s,label,repeat,m) for g,s,label in full
                     for repeat in range(protocol['repeat_count']) for m in ('D','S','H')}
    actual_roster={tuple(row.get(k) for k in ('graph','stream','seed_label','repeat_id','method')) for row in roster}
    gates=dict(full_roster_retained=(analysis['planned_workers']==planned and
        analysis['observed_result_workers']==planned and len(roster)==planned and set(lookup)==full and
        actual_roster==expected_roster and all(type(row.get(k)) is bool for row in roster
                                               for k in ('complete','resource_failure'))),
        original_integrity_and_output_agreement=analysis['correctness_accounting_pass'],
        structural_removal=analysis['structural_removal_observed'],
        independent_selected_full_catalog=(replay['status']=='passed' and
            replay['protocol']==protocol['id'] and replay['manifest_sha256']==analysis['input_manifest_sha256'] and
            len(replay['results'])==len(expected_replay) and observed_replay==expected_replay and
            all(r['status']=='passed' and r['proof']['proof']['full_catalog_decisions_certified'] and
                r['proof']['proof']['termination_certified'] for r in replay['results'])),
        all_primary_Q4_completed=True,primary_cell_cost_reduction=True,
        every_primary_paired_repeat_not_slower=True,no_H_specific_completion_failure=True,
        no_H_resource_failure=True)
    results=[]
    for key in sorted(required):
        cell=lookup.get(key)
        costs={} if cell is None else cell['costs']
        complete=cell is not None and all(costs[m]['complete_workers']==protocol['repeat_count'] and
            len(costs[m]['Q4_seconds'])==protocol['repeat_count'] and
            all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in costs[m]['Q4_seconds'])
            for m in ('D','S','H'))
        item=dict(graph=key[0],stream=key[1],seed_label=key[2],complete=complete,
                  cell_ratio=None,paired_ratios=None,reference_methods=None)
        if complete:
            h=costs['H']['Q4_seconds'];d=costs['D']['Q4_seconds'];s=costs['S']['Q4_seconds']
            ref=[min(a,b) for a,b in zip(d,s,strict=True)]
            ratios=[a/b for a,b in zip(h,ref,strict=True)]
            ratio=sum(h)/sum(ref)
            item.update(cell_ratio=ratio,paired_ratios=ratios,
                reference_methods=['D' if a<=b else 'S' for a,b in zip(d,s,strict=True)],
                H_seconds=h,D_seconds=d,S_seconds=s,
                cell_cost_gate=ratio<=rule['maximum_primary_cell_ratio'],
                repeat_gate=max(ratios)<=rule['maximum_primary_paired_ratio'])
            gates['primary_cell_cost_reduction'] &= item['cell_cost_gate']
            gates['every_primary_paired_repeat_not_slower'] &= item['repeat_gate']
        else:
            gates['all_primary_Q4_completed']=False
            gates['primary_cell_cost_reduction']=False
            gates['every_primary_paired_repeat_not_slower']=False
        results.append(item)
    outcomes={}
    for row in roster:
        key=tuple(row[k] for k in ('graph','stream','seed_label','repeat_id'))
        outcomes.setdefault(key,{})[row['method']]=row
    for methods in outcomes.values():
        h=methods.get('H',{})
        if h.get('resource_failure'):gates['no_H_resource_failure']=False
        if not h.get('complete',False) and any(methods.get(m,{}).get('complete',False) for m in ('D','S')):
            gates['no_H_specific_completion_failure']=False
    return dict(status='passed' if all(gates.values()) else 'failed',gates=gates,
        claim=standard['claim'],primary_results=results,
        planned_workers=planned,complete_workers=analysis['complete_workers'],
        full_roster_status_counts=analysis['status_counts'],
        failed_gates=[k for k,v in gates.items() if not v],
        scope='Only the frozen named scenarios are the time claim; every other registered cell and failure remains. '
              'This is a finite-workload empirical gate, not an originality, population, wall-time bound or instance-optimality claim.')


def evaluate_bound(protocol_bytes,analysis,replay):
    if analysis.get('protocol_sha256')!=hashlib.sha256(protocol_bytes).hexdigest():
        raise ValueError('analysis was not produced under these frozen protocol bytes')
    return evaluate(json.loads(protocol_bytes),analysis,replay)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','analysis','replay','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('do not replace a confirmation judgment')
    inputs={key:getattr(args,key).read_bytes() for key in ('protocol','analysis','replay')}
    result=evaluate_bound(inputs['protocol'],json.loads(inputs['analysis']),json.loads(inputs['replay']))
    result['input_sha256']={key:hashlib.sha256(value).hexdigest() for key,value in inputs.items()}
    result['evaluator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],gates=result['gates']),indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
