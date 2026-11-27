#!/usr/bin/env python3
"""Recompute complete-service comparisons from frozen raw JSON; no solver."""
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict, Counter
from math import fsum, isclose
import argparse, csv, json

ROOT = Path(__file__).resolve().parents[1]

def need(condition, detail):
    if not condition:
        raise ValueError(detail)

def load_workers(archive, prefix):
    with ZipFile(ROOT / 'artifacts/frozen' / archive) as z:
        for name in sorted(z.namelist()):
            if not (name.startswith(prefix) and name.endswith('.json')):
                continue
            worker = json.loads(z.read(name))
            if isinstance(worker.get('requests'), list) and 'prefixes' in worker and 'method' in worker:
                yield name, worker

def aggregate(archive, prefix, arms, cohort, expected_counts):
    rows, groups = [], defaultdict(dict)
    for name,w in load_workers(archive,prefix):
        method=w['method']; need(method in arms,name)
        key=(w['graph'],w['stream'],w.get('seed_label',''),w['repeat_id'])
        complete=w['status'] in ('completed','completed_budget')
        need(len(w['requests'])==4,name+'/Q4 count')
        recorded=w.get('complete_service_seconds',w['prefixes'].get('4') if complete else None)
        if complete:
            need(recorded is not None and recorded>0,name+'/complete service')
        else:
            need(recorded is None,name+'/partial must not have complete cost')
        row=dict(cohort=cohort,graph=key[0],stream=key[1],input_label=key[2],repeat=key[3],method=method,status=w['status'],complete_Q4=complete,complete_service_seconds=recorded,raw_member=name)
        need(method not in groups[key],name+'/duplicate')
        groups[key][method]=row; rows.append(row)
    need(all(set(g)==set(arms) for g in groups.values()),cohort+'/paired roster')
    complete_groups=[g for g in groups.values() if all(w['complete_Q4'] for w in g.values())]
    need((len(rows),len(complete_groups))==expected_counts,cohort+'/denominator')
    totals={m:fsum(g[m]['complete_service_seconds'] for g in complete_groups) for m in arms}
    anchor='H' if 'G' not in arms else 'G'
    ratios={f'{m}/{anchor}':totals[m]/totals[anchor] for m in arms if m!=anchor}
    wins={f'{anchor}_faster_than_{m}':sum(g[anchor]['complete_service_seconds']<g[m]['complete_service_seconds'] for g in complete_groups) for m in arms if m!=anchor}
    return rows,dict(workers=len(rows),complete_workers=sum(w['complete_Q4'] for w in rows),matched_complete_blocks=len(complete_groups),total_seconds=totals,ratios_of_sums=ratios,complete_pair_wins=wins,status_counts=dict(Counter(w['status'] for w in rows)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    out=a.output_dir.resolve();need(not out.exists(),'Choose a new output directory.');out.mkdir(parents=True)
    configs=[('V2_Evidence.zip','raw/a2_tie_retest/',('D','S','H'),'fixed_input_revision_retest',(108,36)),('V2_Evidence.zip','v2/raw_a/',('D','S','H'),'post_freeze_confirmation',(54,15)),('V4_Evidence.zip','v4/raw_hg_light/',('H','G'),'observed_input_light_v1',(36,15))]
    records=[]; results={}
    for archive,prefix,arms,label,counts in configs:
        rows,summary=aggregate(archive,prefix,arms,label,counts);records.extend(rows);results[label]=summary
    expected={'fixed_input_revision_retest':{'D':435.63589438901545,'S':1732.4551205220014,'H':121.47893757598649},'post_freeze_confirmation':{'D':211.019847705,'S':787.462590087,'H':55.624119350},'observed_input_light_v1':{'H':21.537745645999166,'G':18.833572476007248}}
    for cohort, values in expected.items():
        for method, value in values.items():need(isclose(results[cohort]['total_seconds'][method],value,rel_tol=1e-8,abs_tol=1e-8),(cohort,method))
    r=results['observed_input_light_v1'];r['G_percent_less_time']=100*(1-r['total_seconds']['G']/r['total_seconds']['H'])
    with (out/'complete_service_workers.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    report={'status':'passed','scope':'Raw stored timing aggregation only. Partial streams remain in the roster and have no complete-service time. No new timings or independent numerical replay.','cohorts':results}
    (out/'headline_results.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
