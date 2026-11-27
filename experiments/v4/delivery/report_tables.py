"""Independent arithmetic for delivered tables from retained worker JSON only."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
    w=a.workspace.resolve();root=w/'v4';summary=json.loads((root/'reports/hg_light_summary.json').read_text())
    freeze=json.loads((root/'protocols/HG_LIGHT_FROZEN.json').read_text())
    raw={};records={}
    for j in freeze['jobs']:
        file=w/freeze['raw_dir']/j['output_file'];key=(j['cell_index'],j['repeat_id'],j['method'])
        raw[key]=json.loads(file.read_text());records[key]=json.loads((root/'runs'/('hg_light_'+file.stem)/'record.json').read_text())
    totals={m:sum(row['complete_service_seconds'] for (c,r,method),row in raw.items() if method==m and row['complete_Q4']) for m in ('H','G')}
    assert totals['H']==summary['ratio_summary']['H_total_seconds']
    assert totals['G']==summary['ratio_summary']['G_total_seconds']
    assert totals['H']/totals['G']==summary['ratio_summary']['H_over_G_ratio_of_sums']
    for cell in summary['cells']:
        i=cell['cell_index'];ratios=[]
        for repeat in range(3):
            h,g=raw[i,repeat,'H'],raw[i,repeat,'G']
            if h['complete_Q4'] and g['complete_Q4']:ratios.append(h['complete_service_seconds']/g['complete_service_seconds'])
        assert cell['ratio_median']==(statistics.median(ratios) if ratios else None)
    memory={}
    for method in ('H','G'):
        values=[(row,records[key]) for key,row in raw.items() if key[2]==method]
        memory[method]={
            'sampled_RSS_bytes_min':min(r['sampled_peak_rss_bytes'] for row,r in values),
            'sampled_RSS_bytes_max':max(r['sampled_peak_rss_bytes'] for row,r in values),
            'sampled_VMS_bytes_min':min(r['sampled_peak_vms_bytes'] for row,r in values),
            'sampled_VMS_bytes_max':max(r['sampled_peak_vms_bytes'] for row,r in values),
            'checkpoint_ru_maxrss_bytes_min':min(row['process_peak_rss_bytes'] for row,r in values),
            'checkpoint_ru_maxrss_bytes_max':max(row['process_peak_rss_bytes'] for row,r in values)}
    complete=summary['work_aggregates']['complete_pairs_only']
    phase_rows=[]
    for scope in ('structural_exclusive','request_exclusive'):
        for phase in complete['H']['phase_totals'][scope]:
            h=complete['H']['phase_totals'][scope][phase];g=complete['G']['phase_totals'][scope][phase]
            if h or g:phase_rows.append(dict(scope=scope,phase=phase,H_seconds=h,G_seconds=g,H_minus_G_seconds=h-g))
    facts=dict(status='passed',independent_raw_aggregate_check=True,
        H_total_seconds=totals['H'],G_total_seconds=totals['G'],H_over_G=totals['H']/totals['G'],
        G_fractional_cost_reduction=1-totals['G']/totals['H'],
        complete_pair_phase_rows=phase_rows,memory_all_formal_workers=memory,
        numerical_replays_performed=0,scope='read-only table arithmetic on all frozen recorded workers; no service, solve or checker numerical replay')
    target=root/'reports/delivery_table_facts.json'
    with target.open('x') as f:json.dump(facts,f,indent=2);f.write('\n')
    print(json.dumps(facts))


if __name__=='__main__':main()
