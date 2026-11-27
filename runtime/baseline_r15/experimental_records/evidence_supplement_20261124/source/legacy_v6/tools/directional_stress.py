"""Run matched completed-square policies against the stored normwise trajectories."""
import argparse
import csv
from flip_stress import run, ROOT, TOLS, WEIGHTS

def read(name):
    with (ROOT/'data'/f'{name}.csv').open() as handle:
        return list(csv.DictReader(handle))

def write(name,rows):
    with (ROOT/'data'/f'{name}.csv').open('w',newline='') as handle:
        fields = dict.fromkeys(k for row in rows for k in row)
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def experiment(cohort,w):
    prefix = 'flip_stress_'+('alpha_' if cohort=='alpha' else '')
    anchors = [r for r in read(prefix+'steps') if int(r['w'])==w]
    counts,steps,events = [],[],[]
    for tol in TOLS if cohort=='bio' else (1e-7,):
        old_counts,old_steps,old_events = run(w,tol,cohort)
        c,s,e = run(w,tol,cohort,directional=True)
        assert len(old_events)==len(e)
        assert [(r['epoch'],r['winner']) for r in old_steps]==[(r['epoch'],r['winner']) for r in s]
        for before,after in zip(old_counts,c):
            assert (before['b'],before['accepted'])==(after['b'],after['accepted'])
            for key in ('R','state_refines','direct_fallbacks'):
                after['baseline_'+key]=before[key]
        assert [int(r['winner']) for r in anchors if float(r['rtol'])==tol] == [r['winner'] for r in s]
        for before,row in zip(old_events,e):
            assert (before['epoch'],before['edge'])==(row['epoch'],row['edge'])
            row['baseline_radius'] = before['radius']
            row['baseline_gram'] = before['posterior']
        counts.extend(c)
        steps.extend(s)
        events.extend(e)
    for name,rows in [('grid',counts),('steps',steps),('events',events)]:
        write(f'directional_stress_{cohort}_{w}_{name}',rows)

def figure():
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'font.size':8})
    fig, axes = plt.subplots(1,3,figsize=(7.05,2.05),layout='constrained')
    for co,marker in [('bio','o-'),('alpha','s--')]:
        ratios = []
        for w in WEIGHTS:
            events = read(f'directional_stress_{co}_{w}_events')
            ratios.append(np.median([float(r['baseline_radius'])/float(r['radius']) for r in events]))
        axes[0].semilogy(range(4),ratios,marker,ms=3,label=co.capitalize())
    axes[0].set(xticks=range(4),xticklabels=WEIGHTS,xlabel='Flip weight',ylabel='Median paired radius ratio')
    axes[0].legend(fontsize=7)
    def pooled(co):
        return [r for w in WEIGHTS for r in read(f'directional_stress_{co}_{w}_events')]
    bio = [r for r in pooled('bio') if int(r['age'])>0]
    zero = [r for r in bio if float(r['coupling'])==0]
    pos = sorted((r for r in bio if float(r['coupling'])>0),key=lambda r:float(r['coupling']))
    bins = [zero,*np.array_split(pos,4)]
    for key,style in [('error','o-'),('posterior','s--'),('uniform','^:'),('prior','d-.')]:
        axes[1].semilogy(range(5),[np.median([float(r[key]) for r in group if r[key]!='']) for group in bins],style,ms=3,label=key)
    axes[1].set(xticks=range(5),xticklabels=['0','Q1','Q2','Q3','Q4'],xlabel='Bio directional coupling bin',ylabel='Median Gram magnitude')
    axes[1].legend(fontsize=6,ncol=2)
    alpha = pooled('alpha')
    aged = sorted((r for r in alpha if int(r['age'])>0),key=lambda r:float(r['coupling']))
    groups = list(np.array_split(aged,5))
    groups += [[r for r in alpha if lo<=int(r['age'])<=hi] for lo,hi in [(0,0),(1,4),(5,14),(15,19)]]
    groups += [[r for r in alpha if int(r['w'])==w] for w in WEIGHTS]
    for key,style,label in [('baseline_radius','o-','Normwise'),('radius','s--','Directional')]:
        axes[2].plot(range(13),[100*np.mean([float(r[key])>=float(r['margin']) for r in group]) for group in groups],style,ms=2,label=label)
    axes[2].set(xticks=range(13),xticklabels=['1','2','3','4','5','0','1–4','5–14','15–19','1','4','16','64'],ylabel=r'Alpha events with $r_g/m_t\geq1$ (%)',xlabel='Coupling quintile / age / weight')
    axes[2].tick_params(axis='x',labelrotation=90,labelsize=6)
    for x in (4.5,8.5):
        axes[2].axvline(x,color='gray',linewidth=.5)
    axes[2].legend(fontsize=6)
    print('Bio bins',list(map(len,bins)),'Alpha denominators',list(map(len,groups)),
          'Alpha w64 large',sum(float(r['radius'])>=float(r['margin']) for r in groups[-1]))
    fig.savefig(ROOT/'figures'/'directional_stress.pdf')
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('cohort',choices=['bio','alpha','figure'])
    parser.add_argument('weight',nargs='?',type=int,choices=WEIGHTS)
    args=parser.parse_args()
    if args.cohort=='figure':
        figure()
    elif args.weight is None:
        parser.error('weight is required for experiments')
    else:
        experiment(args.cohort,args.weight)
