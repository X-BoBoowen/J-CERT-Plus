"""Plot restored historical same-context work, using deterministic counts only."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullLocator


CASES = (('Alpha', 2), ('OTC', 9))
EPOCHS = tuple(range(1, 16))
COUNT_FIELDS = ('target', 'tau', 'max_drift', 'raw_visits',
                'posting_raw_visits', 'tail_raw_visits')
COLORS = dict(target='#6855A4', drift='#374151', tau='#A6476F',
              postings='#4173A8', tails='#E3AD4F', fresh='#176B57',
              fallback='#A23E29')


def count_series(data, graph, position):
    matches = [case for case in data['cases']
               if (case['graph'], case['stream'], case['position'])
               == (graph, 'independent', position)]
    if len(matches) != 1 or matches[0]['restoration_verified'] is not True:
        raise ValueError(f'Expected one verified restored case for {graph} q{position}')
    case = matches[0]
    rows = {}
    for sample in case['stage_samples']:
        key = (sample['source'], sample['epoch'], sample['repeat'])
        if key in rows:
            raise ValueError(f'Duplicate stage sample for {graph} q{position}: {key}')
        counts = {field: sample[field] for field in COUNT_FIELDS}
        if counts['raw_visits'] != counts['posting_raw_visits'] + counts['tail_raw_visits']:
            raise ValueError(f'Raw visits do not reconcile for {graph} q{position}: {key}')
        rows[key] = counts
    expected = {(source, epoch, repeat) for source in ('old', 'fresh')
                for epoch in EPOCHS for repeat in range(data['repeats'])}
    if rows.keys() != expected or data['repeats'] < 1:
        raise ValueError(f'Incomplete stage samples for {graph} q{position}')
    result = {}
    for source in ('old', 'fresh'):
        for epoch in EPOCHS:
            counts = rows[source, epoch, 0]
            if any(rows[source, epoch, r] != counts for r in range(data['repeats'])):
                raise ValueError(f'Deterministic work counts differ for {graph} q{position}')
        result[source] = [rows[source, epoch, 0] for epoch in EPOCHS]
    if any(old['target'] != fresh['target']
           for old, fresh in zip(result['old'], result['fresh'], strict=True)):
        raise ValueError(f'Old/fresh target differs for {graph} q{position}')
    return result


def guard_markers(regression_path, diagnostic):
    """Read observed event epochs without inferring a guarded work curve."""
    historical = {(c['graph'], c['position']): c for c in diagnostic['cases']
                  if c['stream'] == 'independent'}
    regression = json.loads(regression_path.read_text())
    markers = {}
    for graph, position in CASES:
        matches = [c for c in regression['cases']
                   if (c['graph'], c['stream'], c['q']) == (graph, 'independent', position)]
        if len(matches) != 1:
            raise ValueError(f'Expected one observed guard regression for {graph} q{position}')
        case = matches[0]
        original = historical[graph, position]
        if (case['input_sha256'] != original['frozen_trace_sha256'] or
                case['origin'] != original['origin'] or
                case['same_winners_and_versions'] is not True):
            raise ValueError(f'Unmatched guard regression for {graph} q{position}')
        discovery = case['request']['discovery']
        events = [e for e in discovery['epochs'] if e['action'] == 'fallback']
        if len(events) != 1 or discovery['fallback_count'] != 1:
            raise ValueError(f'Expected a single fallback for {graph} q{position}')
        event = events[0]
        if (event['epoch'] not in EPOCHS or event['anchor_request'] != position or
                event['budget'] != case['visit_budget'] or
                event['consumed'] != discovery['old_consumed'] or
                not 0 <= event['consumed'] <= event['budget']):
            raise ValueError(f'Invalid fallback observation for {graph} q{position}')
        markers[graph, position] = dict(epoch=event['epoch'], consumed=event['consumed'],
                                        budget=event['budget'], coefficient=Fraction(str(case['coefficient'])))
    if len({m['coefficient'] for m in markers.values()}) != 1:
        raise ValueError('Guard regression markers use different budget coefficients')
    return markers


def plot(input_path, output_stem, regression_path=None):
    data = json.loads(input_path.read_text())
    markers = guard_markers(regression_path, data) if regression_path else {}
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.edgecolor': '#C9D0D8',
        'axes.labelcolor': '#263342',
        'text.color': '#1D2937',
        'xtick.color': '#4B5563',
        'ytick.color': '#4B5563',
        'svg.fonttype': 'none',
    })
    fig, axs = plt.subplots(3, 2, figsize=(11.1, 7.6), sharex='col', sharey='row',
                            gridspec_kw={'height_ratios': [1, 1.18, 1.69]})
    fig.subplots_adjust(left=.117, right=.973, top=.84, bottom=.143, hspace=.29, wspace=.22)
    fig.suptitle('As targets fall, retained-index work shifts to postings', y=.966,
                 fontsize=17, weight='semibold')
    subtitle = 'Two frozen independent-stream requests · same-context index probes'
    if markers:
        coefficient = next(iter(markers.values()))['coefficient']
        subtitle = f'Historical work curves · event markers from the observed {coefficient}-budget guard replay'
    fig.text(.5, .916, subtitle,
             ha='center', color='#526172', fontsize=11)
    for col, (graph, position) in enumerate(CASES):
        series = count_series(data, graph, position)
        old, fresh = series['old'], series['fresh']
        target_ax, drift_ax, work_ax = axs[:, col]

        target_ax.set_title(f'{graph} | request {position}', loc='left', pad=9,
                            fontsize=12, weight='bold')
        target = [100 * row['target'] / old[0]['target'] for row in old]
        target_ax.plot(EPOCHS, target, color=COLORS['target'], linewidth=2,
                       marker='o', markersize=3.2)
        target_ax.set_ylim(0, 112)
        target_ax.set_yticks([0, 50, 100])
        target_ax.annotate(f'{target[-1]:.1f}%', (15, target[-1]),
                           xytext=(-6, 9), textcoords='offset points',
                           ha='right', color=COLORS['target'], weight='semibold')

        drift_ax.set_yscale('log', base=2)
        drift_ax.set_ylim(2**11.7, 2**17.6)
        drift_ax.set_yticks([2**12, 2**14, 2**16, 2**17])
        drift_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
        drift_ax.yaxis.set_minor_locator(NullLocator())
        drifts = [row['max_drift'] for row in old]
        taus = [row['tau'] for row in old]
        drift_ax.plot(EPOCHS, drifts, color=COLORS['drift'], linewidth=1.9, linestyle='--')
        drift_ax.step(EPOCHS, taus, where='mid', color=COLORS['tau'], linewidth=2.1)
        drift_ax.scatter(EPOCHS, taus, c=COLORS['tau'], s=13, zorder=3)

        postings = [row['posting_raw_visits'] for row in old]
        tails = [row['tail_raw_visits'] for row in old]
        fresh_raw = [row['raw_visits'] for row in fresh]
        work_ax.bar(EPOCHS, postings, color=COLORS['postings'], width=.74, zorder=2)
        work_ax.bar(EPOCHS, tails, bottom=postings, color=COLORS['tails'], width=.74, zorder=3)
        work_ax.plot(EPOCHS, fresh_raw, color=COLORS['fresh'], marker='o',
                     linewidth=1.65, markersize=3.5, linestyle='--', zorder=4)
        work_ax.set_ylim(0, 720_000)
        work_ax.set_yticks([0, 200_000, 400_000, 600_000])
        work_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x // 1000)}k' if x else '0'))
        pct = 100 * postings[-1] / old[-1]['raw_visits']
        work_ax.annotate(f'{pct:.1f}% postings', (15, old[-1]['raw_visits']),
                         xytext=(-1, 9), textcoords='offset points', ha='right',
                         color='#244A70', fontsize=9, weight='semibold')
        if markers:
            event = markers[graph, position]
            work_ax.axvline(event['epoch'], color=COLORS['fallback'],
                            linestyle=':', linewidth=1.7, zorder=5)
            label = (f"Observed fallback at epoch {event['epoch']}\n"
                     f"{event['consumed']:,} / {event['budget']:,} old visits admitted")
            work_ax.text(event['epoch'] + .25, .955, label,
                         transform=work_ax.get_xaxis_transform(), va='top',
                         fontsize=8.5, color=COLORS['fallback'], zorder=6,
                         bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2.2})
        work_ax.set_xlabel('Cold-query epoch', labelpad=7)
        work_ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
        for ax in axs[:, col]:
            ax.set_xlim(.5, 15.6)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(axis='both', length=0, pad=5)
            ax.grid(axis='y', color='#E5E9EF', linewidth=.8)
            ax.set_axisbelow(True)

    axs[0, 0].set_ylabel('Target (% of epoch 1)')
    axs[1, 0].set_ylabel('Drift / cutoff (quanta)')
    axs[2, 0].set_ylabel('Raw visits before dedup')
    legend = [
        Line2D([], [], color=COLORS['drift'], linestyle='--', label='Max retained drift'),
        Line2D([], [], color=COLORS['tau'], marker='o', markersize=4, label='Selected cutoff τ'),
        Patch(facecolor=COLORS['postings'], label='Retained postings'),
        Patch(facecolor=COLORS['tails'], label='Retained group tails'),
        Line2D([], [], color=COLORS['fresh'], marker='o', markersize=4,
               linestyle='--', label='Fresh raw visits'),
    ]
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .035),
               ncol=5, frameon=False, fontsize=9, columnspacing=1.35, handlelength=2)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix('.png'), dpi=200, facecolor='white',
                metadata={'Title': 'Restored old-index mechanism'})
    fig.savefig(output_stem.with_suffix('.svg'), facecolor='white',
                metadata={'Title': 'Restored old-index mechanism', 'Date': None})
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='raw/access_diagnostic.json', type=Path)
    parser.add_argument('--guard-regression', type=Path)
    parser.add_argument('--output-stem', type=Path)
    args = parser.parse_args()
    default_name = 'MECHANISM_GUARD_OVERLAY' if args.guard_regression else 'MECHANISM_OLD_EVIDENCE'
    plot(args.input, args.output_stem or Path('analysis') / default_name, args.guard_regression)
