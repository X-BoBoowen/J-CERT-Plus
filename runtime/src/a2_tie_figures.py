"""Three publication figures for the diagnosed fixed-input revision retest.

Read recorded analysis/summary only; never import or run a numerical service.
Render after timing finishes, e.g. taskset -c 1 python -m src.a2_tie_figures.
The exact plotted observations, aggregations and source hashes are exported.
"""
import argparse
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

from .a2_figures import (COLORS, FAMILIES, LABELS, MARKERS, METHODS,
                         phase_buckets, save_figure, setup_matplotlib, summary)


ROOT = Path(__file__).resolve().parents[1]
GRAPHS = ('Alpha', 'OTC')
STREAMS = ('slow', 'independent', 'signed-neighbor')
SEEDS = ('confirmation-1', 'confirmation-2')
STUDY = 'Fixed-input revision retest | Generated opinions'
PHASES = ('Root setup', 'Candidate access + screening',
          'Numerical response + certification', 'Reset + other')
PHASE_COLORS = ('#CCD7E2', '#537C9D', '#D7A258', '#C4C0BC')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return tuple(row[name] for name in ('graph', 'stream', 'seed_label'))


def _close(first, second, label):
    if not math.isclose(first, second, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError(label + ' does not reconcile')


def derive(analysis, report):
    """Validate the complete roster before any descriptive aggregation."""
    for source in (analysis, report):
        if (source.get('stage') != 'revision_retest' or
                source.get('unseen_confirmation') is not False or
                source.get('original_confirmation_status') != 'failed'):
            raise ValueError('requires the explicitly labeled revision retest')
    if (analysis.get('planned_workers') != 108 or analysis.get('complete_workers') != 108 or
            not analysis.get('correctness_accounting_pass') or analysis.get('validation_issues') or
            report.get('complete_workers') != 108 or not report.get('all_inputs_complete') or
            report.get('provenance_and_roster_issues') or report.get('raw_read_issues')):
        raise ValueError('complete-cost figure requires all 108 valid workers; no partial subset is substituted')
    expected = list(product(GRAPHS, STREAMS, SEEDS))
    acells = {key(row): row for row in analysis['cells']}
    rcells = {key(row): row for row in report['cells']}
    if len(analysis['cells']) != 12 or len(report['cells']) != 12 or set(acells) != set(expected) or set(rcells) != set(expected):
        raise ValueError('all twelve fixed input cells must remain in the figures')
    workers = {tuple(row[name] for name in ('graph', 'stream', 'seed_label', 'method', 'repeat_id')): row
               for row in analysis['workers']}
    if len(workers) != 108:
        raise ValueError('missing or duplicated worker identity')
    cells, observations = [], []
    for cell_key in expected:
        cell, reported = acells[cell_key], rcells[cell_key]
        plotted = {name: cell[name] for name in ('graph', 'stream', 'seed_label')}
        plotted['methods'] = {}
        for method in METHODS:
            cost = cell['costs'][method]
            recorded = reported['methods'][method]
            values = cost['Q4_seconds']
            if (cost['planned_workers'] != 3 or cost['complete_workers'] != 3 or
                    cost['statuses'] != ['completed'] * 3 or len(values) != 3 or
                    any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values) or
                    recorded['seconds_by_repeat'] != values or recorded['complete_repeats'] != 3):
                raise ValueError('missing or inconsistent three-repeat cell costs')
            plotted['methods'][method] = summary(values)
            metrics = {row['repeat_id']: row for row in cell['work_and_memory'][method]}
            if set(metrics) != {0, 1, 2}:
                raise ValueError('phase/RSS records lost a registered repeat')
            for repeat, value in enumerate(values):
                worker = workers[(*cell_key, method, repeat)]
                metric = metrics[repeat]
                if not worker['complete'] or not worker['valid'] or not metric['complete']:
                    raise ValueError('incomplete observation cannot become a complete plotted cost')
                ledger = phase_buckets(metric, value)
                exclusive = ledger['exclusive_seconds']
                grouped = {
                    'Root setup': exclusive['Root setup'],
                    'Candidate access + screening': exclusive['Candidate access'] + exclusive['Strict screening'],
                    'Numerical response + certification': exclusive['Responses and Gram scores'] + exclusive['Solve, certify and commit'],
                    'Reset + other': exclusive['Reset and other'],
                }
                _close(sum(grouped.values()), value, 'exclusive phase sum')
                rss = metric['process_peak_rss_bytes']
                if type(rss) is not int or rss <= 0:
                    raise ValueError('missing process high-water RSS')
                observations.append(dict(
                    **{name: cell[name] for name in ('graph', 'stream', 'seed_label')},
                    method=method, repeat_id=repeat, schedule_position=worker['position'],
                    Q4_seconds=value, grouped_phases_seconds=grouped, phase_ledger=ledger,
                    reported_process_peak_rss_bytes=rss,
                    externally_sampled_peak_rss_bytes=metric['external_sampled_peak_rss_bytes'],
                    numeric_counters=metric['counters'], access_counters=metric['access'],
                    memory_sample_maxima=metric['memory_sample_maxima'],
                    reached_versions=[{name: event.get(name) for name in (
                        'query_position', 'epoch', 'parent_version_id', 'frontier_rebuilds',
                        'frontier_build_phases', 'seed_build_phases', 'current_legal_candidates')}
                        for event in metric['versions']]))
        pairs = []
        for repeat in range(3):
            d, s, h = (plotted['methods'][method]['values'][repeat] for method in METHODS)
            ref_value, ref_method = min((d, 'D'), (s, 'S'))
            pairs.append(dict(repeat_id=repeat, D_seconds=d, S_seconds=s, H_seconds=h,
                              reference_method=ref_method, reference_seconds=ref_value,
                              ratio_H_over_better=h / ref_value))
        ratio = sum(row['H_seconds'] for row in pairs) / sum(row['reference_seconds'] for row in pairs)
        _close(ratio, reported['comparisons']['H_over_better']['ratio_H_over_reference'], 'paired ratio of sums')
        plotted['paired_H_over_better'] = dict(
            pairs=pairs, ratio_of_sums=ratio,
            individual_ratios=summary(row['ratio_H_over_better'] for row in pairs))
        cells.append(plotted)
    graph_methods = []
    for graph, method in product(GRAPHS, METHODS):
        selected = [row for row in observations if row['graph'] == graph and row['method'] == method]
        if len(selected) != 18 or len({key(row) for row in selected}) != 6:
            raise ValueError('graph summary must include all six cells and 18 workers per method')
        means = {phase: statistics.mean(row['grouped_phases_seconds'][phase] for row in selected) for phase in PHASES}
        mean_cost = statistics.mean(row['Q4_seconds'] for row in selected)
        _close(sum(means.values()), mean_cost, 'graph mean phase sum')
        rss_summary = summary(row['reported_process_peak_rss_bytes'] / 2**20 for row in selected)
        reported_graph = next(row for row in report['graphs'] if row['name'] == graph)
        _close(rss_summary['median'], reported_graph['methods'][method]['process_peak_rss_bytes']['median'] / 2**20,
               'graph process RSS median')
        graph_methods.append(dict(graph=graph, method=method, planned_cells=6, workers=18,
                                  mean_Q4_seconds=mean_cost, mean_phases_seconds=means,
                                  process_high_water_rss_MiB=rss_summary))
    for method in METHODS:
        total = sum(row['Q4_seconds'] for row in observations if row['method'] == method)
        _close(total, analysis['full_schedule_totals_seconds'][method], 'analysis full service total')
        _close(total, report['overall']['full_scope_service_totals_seconds'][method], 'summary full service total')
    return dict(schema_version=1, stage='revision_retest', study_label=analysis['study_label'],
                unseen_confirmation=False, original_confirmation_status='failed',
                planned_input_cells=12, retained_workers=len(observations),
                complete_workers=108, cells=cells, graph_methods=graph_methods, observations=observations,
                overall_service_seconds=analysis['full_schedule_totals_seconds'],
                definitions={
                    'cost': 'Full graph load/setup plus all four completed request walls; every reached-version frontier cost is charged.',
                    'cost_centers': 'Median of the three registered repeats for each input cell and method.',
                    'cost_whiskers': 'Exact minimum and maximum of the same three repeats, not confidence intervals.',
                    'paired_reference': 'For each repeat choose the smaller complete D/S whole-block service time; never splice requests.',
                    'paired_diamonds': 'Sum of H costs / sum of paired better-reference costs across the three repeats.',
                    'paired_dots_and_whiskers': 'All three individual H/better ratios and their exact minimum/maximum.',
                    'phase_population': 'All 18 workers per graph and method: three families, both original seed labels, three repeats.',
                    'phase_statistic': 'Arithmetic mean of exclusive service phase sums; four compatible second-valued groups stack to mean Q4 cost.',
                    'phase_mapping': dict(zip(PHASES, [
                        ['Root setup'], ['Candidate access', 'Strict screening'],
                        ['Responses and Gram scores', 'Solve, certify and commit'], ['Reset and other']], strict=True)),
                    'frontier_cost': 'Included in Candidate access. Nested per-version frontier/index timings are retained for audit, never added twice.',
                    'rss_population': 'Same 18 workers per graph and method; no selected-family examples and no sum of worker peaks.',
                    'rss': 'Reported process-lifetime high-water RSS / 2^20. Externally sampled peaks are separately retained, not added.',
                    'rss_centers_and_ranges': 'Median and exact minimum/maximum across the 18 worker peaks; descriptive workload/repeat spread, not confidence intervals.',
                    'online_and_offline': 'Online equality derivation/checks, bound transfer and witness construction are inside measured service. Independent checker time is outside algorithm time.',
                    'output_accounting': 'Checkpoint serialization/I/O remains in process wall, not setup+Q4 service; in-request witness recording remains in service.',
                    'work': 'Insertion emissions, heap/source pops, Gram counts, response blocks and scalar RHS stay separate in exact observations.',
                })


def _box(axis, x, y, width, height, title, lines, face='#F0F3F6', edge='#63788B'):
    from matplotlib.patches import FancyBboxPatch
    axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.009,rounding_size=0.012',
                                 facecolor=face, edgecolor=edge, linewidth=1.15))
    axis.text(x + width / 2, y + height * .76, title, ha='center', va='center',
              fontsize=10.5, weight='semibold')
    axis.text(x + width / 2, y + height * .36, lines, ha='center', va='center',
              fontsize=9.4, linespacing=1.45)


def _arrow(axis, start, end, **kwargs):
    from matplotlib.patches import FancyArrowPatch
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=13,
                                   linewidth=1.35, color='#63788B', shrinkA=3, shrinkB=4, **kwargs))


def structure_figure(plt):
    fig = plt.figure(figsize=(13.4, 6.5))
    ax = fig.add_axes((.012, .18, .975, .64))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    _box(ax, .025, .59, .19, .23, 'Current parent', 'Certified state + exact opinion\nChosen IDs + version binding')
    _box(ax, .28, .72, .255, .22, 'Implicit insertion frontier', 'All root nonedges; weight 1/16\nSorted rows; one heap head per row', '#E2F0EE', '#3C8E89')
    _box(ax, .28, .405, .255, .22, 'Explicit sparse flip frontier', 'Root negative edges\nNatural absolute edge weights', '#F5ECD9', '#A7853E')
    _box(ax, .60, .59, .165, .23, 'Joint unseen bound', 'Maximum of both heads\nCovers ungenerated rivals')
    _box(ax, .815, .59, .165, .23, 'Shared strict / Gram', 'Strict endpoint bounds\nOn-demand local responses\nCertified gain intervals', '#E8EDF5')
    _box(ax, .59, .075, .39, .30, 'Common interval + exact equality decision',
         'Positive lower; all rivals accounted for; canonical ID\n'
         r'$P A P^T=A,\quad P s=s,\quad P A_e P^T=A_f$' '\n'
         'No positive gain: stop. Unproved overlap: unresolved.', '#E8EDF5', '#657A95')
    _box(ax, .29, .115, .23, .20, 'Commit selected winner', 'Own response / update binding\nRebuild H if another decision')
    _box(ax, .025, .115, .19, .20, 'Next opinion', 'Reset to the fixed root\nNew request evidence')
    for start, end in [((.215, .735), (.28, .83)), ((.215, .665), (.28, .515)),
                       ((.535, .83), (.60, .735)), ((.535, .515), (.60, .665)),
                       ((.765, .705), (.815, .705)), ((.897, .59), (.84, .375)),
                       ((.59, .215), (.52, .215)), ((.12, .315), (.12, .59))]:
        _arrow(ax, start, end)
    ax.text(.792, .755, 'emit', ha='center', fontsize=8.3)
    ax.text(.553, .263, 'accept', ha='center', fontsize=8.6)
    ax.text(.105, .455, 'new\nrequest', ha='right', va='center', fontsize=8.8)
    # The accepted-edit loop returns to the current parent, not the request root.
    ax.plot([.405, .405, .245, .245], [.315, .355, .355, .54], color='#63788B', linewidth=1.35)
    _arrow(ax, (.245, .54), (.20, .59))
    ax.text(.251, .46, 'next parent', ha='left', va='center', fontsize=8.7, rotation=90)
    fig.suptitle('Full-catalog coverage without complete candidate materialization', y=.975,
                 fontsize=16, weight='semibold')
    fig.text(.5, .91, STUDY + ' | H access path; D/S share the common decision kernel',
             ha='center', fontsize=10.4)
    fig.text(.025, .105, 'Classical sorted-frontier / threshold access; the system integration binds coverage, local Grams and current numerical evidence.', fontsize=9.5)
    fig.text(.025, .065, 'Root legality and chosen-edit exclusion apply throughout. Exact equality is algebraic; interval overlap alone is insufficient.', fontsize=9.5)
    fig.text(.025, .025, 'Online witness work is charged inside service. Independent full-catalog checking is a separate validation computation.', fontsize=9.5)
    return fig


def _style_axis(axis, *, grid='y'):
    axis.grid(axis=grid, color='#DFE5EB', linewidth=.7)
    axis.set_axisbelow(True)
    axis.tick_params(length=3, width=.7)


def main_cost_figure(plt, values):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    fig, (cost_ax, ratio_ax) = plt.subplots(2, 1, sharex=True, figsize=(13.4, 8.15),
                                           gridspec_kw={'height_ratios': (1.8, 1.)})
    fig.subplots_adjust(left=.082, right=.985, bottom=.205, top=.79, hspace=.18)
    cells = values['cells']
    all_costs = [x for c in cells for m in METHODS for x in c['methods'][m]['values']]
    cost_ax.set_yscale('log')
    cost_ax.set_ylim(min(all_costs) * .72, max(all_costs) * 1.32)
    cost_ax.set_yticks([2, 5, 10, 20, 50, 100])
    cost_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ratio_ax.set_ylim(0, 1.1)
    ratio_ax.set_yticks([0, .25, .5, .75, 1])
    ratio_ax.axhline(1, linestyle=(0, (4, 3)), color='#77818B', linewidth=1.15)
    ratio_ax.text(11.4, 1.025, 'equal cost', ha='right', fontsize=8.8, color='#596875')
    for ax in (cost_ax, ratio_ax):
        ax.set_xlim(-.55, 11.55)
        _style_axis(ax)
        for start in (0, 4, 8):
            ax.axvspan(start - .5, start + 1.5, color='#F2F5F7', zorder=0)
        ax.axvline(5.5, color='#AAB6C1', linewidth=1.1)
        for boundary in (1.5, 3.5, 7.5, 9.5):
            ax.axvline(boundary, color='#E4E9EE', linewidth=.7)
    for i, cell in enumerate(cells):
        for j, method in enumerate(METHODS):
            stat = cell['methods'][method]
            cost_ax.errorbar(i + (j - 1) * .22, stat['median'],
                            yerr=[[stat['median'] - stat['minimum']], [stat['maximum'] - stat['median']]],
                            fmt=MARKERS[method], color=COLORS[method], markersize=6.2,
                            capsize=3.4, elinewidth=1.4, markeredgewidth=1.1, zorder=3)
        pairs = cell['paired_H_over_better']
        stat = pairs['individual_ratios']
        center = pairs['ratio_of_sums']
        ratio_ax.errorbar(i, center, yerr=[[center - stat['minimum']], [stat['maximum'] - center]],
                         fmt='D', markersize=5.5, color=COLORS['H'], elinewidth=1.3,
                         capsize=3.3, zorder=4)
        ratio_ax.scatter([i - .075, i, i + .075], stat['values'], color=COLORS['H'],
                         s=13, alpha=.5, edgecolors='none', zorder=3)
    cost_ax.set_ylabel('Full setup + Q4 service\n(seconds; log scale)', fontsize=11)
    ratio_ax.set_ylabel('H / faster D–S\n(paired cost ratio)', fontsize=11)
    ratio_ax.set_xticks(range(12), ['C1' if c['seed_label'] == SEEDS[0] else 'C2' for c in cells], fontsize=9.4)
    for center in (.5, 2.5, 4.5, 6.5, 8.5, 10.5):
        ratio_ax.text(center, -.22, FAMILIES[cells[int(center)]['stream']],
                      transform=ratio_ax.get_xaxis_transform(), ha='center', va='top', fontsize=9.5)
    for x, graph in ((2.5, 'Alpha'), (8.5, 'OTC')):
        cost_ax.text(x, 1.04, graph, transform=cost_ax.get_xaxis_transform(), ha='center',
                     fontsize=12, weight='semibold')
    fig.legend([Line2D([], [], linestyle='none', marker=MARKERS[m], color=COLORS[m], markersize=7)
                for m in METHODS], [LABELS[m] for m in METHODS], loc='upper center',
               bbox_to_anchor=(.5, .89), frameon=False, ncol=3, fontsize=10.7)
    fig.suptitle('Complete service cost on all twelve fixed input cells', y=.975, fontsize=16, weight='semibold')
    fig.text(.5, .925, STUDY + ' | Q4, budget 15 | 108/108 workers complete', ha='center', fontsize=10.8)
    fig.text(.025, .125, 'Top: medians with exact min–max whiskers over three registered repeats per method and cell; ranges are not confidence intervals.', fontsize=9.2)
    fig.text(.025, .087, 'Bottom: diamonds = ratio of paired cost sums; dots/whiskers = all three individual H / min(D, S) ratios and their min–max range.', fontsize=9.2)
    fig.text(.025, .049, 'Every input cell is retained. C1/C2 are the original seed labels. Each service cost includes setup and every reached-version rebuild.', fontsize=9.2)
    fig.text(.025, .011, 'Descriptive revision results on diagnosed inputs; the original unseen-input confirmation remains failed.', fontsize=9.2)
    return fig


def cost_memory_figure(plt, values):
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(12.6, 7.75))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.35, 1.), left=.095, right=.98,
                         top=.77, bottom=.19, wspace=.29, hspace=.52)
    lookup = {(row['graph'], row['method']): row for row in values['graph_methods']}
    max_rss = max(row['process_high_water_rss_MiB']['maximum'] for row in values['graph_methods'])
    for col, graph in enumerate(GRAPHS):
        phase_ax = fig.add_subplot(gs[0, col])
        rss_ax = fig.add_subplot(gs[1, col])
        largest = max(lookup[graph, m]['mean_Q4_seconds'] for m in METHODS)
        for row, method in enumerate(METHODS):
            record = lookup[graph, method]
            start = 0.
            for phase, color in zip(PHASES, PHASE_COLORS, strict=True):
                amount = record['mean_phases_seconds'][phase]
                phase_ax.barh(row, amount, left=start, color=color, height=.53,
                              edgecolor='white', linewidth=.5)
                start += amount
            phase_ax.annotate(f'{start:.2f} s', (start, row), xytext=(5, 0), textcoords='offset points',
                              va='center', fontsize=9.3)
            stat = record['process_high_water_rss_MiB']
            rss_ax.errorbar(stat['median'], row,
                            xerr=[[stat['median'] - stat['minimum']], [stat['maximum'] - stat['median']]],
                            fmt=MARKERS[method], color=COLORS[method], markersize=6.2,
                            capsize=4, elinewidth=1.4)
            rss_ax.annotate(f"{stat['median']:.1f} [{stat['minimum']:.1f}, {stat['maximum']:.1f}]",
                            (stat['maximum'], row), xytext=(8, 0), textcoords='offset points',
                            va='center', fontsize=8.8, color=COLORS[method])
        phase_ax.set_title(graph + ': service phases', loc='left', fontsize=11.7, pad=9)
        rss_ax.set_title(graph + ': actual process memory', loc='left', fontsize=11.7, pad=9)
        phase_ax.set_xlim(0, largest * 1.25)
        # Reserve right-hand space for all explicit median [min,max] labels.
        rss_ax.set_xlim(0, max_rss * 1.47)
        for ax in (phase_ax, rss_ax):
            ax.set_yticks(range(3), [LABELS[m] for m in METHODS], fontsize=9.4)
            ax.set_ylim(2.55, -.55)
            _style_axis(ax, grid='x')
        phase_ax.set_xlabel('Mean full setup + Q4 per worker (s)', fontsize=9.8)
        rss_ax.set_xlabel('Process high-water RSS (MiB)', fontsize=9.8)
        rss_ax.set_xticks([0, 500, 1000, 1500])
    fig.legend([Patch(facecolor=color) for color in PHASE_COLORS], PHASES,
               loc='upper center', bbox_to_anchor=(.5, .875), ncol=2, frameon=False, fontsize=10)
    fig.suptitle('Where service time and process memory are spent', y=.975, fontsize=16, weight='semibold')
    fig.text(.5, .925, STUDY + ' | All six cells and 18 workers per graph and method',
             ha='center', fontsize=10.5)
    fig.text(.025, .115, 'Top: arithmetic means of exclusive service phases across all 18 workers; stacks sum to mean setup + Q4. Nested timers are not added twice.', fontsize=9.1)
    fig.text(.025, .080, 'Bottom: median [minimum, maximum] of the 18 reported process-lifetime RSS peaks; no sum of peaks. Ranges are descriptive, not confidence intervals.', fontsize=9.1)
    fig.text(.025, .045, 'Candidate work includes all reached-version frontier setup/access and screening. Numerical work includes responses, Grams, certification and transport.', fontsize=9.1)
    fig.text(.025, .010, 'Reset + other retains the remaining measured request overhead. Online equality work is included; independent checker time is separate.', fontsize=9.1)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ROOT / 'reports/TIE_RETEST_ANALYSIS.json')
    parser.add_argument('--summary', type=Path, default=ROOT / 'reports/A2_TIE_RETEST_SUMMARY.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'figures/tie_revision')
    parser.add_argument('--png-dpi', type=int, default=200)
    args = parser.parse_args()
    if not 96 <= args.png_dpi <= 360:
        parser.error('--png-dpi must be between 96 and 360')
    sources = [args.analysis, args.summary, ROOT / 'reports/A2_COVERAGE_PROOF.txt',
               ROOT / 'reports/TIE_EQUALITY_PROOF.txt', ROOT / 'src/a2_figures.py', Path(__file__)]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    values = derive(json.loads(args.analysis.read_text()), json.loads(args.summary.read_text()))
    values['sources_sha256'] = hashes
    values['structure_scope'] = {
        'domain': 'All root nonedges inserted with weight 1/16 plus natural-weight negative-root-edge sign flips.',
        'access': 'Classical sorted insertion row frontier and explicit sparse weighted-flip frontier; H path shown.',
        'shared_kernel': 'D/S/H share strict bounds, local responses/Grams, interval comparison, exact equality and commit.',
        'equality': ['P A P^T = A', 'P s = s', 'P A_e P^T = A_f'],
        'acceptance': 'Strictly positive lower; every legal rival covered or resolved; canonical winner with its own response binding.',
        'state_transition': 'Commit advances the parent; H rebuilds if another decision is reached. Only a new opinion resets the structural root.',
        'unsupported': 'Unproved overlap remains unresolved; interval overlap or close scores never imply equality.',
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.mpl-cache-', dir=args.output_dir) as cache:
        old_config = os.environ.get('MPLCONFIGDIR')
        os.environ['MPLCONFIGDIR'] = cache
        try:
            plt = setup_matplotlib()
            plt.rcParams.update({'svg.hashsalt': 'a2-tie-revision-publication', 'axes.labelsize': 10})
            builders = [('a2_tie_coverage', lambda: structure_figure(plt)),
                        ('a2_tie_complete_cost', lambda: main_cost_figure(plt, values)),
                        ('a2_tie_cost_memory', lambda: cost_memory_figure(plt, values))]
            outputs = {}
            for stem, builder in builders:
                figure = builder()
                try:
                    outputs[stem] = save_figure(figure, args.output_dir, stem, args.png_dpi)
                    # Matplotlib emits trailing spaces inside SVG path lines.
                    # Normalize whitespace without changing the vector geometry.
                    for output in outputs[stem]:
                        output = Path(output)
                        if output.suffix == '.svg':
                            output.write_text('\n'.join(line.rstrip() for line in output.read_text().splitlines()) + '\n')
                finally:
                    plt.close(figure)
        finally:
            if old_config is None:
                os.environ.pop('MPLCONFIGDIR', None)
            else:
                os.environ['MPLCONFIGDIR'] = old_config
    if any(sha(ROOT / path) != frozen for path, frozen in hashes.items()):
        raise ValueError('a figure source changed during rendering')
    values['publication_figures'] = {name: [{'path': str(Path(path).relative_to(ROOT)), 'sha256': sha(path)}
                                           for path in paths] for name, paths in outputs.items()}
    target = args.output_dir / 'a2_tie_plot_data.json'
    target.write_text(json.dumps(values, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'figures': outputs, 'exact_plotted_data': str(target),
                      'retained_cells': len(values['cells']), 'retained_workers': values['retained_workers']}, indent=2))


if __name__ == '__main__':
    main()
