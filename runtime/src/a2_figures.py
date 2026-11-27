"""Three publication figures from frozen A2 analysis, without rerunning service.

Run only after timed batches are idle. No numerical/runtime module is imported.
The same three figure names are reused for development or confirmation; SVG and
PNG are alternative exports of each figure, not additional figure designs.
All observations, range endpoints, phase sums and exact emission numerators and
denominators used by the figures are saved in a stage-specific JSON artifact.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
METHODS = ('D', 'S', 'H')
LABELS = {'D': 'D-all', 'S': 'Stream-all', 'H': 'Implicit-all'}
COLORS = {'D': '#596D83', 'S': '#B67D18', 'H': '#007F87'}
MARKERS = {'D': 's', 'S': '^', 'H': 'o'}
FAMILIES = {'slow': 'Slow', 'independent': 'Independent', 'signed-neighbor': 'Signed neighbor'}
PHASE_ORDER = ('Root setup', 'Candidate access', 'Strict screening',
               'Responses and Gram scores', 'Solve, certify and commit', 'Reset and other')
PHASE_COLORS = ('#D3DCE6', '#547C9C', '#87B4B7', '#D4A257', '#8876A6', '#D4D2CC')
ACCESS_PHASES = {'caps_build', 'caps_sort', 'posting/index_build', 'frontier_prepare',
                 'seed_access', 'candidate_access', 'discovery'}
RESPONSE_PHASES = {'response_acquisition', 'comparison/refinement'}
CERTIFY_PHASES = {'state_solve_and_cert', 'transport_local'}
TITLE_SCOPE = 'All nonedges + weighted flips; generated opinions'


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def nonnegative(value):
    return finite_number(value) and value >= 0


def summary(values):
    values = list(values)
    return dict(values=values, count=len(values),
                minimum=min(values) if values else None,
                maximum=max(values) if values else None,
                median=statistics.median(values) if values else None,
                mean=statistics.mean(values) if values else None)


def maximum_present(values):
    present = [value for value in values if nonnegative(value)]
    return max(present) if present else None


def phase_buckets(metric, complete_cost):
    """Use the exclusive service ledger; never add nested build subphases twice."""
    buckets = dict.fromkeys(PHASE_ORDER, 0.)
    structural = metric.get('structural_seconds')
    buckets['Root setup'] = structural if nonnegative(structural) else 0.
    for phase, value in metric.get('request_phases_seconds', {}).items():
        if not finite_number(value) or value < -1e-8:
            raise ValueError(f'invalid request phase duration: {phase}={value!r}')
        group = ('Candidate access' if phase in ACCESS_PHASES else
                 'Strict screening' if phase == 'endpoint_screen' else
                 'Responses and Gram scores' if phase in RESPONSE_PHASES else
                 'Solve, certify and commit' if phase in CERTIFY_PHASES else 'Reset and other')
        buckets[group] += value
    adjustment = None
    if complete_cost is not None:
        adjustment = complete_cost - sum(buckets.values())
        if abs(adjustment) > max(1e-7, 1e-9 * complete_cost):
            raise ValueError('complete Q4 cost does not reconcile with exclusive phase durations')
        buckets['Reset and other'] += adjustment
    return dict(exclusive_seconds=buckets, reconciliation_adjustment_seconds=adjustment,
                nested_frontier_build_seconds=metric.get('frontier_build_phases_seconds', {}),
                nested_index_build_seconds=metric.get('index_build_phases_seconds', {}),
                raw_structural_phases_seconds=metric.get('structural_phases'),
                raw_request_phases_seconds=metric.get('request_phases_seconds', {}))


def emission_values(metric):
    """K is legal insertion emissions, not heap pops, Grams or response RHS."""
    versions = []
    total_k = total_flips = total_live = 0
    coverage_known = True
    for event in metric.get('versions', []):
        k = event.get('insertion_emissions')
        flips = event.get('flips_emitted')
        live = event.get('current_legal_candidates')
        if any(type(value) is not int or value < 0 for value in (k, flips, live)):
            coverage_known = False
        else:
            total_k += k
            total_flips += flips
            total_live += live
        versions.append({name: event.get(name) for name in (
            'query_position', 'epoch', 'parent_version_id', 'registered_catalog_size',
            'current_legal_candidates', 'insertion_emissions', 'flips_emitted',
            'raw_heap_pops', 'legal_insertion_pops', 'processed_insertion_skips',
            'processed_flip_skips', 'processed_skips', 'graph_rejects',
            'committed_rejects', 'total_source_pops', 'stream_pairs_examined',
            'emitted_for_screen', 'source_complete', 'frontier_rebuilds')})
    available = coverage_known and bool(versions) and total_live > 0
    return dict(
        definition='sum legal insertion emissions K / sum current legal catalog sizes over actual decisions',
        denominator_includes='both current legal insertions and current legal weighted flips',
        legal_insertion_emissions=total_k if coverage_known else None,
        explicit_flip_emissions=total_flips if coverage_known else None,
        current_legal_catalog_denominator=total_live if coverage_known else None,
        insertion_fraction=total_k / total_live if available else None,
        insertion_fraction_exact=f'{total_k}/{total_live}' if available else None,
        all_emission_fraction=(total_k + total_flips) / total_live if available else None,
        all_emission_fraction_exact=f'{total_k + total_flips}/{total_live}' if available else None,
        coverage_known=coverage_known, version_count=len(versions), versions=versions)


def payload_maxima(metric):
    samples = metric.get('memory_samples', [])
    fields = ('explicit_catalog_bytes', 'incidence_bytes', 'index_private_bytes',
              'structural_weight_grouping', 'peak_frontier_owned_bytes',
              'peak_emitted_batch_array_bytes', 'shared_response_bytes',
              'gram_python_bytes', 'gram_count', 'peak_gram_count',
              'actual_candidate_cache_python_bytes','decision_and_access_log_python_bytes',
              'low_rank_steps', 'peak_low_rank_steps', 'graph_csr', 'root_operator_csr')
    result = {field: maximum_present(sample.get(field) for sample in samples) for field in fields}
    result['maximum_observed_gram_entries'] = maximum_present(
        sample.get(field) for sample in samples for field in ('gram_count', 'peak_gram_count'))
    result['scope'] = ('Per-request ownership snapshots and recorded payload maxima. Shared response is a '
                       'boundary snapshot; Gram bytes remain unknown when not measured. These are not process RSS.')
    return result


def raw_commit_counts(raw_directory, worker, analysis, raw_sources):
    if raw_directory is None:
        return dict(available=False, reason='accepted edit kinds are absent from analysis; no raw directory supplied')
    filename = (f"{worker['position']:03d}_{worker['graph']}_{worker['stream']}_"
                f"{worker['seed_label']}_{worker['label']}.json")
    path = raw_directory / filename
    if not path.is_file():
        return dict(available=False, reason='registered raw result is unavailable', file=filename)
    content = path.read_bytes()
    raw_sources[filename] = hashlib.sha256(content).hexdigest()
    trace = json.loads(content)
    for key in ('schedule_position', 'graph', 'stream', 'seed_label', 'seed', 'method', 'label', 'repeat_id'):
        expected = worker['position'] if key == 'schedule_position' else worker[key]
        if trace.get(key) != expected:
            raise ValueError(f'{filename}: raw result identity mismatch for {key}')
    for key in ('protocol_sha256', 'input_manifest_sha256'):
        if trace.get(key) != analysis[key]:
            raise ValueError(f'{filename}: raw result is from a different frozen analysis')
    rows = []
    for q, request in enumerate(trace.get('requests', []), 1):
        insertions = flips = 0
        for decision in request.get('decisions', []):
            if decision.get('status') != 'accepted':
                continue
            if decision.get('flip') == 1:
                flips += 1
            elif decision.get('flip') == 0:
                insertions += 1
            else:
                raise ValueError(f'{filename}: accepted decision has no insertion/flip kind')
        rows.append(dict(query_position=q, request_status=request.get('status'),
                         accepted_insertions=insertions, accepted_flips=flips,
                         accepted_total=insertions + flips))
    return dict(available=True, file=filename, per_request=rows,
                accepted_insertions=sum(row['accepted_insertions'] for row in rows),
                accepted_flips=sum(row['accepted_flips'] for row in rows),
                accepted_total=sum(row['accepted_total'] for row in rows))


def derive(analysis, analysis_path, analysis_sha256, raw_directory=None):
    stage = analysis.get('stage')
    if stage not in ('development', 'confirmation'):
        raise ValueError('expected A2 development or confirmation analysis')
    cells = analysis.get('cells', [])
    expected_cells = 6 if stage == 'development' else 12
    if len(cells) != expected_cells or analysis.get('planned_workers') != expected_cells * 9:
        raise ValueError('analysis does not preserve the full registered 54/108-worker denominator')
    workers = {(row['graph'], row['stream'], row['seed_label'], row['repeat_id'], row['method']): row
               for row in analysis['workers']}
    if len(workers) != analysis['planned_workers']:
        raise ValueError('missing or duplicate registered workers in the analysis')
    raw_sources, derived_cells, all_repeats = {}, [], {method: [] for method in METHODS}
    for cell in cells:
        entry = {key: cell[key] for key in ('graph', 'stream', 'seed_label')}
        entry['methods'] = {}
        for method in METHODS:
            costs = cell['costs'][method]
            if costs['planned_workers'] != 3 or len(costs['Q4_seconds']) != 3:
                raise ValueError('each graph/family/seed/method must retain three registered repeats')
            metrics = {row['repeat_id']: row for row in cell['work_and_memory'][method]}
            if set(metrics) != {0, 1, 2}:
                raise ValueError('phase/memory records do not retain all three repeat IDs')
            repeats = []
            for repeat in range(3):
                worker = workers[cell['graph'], cell['stream'], cell['seed_label'], repeat, method]
                metric = metrics[repeat]
                q4 = costs['Q4_seconds'][repeat]
                complete = bool(worker['complete'])
                if complete != bool(metric['complete']) or complete != (q4 is not None):
                    raise ValueError('analysis disagrees about complete worker/prefix eligibility')
                if q4 is not None and (not nonnegative(q4) or q4 <= 0):
                    raise ValueError('complete service time must be finite and positive')
                reported_rss = metric.get('process_peak_rss_bytes')
                sampled_rss = metric.get('external_sampled_peak_rss_bytes')
                observation = dict(
                    graph=cell['graph'], stream=cell['stream'], seed_label=cell['seed_label'],
                    method=method, label=LABELS[method], repeat_id=repeat,
                    position=worker['position'], complete=complete, valid=worker.get('valid'),
                    status=worker['status'], trace_status=worker.get('trace_status'),
                    resource_failure=worker.get('resource_failure'), failure_reason=worker.get('failure_reason'),
                    Q1_seconds=costs['Q1_seconds'][repeat], Q4_seconds=q4,
                    retained_Q4_checkpoint_seconds=worker.get('Q4_seconds'),
                    process_wall_seconds=worker.get('process_wall_seconds'),
                    reported_process_peak_rss_bytes=reported_rss,
                    sampled_process_peak_rss_bytes=sampled_rss,
                    observed_process_peak_rss_bytes=maximum_present((reported_rss, sampled_rss)),
                    registered_catalog_size=metric.get('catalog_size'),
                    phases=phase_buckets(metric, q4), emissions=emission_values(metric),
                    payload=payload_maxima(metric), numeric_counters=metric.get('counters', {}),
                    access_counters=metric.get('access', {}),
                    accepted_edits=raw_commit_counts(raw_directory, worker, analysis, raw_sources))
                repeats.append(observation)
                all_repeats[method].append(observation)
            eligible = [row for row in repeats if row['complete']]
            entry['methods'][method] = dict(
                planned_repeats=3, complete_repeats=len(eligible), observations=repeats,
                full_Q4_seconds=summary(row['Q4_seconds'] for row in eligible),
                legal_insertion_emission_fraction=summary(
                    row['emissions']['insertion_fraction'] for row in eligible
                    if row['emissions']['insertion_fraction'] is not None))
        derived_cells.append(entry)
    aggregates = {}
    for method, repeats in all_repeats.items():
        completed = [row for row in repeats if row['complete']]
        rss = [row['observed_process_peak_rss_bytes'] for row in repeats
               if nonnegative(row['observed_process_peak_rss_bytes'])]
        phase_totals = {phase: sum(row['phases']['exclusive_seconds'][phase] for row in completed)
                        for phase in PHASE_ORDER}
        known_edits = [row for row in repeats if row['accepted_edits']['available']]
        completed_edits = [row for row in known_edits if row['complete']]
        aggregates[method] = dict(
            label=LABELS[method], planned_workers=len(repeats), complete_workers=len(completed),
            worker_status_counts=dict(Counter(row['status'] for row in repeats)),
            full_scheduled_Q4_total_seconds=analysis['full_schedule_totals_seconds'][method],
            completed_worker_service_sum_seconds=sum(row['Q4_seconds'] for row in completed),
            completed_Q4_seconds=summary(row['Q4_seconds'] for row in completed),
            completed_phase_totals_seconds=phase_totals,
            completed_phase_mean_seconds={phase: value / len(completed) if completed else None
                                          for phase, value in phase_totals.items()},
            observed_process_peak_rss_bytes=summary(rss),
            payload_maxima={field: maximum_present(row['payload'].get(field) for row in repeats)
                            for field in ('explicit_catalog_bytes', 'shared_response_bytes', 'gram_python_bytes',
                                          'maximum_observed_gram_entries', 'peak_frontier_owned_bytes',
                                          'peak_emitted_batch_array_bytes', 'incidence_bytes', 'index_private_bytes')},
            accepted_edits=dict(
                workers_with_raw_kind_records=len(known_edits),
                all_registered_workers_have_kind_records=len(known_edits) == len(repeats),
                observed_insertions=(sum(row['accepted_edits']['accepted_insertions'] for row in known_edits)
                                     if known_edits else None),
                observed_flips=(sum(row['accepted_edits']['accepted_flips'] for row in known_edits)
                                if known_edits else None),
                completed_worker_insertions=(sum(row['accepted_edits']['accepted_insertions'] for row in completed_edits)
                                            if completed_edits else None),
                completed_worker_flips=(sum(row['accepted_edits']['accepted_flips'] for row in completed_edits)
                                       if completed_edits else None)))
    return dict(
        schema_version=1, stage=stage, scope=TITLE_SCOPE,
        source_analysis=str(analysis_path.resolve()), source_analysis_sha256=analysis_sha256,
        protocol_sha256=analysis['protocol_sha256'], input_manifest_sha256=analysis['input_manifest_sha256'],
        source_raw_directory=str(raw_directory.resolve()) if raw_directory is not None else None,
        raw_result_files_sha256=raw_sources,
        analysis_consistency_pass=analysis.get('correctness_accounting_pass'),
        validation_issues=analysis.get('validation_issues', []),
        planned_workers=analysis['planned_workers'], complete_workers=analysis['complete_workers'],
        complete_workers_by_method=analysis.get('complete_workers_by_method'),
        status_counts=analysis.get('status_counts'), cells=derived_cells, methods=aggregates,
        aggregate_comparisons=analysis.get('aggregate_comparisons'),
        paired_coverage=analysis.get('paired_coverage'), negative_cells=analysis.get('negative_cells'),
        reference_resource_failures=analysis.get('reference_resource_failures'),
        observed_service_seconds_including_partial=analysis.get('observed_service_seconds_including_partial'),
        observed_process_seconds_including_partial=analysis.get('observed_process_seconds_including_partial'),
        figure_conventions={
            'cost': 'full structural setup plus all four requests; includes every charged version rebuild',
            'cost_ranges': 'median with exact min/max of completed registered repeats; not a confidence interval',
            'failed_workers': 'retained in counts and exact observations; no imputed complete cost or infinite ratio',
            'emissions': 'K / sum current legal catalog sizes; K is legal insertion emissions, including later screened rows',
            'denominator': 'the current legal full catalog includes both insertion and weighted-flip candidates',
            'work_categories': 'heap/source pops, emissions, endpoint screening, Grams, response blocks and scalar RHS stay separate',
            'rss': 'max of reported process peak and externally sampled process RSS, including failed started workers',
            'payload': 'ownership snapshots are separate from total process RSS; response is a boundary snapshot',
            'phase_totals': 'exclusive ledger only; nested frontier/index subphases are stored separately and not added twice',
            'phase_population': 'separate Alpha/OTC slow cells at the first registered seed label, mean of their completed repeats 0..2; no pooled replacement of the full schedule',
            'accepted_kinds': 'only from explicitly supplied, identity-matched raw traces; unavailable is not zero',
        })


def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'semibold', 'axes.labelcolor': '#25313B',
        'text.color': '#25313B', 'xtick.color': '#40515F', 'ytick.color': '#40515F',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'svg.fonttype': 'none', 'svg.hashsalt': 'a2-figures-v1',
        'savefig.facecolor': 'white',
    })
    return plt


def save_figure(figure, directory, stem, dpi):
    paths = []
    for extension in ('svg', 'png'):
        path = directory / f'{stem}.{extension}'
        metadata = ({'Creator': 'A2 frozen-result figure builder', 'Date': None} if extension == 'svg'
                    else {'Software': 'A2 frozen-result figure builder'})
        figure.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=.18, metadata=metadata)
        paths.append(str(path.resolve()))
    return paths


def structure_figure(plt):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    figure = plt.figure(figsize=(13.2, 5.2))
    axis = figure.add_axes((.015, .15, .97, .69))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis('off')

    def box(x, y, w, h, text, color='#EDF2F6', border='#63788B'):
        axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02',
                                     linewidth=1.25, edgecolor=border, facecolor=color))
        axis.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=10.3, linespacing=1.45)

    def arrow(start, end):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=13,
                                       linewidth=1.5, color='#63788B', shrinkA=3, shrinkB=4))

    box(.02, .39, .17, .23, 'Certified state\nat the current parent')
    box(.27, .65, .22, .18, 'Insertion row frontier\nOne head per active row', '#E1F1F0', '#378B8C')
    box(.27, .20, .22, .18, 'Explicit sparse flip frontier\nNatural root edge weights', '#F6EDDB', '#B18B40')
    box(.56, .39, .19, .23, 'Maximum of both heads\nBounds the remaining catalog')
    box(.81, .39, .17, .23, 'Shared strict screening\nand certified Gram scores', '#E8EEF6')
    box(.66, .035, .30, .18, 'Commit the canonical winner;\notherwise stop or return unresolved', '#E8EEF6')
    arrow((.19, .54), (.27, .74))
    arrow((.19, .47), (.27, .29))
    arrow((.49, .74), (.56, .54))
    arrow((.49, .29), (.56, .47))
    arrow((.75, .505), (.81, .505))
    arrow((.895, .39), (.83, .215))
    axis.text(.78, .58, 'emit', ha='center', fontsize=8.5)
    axis.plot((.66, .105), (.125, .125), color='#63788B', linewidth=1.5)
    arrow((.105, .125), (.105, .39))
    axis.text(.38, .065, 'After an accepted edit: certify the new state\nand rebuild H at the next reached parent',
              ha='center', va='center', fontsize=9.2)
    figure.suptitle('Candidate-lazy access with full-catalog coverage', y=.965, fontsize=15, weight='semibold')
    figure.text(.5, .89, TITLE_SCOPE, ha='center', fontsize=10.5)
    figure.text(.035, .065, 'H access path shown. D-all and Stream-all use the same strict screening, numerical certification and commit kernel.', fontsize=9)
    figure.text(.035, .025, 'Root legality and committed-edit exclusion apply throughout. The current H frontier is rebuilt at every reached parent version.', fontsize=9)
    return figure


def cell_label(cell):
    label = f"{cell['graph']}\n{FAMILIES.get(cell['stream'], cell['stream'])}"
    if cell['seed_label'] != 'development':
        short = {'confirmation-1': 'C1', 'confirmation-2': 'C2'}.get(cell['seed_label'], cell['seed_label'])
        label += '\n' + short
    return label


def cost_emission_figure(plt, values):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    cells = values['cells']
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(max(11.5, 1.25 * len(cells) + 2), 8.5),
                                gridspec_kw={'height_ratios': (1.8, 1.)})
    figure.subplots_adjust(top=.84, bottom=.22, hspace=.16, left=.085, right=.985)
    costs_axis, emissions_axis = axes
    all_costs = [value for cell in cells for method in METHODS for value in cell['methods'][method]['full_Q4_seconds']['values']]
    if all_costs:
        costs_axis.set_yscale('log')
        costs_axis.set_ylim(min(all_costs) * .68, max(all_costs) * 1.75)
    else:
        costs_axis.set_ylim(0, 1)
        costs_axis.text(.5, .6, 'No completed Q4 workers', transform=costs_axis.transAxes, ha='center')
    emission_values_all = [value for cell in cells for method in METHODS
                          for value in cell['methods'][method]['legal_insertion_emission_fraction']['values']]
    # Keep exact zero emissions visible while separating small nonzero fractions.
    # This is only a display transform; observations and derived ratios are unchanged.
    emissions_axis.set_yscale('symlog', linthresh=1e-8, linscale=1., base=10)
    emissions_axis.set_ylim(0, max([1.] + emission_values_all) * 1.13)
    emissions_axis.set_yticks((0., 1e-8, 1e-6, 1e-4, 1e-2, 1.))
    emissions_axis.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: '0%' if value == 0 else f'{100 * value:.0e}%'))
    for axis in axes:
        axis.set_xlim(-.55, len(cells) - .45)
        axis.grid(axis='y', color='#E1E6EB', linewidth=.7)
        axis.set_axisbelow(True)
        for index in range(1, len(cells)):
            if cells[index]['graph'] != cells[index - 1]['graph']:
                axis.axvline(index - .5, color='#A9B7C4', linewidth=1.)
    for cell_index, cell in enumerate(cells):
        for method_index, method in enumerate(METHODS):
            x = cell_index + (method_index - 1) * .23
            row = cell['methods'][method]
            for axis, key in ((costs_axis, 'full_Q4_seconds'), (emissions_axis, 'legal_insertion_emission_fraction')):
                data = row[key]
                if data['count']:
                    median, lower, upper = data['median'], data['minimum'], data['maximum']
                    axis.errorbar(x, median, yerr=[[median - lower], [upper - median]],
                                  fmt=MARKERS[method], color=COLORS[method], ecolor=COLORS[method],
                                  markersize=6, capsize=3.5, elinewidth=1.5, markeredgewidth=1.25,
                                  clip_on=axis is costs_axis, zorder=3)
                    if axis is costs_axis:
                        axis.annotate(f"{data['count']}/3", (x, upper), xytext=(0, 7),
                                      textcoords='offset points', ha='center', fontsize=7.5, color=COLORS[method])
                else:
                    axis.text(x, .05, '0/3' if axis is costs_axis else 'n/a',
                              transform=axis.get_xaxis_transform(), ha='center', va='bottom',
                              fontsize=7.5, color=COLORS[method], weight='semibold')
    costs_axis.set_ylabel('Full setup + Q4 service (seconds; log scale)' if all_costs else 'Full setup + Q4 service (seconds)')
    emissions_axis.set_ylabel('Legal insertion emissions K\n/ sum of current legal catalog sizes\n(symlog; linear through 1e-6%)')
    emissions_axis.set_xticks(range(len(cells)), [cell_label(cell) for cell in cells], fontsize=8.7)
    legend = [Line2D([], [], color=COLORS[method], marker=MARKERS[method], linestyle='none',
                     markersize=7, label=LABELS[method]) for method in METHODS]
    costs_axis.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, 1.03), ncol=3, frameon=False)
    stage = values['stage'].capitalize()
    figure.suptitle(f'{stage}: complete same-domain cost and insertion emissions', y=.975, fontsize=15, weight='semibold')
    figure.text(.5, .93, TITLE_SCOPE + ' | Q4, budget 15, three registered repeats per cell', ha='center', fontsize=10)
    figure.text(.035, .104, 'Dots: completed-repeat medians. Whiskers: exact minimum and maximum, not confidence intervals. Counts: completed / planned.', fontsize=8.3)
    figure.text(.035, .076, 'All structural work and every reached-version rebuild are charged. Failed workers remain in the roster and have no imputed complete cost.', fontsize=8.3)
    figure.text(.035, .048, 'Emission axis: symlog; linear from 0 through 1e-8 of the catalog (1e-6%); exact zeros retained. Ticks show percentages.', fontsize=8.3)
    figure.text(.035, .020, 'Denominator: legal insertions and weighted flips. Heap/source pops, emissions, Grams and response RHS remain separate work measures.', fontsize=8.3)
    if values['analysis_consistency_pass'] is not True:
        figure.text(.99, .89, 'Consistency validation not passed; see exact-values JSON.', ha='right', color='#A03F36', fontsize=8.5)
    return figure


def display_number(value, divisor=1., digits=1):
    if value is None:
        return 'n/a'
    return f'{value / divisor:,.{digits}f}'


def cost_memory_figure(plt, values):
    from matplotlib.patches import Patch
    # Fixed descriptive cells, chosen by the protocol's existing slow/C1 rule.
    # Never replace these with a pooled mean over whichever other cells finished.
    graphs = ('Alpha', 'OTC')
    seed_label = 'confirmation-1' if values['stage'] == 'confirmation' else 'development'
    selected_name = 'C1' if values['stage'] == 'confirmation' else 'development'
    selected = {graph: next(cell for cell in values['cells'] if cell['graph'] == graph
                            and cell['stream'] == 'slow' and cell['seed_label'] == seed_label)
                for graph in graphs}
    figure = plt.figure(figsize=(14.4, 9.8))
    grid = figure.add_gridspec(2, 2, height_ratios=(2.2, 3.2), top=.775, bottom=.17,
                              left=.085, right=.985, hspace=.42, wspace=.29)
    phase_axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
    table_axis = figure.add_subplot(grid[1, :])
    table_axis.axis('off')
    for graph, axis in zip(graphs, phase_axes, strict=True):
        largest_mean = 0.
        for index, method in enumerate(METHODS):
            by_repeat = {row['repeat_id']: row for row in selected[graph]['methods'][method]['observations']}
            observations = [by_repeat.get(repeat) for repeat in (0, 1, 2)]
            if not all(row is not None and row['complete'] for row in observations):
                axis.text(.01, index, 'Selected Q4 block incomplete', transform=axis.get_yaxis_transform(),
                          va='center', fontsize=8.5, color=COLORS[method])
                continue
            left = 0.
            for phase, color in zip(PHASE_ORDER, PHASE_COLORS, strict=True):
                amount = statistics.mean(row['phases']['exclusive_seconds'][phase] for row in observations)
                axis.barh(index, amount, left=left, height=.56, color=color, edgecolor='white', linewidth=.4)
                left += amount
            largest_mean = max(largest_mean, left)
            axis.annotate(f'{left:.2f} s', (left, index), xytext=(6, 0), textcoords='offset points',
                          va='center', fontsize=8.5)
        axis.set_title(graph + ' / slow / ' + selected_name, fontsize=11, pad=10)
        axis.set_yticks(range(3), [LABELS[method] for method in METHODS])
        axis.invert_yaxis()
        axis.set_xlim(0, largest_mean * 1.24 if largest_mean else 1.)
        axis.set_xlabel('Full setup + Q4 seconds; mean of repeats 0-2', fontsize=9)
        axis.grid(axis='x', color='#E1E6EB', linewidth=.7)
        axis.set_axisbelow(True)
    figure.legend([Patch(facecolor=color) for color in PHASE_COLORS], PHASE_ORDER,
                  loc='upper center', bbox_to_anchor=(.5, .865), ncol=3, frameon=False, fontsize=8.7)
    rows = []
    for graph in graphs:
        for method in METHODS:
            observations = [row for cell in values['cells'] if cell['graph'] == graph
                            for row in cell['methods'][method]['observations']]
            started = [row for row in observations if nonnegative(row.get('process_wall_seconds'))]
            rss = summary(row['observed_process_peak_rss_bytes'] for row in started
                          if nonnegative(row.get('observed_process_peak_rss_bytes')))
            payload = {field: maximum_present(row['payload'].get(field) for row in observations)
                       for field in ('explicit_catalog_bytes', 'shared_response_bytes',
                                     'maximum_observed_gram_entries', 'peak_frontier_owned_bytes',
                                     'actual_candidate_cache_python_bytes')}
            rss_text = (f"{rss['median'] / 2**20:,.1f} [{rss['minimum'] / 2**20:,.1f}, {rss['maximum'] / 2**20:,.1f}]\n"
                        f"n={rss['count']}/{len(started)} started" if rss['count'] else 'n/a')
            rows.append([graph, LABELS[method], f"{sum(row['complete'] for row in observations)}/{len(observations)}",
                         rss_text, display_number(payload['explicit_catalog_bytes'], 2**20, 2),
                         display_number(payload['peak_frontier_owned_bytes'], 2**20, 3),
                         display_number(payload['actual_candidate_cache_python_bytes'], 2**20, 3),
                         display_number(payload['shared_response_bytes'], 2**20, 3),
                         display_number(payload['maximum_observed_gram_entries'], digits=0)])
    columns = ('Graph', 'Method', 'Complete\nQ4 workers',
               'All-started peak RSS, MiB\nmedian [minimum, maximum]',
               'Explicit catalog\nMiB max', 'Frontier payload\nMiB max',
               'Candidate cache\nMiB max', 'Response snapshot\nMiB max', 'Gram entries\nmax')
    table_axis.set_title('All families and seed labels, by graph and method; started failures retained',
                         loc='left', fontsize=10, pad=13)
    table = table_axis.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center',
                             colWidths=(.055, .09, .07, .24, .105, .105, .12, .115, .10))
    table.auto_set_font_size(False)
    table.set_fontsize(8.25)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#D9E0E6')
        cell.set_linewidth(.6)
        cell.set_height(.14 if row else .145)
        if row == 0:
            cell.set_facecolor('#EDF2F6')
            cell.set_text_props(weight='semibold')
        else:
            if row > 3:
                cell.set_facecolor('#F6F8FA')
            if column == 1:
                cell.set_text_props(color=COLORS[METHODS[(row - 1) % 3]], weight='semibold')
    figure.suptitle(f'Selected {selected_name} slow Q4, descriptive phases', y=.975, fontsize=15, weight='semibold')
    figure.text(.5, .93, TITLE_SCOPE + ' | ' + values['stage'].capitalize(), ha='center', fontsize=10.5)
    full_complete = values['complete_workers'] == values['planned_workers']
    figure.text(.5, .894, f"Full scheduled Q4 cost: {'Complete' if full_complete else 'Incomplete'} "
                 f"({values['complete_workers']}/{values['planned_workers']} workers complete)",
                 ha='center', fontsize=10.5, weight='semibold', color='#25313B' if full_complete else '#A03F36')
    figure.text(.035, .108, f'Top panels: fixed Alpha/OTC slow, {seed_label}, repeats 0-2. These descriptive phase examples do not replace the full-schedule result.', fontsize=8.5)
    figure.text(.035, .076, 'The table includes every started worker for each graph and method, including unresolved requests. No remaining-cell total or new aggregate ratio is formed.', fontsize=8.5)
    figure.text(.035, .044, 'RSS uses the larger reported/sampled process peak. Owned catalog, frontier and candidate-cache payloads are separate from process RSS.', fontsize=8.5)
    figure.text(.035, .012, 'Response bytes are boundary snapshots. Full cache/log records remain in the exact-values JSON; native solver workspace is covered by process RSS.', fontsize=8.5)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', required=True, type=Path)
    parser.add_argument('--raw-directory', type=Path,
                        help='optional matching raw results, to retain accepted insertion versus flip counts')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/a2')
    parser.add_argument('--png-dpi', type=int, default=180)
    arguments = parser.parse_args()
    if not 96 <= arguments.png_dpi <= 360:
        parser.error('--png-dpi must be between 96 and 360')
    content = arguments.analysis.read_bytes()
    analysis = json.loads(content)
    values = derive(analysis, arguments.analysis, hashlib.sha256(content).hexdigest(), arguments.raw_directory)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()
    builders = (
        ('a2_structure', lambda: structure_figure(plt)),
        ('a2_main_cost_emissions', lambda: cost_emission_figure(plt, values)),
        ('a2_cost_memory', lambda: cost_memory_figure(plt, values)),
    )
    outputs = {}
    for stem, builder in builders:
        figure = builder()
        try:
            outputs[stem] = save_figure(figure, arguments.output_dir, stem, arguments.png_dpi)
        finally:
            plt.close(figure)
    values['publication_figures'] = outputs
    data_path = arguments.output_dir / f"a2_{values['stage']}_figure_values.json"
    temporary = data_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + '\n')
    temporary.replace(data_path)
    print(json.dumps({'stage': values['stage'], 'figures': outputs, 'exact_values': str(data_path.resolve())}, indent=2))


if __name__ == '__main__':
    main()
