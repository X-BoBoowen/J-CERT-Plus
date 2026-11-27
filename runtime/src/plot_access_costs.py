"""Render the complete confirmation matrix's full-cost prefixes and work table."""
import argparse
from fractions import Fraction
from itertools import product
import json
import math
from pathlib import Path
from statistics import median

from .access_bench import jobs, read_protocol


RESPONSE_RHS = ('candidate_rhs', 'winner_rhs', 'policy_refresh_rhs', 'numerical_reanchor_rhs')
METHOD_STYLE = {
    'rebuild-always': ('#4B5F75', '--', 's'),
    'retain-always': ('#BC7533', ':', '^'),
    'budget-guard': ('#156B57', '-', 'o'),
}
FAMILIES = {'slow': 'Slow', 'independent': 'Independent', 'signed-neighbor': 'Signed neighbor'}
MIB = 2**20


def read_json(path):
    return json.loads(path.read_text())


def summary(values):
    values = list(values)
    return dict(median=median(values), minimum=min(values), maximum=max(values))


def close(observed, registered):
    return math.isclose(observed, registered, rel_tol=0., abs_tol=1e-7)


def read_worker(path, cfg, job, position, freeze):
    worker = read_json(path)
    if (worker['status'] != 'completed' or len(worker['requests']) != cfg['requests'] or
            worker['schedule_position'] != position or
            any(worker.get(field) != value for field, value in job.items()) or
            any(worker[key] != freeze[key] for key in
                ('protocol_sha256', 'input_manifest_sha256', 'source_files_sha256'))):
        raise ValueError(f'Incomplete or unbound confirmation worker: {path.name}')
    elapsed = worker['structural_seconds']
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError(f'Invalid structural time: {path.name}')
    prefix_times = {}
    state_rhs = response_rhs = response_blocks = peak_caps = 0
    for q, request in enumerate(worker['requests'], 1):
        request_seconds = request['request_seconds']
        if (request['query_position'] != q or request['status'] not in ('budget_stop', 'tolerance_stop') or
                not math.isfinite(request_seconds) or request_seconds < 0):
            raise ValueError(f'Invalid request {q}: {path.name}')
        elapsed += request_seconds
        if q in cfg['prefixes']:
            if not close(elapsed, worker['prefixes'][str(q)]):
                raise ValueError(f'Prefix {q} does not equal setup plus request costs: {path.name}')
            prefix_times[q] = elapsed
        counter = request['counters']
        state_rhs += counter['state_rhs']
        response_rhs += sum(counter[field] for field in RESPONSE_RHS)
        response_blocks += counter['response_blocks']
        peak_caps = max(peak_caps, request['memory']['incremental_root_caps_bytes'])
    if elapsed <= 0:
        raise ValueError(f'Nonpositive full-cost time: {path.name}')
    return dict(prefix_times=prefix_times,
                peak_process_rss_mib=worker['process_peak_rss_bytes'] / MIB,
                peak_retained_root_caps_mib=peak_caps / MIB,
                state_rhs=state_rhs, response_rhs=response_rhs,
                response_blocks=response_blocks)


def complete_confirmation(protocol_path, report_path, raw_directory):
    cfg, protocol_sha256 = read_protocol(protocol_path)
    grid = (len(cfg['graphs']), len(cfg['streams']), len(cfg['seeds']),
            cfg['repeat_count'], len(cfg['methods']))
    if grid != (2, 3, 2, 3, 3):
        raise ValueError(f'Expected the registered 108-worker confirmation matrix, got {grid}')
    schedule = list(jobs(cfg))
    report = read_json(report_path)
    if (report['protocol_sha256'] != protocol_sha256 or
            report['expected_workers'] != len(schedule) or report['observed_workers'] != len(schedule) or
            report['correctness_accounting_pass'] is not True or report['validation_issues']):
        raise ValueError('Confirmation validation is incomplete or failed. Keep the gate and raw evidence.')
    freeze = read_json(raw_directory / 'FROZEN_JOBS.json')
    dispatch = read_json(raw_directory / 'SCHEDULE_RESULTS.json')
    if (freeze['jobs'] != schedule or freeze['protocol_sha256'] != protocol_sha256 or
            dispatch['status'] != 'completed' or len(dispatch['jobs']) != len(schedule)):
        raise ValueError('Confirmation scheduler has no complete matching roster')

    workers = {}
    for position, (row, job) in enumerate(zip(dispatch['jobs'], schedule, strict=True)):
        filename = f"{position:03d}_{job['graph']}_{job['stream']}_{job['seed']}_{job['label']}.json"
        if (row['position'] != position or row['job'] != job or row['result'] != filename or
                row['status'] != 'completed' or row['returncode'] != 0 or
                row['completed_requests'] != cfg['requests']):
            raise ValueError(f'Incomplete confirmation dispatch at position {position}')
        key = (job['graph'], job['stream'], job['seed'], job['repeat_id'], job['label'])
        workers[key] = read_worker(raw_directory / filename, cfg, job, position, freeze)

    report_cells = {(c['graph'], c['stream'], c['seed'], c['label']): c for c in report['cells']}
    method_labels = [m['label'] for m in cfg['methods']]
    groups = list(product(cfg['graphs'], cfg['streams'], cfg['seeds'], method_labels))
    if set(report_cells) != set(groups) or len(report_cells) != len(report['cells']):
        raise ValueError('Analyzed report cells do not cover the complete confirmation roster')
    for graph, stream, seed, label in groups:
        cell = report_cells[graph, stream, seed, label]
        repeats = [workers[graph, stream, seed, repeat, label] for repeat in range(cfg['repeat_count'])]
        for prefix in cfg['prefixes']:
            observed = median(row['prefix_times'][prefix] for row in repeats)
            if not close(observed, cell['prefixes_seconds_median'][str(prefix)]):
                raise ValueError(f'Report/raw prefix mismatch for {(graph, stream, seed, label, prefix)}')
    for label in method_labels:
        total = math.fsum(row['prefix_times'][cfg['requests']] for key, row in workers.items()
                          if key[-1] == label)
        if not close(total, report['total_scheduled_service_seconds'][label]):
            raise ValueError(f'Report/raw full-cost total mismatch for {label}')
    return cfg, report, workers


def pool_profiles(cfg, workers):
    control = next(m['label'] for m in cfg['methods'] if m['arm'] == 'rebuild-always')
    pairs = list(product(cfg['seeds'], range(cfg['repeat_count'])))
    data = {}
    for graph, stream in product(cfg['graphs'], cfg['streams']):
        base = [workers[graph, stream, seed, repeat, control]['prefix_times'][cfg['requests']]
                for seed, repeat in pairs]
        for method in cfg['methods']:
            label = method['label']
            rows = [workers[graph, stream, seed, repeat, label] for seed, repeat in pairs]
            costs = {q: summary(row['prefix_times'][q] for row in rows) for q in cfg['prefixes']}
            paired = summary(row['prefix_times'][cfg['requests']] / ref
                             for row, ref in zip(rows, base, strict=True))
            work = {key: summary(row[key] for row in rows)
                    for key in ('peak_process_rss_mib', 'peak_retained_root_caps_mib',
                                'state_rhs', 'response_rhs', 'response_blocks')}
            data[graph, stream, label] = dict(costs=costs, paired_q16_ratio=paired, work=work)
    return data


def render(cfg, data, output_stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': '#CBD3DB',
        'axes.labelcolor': '#293643', 'text.color': '#1B2635',
        'xtick.color': '#4B5563', 'ytick.color': '#4B5563', 'svg.fonttype': 'none',
    })
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharey='row')
    fig.subplots_adjust(left=.08, right=.977, top=.81, bottom=.157, hspace=.32, wspace=.17)
    fig.suptitle('Full confirmation service cost', fontsize=19, weight='semibold', y=.96)
    fig.text(.5, .899, 'Setup plus all request costs · median and observed range across two seeds × three matched runs',
             ha='center', color='#526172', fontsize=10.5)
    x = list(range(len(cfg['prefixes'])))
    guard = next(m for m in cfg['methods'] if m['arm'] == 'budget-guard')
    handles = []
    for method in cfg['methods']:
        arm = method['arm']
        color, line, marker = METHOD_STYLE[arm]
        name = {'rebuild-always': 'Rebuild always', 'retain-always': 'Retain always',
                'budget-guard': 'Budget guard'}[arm]
        if arm == 'budget-guard':
            name += f" ({Fraction(str(method['coefficient']))} of catalog)"
        handles.append(Line2D([], [], label=name, color=color, linestyle=line,
                              marker=marker, markersize=5, linewidth=1.8))

    for row, graph in enumerate(cfg['graphs']):
        row_peak = max(data[graph, stream, method['label']]['costs'][cfg['requests']]['maximum']
                       for stream in cfg['streams'] for method in cfg['methods'])
        for col, stream in enumerate(cfg['streams']):
            ax = axes[row, col]
            ax.axvspan(x[-1] - .39, x[-1] + .39, facecolor='#E8EEF4', alpha=.7, zorder=0)
            for method in cfg['methods']:
                color, line, marker = METHOD_STYLE[method['arm']]
                points = data[graph, stream, method['label']]['costs']
                centers = [points[q]['median'] for q in cfg['prefixes']]
                ax.fill_between(x, [points[q]['minimum'] for q in cfg['prefixes']],
                                [points[q]['maximum'] for q in cfg['prefixes']],
                                color=color, linewidth=0, alpha=.15, zorder=2)
                ax.plot(x, centers, color=color, linestyle=line, marker=marker,
                        markersize=4, linewidth=1.8, zorder=3)
                ax.scatter(x[-1], centers[-1], color=color, marker=marker, s=45,
                           edgecolor='white', linewidth=.45, zorder=4)
            ratio = data[graph, stream, guard['label']]['paired_q16_ratio']['median']
            ax.text(.045, .95, f'Q16 guard / rebuild\n{ratio:.2f}× matched median',
                    transform=ax.transAxes, va='top', fontsize=8.5, color='#2D4B43')
            ax.set_xlim(-.24, x[-1] + .39)
            ax.set_ylim(0, 1.13 * row_peak)
            ax.set_xticks(x, [f'Q{p}' for p in cfg['prefixes']])
            ax.get_xticklabels()[-1].set_fontweight('bold')
            ax.tick_params(axis='both', length=0)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='#E1E7ED', linewidth=.8)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(FAMILIES[stream], loc='left', fontsize=12, weight='semibold', pad=8)
            if col == 0:
                ax.set_ylabel(f'{graph}\nService cost (s)')

    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(.5, .067),
               fontsize=10, frameon=False)
    fig.text(.5, .039, 'Bands show the observed min–max. Repeated runs and nested prefixes are correlated.',
             ha='center', color='#526172', fontsize=9)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix('.png'), dpi=200, facecolor='white')
    fig.savefig(output_stem.with_suffix('.svg'), facecolor='white', metadata={'Date': None})
    plt.close(fig)


def save_work_table(output_path, cfg, report, data):
    table = dict(protocol_sha256=report['protocol_sha256'], gate_pass=report['gate_pass'],
                 seed_blocks=cfg['seeds'], repeats_per_seed=cfg['repeat_count'],
                 samples_per_cell=len(cfg['seeds']) * cfg['repeat_count'],
                 summary='Median and observed min–max across matched Q16 runs. Repeats are correlated.',
                 units=dict(peak_process_rss_mib='worker rusage peak resident MiB',
                            peak_retained_root_caps_mib='maximum incremental retained cap payload across requests, MiB',
                            state_rhs='Q16 state RHS count',
                            response_rhs='Q16 candidate, winner, policy refresh and numerical reanchor RHS',
                            response_blocks='Q16 response block count'),
                 cells=[dict(graph=g, stream=stream, label=label, **row['work'])
                        for (g, stream, label), row in data.items()])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(table, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('protocol/ACCESS_CONFIRMATION.json'))
    parser.add_argument('--report', type=Path, default=Path('reports/ACCESS_CONFIRMATION_RESULTS.json'))
    parser.add_argument('--raw-directory', type=Path, default=Path('raw/access_confirmation'))
    parser.add_argument('--output-stem', type=Path, default=Path('analysis/ACCESS_CONFIRMATION_COSTS'))
    parser.add_argument('--table-output', type=Path, default=Path('analysis/ACCESS_CONFIRMATION_MEMORY_RHS.json'))
    args = parser.parse_args()
    config, analyzed, raw_workers = complete_confirmation(args.protocol, args.report, args.raw_directory)
    profiles = pool_profiles(config, raw_workers)
    render(config, profiles, args.output_stem)
    save_work_table(args.table_output, config, analyzed, profiles)
