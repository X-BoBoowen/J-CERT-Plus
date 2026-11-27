"""Freeze exactly the first twelve V3 paired jobs; performs no numerical work."""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    a = p.parse_args()
    w = a.workspace.resolve()
    runtime = w / 'v3_runtime'
    output = w / 'v3/resumed/protocols/HG_FROZEN.json'
    if output.exists():
        raise FileExistsError('never overwrite a frozen timing batch')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=runtime, text=True).strip():
        raise ValueError('tracked runtime files must be committed before timing')
    baseline = json.loads((runtime / 'reproduction/SOURCE_BINDING.json').read_text())
    files = {}
    allowed = {'src/a2_session.py', 'src/a2_checker.py'}
    changed = {}
    for filename, old in baseline['source_files_sha256'].items():
        current = sha(runtime / filename)
        if current != old:
            if filename not in allowed:
                raise ValueError('unapproved numerical baseline change: '+filename)
            changed[filename] = {'v2': old, 'v3': current}
        files[filename] = current
    for folder in ('src', 'tests'):
        for path in (runtime / folder).glob('*.py'):
            files[str(path.relative_to(runtime))] = sha(path)
    manifest = json.loads((runtime / 'v2/INPUTS.json').read_text())
    jobs = []
    cells = [f for f in manifest['fixtures'] if f['graph'] in ('Alpha', 'OTC')]
    if len(cells) != 6:
        raise ValueError('six retained streams required')
    for cell_index, fixture in enumerate(cells):
        order = ('H', 'G') if cell_index % 2 == 0 else ('G', 'H')
        for method in order:
            position = len(jobs)
            jobs.append(dict(position=position, cell_index=cell_index, graph=fixture['graph'],
                             stream=fixture['stream'], method=method, repeat_id=0,
                             requests=4, budget_per_request=15,
                             output_file=f"{position:02d}_{fixture['graph']}_{fixture['stream']}_{method}_r0.json"))
    result = dict(id='J-CERT+-V3-six-old-streams-paired-HG-initial-batch',
        frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(), status='frozen_before_first_timed_worker',
        runtime_commit=commit, v2_baseline_commit=baseline['commits']['runtime_commit'],
        source_files_sha256=files,
        source_digest=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        intentional_baseline_changes=changed,
        numerical_core_change=False,
        core_change_description='G-only scheduling/access, authentic G checker method label and new worker; weighted gain/state/response/resolver/transport primitives unchanged',
        environment=dict(python=sys.version, platform=platform.platform(),
            packages={name: importlib.metadata.version(name) for name in ('numpy','scipy','psutil','threadpoolctl','sympy')}),
        resources=dict(seconds=180, memory_bytes=2*1024**3, cpu=min(os.sched_getaffinity(0)), numerical_threads=1),
        jobs=jobs, initial_workers=12, repeat_policy='No automatic extra repeats; an added batch requires a decision-relevant frozen rationale, total<=36 workers.',
        v2_input_manifest_sha256=sha(runtime / 'v2/INPUTS.json'),
        input_fixtures=cells, input_graphs={g:manifest['graphs'][g] for g in ('Alpha','OTC')},
        old_raw_dir='v2_received/evidence/v2/raw_a', current_raw_dir='v3/resumed/raw_hg',
        source_selection_file='v3/resumed/reports/state_selection_proposal.json',
        source_selection_sha256=sha(w / 'v3/resumed/reports/state_selection_proposal.json'),
        selected_state_observation='Read the seven preselected parent states from matching G version events in this batch; no additional opinions or seeds.',
        contracts=dict(delta=0, insertion_weight='1/16', hot_size=64, max_commits=15,
                       operator_scale=16, opinion_denominator=65536, proof_unit=2**48, screen_unit=2**20,
                       input_status='all V2 inputs are observed development/regression inputs',
                       resolver='unchanged exact single transposition; small n<=16 rational fallback; otherwise unresolved',
                       reset='fixed root factor within worker; empty request Gram/answer caches'),
        g_policy=dict(flips_first='both warm and cold flips before insertions',
                      seed='paid top64 sparse flip caps only', flip_batch=256, insertion_batch=1,
                      whole_family_bound='min(4 omega, omega(range(q)+2r)^2)',
                      fallback='independent lazy binary triangular pair blocks with diagonal denominator; no H frontier'),
        comparison='Complete-service ratios only for both methods completing the same Q4. Keep all partial costs, unresolved/errors and input units distinct. No ratios against historical environment.',
        verification='Compare all returned accepted edit/version/contract traces with paired H and retained V2. Any genuinely newly completed decision requires independent replay before a completion claim. Global checker scope and interval consistency remain separate.',
        correctness_gates={name:sha(w / 'v3/runs' / name / 'record.json') for name in (
            'resumed_production_certificate_micro','resumed_g_core_checks',
            'resumed_g_mechanism_and_integration','resumed_g_rejection_gate','resumed_a1_verified_log_audit')},
        process_sidecars='v3/runs/hg_XX_<graph>_<stream>_<method>_r0/record.json',
        scheduler_sha256=sha(w / 'v3/control/resumed_budget_run.py'),
        creator_sha256=sha(Path(__file__)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), sha256=sha(output), runtime_commit=commit, jobs=len(jobs))))


if __name__ == '__main__':
    main()
