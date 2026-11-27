# J-CERT+ V3: code, dependencies, and reproduction entry points

## Version identity

This package is a diagnostic/minimal-control extension of V2, not unmodified V2.

- Baseline: `4c2c7b7e84c7c03ee56d45453fea6950c94817bd`.
- Actual H/G run: `c6b752630bd8dbe1f9fd296a7ed02f249f834575`.
- The historical overlay in V2 Code and native `index.so` and `screen.so` must also be applied; checking out the commit alone does not provide a complete environment.
- `runtime_delta/V3_Runtime_Delta.bundle` uses that V2 commit as its exact baseline; `runtime_delta/V3_Runtime.patch` is a review copy of the same delta; `runtime_files/` provides the six files actually modified or added.
- `v3/code/` contains an independent physical rational oracle and construction checks; it does not import the production gain kernel.
- Actual environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, psutil 7.2.2, threadpoolctl 3.6.0, SymPy 1.14.0. Full platform, dependency, and source-file mappings are in `v3/resumed/protocols/HG_FROZEN.json`.

## Required originals

Downloaded filename suffixes may differ; identify files by digest. Do not replace missing materials with fabricated fixtures.

| Original | SHA256 |
| --- | --- |
| J_CERT_Plus_V2_Code_20261126.zip | `4d9d96167fdcaee440aced8b280a36118e0d142bc5d8550707316311d8433dfb` |
| J_CERT_Plus_V2_Evidence_20261126.zip | `1e98f70bef0c22fd2f6d8b528490ab42ceedf75c7427f292d9b58965c86b9fcf` |
| J_CERT_Plus_V2_Inputs_20261126.zip | `b643dde2a09cc5c6866a1295d9923480f53b5a6b426beb2b0c557949e4346308` |
| J_CERT_Plus_PVLDB_Source_20261126.zip (manuscript verification) | `1fed7ed8013f77174d782f02dd648d61ecc262eb06593091b7ec560e098e4deed` |

This package does not nest duplicate historical ZIPs or large NPZs. Checks of the originals' CRCs, paths, member byte counts, and digests are preserved in Evidence at `v3/resumed/inspection/archive_verification.json`.

## Assemble the runtime tree

Extract this V3 Code into a separate directory; first safely verify and extract V2 Code into `v2_code_package/`. The V2 overlay program validates every archive member and refuses to overwrite different content. The original filenames below may be replaced by downloaded names with identical digests.

```sh
git clone v2_code_package/reproduction/RESEARCH_HISTORY_V2.bundle v3_runtime
git -C v3_runtime checkout 4c2c7b7e84c7c03ee56d45453fea6950c94817bd
python v2_code_package/reproduction/overlay_packages.py v3_runtime J_CERT_Plus_V2_Code_20261126.zip J_CERT_Plus_V2_Inputs_20261126.zip J_CERT_Plus_V2_Evidence_20261126.zip
git -C v3_runtime fetch ../runtime_delta/V3_Runtime_Delta.bundle research/v3-value-diagnostics-20261126
git -C v3_runtime checkout --detach c6b752630bd8dbe1f9fd296a7ed02f249f834575
```

The overlay must precede the V3 checkout. `git bundle verify` requires the exact baseline object to be present. Do not remove the check for the missing old A1 fixture or fabricate that fixture; this round validates only paths related to the changes.

## Actual commands, budgets, and status

Evidence's `v3/budget_ledger.jsonl` and `v3/runs/*/record.json` preserve each actual subprocess's complete command, working directory, source/input/protocol, single-thread settings and CPU affinity, RLIMIT_AS, process wall time, and exit status. The root scheduler runs all numerical tasks serially; outer waiting is not charged again.

Research computation for this round has stopped. The entry points below document exact reproduction; they do not automatically authorize new batches, seed changes, or overwriting original observations. Reproduction should use a separate output directory and an explicit budget; when the environment differs, retain rejection by the frozen checks instead of deleting those checks.

### Original formal-worker command structure

```sh
python v3/control/resumed_budget_run.py --root v3 --runtime v3_runtime --task hg_00_Alpha_slow_H_r0 --category small_timing --decision 'Reproduce the frozen paired worker' --timeout 180 --memory-gib 2 --binding v3/resumed/protocols/HG_FROZEN.json --cwd /absolute/workspace/v3_runtime -- python -m src.v3_worker --freeze /absolute/workspace/v3/resumed/protocols/HG_FROZEN.json --job 0 --output /absolute/workspace/v3/resumed/raw_hg/00_Alpha_slow_H_r0.json
```

`--root` must point to the V3 directory containing the protocols and budget ledger. Existing task names or outputs are not overwritten. The order of jobs 0–11, the six existing streams, Q4, and the cap of 15 commits per request are frozen. Primary timing retains its original scope: graph reading/preparation plus actual requests; process wall time and input binding are recorded separately. Original numbers including reporting overhead have not been replaced.

### Diagnostic and verification entry points

The following scripts are in `v3/resumed/analysis/` and are also run through the capped scheduler:

- `production_certificate_checks.py --runtime v3_runtime --oracle v3/code/exact_oracle.py --output <new.json>`: 174 miniature production-boundary checks. weight64 is validated through the general local entry point; A2's rejection of that weight is recorded separately.
- `alpha_relation_diagnostic.py --workspace <workspace> --protocol <ALPHA_DIAGNOSTIC.json> --output-dir <new-dir> --stage structure`: independently reconstructs the real parent state and performs bounded exact permutations. Existing results were already exact_equal; no second numerical diagnostic process was started in this round.
- `hg_analyze.py --freeze <HG_FROZEN.json> --raw-dir <raw-dir> --out-dir <new-dir>`: summarizes existing records without running the service.
- `check_hg_structure.py --workspace <workspace> --output <new.json>`: independently checks actual versions, legality, and declared witnesses; this is not a full-catalog numerical replay.
- `profile_memory_accounting.py --workspace <workspace> --protocol <MEMORY_PROFILE.json> --output <new.json>`: the sole fixed-stream reporting-cost diagnostic. Reporting remains enabled; the remainder after subtraction is not timing without observation overhead.

Source analysis requires safely extracting the original Code/Evidence/Inputs into `v2_received/code`, `v2_received/evidence`, and `v2_received/inputs`, respectively; the manuscript goes in `v2_received/paper`. The task specification and frozen selection table have been retained.

### Acceptance of a copy extracted from the actual archive

```sh
python v3/control/fresh_import_checks.py --runtime /absolute/fresh/v3_runtime --code-root /absolute/fresh/code --freeze /absolute/fresh/code/v3/resumed/protocols/HG_FROZEN.json --output /absolute/new-check-report.json
```

Use the same scheduler to impose single-thread execution and limits of 120 seconds/2 GiB. This entry point verifies all frozen source mappings, imports the production code and independent checker, and runs 17 independent mathematical test methods and 11 G-related methods without rerunning performance batches. The actual delivery used this process to validate the extracted contents of the real Code ZIP; logs are in Evidence.

## Deliberately excluded extensions

The shared gain, residual, transport, and tie resolver were not changed. No online composite-permutation completion procedure, general graph isomorphism, arbitrary-precision platform, shared response-region system, GPU backend, or distributed backend was implemented; Slashdot was not rerun. G's internal block objects are not a public certification API. The independent checker's non-point interval intersection checks were not upgraded to proofs of all endpoints.
