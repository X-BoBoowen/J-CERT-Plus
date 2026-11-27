# J-CERT+ V4 — shared lightweight measurement and the final H/G method selection

This package is a measurement-policy patch, not unmodified V3. Original V2/V3 records retain their original identities. V4 changes only reporting policy and observation fields; H/G access logic, numerical kernels, resolver, weights, opinions, tolerances, and caching policies remain unchanged.

## Exact versions and dependencies

- V2：`4c2c7b7e84c7c03ee56d45453fea6950c94817bd`。
- V3：`c6b752630bd8dbe1f9fd296a7ed02f249f834575`。
- Actual V4 run: `9d09794b6a6de8a4c97f615abf3f3cae1f458b75`.
- Python 3.12.14; NumPy 2.3.5; SciPy 1.17.0; psutil 7.2.2; threadpoolctl 3.6.0; SymPy 1.14.0. The full platform and source-file digests are in the frozen protocol.
- Retain V2 Code's `baseline_r15/` overlay and native libraries; a Git checkout alone is not a complete environment. Formal workers reject mismatches between the frozen protocol and source, inputs, dependencies, CPU, or hard resource limits.

Exact originals and package digests are listed in `v4/protocols/BASELINE_BINDING.json`. Minimal source restoration uses V2 Code, V3 Code, and this V4 Code; actual six-stream execution also requires V2 Inputs. Historical semantic verification requires the twelve original workers from V3 Evidence; V4 Evidence retains copies of the corresponding records actually used in this round. V2 Evidence's provenance binding is retained without rerunning its historical performance experiments.

## Restore an isolated runtime tree from the exact packages

First safely extract the originals and verify their internal manifests. The paths below specify one concrete layout; downloaded filename suffixes may differ, but digests must match. Every destination directory must be new.

```sh
git clone v2_code/reproduction/RESEARCH_HISTORY_V2.bundle v4_runtime
git -C v4_runtime checkout --detach 4c2c7b7e84c7c03ee56d45453fea6950c94817bd
python v2_code/reproduction/overlay_packages.py v4_runtime J_CERT_Plus_V2_Code_20261126.zip J_CERT_Plus_V2_Inputs_20261126.zip
git -C v4_runtime fetch ../v3_code/runtime_delta/V3_Runtime_Delta.bundle research/v3-value-diagnostics-20261126
git -C v4_runtime checkout --detach c6b752630bd8dbe1f9fd296a7ed02f249f834575
git -C v4_runtime fetch ../v4_code/runtime_delta/V4_Runtime_Delta.bundle research/v4-measurement-20261126
git -C v4_runtime checkout --detach 9d09794b6a6de8a4c97f615abf3f3cae1f458b75
```

V4's `runtime_delta/V4_Runtime.patch` is the same delta for review; `runtime_files/` contains the six files actually modified or added. The Git bundle and patch are alternative restoration methods; do not apply an already applied patch again. `v4/analysis/` contains exact log projection, evidence-gate, and summary code; diagnostic digest traversal is enabled only in the four diagnostic workers.

## Measurement contract

`A2Session(..., measurement_policy="legacy_deep")` retains the original behavior. Formal comparisons explicitly use `light_v1`, reading a fixed number of lengths, array payloads, and existing counters only at the shared request begin/version open/version finish/request end boundaries. Unknown policies are rejected. Unmeasured deep byte counts/legacy high-water marks are recorded as `null`, not 0.

Primary cost is the complete wall time for graph reading/structural preparation plus four actual requests. Opinion/provenance validation and whole-process wall time are listed separately; solve timings are overlapping observations within phases and cannot be added to total cost again. Returned request/preparation costs from incomplete workers represent partial work, not a complete Q4.

RSS/VMS are maxima sampled by the supervisor every 100 ms; worker `ru_maxrss` covers time up to its final read and is not guaranteed to include the peak of final JSON serialization. Array payloads, container entry counts, shared payloads, and process memory cannot substitute for one another or be summed and called an exclusive peak.

## Actual commands and one-time execution rules

The following is the actual command structure; original commands and results are in the budget ledger and `runs/`. After research execution for this round ends, these commands must not be treated as automatic authorization for new batches. Every command must use a new destination and retain formal inputs, order, constraints, and source checks.

```sh
python v4/control/budget_run.py --root v4 --runtime v4_runtime --task hg_light_00_Alpha_slow_H_r0 --category formal --decision 'Prespecified paired H/G cost' --timeout 180 --memory-gib 2 --binding v4/protocols/HG_LIGHT_FROZEN.json --cwd /absolute/workspace/v4_runtime -- python -m src.v3_worker --freeze /absolute/workspace/v4/protocols/HG_LIGHT_FROZEN.json --job 0 --output /absolute/workspace/v4/raw_hg_light/00_Alpha_slow_H_r0.json
```

`control/run_frozen_batch.py --workspace /absolute/workspace` schedules jobs serially in frozen order and does not overwrite attempts already started; if the first repetition fails the behavioral gate, later repetitions do not run. Once the first repetition has passed completely, speed does not determine whether to continue. The original frozen protocol must not be regenerated and misrepresented as the original.

Recompute from records only: `analysis/verify_formal_repeat.py --workspace ... --repeat 0` (and 1, 2) requires a new gate-output path; `analysis/summarize_hg_light.py --workspace ...` outputs complete/partial costs, gates, and the fixed engineering choice. All numerical checks and record analyses are charged through the budget scheduler.

## Change-related checks and actual archive acceptance

Policy tests must explicitly point to the external comparator:

```sh
V4_ANALYSIS_ROOT=/absolute/workspace/v4/analysis python -m unittest discover -s tests -p test_measurement_policy.py -v
V3_ORACLE=/absolute/v3_code/v3/code/exact_oracle.py python -m unittest discover -s tests -p test_v3_g.py -v
```

These commands must also run inside the single-threaded budget scheduler with limits of 120 seconds/2 GiB. A test failure caused by a missing environment variable does not establish a production numerical error. Independent V4 archive acceptance additionally runs miniature tests of diagnostic evidence gates and fixed selection rules; exact method counts, results, and cumulative budgets are in the actual acceptance report. Assertion counts are not independent samples.

Acceptance does not rerun the formal batch or the complete historical test suite, and does not fabricate old missing fixtures. The integer permutation relation for the real Alpha graph is carried forward from the V3 archive; this round has no new real full-catalog numerical replay. Final cache digests from the four diagnostics do not observe all intermediate caches or local-set contents.
