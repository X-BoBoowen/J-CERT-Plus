# Reproducibility

## 1. Evidence-only checks (recommended first)

From the repository root, run:

```bash
python3 scripts/verify_artifact.py --output-dir ../jcert-audit
```

Use a fresh path outside the repository. Python 3.10+ is sufficient. The three sub-analyses read stored JSON/CSV, hash the exact archives, parse NPY payload headers, and compare stored rational interval endpoints. They do not import production code, run linear solvers, reconstruct numerical parent matrices or replay a decision. See `verification.json`, `timing/headline_results.json`, `g_mechanism.json` and `g_direct_provenance.json` in the new output directory.

The archive-analysis entry points use the seven scientific archives directly. Their counts, bounds, and request relations match the stored audit records. See `verification/README.md` for the scope of these checks.

## 2. Run a small correctness check

The repository exposes the current code directly in `runtime/`. All 178 files listed in `runtime/FROZEN_SOURCE_SHA256.json` match V4. The numerical runtime and its historical report generator retain their original bytes so that frozen experiment bindings remain valid. Public documentation is in English; original archived records retain their original language. The legacy source manifests cover unchanged scientific files and exclude the translated legacy documentation. The code includes original Linux x86-64 shared objects required by the historical source binding. The pinned environment is in `runtime/requirements-research.txt`. Use the virtual-environment setup in the root README, then:

```bash
python scripts/smoke_test.py
```

The selected tests check catalog coverage, G's block partition and bounds, an exact small-graph oracle, and the measurement policy. These are limited tests, not an independent rerun of all experiments.

## 3. Restore original commit identities

The browseable `runtime/` is a source snapshot, not a checkout of the historical Git object. To satisfy the original commit checks, use:

```bash
python3 scripts/restore_frozen_runtime.py ../jcert-runtime --version V4
```

Versions V2 and V3 can also be selected. The script uses the V2 Git bundle, verified V2 overlay and the original V3/V4 Git deltas. The destination must be new. It performs no benchmarks. It checks the resulting commit and all 178 listed source files when restoring V4. Do not apply the patches again after checkout. Keep archives immutable.

The V2 native `.so` assets are Linux x86-64 binaries. Their matching C sources are included. See [the original V2 reproduction guide](original_v2/README_REPRODUCTION.md) for compiler commands, dependency information and binding details. Recompiled native libraries need not have identical bytes; do not claim they satisfy the original frozen hashes. The recorded main environment used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, psutil 7.2.2, threadpoolctl 3.6.0, SymPy 1.14.0 and Matplotlib 3.10.8.

## 4. Historical performance workers

Read the original protocol and version-specific instructions first:

- `experiments/v2/PRE_REGISTRATION.json` and `experiments/v2/PROTOCOL.json`.
- `experiments/v3/README_V3_REPRODUCE.md` (V3 diagnostics and H/G).
- `experiments/v4/README_V4_REPRODUCE.md` (common `light_v1` H/G).

These directories are browsable copies of original scripts and protocols. For a strict historical rerun, restore the matching runtime, unpack the matching Code and Evidence ZIP into separate clean locations, and follow their original workspace-layout instructions. The recorded worker checks environment, source, input, CPU, resource and protocol identities; it may reject a different environment. Preserve that refusal rather than disabling frozen checks and labelling the run as the original experiment. Use a separate output location and do not overwrite raw files. The supplied evidence archives include complete schedules, inputs, source bindings, partial/unresolved observations and actual logs.

For the main all-nonedge service, a request supplies an int64 opinion numerator vector scaled by 65,536. The root graph is a symmetric sparse matrix with signed half-integer weights. Use `A2Session(W, "H", measurement_policy="light_v1")` for the recommended lightweight configuration; G is an alternative. D/S use the original supported measurement contract. The legacy constructor default is `legacy_deep`.

## 5. Independent numerical checker (separate optional execution)

Original checker source: `runtime/src/a2_checker.py`. Original V2 reports: `verification/reports/v2_alpha_full_catalog.json` and `v2_otc_full_catalog.json`. The checker reconstructs the legal catalog and uses independent residual/energy evidence. A fresh check runs numerical work; it is not part of archive-only verification. Use the V2 checkout for an original V2 trace. Read `python -m src.a2_checker --help` in that checkout to supply the graph, opinions, trace, request number and a new output path. Maintain explicit limits. A full-domain replay can be expensive; the existing records certify the two preselected 15-step requests.

The archive-linked G check proves inclusion of the existing independent intervals in G's reported intervals on the same 30 states, 90 observations across three repeats. It is not a new full-domain replay, a separate validation of every G block cut, or proof that the service completes all exact requests.

## 6. Earlier studies

See [legacy/README.md](../legacy/README.md). Fixed-catalog maintenance and numerical stress evidence use their own Windows/native or Python environments and protocols. They are not pooled with the main Linux all-nonedge timing experiments. The earlier unbounded opinion scale observations are historical; the corrected bounded scale study is separate. No incompatible timing ratios are formed.
