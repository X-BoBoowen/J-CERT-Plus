# J-CERT+ V2 research package

## What is frozen, and what reproduction establishes

The executable V2 code is the frozen commit `4c2c7b7` (its full object identity is
recorded in PACKAGE_MANIFEST.json).  The historical exact-tie runtime was
`531522d09b962286afdcde9038ad1a1b0b5d90fa`; the subsequent retained-input binding
was `9de16a76f5cddcdd4aba0c82d555f2ed42e46fe2`.  The latter added an input manifest,
not a new numerical runtime.  Original A2 code is identified by commit `365bc8e`.
Exact binary-capable Git patches from both code baselines and a full history
bundle are included in `reproduction/`.

Re-executing the supplied opinions reproduces the same inputs.  It is not a new
unseen confirmation.  The retained twelve-cell retest is historical evidence.
The seven V2 opinion files contain six small-graph streams and one Slashdot
stream; each small stream has four requests, with three method arms and three
timing repetitions.  Repetitions do not create independent opinion streams.
Consult the frozen protocol and final evidence for status, resource limits,
unresolved outcomes, and the actual population of completed jobs.

The V2 input-boundary change admits `n <= 82140`.  The A2 numerical backend remains
the existing sparse SuperLU factor with low-rank updates and the existing fixed
directional scoring path.  This is not a new preconditioner or solver.  Neither
the code package nor these instructions assert universal request completion.

## Three separate, root-relative packages

Extract Code, Evidence, and Inputs into three fresh directories if they are to be
inspected independently.  Each ZIP is independently extractable and has its own
uniquely named manifest.  For execution or source-bound record checking, combine
them in a new directory using `reproduction/overlay_packages.py`.  Every payload
path is relative to the research root: `src/`, `baseline_r15/`, `data/`, `raw/`,
`protocol/`, `v2/`, and audit directories retain their meanings.  Identical
cross-package files may be shared; a different file at an existing path is an
error.  Do not overwrite the immutable originals or modify the frozen inputs.

Code contains the full current `src/`, every asset actually named by
`v2/CODE_FREEZE.json:source_files_sha256` (including legacy Python, C, and exact
native libraries), available tests and proof/documentation sources, pinned
requirements, exact patches, and the Git bundle.  Evidence contains the retained
trace directories and V2 outputs plus relevant audits.  Input arrays are in the
Inputs ZIP; Evidence deliberately contains no NPZ members.  Internal manifests
bind all payload members by byte length and SHA-256.  A manifest cannot hash
itself; PACKAGE_MANIFEST.json additionally binds each internal manifest and each
finished ZIP, and reports actual-archive verification results.

## Dependencies and native builds

Use Python 3.12 and the top-level `requirements-research.txt` pins, which are the
research environment specification.  Legacy copied requirements describe older
environments and do not supersede these pins.  The recorded environment used
NumPy 2.3.5, SciPy 1.17.0, psutil 7.2.2, threadpoolctl 3.6.0, SymPy 1.14.0, and
Matplotlib 3.10.8.  Install into a separate environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
```

These pins support source inspection and imports in a separate environment.
Strict experiment replay additionally checks the recorded Python build, resolved
interpreter path, operating-system platform, and package versions.  The frozen
runner also uses CPU affinity 0 for workers and 8 for its supervisor.  A machine
that does not match those bindings may import the code successfully while the
strict worker correctly refuses replay.  Package verification is performed in
the recorded environment; it is not a claim of unchanged performance portability.

The frozen `.so` files are Linux x86-64 ELF assets.  The A2 H/Stream paths do not
call the old native directory interface, but those binaries remain in the
source binding and are retained byte-for-byte.  The source and exact commands
used for fresh verification appear in PACKAGE_MANIFEST.json.  Equivalent build
commands, with paths supplied by the source manifest, are:

```sh
mkdir new-build
legacy=baseline_r15/experimental_records/evidence_supplement_20261124/source/legacy_v6/tools
cc -O3 -fPIC -shared "$legacy/promotion_screen.c" -o new-build/screen.so
cc -O3 -fPIC -shared "$legacy/promotion_index_profile.c" -o new-build/index.so
```

Build into a new directory.  Do not replace a frozen binary and then claim its
original source hash.  Rebuilt native binaries need not be byte-identical across
toolchains.  SciPy's SuperLU and its numerical-library dependencies are provided
by the pinned Python environment; the old C interfaces are separate components.
Other C files named by the freeze are retained because the source binding covers
the historical tool directory.  The verification builds the two C sources for
the two frozen libraries; it does not activate or build dormant alternative
factorization/preconditioning backends.

## Reproduction entry points and resource discipline

`python -m src.v2_runner --help` describes the frozen runner commands.  Read
`v2/PRE_REGISTRATION.json` and `v2/PROTOCOL.json` before running anything.  The
runner verifies frozen source, environment, protocol, roster, seed, and input
bindings.  A reproduction must preserve these checks and its hard resource
limits.  Do not invoke preparation to overwrite a frozen study or draw replacement
inputs after seeing an outcome.  Use a separate destination for reproduced
results.  Inspect the parser and available destination flags before invoking a
worker; the supplied raw evidence is an immutable observation, not a workspace.

For a same-input rerun in a matching environment, create another fresh directory
and overlay Code plus Inputs there, without the Evidence ZIP.  It then has the
frozen V2 manifests and inputs but no prior `v2/raw_a` or `v2/raw_b` directories.
From that new research root, `python -m src.v2_runner run-a` and
`python -m src.v2_runner run-b` are the frozen batch entry points.  They enforce
the original registration and refuse to overwrite an already-started batch.
Running either command is a new execution of the supplied inputs, not part of
the package builder's verification.  An individual recorded job can instead be
run with `worker --job path/to/recorded.job.json --output path/to/new-result.json`;
the worker requires a new output path.  Keep such results separate from the
delivered observations and report their actual environment and resource status.

Historical retest records bind the historical source hashes, not the V2 source
hashes.  Use the supplied history bundle to materialize the recorded historical
commit in another directory when checking those old source bindings.  The
unchanged `baseline_r15/` source/native assets from Code and the corresponding
Inputs/Evidence trees supply the non-Git assets.  Do not substitute the V2
`src/a2_session.py` for its historical version in an old-source check.  The exact
patches make the version difference reviewable.

### Rechecking the delivered V2 observations

The post-measurement wrapper `validation/v2_validate.py` requires a real Git
checkout at the exact frozen V2 commit.  Its default runtime is a sibling
`runtime/` directory beside its `validation/` directory.  A plain ZIP extraction
is not such a checkout.  The following layout makes both requirements explicit;
run it from the directory containing the three ZIPs, using new directory names
and the matching recorded Python environment:

```sh
mkdir package-code
python -m zipfile -e J_CERT_Plus_V2_Code_20261126.zip package-code
git clone package-code/reproduction/RESEARCH_HISTORY_V2.bundle verify-runtime
git -C verify-runtime checkout --detach 4c2c7b7e84c7c03ee56d45453fea6950c94817bd
python package-code/reproduction/overlay_packages.py verify-runtime J_CERT_Plus_V2_Code_20261126.zip J_CERT_Plus_V2_Inputs_20261126.zip J_CERT_Plus_V2_Evidence_20261126.zip
mkdir -p verify-runtime/logs
python -c "from pathlib import Path; Path('verify-runtime/logs/.measurement.lock').open('xb').close()"
recheck_root="$PWD/verify-runtime"
```

The empty lock is created exclusively in this new, inactive checkout.  The
validator opens the existing lock read-only and acquires it nonblocking before
reading evidence.  Do not create or replace it in an active measurement tree.
The overlay preserves the original relative source, graph, opinion, and record
paths and refuses a nonidentical existing file.  The checkout preserves the Git
HEAD identity required by the validator.

Always pass `--runtime` explicitly for this layout.  Also pass a new `--output`
JSON path **inside this copy's `validation/` directory**; arbitrary external
output directories are deliberately rejected.  Aggregation reserves matching
`_cells.csv` and `_requests.csv` names, so those must also be absent:

```sh
python "$recheck_root/validation/v2_validate.py" aggregate --runtime "$recheck_root" --output "$recheck_root/validation/recheck_aggregate.json"
python "$recheck_root/validation/v2_validate.py" chains --runtime "$recheck_root" --output "$recheck_root/validation/recheck_parent_witness.json"
```

An optional independent numerical recheck is a separate operation.  Run the two
prespecified samples serially, with new output names, and report its cost apart
from algorithm service:

```sh
python "$recheck_root/validation/v2_validate.py" full-catalog --runtime "$recheck_root" --sample Alpha --max-seconds 1800 --output "$recheck_root/validation/recheck_full_catalog_Alpha.json"
python "$recheck_root/validation/v2_validate.py" full-catalog --runtime "$recheck_root" --sample OTC --max-seconds 1800 --output "$recheck_root/validation/recheck_full_catalog_OTC.json"
```

The delivered `a_aggregate.json`, `a_parent_witness.json`, and selected numerical
reports remain unchanged.  The wrapper refuses existing output files; use a new
basename or another fresh checkout for a later recheck.  These instructions do
not redraw opinions, change runtime/protocol bytes, or turn same-input replay
into unseen confirmation.  Packaging checks the layout and imports only; it
does not execute these aggregation, chain, or numerical commands.

The package builder itself performs no request, root factorization, historical
replay, or benchmark.  Its fresh-directory verification checks the actual ZIPs,
CRC, safe paths, member counts, bytes, hashes, Python compilation and real module
imports, frozen native-library loading, isolated native compilation, and history
bundle availability.  Exact results are in PACKAGE_MANIFEST.json.

## Test scope and a known legacy fixture limitation

The pre-freeze targeted run reported 117 passing tests, including nine V2 tests.
The prior full run executed 202 tests, with 201 successful tests and one legacy
error because the retained
`A1_CONFIRMATION_PROTOCOL` fixture is absent.  The package does not fabricate that
old protocol.  Relevant tests are retained for transparency, but this package
does not claim that the entire inherited suite passes.  Some integration tests
also need the separate Evidence and Inputs packages overlaid at their original
relative paths.  Fresh package verification is narrower than a complete test
suite and does not replay numerical experiments.  Consult the included validation
logs and TEST_SCOPE.txt for the exact retained scope and any additional limits.
