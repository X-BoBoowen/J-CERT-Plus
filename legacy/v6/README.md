# J-CERT+: historical directional certification, full-graph promotion, and demand-gated planning

This directory preserves the sixth-edition studies. Reported timings came from actual execution records and all six graphs used full operators. Alpha, OTC, and Slashdot originate from SNAP; Conflict, Bio, and Fin are fixed generated topologies. Small rational cases support algebraic regression tests. These historical studies have separate experimental identities from the current manuscript's all-nonedge experiments.

The old manuscript and its build assets are not included here. Table and figure numbers below refer to that historical manuscript. All-in-one paper checks may require files outside this subset; use individual experiment scripts for numerical work.

## Dependencies and reruns

The recorded Python version was 3.11. From this directory, install dependencies with `python -m pip install -r requirements.txt`; Windows can use `py -3.11` in place of `python`. `tools/reproduce.py --check` describes the original offline checks of graph sizes, conditions, winners, solve counts, versions, schedules, integer witnesses, and rejection of damaged logs. `tools/reproduce.py --rerun` rebuilds and replaces data files: run in a separate copy to retain supplied timings. Full runs require loadable SuiteSparse CHOLMOD with its Modify module. Slashdot candidates and dynamic factors require several GB of available memory. Consult the driver for individual commands and ordering.

The recorded SuiteSparse version was 7.12.2, with memory accounting in `tools/cholmod_bridge.c`. On Linux with development headers:

```bash
cc -O3 -fPIC -shared tools/cholmod_bridge.c -I/usr/include/suitesparse -lcholmod -o tools/libjcert_cholmod.so
```

On macOS use the corresponding header and library paths and output `tools/libjcert_cholmod.dylib`. In Windows MSYS2 UCRT64:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-suitesparse
gcc -O3 -shared tools/cholmod_bridge.c -I/ucrt64/include/suitesparse -L/ucrt64/lib -lcholmod -o tools/jcert_cholmod.dll
gcc -O3 -shared tools/scale_symbolic.c -I/ucrt64/include/suitesparse -L/ucrt64/lib -lcholmod -o tools/scale_symbolic.dll
```

SuiteSparse, OpenBLAS, and dependencies must be on the dynamic-library search path when Python starts. On Windows add UCRT64's bin directory to the current process PATH. `JCERT_CHOLMOD_LIB` can specify a bridge at another location. The ABI includes cloning of independent mutable factors; compile the supplied C source. `tools/scale_symbolic.c` reads CHOLMOD symbolic-analysis results. In PowerShell:

```powershell
$env:JCERT_CHOLMOD_LIB = (Resolve-Path tools/jcert_cholmod.dll).Path
$env:JCERT_SCALE_SYMBOLIC = (Resolve-Path tools/scale_symbolic.dll).Path
```

See `DEPENDENCIES.txt` for full dependency details.

## Experiment map

| Historical item | Programs in `tools/` | Main records in `data/` or `datasets/` |
|---|---|---|
| Table III: full-catalog row endpoint bounds, drift index, transport | `promotion_v4.py --edition 6`, `promotion_strict.py`, `promotion_table_v6.py` | `promotion_v6_*`; fifth-edition same-input `*fine*` staged-screening controls |
| Table IV and Fig. 2: routing and external methods | `planned_report.py`, `direct6_matrix.py`, `native_planner.py`, `linview_delta.py`, `full_quad.py` | `persistent_batch_*`, `persistent_native_*`, `direct6_slots_*`, `linview_registered_persistent_*`, `linview_persistent_*`, `persistent_quad_*` |
| Fig. 3: four complete queries on the same persistent contender catalog | `scale_frontier.py --v6`, `native_scale_frontier.py` | `scale_frontier_v6_*`, `scale_social_v5_*` topology and symbolic-analysis records |
| Table V: directional posterior bounds and reanchoring | `flip_stress.py`, `directional_stress.py`, `margin_stress.py` | `flip_stress_*`, `directional_stress_*`, `margin_stress.csv`, `margin_path.csv` |
| Table VI: shared prefixes, deep DFS, two-step beam search | `branch_queries.py --move`, `sparse_search.py`, `sparse_traps.py`, `sparse_search_report.py`, `sparse_search_verify.py` | `branch_move_*`, `sparse_deep_*`, `sparse_trap_cohort_*` |
| Independent integer replay of three main trajectories | `standalone_verifier.py`, `main_witness.py`, `main_witness_emit.py` | `main_witness_universe6`, `main_witness_persistent6`, `main_witness_branch` timing JSON and proof vectors |

### Full-catalog bounds and timing

The full endpoint-bound mode computes per-vertex residual upper bounds at each version. Signed diagonal dominance converts these to equilibrium-coordinate error enclosures; accessed candidates receive the bounds of Theorem 3. F and T enumerate every remaining candidate. D selects cold index records using Theorem 3B cut points, then applies the same calculations. F computes responses cold; T and D transport local Gram views across versions. S searches 64 static root candidates. All four use residual-verified Woodbury state updates.

F/D compares a complete cold scan against indexing plus transport; T/D compares indexing against scanning with transport on both sides. Ten-query ratios include one shareable graph, opinion vector, catalog, root state, and necessary index and bindings, plus ten measured online queries with reset cursors. Staged screening stops when coarse bounds eliminate a candidate and applies fine screening to survivors. Boolean flags and inverted lists are cursor-specific; repeated candidates are refined once. A/W/R count promotions, out-of-view winners, and those winners with certified strictly negative root gains, respectively.

### Common backend, external methods, and routing

Table IV's six left-side combinations retain paired timings with the same sparse backend. J on the right is a separately timed deployment plan. CHOLMOD symbolic analysis predicts root-factor nonzeros; the rule selects a backend using 12 bytes per entry and a 256 MiB limit. Registered operators reuse these statistics; symbolic-analysis costs for new operators are recorded separately. Slashdot uses fused native Jacobi-CG; the other five graphs use a direct planner with fixed CSR edit positions. All external ratios use the rule-selected J as denominator.

Registered LINVIEW-Delta declares the complete candidate projection table as output. It initializes projections from checked root responses, applies low-rank increments to all registered rows after commits, and scores all active candidates in batches. The lazy variant retains global factor pairs and projects unconsumed increments on demand. Alpha and Fin also retain registered-update/screened-read combinations separating maintenance from reading. Both main variants locally implement the original low-rank increment rule, with declared computation and certification included in timings. Original paper: https://doi.org/10.1145/2588555.2610519.

### Scale curves and search consumers

The four curves use the same multiscale signed networks, 32 candidates per graph, and a 15-step budget. Root gains are calibrated as four groups of eight contenders. The largest graph has 500,000 vertices and 1,644,524 undirected edges. Query timing includes numerical initialization, decisions, and commits; input calibration is recorded separately. Peak RSS covers input construction and the entire query. The direct method reserves structural positions for 16 possible insertions. The two CG curves use the same native single-thread sparse kernel; SciPy results remain sensitivity controls. For the bounded-opinion model, use the corrected study in `../bounded_scale/` instead of these earlier unbounded-input observations.

Table VI separates shared structure, root factor, private mutable state, online query, first complete query, and ten-query costs. The online phase begins at a reusable root and includes cache reset, certification, search, numerical updates, and rollback. Each of two graphs has ten 15-edit deadlock sets and ten two-edit synergy-trap catalogs. Success means complete safe deployment for the former and reaching the exhaustively verified budget-two optimum for the latter. Within-group median gains and a separate three-edit reference instance are labeled separately. Exhaustive enumeration covers 121 sets and 210 ordered pairs per trap catalog.

### Independent checks

After compiling the bridge and installing dependencies, the numerical replay entry points are:

```bash
python tools/standalone_verifier.py --main universe6
python tools/standalone_verifier.py --main persistent6
python tools/standalone_verifier.py --main branch
```

Full-catalog proof vectors are in `main_witness_clamp.npz`; persistent-catalog vectors are in `main_witness_persistent5.npz`. These inputs match their historical queries. The sixth-edition checker binds them to the declared runs. Raw repeats, numerical counts, medians, quartiles, and interval summaries are retained. Remeasured timings depend on hardware and execution environment.
