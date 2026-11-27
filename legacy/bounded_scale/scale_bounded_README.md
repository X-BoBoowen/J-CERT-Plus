# Scale experiments with bounded opinion inputs

This study reproduces the comparison at four graph sizes using the historical graph generator and candidate catalog. Seeds and tolerances are in `scale_bounded_protocol.md`. The recorded environment was Windows 10.0.26200, Intel Core i7-12700H, 16 GiB RAM, CPython 3.11.9, NumPy 2.4.6, SciPy 1.17.1, and SuiteSparse 7.12.2. Methods ran in separate processes with one numerical thread on the same machine. GCC 15.2 compiled the native PCG library outside timing. The direct method used the CHOLMOD bridge in UCRT64.

## Recorded results

The table reports median seconds over ten paired repeats. Each size used 32 candidates. All ten runs of all four methods completed 15 steps and selected the same trajectory at each size. At 10k, candidate IDs were 0, 1, 2, 3, 5, 4, then 6 through 14; at the other sizes they were 0 through 14. Independent verification obtained strictly positive winner-gain and winner-margin lower bounds for all 60 decisions; the smallest margin lower bound was approximately 1.485e-6. This is a fixed-catalog study, not an all-nonedge experiment.

| Vertices | PCG Cold | PCG Planner | CHOLMOD Planner | LINVIEW-Delta (registered) |
|---:|---:|---:|---:|---:|
| 10,000 | 0.168 | 0.115 | 0.139 | 0.149 |
| 50,000 | 0.960 | 0.551 | 0.977 | 1.076 |
| 100,000 | 2.296 | 1.298 | 2.381 | 2.697 |
| 500,000 | 13.653 | 7.665 | 29.654 | 30.399 |

At 500k, the median paired PCG Cold/Planner ratio was 1.763 (95% bootstrap interval 1.726-1.799). Median peak process RSS for the four methods was 272, 358, 2,054, and 2,132 MiB, respectively. Including opinion calibration and scaling, median times were 24.30, 18.25, 40.60, and 41.32 seconds.

## Data

- `data/scale_bounded_input_<n>.npz` contains original opinions `s_original`, measured opinions `s`, the scaling factor and divisor, and zero-based candidate arrays `edit_i, edit_j, edit_w, edit_flip`. `fixture(n)` in `tools/scale_views.py` generates the graph.
- `data/scale_bounded_ranges.csv` records graph and candidate counts, ranges before and after scaling, and counts of out-of-bound coordinates.
- Each row of `data/scale_bounded_<n>.csv` is an actual timed process. `rep,position,method` identify the repeat block, execution order, and method. `sequence` records committed candidate IDs; together with the NPZ it reconstructs each edge and edit type. `seconds` includes numerical initialization and the query; `setup_seconds` is its initialization component. `inclusive_seconds` additionally includes opinion calibration and scaling. Input comparison is recorded separately. `peak_mib` is peak process resident memory.
- `data/scale_bounded_summary.csv` and `figures/scale_bounded.pdf` summarize ten repeats for four methods. Times are seconds and error bars are interquartile ranges. Paired speedups divide PCG Cold by the corresponding method within the same repeat block.
- `data/scale_bounded_verified.csv` records independently verified winner-gain intervals and margin lower bounds by size, trajectory, and step. Steps and candidate IDs are zero-based.

## Verification and retiming

Install dependencies from `requirements.txt` and run these commands from this study's directory. The independent verifier reconstructs the signed graph and recomputes bounds.

```bash
python tools/scale_bounded.py --check
python tools/scale_bounded_verify.py
python tools/scale_bounded_report.py --check
```

Run `python tools/scale_bounded_report.py` to regenerate the plot. For retiming, work in a copy and move aside the four existing raw timing CSVs: the driver skips successfully recorded repeats. On Windows, add SuiteSparse's bin directory to PATH and set `JCERT_CHOLMOD_LIB` to the absolute path of the compiled bridge. Build `tools/cholmod_bridge.c` in MSYS2 UCRT64 with:

```bash
gcc -O3 -shared tools/cholmod_bridge.c -I/ucrt64/include/suitesparse -L/ucrt64/lib -lcholmod -o tools/jcert_cholmod.dll
```

Ensure gcc is on PATH, then run sequentially in the same environment:

```bash
python tools/scale_bounded.py --inputs
python tools/scale_bounded.py --run
python tools/scale_bounded_verify.py
python tools/scale_bounded_report.py
```
