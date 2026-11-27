# J-CERT+

**Certified Full-Catalog Intervention Selection in Signed Graphs**

Bowen Xue, Tianyi Chen, and Yuze Xia — Hangzhou Normal University, China.

J-CERT+ separates reusable numerical response summaries from the evidence needed to cover the current legal edit catalog. It maintains endpoint Gram views of size at most 2 × 2 and uses certified gain bounds to focus response computation on contenders. In the structured mixed catalog, row-frontier and block-cover access avoid eagerly generating the complete candidate directory.

This repository accompanies the manuscript and provides the frozen V4 numerical implementation, exact experimental inputs, raw observations, and independent verification records.

## Start here

- [Paper and LaTeX source](paper/README.md)
- [Reproduction guide](docs/REPRODUCIBILITY.md)
- [Experiment-to-evidence map](docs/EVIDENCE_MAP.md)
- [Dataset definitions and input provenance](docs/DATASETS.md)
- [Scientific versions and historical studies](docs/HISTORY.md)
- [Recomputed main results](results/summary/headline_results.json)

## Main experimental findings

The main structured catalog contains every root nonedge as a uniform-weight positive insertion and every negative root edge as a magnitude-preserving flip. The Bitcoin Alpha and OTC projections contain 7,140,884 and 17,271,859 logical candidates, respectively. Q4 comprises four opinion requests with structure preparation charged once.

| Evidence cohort | Complete matched blocks | Recorded complete-service comparison |
| --- | ---: | --- |
| Fixed-input revision retest (12 streams; original instrumentation) | 36 | H: 3.59× faster than D; 14.26× faster than S |
| Post-freeze V2 confirmation (5 of 6 streams; original instrumentation) | 15 | H: 3.79× faster than D; 14.16× faster than S |
| V4 H/G comparison (same observed V2 streams; common `light_v1`) | 15 | G: 12.56% less aggregate time than H; faster in 14/15 complete pairs |

These ratios are sums of matched complete-service times, not ratios of displayed medians. The remaining Alpha/correlated V2 stream has a shared unresolved request. All raw results, including the incomplete runs, are retained. On fixed-input revision retests, median peak process RSS falls from approximately 1,217 MiB (D) to 125 MiB (H) on OTC. The V4 study does not remeasure D/S, so ratios across measurement policies are not composed.

## Reproduce the stored results

Python 3.10+ and the standard library are sufficient for the archive checks. No solver or performance experiment is started by this command.

```bash
python3 scripts/verify_artifact.py --output-dir ../jcert-audit
```

The command validates all seven original scientific archives, the 178 original V4 source files, and the supplied inputs. It recomputes complete-service results, the G mechanism statistics, and the stored interval containment checks for 30 previously replayed states (90 G observations). It refuses to overwrite an existing output directory. Use `--validate-only` to skip the record analyses.

## Inspect or run the implementation

The numerical code under `runtime/` preserves the frozen scientific implementation; accompanying documentation is presented in English. The original archives additionally preserve the exact Git history and version deltas.

For a Linux x86-64 environment, the recorded Python version is 3.12.14. Install the pinned dependencies in a separate virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_test.py
```

The smoke test runs a small selection of existing correctness tests, not the full benchmark suite. For an explicit lightweight configuration, use the following example from the `runtime/` directory:

```python
import numpy as np
from scipy import sparse
from src.a2_session import A2Session

W = sparse.csr_matrix([[0., -1., 0.], [-1., 0., -1.], [0., -1., 0.]])
s = np.array([65536, -19661, -32768], dtype=np.int64)
result = A2Session(W, "H", measurement_policy="light_v1").request(s, budget=2, delta=0)
print(result["status"], result["commits"])
```

The opinion vector uses the implementation's exact integer numerators with denominator 65,536; the example is a demonstration, not a reported benchmark. Use `"G"` for the alternative block cover. The compatibility constructor default is `legacy_deep`; explicitly select `light_v1` for H/G. The frozen main service uses `delta=0`, accepts only certified decisions, and may return `unresolved` rather than guess a winner.

To reconstruct a Git checkout with the original V4 identity and required overlays:

```bash
python3 scripts/restore_frozen_runtime.py ../jcert-runtime --version V4
```

Full historical measurement runners enforce their recorded environment, resources and input bindings. See the reproduction guide before running any of them. Re-execution on another machine is not a new observation of the original frozen timings.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `runtime/` | Current implementation, tests, required historical numerical dependencies, graphs and opinion inputs |
| `experiments/` | Frozen protocols, analysis scripts and original runner documentation |
| `artifacts/frozen/` | Seven immutable V2–V4 code, evidence and input archives, with original internal manifests |
| `verification/` | Archive-based checks, original independent replay reports and bounded diagnostic records |
| `results/` | Compact, browsable summaries and selected data tables |
| `legacy/` | Earlier fixed-catalog studies and corrected bounded-input scale study; separate experimental identities |
| `paper/` | Manuscript, current figures, references and build inputs |
| `scripts/` | Portable verification, historical source restoration and small tests |
| `docs/` | Evidence mapping, datasets, history and reproduction instructions |

## Citation and availability

Please cite the manuscript *J-CERT+: Certified Full-Catalog Intervention Selection in Signed Graphs* and the relevant original dataset sources. See [CITATION.cff](CITATION.cff). The repository is <https://github.com/X-BoBoowen/J-CERT-Plus>.
