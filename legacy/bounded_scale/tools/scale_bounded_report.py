"""Validate and summarize the separate bounded-opinion four-arm experiment.

Ratios are medians of within-repetition cold/arm ratios. Percentile bootstrap
intervals resample entire matched repetition blocks (20,000 draws, seed 651).
No timing trials are launched here. Different certified trajectories are kept
visible rather than silently replaced by an assumed order.
"""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("cold", "planner", "cholmod", "registered")
SIZES = (10000, 50000, 100000, 500000)
LABELS = ("PCG Cold", "PCG Planner", "CHOLMOD Planner", "LINVIEW-Delta")
METRICS = ("seconds", "peak_mib", "actual_rhs", "setup_seconds",
           "calibration_seconds", "normalization_seconds", "inclusive_seconds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def actual_rhs(row):
    """Count solves, avoiding duplicate population and C/F/W ledger aliases."""
    ledger = json.loads(row["counters"])
    if row["method"] == "registered":
        # This fixture populates candidate incidence columns using base_action;
        # population_rhs is a subset of base_action_rhs, not extra solves.
        require(ledger["population_rhs"] == 48, "unexpected population strategy")
        return ledger["base_action_rhs"] + ledger["state_rhs"]
    return sum(ledger.get(k, 0) for k in ("refine_rhs", "winner_rhs", "state_rhs"))


def check_input(n, reference):
    from scale_views import fixture
    A, edits, _, metadata = fixture(n)
    require(metadata["edges"] == int(reference["edges"]), f"{n}: graph identity differs")
    del A
    with np.load(ROOT / "data" / f"scale_bounded_input_{n}.npz", allow_pickle=False) as data:
        original, bounded = data["s_original"], data["s"]
        require(original.shape == bounded.shape == (n,) and np.isfinite(original).all()
                and np.isfinite(bounded).all(), f"{n}: invalid opinion arrays")
        divisor = max(1., float(np.max(np.abs(original))))
        require(float(data["scaling_divisor"]) == divisor == float(reference["scaling_divisor"])
                and float(data["scaling_factor"]) == 1/divisor == float(reference["scaling_factor"]),
                f"{n}: incorrect uniform scaling")
        require(np.array_equal(bounded, original / divisor), f"{n}: scaled input differs")
        for prefix, values in (("before", original), ("after", bounded)):
            expected = dict(min=np.min(values), max=np.max(values), maxabs=np.max(np.abs(values)),
                            count=values.size)
            require(all(float(reference[f"{prefix}_{key}"]) == value for key, value in expected.items()),
                    f"{n}: incorrect {prefix} input diagnostics")
        for key in ("i", "j", "w", "flip"):
            require(np.array_equal(data[f"edit_{key}"], [getattr(e, key) for e in edits]),
                    f"{n}: candidate {key} identity differs")


def load_size(n):
    with (ROOT / "data" / f"scale_bounded_{n}.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    grid = {(int(r["rep"]), r["method"]): r for r in rows}
    require(len(rows) == len(grid) == 40 and
            set(grid) == {(rep, arm) for rep in range(10) for arm in ARMS},
            f"{n}: expected ten complete matched four-arm blocks")
    identity = ("edges", "scaling_factor", "scaling_divisor", "delta",
                "before_min", "before_max", "before_maxabs", "before_count",
                "after_min", "after_max", "after_maxabs", "after_count")
    reference = rows[0]
    check_input(n, reference)
    for rep in range(10):
        require(sorted(int(grid[rep, arm]["position"]) for arm in ARMS) == list(range(4)),
                f"{n}/{rep}: missing or duplicate randomized slots")
        scheduled = np.random.default_rng(871+rep).permutation(ARMS)
        require(all(scheduled[int(grid[rep, arm]["position"])] == arm for arm in ARMS),
                f"{n}/{rep}: randomized arm order differs")
    for r in rows:
        context = f"{n}/{r['rep']}/{r['method']}"
        require(int(r["n"]) == n and r["batch"] == "bounded_native", context + ": wrong input batch")
        require(all(r[k] == reference[k] for k in identity), context + ": input identity differs")
        require(float(r["delta"]) == 1e-9 and int(r["after_count"]) == n
                and float(r["after_maxabs"]) <= 1, context + ": unbounded opinion or wrong delta")
        sequence = json.loads(r["sequence"])
        require(len(sequence) <= 15 and len(sequence) == len(set(sequence)) and
                all(type(e) is int and 0 <= e < 32 for e in sequence), context + ": invalid path")
        require(int(r["commits"]) == len(sequence) and r["stop_reason"] ==
                ("budget" if len(sequence) == 15 else "certified_no_candidate"),
                context + ": incorrect stopping metadata")
        for key in METRICS:
            value = actual_rhs(r) if key == "actual_rhs" else float(r[key])
            require(np.isfinite(value) and value >= 0, context + ": invalid " + key)
        require(float(r["seconds"]) > float(r["setup_seconds"]) > 0 and float(r["peak_mib"]) > 0,
                context + ": invalid time/RSS")
        require(np.isclose(float(r["inclusive_seconds"]), sum(float(r[k]) for k in
                           ("seconds", "calibration_seconds", "normalization_seconds")), rtol=1e-12),
                context + ": inclusive time mismatch")
    for arm in ARMS:
        require(len({r["sequence"] for r in rows if r["method"] == arm}) == 1,
                f"{n}/{arm}: trajectory varies across repetitions")
    paths = {arm: json.loads(grid[0, arm]["sequence"]) for arm in ARMS}
    agreement = len({tuple(p) for p in paths.values()}) == 1
    print(n, "trajectory agreement:", agreement, "actual paths:", paths)
    return grid, agreement


def summarize(sizes=SIZES):
    summary = []
    with (ROOT / "data" / "scale_bounded_verified.csv").open(newline="") as stream:
        verified = list(csv.DictReader(stream))
    verification_keys = [(int(r["n"]), int(r["path"]), int(r["step"])) for r in verified]
    require(len(verification_keys) == len(set(verification_keys)), "duplicate independent verification steps")
    draws = np.random.default_rng(651).integers(0, 10, (20000, 10))
    for n in sizes:
        grid, agreement = load_size(n)
        paths = sorted({tuple(json.loads(r["sequence"])) for r in grid.values()})
        expected = {(path_id, step): winner for path_id, path in enumerate(paths)
                    for step, winner in enumerate(path)}
        evidence = {(int(r["path"]), int(r["step"])): r for r in verified if int(r["n"]) == n}
        require(set(evidence) == set(expected) and bool(expected), f"{n}: independent replay coverage differs")
        for key, winner in expected.items():
            record = evidence[key]
            require(int(record["winner"]) == winner and record["status"] == "interval_certified"
                    and float(record["delta"]) == 1e-9, f"{n}/{key}: independent winner replay differs")
        matrices = {arm: np.array([[actual_rhs(grid[rep, arm]) if metric == "actual_rhs"
                                   else float(grid[rep, arm][metric]) for metric in METRICS]
                                  for rep in range(10)]) for arm in ARMS}
        for arm in ARMS:
            data = matrices[arm]
            row = dict(n=n, method=arm, repetitions=10, trajectory_agreement=int(agreement),
                       sequence=grid[0, arm]["sequence"], commits=len(json.loads(grid[0, arm]["sequence"])))
            for j, metric in enumerate(METRICS):
                for quantile, value in zip(("q1", "median", "q3"), np.quantile(data[:, j], [.25, .5, .75])):
                    row[f"{metric}_{quantile}"] = float(value)
                if metric in ("seconds", "peak_mib", "actual_rhs", "inclusive_seconds"):
                    ratios = matrices["cold"][:, j] / data[:, j]
                    lo, hi = np.quantile(np.median(ratios[draws], axis=1), [.025, .975])
                    row[f"cold_over_arm_{metric}_median"] = float(np.median(ratios))
                    row[f"cold_over_arm_{metric}_ci_low"] = float(lo)
                    row[f"cold_over_arm_{metric}_ci_high"] = float(hi)
            summary.append(row)
            print(n, arm, "seconds/RSS/RHS medians", [row[f"{m}_median"] for m in METRICS[:3]],
                  "paired cold/arm time ratio (95% CI)",
                  [row[f"cold_over_arm_seconds_{s}"] for s in ("median", "ci_low", "ci_high")])
    return summary


def render(summary):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 8})
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6))
    for ax, metric, label in zip(axes, METRICS[:3], ("Time including setup (s)", "Peak RSS (MiB)", "Actual solved RHS")):
        for arm, title, color, marker, style in zip(ARMS, LABELS,
                ("#0072B2", "#D55E00", "#009E73", "#CC79A7"), "os^D", ("--", "-", "-.", ":")):
            records = [r for r in summary if r["method"] == arm]
            n = [r["n"] for r in records]
            lo, mid, hi = np.array([[r[f"{metric}_{q}"] for r in records] for q in ("q1", "median", "q3")])
            ax.errorbar(n, mid, yerr=[mid-lo, hi-mid], label=title, color=color,
                        marker=marker, linestyle=style, lw=1.2, ms=4, mfc="white", capsize=2)
        ax.set(xscale="log", yscale="log", xlabel="Vertices", ylabel=label)
        ax.set_xticks(n, [f"{x//1000}k" for x in n], rotation=25)
        ax.set_xticks([], minor=True)
        ax.grid(axis="y", alpha=.25)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4,
               frameon=False, fontsize=8, columnspacing=1.3)
    fig.subplots_adjust(left=.075, right=.99, bottom=.24, top=.79, wspace=.48)
    fig.savefig(ROOT / "figures" / "scale_bounded.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the committed CSV summary without rewriting")
    args = parser.parse_args()
    summary = summarize()
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
    target = ROOT / "data" / "scale_bounded_summary.csv"
    if args.check:
        require(target.read_text() == buffer.getvalue().replace("\r\n", "\n"), "bounded summary differs")
    else:
        target.write_text(buffer.getvalue(), newline="")
        render(summary)


if __name__ == "__main__":
    main()
