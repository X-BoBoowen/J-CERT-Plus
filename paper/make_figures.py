"""Render manuscript figures from retained measurements; never runs experiments.

Run only after the serial performance window has ended:
  python3 make_figures.py --new-aggregate ../validation/a_aggregate.json

The right panel withholds complete-Q4 points whenever any repeat of that
cell/method is incomplete. Partial elapsed costs belong in the result table,
not on the complete-service axis. Input row counts and figure values are saved.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "review" / "mpl_cache"))


def read_old(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 36, f"Expected all 36 old cell-method records: {len(rows)}"
    out = []
    for row in rows:
        costs = [float(row[f"seconds_r{i}"]) for i in range(3)]
        out.append({"graph": row["graph"], "stream": row["stream"],
                    "label": row["seed_label"], "method": row["method"],
                    "complete": True, "costs": costs, "n": len(costs)})
    return out


def read_new(path: Path) -> list[dict]:
    summary = json.loads(path.read_text())
    cells = summary["cells"]
    assert len(cells) == 6, f"Expected all six frozen new cells: {len(cells)}"
    out = []
    for cell_index, cell in enumerate(cells):
        for method in ["D", "S", "H"]:
            row = cell["methods"][method]
            assert row["planned_workers"] == 3
            complete = row["completed_service_workers"] == 3
            values = row["seconds_by_repeat"]
            assert len(values) == 3
            if complete:
                assert all(value is not None for value in values)
                costs = [float(value) for value in values]
            else:
                costs = []
            out.append({"graph": cell["graph"], "stream": cell["stream"],
                        "method": method, "label": "unseen", "complete": complete,
                        "costs": costs, "n": row["planned_workers"],
                        "complete_workers": row["completed_service_workers"],
                        "sources": [str(path), f"cells[{cell_index}].methods.{method}"]})
    return out


def plot_panel(ax, rows: list[dict], old: bool) -> None:
    import numpy as np
    from matplotlib.ticker import ScalarFormatter
    families = ["slow", "independent", "signed-neighbor"]
    labels = ["confirmation-1", "confirmation-2"] if old else ["unseen"]
    cells = [(g, f, label) for g in ["Alpha", "OTC"] for f in families for label in labels]
    lookup = {(r["graph"], r["stream"], r["label"], r["method"]): r for r in rows}
    style = {"D": ("#3B638B", "o", -0.22),
             "S": ("#6B6B6B", "s", 0),
             "H": ("#B45523", "^", 0.22)}
    missing_cells = set()
    for method, (color, marker, offset) in style.items():
        xs, meds, lows, highs = [], [], [], []
        for x, cell in enumerate(cells):
            row = lookup[(*cell, method)]
            if not row["complete"]:
                missing_cells.add(x)
                continue
            values = row["costs"]
            mid = median(values)
            xs.append(x + offset)
            meds.append(mid)
            lows.append(mid - min(values))
            highs.append(max(values) - mid)
        ax.errorbar(xs, meds, yerr=np.asarray([lows, highs]), fmt=marker,
                    markersize=4.3, color=color, elinewidth=1, capsize=2,
                    markeredgewidth=.7, label=method, linestyle="none")
    short = {"slow": "sl", "independent": "u", "signed-neighbor": "c"}
    xt = [short[f] + (label[-1] if old else "") + ("\u2020" if x in missing_cells else "")
          for x, (_, f, label) in enumerate(cells)]
    ax.set_xticks(range(len(cells)), xt)
    split = len(cells) / 2
    ax.axvline(split - .5, color="#BBBBBB", linewidth=.7)
    for x, label in [(split/2-.5, "Alpha"), (3*split/2-.5, "OTC")]:
        ax.text(x, -.20, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.3)
    ax.set_xlim(-.6, len(cells)-.4)
    ax.set_yscale("log")
    values = [v for row in rows for v in row["costs"]]
    top = max(120, max(values) * 1.3)
    ax.set_ylim(1, top)
    ax.set_yticks([v for v in [1, 3, 10, 30, 100, 300] if v < top])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-csv", type=Path,
                        default=HERE / "review" / "a2_tie_cell_methods.csv")
    parser.add_argument("--new-aggregate", type=Path,
                        default=HERE / "review" / "a_aggregate.json")
    args = parser.parse_args()
    old, new = read_old(args.old_csv), read_new(args.new_aggregate)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.titlesize": 9, "axes.labelsize": 8.5,
                         "xtick.labelsize": 7.6, "ytick.labelsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    plot_panel(axes[0], old, True)
    plot_panel(axes[1], new, False)
    axes[0].set_title("(a) Fixed-input retest", loc="left", pad=10)
    axes[1].set_title("(b) Post-freeze confirmation", loc="left", pad=10)
    axes[0].set_ylabel("Complete setup + Q4 (s; log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["D: explicit", "S: streaming", "H: implicit"],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.53, 1.00))
    fig.subplots_adjust(left=.085, right=.99, bottom=.25, top=.80, wspace=.25)
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "complete_costs.pdf")
    fig.savefig(HERE / "figures" / "complete_costs.png", dpi=220)
    (HERE / "editorial" / "narrative_revision" / "figure_values.json").write_text(
        json.dumps({"old": old, "new": new,
                    "incomplete_policy": "No complete-Q4 point; visibly retained cell",
                    "whiskers": "minimum-maximum over three repeats, not confidence intervals"}, indent=2))


if __name__ == "__main__":
    main()
