"""Plot frozen H/G service records. No solver, tests, or experiments are run."""
from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "review" / "mpl_cache"))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((HERE / "review" / "hg_light_figure_values.json").read_text())
    rows = data["cells"]
    assert len(rows) == 6
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    })
    fig, ax = plt.subplots(figsize=(3.42, 2.58))
    styles = {"H": ("#B45523", "^", -0.15), "G": ("#246B7A", "o", 0.15)}
    xmax = max(v for row in rows if row["complete"]
               for method in styles for v in row[f"{method}_seconds"]) * 1.13
    for method, (color, marker, offset) in styles.items():
        first = True
        for i, row in enumerate(rows):
            if not row["complete"]:
                continue
            values = row[f"{method}_seconds"]
            assert len(values) == 3
            center = median(values)
            ax.errorbar(center, i + offset,
                        xerr=[[center - min(values)], [max(values) - center]],
                        fmt=marker, color=color, markersize=4.2,
                        capsize=2.2, elinewidth=1, markeredgewidth=.7,
                        label=method if first else None, zorder=3)
            first = False
    labels = []
    names = {"slow": "slow", "independent": "uniform", "signed-neighbor": "signed"}
    for i, row in enumerate(rows):
        suffix = "\u2020" if not row["complete"] else ""
        labels.append(f'{row["graph"]} / {names[row["stream"]]}{suffix}')
    ax.set_yticks(range(len(rows)), labels)
    ax.set_ylim(len(rows) - .55, -.55)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Complete setup + Q4 (seconds)", labelpad=4)
    ax.grid(axis="x", color="#DDDDDD", linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(.51, 1.015),
              ncol=2, frameon=False, handletextpad=.25, columnspacing=1.2,
              borderaxespad=0)
    fig.subplots_adjust(left=.32, right=.99, top=.88, bottom=.19)
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "hg_light_costs.pdf")
    fig.savefig(HERE / "figures" / "hg_light_costs.png", dpi=220)


if __name__ == "__main__":
    main()
