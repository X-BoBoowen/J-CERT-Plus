"""Write the left panel of the branching table from ten measured trials."""
import csv
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def render():
    lines = []
    for graph in ("Alpha", "OTC"):
        with (ROOT / "data" / f"branch_move_{graph.lower()}.csv").open() as f:
            data = list(csv.DictReader(f))
        require = {0, 1, 2, 3}
        for level in sorted(require):
            rows = [r for r in data if int(r["level"]) == level]
            if len(rows) != 10 or {int(r["rep"]) for r in rows} != set(range(10)):
                raise ValueError("Expected ten measured repetitions per level")
            if any(r["paths"] != data[0]["paths"] for r in rows):
                raise ValueError("Branching paths differ")
            counts = json.loads(rows[0]["counters"])
            times = [float(r["seconds"]) for r in rows]
            entries = tuple(counts.get(k, 0) for k in ("C", "F", "W", "R"))
            rhs = sum(counts.get(k + "_rhs", 0) for k in ("C", "F", "W", "R"))
            state = counts["state_rhs"]
            paths = json.loads(rows[0]["paths"])
            group = [graph, f'{rows[0]["flips"]}/{sum(map(len, paths))}', "", ""][level]
            lines.append(f"{group} & L{level} & {median(times):.2f} & {state:,} & "
                         f"{rhs:,} & {state+rhs:,} & {'/'.join(map(str,entries))} " + r"\\")
        if graph == "Alpha":
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    (ROOT / "sections" / "branch_rows.tex").write_text(render())

