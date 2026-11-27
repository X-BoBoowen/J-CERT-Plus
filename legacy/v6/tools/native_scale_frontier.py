"""Matched native Jacobi-PCG cold/planner measurements on the v6 frontier fixture."""
import argparse
import csv
import json
from functools import partial
from pathlib import Path
import subprocess
import sys
import tempfile
from time import sleep

from scale_frontier import ROOT, SIZES, dataset, headroom_mib, np, records, trial
from native_planner import NativeCostPlanner
from paired_pcg import build_paired_pcg, load_paired_pcg


def output(n):
    return ROOT/"data"/f"scale_frontier_v6_native_{n}.csv"


def validate(n, row):
    source = dataset(n) if n != 500000 else ROOT/"data"/"scale_frontier_v5_500000.csv"
    reference = next(r for r in records(source) if r["method"] == row["method"])
    for key in ("n", "edges", "sequence", "frontiers"):
        if str(row[key]) != reference[key]:
            raise ValueError(f"{n} {row['method']}: changed {key}")
    counters = json.loads(row["counters"])
    expected = json.loads(reference["counters"])
    for ledger in (counters, expected):
        ledger.pop("historical_ns", None)
        # Last-bit differences may change which of two valid radii is tighter.
        tighter = ledger.pop("directional_blocks_tighter", 0)
        if not 0 <= tighter <= ledger.get("directional_blocks",0):
            raise ValueError("invalid directional radius choice count")
    if counters != expected:
        changed = [k for k in counters.keys() | expected.keys() if counters.get(k) != expected.get(k)]
        raise ValueError(f"{n} {row['method']}: changed counters {changed}")
    metrics = [float(row[k]) for k in ("setup_seconds","seconds","peak_mib","calibration_seconds")]
    if not all(np.isfinite(value) and value > 0 for value in metrics):
        raise ValueError("invalid timing or memory measurement")
    if metrics[0] >= metrics[1]:
        raise ValueError("invalid measured setup/query times")


def run_child(n, action, library):
    cmd = [sys.executable, __file__, "--n", str(n), "--child", action,
           "--library", str(library)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    while True:
        while min(headroom_mib()) < 1100:
            sleep(10)
        with subprocess.Popen(cmd,stdout=subprocess.PIPE,text=True,creationflags=flags) as proc:
            while True:
                try:
                    out, _ = proc.communicate(timeout=1)
                    if proc.returncode:
                        raise RuntimeError(f"{action} {n} exited {proc.returncode}")
                    return json.loads(out)
                except subprocess.TimeoutExpired:
                    if min(headroom_mib()) < 640:
                        proc.kill()
                        proc.communicate()
                        break


def check(n):
    rows = records(output(n))
    pairs = {(int(r["rep"]),r["method"]) for r in rows}
    if len(rows)!=20 or pairs!={(rep,m) for rep in range(10) for m in ("cold","planner")}:
        raise ValueError(f"Incomplete native pairs at {n}")
    for row in rows:
        order = np.random.default_rng(871+int(row["rep"])).permutation(("cold","planner"))
        if row["batch"] != "native" or order[int(row["position"])] != row["method"]:
            raise ValueError("unexpected native pair scheduling")
        validate(n,row)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n",type=int,nargs="+",choices=SIZES,default=SIZES)
    p.add_argument("--child",choices=("cold","planner"))
    p.add_argument("--check",action="store_true")
    p.add_argument("--library",type=Path)
    args=p.parse_args()
    if args.child:
        factory = partial(NativeCostPlanner,native_pcg=load_paired_pcg(args.library))
        print(json.dumps(trial(args.n[0],args.child,cg_factory=factory)))
        return
    if args.check:
        for n in args.n:
            check(n)
        return
    with tempfile.TemporaryDirectory() as directory:
        library = build_paired_pcg(directory)
        for n in args.n:
            done = {(int(r["rep"]),r["method"]) for r in records(output(n))}
            if len(done)==20:
                check(n)
                continue
            for rep in range(10):
                order = np.random.default_rng(871+rep).permutation(("cold","planner"))
                for pos,method in enumerate(order):
                    if (rep,method) in done:
                        continue
                    row = run_child(n,method,library)
                    validate(n,row)
                    row = dict(rep=rep,position=pos,**row,batch="native")
                    with output(n).open("a",newline="") as f:
                        writer=csv.DictWriter(f,fieldnames=row)
                        if f.tell()==0:
                            writer.writeheader()
                        writer.writerow(row)
                    print(n,rep,method,round(row["seconds"],3),flush=True)
            check(n)


if __name__ == "__main__":
    main()
