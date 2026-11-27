"""Fixed-input timing intervals for the branching, scaling and precision studies."""
import argparse
import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
FIELDS = ("table","graph","M","budget","backend","policy","quantity","n",
          "q25","median","q75","CI_2.5","CI_97.5")


def timing_intervals():
    groups = {}
    def insert(key, row):
        rep = int(row["rep"])
        group = groups.setdefault(key, {})
        if rep in group:
            raise ValueError(f"duplicate repetition for {key}")
        group[rep] = float(row["seconds"])
    expected = set()
    for graph in ("Alpha","OTC"):
        for level in range(4):
            expected.add(("VI",graph,1000,"10/15/20/25","cholmod",f"L{level}"))
        with (DATA/f"branch_move_{graph.lower()}.csv").open(newline="") as raw:
            for r in csv.DictReader(raw):
                insert(("VI",graph,1000,"10/15/20/25","cholmod","L"+r["level"]),r)
    if set(groups) != expected or any(set(v) != set(range(10)) for v in groups.values()):
        raise ValueError("missing or unexpected fixed-input timing cells")
    rng = np.random.default_rng(113)
    ratios = {"L1":("L0",),"L2":("L1",),"L3":("L0","L2")}
    for key, observations in sorted(groups.items()):
        times = np.array([observations[i] for i in range(10)])
        samples = {"seconds":times}
        for ref in ratios.get(key[-1],()):
            baseline = groups[(*key[:-1],ref)]
            samples[f"{ref}/policy"] = [baseline[i]/times[i] for i in range(10)]
        for quantity, values in samples.items():
            if np.min(values) <= 0 or not np.isfinite(values).all():
                raise ValueError(f"invalid timing for {key}")
            bootstrap = np.median(rng.choice(values, (10000,10)), axis=1)
            yield (*key,quantity,10,*np.quantile(values,[.25,.5,.75]),
                   *np.quantile(bootstrap,[.025,.975]))


SUPPORT_FIELDS = ("study","size_or_target","method_or_step","quantity","n",
                  "q25","median","q75","CI_2.5","CI_97.5","seed","resamples")


def observations(rows, column):
    ordered=sorted(rows,key=lambda r:int(r["rep"]))
    if [int(r["rep"]) for r in ordered]!=list(range(10)):
        raise ValueError("missing or duplicate timing repetitions")
    values=np.array([float(r[column]) for r in ordered])
    if not np.isfinite(values).all() or min(values)<=0:
        raise ValueError(f"invalid measured value in {column}")
    return values


def describe(key,values,rng,seed):
    sample_medians=np.median(rng.choice(values,(10000,10)),axis=1)
    return (*key,len(values),*np.quantile(values,[.25,.5,.75]),
            *np.quantile(sample_medians,[.025,.975]),seed,len(sample_medians))


def scale_intervals():
    from scale_frontier import ARMS, SIZES, dataset, records
    from native_scale_frontier import output
    seed = 651
    draws = np.random.default_rng(seed).integers(0,10,(20000,10))
    for n in SIZES:
        rows = records(output(n)) + [r for r in records(dataset(n)) if r["method"] in ("registered","cholmod")]
        if len(rows) != 40 or any(int(r["n"]) != n for r in rows):
            raise ValueError(f"incomplete four-arm frontier at {n}")
        values = {}
        for method in ARMS:
            arm = [r for r in rows if r["method"] == method]
            for column in ("seconds","peak_mib","setup_seconds","calibration_seconds"):
                values[method,column] = observations(arm,column)
            values[method,"inclusive_seconds"] = values[method,"seconds"] + values[method,"calibration_seconds"]
        values["planner","cold/planner_seconds"] = values["cold","seconds"]/values["planner","seconds"]
        for (method,column),v in values.items():
            yield ("frontier",n,method,column,10,*np.quantile(v,[.25,.5,.75]),
                   *np.quantile(np.median(v[draws],axis=1),[.025,.975]),seed,len(draws))


def search_intervals():
    """Resample ten independent catalog indices, preserving paired policy/reset observations."""
    for graph in ("Alpha","OTC"):
        for prefix,mode in (("sparse_deep","dfs"),("sparse_trap_cohort","beam")):
            with (DATA/f"{prefix}_{graph}.csv").open(newline="") as f:
                rows = [r for r in csv.DictReader(f) if r["mode"] == mode]
            values = {}
            for case in range(10):
                arms = {}
                for policy in ("cold","transport"):
                    group = sorted((r for r in rows if int(r["case"])==case and r["policy"]==policy),
                                   key=lambda r:int(r["repeat"]))
                    if [int(r["repeat"]) for r in group] != list(range(10)):
                        raise ValueError(f"incomplete search block {graph}/{case}/{policy}")
                    arm = dict(seconds=np.array([float(r["seconds"]) for r in group]))
                    for q in ("setup_s","first_query_s","ten_query_s"):
                        arm[q] = float(group[0][q])
                    arms[policy] = arm
                    for q,v in arm.items():
                        values.setdefault((policy,q),[]).append(np.median(v))
                for q in arms["cold"]:
                    pair = arms["cold"][q]/arms["transport"][q]
                    values.setdefault(("cold/transport",q),[]).append(np.median(pair))
            for (policy,q),v in values.items():
                if len(rows)!=200 or not np.isfinite(v).all() or min(v)<=0:
                    raise ValueError(f"invalid search timing {graph}/{mode}/{q}")
                yield describe((mode,graph,policy,q),v,np.random.default_rng(408),408)


def support_intervals():
    rng=np.random.default_rng(114)
    with (DATA/"margin_path.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 80:
        raise ValueError("expected 80 margin observations")
    for target in (.01,.0001,.000001,.00000001):
        for step in (1,2):
            group = [r for r in rows if float(r["target"])==target and int(r["step"])==step]
            yield describe(("margin",target,step,"query_s"),observations(group,"query_s"),rng,114)


def write(name, fields, records):
    rows = list(records)
    with (DATA/name).open("w",newline="") as f:
        out = csv.writer(f)
        out.writerow(fields)
        out.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-only", action="store_true",
                        help="summarize the scale, frontier and margin runs")
    if not parser.parse_args().support_only:
        write("branch_timing_intervals.csv",FIELDS,timing_intervals())
    write("margin_timing_intervals.csv",SUPPORT_FIELDS,support_intervals())
    write("search_timing_intervals.csv",SUPPORT_FIELDS,search_intervals())
    write("scale_timing_intervals.csv",SUPPORT_FIELDS,scale_intervals())
