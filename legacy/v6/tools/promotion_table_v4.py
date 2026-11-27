"""Render Table III from the measured full-graph cursor runs."""
import csv
import argparse
from pathlib import Path
from statistics import median
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("Alpha", "OTC", "Slashdot", "Conflict", "Bio", "Fin", "AlphaClamp")


def read(name, kind):
    with (ROOT / "data" / f"promotion_v4_{name}_{kind}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def setup_cost(row):
    return sum(float(row[k]) for k in ("graph_seconds","catalog_seconds","common_seconds",
                                       "rank_seconds","index_seconds","bind_seconds"))


def timing_statistics(name):
    """Bootstrap paired query cursors, conditioning on the single observed preparation."""
    runs = read(name,"runs")
    groups = {m: sorted((r for r in runs if r["method"]==m),key=lambda r:int(r["query"]))
              for m in ("static","adaptive","full","fixed")}
    draws = np.random.default_rng(108).integers(0,10,(10000,10))
    full = np.array([float(r["online_seconds"]) for r in groups["full"]])
    full_setup = setup_cost(groups["full"][0])
    output = []
    for method,group in groups.items():
        assert [int(r["query"]) for r in group]==list(range(10))
        online = np.array([float(r["online_seconds"]) for r in group])
        paired = full/online
        setup = setup_cost(group[0])
        def profile(values):
            medians = np.median(values[draws],axis=1)
            return (*np.quantile(values,[.25,.5,.75]),*np.quantile(medians,[.025,.975]))
        item = dict(graph=name,method=method,cursors=10,resamples=10000,bootstrap_seed=108,
                    setup_seconds=setup)
        for label,values in (("online_seconds",online),("paired_full_over_method",paired)):
            for k,v in zip(("q25","median","q75","median_ci_low","median_ci_high"),profile(values)):
                item[label+"_"+k]=float(v)
        item["total_10_seconds"] = setup+online.sum()
        item["total_10_full_over_method"] = (full_setup+full.sum())/item["total_10_seconds"]
        conditional = (full_setup+full[draws].sum(axis=1))/(setup+online[draws].sum(axis=1))
        for k,v in zip(("conditional_ci_low","conditional_ci_high"),np.quantile(conditional,[.025,.975])):
            item["total_10_full_over_method_"+k]=float(v)
        output.append(item)
    return output


def table_row(name):
    runs = read(name, "runs")
    rounds = read(name, "rounds")
    grouped = {m: [r for r in runs if r["method"] == m] for m in ("static", "adaptive", "full", "fixed")}
    for group in grouped.values():
        assert len(group) == 10 and {int(r["query"]) for r in group} == set(range(10))
        assert all(r["directional"]=="1" and r["guarded"]=="1" for r in group)
    paths = {m: [r for r in rounds if r["method"] == m and int(r["query"]) == 0] for m in grouped}
    adaptive = paths["adaptive"]
    winners = [int(r["winner_id"]) for r in adaptive]
    assert len(winners) == len(set(winners)) == 15
    for method in grouped:
        for query in range(10):
            path = [r for r in rounds if r["method"]==method and int(r["query"])==query]
            assert [int(r["epoch"]) for r in path]==list(range(1,16))
            assert all(0<int(r["low"])<=int(r["high"]) for r in path)
            if method!="static":
                assert [int(r["winner_id"]) for r in path]==winners
                assert all(max(int(a["low"]),int(b["low"]))<=min(int(a["high"]),int(b["high"]))
                           for a,b in zip(path,adaptive))
            assert path[0]["state_source"]=="shared"
            assert all(int(r["child_state_residual"])>=0 for r in path)
    admitted = read(name,"admissions") if any(int(row["promoted"]) for row in adaptive) else []
    admitted_ids = {int(r["candidate"]) for r in admitted}
    negative = {int(r["candidate"]) for r in admitted if r["root_high"] and int(r["root_high"]) < 0}
    a,w,r = len(admitted_ids),len(set(winners)&admitted_ids),len(set(winners)&negative)
    assert a == sum(int(row["promoted"]) for row in adaptive)
    if name in ("AlphaClamp","AlphaMotif"):
        assert r > 0 and int(adaptive[1]["winner_id"]) in negative
    universe = int(adaptive[0]["universe"])
    visits = 100*sum(int(r["visited"]) for r in adaptive)/sum(universe-t for t in range(15))
    def online(method):
        return median(float(row["online_seconds"]) for row in grouped[method])
    def ten(method):
        rows = grouped[method]
        return setup_cost(rows[0]) + sum(float(row["online_seconds"]) for row in rows)
    miss = sum(int(row["winner_id"])!=winner for row,winner in zip(paths["static"],winners))
    def total_gain(method, bound):
        return sum(int(row[bound]) for row in paths[method])
    # The upper endpoint is sufficient for the rounded static loss shown in the table.
    loss = (100*(total_gain("full","high")-total_gain("static","low")) /
            total_gain("full","high") if miss else 0.)
    label = "Alpha (clamped S1)" if name == "AlphaClamp" else name
    setup = float(grouped["adaptive"][0]["index_seconds"])
    return (f"{label} & {universe:,} & {miss}/{loss:.3g} & {a}/{w}/{r} & {visits:.3f} & {setup:.3g} & "
            f"{online('static'):.2f} & {online('adaptive'):.2f} & {online('full'):.2f} & "
            f"{ten('full')/ten('adaptive'):.2f} " + r"\\")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true",help="replace the measured table body in the manuscript")
    args = parser.parse_args()
    body = "\n".join(table_row(name) for name in NAMES)
    if args.write:
        path = ROOT/"sections"/"promotion_results.tex"
        source = path.read_text(encoding="utf-8")
        prefix,tail = source.split(r"\midrule",1)
        _,suffix = tail.split(r"\bottomrule",1)
        path.write_text(prefix+r"\midrule"+"\n"+body+"\n"+r"\bottomrule"+suffix,encoding="utf-8")
        statistics = [row for name in NAMES for row in timing_statistics(name)]
        with (ROOT/"data"/"promotion_v4_timing_summary.csv").open("w",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=statistics[0])
            writer.writeheader()
            writer.writerows(statistics)
    else:
        print(body)
