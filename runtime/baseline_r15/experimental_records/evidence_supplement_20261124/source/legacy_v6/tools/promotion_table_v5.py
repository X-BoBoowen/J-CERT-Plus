"""Render the full-catalog comparison from matched, measured 15-commit cursors."""
import argparse
import csv
import io
from pathlib import Path

import numpy as np

from promotion_table_v4 import setup_cost
from workloads import NAMES


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("static", "full", "transport", "adaptive")
SCAN_PHASES = ("seed_screen_s", "cold_query_s", "cold_screen_s")


def read(graph, kind, fine):
    tag = "_fine" if fine else ""
    with (ROOT/"data"/f"promotion_v5_{graph}{tag}_{kind}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def profile(series, draws):
    quantiles = np.quantile(series, [.25,.5,.75])
    ci = np.quantile(np.median(series[draws], axis=1), [.025,.975])
    return (*quantiles,*ci)


def describe(graph, fine=True, *, read=read, strict=False):
    runs, rounds = read(graph,"runs",fine), read(graph,"rounds",fine)
    if len(runs)!=40 or len(rounds)!=600:
        raise ValueError(f"incomplete cursor records for {graph}")
    groups, paths, scan, online, setup = {}, {}, {}, {}, {}
    for method in METHODS:
        group = sorted((r for r in runs if r["method"]==method), key=lambda r:int(r["query"]))
        assert [int(r["query"]) for r in group]==list(range(10))
        assert all(int(r["guarded"])==1 and int(r["directional"])==1 for r in group)
        assert all(int(r.get("fine_screen",0))==int(fine) for r in group)
        assert all(bool(int(r.get("strict_screen",0)))==strict for r in group)
        groups[method] = group
        setup[method] = setup_cost(group[0])
        assert all(setup_cost(r)==setup[method] for r in group)
        times = []
        for cursor in range(10):
            path = sorted((r for r in rounds if r["method"]==method and int(r["query"])==cursor),
                          key=lambda r:int(r["epoch"]))
            assert [int(r["epoch"]) for r in path]==list(range(1,16))
            assert all(0<int(r["low"])<=int(r["high"]) for r in path)
            if method=="full":
                assert int(path[-1]["transfers"])==0 and int(path[-1]["hits"])==0
            if strict:
                for t,row in enumerate(path):
                    count = int(row["strict_rows"])
                    assert count==int(row["seed_visits"])+int(row["catalog_visits"])
                    if method in ("full","transport"):
                        assert count==int(row["universe"])-t
            paths[method,cursor] = path
            times.append(sum(float(row[k]) for row in path for k in SCAN_PHASES))
        scan[method] = np.array(times)
        online[method] = np.array([float(r["online_seconds"]) for r in group])
        assert np.all((scan[method]>0) & (scan[method]<=online[method]))

    active = paths["adaptive",0]
    winners = [int(r["winner_id"]) for r in active]
    assert len(set(winners))==15
    for cursor in range(10):
        for method in ("full","transport","adaptive"):
            peer = paths[method,cursor]
            assert [int(r["winner_id"]) for r in peer]==winners
            assert all(max(int(a["low"]),int(b["low"]))<=min(int(a["high"]),int(b["high"]))
                       for a,b in zip(active,peer))
    outsiders = read(graph,"admissions",fine) if any(int(r["promoted"]) for r in active) else []
    admitted = {int(r["candidate"]) for r in outsiders}
    negative = {int(r["candidate"]) for r in outsiders if r["root_high"] and int(r["root_high"])<0}
    assert len(admitted)==sum(int(r["promoted"]) for r in active)
    m = int(active[0]["universe"])
    cold_visits = sum(int(r["catalog_visits"]) for r in active)
    cold_slots = sum(m-k-int(r["seed_visits"]) for k,r in enumerate(active))
    assert 0<=cold_visits<=cold_slots
    static = paths["static",0]
    miss = sum(int(r["winner_id"])!=w for r,w in zip(static,winners))
    def gain(path,field):
        return sum(int(r[field]) for r in path)
    loss_lo = 100*(1-gain(static,"high")/gain(active,"low")) if miss else 0.
    loss_hi = 100*(1-gain(static,"low")/gain(active,"high")) if miss else 0.

    draws = np.random.default_rng(108).integers(0,10,(10000,10))
    stats = {}
    for method in METHODS:
        total = setup[method]+online[method].sum()
        bootstrap = ((setup["full"]+online["full"][draws].sum(axis=1))/
                     (setup[method]+online[method][draws].sum(axis=1)))
        item = dict(graph=graph,method=method,n=10,setup_seconds=setup[method],total_10_seconds=total,
                    total_10_full_over_method=(setup["full"]+online["full"].sum())/total,
                    total_10_ci_low=np.quantile(bootstrap,.025),total_10_ci_high=np.quantile(bootstrap,.975))
        for label,values in (("scan",scan[method]),("online",online[method]),
                             ("scan_full_over_method",scan["full"]/scan[method]),
                             ("online_full_over_method",online["full"]/online[method]),
                             ("online_transport_over_method",online["transport"]/online[method])):
            for field,value in zip(("q25","median","q75","ci_low","ci_high"),profile(values,draws)):
                item[f"{label}_{field}"] = value
        stats[method] = item
    info = dict(graph=graph,universe=m,admitted=len(admitted),outside=len(set(winners)&admitted),
                revived=len(set(winners)&negative),miss=miss,loss_lo=loss_lo,loss_hi=loss_hi,
                cold_pct=100*cold_visits/cold_slots,cold_visits=cold_visits,cold_slots=cold_slots,
                index=float(groups["adaptive"][0]["index_seconds"]))
    return info,stats


def seconds(value):
    return f"{value:.2f}" if value>=.1 else f"{value:.3g}"


def table_row(info,stats):
    d = stats["adaptive"]
    cells = [f"{info['graph']} ({info['universe']:,})",f"{info['miss']}/{info['loss_hi']:.3g}",
             f"{info['admitted']}/{info['outside']}/{info['revived']}",f"{info['cold_pct']:.3g}",
             seconds(info["index"])]
    cells.extend(seconds(stats[k]["scan_median"]) for k in ("full","adaptive"))
    cells.extend(seconds(stats[k]["online_median"]) for k in METHODS)
    cells.extend(f"{v:.2f}" for v in (d["online_full_over_method_median"],d["total_10_full_over_method"]))
    return " & ".join(cells)+r" \\"


def render():
    measured = {graph:describe(graph) for graph in NAMES}
    rows = [table_row(info,stats) for info,stats in measured.values()]
    all_stats = [item for _,stats in measured.values() for item in stats.values()]
    info,stats = measured["Slashdot"]
    d = stats["adaptive"]
    totals = {key:sum(i[key] for i,_ in measured.values()) for key in ("admitted","outside","revived")}
    def ratio(key):
        return "$"+f"{d[key]:.2f}"+r"\times$"
    findings = (f"All {15*len(NAMES)} indexed choices match both full-scan policies. Across the six "
                f"trajectories, {totals['admitted']:,} admissions produce {totals['outside']} outside "
                f"winners, including {totals['revived']} with strictly negative root gain. "
                f"The static view disagrees at {sum(i['miss'] for i,_ in measured.values())} positions. "
                f"On Slashdot, the index inspects {info['cold_visits']:,} of {info['cold_slots']:,} "
                "eligible cold records across fifteen versions. The paired full/indexed scan ratio "
                f"is {ratio('scan_full_over_method_median')}. With transport enabled in both policies, "
                f"the paired full/indexed online ratio is {ratio('online_transport_over_method_median')}. "
                f"Discovery and transport together give {ratio('online_full_over_method_median')} "
                f"relative to full cold (95\\% paired bootstrap interval "
                f"[{d['online_full_over_method_ci_low']:.2f},{d['online_full_over_method_ci_high']:.2f}]).")
    _, coarse = describe("Alpha",fine=False)
    _, fine = measured["Alpha"]
    findings += (f" On Alpha, rechecking survivors at proof precision reduced median cold time from "
                 f"{seconds(coarse['full']['online_median'])} to {seconds(fine['full']['online_median'])} s "
                 f"and indexed transport time from {seconds(coarse['adaptive']['online_median'])} "
                 f"to {seconds(fine['adaptive']['online_median'])} s.")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer,fieldnames=all_stats[0],lineterminator="\n")
    writer.writeheader()
    writer.writerows(all_stats)
    return {"sections/promotion_rows5.tex":"\n".join(rows)+"\n",
            "sections/promotion_findings5.tex":findings+"\n",
            "data/promotion_v5_timing_summary.csv":buffer.getvalue()}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write",action="store_true")
    mode.add_argument("--check",action="store_true")
    args = parser.parse_args()
    for name,body in render().items():
        dest = ROOT/name
        if args.write:
            dest.write_text(body,encoding="utf8")
        elif dest.read_text(encoding="utf8")!=body:
            raise ValueError(f"Regenerate {name} from the published cursor measurements.")
