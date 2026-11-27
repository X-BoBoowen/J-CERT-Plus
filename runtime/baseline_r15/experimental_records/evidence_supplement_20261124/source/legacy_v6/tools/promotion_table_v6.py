"""Render strict per-row promotion measurements alongside the staged cohort."""
import argparse
import csv
import io

import numpy as np

from promotion_table_v5 import ROOT, METHODS, NAMES, SCAN_PHASES, describe, read as read_staged, seconds


def read_strict(graph, kind, fine=False):
    if fine:
        raise ValueError("strict and staged measurements have separate cohorts")
    with (ROOT / "data" / f"promotion_v6_{graph}_{kind}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def render(names=NAMES):
    rows, summaries, staged_rows, infos = [], [], [], []
    cross_cohort_ratios = []
    staged_scan_wins = {}
    for graph in names:
        info, stats = describe(graph, fine=False, read=read_strict, strict=True)
        _, staged = describe(graph)
        infos.append(info)
        d = stats["adaptive"]
        cross_cohort_ratios.append(staged["transport"]["online_median"]/d["online_median"])
        # Screening changes the admitted view, but every full-catalog winner stays certified.
        current_path = [r for r in read_strict(graph,"rounds") if r["method"]=="full" and r["query"]=="0"]
        staged_path = [r for r in read_staged(graph,"rounds",True) if r["method"]=="full" and r["query"]=="0"]
        assert len(current_path)==len(staged_path)==15
        for current, previous in zip(current_path,staged_path):
            assert current["winner_id"]==previous["winner_id"]
            assert max(int(current["low"]),int(previous["low"]))<=min(int(current["high"]),int(previous["high"]))
        cells = [f"{graph} ({info['universe']:,})",f"{info['miss']}/{info['loss_hi']:.3g}",
                 f"{info['admitted']}/{info['outside']}/{info['revived']}",f"{info['cold_pct']:.3g}"]
        cells += [seconds(stats[k]["scan_median"]) for k in ("full","adaptive")]
        cells += [seconds(stats[k]["online_median"]) for k in METHODS]
        cells += [f"{d[key]:.2f}" for key in ("online_transport_over_method_median",
                                               "online_full_over_method_median","total_10_full_over_method")]
        rows.append(" & ".join(cells)+r" \\")
        sd = staged["adaptive"]
        staged_rounds = read_staged(graph,"rounds",True)
        def scan_by_cursor(method):
            scan = np.zeros(10)
            for row in staged_rounds:
                if row["method"]==method:
                    scan[int(row["query"])] += sum(float(row[k]) for k in SCAN_PHASES)
            return scan
        staged_scan_wins[graph] = int((scan_by_cursor("full")>scan_by_cursor("adaptive")).sum())
        staged_rows.append(" & ".join([graph]+[seconds(staged[k]["scan_median"]) for k in ("full","adaptive")]
                           +[seconds(staged[k]["online_median"]) for k in ("full","transport","adaptive")]
                           +[f"{sd['online_full_over_method_median']:.2f}",f"{sd['total_10_full_over_method']:.2f}"])+r" \\")
        times = read_strict(graph,"runs")
        assert all(int(r["fused_threshold"])==(r["method"]=="adaptive") and int(r["reuse_certified_rows"])==1 for r in times)
        draws = np.random.default_rng(108).integers(0,10,(10000,10))
        def online(method):
            records = sorted((r for r in times if r["method"]==method),key=lambda r:int(r["query"]))
            return np.array([float(r["online_seconds"]) for r in records])
        transport = online("transport")
        for method in METHODS:
            item = stats[method]
            m = online(method)
            ref_setup = stats["transport"]["setup_seconds"]
            item["total_10_transport_over_method"] = stats["transport"]["total_10_seconds"]/item["total_10_seconds"]
            boot = ((ref_setup+transport[draws].sum(axis=1))/
                    (item["setup_seconds"]+m[draws].sum(axis=1)))
            item["total_10_transport_ci_low"],item["total_10_transport_ci_high"] = np.quantile(boot,[.025,.975])
            summaries.append(item)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream,fieldnames=summaries[0].keys(),lineterminator="\n")
    writer.writeheader()
    writer.writerows(summaries)
    findings = "Slashdot measurements are in progress."
    if tuple(names)==NAMES:
        trace = next(i for i in infos if i["graph"]=="Slashdot")
        totals = {key:sum(i[key] for i in infos) for key in ("outside","revived","admitted","miss")}
        slash = next(r for r in summaries if r["graph"]=="Slashdot" and r["method"]=="adaptive")
        findings = (f"Across the six trajectories, all 90 indexed winners match both full scans "
                    f"and the staged cohort. The static view disagrees at {totals['miss']} positions. "
                    f"Of {totals['admitted']:,} admissions, {totals['outside']} become winners. "
                    f"{totals['revived']} winners have strictly negative root gain. "
                    f"On Slashdot, D inspects {trace['cold_visits']:,} of {trace['cold_slots']:,} "
                    f"cold row-epochs. Its paired scan F/D is {slash['scan_full_over_method_median']:.2f}"
                    r"$\times$"
                    f" and its paired online F/D has 95\\% bootstrap interval "
                    f"[{slash['online_full_over_method_ci_low']:.2f},{slash['online_full_over_method_ci_high']:.2f}].")
        adaptive = [r for r in summaries if r["method"]=="adaptive"]
        large = sum(r["scan_full_over_method_median"]>=3 for r in adaptive)
        fin = next(r for r in adaptive if r["graph"]=="Fin")
        findings += (f" Strict scan F/D reaches three on {large} graphs. Fin reaches "
                     f"{fin['scan_full_over_method_median']:.2f}" + r"$\times$" +
                     f". In the staged cohort D wins the scan comparison in "
                     f"{staged_scan_wins['Conflict']}/10 Conflict cursors and "
                     f"{staged_scan_wins['Fin']}/10 Fin cursors.")
        findings += (f" Across the separately timed cohorts, the ratio of staged T to strict D "
                     f"online medians spans {min(cross_cohort_ratios):.2f}--{max(cross_cohort_ratios):.2f}"
                     +r"$\times$.")
    return {"sections/promotion_rows6.tex":"\n".join(rows)+"\n",
            "sections/promotion_staged6.tex":"\n".join(staged_rows)+"\n",
            "sections/promotion_findings6.tex":findings+"\n",
            "data/promotion_v6_timing_summary.csv":stream.getvalue()}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write",action="store_true")
    group.add_argument("--check",action="store_true")
    args = parser.parse_args()
    for name,body in render().items():
        path = ROOT/name
        if args.write:
            path.write_text(body,encoding="utf8")
        elif path.read_text(encoding="utf8")!=body:
            raise ValueError(f"Regenerate {name} from measured promotion data")
