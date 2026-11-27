"""Rebuild Table IV and Figure 2 from measured, decision-checked search rows."""
import argparse
import csv
import io
import json
import math
from collections import defaultdict
from statistics import geometric_mean, mean, median
from time import perf_counter

from planned_matrix import matrix_inputs
from orthogonal import ROOT
from policy_kernel import PolicyKernel
from workloads import NAMES


BACKENDS = ("cg", "cholmod")
MODES = ("cold", "same", "planner")
SIZES = (1000, 2500)
BUDGETS = (20, 40)
EXTERNAL = {"q": "quad", "l": "registered", "d": "lv"}


def profile_routes():
    from scale_views import symbolic
    from workloads import load_graph, system
    rows = []
    for graph in NAMES:
        operator = system(load_graph(graph))
        start = perf_counter()
        profile = symbolic(operator)
        seconds = perf_counter() - start
        rows.append(dict(graph=graph,symbolic_nnz=profile["symbolic_nnz"],seconds=seconds))
    report_csv(ROOT/"data/persistent_routing.csv",rows,False)


def catalogue_routes():
    """Apply the factor-entry storage allowance to pre-indexed root metadata."""
    with (ROOT/"data/persistent_routing.csv").open() as source:
        routes = {}
        for r in csv.DictReader(source):
            nnz, seconds = int(r["symbolic_nnz"]), float(r["seconds"])
            if r["graph"] in routes or nnz <= 0 or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("invalid root symbolic profile")
            routes[r["graph"]] = ("cholmod" if 12*nnz <= 256*2**20 else "cg", seconds)
    if set(routes) != set(NAMES):
        raise ValueError("incomplete root-fill catalogue")
    return routes


def read_runs(prefix):
    groups = defaultdict(dict)
    for graph in NAMES:
        for m in SIZES:
            stem = f"{prefix}_{graph}_{m}"
            canonical = ROOT / "data" / (stem + ".csv")
            sources = [canonical] if canonical.exists() else sorted((ROOT/"data").glob(stem + ".*.csv"))
            if (prefix == "clustered_fast" or prefix.startswith("direct6") or "persistent" in prefix) and canonical.exists():
                sources.extend(sorted((ROOT/"data").glob(stem + ".*.csv")))
            for path in sources:
                with path.open() as f:
                    for row in csv.DictReader(f):
                        backend = (prefix if prefix.startswith("direct6") else
                                   "native" if prefix=="persistent_native" else
                                   "lookup" if prefix.startswith("linview_registered_lookup_") else
                                   "registered" if prefix.startswith("linview_registered_") else
                                   "quad" if prefix.endswith("_quad") else
                                   "lv" if prefix.startswith("linview_") else row["backend"])
                        key = graph, m, int(row["b"]), backend, backend if backend in (*EXTERNAL.values(),"lookup") else row["mode"]
                        rep = int(row["rep"])
                        if "persistent" in prefix or prefix.startswith("direct6"):
                            seq = json.loads(row["sequence"])
                            if (row["graph"] != graph or int(row["M"]) != m or key[2] not in BUDGETS
                                    or len(seq) != int(row["commits"]) or len(seq) > key[2]
                                    or len(set(seq)) != len(seq) or any(i not in range(m) for i in seq)
                                    or rep not in range(10) or not math.isfinite(float(row["seconds"]))
                                    or float(row["seconds"]) <= 0):
                                raise ValueError(f"invalid checkpoint at {path}, repetition {rep}")
                            if prefix in ("persistent_fast","persistent_batch") and (backend not in BACKENDS or
                                    row["mode"] not in MODES or not row["schedule"]):
                                raise ValueError(f"invalid fast method or missing schedule in {path}")
                            if prefix == "persistent_native" and (row["backend"],row["mode"]) != ("cg","planner"):
                                raise ValueError(f"invalid native planner in {path}")
                            if prefix == "direct6" and (row["backend"]!="cholmod" or
                                                        row["mode"] not in ("planner","planner6")):
                                raise ValueError(f"invalid direct planner in {path}")
                            if prefix == "direct6_slots" and (row["backend"]!="cholmod" or
                                                              row["mode"] not in ("planner6","slots6")):
                                raise ValueError(f"invalid fixed-slot planner in {path}")
                            if prefix == "linview_registered_persistent" and row["mode"] != "linview_registered":
                                raise ValueError(f"invalid registered LINVIEW mode in {path}")
                            if prefix == "linview_registered_lookup_persistent" and row["mode"] != "linview_registered_lookup":
                                raise ValueError(f"invalid registered lookup LINVIEW mode in {path}")
                        if rep in groups[key]:
                            raise ValueError(f"duplicate repetition {rep} in {key}")
                        groups[key][rep] = row
    return groups


def root_signs(graph, m, seq, clustered=False, persistent=False):
    A, s, edits = matrix_inputs(graph, m, clustered, persistent)
    kernel = PolicyKernel(A, edits, cached=False)
    try:
        z, rho = kernel.solve(s)
        signs = {}
        for edit_id in seq:
            lower, upper = kernel._interval(edit_id, z, rho)
            signs[edit_id] = -1 if upper < 0 else int(lower > 0)
        return signs
    finally:
        kernel.close()


def check_trajectory(graph, m, runs):
    by_budget = {b: [] for b in BUDGETS}
    for (g, size, b, _, _), reps in runs.items():
        if (g, size) == (graph, m):
            by_budget[b].extend(json.loads(r["sequence"]) for r in reps.values())
    seq = next(iter(by_budget[40] or by_budget[20]), [])
    for b in BUDGETS:
        if any(s != seq[:b] for s in by_budget[b]):
            raise ValueError(f"decision mismatch at {graph} {m} {b}")
    return seq


def median_time(group, provisional):
    if len(group) != 10 and not provisional:
        raise ValueError(f"expected ten completed repetitions, found {len(group)}")
    if not group:
        return "--"
    t = median(float(r["seconds"]) for r in group.values())
    return f"{t:.3g}" if len(group) == 10 else f"{t:.3g}$^{{{len(group)}}}$"


def persistent_findings(summary, runs, external):
    largest = [r for r in summary if r["M"] == 2500 and r["b"] == 40]
    groups = [[r for r in largest if r["backend"] == backend] for backend in BACKENDS]
    speed = [geometric_mean(r["ratio"] for r in rows) for rows in groups]
    incl = [geometric_mean(r["inclusive_ratio"] for r in rows) for rows in groups]
    neg = [r["root_negative"] for r in summary]
    if min(neg) == max(neg):
        count = "four" if neg[0] == 4 else str(neg[0])
        negative = f"each trajectory selected {count} edits with strictly negative root gain"
    else:
        negative = f"trajectories selected {min(neg)}--{max(neg)} edits with strictly negative root gain"
    total = 0
    outside = defaultdict(list)
    for graph in NAMES:
        for m in SIZES:
            cold = runs[graph,m,40,"cholmod","cold"]
            for t, k in enumerate(json.loads(next(iter(cold.values()))["k_rounds"])[20:], 21):
                total += 1
                if not 30 <= k <= 60:
                    outside[graph,t,k].append(m)
    exceptions = ", ".join(
        f"{g} round {t} had {k} at " + ("both sizes" if sizes == list(SIZES)
                                           else "$M=" + ",".join(map(str,sizes)) + "$")
        for (g,t,k),sizes in outside.items())
    in_range = total - sum(map(len,outside.values()))
    exception_text = f" ({exceptions})" if exceptions else ""
    def summed(key):
        return [sum(r[key] for r in rows) for rows in groups]
    cold, planned = summed("cold_refinements"), summed("planner_refinements")
    rhs_c, rhs_p = summed("cold_rhs"), summed("planner_rhs")
    corrected = summed("state_corrections")
    measured = [[r[label+"_over_j"] for r in external.values() if r[label+"_hi"]!=""] for label in EXTERNAL]
    wins = [sum(value > 1 for value in values) for values in measured]
    comparison = (f"J beat Q/L/D on median time in {wins[0]}/{wins[1]}/{wins[2]} of 24 cells. "
                  if all(len(values)==24 for values in measured) else
                  "J beat Q/L/D on median time in " + ", ".join(f"{n}/{len(values)}" for n,values in zip(wins,measured))
                  + " completed cells, respectively. ")
    q_min = min(r["q_over_j"] for r in external.values())
    q_max = max(r["q_over_j"] for r in external.values())
    q_unprofiled = min(r["q_over_unprofiled_j"] for r in external.values())
    fin = external["Fin",2500,40]
    slash = [external["Slashdot",m,20] for m in SIZES]
    return (f"All paths agreed and {negative}. Cold CHOLMOD refined 30--60 candidates in "
            f"{in_range}/{total} late rounds{exception_text}. At $M=2500,b=40$, graph-geometric "
            f"paired C/P ratios were {speed[0]:.2f} (CG) and {speed[1]:.2f} (CHOLMOD), "
            f"or {incl[0]:.2f}/{incl[1]:.2f} with catalog setup. The planners used "
            f"{planned[0]:,}/{planned[1]:,} of {cold[0]:,} cold refinements and {rhs_p[0]:,}/{rhs_p[1]:,} of "
            f"{rhs_c[0]:,} response RHS including winners and refreshes. Certified corrections "
            f"replaced {corrected[0]}/{corrected[1]} of {summed('cold_state_rhs')[0]} state solves. "
            + comparison +
            f"Q/J ranged from {q_min:.2f} to {q_max:.2f} (minimum {q_unprofiled:.2f} including profiling). "
            f"Fin J/L/D took {fin['j_seconds']:.3f}/{fin['l_seconds']:.3f}/{fin['d_seconds']:.3f} s at the largest "
            f"setting. Slashdot $b=20$ Q/J was "
            f"{slash[0]['q_over_j']:.2f} and {slash[1]['q_over_j']:.2f} in increasing catalog size.\n")

def rhs(counter):
    return sum(counter.get(k + "_rhs", 0) for k in "CFWR")


def matched_reps(group, ref, scheduled):
    return [i for i in sorted(group.keys() & ref.keys())
            if not scheduled or group[i].get("schedule") and group[i]["schedule"] == ref[i].get("schedule")]


def paired_summary(groups, fixture, scheduled=False):
    cold, same, planner = groups
    pairs = matched_reps(cold, planner, scheduled)
    if not pairs:
        return {}
    cc = json.loads(cold[pairs[0]]["counters"])
    pc = json.loads(planner[pairs[0]]["counters"])
    timings = [(float(cold[r]["seconds"]), float(planner[r]["seconds"])) for r in pairs]
    ratios = sorted(c / p for c, p in timings)
    result = dict(
        pairs=len(pairs), cold_seconds=median(float(r["seconds"]) for r in cold.values()),
        same_seconds=median(float(r["seconds"]) for r in same.values()) if same else "",
        planner_seconds=median(float(r["seconds"]) for r in planner.values()), ratio=median(ratios),
        lo=ratios[1] if len(ratios)==10 else "", hi=ratios[-2] if len(ratios)==10 else "",
        cold_rhs=rhs(cc), planner_rhs=rhs(pc), saved_rhs=rhs(cc)-rhs(pc),
        net_pct=100*(1-rhs(pc)/rhs(cc)), fixture_seconds=fixture,
        inclusive_ratio=median((fixture+c)/(fixture+p) for c,p in timings),
        gate_requests=pc.get("gate_requests",0), gate_opens=pc.get("gate_opens",0),
        winner_rhs=pc.get("W_rhs",0), forecast_rows=pc.get("forecast_rows",0),
        forecast_realized=pc.get("forecast_realized",0))


    if scheduled:
        result.update(cold_refinements=sum(cc.get(k,0) for k in "CFR"),
                      planner_refinements=sum(pc.get(k,0) for k in "CFR"),
                      cold_state_rhs=cc["state_rhs"],planner_state_rhs=pc["state_rhs"],
                      state_corrections=pc.get("state_warm_hits",0))
    return result

def independent_ratio(group, ref):
    import numpy as np
    x = np.array([float(group[i]["seconds"]) for i in sorted(group)])
    y = np.array([float(ref[i]["seconds"]) for i in sorted(ref)])
    if not len(x) or not len(y):
        return "", "", ""
    point = np.median(x)/np.median(y)
    if min(len(x), len(y)) < 10:
        return point, "", ""
    rng = np.random.default_rng(719)
    sample_x = np.median(rng.choice(x,(20000,len(x))),axis=1)
    sample_y = np.median(rng.choice(y,(20000,len(y))),axis=1)
    lo, hi = np.quantile(sample_x/sample_y,[.025,.975])
    return point, lo, hi


def routed_group(graph, m, b, runs, routes):
    if routes[graph][0] == "cg":
        return runs[graph,m,b,"native","planner"]
    return runs[graph,m,b,"direct6_slots","slots6"]


def external_cell(graph, m, b, runs, routes):
    backend, profile_seconds = routes[graph]
    ref = routed_group(graph,m,b,runs,routes)
    cell = dict(graph=graph,M=m,b=b,j_backend="native_cg" if backend=="cg" else backend,
                j_seconds=median(float(r["seconds"]) for r in ref.values()) if ref else "")
    for label, method in EXTERNAL.items():
        group = runs[graph,m,b,method,method]
        cell[label+"_seconds"] = median(float(r["seconds"]) for r in group.values()) if group else ""
    for label, method in EXTERNAL.items():
        point, lo, hi = independent_ratio(runs[graph,m,b,method,method],ref)
        cell.update({label+"_over_j":point,label+"_lo":lo,label+"_hi":hi})
    cell["profile_seconds"] = profile_seconds
    for label in EXTERNAL:
        seconds = cell[label+"_seconds"]
        cell[label+"_over_unprofiled_j"] = seconds/(cell["j_seconds"]+profile_seconds) if seconds and ref else ""
    return cell

def clustered_method_summary(runs, prefix="clustered"):
    """Retain checkpoint costs and compare every method with the direct planner."""
    rows = []
    for (graph, m, b, backend, mode), group in sorted(runs.items()):
        if not group:
            continue
        ref = runs[graph,m,b,"cholmod","planner"]
        pairs = matched_reps(group, ref, prefix=="persistent")
        ratios = sorted(float(group[i]["seconds"])/float(ref[i]["seconds"]) for i in pairs)
        point, lo, hi = (median(ratios) if ratios else "", ratios[1] if len(ratios)==10 else "",
                         ratios[-2] if len(ratios)==10 else "")
        external = (backend in EXTERNAL.values() or backend in ("native","direct6","direct6_slots","lookup") or (prefix=="clustered" and graph=="Slashdot" and backend=="cg")
                    or (prefix=="persistent" and not pairs))
        if external:
            point, lo, hi = independent_ratio(group,ref)
        counters = [json.loads(r["counters"]) for r in group.values()]
        def checkpoint_median(key):
            values = [int(r[key]) for r in group.values() if r.get(key)]
            return median(values) if values else ""
        times = sorted(float(r["seconds"]) for r in group.values())
        rows.append(dict(graph=graph,M=m,b=b,backend=backend,mode=mode,reps=len(times),
                         seconds=median(times),seconds_lo=times[1] if len(times)==10 else "",
                         seconds_hi=times[-2] if len(times)==10 else "",pairs="" if external else len(pairs),
                         over_cholmod_planner=point,
                         ratio_lo=lo,ratio_hi=hi,
                         memory_bytes=checkpoint_median("memory_bytes"),
                         factor_peak_bytes=checkpoint_median("factor_peak_bytes"),
                         counters=json.dumps({k:median(c.get(k,0) for c in counters)
                             for k in sorted(set().union(*counters))},sort_keys=True)))
    return rows


def report_text(path, content, check):
    if check:
        if path.read_text() != content:
            raise ValueError(f"stored report differs from measurements: {path}")
    else:
        path.write_text(content)


def report_csv(path, rows, check):
    if rows:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        report_text(path, buffer.getvalue(), check)

def findings(summary, provisional, controls):
    if provisional:
        complete = sum(r["pairs"]==10 for r in summary)
        return (f"Measurements are in progress. Superscripts give available sample sizes below ten. "
                f"{complete} of 48 graph, size, budget and backend comparisons currently have "
                "ten matched pairs. The figure displays measured pairs only.\n")
    largest = {backend: [r for r in summary if r["backend"]==backend and r["M"]==2500 and r["b"]==40]
               for backend in BACKENDS}
    ratios = [[r["ratio"] for r in largest[backend]] for backend in BACKENDS]
    speed = [f"{geometric_mean(x):.2f} (range {min(x):.2f}--{max(x):.2f})" for x in ratios]
    inclusive = [geometric_mean(r["inclusive_ratio"] for r in largest[b]) for b in BACKENDS]
    wins = [sum(r["ratio"]>1 for r in summary if r["backend"]==b) for b in BACKENDS]
    saved = [sum(r["saved_rhs"] for r in largest[b]) for b in BACKENDS]
    total = sum(r["cold_rhs"] for r in largest["cg"])
    paragraphs = [
        f"At $M=2500,b=40$ the equal-graph geometric mean of the median paired cold/planner "
        f"ratios was {speed[0]} for CG and {speed[1]} for CHOLMOD. Charging both methods the "
        f"median fixture preparation cost gives ratios of {inclusive[0]:.2f} and {inclusive[1]:.2f}. "
        f"The planner reduced median paired time in {wins[0]}/24 CG and {wins[1]}/24 CHOLMOD settings. "
        f"At the largest setting, mean cold demand was {mean(r['K'] for r in largest['cg']):.1f} "
        f"refinements per edit. Of {total:,} response right-hand sides, including winner "
        f"solves and refreshes, CG saved {saved[0]:,} and CHOLMOD saved {saved[1]:,}."]
    direct = [r for r in summary if r["backend"]=="cholmod" and r["b"]==40]
    forecast = sum(r["forecast_rows"] for r in direct)
    realized = sum(r["forecast_realized"] for r in direct)
    paragraphs.append(f"Across twelve CHOLMOD trajectories, {realized:,} of {forecast:,} forecast "
                      f"requests materialized ({forecast-realized:,} fewer than forecast). "
                      "The gate buys no historical winner solve for a singleton request.")
    winners = defaultdict(list)
    for graph in NAMES:
        times = {"Quad-Refine":median(float(r["seconds"]) for r in controls[graph,2500,40,"quad","quad"].values())}
        if (graph,2500,40,"lv","lv") in controls:
            times["LINVIEW-Delta"] = median(float(r["seconds"]) for r in controls[graph,2500,40,"lv","lv"].values())
        for backend in BACKENDS:
            cell = next(r for r in largest[backend] if r["graph"]==graph)
            for mode in MODES:
                times[backend.upper()+" "+mode] = cell[mode+"_seconds"]
        winners[min(times,key=times.get)].append(graph)
    mem = [int(controls["Slashdot",2500,40,b,"planner"][0]["memory_bytes"])/2**20 for b in BACKENDS]
    factor = int(controls["Slashdot",2500,40,"cholmod","planner"][0]["factor_peak_bytes"])/2**20
    def graph_list(graphs):
        if len(graphs) < 3:
            return " and ".join(graphs)
        return ", ".join(graphs[:-1])+", and "+graphs[-1]
    paragraphs.append("At the largest setting the fastest median was "+
                      ", ".join(method+" on "+graph_list(graphs) for method,graphs in winners.items())+". "
                      f"After 40 Slashdot commits the planner retained {mem[0]:.0f} MiB for CG and "
                      f"{mem[1]:.0f} MiB for CHOLMOD, counting matrix arrays, cached payloads and the "
                      f"mutable factor (allocator peak {factor:.0f} MiB), excluding the shared root and Python overhead.")
    if (NAMES[0],2500,40,"lv","lv") in controls:
        neg = [r["root_negative"] for r in summary if r["backend"]=="cg"]
        negative_count = str(min(neg)) if min(neg)==max(neg) else f"{min(neg)}--{max(neg)}"
        fastest = ", ".join(f"{method} on {graph_list(graphs)}" for method,graphs in winners.items())
        return (f"At $M=2500,b=40$, equal-graph geometric mean cold/planner ratios were "
                f"{geometric_mean(ratios[0]):.2f} for CG and {geometric_mean(ratios[1]):.2f} "
                f"for CHOLMOD ({inclusive[0]:.2f} and {inclusive[1]:.2f} including catalog preparation). "
                f"The planner saved {saved[0]:,} of {total:,} cold response right-hand sides for CG "
                f"and {saved[1]:,} of {sum(r['cold_rhs'] for r in largest['cholmod']):,} for CHOLMOD, "
                f"counting winner solves and refreshes. Across both sizes and budgets, paired median time "
                f"fell in {wins[0]}/24 CG and {wins[1]}/24 CHOLMOD settings. "
                f"All eight methods agreed. Among accepted edits, the count with certified negative root gain "
                f"was {negative_count} per trajectory across 24 settings. At the largest setting the fastest "
                f"median was {fastest}.\n")
    return "\n\n".join(paragraphs)+"\n"


def render_figure(summary, trajectory, controls=None, check=False, external=None):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype":42,"font.size":8})
    if controls is None:
        fig, axes = plt.subplots(1,3,figsize=(7.15,2.45),layout="constrained",
                                 gridspec_kw={"width_ratios":[1,1,1.15]})
    else:
        fig, axes = plt.subplots(1,4,figsize=(7.15,2.6),layout="constrained")
        compare = axes[3]
        series = ([("q","x","Quad"),("l","o","Registered"),("d","+","Delayed")] if external is not None
                  else [("quad","x","Quad"),("lv","+","LINVIEW")])
        for j,(backend, mark, label) in enumerate(series):
            for i, graph in enumerate(NAMES):
                if external is not None:
                    row = external[graph,2500,40]
                    y,lo,hi = [row[backend+suffix] for suffix in ("_over_j","_lo","_hi")]
                else:
                    group = controls.get((graph,2500,40,backend,backend),{})
                    ref = controls.get((graph,2500,40,"cholmod","planner"),{})
                    y, lo, hi = independent_ratio(group,ref)
                if y != "":
                    ends = [[y-lo],[hi-y]] if lo != "" else None
                    compare.errorbar(i+(j-1)*.14,y,yerr=ends,fmt=mark,color=f"C{j}",
                                     label=label if i==0 else None,lw=.7,ms=3)
        compare.set_yscale("log")
        compare.axhline(1,color="gray",lw=.8)
        compare.set_xticks(range(len(NAMES)),NAMES,rotation=35,ha="right",fontsize=7)
        compare.set_title("Time / routed J" if external is not None else "Time / direct P")
        if external is None and compare.get_legend_handles_labels()[0]:
            compare.legend(fontsize=6)
    axes[1].sharey(axes[0])
    axes[1].tick_params(labelleft=False)
    conditions = [(1000,20,"o",-.24),(1000,40,"s",-.08),(2500,20,"^",.08),(2500,40,"D",.24)]
    for ax, backend in zip(axes,BACKENDS):
        for m, b, mark, offset in conditions:
            part = [r for r in summary if (r["backend"],r["M"],r["b"])==(backend,m,b)]
            ax.errorbar([NAMES.index(r["graph"])+offset for r in part], [r["ratio"] for r in part],
                        yerr=[[r["ratio"]-r["lo"] if r["lo"]!="" else 0 for r in part],
                              [r["hi"]-r["ratio"] if r["hi"]!="" else 0 for r in part]],
                        label=f"{m}, {b}",fmt=mark,markersize=3,lw=.8,capsize=1)
        ax.axhline(1,color="gray",lw=.8)
        ax.set_xticks(range(len(NAMES)),NAMES,rotation=35,ha="right")
        ax.set_title(backend.upper())
    axes[0].set_ylabel("Cold / planned wall time")
    ax = axes[2]
    ax.set_xlim(-.6,3.55)
    ax.set_ylim(5.5,-.5)
    ax.set_xticks(range(4),["1k/20","1k/40","2.5k/20","2.5k/40"],rotation=30,ha="right",fontsize=6)
    ax.set_yticks(range(6),NAMES,fontsize=7)
    ax.tick_params(length=0)
    ax.set_title("Accepted flips / edits")
    for col,(m,b,_,_) in enumerate(conditions):
        for i,graph in enumerate(NAMES):
            ax.text(col,i,trajectory[graph,m,b][1],ha="center",va="center",fontsize=6.5)
    handles, labels = axes[0].get_legend_handles_labels()
    if external is not None:
        methods, names = compare.get_legend_handles_labels()
        handles.extend(methods)
        labels.extend(names)
    fig.legend(handles,labels,loc="outside upper center",ncols=len(labels),
               title="Catalog size and budget (left)     External method (right)" if external is not None
               else "Catalog size, budget",fontsize=7)
    path = ROOT/"figures/baseline_ratios.pdf"
    target = io.BytesIO() if check else path
    fig.savefig(target,format="pdf",metadata={"CreationDate":None})
    plt.close(fig)
    if check and target.getvalue() != path.read_bytes():
        raise ValueError(f"stored figure differs from measurements: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-routes",action="store_true",
                        help="time root symbolic analysis once per graph and store the routing catalogue")
    parser.add_argument("--provisional", action="store_true",
                        help="display available repetitions with sample sizes while the matrix runs")
    fixtures = parser.add_mutually_exclusive_group()
    fixtures.add_argument("--clustered", action="store_true")
    fixtures.add_argument("--persistent", action="store_true")
    parser.add_argument("--check",action="store_true",
                        help="compare final summary CSVs, table, findings and figure without writing")
    args = parser.parse_args()
    if args.profile_routes:
        profile_routes()
        return
    if args.check and (not args.persistent or args.provisional):
        parser.error("--check requires --persistent and a complete matrix")
    prefix = "persistent" if args.persistent else "clustered" if args.clustered else "planned"
    with_external = args.clustered or args.persistent
    runs = read_runs("persistent_batch" if args.persistent else prefix + "_fast")
    runs.update(read_runs(prefix + "_quad"))
    if with_external:
        runs.update(read_runs("linview_" + prefix))
    external = {} if args.persistent else None
    lookups = []
    if args.persistent:
        runs.update(read_runs("linview_registered_persistent"))
        runs.update(read_runs("linview_registered_lookup_persistent"))
        runs.update(read_runs("persistent_native"))
        runs.update(read_runs("direct6"))
        runs.update(read_runs("direct6_slots"))
        routes = catalogue_routes()
    setup_file = prefix + "_setup.csv" if with_external else "matched_setup.csv"
    with (ROOT/"datasets"/setup_file).open() as f:
        setup = {(r["graph"],int(r["M"])):float(r["seconds"]) for r in csv.DictReader(f)}
    summary, trajectory = [], {}
    for graph in NAMES:
        for m in SIZES:
            seq = check_trajectory(graph,m,runs)
            signs = root_signs(graph,m,seq,args.clustered,args.persistent) if seq else {}
            for b in BUDGETS:
                ref = seq[:b]
                neg = sum(signs[e]<0 for e in ref)
                unresolved = sum(signs[e]==0 for e in ref)
                times = []
                demand = None
                flips = None
                for backend in BACKENDS:
                    groups = [runs[graph,m,b,backend,mode] for mode in MODES]
                    for g in groups:
                        times.append(median_time(g,args.provisional))
                        if g:
                            flips = int(next(iter(g.values()))["flips"])
                    if groups[0]:
                        counter = json.loads(next(iter(groups[0].values()))["counters"])
                        demand = sum(counter.get(k,0) for k in "CFR")/len(ref) if ref else None
                    stats = paired_summary(groups,setup[graph,m],args.persistent)
                    if args.persistent and not args.provisional and stats.get("pairs",0) != 10:
                        raise ValueError(f"expected ten co-scheduled cold/planner pairs at {graph}/{m}/{backend}/{b}")
                    if stats:
                        if args.persistent:
                            by_round = [json.loads(row["k_rounds"]) for row in groups[0].values()]
                            if any(v != by_round[0] for v in by_round):
                                raise ValueError(f"inconsistent cold demand at {graph}/{m}/{b}/{backend}")
                            stats.update(cold_K_1_20=mean(by_round[0][:20]),
                                         cold_K_21_40=mean(by_round[0][20:]) if by_round[0][20:] else "")
                        summary.append(dict(graph=graph,M=m,b=b,backend=backend,K=demand,flips=flips,
                                            commits=len(ref),root_negative=neg,root_unresolved=unresolved,**stats))
                times.append(median_time(runs[graph,m,b,"quad","quad"],args.provisional))
                if args.persistent:
                    times.append(median_time(runs[graph,m,b,"registered","registered"],args.provisional))
                if with_external:
                    times.append(median_time(runs[graph,m,b,"lv","lv"],args.provisional))
                trajectory[graph,m,b] = [f"{demand:.1f}" if demand is not None else "--",
                                          f"{flips}/{len(ref)}" if flips is not None else "--",
                                          str(neg) if seq else "--",*times]
                if args.persistent:
                    cold = runs[graph,m,b,"cholmod","cold"]
                    values = json.loads(next(iter(cold.values()))["k_rounds"]) if cold else []
                    halves = [f"{mean(part):.1f}" if part else "--"
                              for part in (values[:20],values[20:])]
                    trajectory[graph,m,b][0] = "/".join(halves)
                    cell = external_cell(graph,m,b,runs,routes)
                    external[graph,m,b] = cell
                    trajectory[graph,m,b].append(median_time(routed_group(graph,m,b,runs,routes),args.provisional))
                    for label in EXTERNAL:
                        ratio = cell[label+"_over_j"]
                        trajectory[graph,m,b].append(f"{ratio:.2f}" if ratio != "" else "--")
                    if graph in ("Alpha","Fin"):
                        lookup = runs[graph,m,b,"lookup","lookup"]
                        median_time(lookup,args.provisional)
                        point,lo,hi = independent_ratio(lookup,routed_group(graph,m,b,runs,routes))
                        if lookup:
                            lookups.append(dict(graph=graph,M=m,b=b,reps=len(lookup),
                                lookup_seconds=median(float(r["seconds"]) for r in lookup.values()),
                                j_seconds=cell["j_seconds"],lookup_over_j=point,lo=lo,hi=hi))
    if args.persistent:
        report_csv(ROOT/"data/persistent_external.csv",list(external.values()),args.check)
        report_csv(ROOT/"data/persistent_lookup.csv",lookups,args.check)
    if with_external:
        report_csv(ROOT/"data"/(prefix+"_methods.csv"),clustered_method_summary(runs,prefix),args.check)
    table = []
    for b in BUDGETS:
        table.append(r"\multicolumn{12}{l}{\textit{Budget $b="+str(b)+r"$}} \\")
        if with_external:
            table[-1] = table[-1].replace("{12}", "{18}" if args.persistent else "{13}")
        for graph in NAMES:
            for m in SIZES:
                name = graph + ("/G" if routes[graph][0]=="cg" else "/H") if args.persistent else graph
                table.append(" & ".join([name if m==1000 else "",str(m),*trajectory[graph,m,b]])+r" \\")
    report_csv(ROOT/"data"/(prefix+"_summary.csv"),summary,args.check)
    # The wrapper selects matching captions and columns before expanding these rows.
    header = rf"\def\fixturetablecolumns{{{18 if args.persistent else 13 if with_external else 12}}}"
    rows = "\n".join([header,r"\def\fixturerows{",*table,"}"])
    report_text(ROOT/"sections/orthogonal_rows.tex",rows+"\n",args.check)
    complete_core = (args.persistent and len(summary)==48 and all(r["pairs"]==10 for r in summary)
                     and all(r["q_hi"]!="" for r in external.values()) and external["Fin",2500,40]["l_hi"]!="")
    report = (persistent_findings(summary,runs,external) if complete_core
              else findings(summary,args.provisional,runs))
    report_text(ROOT/"sections/orthogonal_findings.tex",report,args.check)
    render_figure(summary,trajectory,runs if with_external else None,args.check,external)
    if args.check:
        print("Persistent methods, paired summaries, table, findings and figure match the raw checkpoints.")


if __name__=="__main__":
    main()






