"""Check both full-operator search cohorts and generate Table VI's right panel."""
import argparse
import csv
import json
from itertools import product

import numpy as np

from sequencing import DATA

POLICIES = ("cold", "transport")


def records(file):
    with (DATA/file).open() as stream:
        return list(csv.DictReader(stream))


def summarize(graph):
    rows = []
    for task, mode, datafile, fixturefile in (
        ("Deep", "dfs", "sparse_deep", "full_sequencing"),
        ("Trap", "beam", "sparse_trap_cohort", "sparse_trap_cohort"),
    ):
        cases = json.loads((DATA/f"{fixturefile}_{graph}.json").read_text())
        assert len(cases) == 10
        identities = {tuple(sorted(tuple(edit) for edit in c["edits"])) for c in cases}
        assert len(identities) == 10
        if task == "Trap":
            anchor = json.loads((DATA/f"sparse_trap_{graph}.json").read_text())
            assert all(len(c["edits"])==15 for c in cases)
            assert len({tuple(sorted(map(tuple,c["edits"][:2]))) for c in cases})==10
            assert all(set(map(tuple,c["edits"][:3]))!=set(map(tuple,anchor["edits"])) for c in cases)
        data = records(f"{datafile}_{graph}.csv")
        grid = {(int(r["case"]),int(r["repeat"]),r["mode"],r["policy"]):r for r in data}
        assert len(grid) == len(data) == 400
        assert set(grid) == set(product(range(10),range(10),(mode,"greedy"),POLICIES))
        success = {"greedy":0,mode:0}
        ratios, times, columns, gain_ratios, gains, states = [],[],[],[],[],[]
        costs = []
        for i,c in enumerate(cases):
            for m in success:
                ref = grid[i,0,m,"cold"]
                path = [int(j) for j in ref["path"].split()]
                assert len(set(path)) == len(path) and float(ref["gain"]) > 0
                if task == "Deep":
                    assert len(c["edits"]) == 15
                    if m == "dfs":
                        assert path == c["order"] and int(ref["backtracks"]) == c["backtracks"]
                    else:
                        assert sum(1<<j for j in path) == c["greedy_mask"]
                    success[m] += len(path)==15
                else:
                    assert c["budget"] == 2 and float(ref["optimum_gap"])>0
                    assert float(ref["root_e"])<0<float(ref["root_a"])<float(ref["root_d"])
                    assert float(ref["e_after_a"])>0
                    assert float(ref["secondary_e"])<0<float(ref["secondary_a"])
                    assert float(ref["secondary_after_a"])>0
                    assert int(ref["optimal"])==int(float(ref["gain"])==float(ref["optimum_gain"]))
                    success[m] += float(ref["gain"]) == float(ref["optimum_gain"])
                for p in POLICIES:
                    first = grid[i,0,m,p]
                    setup = sum(float(first[key]) for key in ("structure_s","root_factor_s","consumer_state_s","factor_clone_s"))
                    assert all(float(first[key])>0 for key in ("structure_s","root_factor_s","consumer_state_s","factor_clone_s"))
                    assert setup == float(first["setup_s"])
                    ten = sum(float(grid[i,rep,m,p]["seconds"]) for rep in range(10))
                    assert np.isclose(float(first["ten_query_s"]),setup+ten,rtol=1e-12)
                    assert np.isclose(float(first["first_query_s"]),setup+float(first["seconds"]),rtol=1e-12)
                    for rep in range(10):
                        r = grid[i,rep,m,p]
                        for field in ("structure_s","root_factor_s"):
                            assert r[field] == grid[i,0,mode,"cold"][field]
                        for field in ("consumer_state_s","factor_clone_s","setup_s","ten_query_s","first_query_s"):
                            assert r[field] == first[field]
                        assert r["updates"]==r["undos"] and r["factor_clones"]=="1" and r["factors"]=="1"
                        assert r["path"] == ref["path"] and r["gain"] == ref["gain"]
                        assert r["states"] == ref["states"] and r["state_rhs"] == ref["state_rhs"]
                        assert r["response_rhs"] == first["response_rhs"] and float(r["seconds"])>0
                        assert float(r["min_margin"])>0 and float(r["min_rank_slack"])>=0
                        assert (int(r["transported"])>0) == (p=="transport")
                        if task == "Deep":
                            assert r["factors"]=="1" and r["factor_clones"]=="1"
                            assert r["state_rhs"]==r["expansions"] and int(r["rank_checks"])>0
                        else:
                            assert r["optimum_gain"]==ref["optimum_gain"]
                            assert r["optimum_gap"]==ref["optimum_gap"]
            pairs = [[grid[i,r,mode,p] for p in POLICIES] for r in range(10)]
            secs = np.array([[float(r["seconds"]) for r in pair] for pair in pairs])
            costs.append([[float(row[key]) for row in pairs[0]] for key in
                          ("structure_s","root_factor_s","consumer_state_s","factor_clone_s",
                           "setup_s","first_query_s","ten_query_s")])
            ratios.append(np.median(secs[:,0]/secs[:,1]))
            times.append(np.median(secs,axis=0))
            counts = [int(r["response_rhs"]) for r in pairs[0]]
            assert counts[0]>counts[1]
            columns.append(counts)
            states.append(int(pairs[0][0]["states"]))
            greedy_gain = float(grid[i,0,"greedy","cold"]["gain"])
            search_gain = float(pairs[0][0]["gain"])
            gains.append((greedy_gain,search_gain))
            gain_ratios.append(search_gain/greedy_gain)
        ci = np.percentile(np.median(np.random.default_rng(408).choice(ratios,(10000,10)),axis=1),[2.5,97.5])
        ct,tt = np.median(times,axis=0)
        cf,cr = np.sum(columns,axis=0)
        gain = np.array(gain_ratios)
        print(graph,task,"greedy/search",success,"speed",np.median(ratios),"CI",ci,
              "ms",1000*ct,1000*tt,"cols",cf,cr,"states",min(states),max(states),
              "gain greedy/search",np.median(gains,axis=0),
              "gain ratio median/range",np.median(gain),min(gain),max(gain))
        assert success["greedy"] == 0
        gain_cell = "---"
        if mode == "beam":
            anchor_rows = records(f"sparse_trap_{graph}.csv")
            assert len(anchor_rows)==40
            anchor_gains = {}
            for policy_mode in ("beam","greedy"):
                values = {float(r["gain"]) for r in anchor_rows if r["mode"]==policy_mode}
                assert len(values)==1
                anchor_gains[policy_mode] = values.pop()
            anchor_ratio = anchor_gains["beam"]/anchor_gains["greedy"]
            gain_cell = f"{np.median(gain):.2f}/{anchor_ratio:.2f}"
        costs = np.array(costs)
        med = np.median(costs,axis=0)
        net10 = np.median(costs[:,-1,0]/costs[:,-1,1])
        print("structure/factor/state/clone/setup/first/ten ms fresh-transfer",med*1000,"net10",net10)
        rows.append((f"{graph} & {success[mode]}/10 & {cf}/{cr} & "
                     f"{1000*ct:.0f}/{1000*tt:.0f} & {np.median(ratios):.2f} & {gain_cell}" + r"\\",
                     f"{graph} & {1000*med[0,0]:.1f}/{1000*med[1,0]:.1f} & "
                     f"{1000*med[4,0]:.1f}/{1000*med[4,1]:.1f} & "
                     f"{1000*med[5,0]:.0f}/{1000*med[5,1]:.0f} & "
                     f"{med[6,0]:.2f}/{med[6,1]:.2f} & {net10:.2f}" + r"\\"))
    return rows


def render_table():
    by_graph = [summarize(graph) for graph in ("Alpha","OTC")]
    lines = []
    for i,label in enumerate(("A. DFS on 15-edit lockouts", "B. Two-step beam on two-motif catalogs")):
        if i:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{6}{@{}l}{\textit{" + label + r"}}\\")
        lines.append(r"Graph & Hits & RHS & Online ms & Speed & Gain\\")
        lines.extend(rows[i][0] for rows in by_graph)
        lines.append(r"Graph & Str./fac. & Setup & First & Ten s & Speed\\")
        lines.extend(rows[i][1] for rows in by_graph)
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    target = DATA.parent/"sections"/"sparse_search_rows.tex"
    table = render_table()
    if args.check:
        if target.read_text(encoding="utf8")!=table:
            raise SystemExit("Table VI right panel differs from verified measurements")
    else:
        target.write_text(table,encoding="utf8")

