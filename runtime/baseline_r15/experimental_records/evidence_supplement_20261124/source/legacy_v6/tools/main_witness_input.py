"""Trusted Alpha workload adapters and external published-row bindings."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
from workloads import opinions, contention_candidates

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
CASES = ("universe6","persistent6","universe5","persistent5","mixed", "clamp", "clustered", "branch", "motif")
FULL_FAMILY = {"universe6":"Alpha","universe5":"Alpha","mixed":"Alpha", "motif":"AlphaMotif", "clamp":"AlphaClamp", "promotion":"Alpha"}


def instance(case, *, emitting=False):
    graph = load_npz(BASE / "datasets/Alpha.npz").tocsr()
    if case in ("branch","s2","clustered","persistent5","persistent6"):
        if case=="s2":
            s = opinions(graph, "S2", 17)
            edits = contention_candidates(graph, 1000, s, seed=17)
        elif case in ("clustered","persistent5","persistent6"):
            from workloads import clustered_inputs,persistent_inputs
            constructor = persistent_inputs if case in ("persistent5","persistent6") else clustered_inputs
            s, edits = constructor(graph,1000,seed=17)
        else:
            from workloads import matched_inputs
            s, edits = matched_inputs(graph,1000,seed=17)
        i, j, weight, flip = map(np.asarray, zip(*edits))
        return graph, s, (i, j, weight, flip), 20
    if case in ("motif","clamp","universe5","universe6"):
        # Rebuild the public integer forcing without calling the search/index.
        from scipy.sparse import diags
        A = diags(1+np.asarray(abs(graph).sum(axis=1)).ravel(), dtype="int64")-graph
        negative = (graph<0).tocsr()
        degree = A.diagonal()
        a = np.flatnonzero((np.diff(negative.indptr)==2)&(degree<7))[0]
        b,c = sorted(negative[a].indices,key=lambda v:(degree[v],v))
        sites = [a,b,c]
        target = [round(1.5*512/degree[a]),-512,96]
        if case in ("clamp","universe5","universe6"):
            from scipy.sparse.linalg import spsolve
            signed = np.rint(65536*opinions(graph,"S1",17))
            state = spsolve(A.tocsc(),signed)
            factor = 3/max(abs(state))
            rhs = factor*signed + A[:,sites] @ (target-factor*state[sites])
        else:
            z = np.random.default_rng(17).integers(-3,4,size=graph.shape[0])
            z[sites] = target
            rhs = A @ z
        s = np.rint(rhs*min(1,65536/max(abs(rhs)))) / 65536
    elif case == "mixed":
        negative = (graph<0).tocsr()
        i = np.flatnonzero(np.diff(negative.indptr))[0]
        j = min(negative[i].indices)
        s = .75*opinions(graph,"S1",17)
        s[i] += .25
        s[j] -= .25
        s = np.rint(65536*s)/65536
    elif case == "promotion":
        s = np.rint(65536 * opinions(graph, "S1", 17)) / 65536
    else:
        raise ValueError(case)
    if emitting:
        from universe_index import build_universe
        i, j, flip = build_universe(graph)
    else:
        from cold_witness import inputs
        _, edges = inputs(BASE / "datasets/Alpha.npz", -1)
        i, j, flip = map(np.asarray, zip(*edges))
    return graph, s, (i, j, np.where(flip, 1., 1/16), flip), 15


def records(filename):
    with (DATA / filename).open(newline="") as stream:
        return list(csv.DictReader(stream))


def complete(rows, fields, keys):
    actual = [tuple(row[k] for k in fields) for row in rows]
    if len(actual) != len(keys) or set(actual) != keys:
        raise ValueError(f"incomplete or duplicate published rows for {fields}")


def expected(case,pool=None):
    repeats = [str(r) for r in range(10)]
    if case in ("branch","s2"):
        filename = "branch_s2_alpha.csv" if case=="s2" else "branch_move_alpha.csv"
        rows = records(filename)
        complete(rows, ("rep","level"), {(r,str(l)) for r in repeats for l in range(4)})
        sequences = [json.loads(row["paths"])[0] for row in rows]
    elif case in ("clustered","persistent5","persistent6"):
        sequences = []
        methods = {"clustered_fast": [(backend,mode) for backend in ("cg","cholmod")
                                      for mode in ("cold","same","planner")],
                   "clustered_quad": [("cg","quad")], "linview_clustered":[("cholmod","linview")]}
        if case in ("persistent5","persistent6"):
            methods = {"persistent_batch":methods["clustered_fast"],
                       "persistent_quad":methods["clustered_quad"],
                       "linview_persistent":methods["linview_clustered"]}
        if case=="persistent6":
            methods["direct6_slots"] = [("cholmod","slots6")]
            methods["linview_registered_persistent"] = [("cholmod","linview_registered")]
        for prefix, modes in methods.items():
            rows = [row for row in records(f"{prefix}_Alpha_1000.csv") if row["b"]=="20"]
            if prefix=="direct6_slots":
                rows = [r for r in rows if r["mode"]=="slots6"]
            complete(rows, ("rep","backend","mode"), {(r,b,m) for r in repeats for b,m in modes})
            if any(row["graph"]!="Alpha" or row["M"]!="1000" or row["commits"]!="20" for row in rows):
                raise ValueError("clustered query parameters")
            if prefix=="linview_registered_persistent":
                for row in rows:
                    counters = json.loads(row["counters"])
                    if (row["population_reused"]!="False" or
                        [counters.get(k) for k in ("registered_rows_populated",
                         "registered_rows_updated","registered_scores")] != [1000,20000,19810]):
                        raise ValueError("all-active registered score ledger")
            if prefix=="direct6_slots":
                for row in rows:
                    if (row["catalog_seed"],row["opinion_seed"])!=("17","17") or json.loads(row["counters"])["screens"]!=19810:
                        raise ValueError("routed planner query")
            if prefix=="persistent_batch":
                for row in rows:
                    if row["mode"]=="planner":
                        counts = json.loads(row["counters"])
                        if counts.get("batch_consumed",0)<=0 or counts["batch_computed"]!=(
                                counts["batch_consumed"]+counts.get("batch_unused",0)):
                            raise ValueError("published batch ledger")
            sequences.extend(json.loads(row["sequence"]) for row in rows)
    elif case in FULL_FAMILY:
        legacy = case=="promotion"
        name = FULL_FAMILY[case]
        filename = "signed_promotion_Alpha_rounds.csv" if legacy else f"promotion_v{5 if case=='universe5' else 4}_{name}{'_fine' if case=='universe5' else ''}_rounds.csv"
        if case=="universe6":
            filename = f"promotion_v6_{name}_rounds.csv"
        methods = ("promoted","full") if legacy else ("adaptive","fixed","full")
        if case=="universe5":
            methods = ("adaptive","full","transport")
        if case=="universe6":
            methods = ("adaptive","full","transport")
        rep_field = "repetition" if legacy else "query"
        rows = [row for row in records(filename) if row["method"] in methods]
        complete(rows,(rep_field,"method","epoch"),
                 {(r,m,str(t)) for r in repeats for m in methods for t in range(1,16)})
        groups = {}
        for row in rows:
            k=int(row["winner_id"])
            if row["graph"]!=name or (not legacy and (row["seed"]!="17" or
                int(row["proof_unit"])!=(1<<48 if case in ("motif","clamp","universe5","universe6") else 1<<40))):
                raise ValueError("full-universe query parameters")
            if pool is not None and (not 0<=k<len(pool[0]) or int(row["universe"])!=len(pool[0]) or
                    (int(row["i"]),int(row["j"]),int(row["flip"])) !=
                    (int(pool[0][k]),int(pool[1][k]),int(pool[3][k]))):
                raise ValueError("catalog/trace mismatch")
            if case=="universe5":
                method = row["method"]
                if int(row["transport_enabled"]) != int(method!="full"):
                    raise ValueError("full-universe transport policy")
                if method in ("full","transport") and row["catalog_visits"]!=row["universe"]:
                    raise ValueError("full scan coverage")
            if case=="universe6" and (int(row["transport_enabled"])!=int(row["method"]!="full") or int(row["strict_rows"])!=int(row["seed_visits"])+int(row["catalog_visits"]) or (row["method"] in ("full","transport") and int(row["strict_rows"])!=int(row["universe"])-int(row["epoch"])+1)):
                raise ValueError("indexed universe parameters")
            groups.setdefault((row[rep_field],row["method"]),[]).append((int(row["epoch"]),k))
        sequences = [[k for _,k in sorted(group)] for group in groups.values()]
    else:
        raise ValueError(case)
    if not sequences or any(x != sequences[0] for x in sequences):
        raise ValueError("inconsistent main performance sequences")
    return sequences[0]


def bind_gain_intervals(case, result):
    """Prove containment in the published per-edit and cumulative intervals."""
    family = FULL_FAMILY[case]
    methods = ("adaptive","full","fixed")
    version = 5 if case=="universe5" else 4
    if version==5:
        methods = ("adaptive","full","transport")
    if case=="universe6":
        version = 6
        methods = ("adaptive","full","transport")
    den = result["denominator"]
    for table in ("rounds","runs"):
        rows = [r for r in records(f"promotion_v{version}_{family}{'_fine' if version==5 else ''}_{table}.csv") if r["method"] in methods]
        if table=="runs":
            complete(rows,("query","method"),{(str(q),m) for q in range(10) for m in methods})
        for row in rows:
            if row["graph"]!=family or row["seed"]!="17" or int(row["proof_unit"])!=(1<<48 if case in ("motif","clamp","universe5","universe6") else 1<<40):
                raise ValueError("published gain parameters")
            if table=="runs" and (row.get("directional")!="1" or
                                   (case in ("mixed","clamp","universe5","universe6") and row.get("guarded")!="1")):
                raise ValueError("published solver policy")
            if case=="universe6" and table=="runs":
                if (row["strict_screen"],row["reuse_certified_rows"],row["fused_threshold"])!=("1","1",str(int(row["method"]=="adaptive"))):
                    raise ValueError("published strict-screen policy")
            if table=="rounds":
                lo,hi = result["gains"][int(row["epoch"])-1]
                reported_lo,reported_hi = int(row["low"]),int(row["high"])
            else:
                lo,hi = result["objective"]
                reported_lo,reported_hi = int(row["gain_low"]),int(row["gain_high"])
            scale = int(row["proof_unit"])**2
            if not reported_lo*den<=lo*scale<=hi*scale<=reported_hi*den:
                raise ValueError("independent gain outside published interval")
    if case in ("motif","clamp","universe5","universe6"):
        winner = result["epochs"][1]["winner"]
        target = [r for r in records(f"promotion_v{version}_{family}{'_fine' if version==5 else ''}_admissions.csv")
                  if int(r["candidate"])==winner]
        if len(target)!=1 or (target[0]["epoch"],target[0]["flip"],target[0]["revived_at_admission"])!=("2","1","1"):
            raise ValueError("published revival admission")
        row = target[0]
        scale = int(row["state_denominator"])**2
        for (lo,hi),a,b in ((result["root_gain"],"root_gain_low","root_high"),
                            (result["gains"][1],"admission_low","admission_high")):
            if not int(row[a])*den<=lo*scale<=hi*scale<=int(row[b])*den:
                raise ValueError("independent gain outside published revival interval")


def bind_routed_gains(gains):
    for row in records("direct6_slots_Alpha_1000.csv"):
        if row["b"]!="20" or row["mode"]!="slots6":
            continue
        low,high = (json.loads(row[k]) for k in ("winner_lower","winner_upper"))
        if len(low)!=len(gains) or len(high)!=len(gains) or any(
            not a<=l<=u<=b for a,(l,u),b in zip(low,gains,high)):
            raise ValueError("routed planner gain interval")
