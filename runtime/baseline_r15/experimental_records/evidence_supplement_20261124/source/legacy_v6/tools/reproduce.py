"""Regenerate the full-graph experiments or validate their shipped outputs offline."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
PAPER = TOOLS.parent
DATA = PAPER / "data"
GRAPHS = ("Alpha", "OTC", "Slashdot", "Conflict", "Bio", "Fin")


def run(*args):
    print("+", sys.executable, *args, flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MPLBACKEND="Agg", PYTHONOPTIMIZE="",
               PYTHONDONTWRITEBYTECODE="1")
    local_bridge = PAPER.parent / "chol/ucrt64/bin/jcert_diagnostic.dll"
    if local_bridge.is_file():
        env.setdefault("JCERT_CHOLMOD_LIB", str(local_bridge))
    subprocess.run([sys.executable,*map(str,args)],cwd=TOOLS,env=env,check=True)


def check():
    run("full_quad.py", "--check")
    run("planned_report.py","--persistent","--check")
    run("transport_ablation.py","--check")
    run("promotion_table_v6.py","--check")
    run("scale_frontier.py","--v6","--check")
    run("scale_frontier.py","--v6","--scipy","--check")
    run("sparse_search_report.py","--check")
    run("view_memory.py","--persistent","--check")
    for graph in ("Alpha","OTC"):
        run("sparse_search_verify.py",graph)
    run("long_promotion.py", "--selftest")
    run("signed_promotion.py", "--selftest")
    run("promotion_directional.py")
    run("promotion_root.py")
    run("promotion_index_profile.py","--graph")
    run("-m","unittest","test_directional_kernel","test_fast_policy_kernel")
    for script in ("check_witnesses.py", "test_bounds.py", "test_scaling.py",
                   "test_certified_engine.py", "test_policy_kernel.py", "test_sparse_search.py", "test_directional_math.py",
                   "test_cost_planner.py", "test_direct6.py", "test_linview_registered.py",
                   "test_promotion_strict.py",
                   "main_witness_test.py",
                   "check_results.py"):
        run(script)


def check_runtime():
    packages = ("numpy", "scipy", "sympy", "matplotlib", "psutil", "threadpoolctl", "pymupdf")
    missing = [p for p in packages if importlib.util.find_spec(p) is None]
    if missing:
        raise SystemExit("Missing required Python packages: " + ", ".join(missing))
    for path in (PAPER/"datasets"/f"{name}.npz" for name in GRAPHS):
        if not path.is_file():
            raise FileNotFoundError(path)
    run("-c", "from cholmod_backend import _library; _library().jc_factor_entries")
    run("-c", "from scale_views import symbolic_library; symbolic_library().scale_symbolic")


def clear_cohort(prefix, graph, size):
    stem = f"{prefix}_{graph}_{size}"
    (DATA/f"{stem}.csv").unlink(missing_ok=True)
    for shard in DATA.glob(f"{stem}.*.csv"):
        shard.unlink()


def rerun():
    """Replace the published cohorts using the released graphs and fixed sets."""
    run(PAPER/"figures/make_architecture.py")
    run("factor_fill.py","--root")
    run("planned_report.py","--profile-routes")
    run("workloads.py","--response-profile","--persistent-setup","--persistent-witness")
    for graph in GRAPHS:
        for m in (1000,2500):
            for prefix in ("persistent_batch","persistent_quad","linview_persistent",
                           "linview_registered_persistent"):
                clear_cohort(prefix,graph,m)
            if graph == "Slashdot":
                clear_cohort("persistent_native",graph,m)
                run("planned_matrix.py","--persistent","--fast","--batch-transport",
                    "--native-pcg","--backend","cg","--modes","planner","--graph",graph,"--m",m)
            else:
                for prefix in ("direct6","direct6_slots"):
                    clear_cohort(prefix,graph,m)
            if graph in ("Alpha","Fin"):
                clear_cohort("linview_registered_lookup_persistent",graph,m)
                run("linview_delta.py","--lookup","--catalog","persistent","--graph",graph,"--m",m)
            run("linview_delta.py","--registered","--catalog","persistent","--graph",graph,"--m",m)
            run("planned_matrix.py","--persistent","--fast","--batch-transport","--graph",graph,"--m",m,"--reps",10)
            run("full_quad.py","--catalog","persistent","--graph",graph,"--m",m,"--reps",10)
            run("linview_delta.py","--catalog","persistent","--graph",graph,"--m",m,"--reps",10)
    run("direct6_matrix.py")
    run("direct6_matrix.py","--slots")
    run("planned_report.py","--persistent")
    run("transport_ablation.py")
    run("view_memory.py","--persistent")
    for graph in ("Alpha","OTC"):
        (DATA/f"branch_move_{graph.lower()}.csv").unlink(missing_ok=True)
        run("branch_queries.py","--move","--graph",graph,"--reps",10)
        run("branch_queries.py","--move","--graph",graph,"--audit-revivals")
        run("sparse_search.py",graph)
    run("branch_table.py")
    run("sparse_traps.py")
    run("sparse_search_report.py")
    run("promotion_v4.py","--edition",5,"--reps",10,"--fine-screen")
    run("promotion_v4.py","--edition",5,"--graph","Alpha","--reps",10)
    run("promotion_v4.py","--precision-summary","--graph","Alpha")
    run("promotion_v4.py","--edition",6,"--reps",10)
    run("promotion_table_v6.py","--write")
    from scale_frontier import SIZES, dataset
    from native_scale_frontier import output
    from scale_views import fname
    for n in SIZES:
        for audit in ("audit","union_audit"):
            fname(n,audit,"union").unlink(missing_ok=True)
        dataset(n).unlink(missing_ok=True)
        output(n).unlink(missing_ok=True)
    run("scale_views.py","--union-pattern","--audit")
    for n in (10000,50000):
        (DATA/f"scale_frontier_v6_{n}_deferred.csv").unlink(missing_ok=True)
    run("scale_frontier.py","--v6","--n",10000,50000,"--methods","deferred")
    run("scale_frontier.py","--v6","--scipy")
    run("native_scale_frontier.py")
    run("scale_frontier.py","--v6","--report")
    run("margin_stress.py","--refresh-first")
    run("primary_intervals.py")
    run("flip_stress.py","--cohort","both")
    for name in ("bio","alpha"):
        for weight in (1,4,16,64):
            run("directional_stress.py",name,weight)
    run("directional_stress.py","figure")
    run("-c","from check_results import stress_grid; stress_grid(write=True)")
    for case in ("branch","persistent6","universe6"):
        run("main_witness_emit.py",case)
    for case in ("branch","persistent6","universe6"):
        run("main_witness.py",case,"--repeat",10,"--save")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify the released records offline")
    mode.add_argument("--rerun", action="store_true", help="replace experiment outputs, then check")
    args = parser.parse_args()
    try:
        if args.rerun:
            check_runtime()
            rerun()
        check()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
