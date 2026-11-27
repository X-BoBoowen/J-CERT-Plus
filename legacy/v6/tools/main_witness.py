"""Time independent certificate replay against the released performance rows."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from main_witness_input import CASES, DATA, bind_gain_intervals, expected, instance
from main_witness_verify import verify


def one_replay(case):
    log_case = {"universe5":"clamp","universe6":"clamp","persistent6":"persistent5"}.get(case,case)
    log = DATA / f"main_witness_{log_case}.npz"
    start = perf_counter()
    graph, source, pool, budget = instance(case)
    plan = expected(case, pool)
    prepared = perf_counter()
    result = verify(graph, source, pool, plan, budget, log,
                    scale=2**33 if case in ("motif","clamp","universe5","universe6") else 2**26,
                    revived=plan[1] if case in ("motif","clamp","clustered","universe5","universe6") else None,
                    all_root_signs=case in ("persistent5","persistent6"))
    if case in ("mixed","motif","clamp","universe5","universe6"):
        bind_gain_intervals(case,result)
    if case=="persistent6":
        from main_witness_input import bind_routed_gains
        from main_witness_verify import precise_persistent_gains
        begin = perf_counter()
        bind_routed_gains(precise_persistent_gains(graph,source,pool,plan))
        result["replay_seconds"] += perf_counter()-begin
    sample = [prepared - start, result["index_seconds"], result["replay_seconds"]]
    return dict(case=case, performance_delta=0 if case in ("promotion","mixed","motif","clamp","universe5","universe6") else 1e-9,
                certified_delta=0, pool=len(pool[0]), budget=budget, plan=plan,
                bytes=log.stat().st_size, objective=result["objective"],
                denominator=result["denominator"], root_gain=result["root_gain"],
                epochs=result["epochs"], sample=sample,
                **({"root_gains":result["root_gains"]} if case in ("persistent5","persistent6") else {}))


def run(case, repeats=10):
    if repeats < 1:
        raise ValueError("positive repeat count required")
    samples = []
    reference = None
    for _ in range(repeats):
        command = [sys.executable, str(Path(__file__).resolve()), case, "--child"]
        output = subprocess.check_output(command, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        result = json.loads(output)
        samples.append(result.pop("sample"))
        if reference is not None and result != reference:
            raise ValueError("replay results changed between processes")
        reference = result
    times = np.asarray(samples)
    rng = np.random.default_rng(17)
    draws = rng.integers(len(times), size=(10000, len(times)))
    medians = np.median(times[draws], axis=1)
    reference.update(samples=samples, median_seconds=np.median(times, axis=0).tolist(),
                     phases=["input", "index", "online"], independent_processes=repeats,
                     iqr_seconds=np.quantile(times, [.25, .75], axis=0).T.tolist(),
                     median_ci95_seconds=np.quantile(medians, [.025, .975], axis=0).T.tolist(),
                     bootstrap_resamples=len(draws), bootstrap_seed=17)
    return reference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(one_replay(args.case)))
    else:
        report = run(args.case, args.repeat)
        if args.save:
            (DATA / f"main_witness_{args.case}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "epochs")}))
