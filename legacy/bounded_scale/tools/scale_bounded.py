"""Bounded-input four-arm rerun of the original v6 frontier experiment.

Use --inputs to materialize full opinions and ordered edits, then --run.
Every timed arm recalibrates the original RHS, normalizes once, and verifies
exact equivalence to that input outside the original setup/query timer.
Calibration and normalization are separately charged in inclusive_seconds;
fixture construction and input verification remain outside that total.
"""
import argparse
import csv
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter, sleep

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

from scale_frontier import (ARMS, ROOT, SIZES, calibrated_rhs, fixture,
                            headroom_mib, np, records, trial)
from native_scale_frontier import NativeCostPlanner, build_paired_pcg, load_paired_pcg


def output(n):
    return ROOT / "data" / f"scale_bounded_{n}.csv"


def input_path(n):
    return ROOT / "data" / f"scale_bounded_input_{n}.npz"


def edit_arrays(edits):
    return {"edit_i": np.array([e.i for e in edits], dtype=np.int64),
            "edit_j": np.array([e.j for e in edits], dtype=np.int64),
            "edit_w": np.array([e.w for e in edits], dtype=np.float64),
            "edit_flip": np.array([e.flip for e in edits], dtype=bool)}


def normalize(s):
    divisor = max(1., float(np.max(np.abs(s))))
    return s / divisor, divisor


def stats(s, prefix):
    return {f"{prefix}_min": float(np.min(s)),
            f"{prefix}_max": float(np.max(s)),
            f"{prefix}_maxabs": float(np.max(np.abs(s))),
            f"{prefix}_count": int(s.size)}


def prepare(n):
    A, edits, _, _ = fixture(n)
    original = calibrated_rhs(A, edits)
    bounded, divisor = normalize(original)
    payload = dict(n=n, s_original=original, s=bounded,
                   scaling_factor=1./divisor, scaling_divisor=divisor,
                   **edit_arrays(edits))
    path = input_path(n)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(payload) or any(
                    not np.array_equal(saved[k], v) for k, v in payload.items()):
                raise ValueError(f"Existing input differs: {path}")
    else:
        np.savez(path, **payload)
    print(json.dumps(dict(n=n, scaling_factor=1./divisor,
                          **stats(original, "before"), **stats(bounded, "after"))), flush=True)


def bounded_trial(n, method, library):
    details = {}

    def transform(A, edits, original):
        start = perf_counter()
        bounded, divisor = normalize(original)
        details["normalization_seconds"] = perf_counter()-start
        start = perf_counter()
        with np.load(input_path(n), allow_pickle=False) as saved:
            expected = dict(n=n, s_original=original, s=bounded,
                            scaling_factor=1./divisor, scaling_divisor=divisor,
                            **edit_arrays(edits))
            if any(not np.array_equal(saved[k], v) for k, v in expected.items()):
                raise ValueError(f"Input mismatch at {n} {method}")
        details["validation_seconds"] = perf_counter()-start
        details.update(scaling_factor=1./divisor, scaling_divisor=divisor,
                       delta=1e-9, **stats(original, "before"), **stats(bounded, "after"))
        return bounded

    factory = partial(NativeCostPlanner, native_pcg=load_paired_pcg(library))
    row = trial(n, method, cg_factory=factory, rhs_transform=transform,
                allow_early_stop=True)
    row.update(details)
    row["inclusive_seconds"] = (row["seconds"] + row["calibration_seconds"]
                                + row["normalization_seconds"])
    row["commits"] = len(json.loads(row["sequence"]))
    row["stop_reason"] = "budget" if row["commits"] == 15 else "certified_no_candidate"
    return row


def measured(n, method, library):
    # Windows can reclaim standby pages; available commit is the launch/retry
    # guard. Peak RSS is still measured by the original worker instrumentation.
    def available():
        physical, commit = headroom_mib()
        return commit if os.name == "nt" else physical

    needed = 2944 if n == 500000 and method in ("cholmod", "registered") else 1350
    while True:
        while available() < needed:
            print("Waiting for benchmark headroom", n, method, flush=True)
            sleep(10)
        cmd = [sys.executable, "-B", __file__, "--child", method, "--n", str(n),
               "--library", str(library)]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as proc:
            start = perf_counter()
            while True:
                try:
                    out, err = proc.communicate(timeout=1)
                    if proc.returncode:
                        raise RuntimeError(f"{n} {method}: {err or out}")
                    return json.loads(out.strip().splitlines()[-1])
                except subprocess.TimeoutExpired:
                    if available() < 640:
                        proc.kill()
                        proc.communicate()
                        print("Memory reserve reached; retrying", n, method, flush=True)
                        break
                    if perf_counter()-start > 1800:
                        proc.kill()
                        proc.communicate()
                        raise TimeoutError(f"{n} {method}: 30 minute limit")


def validate_rows(n, rows, complete=False):
    keys = [(int(r["rep"]), r["method"]) for r in rows]
    expected = {(rep, method) for rep in range(10) for method in ARMS}
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError(f"Duplicate or unexpected records at {n}")
    if complete and set(keys) != expected:
        raise ValueError(f"Incomplete four-arm measurements at {n}")
    for r in rows:
        order = np.random.default_rng(871+int(r["rep"])).permutation(ARMS)
        seq = json.loads(r["sequence"])
        if (int(r["n"]) != n or order[int(r["position"])] != r["method"]
                or r["batch"] != "bounded_native" or len(seq) > 15
                or len(seq) != len(set(seq)) or any(e < 0 or e >= 32 for e in seq)
                or int(r["commits"]) != len(seq) or float(r["delta"]) != 1e-9
                or float(r["after_maxabs"]) > 1 or int(r["after_count"]) != n):
            raise ValueError(f"Invalid measurement at {n}: {r['method']}")
        for k in ("seconds", "setup_seconds", "calibration_seconds", "normalization_seconds",
                  "validation_seconds", "inclusive_seconds", "peak_mib"):
            if not np.isfinite(float(r[k])) or float(r[k]) <= 0:
                raise ValueError(f"Invalid {k} at {n}")
        if float(r["setup_seconds"]) >= float(r["seconds"]):
            raise ValueError("Setup exceeds query time")
        if not isinstance(json.loads(r["counters"]), dict):
            raise ValueError("Missing work counters")
        if r["method"] != "registered" and len(json.loads(r["frontiers"])) != len(seq):
            raise ValueError("Incomplete frontier trace")
        if r["stop_reason"] != ("budget" if len(seq) == 15 else "certified_no_candidate"):
            raise ValueError("Incorrect stopping reason")
        divisor = max(1., float(r["before_maxabs"]))
        if (int(r["before_count"]) != n or float(r["scaling_divisor"]) != divisor
                or float(r["scaling_factor"]) != 1./divisor
                or not np.isclose(float(r["inclusive_seconds"]),
                                  sum(float(r[k]) for k in ("seconds", "calibration_seconds",
                                                            "normalization_seconds")),
                                  rtol=1e-14, atol=0)):
            raise ValueError("Invalid input scaling or inclusive timing")
    if complete and len({tuple(json.loads(r["sequence"])) for r in rows}) != 1:
        raise ValueError(f"Bounded paths disagree at {n}; all measurements retained")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", nargs="+", type=int, choices=SIZES, default=SIZES)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inputs", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--child", choices=ARMS)
    p.add_argument("--library", type=Path)
    args = p.parse_args()
    if args.inputs:
        for n in args.n:
            prepare(n)
        return
    if args.child:
        if args.library is None:
            p.error("--child requires --library")
        print(json.dumps(bounded_trial(args.n[0], args.child, args.library)))
        return
    if args.check:
        for n in args.n:
            validate_rows(n, records(output(n)), complete=True)
        print("Verified complete bounded four-arm measurements and equal paths.")
        return
    for n in args.n:
        if not input_path(n).exists():
            p.error(f"Missing {input_path(n)}; run --inputs first")
        validate_rows(n, records(output(n)))
    with tempfile.TemporaryDirectory() as directory:
        library = build_paired_pcg(directory)
        for n in args.n:
            rows = records(output(n))
            done = {(int(r["rep"]), r["method"]) for r in rows}
            for rep in range(10):
                for pos, method in enumerate(np.random.default_rng(871+rep).permutation(ARMS)):
                    if (rep, method) in done:
                        continue
                    row = dict(rep=rep, position=pos, **measured(n, method, library), batch="bounded_native")
                    validate_rows(n, [row])
                    with output(n).open("a", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=row)
                        if stream.tell() == 0:
                            writer.writeheader()
                        writer.writerow(row)
                    rows.append(row)
                    print(n, rep, method, round(row["seconds"], 3), row["sequence"], flush=True)
            # Keep measuring remaining sizes even if a completed size disagrees.
    for n in args.n:
        validate_rows(n, records(output(n)), complete=True)


if __name__ == "__main__":
    main()
