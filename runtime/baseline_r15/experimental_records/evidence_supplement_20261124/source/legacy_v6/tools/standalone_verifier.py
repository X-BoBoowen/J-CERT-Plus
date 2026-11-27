"""Replay a flip-only certificate against a caller-supplied graph and query."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import load_npz


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_dot(a, b):
    """Chunk each int64 reduction so its absolute sum fits in int64."""
    limit = int(np.abs(a).max()) * int(np.abs(b).max())
    require(limit < 2**62, "component product too large")
    chunk = max(1, (2**62) // max(limit, 1))
    return sum(int(a[i:i + chunk] @ b[i:i + chunk])
               for i in range(0, len(a), chunk))


def replay(graph_path, query_path, log_path, diagnostics=None):
    graph = load_npz(graph_path).tocsr()
    n = graph.shape[0]
    require(graph.shape == (n, n) and graph.has_canonical_format, "graph layout")
    require((graph != graph.T).nnz == 0 and not graph.diagonal().any()
            and np.all((graph.data == 1) | (graph.data == -1)), "expected signed unit graph")
    graph = graph.astype(np.int64)
    degree = np.diff(graph.indptr).astype(np.int64)
    max_degree = int(degree.max())
    require(0 <= max_degree < n, "invalid degree")
    query = json.loads(Path(query_path).read_text())
    pool = [tuple(x) for x in query["pool"]]
    budget = query["budget"]
    require(query["opinion"] == "even+1,odd-1" and query["alpha"] == 1
            and query["threshold"] == 0 and 0 < budget <= len(pool), "query contract")
    require(len(set(pool)) == len(pool) and all(0 <= i < j < n and graph[i,j] == -1
            for i, j in pool), "illegal flip catalog")
    opinion = 1 - 2 * (np.arange(n, dtype=np.int64) % 2)
    scale = 2**26
    with np.load(log_path, allow_pickle=False) as log:
        require(len(log.files) == len(set(log.files)), "duplicate witness names")
        meta = json.loads(log["meta"].tobytes())
        used = {"meta"}
        def energy(key, flips):
            used.add(key)
            q = log[key]
            require(q.dtype == np.int32 and q.shape == (n,), "invalid witness")
            q = q.astype(np.int64)
            if key != "root":
                q += root_codes
            qmax = int(np.abs(q).max())
            # The base row uses at most (1+2d)qmax; incident flip additions
            # contribute at most another 2d*qmax in any intermediate sum.
            require((4 * max_degree + 2) * qmax + 2 * scale < 2**61, "matvec overflow")
            aq = (1 + degree) * q - graph @ q
            for e in flips:
                i, j = pool[e]
                aq[i] -= 2 * q[j]
                aq[j] -= 2 * q[i]
            r = scale * opinion - aq
            lower = exact_dot(q, scale * opinion + r)
            return lower, lower + exact_dot(r, r)

        chosen = []
        ended = False
        count = 0
        root_codes = log["root"].astype(np.int64)
        initial = current = energy("root", [])
        steps = meta["steps"]
        require(0 < len(steps) <= budget, "invalid number of rounds")
        for t, step in enumerate(steps):
            require(not ended, "round after stop")
            candidates = sorted(set(range(len(pool))) - set(chosen))
            require(step["ids"] == candidates, "incomplete or reordered candidate coverage")
            children = {e: energy(f"r{t}e{e}", chosen + [e]) for e in candidates}
            count += len(children)
            e = step["winner"]
            if e is None:
                require(step["gain"] == 0, "incorrect stop gain")
                require(all(current[1] <= interval[0] for interval in children.values()),
                        "uncertified stop")
                ended = True
            else:
                require(type(e) is int and e in children, "invalid decision")
                lo, hi = children[e]
                require(hi < current[0] and all(hi < value[0] for f, value in children.items()
                                              if f != e), "winner lacks strict separation")
                num, den = float(step["gain"]).as_integer_ratio()
                require((current[0] - hi) * den <= num * scale**2 <=
                        (current[1] - lo) * den, "reported gain outside enclosure")
                if diagnostics is not None:
                    diagnostics.append({"round": t + 1, "gain_lower": (current[0] - hi) / scale**2,
                                        "gap_lower": min(value[0] - hi for f, value in children.items()
                                                         if f != e) / scale**2 if len(children) > 1 else None})
                chosen.append(e)
                current = children[e]
        require(ended or len(chosen) == budget, "premature termination")
        require(chosen == meta["plan"] and set(log.files) == used, "output or witness mismatch")
        num, den = float(meta["objective"]).as_integer_ratio()
        require((initial[0] - current[1]) * den <= num * scale**2 <=
                (initial[1] - current[0]) * den, "final objective outside enclosure")
        if diagnostics is not None:
            diagnostics.append({"objective_lower": (initial[0] - current[1]) / scale**2,
                                "objective_upper": (initial[1] - current[0]) / scale**2})
    return chosen, count


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--main"]:
        from main_witness import one_replay
        from main_witness_input import CASES
        main_parser = argparse.ArgumentParser(description="Replay a full published performance trajectory")
        main_parser.add_argument("--main", required=True, choices=CASES)
        if "--help" in sys.argv[2:] or "-h" in sys.argv[2:]:
            main_parser.print_help()
            sys.exit(0)
        selected = main_parser.parse_args()
        report = one_replay(selected.main)
        print(json.dumps(report))
        sys.exit(0)
    parser = argparse.ArgumentParser(description=__doc__,
        epilog="Use --main {universe6,persistent6,branch} for a released Alpha trajectory.")
    parser.add_argument("--memory", action="store_true",help="sample process resident memory and peak resident memory")
    parser.add_argument("--cold", action="store_true")
    for arg in ("graph", "query", "log"):
        parser.add_argument(arg)
    args = parser.parse_args()
    if args.cold:
        from cold_witness import replay as cold_replay
        probe = None
        if args.memory:
            import psutil
            process = psutil.Process()
            def probe():
                info = process.memory_info()
                if hasattr(info,"peak_wset"):
                    return [info.rss,info.peak_wset]
                import resource
                import sys
                peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                return [info.rss, peak if sys.platform=="darwin" else peak*1024]
        result=cold_replay(args.graph,int(args.query),args.log,memory_probe=probe)
        result.pop("claims")
        print(json.dumps(result))
        raise SystemExit
    start = time.perf_counter()
    details = []
    print(replay(args.graph, args.query, args.log, details), f"{time.perf_counter()-start:.4f}s")
    print(json.dumps(details, indent=2))
