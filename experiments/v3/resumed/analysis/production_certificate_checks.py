"""Bounded actual-production certificate checks; run only by the root scheduler.

The physical reference is v3/code/exact_oracle.py.  Its gains are differences
of independently assembled parent/child objectives, not a local gain formula.
No admission method is patched.  Weight 64 is rejected by A2 and is tested at
the actual generic weighted_adapter.correction boundary with certified inputs.
This file does not execute the historical suite, data streams, or scale runs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import inspect
import json
from math import ceil, floor
from pathlib import Path
import sys
import time


def encode(value):
    if isinstance(value, F):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


class Audit:
    def __init__(self, output):
        self.output = output
        self.started = time.perf_counter()
        self.report = {
            "status": "running", "scope": "bounded mathematical fixtures only",
            "independent_gain_basis": "direct rational parent-minus-child physical objectives",
            "historical_suite_reexecuted": False, "real_data_decisions_verified": 0,
            "checks": [], "fixtures": [], "services": [], "rejections": [],
            "completion_issues": [], "unknown": [],
        }

    def check(self, name, passed, **details):
        item = {"name": name, "passed": bool(passed), **details}
        self.report["checks"].append(item)
        if not passed:
            raise AssertionError(name)

    def rejected(self, name, callback, *, expected=(ValueError, ArithmeticError)):
        try:
            callback()
        except expected as exc:
            self.report["rejections"].append({
                "name": name, "rejected": True, "exception": type(exc).__name__,
                "reason": str(exc),
            })
            self.check(name, True)
        else:
            self.check(name, False, reason="invalid evidence was accepted")

    def save(self):
        self.report["check_count"] = len(self.report["checks"])
        self.report["check_failures"] = sum(not c["passed"] for c in self.report["checks"])
        self.report["harness_wall_seconds"] = time.perf_counter() - self.started
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(self.report, default=encode, indent=2,
                                         allow_nan=False) + "\n")


class ResolveTrace:
    """Observe executed resolver branches without replacing production code."""
    def __init__(self, kernel_class):
        self.code = kernel_class.resolve.__code__
        lines, first = inspect.getsourcelines(kernel_class.resolve)
        self.fallback_line = next(first + i for i, line in enumerate(lines)
                                  if "import sympy as sy" in line)
        self.calls = []
        self.active = {}

    def trace(self, frame, event, arg):
        if frame.f_code is not self.code:
            return None
        key = id(frame)
        if event == "call":
            self.active[key] = {"fallback_executed": False}
        current = self.active.get(key)
        if current is not None and event == "line" and frame.f_lineno >= self.fallback_line:
            current["fallback_executed"] = True
        if current is not None and event == "return":
            values = frame.f_locals
            evidence = values.get("evidence") or {}
            current.update(
                overlap_ids=list(map(int, values.get("overlap", ()))),
                returned=list(arg) if isinstance(arg, tuple) else None,
                structural_witnesses=len(evidence.get("witnesses", ())),
                structural_success="canonical_winner_id" in evidence,
            )
            current["path"] = (
                "small_graph_exact_fallback" if current["fallback_executed"] else
                "structural_node_transposition" if current["structural_success"] else
                "no_unresolved_overlap"
            )
            self.calls.append(current)
            del self.active[key]
        return self.trace

    def __enter__(self):
        self.previous = sys.gettrace()
        sys.settrace(self.trace)
        return self

    def __exit__(self, *args):
        sys.settrace(self.previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.runtime = args.runtime.resolve()
    args.oracle = args.oracle.resolve()
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(args.runtime), str(args.oracle.parent)]
    audit = Audit(args.output.resolve())
    try:
        import numpy as np
        from scipy import sparse
        from threadpoolctl import threadpool_limits
        from exact_oracle import (canonical_reference_decision, edited_edges,
                                  evaluate_catalog, physical_operator, solve_rational)
        from src.common import DEN, SCALE, UNIT, Ledger
        from src.a2_catalog import AllCatalog
        from src.a2_numeric import EndpointKernel
        from src.a2_session import A2Session
        from src.a2_checker import A2CatalogReplay, worker_request
        from src.weighted_adapter import Kernel, correction
        from src.exact_ties import TieContext, certify_exchange
        import long_promotion as lp

        files = [Path(__file__).resolve(), args.oracle,
                 *[args.runtime / "src" / name for name in
                   ("common.py", "a2_catalog.py", "a2_numeric.py", "a2_session.py",
                    "a2_checker.py", "checker.py", "weighted_adapter.py", "exact_ties.py")],
                 Path(lp.__file__),
                 Path(sys.modules["promotion"].__file__),
                 Path(sys.modules["promotion_directional"].__file__)]
        audit.report["source_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        audit.report["units"] = {
            "operator_scale": SCALE, "opinion_denominator": DEN, "proof_unit": UNIT,
            "raw_score_to_physical": "16 / 2^96",
            "published_decision_to_physical": "1 / gain_denominator = 1 / 2^96",
        }
        fixtures = [
            ("path4_equal", 4, ((0, 1, -1), (1, 2, 1), (2, 3, -1)),
             (1, -1, -1, 1)),
            ("path4_near", 4, ((0, 1, -1), (1, 2, 1), (2, 3, -1)),
             (1, -1 + F(1, DEN), -1, 1)),
            ("path4_zero", 4, ((0, 1, -1), (1, 2, 1), (2, 3, -1)),
             (0, 0, 0, 0)),
            ("empty2_insert", 2, (), (1, -1)),
        ]
        saved = {}

        def graph(n, edges):
            array = np.zeros((n, n), dtype=np.float64)
            for i, j, weight in edges:
                array[i, j] = array[j, i] = float(weight)
            return sparse.csr_matrix(array)

        def coefficients(matrix):
            return tuple(tuple(F(int(v)) for v in row) for row in matrix.toarray())

        def basis(n, edit):
            cols = [[F(0)] * n for _ in range(2 if edit.kind == "flip" else 1)]
            cols[0][edit.i], cols[0][edit.j] = F(1), F(-1)
            if edit.kind == "flip":
                cols[1][edit.i] = cols[1][edit.j] = F(1)
            return cols

        def project_exact(cols, response_cols):
            return [[sum((a * b for a, b in zip(left, right)), F(0))
                     for right in response_cols] for left in cols]

        def box_contains(boxes, exact):
            return all(lo <= value <= hi for row, erow in zip(boxes, exact)
                       for (lo, hi), value in zip(row, erow))

        def inspect_numeric(name, n, edges, opinions):
            W = graph(n, edges)
            s = np.asarray([int(F(v) * DEN) for v in opinions], dtype=np.int64)
            rows = evaluate_catalog(n, edges, opinions, F(1, 16))
            expected = {row.edit.canonical_id: row for row in rows}
            catalog = AllCatalog(W)
            kernel = EndpointKernel(catalog)
            A = kernel.operator(W)
            A.sort_indices()
            physical = physical_operator(n, edges)
            audit.check(name + ": physical operator scaling",
                        coefficients(A) == tuple(tuple(SCALE * v for v in row) for row in physical))
            actual_ids = [int(e) for block in catalog.iter_blocks(block_size=2) for e in block.ids]
            audit.check(name + ": complete legal catalog", actual_ids == sorted(expected))
            fac = lp.UpdatedFactor(A)
            certified_rows = []
            state = lp.certified_state(A, fac, s, UNIT, certified_rows=certified_rows)
            z, x, r, xc, rc = state
            exact_state = tuple(v / SCALE for v in solve_rational(physical, opinions))
            audit.check(name + ": state residual enclosure",
                        all(abs(F(int(a), UNIT) - b) <= F(int(r), UNIT)
                            for a, b in zip(x, exact_state)))
            strict = kernel.strict(A, s, x, certified_rows[0])
            audit.check(name + ": endpoint state enclosure",
                        all(abs(F(int(a), UNIT) - b) <= F(int(rad), UNIT)
                            for a, b, rad in zip(x, exact_state, strict.radius)))
            audit.check(name + ": coarse state enclosure",
                        all(abs(F(int(a), lp.SCREEN) - b) <= F(int(rc), lp.SCREEN)
                            for a, b in zip(xc, exact_state)))
            keep, caps = strict.screen(np.asarray(actual_ids, dtype=np.int64), 0)
            cap_by_id = dict(zip(map(int, keep), map(int, caps)))
            views = kernel.views(Ledger(False), defaultdict(int))
            fixture = {"name": name, "catalog_size": len(rows), "operator_integer": A.toarray(),
                       "state_radius_raw": int(r), "state_radius_physical": F(SCALE * int(r), UNIT),
                       "state_exact_integer_units": exact_state,
                       "reference_decision": canonical_reference_decision(rows), "candidates": []}
            audit.report["fixtures"].append(fixture)
            for e in actual_ids:
                row = expected[e]
                U, _, X, rr = views.response(e, A, fac)
                cols = basis(n, row.edit)
                exact_Y = [tuple(v / SCALE for v in solve_rational(physical, col)) for col in cols]
                audit.check(name + f": response enclosure {e}",
                            all(abs(F(int(X[i, k]), UNIT) - exact_Y[k][i]) <= F(int(rr), UNIT)
                                for i in range(n) for k in range(len(cols))))
                g = views.project(e, X, rr)
                h = views.project(e, x[:, None], r)
                audit.check(name + f": Gram enclosure {e}", box_contains(g, project_exact(cols, exact_Y)))
                audit.check(name + f": projection enclosure {e}",
                            box_contains(h, project_exact(cols, [exact_state])))
                views.retained = (e, (U, _, X, rr))
                lo, hi = views.score(e, 0, A, fac, x, r)
                views.retained = None
                low, high = F(SCALE * lo, UNIT ** 2), F(SCALE * hi, UNIT ** 2)
                audit.check(name + f": physical gain enclosure {e}", low <= row.gain <= high,
                            exact=row.gain, interval=[low, high])
                cap = cap_by_id.get(e)
                audit.check(name + f": strict cap or safe exclusion {e}",
                            row.gain <= (F(SCALE * cap, UNIT ** 2) if cap is not None else 0))
                fixture["candidates"].append({
                    "id": e, "kind": row.edit.kind, "physical_weight": row.edit.weight,
                    "integer_weight": int(kernel.d[e]), "exact_gain": row.gain,
                    "physical_interval": [low, high], "raw_interval": [lo, hi],
                    "response_radius_integer_units": int(rr), "Gram_interval": g,
                    "projection_interval": h, "screen_upper_raw": cap,
                })
            saved[name] = dict(W=W, s=s, rows=rows, kernel=kernel, A=A, fac=fac,
                               state=state, views=views, certified_rows=certified_rows,
                               n=n, edges=edges, opinions=opinions)

        def request_check(name, n, edges, opinions, methods=("D", "S", "H")):
            W = graph(n, edges)
            s = np.asarray([int(F(v) * DEN) for v in opinions], dtype=np.int64)
            exact = evaluate_catalog(n, edges, opinions, F(1, 16))
            reference = canonical_reference_decision(exact)
            replay = A2CatalogReplay(W, block_size=2)
            for method in methods:
                session = A2Session(W, method, timers=False, hot_size=0)
                with ResolveTrace(Kernel) as traced:
                    request = session.request(s, budget=1)
                proof = worker_request(replay, request, s, session.graph_hash)
                result = {"fixture": name, "method": method, "request": request,
                          "resolver_execution": traced.calls, "checker": proof}
                audit.report["services"].append(result)
                if request["status"] == "unresolved":
                    audit.check(name + f": unresolved is not certified stop ({method})",
                                request["commits"] == 0 and not proof["termination_certified"])
                    audit.report["completion_issues"].append({"fixture": name, "method": method,
                                                              "reason": request["failure_reason"]})
                    continue
                audit.check(name + f": full toy decision ({method})",
                            proof["full_catalog_decisions_certified"] and
                            proof["checked_candidates"] == len(exact) and
                            request["commits"] == int(reference["accepted"]))
                if reference["accepted"]:
                    row = request["decisions"][0]
                    audit.check(name + f": positive canonical winner ({method})",
                                row["winner_id"] == reference["canonical_id"] and row["low"] > 0)
                    audit.check(name + f": published physical interval ({method})",
                                F(row["low"], request["gain_denominator"]) <= reference["gain"] <=
                                F(row["high"], request["gain_denominator"]))
                else:
                    audit.check(name + f": certified zero stop ({method})",
                                request["status"] == "tolerance_stop" and
                                request["decisions"][0]["winner_id"] is None)
                if method == "H":
                    saved.setdefault(name, {}).update(W=W, s=s, request=request,
                                                       graph_hash=session.graph_hash, replay=replay)

        with threadpool_limits(limits=1):
            for fixture in fixtures:
                inspect_numeric(*fixture)
                request_check(*fixture)

            # Genuine local weighted entrypoint, no A2 graph-admission bypass.
            name, n, edges, opinions = "isolated_flip_weight64", 2, ((0, 1, -64),), (1, -1)
            W = graph(n, edges)
            audit.rejected("A2 rejects weight64", lambda: AllCatalog(W))
            physical = physical_operator(n, edges)
            row, = evaluate_catalog(n, edges, opinions, F(1, 16))
            cols = basis(n, row.edit)
            z = tuple(v / SCALE for v in solve_rational(physical, opinions))
            Y = [tuple(v / SCALE for v in solve_rational(physical, col)) for col in cols]
            G, h = project_exact(cols, Y), project_exact(cols, [z])
            # Outward boxes on the actual dyadic proof grid; input truth comes
            # from direct parent-system solutions, never a gain identity.
            def dyadic(value):
                return F(floor(value * UNIT), UNIT), F(ceil(value * UNIT), UNIT)
            gbox = [[dyadic(v) for v in erow] for erow in G]
            hbox = [[dyadic(erow[0]) for erow in h]]
            raw_lo, raw_hi = correction(gbox, hbox, 64 * SCALE)[0][0]
            lo, hi = F(SCALE * floor(raw_lo * UNIT ** 2), UNIT ** 2), F(SCALE * ceil(raw_hi * UNIT ** 2), UNIT ** 2)
            audit.check("weight64 generic physical gain enclosure", lo <= row.gain <= hi,
                        exact=row.gain, interval=[lo, hi])
            audit.report["fixtures"].append({
                "name": name, "catalog_size": 1, "entrypoint": "src.weighted_adapter.correction",
                "A2_service_admitted": False, "response_input": "outward dyadic boxes of exact direct parent solves",
                "candidates": [{"id": row.edit.canonical_id, "exact_gain": row.gain,
                                "physical_interval": [lo, hi], "Gram_interval": gbox, "projection_interval": hbox}],
                "pivot_units": {"physical_B_inverse_norm": 129,
                                "physical_Bbar_inverse_norm": 8256,
                                "integer_normalized_inverse_norm": 132096,
                                "adapter_transformed_inverse_norm": 1},
            })

            # One current-version transport, including fresh direct child truth.
            case = saved["path4_equal"]
            kernel, views, A, fac = (case[k] for k in ("kernel", "views", "A", "fac"))
            state = case["state"]
            winner = case["rows"][0].edit
            e = winner.canonical_id
            trial = views.advance(e, 0, A, fac, state[0])
            child_A = kernel.child(A, e)
            child_rows = []
            child_state, child_source = lp.certify_child(child_A, fac, case["s"], trial, UNIT, child_rows)
            child_edges = edited_edges(case["n"], case["edges"], winner)
            child_truth = evaluate_catalog(case["n"], child_edges, case["opinions"], F(1, 16))
            exact_child_A = physical_operator(case["n"], child_edges)
            exact_child_z = tuple(v / SCALE for v in solve_rational(exact_child_A, case["opinions"]))
            audit.check("transport: child state enclosure", all(
                abs(F(int(a), UNIT) - b) <= F(int(child_state[2]), UNIT)
                for a, b in zip(child_state[1], exact_child_z)))
            transport_records = []
            for row in child_truth:
                e = row.edit.canonical_id
                cols = basis(case["n"], row.edit)
                exact_Y = [tuple(v / SCALE for v in solve_rational(exact_child_A, col)) for col in cols]
                version, g = views.entries[e]
                audit.check(f"transport: current Gram {e}",
                            version == 1 and box_contains(g, project_exact(cols, exact_Y)))
                low, high = views.score(e, 1, child_A, fac, child_state[1], child_state[2])
                physical_box = [F(SCALE * low, UNIT ** 2), F(SCALE * high, UNIT ** 2)]
                audit.check(f"transport: physical gain {e}", physical_box[0] <= row.gain <= physical_box[1])
                transport_records.append({"id": e, "exact_gain": row.gain, "physical_interval": physical_box})
            audit.report["transport"] = {"committed_id": winner.canonical_id, "child_source": child_source,
                                         "remaining": transport_records, "transfers": views.transfers}
            audit.rejected("stale Gram version", lambda: views.score(
                child_truth[0].edit.canonical_id, 0, child_A, fac, child_state[1], child_state[2]))
            audit.rejected("stale strict operator object", lambda: kernel.strict(
                A.copy(), case["s"], state[1], case["certified_rows"][0]))

            # A separate genuine transposition fixture exercises the witness
            # branch; the plan's disjoint end flips do not have that witness.
            twins = ("twins3_structural", 3, ((1, 2, 3),), (1, -1, -1))
            request_check(*twins)
            twin = saved[twins[0]]
            witness_request = twin["request"]
            resolution = witness_request["decisions"][0].get("comparison_resolution")
            audit.check("actual structural witness emitted", bool(resolution and resolution.get("witnesses")))
            # Force only the independent checker's documented numerical path,
            # preserving all graph and service admission requirements.
            numerical_replay = A2CatalogReplay(twin["W"], block_size=2, exact_limit=0)
            independent = worker_request(numerical_replay, witness_request, twin["s"], twin["graph_hash"])
            audit.check("witness verified on numerical checker path",
                        independent["structural_tie_witnesses_verified"] > 0 and
                        independent["full_catalog_decisions_certified"])
            audit.report["structural_numerical_checker"] = independent

            def verify_case(case, request):
                return worker_request(case["replay"], request, case["s"], case["graph_hash"])

            base = saved["path4_equal"]
            for field, value, label in (
                ("parent_version_id", "stale-parent", "stale parent version"),
                ("d", 17, "wrong edit weight"),
            ):
                attacked = deepcopy(base["request"])
                attacked["decisions"][0][field] = value
                audit.rejected(label, lambda bad=attacked: verify_case(base, bad))
            attacked = deepcopy(base["request"])
            attacked["gain_denominator"] //= SCALE
            audit.rejected("wrong published gain scaling", lambda: verify_case(base, attacked))

            def fabricated_winner(case, e, gain):
                """Adversarial claim, not a synthetic production success."""
                request = deepcopy(case["request"])
                row = request["decisions"][0]
                parent = row["parent_version_id"]
                version = hashlib.sha256(f"{parent}/{e}".encode()).hexdigest()
                i, j, d, flip = case["replay"].candidate(e)
                row.pop("comparison_resolution", None)
                row.update(status="accepted", winner_id=e, i=i, j=j, d=d, flip=int(flip),
                           current_version_id=version, low=floor(gain * UNIT ** 2),
                           high=ceil(gain * UNIT ** 2), scored=1, live_catalog=1,
                           competitor_max_upper=0, cold_upper=0, scored_rival_max_high=0)
                request.update(status="budget_stop", commits=1, current_version_id=version,
                               gain_low=row["low"], gain_high=row["high"])
                return request

            by_id = {row.edit.canonical_id: row.gain for row in base["rows"]}
            omitted_stronger = fabricated_winner(base, 2, by_id[2])
            audit.rejected("missing stronger competitor cannot certify full decision",
                           lambda: verify_case(base, omitted_stronger))
            omitted_canonical = fabricated_winner(base, 11, by_id[11])
            audit.rejected("missing earlier equal competitor cannot certify canonical choice",
                           lambda: verify_case(base, omitted_canonical))
            zero = saved["path4_zero"]
            zero_accept = fabricated_winner(zero, 1, F(0))
            audit.rejected("zero gain cannot be accepted", lambda: verify_case(zero, zero_accept))

            witness_mutations = (
                ("parent_version_id", "stale", "stale witness version"),
                ("operator_scale", 1, "wrong witness operator scale"),
                ("score_unit", UNIT // 2, "wrong witness score scale"),
                ("source_edit", {"i": 0, "j": 1, "d": 2, "flip": False}, "wrong witness weight"),
                ("swap", [0, 2], "invalid witness conjugacy"),
            )
            for field, value, label in witness_mutations:
                attacked = deepcopy(witness_request)
                attacked["decisions"][0]["comparison_resolution"]["witnesses"][0][field] = value
                audit.rejected(label, lambda bad=attacked: verify_case(twin, bad))
            attacked = deepcopy(witness_request)
            attacked["decisions"][0]["comparison_resolution"]["witnesses"] = []
            audit.rejected("missing declared equality witness", lambda: verify_case(twin, attacked))

            # A genuinely near equality cannot inherit an exact witness even
            # after refreshing the opinion digest in the adversarial record.
            near_s = twin["s"].copy()
            near_s[2] += 1
            attacked = deepcopy(witness_request)
            opinion_digest = hashlib.sha256(near_s.tobytes()).hexdigest()
            attacked["opinion_hash"] = opinion_digest
            attacked["decisions"][0]["comparison_resolution"]["witnesses"][0]["opinion_sha256"] = opinion_digest
            audit.rejected("near tie rejects transplanted exact witness", lambda: worker_request(
                twin["replay"], attacked, near_s, twin["graph_hash"]))

            # The standalone producer API also binds the exact integer opinion.
            cat = AllCatalog(twin["W"])
            key = EndpointKernel(cat)
            op = key.operator(twin["W"]); op.sort_indices()
            ctx = TieContext(op, twin["s"], "toy-current-parent", frozenset())
            proof = certify_exchange(key, op, twin["s"] / DEN, 1, 2, ctx)
            audit.check("producer witness on exact twins", proof is not None)
            audit.check("producer rejects near-tie opinion mismatch", certify_exchange(
                key, op, near_s / DEN, 1, 2, ctx) is None)
            audit.check("producer rejects stale operator identity", certify_exchange(
                key, op.copy(), twin["s"] / DEN, 1, 2, ctx) is None)

            import bounds
            constants = bounds.prior_constants(1, 40, 1e-9)
            audit.report["finite_precision_theorem"] = {
                "w": 1, "b": 40, "delta": 1e-9, "epsilon_sufficient": constants["epsilon"],
                "binary64_unit_roundoff": 2.0 ** -53,
                "runtime_delta": 0,
                "applicable_to_runtime_completion": False,
                "reason": "The theorem requires delta>0 and certified attainable complete allowances; the service uses delta=0 and posterior dyadic certificates.",
            }
            audit.check("finite theorem sufficient allowance below binary64 unit roundoff",
                        0 < constants["epsilon"] < 2.0 ** -53)
            audit.report["source_sha256"][str(Path(bounds.__file__))] = hashlib.sha256(
                Path(bounds.__file__).read_bytes()).hexdigest()
            audit.report["unknown"].extend([
                "Weight64 is not an admitted A2 service; no full A2 decision or response-acquisition claim is made for it.",
                "The >12-vertex checker validates intersection of independent/published nonpoint intervals, not containment of every published endpoint.",
                "The producer's standalone tie API cannot authenticate a nonempty parent-version label against external history; the replay checker binds the actual version chain.",
                "No general equality resolver, arbitrary weight interface, or historical full-suite result is inferred from these fixtures.",
            ])
            audit.report["resolver_path_counts"] = dict(Counter(
                call["path"] for service in audit.report["services"] for call in service["resolver_execution"]))
            audit.report["core_fixture_gain_values_checked"] = sum(
                len(fixture["candidates"]) for fixture in audit.report["fixtures"])
            audit.report["status"] = "passed"
            audit.report["correctness_gate"] = "no contradiction found on the explicitly checked paths"
    except Exception as exc:
        audit.report.update(status="failed", exception=type(exc).__name__, failure=str(exc),
                            correctness_gate="closed_pending_failure_classification")
        raise
    finally:
        audit.save()
        print(json.dumps({"status": audit.report["status"], "checks": audit.report["check_count"],
                          "failures": audit.report["check_failures"], "output": str(audit.output)}), flush=True)


if __name__ == "__main__":
    main()
