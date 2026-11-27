"""Emit exact A2 reference evidence; invoke only through the budget scheduler."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from fractions import Fraction as F
from pathlib import Path

from exact_oracle import (
    canonical_reference_decision,
    edited_edges,
    evaluate_catalog,
    physical_operator,
)


V3_ROOT = Path(__file__).resolve().parents[1]


def fraction_json(value):
    if isinstance(value, F):
        return f"{value.numerator}/{value.denominator}"
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _input_digest(value):
    encoded = json.dumps(value, default=fraction_json, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _check(name, observed, expected):
    return {"name": name, "observed": observed, "expected": expected,
            "passed": observed == expected}


def build_report(protocol_path=V3_ROOT / "PROTOCOL.json"):
    started = time.perf_counter()
    path = ((0, 1, F(-1)), (1, 2, F(1)), (2, 3, F(-1)))
    inputs = [
        {"name": "path4_equal", "n": 4, "edges": path,
         "opinions": (F(1), F(-1), F(-1), F(1)), "insertion_weight": F(1, 16)},
        {"name": "path4_perturbed", "n": 4, "edges": path,
         "opinions": (F(1), F(-1) + F(1, 65536), F(-1), F(1)),
         "insertion_weight": F(1, 16)},
        {"name": "path4_zero", "n": 4, "edges": path,
         "opinions": (F(0),) * 4, "insertion_weight": F(1, 16)},
        {"name": "negative2_flip64", "n": 2, "edges": ((0, 1, F(-64)),),
         "opinions": (F(1), F(-1)), "insertion_weight": F(1, 16)},
        {"name": "empty2_insert_1_16", "n": 2, "edges": (),
         "opinions": (F(1), F(-1)), "insertion_weight": F(1, 16)},
    ]
    code_digest = {p.name: _file_digest(p) for p in
                   (Path(__file__), Path(__file__).with_name("exact_oracle.py"),
                    Path(__file__).with_name("test_exact_oracle.py"))}
    protocol_digest = _file_digest(Path(protocol_path)) if protocol_path else None
    source = {"file": "v3/spec/V3_USER_PLAN.md", "section": "A2",
              "sha256": _file_digest(V3_ROOT / "spec" / "V3_USER_PLAN.md")}
    fixtures = []
    total_rhs = 0
    for data in inputs:
        rows = evaluate_catalog(data["n"], data["edges"], data["opinions"],
                                data["insertion_weight"])
        values = {row.edit.canonical_id: row.gain for row in rows}
        decision = canonical_reference_decision(rows)
        checks = []
        relation = None
        extra = {}
        if data["n"] == 4:
            checks.append(_check("complete_legal_catalog_ids", list(values),
                                 [1, 2, 3, 7, 11]))
        if data["name"] == "path4_equal":
            checks.extend([
                _check("all_physical_gains", values,
                       {1: F(32, 21), 2: F(28, 121), 3: F(0),
                        7: F(28, 121), 11: F(32, 21)}),
                _check("canonical_tie_winner", decision["canonical_id"], 1),
            ])
            relation = "exact_equal"
        elif data["name"] == "path4_perturbed":
            extra["right_minus_left_flip_gain"] = values[11] - values[1]
            checks.extend([
                _check("strict_flip_gap", extra["right_minus_left_flip_gain"],
                       F(1, 114688)),
                _check("full_catalog_winner", decision["canonical_id"], 11),
                _check("strict_winner_over_all_other_edits",
                       all(values[11] > values[i] for i in (1, 2, 3, 7)), True),
            ])
            relation = "separated"
        elif data["name"] == "path4_zero":
            checks.extend([
                _check("all_gains_zero", list(values.values()), [F(0)] * 5),
                _check("zero_gain_stop", decision["status"], "certified_stop"),
                _check("zero_gain_never_accepted", decision["accepted"], False),
            ])
            relation = "exact_equal"
        elif data["name"] == "negative2_flip64":
            checks.append(_check("physical_weight_64_gain", values, {1: F(256, 129)}))
        else:
            checks.append(_check("physical_weight_1_16_gain", values, {1: F(2, 9)}))
        passed = all(check["passed"] for check in checks)
        candidates = []
        for row in rows:
            candidate = asdict(row)
            candidate["child_operator"] = physical_operator(
                data["n"], edited_edges(data["n"], data["edges"], row.edit))
            candidates.append(candidate)
        rhs = 1 + len(rows)
        total_rhs += rhs
        fixtures.append({
            "name": data["name"], "source": source,
            "code_digest": code_digest, "protocol_digest": protocol_digest,
            "input_digest": _input_digest(data), "inputs": data,
            "scope": "constructed_exact_mathematical_fixture",
            "status": "passed" if passed else "failed",
            "reason": "all_named_reference_checks_passed" if passed
                      else "at_least_one_named_reference_check_failed",
            "relation_classification": relation,
            "relation_scope": "two_endpoint_flips" if data["n"] == 4 else None,
            "full_decision_verified": False,
            "full_decision_scope": "production_service_unavailable",
            "reference_catalog_decision_verified": passed,
            "parent_operator": physical_operator(data["n"], data["edges"]),
            "candidates": candidates, "reference_decision": decision,
            "checks": checks, "cost": {"category": "checker", "rhs_solves": rhs,
                                         "process_wall_seconds": None}, **extra,
        })
    passed = all(fixture["status"] == "passed" for fixture in fixtures)
    return {
        "schema": "j-cert-plus-v3-exact-micro-reference-v1",
        "source": source, "code_digest": code_digest,
        "protocol_digest": protocol_digest, "input_digest": _input_digest(inputs),
        "scope": "five_constructed_micro_fixtures_alpha_1",
        "status": "passed" if passed else "failed",
        "reason": "all_named_reference_checks_passed" if passed
                  else "at_least_one_named_reference_check_failed",
        "production_path": "unavailable_missing_v2_code",
        "production_validation_status": "not_started_missing_input",
        "full_decision_verified": False,
        "full_decision_scope": "production_service_unavailable",
        "independence": "stdlib Fraction; direct parent and child physical operators; "
                        "Gaussian elimination; no production import or local gain formula",
        "cost": {"category": "checker", "rhs_solves": total_rhs,
                 "internal_wall_seconds": time.perf_counter() - started,
                 "process_wall_seconds": None,
                 "scheduler_record": "v3/budget_ledger.jsonl"},
        "fixtures": fixtures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=V3_ROOT / "PROTOCOL.json")
    args = parser.parse_args()
    report = build_report(args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, default=fraction_json, indent=2,
                                      sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"],
                      "fixtures": len(report["fixtures"]),
                      "production_path": report["production_path"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
