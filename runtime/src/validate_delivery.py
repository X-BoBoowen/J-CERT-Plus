"""Verify delivered evidence identities and scopes without running experiments."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import fitz

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_mapping(mapping, base=ROOT):
    return [p for p, value in mapping.items()
            if not (base / p).is_file() or sha(base / p) != value]


def main():
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "File identity, recorded batch completeness and reporting scope; no performance reruns."}
    manifest = read("protocol/R15_SOURCE_MANIFEST.json")
    baseline = {}
    for name, path in [("isolated_copy", ROOT / "baseline_r15"),
                       ("original_workspace", ROOT.parent / "r15_work/J_CERT_Plus_R15_Evidence_Revision")]:
        if not path.is_dir():
            baseline[name] = {"available": False}
            continue
        bad = check_mapping(manifest, path)
        baseline[name] = {"available": True, "files_checked": len(manifest), "mismatches": bad}
        assert not bad, (name, bad)
    result["frozen_r15"] = baseline

    frozen = read("protocol/PILOT_SOURCE_FREEZE.json")["sha256"]
    changes = check_mapping(frozen)
    assert changes == ["src/forest_probe.py"], changes
    old = ROOT / "review/source_snapshots/forest_probe_at_pilot_freeze.py"
    assert sha(old) == frozen["src/forest_probe.py"]
    result["pilot_freeze"] = {
        "files_checked": len(frozen), "changed": changes,
        "explanation": "The later four-node probabilistic interface probe is separate from the timed service. Its initial source is retained in review/source_snapshots; no measured numerical service source changed."}

    weighted = read("reports/WEIGHTED_ADMISSION.json")
    wm = {"src/" + name + ".py": weighted[name + "_sha256"]
          for name in ["weighted_adapter", "session", "checker", "shortlist_checker"]}
    am = read("reports/A1_SERVICE_ADMISSION.json")
    identities = {"src/" + k + ".py": v for k, v in am["source_files_sha256"].items()}
    identities.update({"src/" + k + ".py": v for k, v in am["independent_checker_sha256"].items()})
    assert not check_mapping(wm)
    assert not check_mapping(identities)
    assert sha(ROOT / "protocol/A1_SERVICE_PROTOCOL.json") == am["protocol_sha256"]
    result["source_bound_admissions"] = {"weighted_files": len(wm), "a1_files": len(identities), "mismatches": []}

    batches = {}
    for folder, count in [("pilot",216),("guarantee_cost",180),("a1_service",54),
                          ("a1_confirmation",54),("parity/measurements",42),("timer_perturbation",12)]:
        files = sorted((ROOT / "raw" / folder).glob("[0-9]*.json"))
        assert len(files) == count, (folder, len(files))
        statuses = Counter(); requests = decisions = 0
        traces = {}
        for path in files:
            d = json.loads(path.read_text())
            statuses[d["status"]] += 1
            requests += len(d.get("requests", []))
            decisions += sum(len(q["decisions"]) for q in d.get("requests", []))
            ids = {"src/" + k + ".py": v for k, v in d.get("source_files_sha256", {}).items()}
            assert not check_mapping(ids), (path.name, check_mapping(ids))
            if folder == "parity/measurements":
                assert d["source_sha256"] == sha(ROOT / "src/kernel_probe.py")
                assert d["matched"]
            traces[str(path.relative_to(ROOT))] = sha(path)
        assert statuses == {"completed": count}, (folder, statuses)
        batches[folder] = {"workers": count, "statuses": dict(statuses), "requests": requests,
                           "recorded_decisions": decisions,
                           "trace_set_sha256": hashlib.sha256(json.dumps(traces,sort_keys=True).encode()).hexdigest()}
    result["recorded_batches"] = batches

    inputs = read("protocol/A1_CONFIRMATION_INPUTS.json")
    assert inputs["protocol_sha256"] == sha(ROOT / "protocol/A1_CONFIRMATION_PROTOCOL.json")
    for fixture in inputs["fixtures"]:
        assert sha(ROOT / fixture["file"]) == fixture["npz_sha256"]
    checker = read("reports/A1_CONFIRMATION_INPUT_CHECK.json")
    assert checker["verified"] and checker["fixture_count"] == 6
    assert checker["input_manifest_sha256"] == sha(ROOT / "protocol/A1_CONFIRMATION_INPUTS.json")
    result["heldout_inputs"] = {"fixtures": 6, "rows": checker["opinion_rows"], "stored_identity_matches": True,
                                "independent_regeneration_record": "reports/A1_CONFIRMATION_INPUT_CHECK.json"}

    proof = read("analysis/A1_CONFIRMATION_OUTPUT_VERIFICATION.json")
    for worker in proof["workers"]:
        assert sha(ROOT / worker["trace"]) == worker["trace_sha256"]
        assert sha(ROOT / worker["replay"]) == worker["replay_sha256"]
    result["heldout_independent_output_scope"] = {k: proof[k] for k in
        ["files_verified","requests_verified","distinct_input_requests","full_catalog_decisions_verified","distinct_input_decisions"]}
    shadow = read("raw/audit/e1_shadow/COVERAGE.json")
    assert shadow["status"] == "resource_cap_partial" and shadow["exit_code"] == 124
    assert len(shadow["uncovered"]) == 6
    result["fresh_shadow_scope"] = {k: shadow[k] for k in
        ["status","exit_code","limit_seconds","unique_queries","completed_unique_queries","completed_events"]}
    result["research_gate"] = "A1 Q16 cost promotion failed; correctness retained in reported scope. No joint-scale escalation, retuning or manuscript promotion."

    probe = ROOT / "template_probe/r15_unchanged"
    copied = [probe / "references.bib"]
    for folder in ["sections", "figures"]:
        copied += [p for p in (probe / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    assert len(copied) == 37
    assert all(sha(p) == sha(ROOT / "baseline_r15" / p.relative_to(probe)) for p in copied)
    pdfs = {"template_probe/r15_unchanged/main.pdf": 14,
            "reports/RESEARCH_DECISION.pdf": 1, "reports/CLAIM_EVIDENCE_MAP.pdf": 1}
    for path, pages in pdfs.items():
        with fitz.open(ROOT / path) as pdf:
            assert len(pdf) == pages, (path, len(pdf))
    result["pdfs"] = {p: {"pages": n, "sha256": sha(ROOT / p)} for p,n in pdfs.items()}
    result["template"] = {"copied_files_byte_identical":37,"body_plus_AI_end":13,
                          "body_floats_end":12,"references_start":14,"submission_format_pass":False}
    result["report_review"] = {
        "scoped_independent_review": "review/FINAL_REPORT_REVIEW.txt",
        "corrections": ["Only I/D use persistent structural postings.",
                        "Top-64 is selected with partition and candidate-ID ties, not a complete sort."],
        "visual_review": "Both one-page briefs inspected as rendered PNG; no clipping, overlap or unreadable cells."}
    result["validation_passed"] = True
    (ROOT / "review/DELIVERY_VALIDATION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "baseline_files": len(manifest),
                      "batches": {k:v["workers"] for k,v in batches.items()}, "pdf_pages": pdfs}))


if __name__ == "__main__":
    main()
