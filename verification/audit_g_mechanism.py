#!/usr/bin/env python3
"""Recalculate G mechanism counts and compare saved independent enclosures.

Only Python's standard library is used. Code archives are read as text.
The audit does not execute their code or recreate any input, graph or solve.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile


ARCHIVES = {
    "v4_evidence": "artifacts/frozen/V4_Evidence.zip",
    "v4_code": "artifacts/frozen/V4_Code.zip",
}
V3_FREEZE = "v3/resumed/protocols/HG_FROZEN.json"
V4_FREEZE = "v4/protocols/HG_LIGHT_FROZEN.json"
V3_RAW = "v3/resumed/raw_hg/"
V4_RAW = "v4/raw_hg_light/"
V3_ANALYSIS = "v3/resumed/reports/hg_verified/"
IDENTITY = ("epoch", "parent_version_id", "current_version_id", "winner_id",
            "i", "j", "d", "flip")
TOP_BINDINGS = ("root_version_id", "operator_hash", "catalog_hash", "graph_hash",
                "n", "catalog_size")


def require(condition, context):
    if not condition:
        raise ValueError(context)


def digest(data):
    return sha256(data).hexdigest()


class Sources:
    def __init__(self, workspace, stack):
        self.archives = {
            label: stack.enter_context(ZipFile(workspace / path))
            for label, path in ARCHIVES.items()
        }
        self.provenance = {
            label: {"path": path, "sha256": digest((workspace / path).read_bytes()),
                    "members": {}}
            for label, path in ARCHIVES.items()
        }

    def read(self, archive, member):
        content = self.archives[archive].read(member)
        self.provenance[archive]["members"][member] = {
            "bytes": len(content), "sha256": digest(content)
        }
        return content

    def json(self, archive, member):
        return json.loads(self.read(archive, member))

    def jsonl(self, archive, member):
        return [json.loads(line) for line in self.read(archive, member).splitlines()]


def identity(entry):
    return tuple(entry[key] for key in IDENTITY)


def read_runtime_source(sources, generation, freeze, name):
    content = sources.read(generation + "_code", "runtime_files/" + name)
    require(digest(content) == freeze["source_files_sha256"][name], name + "/source pin")
    return content


def read_worker(sources, archive, freeze, prefix, job):
    name = prefix + job["output_file"]
    worker = sources.json(archive, name)
    freeze_member = V4_FREEZE if archive == "v4_evidence" else V3_FREEZE
    freeze_hash = sources.provenance[archive]["members"][freeze_member]["sha256"]
    require(worker["job"] == job and worker["freeze_sha256"] == freeze_hash
            and worker["actual_runtime_commit"] == freeze["runtime_commit"], name)
    require(len(worker["requests"]) == job["requests"], name)
    if archive == "v4_evidence":
        require(worker["measurement_policy"] == "light_v1"
                and worker["run_kind"] == freeze["run_kind"] == "formal", name)
    return name, worker


def counts(states):
    return {
        "recorded_decision_states": len(states),
        "whole_insertion_excluded_states": sum(row["whole_insertion_excluded"] for row in states),
        "entered_block_states": sum(row["blocks_tested"] > 0 for row in states),
        "expanded_block_states": sum(row["blocks_expanded"] > 0 for row in states),
        "states_with_at_least_one_leaf": sum(row["leaves_generated"] > 0 for row in states),
        "states_with_at_least_one_emission": sum(row["insertions_emitted"] > 0 for row in states),
        "both_whole_exclusion_and_blocks_states": sum(
            row["whole_insertion_excluded"] and row["blocks_tested"] > 0 for row in states),
        "neither_whole_exclusion_nor_blocks_states": sum(
            not row["whole_insertion_excluded"] and row["blocks_tested"] == 0 for row in states),
        "winner_or_terminal_counts": dict(Counter(row["winner_or_terminal"] for row in states)),
        "whole_exclusion_by_winner_or_terminal": dict(Counter(
            row["winner_or_terminal"] for row in states if row["whole_insertion_excluded"])),
        "block_entry_by_winner_or_terminal": dict(Counter(
            row["winner_or_terminal"] for row in states if row["blocks_tested"] > 0)),
        "blocks_tested": sum(row["blocks_tested"] for row in states),
        "blocks_expanded": sum(row["blocks_expanded"] for row in states),
        "leaves_generated": sum(row["leaves_generated"] for row in states),
        "insertions_emitted": sum(row["insertions_emitted"] for row in states),
        "leaves_by_rejection_type": {
            cause: sum(row["leaf_rejects"][cause] for row in states)
            for cause in ("graph", "chosen", "processed")
        },
        "constant_cap_pass_at_same_cutoff": sum(row["constant_cap_pass_at_same_cutoff"]
                                              for row in states),
        "range_cap_only_pass_at_same_cutoff": sum(row["range_cap_only_pass_at_same_cutoff"]
                                                for row in states),
        "global_caps_strictly_below_constant": sum(
            row["recorded_global_cap_strictly_below_constant"] for row in states),
    }


def audit_mechanism(sources, freeze):
    contracts = freeze["contracts"]
    scale = contracts["operator_scale"]
    proof, screen = contracts["proof_unit"], contracts["screen_unit"]
    code = read_runtime_source(sources, "v4", freeze, "src/g_access.py").decode()
    access = read_runtime_source(sources, "v4", freeze, "src/a2_access.py").decode()
    session = read_runtime_source(sources, "v4", freeze, "src/a2_session.py").decode()
    for snippet in ("return min((2 * SCREEN // SCALE)**2, width**2)",
                    "target = -(-int(best)//RATIO)",
                    "if upper < target or (best == 0 and upper <= 0):"):
        require(snippet in code, "G cap source differs: " + snippet)
    require("RATIO=(UNIT//SCREEN)**2" in access, "coarse/fine ratio source differs")
    require("access['cold_end_lower_fine']=int(best)" in session, "saved cutoff source differs")
    require(contracts["insertion_weight"] == "1/16" and scale == 16 and proof % screen == 0,
            "cap units differ")
    constant = (2 * screen // scale) ** 2
    ratio = (proof // screen) ** 2
    by_state = defaultdict(list)
    for job in freeze["jobs"]:
        if job["method"] != "G":
            continue
        name, worker = read_worker(sources, "v4_evidence", freeze, V4_RAW, job)
        for request in worker["requests"]:
            epochs = request["access"]["epochs"]
            require(len(epochs) == len(request["decisions"]), name)
            for event, decision in zip(epochs, request["decisions"]):
                context = f"{name}/q{request['query_position']}/e{event['epoch']}"
                require(event["source"] == "G" and event["epoch"] == decision["epoch"]
                        and event["parent_version_id"] == decision["parent_version_id"], context)
                winner = decision["status"]
                if winner == "accepted":
                    require(decision["flip"] in (0, 1), context)
                    winner = "flip" if decision["flip"] else "insertion"
                best = event["cold_end_lower_fine"]
                cutoff = -(-best // ratio)
                upper = event["g_global_insertion_upper_coarse"]
                require(type(upper) is int and 0 <= upper <= constant, context)
                reported_pass = event["g_whole_insertion_excluded"]
                require((upper < cutoff or (best == 0 and upper <= 0)) == reported_pass,
                        context + "/range gate")
                constant_pass = constant < cutoff or (best == 0 and constant <= 0)
                leaf_rejections = {kind: event[f"g_pair_{kind}_rejects"]
                                   for kind in ("graph", "chosen", "processed")}
                require(event["g_pair_leaves_generated"] == sum(leaf_rejections.values())
                        + event["g_block_insertions_emitted"], context + "/leaf accounting")
                require(event["g_block_insertions_emitted"] == event["insertion_emissions"],
                        context + "/insertion accounting")
                state_key = tuple(worker[field] for field in
                                  ("graph_hash", "operator_hash", "catalog_hash"))
                state_key += (request["opinion_hash"], event["parent_version_id"])
                row = {
                    "graph": worker["graph"], "stream": worker["stream"],
                    "query_position": request["query_position"], "epoch": event["epoch"],
                    "complete_q4": worker["complete_Q4"], "winner_or_terminal": winner,
                    "winner_id": decision["winner_id"],
                    "child_version_id": decision["current_version_id"],
                    "source_complete": event["source_complete"],
                    "whole_insertion_excluded": reported_pass,
                    "blocks_tested": event["g_pair_blocks_tested"],
                    "blocks_expanded": event["g_pair_blocks_expanded"],
                    "leaves_generated": event["g_pair_leaves_generated"],
                    "insertions_emitted": event["g_block_insertions_emitted"],
                    "deferred_warm_insertions": event["g_deferred_warm_insertions"],
                    "leaf_rejects": leaf_rejections, "coarse_cutoff": cutoff,
                    "cold_end_lower_fine": best, "recorded_global_cap_coarse": upper,
                    "recorded_global_cap_strictly_below_constant": upper < constant,
                    "constant_cap_pass_at_same_cutoff": constant_pass,
                    "range_cap_only_pass_at_same_cutoff": reported_pass and not constant_pass,
                }
                by_state[state_key].append((worker["repeat_id"], name, row))
    work, unique = [], []
    key_fields = ("graph_sha256", "root_operator_sha256", "catalog_sha256",
                  "opinion_sha256", "parent_version_id")
    for state_key, observations in by_state.items():
        observations.sort()
        require([repeat for repeat, _, _ in observations] == [0, 1, 2], state_key)
        representative = observations[0][2]
        require(all(row == representative for _, _, row in observations), state_key)
        unique.append({**dict(zip(key_fields, state_key)), **representative,
                       "source_members": [name for _, name, _ in observations]})
        work.extend(row for _, _, row in observations)
    return {
        "state_key_fields": key_fields,
        "repeat_observation_rule": "Require identical audited fields in repeats 0, 1, and 2.",
        "block_entry_rule": "g_pair_blocks_tested > 0, including testing the root block.",
        "constant_cap_coarse": constant,
        "fine_to_coarse_ratio": ratio,
        "cap_scope": ("Compute ceil(cold_end_lower_fine / RATIO) using the same rounding and "
                      "strict cutoff rule as the access code. Substituting 4 omega at this "
                      "recorded cold-loop end threshold gives a threshold comparison. "
                      "Per-call next_batch/best history is absent, so agreement with the saved "
                      "flag does not establish when the first global gate would have fired, "
                      "the incumbent at its first call, an alternative search, or runtime."),
        "all_recorded": counts(work),
        "complete_q4_recorded": counts([row for row in work if row["complete_q4"]]),
        "all_unique": counts(unique),
        "complete_q4_unique": counts([row for row in unique if row["complete_q4"]]),
        "all_unique_by_stream": {
            graph + "/" + stream: counts([row for row in unique
                                         if (row["graph"], row["stream"]) == (graph, stream)])
            for graph, stream in sorted({(row["graph"], row["stream"]) for row in unique})
        },
        "unique_states": unique,
    }


def interval_words(bounds):
    return [str(value) for value in bounds]



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    with ExitStack() as stack:
        sources = Sources(args.workspace.resolve(), stack)
        freeze = sources.json('v4_evidence', V4_FREEZE)
        result = audit_mechanism(sources, freeze)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'mechanism': result, 'sources': sources.provenance}, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'status':'passed','complete_states':result['complete_q4_unique']['recorded_decision_states'],'all_states':result['all_unique']['recorded_decision_states']}))

if __name__ == '__main__': main()
