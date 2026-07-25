"""Frozen CAP2 symbolic operator suite and anti-cheat scoreboard."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

from slm_training.dsl.operators.contracts import _fingerprint
from slm_training.dsl.schema import load_jsonl
from slm_training.evals.power_protocol import wilson_interval
from slm_training.harnesses.train_data.operator_corpus import (
    OperatorCorpusConfig,
    build_symbolic_operator_corpus,
)

CAP2_SUITE_SCHEMA = "cap2_operator_suite/v1"
CAP2_PREDICTION_SCHEMA = "cap2_operator_prediction/v1"
CAP2_SCORE_SCHEMA = "cap2_operator_score/v1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain only JSON objects")
    return rows


def _arguments_fingerprint(arguments: list[dict[str, Any]]) -> str:
    return _fingerprint(
        {"schema": "cap2_argument_target/v1", "arguments": arguments}
    )


def _effect_fingerprint(application: Mapping[str, Any]) -> str:
    proof = application.get("proof")
    if not isinstance(proof, Mapping):
        raise ValueError("gold application is missing replay proof")
    value = str(proof.get("effect_fingerprint") or "")
    if len(value) != 64:
        raise ValueError("gold application effect fingerprint is invalid")
    return value


def _transition_case(row: Mapping[str, Any]) -> dict[str, Any]:
    action = row["legal_action"]
    application = row["application"]
    preference = row["canonical_preference"]
    trace = row["conversation_trace"]
    if not all(
        isinstance(value, Mapping)
        for value in (action, application, preference, trace)
    ):
        raise ValueError("transition gold is incomplete")
    steps = preference["steps"]
    if not isinstance(steps, list) or len(steps) != 1:
        raise ValueError("transition preference must contain exactly one step")
    stratum = (
        "held_out_composition"
        if row["kind"] == "next_turn"
        else "held_out_transition"
    )
    return {
        "schema": "cap2_operator_case/v1",
        "case_id": f"transition:{row['example_id']}",
        "stratum": stratum,
        "source_record_id": row["source_record_id"],
        "before_ast": row["before_ast"],
        "gold": {
            "accepted_legal_action": True,
            "operator_id": action["operator_id"],
            "semantic_action_id": action["semantic_id"],
            "argument_fingerprint": _arguments_fingerprint(action["arguments"]),
            "result_ast": row["after_ast"],
            "effect_fingerprint": _effect_fingerprint(application),
            "locality_violations": steps[0]["locality_violations"],
            "unintended_edits": 0,
            "final_state_id": trace["current_state_id"],
            "operator_count": 1,
        },
        "evidence": {
            "legal_set_fingerprint": row["legal_set_fingerprint"],
            "application_id": application.get("application_id")
            or action["application_id"],
            "trace_fingerprint": _fingerprint(dict(trace)),
            "intermediate_replay": True,
        },
    }


def _history_case(row: Mapping[str, Any]) -> dict[str, Any]:
    trace = row["conversation_trace"]
    answer = row["answer"]
    if not isinstance(trace, Mapping) or not isinstance(answer, Mapping):
        raise ValueError("history gold is incomplete")
    return {
        "schema": "cap2_operator_case/v1",
        "case_id": f"history:{row['example_id']}",
        "stratum": "history_branch_isolation",
        "source_record_id": row["source_record_id"],
        "before_ast": row["before_ast"],
        "gold": {
            "history_operation": answer["operation"],
            "history_state_id": answer["state_id"],
            "branch_digest": answer["branch_digest"],
            "result_ast": row["after_ast"],
            "intermediate_legal": True,
            "branch_isolated": True,
            "operator_count": 0,
        },
        "evidence": {
            "trace_fingerprint": _fingerprint(dict(trace)),
            "root_state_id": trace["root_state_id"],
            "final_state_id": trace["current_state_id"],
        },
    }


def _collapse_case(row: Mapping[str, Any]) -> dict[str, Any]:
    collapse = row["collapse"]
    trace = row["conversation_trace"]
    if not isinstance(collapse, Mapping) or not isinstance(trace, Mapping):
        raise ValueError("collapse gold is incomplete")
    if collapse["final_state_id"] != trace["current_state_id"]:
        raise ValueError("collapse final state differs from replayed trace")
    return {
        "schema": "cap2_operator_case/v1",
        "case_id": f"collapse:{row['example_id']}",
        "stratum": "sequential_vs_collapsed",
        "source_record_id": row["source_record_id"],
        "before_ast": row["question"]["state_ast"],
        "gold": {
            "result_ast": row["answer"]["result_ast"],
            "required_order": collapse["required_order"],
            "final_state_id": collapse["final_state_id"],
            "intermediate_legal": True,
            "operator_count": len(collapse["applications"]),
        },
        "evidence": {
            "collapse_id": collapse["collapse_id"],
            "turn_ids": collapse["turn_ids"],
            "operator_ids": collapse["operator_ids"],
            "trace_fingerprint": _fingerprint(dict(trace)),
        },
    }


def _negative_cases(
    collapsed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cases = []
    for row in collapsed_rows:
        for index, negative in enumerate(row["collapse"]["hard_negatives"]):
            cases.append(
                {
                    "schema": "cap2_operator_case/v1",
                    "case_id": f"reordered:{row['example_id']}:{index}",
                    "stratum": "reordered_noncommuting",
                    "source_record_id": row["source_record_id"],
                    "before_ast": row["question"]["state_ast"],
                    "gold": {
                        "expected_outcome": negative["outcome"],
                        "rejection_code": negative["conflict_code"],
                        "required_order": row["collapse"]["required_order"],
                        "illegal_action_mass_max": 0.0,
                        "operator_count": len(negative["application_ids"]),
                    },
                    "evidence": {
                        "application_ids": negative["application_ids"],
                        "swapped_step_indices": negative[
                            "swapped_step_indices"
                        ],
                    },
                }
            )
    return cases


def _stale_reference_cases(
    negative_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "schema": "cap2_operator_case/v1",
            "case_id": f"stale:{case['case_id']}",
            "stratum": "stale_reference",
            "source_record_id": case["source_record_id"],
            "before_ast": case["before_ast"],
            "gold": {
                "expected_outcome": "conflict",
                "rejection_code": case["gold"]["rejection_code"],
                "illegal_action_mass_max": 0.0,
                "operator_count": 0,
            },
            "evidence": {
                **case["evidence"],
                "stale_reference_source": "reordered_application",
            },
        }
        for case in negative_cases
        if str(case["gold"].get("rejection_code") or "").startswith("ref.")
    ]


def _undo_redo_cases(
    history_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "schema": "cap2_operator_case/v1",
            "case_id": f"undo-redo:{case['case_id']}",
            "stratum": "history_undo_redo",
            "source_record_id": case["source_record_id"],
            "before_ast": case["before_ast"],
            "gold": {
                "undo_state_id": case["evidence"]["root_state_id"],
                "redo_state_id": case["evidence"]["final_state_id"],
                "intermediate_legal": True,
                "operator_count": 0,
            },
            "evidence": {
                "trace_fingerprint": case["evidence"]["trace_fingerprint"],
                "exact_parent_child_identity": True,
            },
        }
        for case in history_cases
    ]


def _merge_contract_cases() -> list[dict[str, Any]]:
    return [
        {
            "schema": "cap2_operator_case/v1",
            "case_id": "merge:replay-success",
            "stratum": "merge_replay",
            "source_record_id": "cap2_contract",
            "before_ast": "",
            "gold": {
                "merge_replay": True,
                "order_invariant": True,
                "operator_count": 2,
            },
            "evidence": {
                "contract": "branch_merge/v1",
                "authority": "replay_branch_merge",
            },
        },
        {
            "schema": "cap2_operator_case/v1",
            "case_id": "merge:typed-conflict",
            "stratum": "merge_conflict",
            "source_record_id": "cap2_contract",
            "before_ast": "",
            "gold": {
                "conflict_typed": True,
                "state_mutated": False,
                "operator_count": 0,
            },
            "evidence": {
                "contract": "merge_conflict/v1",
                "kinds": [
                    "same_node",
                    "delete_modify",
                    "role_cardinality",
                    "child_order",
                    "scope_binder",
                    "stale_ref",
                    "unsupported_effect",
                ],
            },
        },
    ]


def _permutation_cases(
    transition_cases: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    first_by_source: dict[str, dict[str, Any]] = {}
    for case in transition_cases:
        first_by_source.setdefault(case["source_record_id"], case)
    return [
        {
            "schema": "cap2_operator_case/v1",
            "case_id": f"permutation:{case['case_id']}",
            "stratum": "marker_permutation",
            "source_record_id": case["source_record_id"],
            "before_ast": case["before_ast"],
            "gold": {
                "semantic_action_id": case["gold"]["semantic_action_id"],
                "result_ast": case["gold"]["result_ast"],
                "marker_invariant": True,
                "permutation_seed": seed,
                "operator_count": 1,
            },
            "evidence": {
                "legal_set_fingerprint": case["evidence"][
                    "legal_set_fingerprint"
                ],
                "opaque_serialization_is_not_gold": True,
            },
        }
        for case in first_by_source.values()
    ]


def build_frozen_cap2_suite(
    *,
    manifest_path: Path,
    source_records_path: Path,
    work_dir: Path,
    version_stamp: dict[str, Any],
) -> dict[str, Any]:
    """Regenerate the frozen held-out suite through canonical operator replay."""
    manifest = _json(manifest_path)
    if manifest.get("schema") != "cap2_operator_suite_manifest/v1":
        raise ValueError("unsupported CAP2 suite manifest")
    selected_ids = tuple(manifest["source_record_ids"])
    records_by_id = {
        record.id: record for record in load_jsonl(source_records_path)
    }
    if set(selected_ids) - set(records_by_id):
        raise ValueError("CAP2 manifest names unavailable source records")
    selected = [records_by_id[record_id] for record_id in selected_ids]
    source_fingerprint = _fingerprint(
        {
            "schema": "cap2_source_records/v1",
            "records": [record.to_dict() for record in selected],
        }
    )
    if source_fingerprint != manifest["source_records_fingerprint"]:
        raise ValueError("CAP2 held-out source records drifted")

    generated = build_symbolic_operator_corpus(
        records=selected,
        output_dir=work_dir / "generated",
        version=str(manifest["suite_version"]),
        version_stamp=version_stamp,
        config=OperatorCorpusConfig(
            max_roots=len(selected),
            actions_per_state=int(manifest["generation"]["actions_per_state"]),
            max_combinations_per_operator=int(
                manifest["generation"]["max_combinations_per_operator"]
            ),
            sibling_forks=True,
        ),
    )
    if (
        generated["content_fingerprint"]
        != manifest["operator_corpus_fingerprint"]
    ):
        raise ValueError("CAP2 generated operator corpus drifted")
    operator_rows = _jsonl(Path(generated["records_path"]))
    collapsed_rows = _jsonl(Path(generated["collapsed_records_path"]))
    transitions = [
        _transition_case(row)
        for row in operator_rows
        if row["target_view"] == "dual" and row["outcome"] == "success"
    ]
    histories = [
        _history_case(row)
        for row in operator_rows
        if row["kind"] == "sibling_fork"
    ]
    collapses = [_collapse_case(row) for row in collapsed_rows]
    negative_cases = _negative_cases(collapsed_rows)
    cases = [
        *transitions,
        *histories,
        *_undo_redo_cases(histories),
        *collapses,
        *negative_cases,
        *_stale_reference_cases(negative_cases),
        *_merge_contract_cases(),
        *_permutation_cases(
            transitions, seed=int(manifest["marker_permutation_seed"])
        ),
    ]
    cases.sort(key=lambda case: case["case_id"])
    suite = {
        "schema": CAP2_SUITE_SCHEMA,
        "suite_version": manifest["suite_version"],
        "source_records_fingerprint": source_fingerprint,
        "operator_corpus_fingerprint": generated["content_fingerprint"],
        "cases": cases,
        "thresholds": manifest["thresholds"],
        "contract_inventory": manifest["contract_inventory"],
        "nl": {
            "available": False,
            "reason": "CERT_CAP1_unavailable",
            "dependency_issue": "SLM-379",
        },
        "version_stamp": version_stamp,
    }
    suite_hash = _fingerprint(
        {key: value for key, value in suite.items() if key != "version_stamp"}
    )
    suite["suite_hash"] = suite_hash
    if suite_hash != manifest["suite_hash"]:
        raise ValueError("CAP2 frozen suite hash drifted")
    required = set(manifest["required_strata"])
    actual = {case["stratum"] for case in cases}
    if required - actual:
        raise ValueError(f"CAP2 suite misses strata: {sorted(required - actual)}")
    return suite


def oracle_prediction(case: Mapping[str, Any]) -> dict[str, Any]:
    gold = dict(case["gold"])
    return {
        "schema": CAP2_PREDICTION_SCHEMA,
        "case_id": case["case_id"],
        "accepted_legal_action_mass": (
            1.0 if gold.get("accepted_legal_action") else 0.0
        ),
        **gold,
        "telemetry": {
            "active_nodes": 1,
            "node_passes": 1,
            "remask_phases": 0,
            "model_calls": 0,
            "compiler_calls": 1,
            "verifier_calls": 1,
            "latency_ms": 0.0,
            "peak_memory_bytes": 0,
        },
    }


def _check(
    checks: dict[str, bool],
    name: str,
    prediction: Mapping[str, Any],
    gold: Mapping[str, Any],
    key: str,
) -> None:
    checks[name] = prediction.get(key) == gold.get(key)


def score_cap2_predictions(
    suite: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Score every applicable CAP2 dimension and fail closed on missing rows."""
    cases = list(suite["cases"])
    rows = []
    dimension_counts: dict[str, list[bool]] = defaultdict(list)
    legal_mass = []
    telemetry: dict[str, float] = defaultdict(float)
    for case in cases:
        prediction = predictions.get(case["case_id"], {})
        gold = case["gold"]
        checks: dict[str, bool] = {}
        if case["stratum"] in {
            "held_out_transition",
            "held_out_composition",
        }:
            mass = float(prediction.get("accepted_legal_action_mass", 0.0))
            legal_mass.append(mass)
            checks["accepted_legal_action_mass"] = mass >= float(
                suite["thresholds"]["accepted_legal_action_mass_min"]
            )
            for name, key in (
                ("operator_id", "operator_id"),
                ("typed_arguments", "argument_fingerprint"),
                ("canonical_ast", "result_ast"),
                ("action_effect", "effect_fingerprint"),
                ("locality", "locality_violations"),
                ("unintended_edits", "unintended_edits"),
                ("final_state", "final_state_id"),
            ):
                _check(checks, name, prediction, gold, key)
        elif case["stratum"] == "history_branch_isolation":
            for name, key in (
                ("history_operation", "history_operation"),
                ("history_state", "history_state_id"),
                ("branch_isolation", "branch_isolated"),
                ("canonical_ast", "result_ast"),
                ("intermediate_legality", "intermediate_legal"),
            ):
                _check(checks, name, prediction, gold, key)
        elif case["stratum"] == "history_undo_redo":
            for name, key in (
                ("undo_state", "undo_state_id"),
                ("redo_state", "redo_state_id"),
                ("intermediate_legality", "intermediate_legal"),
            ):
                _check(checks, name, prediction, gold, key)
        elif case["stratum"] == "sequential_vs_collapsed":
            for name, key in (
                ("canonical_ast", "result_ast"),
                ("required_order", "required_order"),
                ("final_state", "final_state_id"),
                ("intermediate_legality", "intermediate_legal"),
            ):
                _check(checks, name, prediction, gold, key)
        elif case["stratum"] == "merge_replay":
            for name, key in (
                ("merge_replay", "merge_replay"),
                ("merge_order_invariant", "order_invariant"),
            ):
                _check(checks, name, prediction, gold, key)
        elif case["stratum"] == "merge_conflict":
            for name, key in (
                ("merge_conflict_typed", "conflict_typed"),
                ("merge_no_mutation", "state_mutated"),
            ):
                _check(checks, name, prediction, gold, key)
        elif case["stratum"] in {
            "reordered_noncommuting",
            "stale_reference",
        }:
            for name, key in (
                ("negative_outcome", "expected_outcome"),
                ("conflict_code", "rejection_code"),
            ):
                _check(checks, name, prediction, gold, key)
            if "required_order" in gold:
                _check(
                    checks,
                    "required_order",
                    prediction,
                    gold,
                    "required_order",
                )
            checks["illegal_action_mass"] = float(
                prediction.get("accepted_legal_action_mass", 1.0)
            ) <= float(gold["illegal_action_mass_max"])
        else:
            for name, key in (
                ("marker_semantic_identity", "semantic_action_id"),
                ("marker_result_invariance", "result_ast"),
                ("marker_invariant", "marker_invariant"),
            ):
                _check(checks, name, prediction, gold, key)
        expected_count = int(gold["operator_count"])
        actual_count = int(prediction.get("operator_count", -1))
        checks["anti_bloat"] = 0 <= actual_count <= expected_count
        result_ast = str(prediction.get("result_ast") or "")
        if result_ast:
            checks["nonempty"] = True
            if case["stratum"] != "history_branch_isolation":
                checks["nontrivial"] = result_ast != str(
                    case.get("before_ast") or ""
                )
        elif "result_ast" in gold:
            checks["nonempty"] = False
            checks["nontrivial"] = False
        for name, passed in checks.items():
            dimension_counts[name].append(passed)
        raw_telemetry = prediction.get("telemetry") or {}
        for name in (
            "active_nodes",
            "node_passes",
            "remask_phases",
            "model_calls",
            "compiler_calls",
            "verifier_calls",
            "latency_ms",
            "peak_memory_bytes",
        ):
            value = float(raw_telemetry.get(name, 0.0))
            if value < 0:
                raise ValueError(f"negative CAP2 telemetry: {name}")
            telemetry[name] += value
        rows.append(
            {
                "case_id": case["case_id"],
                "stratum": case["stratum"],
                "checks": checks,
                "pass": bool(checks) and all(checks.values()),
            }
        )
    dimensions = {}
    min_lower = float(suite["thresholds"]["dimension_wilson_lower_min"])
    for name, values in sorted(dimension_counts.items()):
        successes = sum(values)
        interval = wilson_interval(successes, len(values))
        dimensions[name] = {
            "successes": successes,
            "n": len(values),
            "rate": successes / len(values),
            "interval": interval,
            "pass": (
                successes == len(values)
                and float(interval["low"] or 0.0) >= min_lower
            ),
        }
    case_successes = sum(row["pass"] for row in rows)
    case_interval = wilson_interval(case_successes, len(rows))
    mean_legal_mass = (
        sum(legal_mass) / len(legal_mass) if legal_mass else 0.0
    )
    gate_pass = (
        case_successes == len(rows)
        and float(case_interval["low"] or 0.0)
        >= float(suite["thresholds"]["case_wilson_lower_min"])
        and all(value["pass"] for value in dimensions.values())
        and mean_legal_mass
        >= float(suite["thresholds"]["accepted_legal_action_mass_min"])
    )
    return {
        "schema": CAP2_SCORE_SCHEMA,
        "suite_hash": suite["suite_hash"],
        "case_count": len(rows),
        "case_successes": case_successes,
        "case_pass_rate": case_successes / len(rows),
        "case_interval": case_interval,
        "mean_accepted_legal_action_mass": mean_legal_mass,
        "dimensions": dimensions,
        "rows": rows,
        "telemetry": dict(sorted(telemetry.items())),
        "retention_diagnostics": {
            "cap0": {
                "status": "unavailable",
                "reason": "fixture policy has no learned checkpoint",
            },
            "cap1": {
                "status": "unavailable",
                "reason": "CERT_CAP1 has not been issued",
            },
            "strict_metrics": {
                "status": "unavailable",
                "reason": "fixture contract run is not a model evaluation",
            },
            "agentv": {"status": "published_by_runner"},
        },
        "gate_pass": gate_pass,
    }


def evaluate_fixture_policies(
    suite: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Run the oracle plus three degenerate policies used as anti-cheat controls."""
    cases = list(suite["cases"])
    oracle = {case["case_id"]: oracle_prediction(case) for case in cases}
    first_transition = next(
        case
        for case in cases
        if case["stratum"]
        in {"held_out_transition", "held_out_composition"}
    )
    constant_gold = first_transition["gold"]

    def control(
        transform: Callable[
            [Mapping[str, Any], Mapping[str, Any]], dict[str, Any]
        ],
    ) -> dict[str, dict[str, Any]]:
        return {
            case["case_id"]: {
                "schema": CAP2_PREDICTION_SCHEMA,
                "case_id": case["case_id"],
                "operator_count": 0,
                "telemetry": {},
                **transform(case, case["gold"]),
            }
            for case in cases
        }

    unchanged = control(
        lambda case, _gold: {
            "accepted_legal_action_mass": 0.0,
            "result_ast": case["before_ast"],
        }
    )
    generic = control(
        lambda _case, _gold: {
            "accepted_legal_action_mass": 0.0,
            "result_ast": 'root = TextContent(":generic")',
        }
    )
    constant = control(
        lambda _case, _gold: {
            "accepted_legal_action_mass": 1.0,
            "operator_id": constant_gold["operator_id"],
            "semantic_action_id": constant_gold["semantic_action_id"],
            "argument_fingerprint": constant_gold["argument_fingerprint"],
            "result_ast": constant_gold["result_ast"],
            "effect_fingerprint": constant_gold["effect_fingerprint"],
            "locality_violations": 0,
            "unintended_edits": 0,
            "final_state_id": constant_gold["final_state_id"],
            "operator_count": 1,
        }
    )
    return {
        "oracle": score_cap2_predictions(suite, oracle),
        "unchanged": score_cap2_predictions(suite, unchanged),
        "generic_valid_ast": score_cap2_predictions(suite, generic),
        "constant_operator": score_cap2_predictions(suite, constant),
    }


def suite_hash_payload(suite: Mapping[str, Any]) -> str:
    """Return the canonical public suite identity."""
    return str(suite["suite_hash"])


__all__ = [
    "CAP2_PREDICTION_SCHEMA",
    "CAP2_SCORE_SCHEMA",
    "CAP2_SUITE_SCHEMA",
    "build_frozen_cap2_suite",
    "evaluate_fixture_policies",
    "oracle_prediction",
    "score_cap2_predictions",
    "suite_hash_payload",
]
