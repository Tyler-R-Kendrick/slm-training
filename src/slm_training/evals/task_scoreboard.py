"""Availability-aware per-task and L3-L5 equivalence scoreboards."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from slm_training.data.leakage import norm_text
from slm_training.dsl.parser import ParseError, lexical_tokens, validate, validate_output
from slm_training.dsl.schema import OutputTarget

_BINDER_RE = re.compile(r"(?m)^\s*([a-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_IDENT_RE = re.compile(r"\b([a-z_][A-Za-z0-9_]*)\b")


@dataclass(frozen=True)
class _Tree:
    label: str
    children: tuple[_Tree, ...] = ()

    @property
    def size(self) -> int:
        return 1 + sum(child.size for child in self.children)


def _semantic_tree(value: Any) -> _Tree | None:
    if not isinstance(value, Mapping):
        return None
    if value.get("type") == "element":
        children: list[_Tree] = []
        props = value.get("props")
        if isinstance(props, Mapping):
            stack = list(props.values())
            while stack:
                child = stack.pop(0)
                if isinstance(child, list):
                    stack[0:0] = child
                elif isinstance(child, Mapping):
                    node = _semantic_tree(child)
                    if node is not None:
                        children.append(node)
                    else:
                        stack[0:0] = list(child.values())
        return _Tree(str(value.get("typeName") or "element"), tuple(children))
    for child in value.values():
        node = _semantic_tree(child)
        if node is not None:
            return node
    return None


def _tree_distance(left: _Tree, right: _Tree) -> int:
    substitution = int(left.label != right.label)
    a, b = left.children, right.children
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, child in enumerate(a, start=1):
        dp[i][0] = dp[i - 1][0] + child.size
    for j, child in enumerate(b, start=1):
        dp[0][j] = dp[0][j - 1] + child.size
    for i, child_a in enumerate(a, start=1):
        for j, child_b in enumerate(b, start=1):
            dp[i][j] = min(
                dp[i - 1][j] + child_a.size,
                dp[i][j - 1] + child_b.size,
                dp[i - 1][j - 1] + _tree_distance(child_a, child_b),
            )
    return substitution + dp[-1][-1]


def _tree_features(tree: _Tree | None) -> tuple[Counter[str], Counter[tuple[str, str]]]:
    nodes: Counter[str] = Counter()
    edges: Counter[tuple[str, str]] = Counter()

    def visit(node: _Tree) -> None:
        nodes[node.label] += 1
        for child in node.children:
            edges[(node.label, child.label)] += 1
            visit(child)

    if tree is not None:
        visit(tree)
    return nodes, edges


def _multiset_f1(left: Counter[Any], right: Counter[Any]) -> float:
    total_left, total_right = sum(left.values()), sum(right.values())
    if not total_left and not total_right:
        return 1.0
    overlap = sum((left & right).values())
    precision = overlap / total_left if total_left else 0.0
    recall = overlap / total_right if total_right else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _reference_graph(source: str) -> set[tuple[int, int]]:
    statements = _BINDER_RE.findall(source)
    order = {name: index for index, (name, _) in enumerate(statements)}
    edges: set[tuple[int, int]] = set()
    for owner, rhs in statements:
        scrubbed = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', "", rhs)
        for reference in _IDENT_RE.findall(scrubbed):
            if reference in order:
                edges.add((order[owner], order[reference]))
    return edges


def _available(value: float, n: int = 1) -> dict[str, Any]:
    return {"value": float(value), "n": n, "status": "available", "reason": None}


def _unavailable(reason: str) -> dict[str, Any]:
    return {"value": None, "n": 0, "status": "unavailable", "reason": reason}


def _mean(metrics: Iterable[dict[str, Any]], reason: str) -> dict[str, Any]:
    values = [metric["value"] for metric in metrics if metric.get("value") is not None]
    return (
        _available(sum(values) / len(values), len(values))
        if values
        else _unavailable(reason)
    )


def _parse(source: str) -> tuple[Any | None, str | None]:
    try:
        return validate(source), None
    except (ParseError, ValueError) as exc:
        return None, str(exc)


def _structure_metrics(prediction: str, gold: str) -> dict[str, dict[str, Any]]:
    predicted, error = _parse(prediction)
    expected, gold_error = _parse(gold)
    if expected is None:
        return {"language_validity": _unavailable(f"invalid gold: {gold_error}")}
    metrics = {"language_validity": _available(float(predicted is not None))}
    if predicted is None:
        reason = f"prediction does not parse: {error}"
        for name in (
            "canonical_exact",
            "ast_node_f1",
            "ast_edge_f1",
            "tree_edit_similarity",
            "ref_graph_exact",
        ):
            metrics[name] = _unavailable(reason)
        return metrics
    backend = str((predicted.meta or {}).get("backend") or "")
    if backend == "openui-langcore":
        metrics["canonical_exact"] = _available(
            float(
                norm_text(predicted.serialized or prediction)
                == norm_text(expected.serialized or gold)
            )
        )
    else:
        metrics["canonical_exact"] = _unavailable(
            "official lang-core canonical serialization was not active"
        )
    pred_tree = _semantic_tree(predicted.root)
    gold_tree = _semantic_tree(expected.root)
    pred_nodes, pred_edges = _tree_features(pred_tree)
    gold_nodes, gold_edges = _tree_features(gold_tree)
    metrics["ast_node_f1"] = _available(_multiset_f1(pred_nodes, gold_nodes))
    metrics["ast_edge_f1"] = _available(_multiset_f1(pred_edges, gold_edges))
    if pred_tree is None or gold_tree is None:
        metrics["tree_edit_similarity"] = _unavailable("semantic AST root unavailable")
    else:
        distance = _tree_distance(pred_tree, gold_tree)
        metrics["tree_edit_similarity"] = _available(
            max(0.0, 1.0 - distance / max(pred_tree.size, gold_tree.size, 1))
        )
    metrics["ref_graph_exact"] = _available(
        float(_reference_graph(prediction) == _reference_graph(gold))
    )
    return metrics


def _output_target(value: Any, *, fallback_kind: str, fallback_category: str | None) -> OutputTarget:
    if isinstance(value, Mapping):
        return OutputTarget.from_dict(dict(value))
    return OutputTarget(str(value), fallback_kind, fallback_category)  # type: ignore[arg-type]


def score_output_targets(
    prediction: str,
    targets: Sequence[OutputTarget],
) -> dict[str, float | int | str | None]:
    """Score correctness separately from output-symbol efficiency."""
    normalized_prediction: dict[tuple[str, str | None], str] = {}
    matched: OutputTarget | None = None
    for target in targets:
        key = (target.kind, target.category)
        if key not in normalized_prediction:
            try:
                normalized_prediction[key] = validate_output(
                    prediction, target.kind, target.category
                )
            except (ParseError, ValueError, RuntimeError):
                normalized_prediction[key] = ""
        try:
            normalized_target = validate_output(target.text, target.kind, target.category)
        except (ParseError, ValueError, RuntimeError):
            continue
        if norm_text(normalized_prediction[key]) == norm_text(normalized_target):
            matched = target
            break

    correctness = float(matched is not None)
    prediction_tokens = len(lexical_tokens(prediction))
    valid_target_lengths = [
        len(lexical_tokens(target.text))
        for target in targets
        if lexical_tokens(target.text)
    ]
    minimum_tokens = min(valid_target_lengths, default=0)
    efficiency = (
        min(1.0, minimum_tokens / prediction_tokens)
        if correctness and prediction_tokens
        else 0.0
    )
    return {
        "correctness": correctness,
        "efficiency": efficiency,
        "composite": 0.8 * correctness + 0.2 * efficiency,
        "prediction_tokens": prediction_tokens,
        "minimum_target_tokens": minimum_tokens,
        "matched_kind": matched.kind if matched else None,
    }


def _numeric_evidence(evidence: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = evidence.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _available(float(value))
    if isinstance(value, bool):
        return _available(float(value))
    return _unavailable(f"prediction evidence {key!r} was not supplied")


def _equivalence_metrics(
    abstraction_level: str | int | None,
    evidence: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    level = str(abstraction_level or "").upper().replace("LEVEL", "L")
    if level not in {"L3", "L4", "L5", "3", "4", "5"}:
        return {}
    required = set(map(str, evidence.get("required_facts") or ()))
    forbidden = set(map(str, evidence.get("forbidden_facts") or ()))
    predicted = set(map(str, evidence.get("predicted_facts") or ()))
    if not required and not forbidden:
        constraint = _unavailable("prediction-side constraint facts were not supplied")
    else:
        required_score = len(required & predicted) / len(required) if required else 1.0
        forbidden_score = 1.0 - (
            len(forbidden & predicted) / len(forbidden) if forbidden else 0.0
        )
        constraint = _available((required_score + forbidden_score) / 2)
    metrics = {
        "constraint_satisfaction": constraint,
        "behavior_equivalence": _numeric_evidence(evidence, "behavior_equivalence"),
        "render_equivalence": _numeric_evidence(evidence, "render_equivalence"),
    }
    wanted = [constraint]
    normalized = "L" + level[-1]
    if normalized in {"L4", "L5"}:
        wanted.append(metrics["behavior_equivalence"])
    if normalized == "L5":
        wanted.append(metrics["render_equivalence"])
    metrics["equivalence_score"] = (
        _available(sum(metric["value"] for metric in wanted) / len(wanted))
        if all(metric["value"] is not None for metric in wanted)
        else _unavailable(f"{normalized} prediction evidence is incomplete")
    )
    return metrics


def score_case(case: Mapping[str, Any]) -> dict[str, Any]:
    prediction = str(case.get("prediction") or "")
    gold = str(case.get("gold") or "")
    task = str(case.get("task") or "unknown")
    evidence = case.get("prediction_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    target_kind = str(case.get("target_kind") or "document")
    target_category = case.get("target_category")
    target_category = None if target_category is None else str(target_category)
    targets = [OutputTarget(gold, target_kind, target_category)]  # type: ignore[arg-type]
    targets.extend(
        _output_target(
            item,
            fallback_kind=target_kind,
            fallback_category=target_category,
        )
        for item in case.get("accepted_outputs") or ()
    )
    target_score = score_output_targets(prediction, targets)
    if target_kind == "document":
        metrics = _structure_metrics(prediction, gold)
    else:
        metrics = {
            "language_validity": _available(target_score["correctness"]),
        }
    metrics.update(
        {
            "target_correctness": _available(target_score["correctness"]),
            "target_efficiency": _available(target_score["efficiency"]),
            "target_composite": _available(target_score["composite"]),
            "output_symbol_count": _available(target_score["prediction_tokens"]),
            "minimum_output_symbols": _available(
                target_score["minimum_target_tokens"]
            ),
        }
    )

    if task in {"repair", "completion", "inpaint"}:
        metrics.update(
            {
                "repair_target_match": _available(
                    float(norm_text(prediction) == norm_text(gold))
                ),
                "repair_minimality": _numeric_evidence(evidence, "minimality"),
                "repair_preservation": _numeric_evidence(evidence, "preservation"),
                "repair_localization": _numeric_evidence(evidence, "localization"),
                "valid_input_no_change": _numeric_evidence(
                    evidence, "valid_input_no_change"
                ),
            }
        )
    elif task in {"patch", "edit"}:
        metrics.update(
            {
                "apply_equals_after": _available(
                    float(norm_text(prediction) == norm_text(gold))
                ),
                "edit_minimality": _numeric_evidence(evidence, "minimality"),
                "edit_preservation": _numeric_evidence(evidence, "preservation"),
                "undo_redo_identity": _numeric_evidence(evidence, "undo_redo_identity"),
                "multi_turn_referents": _numeric_evidence(
                    evidence, "multi_turn_referents"
                ),
            }
        )
    for name in (
        "element_recall",
        "grounding_accuracy",
        "position_similarity",
        "responsive_accuracy",
        "a11y_tree_similarity",
        "state_trace_match",
        "query_mutation_tool_args",
        "conditional_visibility",
        "length_accuracy",
        "expand_contract_success",
        "constraint_interventions",
        "steps_to_first_valid",
        "action_macro_f1",
        "production_accuracy",
        "arity_accuracy",
        "production_head_accuracy",
        "arity_head_accuracy",
        "critic_ece",
        "efficiency_score",
        "node_passes",
        "active_peak",
        "phases",
    ):
        if name in evidence:
            metrics[name] = _numeric_evidence(evidence, name)
    metrics.update(_equivalence_metrics(case.get("abstraction_level"), evidence))
    return {"id": str(case.get("id") or ""), "task": task, "metrics": metrics}


def build_task_scoreboard(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [score_case(case) for case in cases]
    task_names = sorted({row["task"] for row in scored})
    tasks: dict[str, Any] = {}
    for task in task_names:
        members = [row for row in scored if row["task"] == task]
        names = sorted({name for row in members for name in row["metrics"]})
        tasks[task] = {
            "n": len(members),
            "metrics": {
                name: _mean(
                    [
                        row["metrics"].get(name, _unavailable("not eligible"))
                        for row in members
                    ],
                    "no eligible prediction evidence",
                )
                for name in names
            },
        }
    equivalence_rows = [
        row["metrics"]["equivalence_score"]
        for row in scored
        if "equivalence_score" in row["metrics"]
    ]
    unavailable = sum(
        metric["value"] is None for row in scored for metric in row["metrics"].values()
    )
    return {
        "n": len(scored),
        "complete": unavailable == 0,
        "unavailable_metric_instances": unavailable,
        "tasks": tasks,
        "equivalence": _mean(equivalence_rows, "no complete L3-L5 prediction evidence"),
        "details": scored,
    }
