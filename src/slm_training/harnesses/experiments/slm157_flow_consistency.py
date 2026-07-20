"""SLM-157 (SPV3-04): flow / consistency / trajectory-imitation wiring fixture.

Deterministic, CPU-only harness that simulates 4--8 step discrete-flow,
consistency, and trajectory-imitation arms over the existing hard-valid
tree-edit state space.  It reuses ``TreeEditSpace``, verified program patches,
and the fixture plan corpus; it trains no GPU model and makes no ship-gate
claim.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from slm_training.data.edits import (
    EditPatch,
    ProgramDocument,
    apply_patch,
    diff_programs,
)
from slm_training.data.semantic_plan.corpus import build_fixture_plan_corpus
from slm_training.data.semantic_plan.seed import PlanSeedBuilder
from slm_training.dsl.pack import get_pack
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.evals.tree_edit_scaling import _enumerate_edits
from slm_training.models.tree_edit_diffusion import (
    ACTION_STOP,
    Edit,
    Statement,
    TreeEditSpace,
    parse_statements,
    render_statements,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "FLOW_CAMPAIGN_ID",
    "PathFamily",
    "ArmFamily",
    "FlowArm",
    "CommonConfig",
    "SimulationRecord",
    "FlowReportRow",
    "FlowManifest",
    "FlowConsistencyReport",
    "build_manifest",
    "validate_manifest",
    "run_fixture_campaign",
    "render_markdown",
]

MATRIX_VERSION = "spv3-04-v1"
MATRIX_SET = "slm157_flow_consistency"
FLOW_CAMPAIGN_ID = "slm157-flow-consistency"

_DEFAULT_INVENTORY = [":content.body", ":label.text", ":placeholder.hint", ":title.text"]
_FLOW_TEMPERATURE = 1.0


class PathFamily(str, Enum):
    """Reference path shape used by an arm."""

    P_short = "P_short"
    P_x22 = "P_x22"
    P_coarse = "P_coarse"
    P_capsule = "P_capsule"
    P_random = "P_random"


class ArmFamily(str, Enum):
    """Arm family for SPV3-04."""

    teacher_long_x22 = "teacher_long_x22"
    direct_trajectory_imitation = "direct_trajectory_imitation"
    consistency_student_x22 = "consistency_student_x22"
    consistency_student_coarse = "consistency_student_coarse"
    discrete_flow_rate = "discrete_flow_rate"
    random_path_control = "random_path_control"
    ar_x22_hybrid_placeholder = "ar_x22_hybrid_placeholder"
    oracle_boundary = "oracle_boundary"


@dataclass(frozen=True)
class CommonConfig:
    """Frozen orthogonal controls shared by every arm."""

    seeds: tuple[int, ...] = (0, 1, 2)
    n_records: int = 8
    steps_list: tuple[int, ...] = (4, 8)
    path_families: tuple[PathFamily, ...] = (
        PathFamily.P_short,
        PathFamily.P_x22,
        PathFamily.P_coarse,
        PathFamily.P_capsule,
        PathFamily.P_random,
    )
    max_path_len: int = 24
    metric_versions: dict[str, str] = field(
        default_factory=lambda: {"meaningful": "2.0.0"}
    )
    root_containers: tuple[str, ...] = ("Stack", "Card")
    leaf_components: tuple[str, ...] = ("TextContent", "Button", "Input", "CardHeader")

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["seeds"] = list(self.seeds)
        data["steps_list"] = list(self.steps_list)
        data["path_families"] = [p.value for p in self.path_families]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommonConfig":
        return cls(
            seeds=tuple(data.get("seeds", [0, 1, 2])),
            n_records=data.get("n_records", 8),
            steps_list=tuple(data.get("steps_list", [4, 8])),
            path_families=tuple(
                PathFamily(p) for p in data.get("path_families", ["P_short"])
            ),
            max_path_len=data.get("max_path_len", 24),
            metric_versions=data.get("metric_versions", {"meaningful": "2.0.0"}),
            root_containers=tuple(data.get("root_containers", ["Stack", "Card"])),
            leaf_components=tuple(
                data.get("leaf_components", ["TextContent", "Button", "Input", "CardHeader"])
            ),
        )


@dataclass(frozen=True)
class FlowArm:
    """One arm in the SLM-157 flow-consistency campaign."""

    arm_id: str
    family: ArmFamily
    path_family: PathFamily
    name: str
    description: str
    promotable: bool = False
    diagnostic: bool = False
    blocked: bool = False
    blocker: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["family"] = self.family.value
        data["path_family"] = self.path_family.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowArm":
        return cls(
            arm_id=data["arm_id"],
            family=ArmFamily(data.get("family", "teacher_long_x22")),
            path_family=PathFamily(data.get("path_family", "P_short")),
            name=data.get("name", ""),
            description=data.get("description", ""),
            promotable=data.get("promotable", False),
            diagnostic=data.get("diagnostic", False),
            blocked=data.get("blocked", False),
            blocker=data.get("blocker", ""),
        )


@dataclass(frozen=True)
class SimulationRecord:
    """Per-record simulation outcome."""

    record_index: int
    source: str
    target: str
    final_state: str
    steps_budget: int
    path_family: str
    states_visited: int
    edits_applied: int
    rollbacks: int
    forwards: int
    verifier_calls: int
    reached_target: bool
    accepted: bool
    boundary_accuracy: float
    trajectory_consistency: float
    transition_entropy: float
    path_length: int
    remaining_distance: int
    detour_ratio: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationRecord":
        return cls(
            record_index=data["record_index"],
            source=data["source"],
            target=data["target"],
            final_state=data["final_state"],
            steps_budget=data["steps_budget"],
            path_family=data["path_family"],
            states_visited=data["states_visited"],
            edits_applied=data["edits_applied"],
            rollbacks=data["rollbacks"],
            forwards=data["forwards"],
            verifier_calls=data["verifier_calls"],
            reached_target=data["reached_target"],
            accepted=data["accepted"],
            boundary_accuracy=data["boundary_accuracy"],
            trajectory_consistency=data["trajectory_consistency"],
            transition_entropy=data["transition_entropy"],
            path_length=data["path_length"],
            remaining_distance=data["remaining_distance"],
            detour_ratio=data["detour_ratio"],
            notes=list(data.get("notes", [])),
        )


@dataclass(frozen=True)
class FlowReportRow:
    """Aggregated result for one arm / seed / steps budget."""

    arm_id: str
    family: ArmFamily
    path_family: PathFamily
    seed: int
    steps: int
    promotable: bool
    diagnostic: bool
    n_records: int
    state_validity: float
    transition_validity: float
    target_reach_rate: float
    accepted_reach_rate: float
    boundary_accuracy: float
    mean_path_length: float
    mean_remaining_distance: float
    mean_detour_ratio: float
    mean_trajectory_consistency: float
    mean_transition_entropy: float
    rollback_rate: float
    mean_forwards: float
    mean_edits_applied: float
    mean_verifier_calls: float
    records: list[SimulationRecord]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["family"] = self.family.value
        data["path_family"] = self.path_family.value
        data["records"] = [r.to_dict() for r in self.records]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowReportRow":
        return cls(
            arm_id=data["arm_id"],
            family=ArmFamily(data.get("family", "teacher_long_x22")),
            path_family=PathFamily(data.get("path_family", "P_short")),
            seed=data["seed"],
            steps=data["steps"],
            promotable=data.get("promotable", False),
            diagnostic=data.get("diagnostic", False),
            n_records=data["n_records"],
            state_validity=data["state_validity"],
            transition_validity=data["transition_validity"],
            target_reach_rate=data["target_reach_rate"],
            accepted_reach_rate=data["accepted_reach_rate"],
            boundary_accuracy=data["boundary_accuracy"],
            mean_path_length=data["mean_path_length"],
            mean_remaining_distance=data["mean_remaining_distance"],
            mean_detour_ratio=data["mean_detour_ratio"],
            mean_trajectory_consistency=data["mean_trajectory_consistency"],
            mean_transition_entropy=data["mean_transition_entropy"],
            rollback_rate=data["rollback_rate"],
            mean_forwards=data["mean_forwards"],
            mean_edits_applied=data["mean_edits_applied"],
            mean_verifier_calls=data["mean_verifier_calls"],
            records=[SimulationRecord.from_dict(r) for r in data.get("records", [])],
            notes=list(data.get("notes", [])),
        )


@dataclass(frozen=True)
class FlowManifest:
    """Preregistered manifest for the SLM-157 campaign."""

    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = FLOW_CAMPAIGN_ID
    hypothesis: str = (
        "Discrete-flow, consistency, and trajectory-imitation policies can be "
        "simulated over the hard-valid tree-edit state space using only the "
        "existing compiler and verified patch surfaces, before any learned scorer "
        "is trained."
    )
    falsifier: str = (
        "The existing legal-edit enumeration cannot produce non-trivial paths "
        "between distinct valid programs, the consistency boundary signal is "
        "indistinguishable from the greedy distance signal, or every synthetic "
        "arm collapses to the same trivial trajectory."
    )
    common_config: CommonConfig = field(default_factory=CommonConfig)
    arms: tuple[FlowArm, ...] = ()
    claim_class: str = "wiring"
    status: str = "not_run"

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["common_config"] = self.common_config.to_dict()
        data["arms"] = [arm.to_dict() for arm in self.arms]
        return data

    def to_json(self, path: Path) -> None:
        path.write_text(
            _json_dumps(self.to_dict()),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowManifest":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", FLOW_CAMPAIGN_ID),
            hypothesis=data.get("hypothesis", ""),
            falsifier=data.get("falsifier", ""),
            common_config=CommonConfig.from_dict(data.get("common_config", {})),
            arms=tuple(FlowArm.from_dict(a) for a in data.get("arms", [])),
            claim_class=data.get("claim_class", "wiring"),
            status=data.get("status", "not_run"),
        )


@dataclass(frozen=True)
class FlowConsistencyReport:
    """Full fixture report for SLM-157."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    manifest: FlowManifest
    rows: list[FlowReportRow]
    version_stamp: dict[str, Any] = field(default_factory=dict)
    claim_class: str = "wiring"

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "manifest": self.manifest.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(_json_dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowConsistencyReport":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", FLOW_CAMPAIGN_ID),
            run_id=data.get("run_id", "slm157_fixture"),
            status=data.get("status", "fixture"),
            manifest=FlowManifest.from_dict(data.get("manifest", {})),
            rows=[FlowReportRow.from_dict(r) for r in data.get("rows", [])],
            version_stamp=data.get("version_stamp", {}),
            claim_class=data.get("claim_class", "wiring"),
        )


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def _inventory_for(source: str, target: str) -> list[str]:
    placeholders = set(extract_placeholders(source)) | set(extract_placeholders(target))
    inventory = list(dict.fromkeys([*list(placeholders), *_DEFAULT_INVENTORY]))
    return [p if p.startswith(":") else f":{p}" for p in inventory]


def _edit_distance(before: str, after: str) -> int:
    try:
        return diff_programs(before, after).ast_operation_count
    except ValueError:
        # diff_programs can fail on intermediate patch states that contain
        # unreachable statements; treat them as maximally distant for policy
        # scoring without aborting the fixture run.
        return 999


def _coarse_distance(before: str, after: str) -> int:
    """Component-expression mismatches for statements that exist in both programs."""
    doc_before = ProgramDocument.from_openui(before)
    doc_after = ProgramDocument.from_openui(after)
    after_by_name = {s.name: s.expression for s in doc_after.statements}
    mismatches = 0
    for statement in doc_before.statements:
        if statement.name in after_by_name and after_by_name[statement.name] != statement.expression:
            mismatches += 1
    return mismatches


def _build_path_short(source: str, target: str) -> list[str]:
    patch = diff_programs(source, target)
    path: list[str] = [source]
    state = source
    for operation in patch.operations:
        if operation.kind.value == "noop":
            continue
        state = apply_patch(
            state,
            EditPatch((operation,), collect_unreachable=False),
        )
        path.append(state)
    if ProgramDocument.from_openui(path[-1]).to_openui() != ProgramDocument.from_openui(target).to_openui():
        path.append(target)
    return path


def _build_path_x22(
    source: str,
    target: str,
    space: TreeEditSpace,
    inventory: list[str],
    max_len: int,
) -> list[str]:
    path: list[str] = [source]
    state_text = source
    for _ in range(max_len):
        if state_text == target:
            break
        statements = parse_statements(state_text)
        if statements is None:
            break
        candidates = _enumerate_edits(statements, inventory, space)
        best: tuple[int, int, int, int, list[Statement]] | None = None
        current_dist = _edit_distance(state_text, target)
        for edit, child in candidates:
            if child is None or edit.action == ACTION_STOP:
                continue
            child_text = render_statements(child)
            dist = _edit_distance(child_text, target)
            key = (dist, edit.action, edit.stmt, edit.comp, edit.slot)
            if best is None or key < best[:5]:
                best = (dist, edit.action, edit.stmt, edit.comp, edit.slot, child)
        if best is None or best[0] >= current_dist:
            break
        state_text = render_statements(best[5])
        path.append(state_text)
    return path


def _build_path_coarse(
    source: str,
    target: str,
    space: TreeEditSpace,
    inventory: list[str],
    max_len: int,
) -> list[str]:
    path: list[str] = [source]
    state_text = source
    for _ in range(max_len):
        if state_text == target:
            break
        statements = parse_statements(state_text)
        if statements is None:
            break
        candidates = _enumerate_edits(statements, inventory, space)
        best: tuple[int, int, int, int, int, list[Statement]] | None = None
        current_coarse = _coarse_distance(state_text, target)
        current_full = _edit_distance(state_text, target)
        for edit, child in candidates:
            if child is None or edit.action == ACTION_STOP:
                continue
            child_text = render_statements(child)
            coarse = _coarse_distance(child_text, target)
            full = _edit_distance(child_text, target)
            # Prefer coarse progress; break ties by full distance, then determinism.
            key = (coarse, full, edit.action, edit.stmt, edit.comp, edit.slot)
            if best is None or key < best[:6]:
                best = (coarse, full, edit.action, edit.stmt, edit.comp, edit.slot, child)
        if best is None or (best[0] >= current_coarse and best[1] >= current_full):
            break
        state_text = render_statements(best[6])
        path.append(state_text)
    return path


def _build_path_capsule(
    source: str,
    target: str,
    space: TreeEditSpace,
    inventory: list[str],
    rng: random.Random,
    max_len: int,
) -> list[str]:
    statements = parse_statements(source)
    if statements is None:
        return _build_path_short(source, target)
    mutated = space.sample_mutation(statements, inventory, rng)
    if mutated is None:
        return _build_path_short(source, target)
    waypoint = render_statements(mutated[0])
    tail = _build_path_short(waypoint, target)
    path = [source, *tail]
    # Keep the capsule within the budget by truncating the detour if needed.
    if len(path) > max_len + 1:
        path = path[: max_len + 1]
    return path


def _build_path_random(
    source: str,
    target: str,
    space: TreeEditSpace,
    inventory: list[str],
    rng: random.Random,
    max_len: int,
) -> list[str]:
    path: list[str] = [source]
    state_text = source
    for _ in range(max_len):
        if state_text == target:
            break
        statements = parse_statements(state_text)
        if statements is None:
            break
        mutated = space.sample_mutation(statements, inventory, rng)
        if mutated is None:
            break
        state_text = render_statements(mutated[0])
        path.append(state_text)
    return path


def _build_reference_path(
    family: PathFamily,
    source: str,
    target: str,
    space: TreeEditSpace,
    inventory: list[str],
    rng: random.Random,
    max_len: int,
) -> list[str]:
    if family is PathFamily.P_short:
        return _build_path_short(source, target)
    if family is PathFamily.P_x22:
        return _build_path_x22(source, target, space, inventory, max_len)
    if family is PathFamily.P_coarse:
        return _build_path_coarse(source, target, space, inventory, max_len)
    if family is PathFamily.P_capsule:
        return _build_path_capsule(source, target, space, inventory, rng, max_len)
    if family is PathFamily.P_random:
        return _build_path_random(source, target, space, inventory, rng, max_len)
    return [source, target]


def _entropy(probs: dict[int, float]) -> float:
    return -sum(
        p * math.log(p) for p in probs.values() if p > 0.0
    )


def _select_candidate(
    arm: FlowArm,
    current_text: str,
    target_text: str,
    reference_path: list[str],
    candidates: list[tuple[Edit, list[Statement] | None]],
    rng: random.Random,
) -> tuple[tuple[Edit, list[Statement]], dict[int, float], bool]:
    """Return (chosen candidate, probability distribution over candidate indices, boundary prediction)."""
    current_dist = _edit_distance(current_text, target_text)
    valid_indices = [
        i for i, (_, child) in enumerate(candidates) if child is not None
    ]
    if not valid_indices:
        raise RuntimeError("no valid candidates; state space should never be empty")

    stop_index = next(
        (i for i, (edit, _) in enumerate(candidates) if edit.action == ACTION_STOP),
        valid_indices[0],
    )

    def _child_text(i: int) -> str:
        edit, child = candidates[i]
        if edit.action == ACTION_STOP:
            return current_text
        return render_statements(child)  # type: ignore[arg-type]

    def _dist(i: int) -> int:
        return _edit_distance(_child_text(i), target_text)

    def _coarse(i: int) -> int:
        return _coarse_distance(_child_text(i), target_text)

    probs: dict[int, float] = {}

    if arm.family in {
        ArmFamily.teacher_long_x22,
        ArmFamily.consistency_student_x22,
    }:
        # Greedy distance-reducing / boundary match.
        best = min(valid_indices, key=lambda i: (_dist(i), candidates[i][0].action, candidates[i][0].stmt, candidates[i][0].comp, candidates[i][0].slot))
        probs[best] = 1.0
        predicted_boundary = _dist(best) == 0 and candidates[best][0].action == ACTION_STOP
        return candidates[best], probs, predicted_boundary

    if arm.family is ArmFamily.consistency_student_coarse:
        best = min(
            valid_indices,
            key=lambda i: (
                _coarse(i),
                _dist(i),
                candidates[i][0].action,
                candidates[i][0].stmt,
                candidates[i][0].comp,
                candidates[i][0].slot,
            ),
        )
        probs[best] = 1.0
        predicted_boundary = _coarse(best) == 0 and candidates[best][0].action == ACTION_STOP
        return candidates[best], probs, predicted_boundary

    if arm.family is ArmFamily.direct_trajectory_imitation:
        desired_index = min(len(reference_path) - 1, 1)
        desired_text = reference_path[desired_index]
        # Prefer the candidate that lands exactly on the next reference state;
        # otherwise move toward the desired state.
        exact = [i for i in valid_indices if _child_text(i) == desired_text]
        if exact:
            best = exact[0]
        else:
            best = min(
                valid_indices,
                key=lambda i: (
                    _edit_distance(_child_text(i), desired_text),
                    candidates[i][0].action,
                    candidates[i][0].stmt,
                    candidates[i][0].comp,
                    candidates[i][0].slot,
                ),
            )
        probs[best] = 1.0
        predicted_boundary = candidates[best][0].action == ACTION_STOP
        return candidates[best], probs, predicted_boundary

    if arm.family is ArmFamily.oracle_boundary:
        if current_text == target_text:
            probs[stop_index] = 1.0
            return candidates[stop_index], probs, True
        # Step to the next state on the short path if it is a legal neighbor.
        path_set = set(reference_path)
        next_states = [
            i
            for i in valid_indices
            if _child_text(i) in path_set and _dist(i) < current_dist
        ]
        if next_states:
            best = next_states[0]
        else:
            best = min(
                valid_indices,
                key=lambda i: (_dist(i), candidates[i][0].action, candidates[i][0].stmt, candidates[i][0].comp, candidates[i][0].slot),
            )
        probs[best] = 1.0
        predicted_boundary = candidates[best][0].action == ACTION_STOP
        return candidates[best], probs, predicted_boundary

    if arm.family is ArmFamily.discrete_flow_rate:
        # Softmax over negative distance (flow-rate policy).
        non_stop = [i for i in valid_indices if candidates[i][0].action != ACTION_STOP]
        active = non_stop or valid_indices
        scores = {}
        for i in active:
            d = _dist(i)
            scores[i] = math.exp(-d / _FLOW_TEMPERATURE)
        total = sum(scores.values())
        if total == 0.0 or total != total:
            uniform = 1.0 / len(active)
            probs = {i: uniform for i in active}
        else:
            probs = {i: scores[i] / total for i in active}
        chosen = rng.choices(list(probs.keys()), weights=list(probs.values()))[0]
        predicted_boundary = candidates[chosen][0].action == ACTION_STOP
        return candidates[chosen], probs, predicted_boundary

    if arm.family is ArmFamily.random_path_control:
        non_stop = [i for i in valid_indices if candidates[i][0].action != ACTION_STOP]
        if current_text == target_text or not non_stop:
            probs[stop_index] = 1.0
            return candidates[stop_index], probs, True
        uniform = 1.0 / len(non_stop)
        probs = {i: uniform for i in non_stop}
        chosen = rng.choice(non_stop)
        predicted_boundary = False
        return candidates[chosen], probs, predicted_boundary

    if arm.family is ArmFamily.ar_x22_hybrid_placeholder:
        # Placeholder: mix greedy X22 step with a random exploration step.
        non_stop = [i for i in valid_indices if candidates[i][0].action != ACTION_STOP]
        if rng.random() < 0.5 and non_stop:
            chosen = rng.choice(non_stop)
        else:
            chosen = min(
                valid_indices,
                key=lambda i: (_dist(i), candidates[i][0].action, candidates[i][0].stmt, candidates[i][0].comp, candidates[i][0].slot),
            )
        probs[chosen] = 1.0
        predicted_boundary = candidates[chosen][0].action == ACTION_STOP
        return candidates[chosen], probs, predicted_boundary

    # Fallback: greedy.
    best = min(valid_indices, key=lambda i: (_dist(i), candidates[i][0].action, candidates[i][0].stmt, candidates[i][0].comp, candidates[i][0].slot))
    probs[best] = 1.0
    predicted_boundary = candidates[best][0].action == ACTION_STOP
    return candidates[best], probs, predicted_boundary


def _simulate_record(
    arm: FlowArm,
    record_index: int,
    source: str,
    target: str,
    reference_path: list[str],
    steps_budget: int,
    space: TreeEditSpace,
    inventory: list[str],
    rng: random.Random,
) -> SimulationRecord:
    state_text = source
    edits_applied = 0
    rollbacks = 0
    forwards_total = 0
    verifier_calls = 0
    boundary_correct = 0
    boundary_total = 0
    consistency_scores: list[float] = []
    entropies: list[float] = []
    states_visited = 1
    notes: list[str] = []

    for step in range(steps_budget):
        statements = parse_statements(state_text)
        if statements is None:
            notes.append(f"step {step}: parser failed")
            break
        candidates = _enumerate_edits(statements, inventory, space)
        forwards_total += len(candidates)
        verifier_calls += len(candidates)

        chosen, probs, predicted_boundary = _select_candidate(
            arm, state_text, target, reference_path, candidates, rng
        )
        edit, child = chosen
        if child is None:
            rollbacks += 1
            notes.append(f"step {step}: selected invalid edit")
            continue
        child_text = render_statements(child) if edit.action != ACTION_STOP else state_text

        # Hard-validity check through TreeEditSpace.
        if space.apply(statements, edit, inventory) is None:
            rollbacks += 1
            notes.append(f"step {step}: TreeEditSpace rejected selected edit")
            continue

        actual_boundary = child_text == target
        boundary_total += 1
        if predicted_boundary == actual_boundary:
            boundary_correct += 1

        entropies.append(_entropy(probs))

        if edit.action == ACTION_STOP:
            break

        prev_dist = _edit_distance(state_text, target)
        next_dist = _edit_distance(child_text, target)
        consistency_scores.append(1.0 if next_dist <= prev_dist else 0.0)

        state_text = child_text
        edits_applied += 1
        states_visited += 1

        # Advance the reference path pointer for direct imitation.
        if arm.family is ArmFamily.direct_trajectory_imitation and len(reference_path) > 1:
            reference_path = reference_path[1:]

    final_text = state_text
    reached_target = final_text == target
    accepted = reached_target and rollbacks == 0
    shortest_distance = max(1, _edit_distance(source, target))
    path_length = edits_applied
    remaining_distance = _edit_distance(final_text, target)
    detour_ratio = path_length / shortest_distance

    return SimulationRecord(
        record_index=record_index,
        source=source,
        target=target,
        final_state=final_text,
        steps_budget=steps_budget,
        path_family=arm.path_family.value,
        states_visited=states_visited,
        edits_applied=edits_applied,
        rollbacks=rollbacks,
        forwards=forwards_total,
        verifier_calls=verifier_calls,
        reached_target=reached_target,
        accepted=accepted,
        boundary_accuracy=boundary_correct / max(1, boundary_total),
        trajectory_consistency=sum(consistency_scores) / max(1, len(consistency_scores)),
        transition_entropy=sum(entropies) / max(1, len(entropies)),
        path_length=path_length,
        remaining_distance=remaining_distance,
        detour_ratio=detour_ratio,
        notes=notes or ["ok"],
    )


def _generate_programs(cfg: CommonConfig, seed: int) -> list[str]:
    """Return a deterministic list of valid OpenUI programs."""
    pack = get_pack("openui")
    builder = PlanSeedBuilder(pack)
    corpus = build_fixture_plan_corpus(
        count=max(cfg.n_records * 4, 32),
        seed=seed,
        root_containers=list(cfg.root_containers),
        leaf_components=list(cfg.leaf_components),
    )
    programs: list[str] = []
    for spec, plan in corpus["train"]:
        result = builder.build(plan)
        if result.ok and result.seed is not None:
            programs.append(result.seed)
        if len(programs) >= cfg.n_records * 2:
            break
    return programs


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def build_manifest() -> FlowManifest:
    """Return the default SLM-157 fixture manifest."""
    arms = (
        FlowArm(
            arm_id="A_teacher_long_x22",
            family=ArmFamily.teacher_long_x22,
            path_family=PathFamily.P_x22,
            name="teacher_long_x22",
            description=(
                "Long-horizon X22 teacher: greedy distance-reducing walk over the "
                "full legal-edit enumeration."
            ),
        ),
        FlowArm(
            arm_id="B_direct_trajectory_imitation",
            family=ArmFamily.direct_trajectory_imitation,
            path_family=PathFamily.P_short,
            name="direct_trajectory_imitation",
            description=(
                "Direct trajectory imitation: follow the short verified patch path "
                "from source to target."
            ),
        ),
        FlowArm(
            arm_id="C_consistency_student_x22",
            family=ArmFamily.consistency_student_x22,
            path_family=PathFamily.P_x22,
            name="consistency_student_x22",
            description=(
                "Consistency student trained on the X22 path family: boundary-state "
                "match using the full edit distance."
            ),
        ),
        FlowArm(
            arm_id="D_consistency_student_coarse",
            family=ArmFamily.consistency_student_coarse,
            path_family=PathFamily.P_coarse,
            name="consistency_student_coarse",
            description=(
                "Consistency student trained on the coarse path family: boundary-state "
                "match using a coarse component-expression distance."
            ),
        ),
        FlowArm(
            arm_id="E_discrete_flow_rate",
            family=ArmFamily.discrete_flow_rate,
            path_family=PathFamily.P_capsule,
            name="discrete_flow_rate",
            description=(
                "Discrete flow-rate policy: softmax over negative remaining distance "
                "on a capsule-shaped reference path."
            ),
        ),
        FlowArm(
            arm_id="F_random_path_control",
            family=ArmFamily.random_path_control,
            path_family=PathFamily.P_random,
            name="random_path_control",
            description=(
                "Random-path control: uniform random legal edits; sanity-check "
                "baseline for reach rates."
            ),
        ),
        FlowArm(
            arm_id="G_ar_x22_hybrid_placeholder",
            family=ArmFamily.ar_x22_hybrid_placeholder,
            path_family=PathFamily.P_x22,
            name="ar_x22_hybrid_placeholder",
            description=(
                "AR/X22 hybrid placeholder: policy wiring only, not a trained "
                "autoregressive scorer."
            ),
        ),
        FlowArm(
            arm_id="H_oracle_boundary",
            family=ArmFamily.oracle_boundary,
            path_family=PathFamily.P_short,
            name="oracle_boundary",
            description=(
                "Oracle boundary diagnostic: perfect boundary prediction and short-path "
                "navigation; upper-bound sanity check."
            ),
            diagnostic=True,
        ),
    )
    return FlowManifest(arms=arms)


def validate_manifest(manifest: FlowManifest) -> list[str]:
    """Validate manifest shape and honest constraints."""
    errors: list[str] = []
    if not manifest.arms:
        errors.append("arms must not be empty")
    seen: set[str] = set()
    for arm in manifest.arms:
        if arm.arm_id in seen:
            errors.append(f"duplicate arm_id: {arm.arm_id}")
        seen.add(arm.arm_id)
        if arm.blocked and arm.promotable:
            errors.append(f"{arm.arm_id}: blocked arm must be non-promotable")
        if arm.diagnostic and arm.promotable:
            errors.append(f"{arm.arm_id}: diagnostic arm must be non-promotable")
    cfg = manifest.common_config
    if cfg.n_records <= 0:
        errors.append("common_config.n_records must be positive")
    if not cfg.steps_list:
        errors.append("common_config.steps_list must not be empty")
    if any(s <= 0 for s in cfg.steps_list):
        errors.append("common_config.steps_list values must be positive")
    return errors


def run_fixture_campaign(
    manifest: FlowManifest | None = None,
    *,
    run_id: str = "slm157_fixture",
    output_dir: Path | None = None,
) -> FlowConsistencyReport:
    """Run the SLM-157 flow-consistency fixture campaign."""
    manifest = manifest or build_manifest()
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))

    cfg = manifest.common_config
    space = TreeEditSpace()
    rows: list[FlowReportRow] = []

    for seed in cfg.seeds:
        programs = _generate_programs(cfg, seed)
        if len(programs) < cfg.n_records * 2:
            raise RuntimeError(
                f"seed {seed}: only {len(programs)} valid programs generated, "
                f"need {cfg.n_records * 2}"
            )
        pairs = [
            (programs[2 * i], programs[2 * i + 1])
            for i in range(cfg.n_records)
        ]

        for arm in manifest.arms:
            if arm.blocked:
                for steps in cfg.steps_list:
                    rows.append(
                        FlowReportRow(
                            arm_id=arm.arm_id,
                            family=arm.family,
                            path_family=arm.path_family,
                            seed=seed,
                            steps=steps,
                            promotable=False,
                            diagnostic=arm.diagnostic,
                            n_records=0,
                            state_validity=1.0,
                            transition_validity=1.0,
                            target_reach_rate=0.0,
                            accepted_reach_rate=0.0,
                            boundary_accuracy=0.0,
                            mean_path_length=0.0,
                            mean_remaining_distance=0.0,
                            mean_detour_ratio=0.0,
                            mean_trajectory_consistency=0.0,
                            mean_transition_entropy=0.0,
                            rollback_rate=0.0,
                            mean_forwards=0.0,
                            mean_edits_applied=0.0,
                            mean_verifier_calls=0.0,
                            records=[],
                            notes=["blocked", arm.blocker],
                        )
                    )
                continue

            for steps in cfg.steps_list:
                records: list[SimulationRecord] = []
                for record_index, (source, target) in enumerate(pairs):
                    record_rng = random.Random(_arm_seed(arm.arm_id, seed, steps, record_index))
                    inventory = _inventory_for(source, target)
                    reference_path = _build_reference_path(
                        arm.path_family, source, target, space, inventory, record_rng, cfg.max_path_len
                    )
                    record = _simulate_record(
                        arm,
                        record_index,
                        source,
                        target,
                        reference_path,
                        steps,
                        space,
                        inventory,
                        record_rng,
                    )
                    records.append(record)

                rows.append(_aggregate_records(arm, seed, steps, records))

    report = FlowConsistencyReport(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=FLOW_CAMPAIGN_ID,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        rows=rows,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm157_flow_consistency",
        ),
        claim_class="wiring",
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm157_flow_consistency_report.json")
    return report


def _arm_seed(arm_id: str, seed: int, steps: int, record_index: int | None = None) -> int:
    base = hash((arm_id, seed, steps))
    if record_index is not None:
        base = hash((base, record_index))
    return base % (2**31)


def _aggregate_records(
    arm: FlowArm, seed: int, steps: int, records: list[SimulationRecord]
) -> FlowReportRow:
    n = len(records)
    return FlowReportRow(
        arm_id=arm.arm_id,
        family=arm.family,
        path_family=arm.path_family,
        seed=seed,
        steps=steps,
        promotable=arm.promotable and not arm.blocked,
        diagnostic=arm.diagnostic,
        n_records=n,
        state_validity=1.0,
        transition_validity=1.0,
        target_reach_rate=sum(1 for r in records if r.reached_target) / max(1, n),
        accepted_reach_rate=sum(1 for r in records if r.accepted) / max(1, n),
        boundary_accuracy=_mean([r.boundary_accuracy for r in records]),
        mean_path_length=_mean([float(r.path_length) for r in records]),
        mean_remaining_distance=_mean([float(r.remaining_distance) for r in records]),
        mean_detour_ratio=_mean([r.detour_ratio for r in records]),
        mean_trajectory_consistency=_mean([r.trajectory_consistency for r in records]),
        mean_transition_entropy=_mean([r.transition_entropy for r in records]),
        rollback_rate=_mean([float(r.rollbacks) for r in records]),
        mean_forwards=_mean([float(r.forwards) for r in records]),
        mean_edits_applied=_mean([float(r.edits_applied) for r in records]),
        mean_verifier_calls=_mean([float(r.verifier_calls) for r in records]),
        records=records,
        notes=[
            f"family={arm.family.value}",
            f"path_family={arm.path_family.value}",
            "fixture-only: synthetic scoring over hard-valid states",
        ],
    )


def render_markdown(report: FlowConsistencyReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-157 (SPV3-04): Flow / consistency / trajectory-imitation fixture ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`",
        "",
        f"Version: `{report.matrix_version}`",
        "",
        f"Status: **{report.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no production "
        "TwoTower wiring was touched, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        report.manifest.hypothesis,
        "",
        "## Falsifier",
        "",
        report.manifest.falsifier,
        "",
        "## Arms",
        "",
        "| Arm | Family | Path family | Promotable | Diagnostic | Description |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in report.manifest.arms:
        lines.append(
            f"| {arm.arm_id} | {arm.family.value} | {arm.path_family.value} | "
            f"{arm.promotable} | {arm.diagnostic} | {arm.description} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Arm | Seed | Steps | Records | Target reach | Accepted reach | "
            "Boundary acc | Path len | Remaining dist | Detour | Consistency | Entropy | Rollbacks |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        lines.append(
            f"| {row.arm_id} | {row.seed} | {row.steps} | {row.n_records} | "
            f"{row.target_reach_rate:.3f} | {row.accepted_reach_rate:.3f} | "
            f"{row.boundary_accuracy:.3f} | {row.mean_path_length:.2f} | "
            f"{row.mean_remaining_distance:.2f} | {row.mean_detour_ratio:.2f} | "
            f"{row.mean_trajectory_consistency:.3f} | {row.mean_transition_entropy:.3f} | "
            f"{row.rollback_rate:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** Every arm is explicitly non-promotable. The "
            "harness proves the wiring and metrics plumbing over synthetic, hard-valid "
            "trajectories, but it does not train or evaluate a real model. The "
            "mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_model`` "
            "until a trained scorer and AgentV evaluation are available.",
            "",
            "## Honest caveats",
            "",
            "- The source/target pairs come from a small fixture plan corpus, not a "
            "  production train/eval split.",
            "- Distance and reach metrics use the existing statement-level patch "
            "  distance, not a rendering or user-judgment proxy.",
            "- Boundary prediction is synthetic: STOP is treated as a boundary "
            "  prediction, and accuracy is measured against the known target.",
            "- Rollbacks are recorded when the selected edit is invalid, but the "
            "  legal-edit enumeration filters invalid candidates before selection.",
            "- No Pareto or ship-gate claim is made; this is wiring evidence only.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm157_flow_consistency_fixture --mode plan-only",
            "python -m scripts.run_slm157_flow_consistency_fixture --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
