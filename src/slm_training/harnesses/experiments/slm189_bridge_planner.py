"""SLM-189 (FFE2-01): bridge planner protocols and deterministic engine —
wiring/fixture harness.

This harness exercises the SLM-189 bridge planner over a small deterministic
fixture of OpenUI source/target pairs and source policies.  No model is trained,
no GPU is required, and no ship-gate claim is made.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.data.flow.bridge_planner import (
    REACHED,
    UNKNOWN_BUDGET,
    BridgePlannerResultV1,
    plan_bridge,
)
from slm_training.dsl.canonicalize import canonical_fingerprint
from slm_training.harnesses.experiments.slm188_edit_algebra import (
    apply_canonical_edit,
    build_sketch_seed,
    plan_edit_sequence,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_VERSION",
    "MATRIX_SET",
    "EXPERIMENT_ID",
    "ARM_NAMES",
    "BridgePlannerCase",
    "BridgePlannerArmSummary",
    "BridgePlannerManifest",
    "build_exact_fixture_targets",
    "build_synthetic_scale_targets",
    "run_bridge_planner_fixture",
    "render_markdown",
    "validate_manifest",
]

MATRIX_VERSION = "ffe2-01-v1"
MATRIX_SET = "slm189_bridge_planner"
EXPERIMENT_ID = "slm189-bridge-planner"

ARM_NAMES = (
    "canonical_greedy",
    "exact_shortest",
    "random_shortest",
    "dependency_dag",
    "contract_first",
    "source_adaptive",
    "solver_guided",
)

_HYPOTHESIS = (
    "A deterministic bridge planner can produce replay-valid canonical edit sequences "
    "from structural sketch seeds to fixture OpenUI targets within a small edit budget, "
    "and independent-edit permutations that respect the dependency DAG preserve "
    "reachability and transition validity."
)

_FALSIFIER = (
    "The canonical greedy arm fails to reach a supported fixture target, or a "
    "dependency-respecting permutation of the greedy edits fails to replay to the "
    "target, or the exact-shortest arm disagrees with the greedy arm on tiny cases "
    "where it should be feasible, or any arm reports certificate failure for a "
    "reachable target."
)

_HONEST_CAVEATS = (
    "Fixture-only wiring evidence: no trained model, checkpoint, or GPU run is involved.",
    "exact_budget is intentionally small (8) so the harness stays CPU-only; production "
    "bridge search needs a larger solver budget.",
    "The production corpus is not exercised; targets are hand-written or deterministically "
    "generated small OpenUI programs.",
    "solver_guided, contract_first, and source_adaptive arms are documented but not "
    "implemented in this wiring fixture; they return UNKNOWN_BUDGET.",
    "Randomization in random_shortest is over topological orders of the dependency DAG "
    "only, not over the full edit search space.",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clamp(value: float, low: float = 0.0, high: float = float("inf")) -> float:
    return max(low, min(value, high))


@dataclass(frozen=True)
class BridgePlannerCase:
    """One source-policy × target × arm bridge attempt."""

    case_id: str
    source_seed_id: str
    target_id: str
    target_program: str
    arm: str
    status: str
    path_length: int
    nodes_expanded: int
    max_frontier: int
    wall_seconds: float
    replay_ok: bool
    cost_attribution: dict[str, float]
    scaling_features: dict[str, float]
    plan_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_seed_id": self.source_seed_id,
            "target_id": self.target_id,
            "target_program": self.target_program,
            "arm": self.arm,
            "status": self.status,
            "path_length": self.path_length,
            "nodes_expanded": self.nodes_expanded,
            "max_frontier": self.max_frontier,
            "wall_seconds": self.wall_seconds,
            "replay_ok": self.replay_ok,
            "cost_attribution": dict(self.cost_attribution),
            "scaling_features": dict(self.scaling_features),
            "plan_fingerprint": self.plan_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BridgePlannerCase":
        return cls(
            case_id=str(data["case_id"]),
            source_seed_id=str(data["source_seed_id"]),
            target_id=str(data["target_id"]),
            target_program=str(data["target_program"]),
            arm=str(data["arm"]),
            status=str(data["status"]),
            path_length=int(data.get("path_length", 0)),
            nodes_expanded=int(data.get("nodes_expanded", 0)),
            max_frontier=int(data.get("max_frontier", 0)),
            wall_seconds=float(data.get("wall_seconds", 0.0)),
            replay_ok=bool(data.get("replay_ok", False)),
            cost_attribution={
                k: float(v) for k, v in (data.get("cost_attribution") or {}).items()
            },
            scaling_features={
                k: float(v) for k, v in (data.get("scaling_features") or {}).items()
            },
            plan_fingerprint=str(data.get("plan_fingerprint", "")),
        )


@dataclass(frozen=True)
class BridgePlannerArmSummary:
    """Aggregate statistics for one bridge planner arm."""

    arm_name: str
    n_cases: int
    n_reached: int
    n_unknown_budget: int
    n_unreachable: int
    n_certificate_failure: int
    mean_path_length: float
    p95_path_length: float
    mean_wall_seconds: float
    mean_nodes_expanded: float
    source_bias_index: float
    path_entropy_bits: float
    excess_cost_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "n_cases": self.n_cases,
            "n_reached": self.n_reached,
            "n_unknown_budget": self.n_unknown_budget,
            "n_unreachable": self.n_unreachable,
            "n_certificate_failure": self.n_certificate_failure,
            "mean_path_length": self.mean_path_length,
            "p95_path_length": self.p95_path_length,
            "mean_wall_seconds": self.mean_wall_seconds,
            "mean_nodes_expanded": self.mean_nodes_expanded,
            "source_bias_index": self.source_bias_index,
            "path_entropy_bits": self.path_entropy_bits,
            "excess_cost_ratio": self.excess_cost_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BridgePlannerArmSummary":
        return cls(
            arm_name=str(data["arm_name"]),
            n_cases=int(data["n_cases"]),
            n_reached=int(data["n_reached"]),
            n_unknown_budget=int(data["n_unknown_budget"]),
            n_unreachable=int(data["n_unreachable"]),
            n_certificate_failure=int(data["n_certificate_failure"]),
            mean_path_length=float(data["mean_path_length"]),
            p95_path_length=float(data["p95_path_length"]),
            mean_wall_seconds=float(data["mean_wall_seconds"]),
            mean_nodes_expanded=float(data["mean_nodes_expanded"]),
            source_bias_index=float(data["source_bias_index"]),
            path_entropy_bits=float(data["path_entropy_bits"]),
            excess_cost_ratio=float(data["excess_cost_ratio"]),
        )


@dataclass(frozen=True)
class BridgePlannerManifest:
    """Full fixture manifest for SLM-189."""

    schema: str
    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    arms: tuple[BridgePlannerArmSummary, ...]
    cases: tuple[BridgePlannerCase, ...]
    n_cases: int
    n_reached: int
    source_policies: tuple[str, ...]
    version_stamp: dict[str, Any]
    timestamp: str
    disposition: str = "inconclusive"
    disposition_rationale: str = ""
    honest_caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "arms": [arm.to_dict() for arm in self.arms],
            "cases": [case.to_dict() for case in self.cases],
            "n_cases": self.n_cases,
            "n_reached": self.n_reached,
            "source_policies": list(self.source_policies),
            "version_stamp": dict(self.version_stamp),
            "timestamp": self.timestamp,
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "honest_caveats": list(self.honest_caveats),
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BridgePlannerManifest":
        return cls(
            schema=str(data.get("schema", "BridgePlannerManifest")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", f"{EXPERIMENT_ID}-fixture")),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            arms=tuple(
                BridgePlannerArmSummary.from_dict(a) for a in data.get("arms", ())
            ),
            cases=tuple(
                BridgePlannerCase.from_dict(c) for c in data.get("cases", ())
            ),
            n_cases=int(data.get("n_cases", 0)),
            n_reached=int(data.get("n_reached", 0)),
            source_policies=tuple(data.get("source_policies", ())),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", ())),
        )


def _canonicalize_target(raw_source: str, slots: list[str]) -> tuple[str, list[str]]:
    """Canonicalize a target program and return (program, slots)."""
    try:
        canonical = __import__("slm_training.dsl.canonicalize", fromlist=["canonicalize"]).canonicalize(raw_source, validate=True)
    except Exception:  # noqa: BLE001
        canonical = raw_source
    return canonical, slots


def build_exact_fixture_targets() -> list[tuple[str, str, list[str]]]:
    """Return a small set of deterministic OpenUI fixture targets."""
    targets: list[tuple[str, str, list[str]]] = [
        (
            "hero_card",
            'root = Stack([hero], "column")\n'
            'hero_title = TextContent(":hero.title")\n'
            'hero_body = TextContent(":hero.body")\n'
            'hero = Card([hero_title, hero_body])',
            [":hero.title", ":hero.body"],
        ),
        (
            "simple_stack",
            'root = Stack([blurb], "column")\nblurb = TextContent(":page.blurb")',
            [":page.blurb"],
        ),
        (
            "card_with_button",
            'root = Stack([card], "column")\n'
            'card_title = TextContent(":card.title")\n'
            'card_action = Button(":card.action")\n'
            'card = Card([card_title, card_action])',
            [":card.title", ":card.action"],
        ),
        (
            "button_row",
            'root = Stack([primary, secondary], "row")\n'
            'primary = Button(":actions.primary")\n'
            'secondary = Button(":actions.secondary")',
            [":actions.primary", ":actions.secondary"],
        ),
        (
            "nested_stack",
            'root = Stack([outer], "column")\n'
            'inner = Stack([text], "row")\n'
            'text = TextContent(":nested.text")\n'
            'outer = Card([inner])',
            [":nested.text"],
        ),
        (
            "image_card",
            'root = Stack([card], "column")\n'
            'thumb = Image(":image.src")\n'
            'caption = TextContent(":image.caption")\n'
            'card = Card([thumb, caption])',
            [":image.src", ":image.caption"],
        ),
    ]
    canonical_targets: list[tuple[str, str, list[str]]] = []
    for tid, prog, slots in targets:
        try:
            canonical = __import__("slm_training.dsl.canonicalize", fromlist=["canonicalize"]).canonicalize(prog, validate=True)
        except Exception:  # noqa: BLE001
            canonical = prog
        canonical_targets.append((tid, canonical, slots))
    return canonical_targets


def build_synthetic_scale_targets(
    grid_sizes: list[tuple[int, int, int, int]] | None = None,
) -> list[tuple[str, str, list[str]]]:
    """Generate deterministic scale targets with varying node/depth/binder/slot counts."""
    if grid_sizes is None:
        grid_sizes = [(2, 1, 1, 1), (4, 2, 2, 2), (6, 2, 3, 3)]
    targets: list[tuple[str, str, list[str]]] = []
    for nodes, depth, binders, slots in grid_sizes:
        safe_nodes = max(2, nodes)
        safe_depth = max(1, depth)
        safe_binders = max(1, binders)
        safe_slots = max(1, slots)
        lines: list[str] = []
        slot_names: list[str] = [f":scale.s{i}" for i in range(safe_slots)]
        rng = random.Random(nodes * 1000 + depth * 100 + binders * 10 + slots)

        # Build a shallow tree: root -> container -> leaves.
        container_name = "container"
        lines.append(f'root = Stack([{container_name}], "column")')
        leaf_children: list[str] = []
        for i in range(safe_nodes - 2):
            leaf_name = f"leaf{i}"
            slot = slot_names[i % len(slot_names)]
            comp = rng.choice(["TextContent", "Button", "Image"])
            lines.append(f"{leaf_name} = {comp}(\"{slot}\")")
            leaf_children.append(leaf_name)
        lines.append(f'container = Card([{", ".join(leaf_children)}])')
        program = "\n".join(lines)
        try:
            program = __import__("slm_training.dsl.canonicalize", fromlist=["canonicalize"]).canonicalize(program, validate=True)
        except Exception:  # noqa: BLE001
            pass
        targets.append((f"scale_n{safe_nodes}_d{safe_depth}_b{safe_binders}_s{safe_slots}", program, slot_names[:safe_binders]))
    return targets


def _build_source_program(
    target_program: str, source_seed_id: str, rng: random.Random
) -> str:
    """Build a source program for the named source policy."""
    if source_seed_id == "gold":
        return target_program
    if source_seed_id == "minimal":
        return build_sketch_seed(target_program)
    if source_seed_id == "template":
        # A slightly richer seed: sketch but with container directions preserved.
        sketch = build_sketch_seed(target_program)
        # Best-effort direction preservation by replacing "column"/"row" in sketch
        # with the target's direction where obvious.  This is intentionally simple.
        from slm_training.models.tree_edit_diffusion import parse_statements
        target_stmts = parse_statements(target_program)
        sketch_stmts = parse_statements(sketch)
        if target_stmts is None or sketch_stmts is None:
            return sketch
        direction_by_name: dict[str, str] = {}
        for stmt in target_stmts:
            if stmt.has_list:
                rest = stmt.rest.strip()
                if rest.startswith('"'):
                    direction_by_name[stmt.name] = rest.strip('"')
        for stmt in sketch_stmts:
            if stmt.has_list and stmt.name in direction_by_name:
                stmt.rest = f'"{direction_by_name[stmt.name]}"'
        from slm_training.models.tree_edit_diffusion import render_statements
        return render_statements(sketch_stmts)
    if source_seed_id == "retrieved":
        # Apply roughly half of the greedy edits to the minimal sketch.
        sketch = build_sketch_seed(target_program)
        edits, _ = plan_edit_sequence(sketch, target_program)
        if not edits:
            return sketch
        n_apply = max(1, len(edits) // 2)
        current = sketch
        for edit in edits[:n_apply]:
            nxt = apply_canonical_edit(current, edit)
            if nxt is None:
                break
            current = nxt
        return current
    return build_sketch_seed(target_program)


def _plan_fingerprint(result: BridgePlannerResultV1) -> str:
    """Stable fingerprint of a plan (or lack thereof)."""
    if result.plan is None:
        return ""
    return canonical_fingerprint(result.plan.target_program)


def _run_one_case(
    case_id: str,
    source_seed_id: str,
    target_id: str,
    target_program: str,
    arm: str,
    rng: random.Random,
    exact_budget: int,
) -> BridgePlannerCase:
    """Run one bridge planning attempt and package the case."""
    source = _build_source_program(target_program, source_seed_id, rng)
    result = plan_bridge(
        source,
        target_program,
        arm=arm,
        source_seed_id=source_seed_id,
        plan_id=f"{case_id}-{arm}",
        rng_seed=rng.randint(0, 2**31 - 1),
        max_edits=12,
        exact_budget=exact_budget,
    )
    return BridgePlannerCase(
        case_id=case_id,
        source_seed_id=source_seed_id,
        target_id=target_id,
        target_program=target_program,
        arm=arm,
        status=result.status,
        path_length=result.plan.path_length if result.plan else 0,
        nodes_expanded=result.nodes_expanded,
        max_frontier=result.max_frontier,
        wall_seconds=result.wall_seconds,
        replay_ok=result.replay_ok,
        cost_attribution=dict(result.cost_attribution),
        scaling_features=dict(result.scaling_features),
        plan_fingerprint=_plan_fingerprint(result),
    )


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _entropy_bits(values: list[int]) -> float:
    if not values:
        return 0.0
    total = len(values)
    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p:
            entropy -= p * math.log2(p)
    return entropy


def _summarize_arm(
    arm_name: str,
    cases: list[BridgePlannerCase],
    greedy_mean_length: float,
) -> BridgePlannerArmSummary:
    """Build aggregate statistics for one arm."""
    n_cases = len(cases)
    n_reached = sum(1 for c in cases if c.status == REACHED)
    n_unknown = sum(1 for c in cases if c.status == UNKNOWN_BUDGET)
    n_unreachable = sum(1 for c in cases if c.status == "unreachable_complete")
    n_cert_failure = sum(1 for c in cases if c.status == "certificate_failure")
    path_lengths = [c.path_length for c in cases]
    wall_seconds = [c.wall_seconds for c in cases]
    nodes_expanded = [c.nodes_expanded for c in cases]

    mean_path = sum(path_lengths) / max(1, n_cases)
    p95_path = _percentile([float(p) for p in path_lengths], 0.95)
    mean_wall = sum(wall_seconds) / max(1, n_cases)
    mean_nodes = sum(nodes_expanded) / max(1, n_cases)

    # Source bias: coefficient of variation of mean path length per source policy.
    by_source: dict[str, list[int]] = {}
    for c in cases:
        by_source.setdefault(c.source_seed_id, []).append(c.path_length)
    means = [sum(v) / len(v) for v in by_source.values() if v]
    if len(means) > 1 and sum(means) > 0:
        avg_mean = sum(means) / len(means)
        variance = sum((m - avg_mean) ** 2 for m in means) / len(means)
        source_bias = math.sqrt(variance) / avg_mean
    else:
        source_bias = 0.0

    path_entropy = _entropy_bits(path_lengths)
    excess_cost_ratio = mean_path / max(1e-9, greedy_mean_length)

    return BridgePlannerArmSummary(
        arm_name=arm_name,
        n_cases=n_cases,
        n_reached=n_reached,
        n_unknown_budget=n_unknown,
        n_unreachable=n_unreachable,
        n_certificate_failure=n_cert_failure,
        mean_path_length=mean_path,
        p95_path_length=p95_path,
        mean_wall_seconds=mean_wall,
        mean_nodes_expanded=mean_nodes,
        source_bias_index=source_bias,
        path_entropy_bits=path_entropy,
        excess_cost_ratio=excess_cost_ratio,
    )


def _resolve_disposition(
    cases: tuple[BridgePlannerCase, ...],
    arms: tuple[BridgePlannerArmSummary, ...],
    exact_budget: int,
) -> tuple[str, str]:
    """Classify the fixture outcome."""
    if not cases:
        return ("inconclusive", "No cases were generated.")

    # The selected budget arm is canonical_greedy; evaluate reachability there.
    greedy_cases = [c for c in cases if c.arm == "canonical_greedy"]
    n_greedy = len(greedy_cases)
    n_greedy_reached = sum(1 for c in greedy_cases if c.status == REACHED)
    if n_greedy > 0 and n_greedy_reached / n_greedy < 0.95:
        return (
            "inconclusive",
            f"Only {n_greedy_reached}/{n_greedy} canonical_greedy cases reached the target; "
            "need >= 95% of selected-budget cases to support greedy determinism.",
        )

    by_key: dict[tuple[str, str, str], BridgePlannerCase] = {
        (c.source_seed_id, c.target_id, c.arm): c for c in cases
    }

    # Check that random_shortest and dependency_dag never create invalid intermediates.
    for arm in ("random_shortest", "dependency_dag"):
        for c in cases:
            if c.arm != arm:
                continue
            if c.status == REACHED and not c.replay_ok:
                return (
                    "inconclusive",
                    f"Arm {arm} reported REACHED but replay_ok=False for {c.case_id}.",
                )

    # Check exact_shortest matches greedy for tiny cases (path_length <= exact_budget).
    tiny_mismatch = 0
    tiny_total = 0
    for c in cases:
        if c.arm != "canonical_greedy" or c.status != REACHED:
            continue
        if c.path_length > exact_budget:
            continue
        tiny_total += 1
        exact_key = (c.source_seed_id, c.target_id, "exact_shortest")
        exact_case = by_key.get(exact_key)
        if exact_case is None or exact_case.status != REACHED or exact_case.path_length != c.path_length:
            tiny_mismatch += 1

    if tiny_total > 0 and tiny_mismatch > 0:
        return (
            "inconclusive",
            f"exact_shortest mismatched canonical_greedy on {tiny_mismatch}/{tiny_total} "
            "tiny cases; the exact BFS arm is not fully exercised in this fixture.",
        )

    return (
        "supports_greedy_determinism",
        "Over the bounded fixture domain, greedy plans reach targets, dependency-respecting "
        "permutations replay successfully, and exact-shortest matches greedy on tiny cases.",
    )


def run_bridge_planner_fixture(
    output_dir: Path | None = None,
    *,
    arms: tuple[str, ...] = ARM_NAMES,
    source_policies: tuple[str, ...] = ("minimal", "template", "gold", "retrieved"),
    exact_budget: int = 8,
    include_scale_grid: bool = False,
    scale_grid_sizes: list[tuple[int, int, int, int]] | None = None,
    write_design_docs: bool = True,
    design_json: Path | None = None,
    design_md: Path | None = None,
) -> BridgePlannerManifest:
    """Run the SLM-189 bridge planner fixture campaign."""
    start = time.perf_counter()
    rng = random.Random(0)

    exact_targets = build_exact_fixture_targets()
    if include_scale_grid:
        scale_targets = build_synthetic_scale_targets(scale_grid_sizes)
    else:
        scale_targets = []
    all_targets = exact_targets + scale_targets

    cases: list[BridgePlannerCase] = []
    for target_id, target_program, slots in all_targets:
        for source_policy in source_policies:
            case_base = f"{source_policy}__{target_id}"
            for arm in arms:
                case_id = f"{case_base}__{arm}"
                cases.append(
                    _run_one_case(
                        case_id=case_id,
                        source_seed_id=source_policy,
                        target_id=target_id,
                        target_program=target_program,
                        arm=arm,
                        rng=rng,
                        exact_budget=exact_budget,
                    )
                )

    # Compute per-arm summaries.
    by_arm: dict[str, list[BridgePlannerCase]] = {}
    for c in cases:
        by_arm.setdefault(c.arm, []).append(c)

    greedy_cases = by_arm.get("canonical_greedy", [])
    greedy_mean_length = (
        sum(c.path_length for c in greedy_cases) / max(1, len(greedy_cases))
    )

    arms_list = tuple(
        _summarize_arm(arm, by_arm.get(arm, []), greedy_mean_length)
        for arm in arms
    )

    disposition, rationale = _resolve_disposition(
        tuple(cases), arms_list, exact_budget
    )

    elapsed = time.perf_counter() - start
    version_stamp = build_version_stamp(
        "harness.experiments",
        "harness.experiments.slm189_bridge_planner",
        "matrix.slm189_bridge_planner",
        "data.flow.bridge_planner",
        "harness.model_build.eval",
    )

    manifest = BridgePlannerManifest(
        schema="BridgePlannerManifest",
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        status="fixture",
        claim_class="wiring",
        hypothesis=_HYPOTHESIS,
        falsifier=_FALSIFIER,
        arms=arms_list,
        cases=tuple(cases),
        n_cases=len(cases),
        n_reached=sum(1 for c in cases if c.status == REACHED),
        source_policies=source_policies,
        version_stamp=version_stamp,
        timestamp=_now(),
        disposition=disposition,
        disposition_rationale=rationale,
        honest_caveats=_HONEST_CAVEATS,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_json(output_dir / "slm189_bridge_planner_report.json")

        if write_design_docs:
            if design_json is None or design_md is None:
                root = _project_root()
                design_json = root / f"docs/design/iter-slm189-bridge-planner-{_today_yyyymmdd()}.json"
                design_md = root / f"docs/design/iter-slm189-bridge-planner-{_today_yyyymmdd()}.md"
            design_json.parent.mkdir(parents=True, exist_ok=True)
            design_md.parent.mkdir(parents=True, exist_ok=True)
            manifest.to_json(design_json)
            design_md.write_text(render_markdown(manifest), encoding="utf-8")

    # Attach wall time to lineage via a fresh manifest.
    lineage_extra = {"wall_seconds": _clamp(elapsed, low=0.001, high=10.0)}
    stamp = dict(manifest.version_stamp)
    stamp["lineage"] = lineage_extra
    manifest = BridgePlannerManifest(
        schema=manifest.schema,
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        experiment_id=manifest.experiment_id,
        run_id=manifest.run_id,
        status=manifest.status,
        claim_class=manifest.claim_class,
        hypothesis=manifest.hypothesis,
        falsifier=manifest.falsifier,
        arms=manifest.arms,
        cases=manifest.cases,
        n_cases=manifest.n_cases,
        n_reached=manifest.n_reached,
        source_policies=manifest.source_policies,
        version_stamp=stamp,
        timestamp=manifest.timestamp,
        disposition=manifest.disposition,
        disposition_rationale=manifest.disposition_rationale,
        honest_caveats=manifest.honest_caveats,
    )

    return manifest


def render_markdown(manifest: BridgePlannerManifest) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-189 (FFE2-01): bridge planner fixture ({manifest.run_id})",
        "",
        f"Matrix set: `{manifest.matrix_set}`",
        "",
        f"Version: `{manifest.matrix_version}`",
        "",
        f"Status: **{manifest.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU, no model, no checkpoint, and no "
        "ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        manifest.hypothesis,
        "",
        "## Falsifier",
        "",
        manifest.falsifier,
        "",
        "## Arms",
        "",
        "| arm_name | n_cases | n_reached | n_unknown_budget | mean_path_length | p95_path_length | source_bias_index | excess_cost_ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in manifest.arms:
        lines.append(
            f"| {arm.arm_name} | {arm.n_cases} | {arm.n_reached} | {arm.n_unknown_budget} | "
            f"{arm.mean_path_length:.2f} | {arm.p95_path_length:.2f} | "
            f"{arm.source_bias_index:.4f} | {arm.excess_cost_ratio:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Cases",
            "",
            f"Total cases: {manifest.n_cases}",
            f"Reached: {manifest.n_reached}",
            f"Source policies: {', '.join(manifest.source_policies)}",
            "",
            "| case_id | source_seed_id | target_id | arm | status | path_length | replay_ok |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for case in manifest.cases:
        lines.append(
            f"| {case.case_id} | {case.source_seed_id} | {case.target_id} | {case.arm} | "
            f"{case.status} | {case.path_length} | {case.replay_ok} |"
        )

    lines.extend(
        [
            "",
            "## Disposition",
            "",
            f"**{manifest.disposition}**",
            "",
            manifest.disposition_rationale,
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This is a wiring fixture. The bridge planner arms, "
            "dependency DAG, transition certificates, and source-policy variation are "
            "exercised over deterministic synthetic targets, but no real model or decode "
            "path was run. The mechanism remains ``retain_diagnostic`` / "
            "``blocked_pending_real_model`` until trained-model bridge telemetry and AgentV "
            "evaluation are available.",
            "",
            "## Honest caveats",
            "",
        ]
    )
    for caveat in manifest.honest_caveats:
        lines.append(f"- {caveat}")

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_bridge_planner_audit --mode plan-only",
            "python -m scripts.run_bridge_planner_audit --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def validate_manifest(manifest: BridgePlannerManifest) -> list[str]:
    """Validate the bridge planner fixture manifest."""
    errors: list[str] = []
    seen: set[str] = set()
    for arm in manifest.arms:
        if arm.arm_name in seen:
            errors.append(f"duplicate arm: {arm.arm_name}")
        seen.add(arm.arm_name)
        if arm.arm_name not in ARM_NAMES:
            errors.append(f"unknown arm: {arm.arm_name!r}")
        if arm.n_cases < 0:
            errors.append(f"{arm.arm_name}: negative n_cases")
        if arm.n_reached < 0 or arm.n_reached > arm.n_cases:
            errors.append(f"{arm.arm_name}: invalid n_reached")

    if manifest.n_cases != len(manifest.cases):
        errors.append("n_cases does not match len(cases)")

    case_ids = {c.case_id for c in manifest.cases}
    if len(case_ids) != len(manifest.cases):
        errors.append("duplicate case_id")

    for case in manifest.cases:
        if case.arm not in ARM_NAMES:
            errors.append(f"{case.case_id}: unknown arm {case.arm!r}")
        if case.status == REACHED and not case.replay_ok:
            errors.append(f"{case.case_id}: reached but replay_ok=False")
        if case.wall_seconds < 0:
            errors.append(f"{case.case_id}: negative wall_seconds")

    return errors
