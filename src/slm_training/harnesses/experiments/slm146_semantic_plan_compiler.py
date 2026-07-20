"""SLM-146 SPV1-03: plan-compiler bridge fixture matrix.

Wiring-only evidence. The fixture exercises the OpenUISemanticPlanCompiler on
a deterministic synthetic corpus, measuring seed validity, soft-feature
attachment, and certified-only restrictions. No production decoder is changed
and no ship claim is made.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.semantic_plan import (
    Evidence,
    EvidenceKind,
    OpenUISemanticPlanCompiler,
)
from slm_training.data.semantic_plan.corpus import build_fixture_plan_corpus
from slm_training.dsl.pack import get_pack
from slm_training.dsl.parser import validate
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "Slm146Arm",
    "Slm146Manifest",
    "Slm146Record",
    "Slm146Report",
    "Slm146Row",
    "build_manifest",
    "render_markdown",
    "run_fixture_matrix",
]

MATRIX_VERSION = "spv1-03-v1"
MATRIX_SET = "slm146_semantic_plan_compiler"


@dataclass(frozen=True)
class Slm146Arm:
    """One diagnostic arm from the SPV1-03 matrix."""

    arm_id: str
    seed_mode: str  # baseline | gold | none
    feature_mode: str  # off | soft
    restriction_mode: str  # compiler_only | certified | unsafe_predicted_hard
    description: str
    promotable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Slm146Record:
    """Per-record diagnostics for one arm."""

    record_id: str
    seed_ok: bool
    seed_valid: bool
    seed_to_gold_ratio: float | None
    role_coverage: float
    topology_coverage: float
    binding_coverage: float
    action_count: int
    soft_feature_count: int
    hard_removal_count: int
    false_hard_prune_count: int

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Slm146Row:
    """Aggregated row for one arm."""

    arm_id: str
    status: str
    promotable: bool
    n_records: int
    seed_ok_count: int
    seed_valid_count: int
    mean_seed_to_gold_ratio: float | None
    mean_role_coverage: float
    mean_topology_coverage: float
    mean_binding_coverage: float
    total_soft_features: int
    total_hard_removals: int
    total_false_hard_prunes: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Slm146Manifest:
    """Preregistered manifest for the SLM-146 fixture matrix."""

    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    hypothesis: str = (
        "A deterministic plan compiler produces valid OpenUI seeds, attaches soft "
        "action features without changing legal membership, and gates hard "
        "restrictions behind certified evidence; unsafe predicted-hard controls "
        "are non-promotable diagnostics."
    )
    falsifier: str = (
        "Either the plan-derived seeds are invalid, soft features alter the legal "
        "candidate set, or any non-certified prediction removes a supported "
        "candidate in a promotable arm."
    )
    arms: list[Slm146Arm] = field(default_factory=list)
    claim_class: str = "wiring"
    status: str = "not_run"

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["arms"] = [arm.to_dict() for arm in self.arms]
        return data

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class Slm146Report:
    """Full fixture report for SLM-146."""

    matrix_set: str
    matrix_version: str
    run_id: str
    status: str
    manifest: Slm146Manifest
    rows: list[Slm146Row]
    version_stamp: dict[str, Any] = field(default_factory=dict)
    claim_class: str = "wiring"

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "manifest": self.manifest.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


def build_manifest() -> Slm146Manifest:
    """Return the default SLM-146 diagnostic matrix manifest."""
    arms = [
        Slm146Arm(
            arm_id="A_baseline",
            seed_mode="baseline",
            feature_mode="off",
            restriction_mode="compiler_only",
            description="Minimal baseline seed, no plan features, compiler-only restrictions.",
        ),
        Slm146Arm(
            arm_id="B_gold_seed",
            seed_mode="gold",
            feature_mode="off",
            restriction_mode="compiler_only",
            description="Gold plan seed, no plan features, compiler-only restrictions.",
        ),
        Slm146Arm(
            arm_id="C_gold_seed_soft",
            seed_mode="gold",
            feature_mode="soft",
            restriction_mode="compiler_only",
            description="Gold plan seed with soft plan features, compiler-only restrictions.",
        ),
        Slm146Arm(
            arm_id="D_baseline_soft",
            seed_mode="baseline",
            feature_mode="soft",
            restriction_mode="compiler_only",
            description="Baseline seed with soft gold-plan features, compiler-only restrictions.",
        ),
        Slm146Arm(
            arm_id="E_certified_restrictions",
            seed_mode="gold",
            feature_mode="soft",
            restriction_mode="certified",
            description="Gold plan seed with soft features and certified-only authored restrictions.",
        ),
        Slm146Arm(
            arm_id="F_unsafe_predicted_hard",
            seed_mode="gold",
            feature_mode="soft",
            restriction_mode="unsafe_predicted_hard",
            description="Unsafe predicted-hard fixture control; non-promotable diagnostic.",
            promotable=False,
        ),
    ]
    return Slm146Manifest(arms=arms)


def _render_fixture_ast(ast: dict[str, Any]) -> str:
    """Render the SLM-144 fixture AST shape back to OpenUI source.

    This is intentionally local to the fixture harness; production rendering
    continues to use the official serializer/canonicalizer. The renderer
    produces the OpenUI positional-string convention for leaf content props
    and the ``Container([children], direction)`` convention for Stacks.
    """

    def _render_node(node: dict[str, Any], name: str) -> str:
        type_name = str(node.get("typeName") or "")
        props = dict(node.get("props") or {})
        children = props.pop("children", None)
        if isinstance(children, list) and children:
            child_lines: list[str] = []
            refs: list[str] = []
            for index, child in enumerate(children):
                child_name = f"n{index}"
                child_lines.append(_render_node(child, child_name))
                refs.append(child_name)
            children_source = f"[{', '.join(refs)}]"
            if type_name == "Stack" and "direction" in props:
                body = f"{children_source}, {json.dumps(props['direction'], ensure_ascii=False)}"
            else:
                body = children_source
            node_line = f"{name} = {type_name}({body})"
            if child_lines:
                return node_line + "\n" + "\n".join(child_lines)
            return node_line

        # Leaf: render the single content prop as a positional string argument.
        if props:
            value = next(iter(props.values()))
            return f"{name} = {type_name}({json.dumps(value, ensure_ascii=False)})"
        return f"{name} = {type_name}()"

    root = ast.get("root")
    if not isinstance(root, dict):
        return ""
    return _render_node(root, "root").strip()


def _canonicalize_source(source: str) -> str | None:
    """Return the canonical serialized form of *source*, or None if invalid."""
    try:
        program = validate(source)
        return program.serialized or source.strip()
    except Exception:  # noqa: BLE001
        return None


def _token_ratio(a: str | None, b: str | None) -> float | None:
    if a is None or b is None:
        return None
    return SequenceMatcher(None, a.split(), b.split()).ratio()


def _coverage_fraction(plan: SemanticPlanV1, seed: str | None) -> tuple[float, float, float]:
    if seed is None:
        return 0.0, 0.0, 0.0
    n_roles = len(plan.role_slots)
    n_edges = len(plan.topology.parent_relation_candidates or ())
    n_bindings = len(plan.bindings)
    # Seed builder preserves all roles and edges that are valid; approximate
    # coverage by checking that the seed mentions role/component tokens.
    role_hits = sum(
        1
        for slot in plan.role_slots
        if slot.component_family and slot.component_family in seed
    )
    edge_hits = sum(
        1
        for edge in plan.topology.parent_relation_candidates or ()
        if str(edge.get("child_role_id") or "") in seed
        or str(edge.get("parent_role_id") or "") in seed
    )
    binding_hits = sum(
        1
        for binding in plan.bindings
        if binding.role_slot_id in seed
    )
    role_coverage = role_hits / n_roles if n_roles else 1.0
    topology_coverage = edge_hits / n_edges if n_edges else 1.0
    binding_coverage = binding_hits / n_bindings if n_bindings else 1.0
    return role_coverage, topology_coverage, binding_coverage


def _actions_for_record(spec: ProgramSpec) -> list[str]:
    """Return a small synthetic action inventory for the record."""
    families = {"Stack", "Card", "TextContent", "Button", "Input", "CardHeader"}
    return sorted(families | set(spec.canonical_openui.split()))[:12]


def _evidence_for_record(
    plan: SemanticPlanV1, mode: str
) -> tuple[list[Evidence], bool]:
    """Return evidence list and whether the compiler should run unsafe mode."""
    evidence: list[Evidence] = []
    if mode == "certified":
        # Author a certified requirement that Stack is allowed (no-op) and
        # a forbidden component restriction that is certified by pack schema.
        evidence.append(
            Evidence(
                evidence_id="NonExistentComponent",
                kind=EvidenceKind.COMPILER_AUTHORED_CERTIFIED,
                certificate="pack_schema:forbidden_component",
            )
        )
    elif mode == "unsafe_predicted_hard":
        evidence.append(
            Evidence(
                evidence_id="Card",
                kind=EvidenceKind.PREDICTION_ONLY,
            )
        )
    return evidence, mode == "unsafe_predicted_hard"


def _run_arm_on_record(
    arm: Slm146Arm,
    spec: ProgramSpec,
    plan: SemanticPlanV1,
    pack: Any,
) -> Slm146Record:
    use_gold_seed = arm.seed_mode == "gold"
    use_features = arm.feature_mode == "soft"
    restriction_mode = arm.restriction_mode

    compiler = OpenUISemanticPlanCompiler(
        honesty_mode="oracle_diagnostic",
        allow_unsafe_predicted_hard_control=(restriction_mode == "unsafe_predicted_hard"),
    )

    seed_plan = plan if use_gold_seed else None
    seed_result = compiler.build_valid_seed(None, seed_plan, pack)
    seed = seed_result.seed

    feature_plan = plan if use_features else None
    actions = _actions_for_record(spec)
    features = compiler.annotate_actions(None, actions, feature_plan)
    soft_feature_count = sum(
        1
        for f in features
        if f.matches_predicted_role or f.component_family_compatible
    )

    evidence, _ = _evidence_for_record(plan, restriction_mode)
    restriction_result = compiler.certified_restrictions(
        None, None, plan if use_features else None, evidence
    )

    role_cov, topo_cov, bind_cov = _coverage_fraction(plan, seed)
    gold_source = _render_fixture_ast(spec.ast)
    seed_canonical = _canonicalize_source(seed) if seed else None
    ratio = _token_ratio(seed_canonical, gold_source)

    return Slm146Record(
        record_id=spec.id,
        seed_ok=seed_result.ok,
        seed_valid=bool(seed and seed_canonical is not None),
        seed_to_gold_ratio=ratio,
        role_coverage=role_cov,
        topology_coverage=topo_cov,
        binding_coverage=bind_cov,
        action_count=len(actions),
        soft_feature_count=soft_feature_count,
        hard_removal_count=len(restriction_result.hard_removals),
        false_hard_prune_count=restriction_result.false_hard_prune_count,
    )


def _aggregate_records(arm: Slm146Arm, records: list[Slm146Record]) -> Slm146Row:
    n = len(records)
    if not n:
        return Slm146Row(
            arm_id=arm.arm_id,
            status="empty",
            promotable=arm.promotable,
            n_records=0,
            seed_ok_count=0,
            seed_valid_count=0,
            mean_seed_to_gold_ratio=None,
            mean_role_coverage=0.0,
            mean_topology_coverage=0.0,
            mean_binding_coverage=0.0,
            total_soft_features=0,
            total_hard_removals=0,
            total_false_hard_prunes=0,
        )

    ratios = [r.seed_to_gold_ratio for r in records if r.seed_to_gold_ratio is not None]
    mean_ratio = sum(ratios) / len(ratios) if ratios else None

    notes = [
        f"seed_mode={arm.seed_mode}",
        f"feature_mode={arm.feature_mode}",
        f"restriction_mode={arm.restriction_mode}",
        "fixture-only: synthetic corpus, no production decoder",
    ]
    if not arm.promotable:
        notes.append("non-promotable diagnostic arm")

    return Slm146Row(
        arm_id=arm.arm_id,
        status="fixture",
        promotable=arm.promotable,
        n_records=n,
        seed_ok_count=sum(1 for r in records if r.seed_ok),
        seed_valid_count=sum(1 for r in records if r.seed_valid),
        mean_seed_to_gold_ratio=mean_ratio,
        mean_role_coverage=sum(r.role_coverage for r in records) / n,
        mean_topology_coverage=sum(r.topology_coverage for r in records) / n,
        mean_binding_coverage=sum(r.binding_coverage for r in records) / n,
        total_soft_features=sum(r.soft_feature_count for r in records),
        total_hard_removals=sum(r.hard_removal_count for r in records),
        total_false_hard_prunes=sum(r.false_hard_prune_count for r in records),
        notes=notes,
    )


def run_fixture_matrix(
    corpus: dict[str, list[tuple[ProgramSpec, SemanticPlanV1]]] | None = None,
    *,
    run_id: str = "slm146_fixture",
    output_dir: Path | None = None,
) -> Slm146Report:
    """Run the SLM-146 diagnostic matrix on the fixture corpus."""
    manifest = build_manifest()
    if corpus is None:
        corpus = build_fixture_plan_corpus(
            count=64,
            seed=0,
            root_containers=["Stack", "Card"],
            leaf_components=["TextContent", "Button"],
        )
    pack = get_pack("openui")
    val_records = corpus.get("val", [])

    rows: list[Slm146Row] = []
    for arm in manifest.arms:
        per_record: list[Slm146Record] = []
        for spec, plan in val_records:
            per_record.append(_run_arm_on_record(arm, spec, plan, pack))
        rows.append(_aggregate_records(arm, per_record))

    report = Slm146Report(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        rows=rows,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm146_plan_compiler",
        ),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm146_semantic_plan_compiler_report.json")
    return report


def render_markdown(report: Slm146Report) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-146 / SPV1-03: Plan-compiler bridge fixture matrix ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`  ",
        f"Version: `{report.matrix_version}`  ",
        f"Status: **{report.status}**  ",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no production "
        "decoder was changed, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        report.manifest.hypothesis,
        "",
        "## Falsifier",
        "",
        report.manifest.falsifier,
        "",
        "## Manifest",
        "",
        "| Arm | Seed | Features | Restrictions | Promotable |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in report.manifest.arms:
        lines.append(
            f"| {arm.arm_id} | {arm.seed_mode} | {arm.feature_mode} | "
            f"{arm.restriction_mode} | {arm.promotable} |"
        )

    lines.extend(["", "## Results", ""])
    for row in report.rows:
        lines.append(f"### {row.arm_id}")
        lines.append(f"- records: {row.n_records}")
        lines.append(f"- seed ok: {row.seed_ok_count}")
        lines.append(f"- seed valid: {row.seed_valid_count}")
        if row.mean_seed_to_gold_ratio is not None:
            lines.append(
                f"- mean seed-to-gold token ratio: {row.mean_seed_to_gold_ratio:.3f}"
            )
        lines.append(f"- mean role coverage: {row.mean_role_coverage:.3f}")
        lines.append(f"- mean topology coverage: {row.mean_topology_coverage:.3f}")
        lines.append(f"- mean binding coverage: {row.mean_binding_coverage:.3f}")
        lines.append(f"- total soft features: {row.total_soft_features}")
        lines.append(f"- total hard removals: {row.total_hard_removals}")
        lines.append(f"- total false hard prunes: {row.total_false_hard_prunes}")
        for note in row.notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(
        [
            "## Verdict",
            "",
            "If any promotable arm reports `false_hard_prune_count > 0`, the plan "
            "compiler has violated the certified-only restriction boundary. "
            "The unsafe predicted-hard arm is expected to show hard removals and "
            "is explicitly non-promotable.",
            "",
        ]
    )
    return "\n".join(lines)
