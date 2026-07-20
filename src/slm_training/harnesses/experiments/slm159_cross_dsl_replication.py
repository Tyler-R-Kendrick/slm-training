"""SLM-159 (SPV4-01): cross-DSL replication of the semantic-plan stack.

Fixture/wiring-only harness.  Ports the pack-neutral ``SemanticPlanV1``
extractor/seed pattern to GraphQL (SLM-43) and records a concrete readiness
blocker for the preregistered second-pack candidates (SLM-44/SLM-45).  No
production TwoTower wiring is touched and no ship-gate claim is made.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.progspec.semantic_plan import (
    PlanArchetype,
    PlanBinding,
    PlanConfidenceCalibration,
    PlanCoverage,
    PlanIdentity,
    PlanSymbol,
    PlanTopology,
    RoleSlot,
    SemanticPlanV1,
)
from slm_training.data.semantic_plan.seed import SeedResult
from slm_training.dsl.pack import DslPack, get_pack, list_packs
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "REPLICATION_CAMPAIGN_ID",
    "PackFamily",
    "PackArm",
    "CommonConfig",
    "ReadinessChecklist",
    "PackReadinessReport",
    "ReplicationRow",
    "CrossDslManifest",
    "CrossDslReplicationReport",
    "GraphQLSemanticPlanExtractor",
    "GraphQLPlanSeedBuilder",
    "build_manifest",
    "validate_manifest",
    "assess_pack_readiness",
    "extract_graphql_plan",
    "build_graphql_seed",
    "run_fixture_campaign",
    "render_markdown",
]

MATRIX_VERSION = "spv4-01-v1"
MATRIX_SET = "slm159_cross_dsl_replication"
REPLICATION_CAMPAIGN_ID = "slm159-cross-dsl-replication"

_CANDIDATE_SECOND_PACKS = ("design-patterns", "nomenclature", "ontology", "expert-nomenclature")


class PackFamily(str, Enum):
    """Pack family for SPV4-01 replication."""

    GRAPHQL = "graphql"
    SECOND_PACK = "second_pack"


@dataclass(frozen=True)
class CommonConfig:
    """Frozen orthogonal controls shared by every pack arm."""

    n_graphql_records: int = 8
    graphql_depth: int = 1
    seeds: tuple[int, ...] = (0, 1)
    max_records_per_root: int = 8
    metric_versions: dict[str, str] = field(default_factory=lambda: {"meaningful": "2.0.0"})

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["seeds"] = list(self.seeds)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommonConfig":
        return cls(
            n_graphql_records=data.get("n_graphql_records", 8),
            graphql_depth=data.get("graphql_depth", 1),
            seeds=tuple(data.get("seeds", [0, 1])),
            max_records_per_root=data.get("max_records_per_root", 8),
            metric_versions=data.get("metric_versions", {"meaningful": "2.0.0"}),
        )


@dataclass(frozen=True)
class PackArm:
    """One pack arm in the cross-DSL replication campaign."""

    arm_id: str
    family: PackFamily
    pack_id: str
    name: str
    description: str
    promotable: bool = True
    reference: bool = True
    blocked: bool = False
    blocker: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["family"] = self.family.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PackArm":
        return cls(
            arm_id=data["arm_id"],
            family=PackFamily(data.get("family", "graphql")),
            pack_id=data.get("pack_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            promotable=data.get("promotable", True),
            reference=data.get("reference", True),
            blocked=data.get("blocked", False),
            blocker=data.get("blocker", ""),
        )


@dataclass(frozen=True)
class ReadinessChecklist:
    """Machine-readable pack-readiness rubric from the issue."""

    versioned_grammar_parser: bool = False
    schema_type_scope_oracle: bool = False
    exact_legal_action_or_valid_edit_enumeration: bool = False
    typed_ast_or_program_spec_generator: bool = False
    canonical_equivalence_fingerprint: bool = False
    placeholder_symbol_binder_policy: bool = False
    request_prompt_contract: bool = False
    positive_corpus_and_disjoint_eval: bool = False
    pack_native_semantic_metric_and_audit: bool = False
    bounded_finite_cases: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReadinessChecklist":
        return cls(**{k: bool(data.get(k, False)) for k in asdict(cls()).keys()})


@dataclass(frozen=True)
class PackReadinessReport:
    """Readiness assessment for one candidate pack."""

    pack_id: str
    pack_available: bool
    parser_available: bool
    oracle_available: bool
    generator_available: bool
    canonicalizer_available: bool
    placeholder_policy_available: bool
    symbol_table: dict[str, list[str]] = field(default_factory=dict)
    checklist: ReadinessChecklist = field(default_factory=ReadinessChecklist)
    unsupported_features: list[str] = field(default_factory=list)
    readiness_pass: bool = False
    blocker: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["checklist"] = self.checklist.to_dict()
        data["symbol_table"] = dict(self.symbol_table)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PackReadinessReport":
        return cls(
            pack_id=data.get("pack_id", ""),
            pack_available=data.get("pack_available", False),
            parser_available=data.get("parser_available", False),
            oracle_available=data.get("oracle_available", False),
            generator_available=data.get("generator_available", False),
            canonicalizer_available=data.get("canonicalizer_available", False),
            placeholder_policy_available=data.get("placeholder_policy_available", False),
            symbol_table=dict(data.get("symbol_table", {})),
            checklist=ReadinessChecklist.from_dict(data.get("checklist", {})),
            unsupported_features=list(data.get("unsupported_features", [])),
            readiness_pass=data.get("readiness_pass", False),
            blocker=data.get("blocker", ""),
        )


@dataclass(frozen=True)
class ReplicationRow:
    """Aggregated replication result for one pack arm/seed."""

    arm_id: str
    pack_id: str
    seed: int
    promotable: bool
    n_records: int
    extraction_coverage: float
    seed_validity: float
    round_trip_equal: float
    mean_latency_ms: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplicationRow":
        return cls(
            arm_id=data["arm_id"],
            pack_id=data["pack_id"],
            seed=data["seed"],
            promotable=data.get("promotable", True),
            n_records=data["n_records"],
            extraction_coverage=data["extraction_coverage"],
            seed_validity=data["seed_validity"],
            round_trip_equal=data["round_trip_equal"],
            mean_latency_ms=data["mean_latency_ms"],
            notes=list(data.get("notes", [])),
        )


@dataclass(frozen=True)
class CrossDslManifest:
    """Preregistered manifest for the SLM-159 campaign."""

    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = REPLICATION_CAMPAIGN_ID
    hypothesis: str = (
        "Pack-neutral SemanticPlanV1 extraction, seed construction, and oracle-backed "
        "validation transfer from OpenUI to GraphQL and a structurally different second pack."
    )
    falsifier: str = (
        "Plan factors cannot be defined from GraphQL's schema/selection semantics, the seed "
        "builder cannot reproduce schema-valid queries, or the second-pack candidates lack "
        "the grammar/parser/oracle/data contract required for a non-toy replication."
    )
    common_config: CommonConfig = field(default_factory=CommonConfig)
    arms: tuple[PackArm, ...] = ()
    claim_class: str = "wiring"
    status: str = "not_run"

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["common_config"] = self.common_config.to_dict()
        data["arms"] = [arm.to_dict() for arm in self.arms]
        return data

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrossDslManifest":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", REPLICATION_CAMPAIGN_ID),
            hypothesis=data.get("hypothesis", ""),
            falsifier=data.get("falsifier", ""),
            common_config=CommonConfig.from_dict(data.get("common_config", {})),
            arms=tuple(PackArm.from_dict(a) for a in data.get("arms", [])),
            claim_class=data.get("claim_class", "wiring"),
            status=data.get("status", "not_run"),
        )


@dataclass(frozen=True)
class CrossDslReplicationReport:
    """Full fixture report for SLM-159."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    manifest: CrossDslManifest
    readiness_reports: list[PackReadinessReport]
    rows: list[ReplicationRow]
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
            "readiness_reports": [r.to_dict() for r in self.readiness_reports],
            "rows": [row.to_dict() for row in self.rows],
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrossDslReplicationReport":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", REPLICATION_CAMPAIGN_ID),
            run_id=data.get("run_id", "slm159_fixture"),
            status=data.get("status", "fixture"),
            manifest=CrossDslManifest.from_dict(data.get("manifest", {})),
            readiness_reports=[
                PackReadinessReport.from_dict(r) for r in data.get("readiness_reports", [])
            ],
            rows=[ReplicationRow.from_dict(r) for r in data.get("rows", [])],
            version_stamp=data.get("version_stamp", {}),
            claim_class=data.get("claim_class", "wiring"),
        )


class GraphQLSemanticPlanExtractor:
    """Deterministic gold plan extraction from a GraphQL source string.

    Maps GraphQL operation/selection/variable/fragment semantics onto the
    pack-neutral ``SemanticPlanV1`` factors without embedding any OpenUI-specific
    role assumptions.
    """

    def extract(self, program_spec: ProgramSpec, pack: DslPack) -> SemanticPlanV1:
        if pack.pack_id != "graphql":
            raise ValueError(f"expected graphql pack, got {pack.pack_id}")
        source = program_spec.canonical_openui or str(program_spec.ast)
        op, variables, body = _parse_graphql_operation(source)
        selections = _parse_selection_set(body)

        contract_hash = program_spec.contract_id
        if not re.fullmatch(r"^[0-9a-f]{16}$", contract_hash or ""):
            contract_hash = None

        identity = PlanIdentity(
            pack_id=pack.pack_id,
            contract_hash=contract_hash,
            source_program_fingerprint=None,
            prompt_context_hash=None,
            provenance="gold",
        )
        archetype = PlanArchetype(id=op, confidence=1.0)

        role_slots: list[RoleSlot] = []
        symbols: list[PlanSymbol] = [
            PlanSymbol(
                symbol_id=f"${name}",
                semantic_role=var_type,
                allowed_pointer_targets=None,
            )
            for name, var_type in variables.items()
        ]
        bindings: list[PlanBinding] = []
        topology_edges: list[dict[str, Any]] = []
        named_requirements: list[str] = []

        def walk(
            items: list[dict[str, Any]],
            parent_role_id: str | None,
            path: tuple[int | str, ...],
        ) -> None:
            for idx, sel in enumerate(items):
                if sel.get("kind") != "field":
                    continue
                field = sel["name"]
                role_id = "/".join(str(p) for p in (*path, field))
                named_requirements.append(field)
                role_slots.append(
                    RoleSlot(
                        role_id=role_id,
                        component_family=field,
                        required=None,
                        evidence_spans=tuple(
                            f"{arg_name}:{value}" for arg_name, value in sel.get("args", [])
                        )
                        or None,
                    )
                )
                if parent_role_id is not None:
                    topology_edges.append(
                        {
                            "parent_role_id": parent_role_id,
                            "child_role_id": role_id,
                            "relation": "contains",
                        }
                    )
                used_vars = [
                    f"${name}"
                    for _arg_name, value in sel.get("args", [])
                    for name in re.findall(r"\$([A-Za-z_]\w*)", str(value))
                ]
                if used_vars:
                    bindings.append(
                        PlanBinding(
                            role_slot_id=role_id,
                            candidate_symbols=tuple(used_vars),
                            placeholder_fallback=True,
                        )
                    )
                if sel.get("selections"):
                    walk(sel["selections"], role_id, (*path, idx))

        walk(selections, None, ())

        topology = PlanTopology(
            parent_relation_candidates=tuple(topology_edges) or None,
        )
        coverage = PlanCoverage(
            named_requirements_accounted_for=tuple(sorted(set(named_requirements)))
            or None,
            unresolved_requirements=None,
        )
        return SemanticPlanV1(
            identity=identity,
            archetype=archetype,
            role_slots=tuple(role_slots),
            topology=topology,
            symbols=tuple(symbols),
            bindings=tuple(bindings),
            coverage=coverage,
            confidence_calibration=PlanConfidenceCalibration(
                per_factor_confidence={
                    "archetype": 1.0,
                    "role_slots": 1.0,
                    "topology": 1.0,
                    "symbols": 1.0,
                    "bindings": 1.0,
                }
            ),
        )


class GraphQLPlanSeedBuilder:
    """Build a schema-valid GraphQL seed from a pack-neutral plan.

    The builder is a soft preference consumer: it renders the plan's predicted
    structure and re-validates through the GraphQL pack oracle.  It never
    implements its own legality checks.
    """

    def __init__(self, pack: DslPack) -> None:
        self.pack = pack

    def build(self, plan: SemanticPlanV1) -> SeedResult:
        has_actionable = (
            plan.role_slots
            or plan.topology.parent_relation_candidates is not None
            or plan.symbols
        )
        if not has_actionable:
            return SeedResult(seed=None, ok=True, reason="baseline: no actionable plan")

        backend_available = getattr(self.pack.backend, "available", lambda: True)()
        if not backend_available:
            return SeedResult(
                seed=None, ok=False, reason="graphql bridge unavailable"
            )

        role_lookup = {slot.role_id: slot for slot in plan.role_slots}
        children_map: dict[str, list[str]] = {}
        root_roles = set(role_lookup)
        for edge in plan.topology.parent_relation_candidates or ():
            parent = str(edge.get("parent_role_id") or "")
            child = str(edge.get("child_role_id") or "")
            if parent and child:
                children_map.setdefault(parent, []).append(child)
                root_roles.discard(child)
        if len(root_roles) != 1:
            return SeedResult(
                seed=None,
                ok=False,
                reason=f"expected exactly one root role, got {len(root_roles)}",
            )
        root_role_id = next(iter(root_roles))

        var_decls = [
            f"{sym.symbol_id}: {sym.semantic_role or 'String'}"
            for sym in plan.symbols
        ]

        def render(role_id: str, depth: int = 0) -> str:
            slot = role_lookup.get(role_id)
            if slot is None:
                raise ValueError(f"unknown role {role_id}")
            field = slot.component_family or ""
            args: list[str] = []
            for span in slot.evidence_spans or ():
                if ":" in span:
                    args.append(span)
            args_str = f"({', '.join(args)})" if args else ""
            children = children_map.get(role_id, [])
            indent = "  " * depth
            if children:
                inner = "\n".join(render(child, depth + 1) for child in children)
                return f"{indent}{field}{args_str} {{\n{inner}\n{indent}}}"
            return f"{indent}{field}{args_str}"

        selection_set = render(root_role_id, depth=0)
        header = "query"
        if var_decls:
            header = f"query({', '.join(var_decls)})"
        seed_text = f"{header} {{\n{selection_set}\n}}"

        try:
            oracle = self.pack.require("oracle")
            oracle(seed_text)
        except Exception as exc:  # noqa: BLE001
            return SeedResult(seed=None, ok=False, reason=f"oracle rejected seed: {exc}")
        return SeedResult(seed=seed_text, ok=True, reason=None)


def _parse_graphql_operation(source: str) -> tuple[str, dict[str, str], str]:
    """Return (operation, {var: type}, selection-body)."""
    s = source.strip()
    m = re.match(r"\s*(query|mutation|subscription)\b", s, re.IGNORECASE)
    op = m.group(1).lower() if m else "query"
    rest = s[m.end() :] if m else s

    vm = re.match(r"\s*\(([^)]*)\)", rest)
    variables: dict[str, str] = {}
    if vm:
        variables = _parse_var_declarations(vm.group(1))
        rest = rest[vm.end() :]

    body = _extract_braced(rest)
    return op, variables, body


_VAR_DECL_RE = re.compile(r"\$\s*([A-Za-z_]\w*)\s*:\s*([^,\)]+)")


def _parse_var_declarations(text: str) -> dict[str, str]:
    return {
        name.strip(): var_type.strip()
        for name, var_type in _VAR_DECL_RE.findall(text)
    }


def _extract_braced(text: str) -> str:
    """Extract the contents of the first balanced {...} block."""
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return ""


def _split_selections(body: str) -> list[str]:
    """Split a selection-set body into top-level selection segments."""
    items: list[str] = []
    i = 0
    n = len(body)
    while i < n:
        while i < n and body[i] in " \t\n,":
            i += 1
        if i >= n:
            break
        start = i
        brace_depth = 0
        paren_depth = 0
        while i < n:
            ch = body[i]
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1
            elif ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth < 0:
                    break
            elif ch == "," and brace_depth == 0 and paren_depth == 0:
                break
            elif ch in " \t\n" and brace_depth == 0 and paren_depth == 0:
                # A whitespace separator ends the current selection unless the
                # next non-space token is a '{' that belongs to it.
                j = i + 1
                while j < n and body[j] in " \t\n":
                    j += 1
                if j < n and body[j] == "{":
                    i = j
                    continue
                break
            i += 1
        segment = body[start:i].strip()
        if segment:
            items.append(segment)
        i += 1
    return items


def _parse_selection_set(body: str) -> list[dict[str, Any]]:
    """Parse top-level selections inside a selection-set body."""
    return [_parse_selection(segment) for segment in _split_selections(body) if segment.strip()]


def _parse_selection(segment: str) -> dict[str, Any]:
    segment = segment.strip()
    if segment.startswith("..."):
        return {"kind": "fragment", "name": segment[3:].strip()}

    # Field with optional args and optional nested selection set.
    m = re.match(r"^(\w+)(?:\s*\(([^()]*)\))?\s*(.*)$", segment, re.DOTALL)
    if not m:
        return {"kind": "unknown", "raw": segment}
    name = m.group(1)
    args_str = m.group(2) or ""
    rest = m.group(3).strip()
    args = _parse_args(args_str)
    selections: list[dict[str, Any]] = []
    if rest.startswith("{"):
        inner = _extract_braced(rest)
        selections = _parse_selection_set(inner)
    return {"kind": "field", "name": name, "args": args, "selections": selections}


def _parse_args(args_str: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not args_str.strip():
        return out
    for part in args_str.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\w+)\s*:\s*(.+)$", part)
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


def _graphql_contract_id(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _graphql_program_spec(source: str) -> ProgramSpec:
    return ProgramSpec(
        id=f"graphql-fixture-{_graphql_contract_id(source)}",
        ast={},
        canonical_openui=source,
        facts={"dsl": "graphql"},
        contract_id=_graphql_contract_id(source),
        program_family_id="graphql",
        lineage_id="slm159-fixture",
        split_group_id="slm159-fixture",
    )


def build_manifest() -> CrossDslManifest:
    """Return the default SLM-159 fixture manifest."""
    arms = (
        PackArm(
            arm_id="G1_graphql",
            family=PackFamily.GRAPHQL,
            pack_id="graphql",
            name="graphql_replication",
            description=(
                "Replicate SemanticPlanV1 extraction and seed building on the "
                "SLM-43 GraphQL pack, using the schema as the symbol table."
            ),
        ),
        PackArm(
            arm_id="S1_second_pack",
            family=PackFamily.SECOND_PACK,
            pack_id="design-patterns-or-nomenclature",
            name="second_pack_readiness",
            description=(
                "Preregistered second-pack candidate from SLM-44 (design-patterns DSL) "
                "or SLM-45 (expert-nomenclature/ontology pack).  The issue authorizes "
                "recording a concrete readiness blocker if neither pack/oracle contract "
                "is available."
            ),
            promotable=False,
            blocked=True,
            blocker=(
                "No SLM-44 or SLM-45 pack is registered; only design documents exist. "
                "A syntax-only toy pack cannot satisfy the readiness rubric, so the "
                "second-pack replication is blocked pending a real pack implementation."
            ),
        ),
    )
    return CrossDslManifest(arms=arms)


def validate_manifest(manifest: CrossDslManifest) -> list[str]:
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
    cfg = manifest.common_config
    if cfg.n_graphql_records <= 0:
        errors.append("common_config.n_graphql_records must be positive")
    if cfg.graphql_depth < 0:
        errors.append("common_config.graphql_depth must be non-negative")
    return errors


def assess_pack_readiness(pack_id: str) -> PackReadinessReport:
    """Run the issue's pack-readiness rubric for *pack_id*.

    For GraphQL this exercises the registered pack slots.  For the second-pack
    candidates it checks registration and reports the exact missing contract.
    """
    available_packs = set(list_packs())
    pack_available = pack_id in available_packs
    if not pack_available:
        return PackReadinessReport(
            pack_id=pack_id,
            pack_available=False,
            parser_available=False,
            oracle_available=False,
            generator_available=False,
            canonicalizer_available=False,
            placeholder_policy_available=False,
            symbol_table={},
            checklist=ReadinessChecklist(),
            unsupported_features=["pack not registered"],
            readiness_pass=False,
            blocker=f"pack {pack_id!r} is not registered in the DSL pack registry",
        )

    pack = get_pack(pack_id)
    parser_available = bool(pack.backend and getattr(pack.backend, "available", lambda: False)())
    oracle_available = parser_available and pack.oracle is not None
    generator_available = pack.corpus_generator is not None
    canonicalizer_available = pack.canonicalize is not None
    placeholder_policy_available = pack.placeholder_policy is not None

    symbol_table: dict[str, list[str]] = {}
    if parser_available and hasattr(pack.backend, "library_schema"):
        try:
            symbol_table = dict(pack.backend.library_schema().get("types", {}))
        except Exception:  # noqa: BLE001
            symbol_table = {}

    # The GraphQL fixture schema gives us bounded finite cases through the generator.
    has_bounded_finite_cases = generator_available
    checklist = ReadinessChecklist(
        versioned_grammar_parser=parser_available,
        schema_type_scope_oracle=oracle_available,
        exact_legal_action_or_valid_edit_enumeration=False,  # not exposed by pack contract yet
        typed_ast_or_program_spec_generator=generator_available,
        canonical_equivalence_fingerprint=canonicalizer_available,
        placeholder_symbol_binder_policy=placeholder_policy_available,
        request_prompt_contract=generator_available,
        positive_corpus_and_disjoint_eval=False,  # requires train/test split corpus
        pack_native_semantic_metric_and_audit=False,  # not implemented
        bounded_finite_cases=has_bounded_finite_cases,
    )
    readiness_pass = (
        checklist.versioned_grammar_parser
        and checklist.schema_type_scope_oracle
        and checklist.typed_ast_or_program_spec_generator
        and checklist.canonical_equivalence_fingerprint
        and checklist.placeholder_symbol_binder_policy
        and checklist.bounded_finite_cases
    )
    blocker = ""
    if not readiness_pass:
        missing = [k for k, v in asdict(checklist).items() if not v]
        blocker = f"readiness checklist incomplete: {', '.join(missing)}"
    return PackReadinessReport(
        pack_id=pack_id,
        pack_available=True,
        parser_available=parser_available,
        oracle_available=oracle_available,
        generator_available=generator_available,
        canonicalizer_available=canonicalizer_available,
        placeholder_policy_available=placeholder_policy_available,
        symbol_table=symbol_table,
        checklist=checklist,
        unsupported_features=[],
        readiness_pass=readiness_pass,
        blocker=blocker,
    )


def extract_graphql_plan(source: str) -> SemanticPlanV1:
    """Convenience wrapper: extract a gold plan from a GraphQL source string."""
    extractor = GraphQLSemanticPlanExtractor()
    return extractor.extract(_graphql_program_spec(source), get_pack("graphql"))


def build_graphql_seed(plan: SemanticPlanV1) -> SeedResult:
    """Convenience wrapper: build a GraphQL seed from a plan."""
    return GraphQLPlanSeedBuilder(get_pack("graphql")).build(plan)


def _sample_graphql_queries(cfg: CommonConfig, seed: int) -> list[str]:
    """Return a deterministic sample of GraphQL queries from the pack generator."""
    pack = get_pack("graphql")
    generator = pack.require("corpus_generator")
    records = generator(
        root_id=f"slm159-fixture-{seed}",
        depth=cfg.graphql_depth,
        max_records=cfg.max_records_per_root,
    )
    queries = [r.openui for r in records[: cfg.n_graphql_records]]
    # Deterministic shuffle so different seeds see different order.
    rng = __import__("random").Random(seed)
    rng.shuffle(queries)
    return queries[: cfg.n_graphql_records]


def _run_graphql_arm(arm: PackArm, cfg: CommonConfig, seed: int) -> ReplicationRow:
    pack = get_pack(arm.pack_id)
    extractor = GraphQLSemanticPlanExtractor()
    builder = GraphQLPlanSeedBuilder(pack)
    canonicalize = pack.require("canonicalize")

    queries = _sample_graphql_queries(cfg, seed)
    n = len(queries)
    covered = 0
    valid = 0
    round_trip = 0
    start = time.perf_counter()
    notes: list[str] = []

    for source in queries:
        try:
            program_spec = _graphql_program_spec(source)
            plan = extractor.extract(program_spec, pack)
            original_canonical = canonicalize(source)
            covered += 1

            seed_result = builder.build(plan)
            if seed_result.ok and seed_result.seed is not None:
                valid += 1
                seed_canonical = canonicalize(seed_result.seed)
                if seed_canonical == original_canonical:
                    round_trip += 1
        except Exception as exc:  # noqa: BLE001
            notes.append(f"query failed: {exc}")

    wall_ms = (time.perf_counter() - start) * 1000.0
    return ReplicationRow(
        arm_id=arm.arm_id,
        pack_id=arm.pack_id,
        seed=seed,
        promotable=arm.promotable and not arm.blocked,
        n_records=n,
        extraction_coverage=covered / max(1, n),
        seed_validity=valid / max(1, n),
        round_trip_equal=round_trip / max(1, n),
        mean_latency_ms=wall_ms / max(1, n),
        notes=notes or ["fixture GraphQL plan round-trip"],
    )


def _run_second_pack_arm(arm: PackArm, cfg: CommonConfig, seed: int) -> ReplicationRow:
    """Blocked arm: only readiness assessment is recorded."""
    del cfg
    return ReplicationRow(
        arm_id=arm.arm_id,
        pack_id=arm.pack_id,
        seed=seed,
        promotable=False,
        n_records=0,
        extraction_coverage=0.0,
        seed_validity=0.0,
        round_trip_equal=0.0,
        mean_latency_ms=0.0,
        notes=[arm.blocker or "blocked", f"seed={seed}"],
    )


def run_fixture_campaign(
    manifest: CrossDslManifest | None = None,
    *,
    run_id: str = "slm159_fixture",
    output_dir: Path | None = None,
) -> CrossDslReplicationReport:
    """Run the SLM-159 cross-DSL replication fixture campaign."""
    manifest = manifest or build_manifest()
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))

    cfg = manifest.common_config
    rows: list[ReplicationRow] = []
    readiness_reports: list[PackReadinessReport] = []

    # GraphQL readiness assessment.
    readiness_reports.append(assess_pack_readiness("graphql"))

    # Second-pack candidate readiness assessments.
    for candidate in _CANDIDATE_SECOND_PACKS:
        readiness_reports.append(assess_pack_readiness(candidate))

    for arm in manifest.arms:
        if arm.blocked:
            for seed in cfg.seeds:
                rows.append(_run_second_pack_arm(arm, cfg, seed))
            continue
        if arm.pack_id == "graphql":
            for seed in cfg.seeds:
                rows.append(_run_graphql_arm(arm, cfg, seed))
            continue
        # Unknown pack: fail-closed note.
        for seed in cfg.seeds:
            rows.append(
                ReplicationRow(
                    arm_id=arm.arm_id,
                    pack_id=arm.pack_id,
                    seed=seed,
                    promotable=False,
                    n_records=0,
                    extraction_coverage=0.0,
                    seed_validity=0.0,
                    round_trip_equal=0.0,
                    mean_latency_ms=0.0,
                    notes=["unsupported pack arm"],
                )
            )

    report = CrossDslReplicationReport(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=REPLICATION_CAMPAIGN_ID,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        readiness_reports=readiness_reports,
        rows=rows,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm159_cross_dsl_replication",
        ),
        claim_class="wiring",
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm159_cross_dsl_replication_report.json")
    return report


def render_markdown(report: CrossDslReplicationReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-159 (SPV4-01): Cross-DSL semantic-plan replication fixture ({report.run_id})",
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
        "| Arm | Pack | Family | Promotable | Blocked | Description |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in report.manifest.arms:
        lines.append(
            f"| {arm.arm_id} | {arm.pack_id} | {arm.family.value} | "
            f"{arm.promotable} | {arm.blocked} | {arm.description} |"
        )

    lines.extend(
        [
            "",
            "## Pack readiness",
            "",
            "| Pack | Available | Parser | Oracle | Generator | Canonicalizer | Placeholder | Pass | Blocker |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in report.readiness_reports:
        lines.append(
            f"| {r.pack_id} | {r.pack_available} | {r.parser_available} | "
            f"{r.oracle_available} | {r.generator_available} | {r.canonicalizer_available} | "
            f"{r.placeholder_policy_available} | {r.readiness_pass} | {r.blocker or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Arm | Seed | Records | Extraction | Seed validity | Round-trip | Latency ms |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        lines.append(
            f"| {row.arm_id} | {row.seed} | {row.n_records} | "
            f"{row.extraction_coverage:.2f} | {row.seed_validity:.2f} | "
            f"{row.round_trip_equal:.2f} | {row.mean_latency_ms:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "This is a fixture wiring run. GraphQL plan extraction and seed building "
            "are exercised through the pack's own oracle; the second-pack replication "
            "is intentionally blocked until SLM-44 or SLM-45 provides a real "
            "grammar/parser/oracle/data contract. Real claims require pack-native "
            "semantic metrics, causal plan-oracle substitution, learned plan recovery, "
            "and independent ship-gate evaluation.",
            "",
        ]
    )
    return "\n".join(lines)
