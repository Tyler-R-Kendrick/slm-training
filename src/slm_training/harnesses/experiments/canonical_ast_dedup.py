"""SLM-130 EFS3-05 canonical AST deduplication and valid semantic-mode coverage.

Wiring/fixture harness only. Real measurement requires frozen X22/compiler-tree
 candidate pools from trained checkpoints and a labeled semantic corpus. No ship
claim, no GPU run, no full factorial eval.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, TypeVar

from slm_training.dsl.canonicalize import canonical_fingerprint
from slm_training.dsl.grammar.backends.ast_utils import ast_fingerprint
from slm_training.dsl.parser import validate

__all__ = [
    "ABSTRACT_MODE_SIGNATURE_SCHEMA",
    "CANONICAL_AST_FINGERPRINT_SCHEMA",
    "CANDIDATE_EQUIVALENCE_GROUP_SCHEMA",
    "AbstractModeSignatureV1",
    "CanonicalAstFingerprintV1",
    "CandidateEquivalenceGroupV1",
    "DiversityCoverageReportV1",
    "HardLevel",
    "RepresentativePolicy",
    "build_abstract_mode_signature",
    "build_canonical_ast_fingerprint",
    "compute_diversity_coverage",
    "dedup_arms_for_pool",
    "group_candidates_by_canonical_ast",
    "select_group_representative",
    "unique_slot_truncation",
]

CANONICAL_AST_FINGERPRINT_SCHEMA = "CanonicalAstFingerprintV1"
ABSTRACT_MODE_SIGNATURE_SCHEMA = "AbstractModeSignatureV1"
CANDIDATE_EQUIVALENCE_GROUP_SCHEMA = "CandidateEquivalenceGroupV1"
DIVERSITY_COVERAGE_REPORT_SCHEMA = "DiversityCoverageReportV1"


class RepresentativePolicy(str, Enum):
    FIRST = "first"
    BEST_GENERATOR_SCORE = "best_generator_score"
    BEST_SELECTOR_SCORE = "best_selector_score"
    DETERMINISTIC_LEXICOGRAPHIC = "deterministic_lexicographic"


class HardLevel(str, Enum):
    VALID = "VALID"
    CONTRACT_SATISFIED = "CONTRACT_SATISFIED"
    UNKNOWN = "UNKNOWN"
    INVALID = "INVALID"


T = TypeVar("T")


@dataclass(frozen=True)
class CanonicalAstFingerprintV1:
    """Stable semantic-equivalence fingerprint for an OpenUI program."""

    canonical_fingerprint: str
    structural_fingerprint: str
    dsl_pack: str | None
    canonicalizer_version: str
    parser_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CANONICAL_AST_FINGERPRINT_SCHEMA,
            "canonical_fingerprint": self.canonical_fingerprint,
            "structural_fingerprint": self.structural_fingerprint,
            "dsl_pack": self.dsl_pack,
            "canonicalizer_version": self.canonicalizer_version,
            "parser_version": self.parser_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CanonicalAstFingerprintV1:
        if value.get("schema") != CANONICAL_AST_FINGERPRINT_SCHEMA:
            raise ValueError(f"unsupported fingerprint schema: {value.get('schema')!r}")
        return cls(
            canonical_fingerprint=str(value["canonical_fingerprint"]),
            structural_fingerprint=str(value["structural_fingerprint"]),
            dsl_pack=value.get("dsl_pack"),
            canonicalizer_version=str(value.get("canonicalizer_version", "unknown")),
            parser_version=str(value.get("parser_version", "unknown")),
        )


@dataclass(frozen=True)
class AbstractModeSignatureV1:
    """Diagnostic coarse signature for coverage reporting only.

    Never hard equivalence authority. Collisions against canonical fingerprints
    and semantic reports must be audited.
    """

    signature: str
    normalization_rules: tuple[str, ...]
    component_topology_hash: str
    slot_contract_shape_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ABSTRACT_MODE_SIGNATURE_SCHEMA,
            "signature": self.signature,
            "normalization_rules": list(self.normalization_rules),
            "component_topology_hash": self.component_topology_hash,
            "slot_contract_shape_hash": self.slot_contract_shape_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AbstractModeSignatureV1:
        if value.get("schema") != ABSTRACT_MODE_SIGNATURE_SCHEMA:
            raise ValueError(f"unsupported signature schema: {value.get('schema')!r}")
        return cls(
            signature=str(value["signature"]),
            normalization_rules=tuple(value.get("normalization_rules") or ()),
            component_topology_hash=str(value.get("component_topology_hash", "")),
            slot_contract_shape_hash=str(value.get("slot_contract_shape_hash", "")),
        )


@dataclass(frozen=True)
class CandidateEquivalenceGroupV1:
    """One canonical-AST equivalence class with complete provenance."""

    group_id: str
    canonical_fingerprint: CanonicalAstFingerprintV1
    abstract_mode_signature: AbstractModeSignatureV1 | None
    member_candidate_ids: tuple[str, ...]
    member_generator_scores: tuple[float | None, ...]
    member_selector_scores: tuple[float | None, ...]
    member_hard_levels: tuple[str, ...]
    member_semantic_report_hashes: tuple[str, ...]
    selected_representative_id: str
    representative_policy: RepresentativePolicy
    multiplicity: int
    first_generation_rank: int
    last_generation_rank: int
    has_hard_disagreement: bool
    has_semantic_disagreement: bool

    def __post_init__(self) -> None:
        if self.multiplicity != len(self.member_candidate_ids):
            raise ValueError(
                f"multiplicity {self.multiplicity} != member count "
                f"{len(self.member_candidate_ids)}"
            )
        if self.selected_representative_id not in self.member_candidate_ids:
            raise ValueError("selected_representative_id must be a member")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = dict(asdict(self))
        data["schema"] = CANDIDATE_EQUIVALENCE_GROUP_SCHEMA
        data["canonical_fingerprint"] = self.canonical_fingerprint.to_dict()
        data["abstract_mode_signature"] = (
            self.abstract_mode_signature.to_dict() if self.abstract_mode_signature else None
        )
        data["member_generator_scores"] = list(self.member_generator_scores)
        data["member_selector_scores"] = list(self.member_selector_scores)
        data["member_hard_levels"] = list(self.member_hard_levels)
        data["member_semantic_report_hashes"] = list(self.member_semantic_report_hashes)
        data["representative_policy"] = self.representative_policy.value
        data["member_candidate_ids"] = list(self.member_candidate_ids)
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateEquivalenceGroupV1:
        if value.get("schema") != CANDIDATE_EQUIVALENCE_GROUP_SCHEMA:
            raise ValueError(f"unsupported group schema: {value.get('schema')!r}")
        fp = CanonicalAstFingerprintV1.from_dict(value["canonical_fingerprint"])
        abstract = value.get("abstract_mode_signature")
        return cls(
            group_id=str(value["group_id"]),
            canonical_fingerprint=fp,
            abstract_mode_signature=AbstractModeSignatureV1.from_dict(abstract) if abstract else None,
            member_candidate_ids=tuple(value.get("member_candidate_ids") or ()),
            member_generator_scores=tuple(
                _optional_float(x) for x in value.get("member_generator_scores") or ()
            ),
            member_selector_scores=tuple(
                _optional_float(x) for x in value.get("member_selector_scores") or ()
            ),
            member_hard_levels=tuple(str(x) for x in value.get("member_hard_levels") or ()),
            member_semantic_report_hashes=tuple(
                str(x) for x in value.get("member_semantic_report_hashes") or ()
            ),
            selected_representative_id=str(value["selected_representative_id"]),
            representative_policy=RepresentativePolicy(
                value.get("representative_policy", RepresentativePolicy.FIRST.value)
            ),
            multiplicity=int(value["multiplicity"]),
            first_generation_rank=int(value["first_generation_rank"]),
            last_generation_rank=int(value["last_generation_rank"]),
            has_hard_disagreement=bool(value.get("has_hard_disagreement", False)),
            has_semantic_disagreement=bool(value.get("has_semantic_disagreement", False)),
        )


@dataclass(frozen=True)
class DiversityCoverageReportV1:
    """Within-prompt coverage/diversity numbers for one dedup arm."""

    arm: str
    prompt_hash: str
    pool_size: int
    raw_valid_count: int
    unique_output_strings: int
    unique_canonical_ast: int
    unique_abstract_mode_signatures: int
    duplicate_multiplicity: int
    effective_finalist_occupancy: int
    hard_valid_pass_at_k: float
    semantic_pass_at_k: float | None
    semantic_report_disagreements: int
    canonical_group_hard_disagreements: int
    latency_ms: float | None
    memory_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = dict(asdict(self))
        data["schema"] = DIVERSITY_COVERAGE_REPORT_SCHEMA
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DiversityCoverageReportV1:
        if value.get("schema") != DIVERSITY_COVERAGE_REPORT_SCHEMA:
            raise ValueError(f"unsupported report schema: {value.get('schema')!r}")
        return cls(
            arm=str(value["arm"]),
            prompt_hash=str(value["prompt_hash"]),
            pool_size=int(value["pool_size"]),
            raw_valid_count=int(value["raw_valid_count"]),
            unique_output_strings=int(value["unique_output_strings"]),
            unique_canonical_ast=int(value["unique_canonical_ast"]),
            unique_abstract_mode_signatures=int(value["unique_abstract_mode_signatures"]),
            duplicate_multiplicity=int(value["duplicate_multiplicity"]),
            effective_finalist_occupancy=int(value["effective_finalist_occupancy"]),
            hard_valid_pass_at_k=float(value["hard_valid_pass_at_k"]),
            semantic_pass_at_k=_optional_float(value.get("semantic_pass_at_k")),
            semantic_report_disagreements=int(value.get("semantic_report_disagreements", 0)),
            canonical_group_hard_disagreements=int(
                value.get("canonical_group_hard_disagreements", 0)
            ),
            latency_ms=_optional_float(value.get("latency_ms")),
            memory_bytes=int(value["memory_bytes"]) if value.get("memory_bytes") else None,
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _hard_level(valid: bool, contract_satisfied: bool, unknown: bool = False) -> HardLevel:
    if unknown:
        return HardLevel.UNKNOWN
    if not valid:
        return HardLevel.INVALID
    if contract_satisfied:
        return HardLevel.CONTRACT_SATISFIED
    return HardLevel.VALID


def _semantic_report_hash(report: Mapping[str, Any] | None) -> str:
    if report is None:
        return ""
    canonical = json.dumps(report, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_canonical_ast_fingerprint(
    source: str,
    *,
    dsl: str | None = None,
    parser_version: str = "openui_parser_v1",
    canonicalizer_version: str = "d2_canonicalizer_v1",
) -> CanonicalAstFingerprintV1:
    """Return a versioned canonical-AST fingerprint for ``source``.

    Parses the source, computes the D2 canonical fingerprint, and computes a
    structural component-multiset fingerprint from the parsed AST. Raises if the
    source does not parse.
    """
    program = validate(source, dsl=dsl)
    canonical_fp = canonical_fingerprint(source, dsl=dsl)
    structural = ast_fingerprint(program.root)
    return CanonicalAstFingerprintV1(
        canonical_fingerprint=canonical_fp,
        structural_fingerprint=structural,
        dsl_pack=dsl,
        canonicalizer_version=canonicalizer_version,
        parser_version=parser_version,
    )


def build_abstract_mode_signature(
    source: str,
    *,
    dsl: str | None = None,
    normalize_literals: bool = True,
) -> AbstractModeSignatureV1:
    """Diagnostic abstract-mode signature: collapses literal payloads only.

    This is a coarse coverage signal, never a hard-equivalence key.
    """
    program = validate(source, dsl=dsl)
    root = program.root

    # Walk component topology without literal values.
    topology_parts: list[str] = []
    slot_shape_parts: list[str] = []

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type == "element":
                topology_parts.append(f"{path}:element:{node.get('typeName', '?')}")
                props = node.get("props") or {}
                for prop_name in sorted(props.keys()):
                    _walk(props[prop_name], f"{path}.{prop_name}")
            elif node_type == "binding":
                topology_parts.append(f"{path}:binding:{node.get('kind', '?')}")
                slot_shape_parts.append(f"{path}:binding:{node.get('kind', '?')}")
                _walk(node.get("target"), f"{path}.target")
            elif node_type == "placeholder":
                topology_parts.append(f"{path}:placeholder:{node.get('role', '?')}")
                slot_shape_parts.append(f"{path}:placeholder:{node.get('role', '?')}")
            else:
                for k, v in sorted(node.items()):
                    _walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                _walk(item, f"{path}[{i}]")
        else:
            leaf = "_LITERAL_" if normalize_literals else repr(node)
            topology_parts.append(f"{path}:literal:{leaf}")

    _walk(root, "root")
    topology_hash = hashlib.sha256("\n".join(topology_parts).encode("utf-8")).hexdigest()
    slot_shape_hash = hashlib.sha256("\n".join(slot_shape_parts).encode("utf-8")).hexdigest()
    signature_payload = f"{topology_hash}:{slot_shape_hash}"
    signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()
    rules = (
        "collapse_literal_payloads",
        "preserve_component_topology",
        "preserve_binding_kind",
        "preserve_placeholder_role",
    )
    if normalize_literals:
        rules += ("normalize_literals",)
    return AbstractModeSignatureV1(
        signature=signature,
        normalization_rules=rules,
        component_topology_hash=topology_hash,
        slot_contract_shape_hash=slot_shape_hash,
    )


@dataclass(frozen=True)
class _Member:
    candidate_id: str
    generator_score: float | None
    selector_score: float | None
    hard_level: HardLevel
    semantic_report_hash: str
    generation_rank: int


def group_candidates_by_canonical_ast(
    candidates: Sequence[tuple[str, str, Mapping[str, Any] | None]],
    *,
    dsl: str | None = None,
    policy: RepresentativePolicy = RepresentativePolicy.DETERMINISTIC_LEXICOGRAPHIC,
) -> tuple[CandidateEquivalenceGroupV1, ...]:
    """Group candidates by canonical AST fingerprint.

    ``candidates`` is a sequence of ``(candidate_id, source, context)`` where
    ``context`` may carry ``generator_score``, ``selector_score``,
    ``contract_satisfied``, ``valid``, ``unknown``, and ``semantic_report``.

    Returns deterministic ordered groups by first-seen rank.
    """
    by_fp: dict[str, list[_Member]] = {}
    fingerprints: dict[str, CanonicalAstFingerprintV1] = {}
    abstracts: dict[str, AbstractModeSignatureV1] = {}

    for rank, (candidate_id, source, context) in enumerate(candidates):
        ctx = context or {}
        valid = bool(ctx.get("valid", False))
        contract_satisfied = bool(ctx.get("contract_satisfied", False))
        unknown = bool(ctx.get("unknown", False))
        try:
            fp = build_canonical_ast_fingerprint(source, dsl=dsl)
            abstract = build_abstract_mode_signature(source, dsl=dsl)
        except Exception:
            # Parsing/canonicalization failure: unique invalid-candidate ID.
            invalid_id = f"__invalid__:{candidate_id}"
            fp = CanonicalAstFingerprintV1(
                canonical_fingerprint=invalid_id,
                structural_fingerprint=invalid_id,
                dsl_pack=dsl,
                canonicalizer_version="d2_canonicalizer_v1",
                parser_version="openui_parser_v1",
            )
            abstract = None

        key = fp.canonical_fingerprint
        fingerprints[key] = fp
        if abstract is not None:
            abstracts[key] = abstract
        member = _Member(
            candidate_id=candidate_id,
            generator_score=_optional_float(ctx.get("generator_score")),
            selector_score=_optional_float(ctx.get("selector_score")),
            hard_level=_hard_level(valid, contract_satisfied, unknown),
            semantic_report_hash=_semantic_report_hash(ctx.get("semantic_report")),
            generation_rank=rank,
        )
        by_fp.setdefault(key, []).append(member)

    groups: list[CandidateEquivalenceGroupV1] = []
    for key, members in by_fp.items():
        representative_id = select_group_representative(members, policy=policy)
        hard_hashes = {m.semantic_report_hash for m in members if m.hard_level != HardLevel.INVALID}
        semantic_hashes = {m.semantic_report_hash for m in members}
        groups.append(
            CandidateEquivalenceGroupV1(
                group_id=f"group:{key[:16]}",
                canonical_fingerprint=fingerprints[key],
                abstract_mode_signature=abstracts.get(key),
                member_candidate_ids=tuple(m.candidate_id for m in members),
                member_generator_scores=tuple(m.generator_score for m in members),
                member_selector_scores=tuple(m.selector_score for m in members),
                member_hard_levels=tuple(m.hard_level.value for m in members),
                member_semantic_report_hashes=tuple(m.semantic_report_hash for m in members),
                selected_representative_id=representative_id,
                representative_policy=policy,
                multiplicity=len(members),
                first_generation_rank=min(m.generation_rank for m in members),
                last_generation_rank=max(m.generation_rank for m in members),
                has_hard_disagreement=len(hard_hashes) > 1,
                has_semantic_disagreement=len(semantic_hashes) > 1,
            )
        )
    # Deterministic order by first generation rank.
    groups.sort(key=lambda g: g.first_generation_rank)
    return tuple(groups)


def select_group_representative(
    members: Sequence[_Member],
    *,
    policy: RepresentativePolicy = RepresentativePolicy.DETERMINISTIC_LEXICOGRAPHIC,
) -> str:
    """Pick one representative from an equivalence group.

    Lexicographic policy: CONTRACT_SATISFIED > VALID > UNKNOWN > INVALID, then
    higher generator score, then higher selector score, then lower generation
    rank, then lexicographic candidate_id.
    """
    if not members:
        raise ValueError("empty group")
    if policy is RepresentativePolicy.FIRST:
        return members[0].candidate_id
    if policy is RepresentativePolicy.BEST_GENERATOR_SCORE:
        best = max(
            members,
            key=lambda m: (
                _score_or_neg_inf(m.generator_score),
                -m.generation_rank,
                m.candidate_id,
            ),
        )
        return best.candidate_id
    if policy is RepresentativePolicy.BEST_SELECTOR_SCORE:
        best = max(
            members,
            key=lambda m: (
                _score_or_neg_inf(m.selector_score),
                _score_or_neg_inf(m.generator_score),
                -m.generation_rank,
                m.candidate_id,
            ),
        )
        return best.candidate_id
    # DETERMINISTIC_LEXICOGRAPHIC
    level_order = {
        HardLevel.CONTRACT_SATISFIED: 0,
        HardLevel.VALID: 1,
        HardLevel.UNKNOWN: 2,
        HardLevel.INVALID: 3,
    }
    best = min(
        members,
        key=lambda m: (
            level_order.get(m.hard_level, 4),
            -(_score_or_neg_inf(m.generator_score)),
            -(_score_or_neg_inf(m.selector_score)),
            m.generation_rank,
            m.candidate_id,
        ),
    )
    return best.candidate_id


def _score_or_neg_inf(score: float | None) -> float:
    return float("-inf") if score is None else score


def unique_slot_truncation(
    candidates: Sequence[tuple[str, str, Mapping[str, Any] | None]],
    k: int,
    *,
    dsl: str | None = None,
    policy: RepresentativePolicy = RepresentativePolicy.DETERMINISTIC_LEXICOGRAPHIC,
) -> tuple[str, ...]:
    """Select up to ``k`` finalists with at most one slot per canonical AST group.

    Refills from already-scored next groups without extra generation. Invalid
    candidates are retained in raw traces but never selected.
    """
    groups = group_candidates_by_canonical_ast(candidates, dsl=dsl, policy=policy)
    selected: list[str] = []
    for group in groups:
        if len(selected) >= k:
            break
        if group.selected_representative_id:
            selected.append(group.selected_representative_id)
    return tuple(selected)


def compute_diversity_coverage(
    candidates: Sequence[tuple[str, str, Mapping[str, Any] | None]],
    *,
    arm: str,
    prompt_hash: str,
    dsl: str | None = None,
) -> DiversityCoverageReportV1:
    """Compute within-prompt diversity/coverage numbers."""
    valid_count = 0
    unknown_count = 0
    semantic_success_count = 0
    output_strings = set()
    for candidate_id, source, context in candidates:
        ctx = context or {}
        output_strings.add(source)
        if ctx.get("valid"):
            valid_count += 1
        if ctx.get("unknown"):
            unknown_count += 1
        if ctx.get("semantic_success"):
            semantic_success_count += 1

    groups = group_candidates_by_canonical_ast(candidates, dsl=dsl)
    valid_groups = [g for g in groups if HardLevel(g.member_hard_levels[0]) != HardLevel.INVALID]
    abstract_sigs = {g.abstract_mode_signature.signature for g in groups if g.abstract_mode_signature}
    duplicate_multiplicity = sum(max(0, g.multiplicity - 1) for g in groups)
    hard_disagreements = sum(1 for g in groups if g.has_hard_disagreement)
    semantic_disagreements = sum(1 for g in groups if g.has_semantic_disagreement)
    pool_size = len(candidates)
    pass_at_k = len(valid_groups) / pool_size if pool_size else 0.0
    semantic_pass = semantic_success_count / pool_size if pool_size else None

    return DiversityCoverageReportV1(
        arm=arm,
        prompt_hash=prompt_hash,
        pool_size=pool_size,
        raw_valid_count=valid_count,
        unique_output_strings=len(output_strings),
        unique_canonical_ast=len(groups),
        unique_abstract_mode_signatures=len(abstract_sigs),
        duplicate_multiplicity=duplicate_multiplicity,
        effective_finalist_occupancy=min(len(groups), pool_size),
        hard_valid_pass_at_k=pass_at_k,
        semantic_pass_at_k=semantic_pass,
        semantic_report_disagreements=semantic_disagreements,
        canonical_group_hard_disagreements=hard_disagreements,
        latency_ms=None,
        memory_bytes=None,
    )


def dedup_arms_for_pool(
    candidates: Sequence[tuple[str, str, Mapping[str, Any] | None]],
    *,
    prompt_hash: str,
    dsl: str | None = None,
) -> dict[str, DiversityCoverageReportV1]:
    """Compare the five preregistered dedup arms for a fixed candidate pool.

    Arms:
      A: raw top-K, no dedup (counts raw candidates)
      B: exact output-string dedup
      C: terminal canonical-AST grouping
      D: canonical-AST unique-slot truncation (k = unique canonical groups)
      E: abstract-mode spread diagnostic
    """
    raw = compute_diversity_coverage(candidates, arm="A_raw_no_dedup", prompt_hash=prompt_hash, dsl=dsl)

    # B: exact output-string dedup
    seen_outputs: dict[str, tuple[str, str, Mapping[str, Any] | None]] = {}
    for item in candidates:
        seen_outputs.setdefault(item[1], item)
    output_deduped = tuple(seen_outputs.values())
    string_dedup = compute_diversity_coverage(
        output_deduped, arm="B_exact_output_dedup", prompt_hash=prompt_hash, dsl=dsl
    )

    # C: terminal canonical-AST grouping
    canonical = compute_diversity_coverage(
        candidates, arm="C_terminal_canonical_ast", prompt_hash=prompt_hash, dsl=dsl
    )

    # D: unique-slot truncation with k = number of canonical groups
    groups = group_candidates_by_canonical_ast(candidates, dsl=dsl)
    truncated_ids = unique_slot_truncation(candidates, k=len(groups), dsl=dsl)
    truncated = [item for item in candidates if item[0] in truncated_ids]
    slot_trunc = compute_diversity_coverage(
        truncated, arm="D_unique_slot_truncation", prompt_hash=prompt_hash, dsl=dsl
    )

    # E: abstract-mode spread diagnostic (same as C but counts abstract sigs)
    abstract = compute_diversity_coverage(
        candidates, arm="E_abstract_mode_spread", prompt_hash=prompt_hash, dsl=dsl
    )

    return {
        "A_raw_no_dedup": raw,
        "B_exact_output_dedup": string_dedup,
        "C_terminal_canonical_ast": canonical,
        "D_unique_slot_truncation": slot_trunc,
        "E_abstract_mode_spread": abstract,
    }
