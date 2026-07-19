"""SDE4-03 (SLM-181) teacher-paraphrase activation/budget manifest.

This module provides a frozen, versioned, plan-only manifest that must exist
before any teacher spend is authorized. It records the preregistered provider,
budget caps, activation gates, and corpus arms for the H19 teacher-paraphrase
experiment, plus a pack-neutral canonical request renderer derived from the
typed ``ProgramSpec`` target contract.

The slice is intentionally wiring-only: no teacher calls, no model training, and
no ship claim. It only emits the machine-readable activation/budget manifest and
canonical-request contract that later slices must satisfy before spending budget.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.quality import (
    render_semantic_contract_prompt,
    semantic_contract_for_openui,
)

MANIFEST_SCHEMA = "teacher_paraphrase_activation/v1"

ACTIVATION_VERDICTS = frozenset(
    {
        "activation_blocked",
        "ready_to_spend",
        "teacher_paraphrases_not_prioritized",
        "budget_or_yield_blocked",
        "unrun",
    }
)

CAMPAIGN_VERDICTS = frozenset(
    {
        "teacher_paraphrases_improve_language_generalization",
        "deterministic_templates_sufficient",
        "teacher_signal_low_quality",
        "root_not_prompt_limited",
        "budget_or_yield_blocked",
        "inconclusive",
        "unrun",
    }
)

CORPUS_VARIANTS = frozenset(
    {
        "canonical_only",
        "deterministic_templates",
        "teacher_paraphrases",
        "mixed_50_50",
        "teacher_shuffled_target",
        "teacher_low_diversity",
    }
)

PARAPHRASE_STYLES = frozenset(
    {
        "concise",
        "detailed",
        "business_user_story",
        "imperative",
        "multi_constraint",
    }
)

_BINDER_ASSIGNMENT_RE = re.compile(r"(?m)^[a-z_][A-Za-z0-9_]*\s*=")


def _stable_hash(parts: Mapping[str, Any]) -> str:
    """Return a stable 16-hex hash of a JSON-sortable mapping."""
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class TeacherProviderConfig:
    """Pinned external teacher/provider configuration.

    Template hashes are recorded in place of raw prompts so the manifest stays
    compact and the actual prompts can be audited separately.
    """

    provider: str
    model: str
    revision: str | None = None
    system_prompt_template_hash: str | None = None
    user_prompt_template_hash: str | None = None
    sampling_parameters: dict[str, Any] | None = None
    max_tokens: int | None = None
    retry_policy: dict[str, Any] | None = None
    cost_per_1k_input_usd: float | None = None
    cost_per_1k_output_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
        }
        for key in (
            "revision",
            "system_prompt_template_hash",
            "user_prompt_template_hash",
            "sampling_parameters",
            "max_tokens",
            "retry_policy",
            "cost_per_1k_input_usd",
            "cost_per_1k_output_usd",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TeacherProviderConfig":
        return cls(
            provider=str(data["provider"]),
            model=str(data["model"]),
            revision=_optional_str(data.get("revision")),
            system_prompt_template_hash=_optional_str(
                data.get("system_prompt_template_hash")
            ),
            user_prompt_template_hash=_optional_str(
                data.get("user_prompt_template_hash")
            ),
            sampling_parameters=_optional_dict(data.get("sampling_parameters")),
            max_tokens=data.get("max_tokens"),
            retry_policy=_optional_dict(data.get("retry_policy")),
            cost_per_1k_input_usd=data.get("cost_per_1k_input_usd"),
            cost_per_1k_output_usd=data.get("cost_per_1k_output_usd"),
        )


@dataclass(frozen=True)
class ActivationGate:
    """A pre-condition that must be satisfied before teacher spend begins."""

    gate_id: str
    depends_on_issue_id: str
    required_status: str
    available: bool
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "gate_id": self.gate_id,
            "depends_on_issue_id": self.depends_on_issue_id,
            "required_status": self.required_status,
            "available": self.available,
        }
        if self.evidence is not None:
            data["evidence"] = self.evidence
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActivationGate":
        return cls(
            gate_id=str(data["gate_id"]),
            depends_on_issue_id=str(data["depends_on_issue_id"]),
            required_status=str(data["required_status"]),
            available=bool(data["available"]),
            evidence=_optional_str(data.get("evidence")),
        )


@dataclass(frozen=True)
class BudgetCap:
    """Hard monetary and token budget cap for the teacher campaign."""

    max_dollars: float | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key in ("max_dollars", "max_input_tokens", "max_output_tokens"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BudgetCap":
        return cls(
            max_dollars=data.get("max_dollars"),
            max_input_tokens=data.get("max_input_tokens"),
            max_output_tokens=data.get("max_output_tokens"),
        )


@dataclass(frozen=True)
class TeacherParaphraseArm:
    """One preregistered corpus or control arm in the H19 campaign."""

    arm_id: str
    corpus_variant: str
    eligible: bool
    styles: tuple[str, ...] = ()
    omission_reason: str | None = None

    def __post_init__(self) -> None:
        if self.corpus_variant not in CORPUS_VARIANTS:
            raise ValueError(
                f"invalid corpus_variant {self.corpus_variant!r}; "
                f"expected one of {sorted(CORPUS_VARIANTS)}"
            )
        invalid = sorted(set(self.styles) - PARAPHRASE_STYLES)
        if invalid:
            raise ValueError(f"invalid styles {invalid!r}")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "arm_id": self.arm_id,
            "corpus_variant": self.corpus_variant,
            "eligible": self.eligible,
        }
        if self.styles:
            data["styles"] = list(self.styles)
        if self.omission_reason is not None:
            data["omission_reason"] = self.omission_reason
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TeacherParaphraseArm":
        return cls(
            arm_id=str(data["arm_id"]),
            corpus_variant=str(data["corpus_variant"]),
            eligible=bool(data["eligible"]),
            styles=tuple(str(s) for s in data.get("styles") or ()),
            omission_reason=_optional_str(data.get("omission_reason")),
        )


@dataclass(frozen=True)
class TeacherParaphraseActivationManifest:
    """Frozen activation/budget manifest for the H19 teacher-paraphrase experiment."""

    manifest_id: str
    schema_version: str
    hypothesis_id: str
    activation_status: str
    activation_verdict: str
    campaign_verdict: str
    activation_gates: tuple[ActivationGate, ...]
    provider: TeacherProviderConfig
    budget: BudgetCap
    arms: tuple[TeacherParaphraseArm, ...]
    primary_metric: str
    seeds: tuple[int, ...]
    max_derivatives_per_root: int
    manifest_hash: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "hypothesis_id": self.hypothesis_id,
            "activation_status": self.activation_status,
            "activation_verdict": self.activation_verdict,
            "campaign_verdict": self.campaign_verdict,
            "activation_gates": [g.to_dict() for g in self.activation_gates],
            "provider": self.provider.to_dict(),
            "budget": self.budget.to_dict(),
            "arms": [a.to_dict() for a in self.arms],
            "primary_metric": self.primary_metric,
            "seeds": list(self.seeds),
            "max_derivatives_per_root": self.max_derivatives_per_root,
            "manifest_hash": self.manifest_hash,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TeacherParaphraseActivationManifest":
        return cls(
            manifest_id=str(data["manifest_id"]),
            schema_version=str(data.get("schema_version", MANIFEST_SCHEMA)),
            hypothesis_id=str(data["hypothesis_id"]),
            activation_status=str(data["activation_status"]),
            activation_verdict=str(data["activation_verdict"]),
            campaign_verdict=str(data["campaign_verdict"]),
            activation_gates=tuple(
                ActivationGate.from_dict(g) for g in data["activation_gates"]
            ),
            provider=TeacherProviderConfig.from_dict(data["provider"]),
            budget=BudgetCap.from_dict(data["budget"]),
            arms=tuple(TeacherParaphraseArm.from_dict(a) for a in data["arms"]),
            primary_metric=str(data["primary_metric"]),
            seeds=tuple(int(s) for s in data["seeds"]),
            max_derivatives_per_root=int(data["max_derivatives_per_root"]),
            manifest_hash=str(data["manifest_hash"]),
            note=str(data["note"]),
        )


@dataclass(frozen=True)
class CanonicalRequest:
    """A deterministic, pack-neutral language-like request for a typed root."""

    request_text: str
    semantic_contract: dict[str, Any]
    output_kind: str
    request_hash: str
    leakage_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_text": self.request_text,
            "semantic_contract": self.semantic_contract,
            "output_kind": self.output_kind,
            "request_hash": self.request_hash,
            "leakage_flags": list(self.leakage_flags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CanonicalRequest":
        return cls(
            request_text=str(data["request_text"]),
            semantic_contract=dict(data.get("semantic_contract") or {}),
            output_kind=str(data.get("output_kind", "document")),
            request_hash=str(data["request_hash"]),
            leakage_flags=tuple(str(f) for f in data.get("leakage_flags") or ()),
        )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_dict(value: Any) -> dict[str, Any] | None:
    return None if value is None else dict(value)


def _compute_manifest_hash(
    *,
    manifest_id: str,
    hypothesis_id: str,
    activation_gates: Iterable[ActivationGate],
    provider: TeacherProviderConfig,
    budget: BudgetCap,
    arms: Iterable[TeacherParaphraseArm],
    primary_metric: str,
    seeds: Iterable[int],
    max_derivatives_per_root: int,
    note: str,
) -> str:
    gate_hashes = [_stable_hash(g.to_dict()) for g in activation_gates]
    arm_hashes = [_stable_hash(a.to_dict()) for a in arms]
    return _stable_hash(
        {
            "manifest_id": manifest_id,
            "hypothesis_id": hypothesis_id,
            "activation_gates": gate_hashes,
            "provider": _stable_hash(provider.to_dict()),
            "budget": _stable_hash(budget.to_dict()),
            "arms": arm_hashes,
            "primary_metric": primary_metric,
            "seeds": sorted(seeds),
            "max_derivatives_per_root": max_derivatives_per_root,
            "note": note,
        }
    )


def _choose_activation_verdict(
    *,
    slm171_outcome: str | None,
    gates: Iterable[ActivationGate],
    budget: BudgetCap,
) -> tuple[str, str]:
    """Return (activation_status, activation_verdict).

    ``slm171_outcome`` is the result of the SLM-171 diversity-economics issue:
    ``prompt_diversity_limited`` means teacher paraphrases are plausibly useful;
    ``root_diversity_limited`` means the limitation is root coverage, so close as
    ``teacher_paraphrases_not_prioritized``.
    """
    if slm171_outcome == "root_diversity_limited":
        return ("closed", "teacher_paraphrases_not_prioritized")

    unavailable = [g.gate_id for g in gates if not g.available]
    if unavailable:
        return ("blocked", "activation_blocked")

    if slm171_outcome != "prompt_diversity_limited":
        return ("blocked", "activation_blocked")

    if (budget.max_dollars is None or budget.max_dollars <= 0.0) and (
        budget.max_input_tokens is None or budget.max_input_tokens <= 0
    ) and (
        budget.max_output_tokens is None or budget.max_output_tokens <= 0
    ):
        return ("blocked", "budget_or_yield_blocked")

    return ("ready", "ready_to_spend")


def build_teacher_paraphrase_activation_manifest(
    *,
    manifest_id: str,
    activation_gates: Iterable[ActivationGate],
    provider: TeacherProviderConfig,
    budget: BudgetCap,
    arms: Iterable[TeacherParaphraseArm],
    slm171_outcome: str | None = None,
    primary_metric: str = "binding_aware_meaningful_program_rate",
    seeds: Iterable[int] = (0, 1, 2),
    max_derivatives_per_root: int = 5,
    campaign_verdict: str = "unrun",
    note: str = "SDE4-03 teacher-paraphrase activation/budget manifest (wiring slice).",
) -> TeacherParaphraseActivationManifest:
    """Build a deterministic, versioned activation/budget manifest.

    The manifest is plan-only: it does not call the teacher, run training, or
    claim a result. It records the preregistered conditions under which teacher
    spend may begin.
    """
    gate_tuple = tuple(activation_gates)
    arm_tuple = tuple(arms)
    seed_tuple = tuple(seeds)
    activation_status, activation_verdict = _choose_activation_verdict(
        slm171_outcome=slm171_outcome,
        gates=gate_tuple,
        budget=budget,
    )
    manifest_hash = _compute_manifest_hash(
        manifest_id=manifest_id,
        hypothesis_id="H19",
        activation_gates=gate_tuple,
        provider=provider,
        budget=budget,
        arms=arm_tuple,
        primary_metric=primary_metric,
        seeds=seed_tuple,
        max_derivatives_per_root=max_derivatives_per_root,
        note=note,
    )
    return TeacherParaphraseActivationManifest(
        manifest_id=manifest_id,
        schema_version=MANIFEST_SCHEMA,
        hypothesis_id="H19",
        activation_status=activation_status,
        activation_verdict=activation_verdict,
        campaign_verdict=campaign_verdict,
        activation_gates=gate_tuple,
        provider=provider,
        budget=budget,
        arms=arm_tuple,
        primary_metric=primary_metric,
        seeds=seed_tuple,
        max_derivatives_per_root=max_derivatives_per_root,
        manifest_hash=manifest_hash,
        note=note,
    )


def validate_teacher_paraphrase_activation_manifest(
    manifest: Mapping[str, Any],
) -> list[str]:
    """Return validation errors; empty means valid."""
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA!r}")
    if not isinstance(manifest.get("manifest_id"), str) or not manifest.get("manifest_id"):
        errors.append("manifest_id must be a non-empty string")
    if manifest.get("hypothesis_id") != "H19":
        errors.append("hypothesis_id must be 'H19'")

    activation_verdict = manifest.get("activation_verdict")
    if activation_verdict not in ACTIVATION_VERDICTS:
        errors.append(
            f"activation_verdict must be one of {sorted(ACTIVATION_VERDICTS)}"
        )

    campaign_verdict = manifest.get("campaign_verdict")
    if campaign_verdict not in CAMPAIGN_VERDICTS:
        errors.append(
            f"campaign_verdict must be one of {sorted(CAMPAIGN_VERDICTS)}"
        )

    gates = manifest.get("activation_gates")
    if not isinstance(gates, list) or not gates:
        errors.append("activation_gates must be a non-empty list")
    else:
        for idx, gate in enumerate(gates):
            prefix = f"activation_gates[{idx}]"
            if not isinstance(gate, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for key in ("gate_id", "depends_on_issue_id", "required_status"):
                if not gate.get(key):
                    errors.append(f"{prefix} missing or empty {key!r}")
            if "available" not in gate:
                errors.append(f"{prefix} missing 'available'")

    provider = manifest.get("provider")
    if not isinstance(provider, dict):
        errors.append("provider must be an object")
    else:
        for key in ("provider", "model"):
            if not provider.get(key):
                errors.append(f"provider missing or empty {key!r}")

    budget = manifest.get("budget")
    if not isinstance(budget, dict):
        errors.append("budget must be an object")
    else:
        if all(
            budget.get(key) is None
            for key in ("max_dollars", "max_input_tokens", "max_output_tokens")
        ):
            errors.append("budget must specify at least one cap")

    arms = manifest.get("arms")
    if not isinstance(arms, list) or not arms:
        errors.append("arms must be a non-empty list")
        return errors
    for idx, arm in enumerate(arms):
        prefix = f"arms[{idx}]"
        if not isinstance(arm, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if not arm.get("arm_id"):
            errors.append(f"{prefix} missing or empty 'arm_id'")
        variant = arm.get("corpus_variant")
        if variant not in CORPUS_VARIANTS:
            errors.append(
                f"{prefix} corpus_variant {variant!r} not in {sorted(CORPUS_VARIANTS)}"
            )
        if "eligible" not in arm:
            errors.append(f"{prefix} missing 'eligible'")
        elif arm.get("eligible") is False and not arm.get("omission_reason"):
            errors.append(f"{prefix} omitted arm must have omission_reason")
        invalid_styles = sorted(
            set(arm.get("styles") or ()) - PARAPHRASE_STYLES
        )
        if invalid_styles:
            errors.append(f"{prefix} invalid styles {invalid_styles!r}")

    if not isinstance(manifest.get("primary_metric"), str) or not manifest.get("primary_metric"):
        errors.append("primary_metric must be a non-empty string")
    seeds = manifest.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        errors.append("seeds must be a non-empty list")
    if not isinstance(manifest.get("max_derivatives_per_root"), int):
        errors.append("max_derivatives_per_root must be an integer")
    elif manifest["max_derivatives_per_root"] <= 0:
        errors.append("max_derivatives_per_root must be positive")

    return errors


def render_canonical_request(
    spec: ProgramSpec,
    *,
    design_md: str | None = None,
    output_kind: str | None = None,
) -> CanonicalRequest:
    """Render a deterministic, pack-neutral canonical request from ``spec``.

    The renderer uses only the typed target contract: component types, roles,
    cardinalities, placeholders, reference graph, and declared design context.
    It deliberately excludes production token IDs, binder indices, hidden gold
    slot order, compiler-internal states, and the full serialized answer.
    """
    contract = semantic_contract_for_openui(spec.canonical_openui)
    request_text = render_semantic_contract_prompt(contract)
    active_output_kind = output_kind or spec.provenance.get("output_kind") or "document"
    if active_output_kind != "document":
        request_text += f" Output kind: {active_output_kind}."
    if design_md:
        snippet = design_md.strip().replace("\n", " ")[:400]
        request_text += f" Design constraints: {snippet}"

    leakage_flags: list[str] = []
    if spec.canonical_openui in request_text:
        leakage_flags.append("raw_openui")
    if _BINDER_ASSIGNMENT_RE.search(request_text):
        leakage_flags.append("binder_assignment")
    if re.search(r"\broot\s*=", request_text, re.IGNORECASE):
        leakage_flags.append("root_assignment")

    request_hash = _stable_hash(
        {
            "request_text": request_text,
            "semantic_contract": contract,
            "output_kind": active_output_kind,
        }
    )
    return CanonicalRequest(
        request_text=request_text,
        semantic_contract=contract,
        output_kind=active_output_kind,
        request_hash=request_hash,
        leakage_flags=tuple(leakage_flags),
    )


__all__ = [
    "ACTIVATION_VERDICTS",
    "CAMPAIGN_VERDICTS",
    "CORPUS_VARIANTS",
    "PARAPHRASE_STYLES",
    "ActivationGate",
    "BudgetCap",
    "CanonicalRequest",
    "TeacherParaphraseActivationManifest",
    "TeacherParaphraseArm",
    "TeacherProviderConfig",
    "build_teacher_paraphrase_activation_manifest",
    "render_canonical_request",
    "validate_teacher_paraphrase_activation_manifest",
]
