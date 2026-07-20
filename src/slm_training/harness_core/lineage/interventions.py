"""LDI4-03 unified intervention manifest, registry, and promotion gates (SLM-137).

One common identity / loading / evaluation / lineage / promotion contract for every
intervention kind -- causal PEFT adapter, TwoTower low-rank delta, ReFT representation
intervention, and SAE diagnostic -- so no artifact type may bypass model-compatibility,
immutable evidence, the five-suite ship gates, or the promotion process.

This **consolidates proven owners**: it wraps around each model's existing
``artifact_identity()`` / ``compatibility_fingerprint()`` / ``save_*`` / ``load_*`` and the
per-kind spec fingerprints; it does **not** replace the ``ModelPlugin`` protocol, add a
trainer/scheduler, or invent a new event schema. It is torch-free -- the actual load into
a real model, merge/export parity, dashboard rendering, and bucket upload are deferred to
the follow-on integration.

Design invariants (fail closed): unknown kinds/fields/versions are rejected; base
compatibility (architecture, tokenizer, module/site shapes) is checked before any load;
a diagnostic-only SAE can never be a deployable production intervention; the runtime
supports **one active intervention** (multiple parent ids are provenance only, not
runtime stacking); intervention lineage is explicit and acyclic; missing required
evaluation evidence makes an artifact *ineligible*, never a silent pass; and no scalar
score can promote past a failed protected ship gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from slm_training.harness_core.checkpoint_reference import FileArtifact
from slm_training.harness_core.lineage.records import content_sha

__all__ = [
    "MANIFEST_VERSION",
    "INTERVENTION_KINDS",
    "PromotionStatus",
    "RUN_OUTCOMES",
    "BaseIdentity",
    "InterventionManifest",
    "InterventionError",
    "InterventionRegistry",
    "EvaluationBundle",
    "PROMOTION_TRANSITIONS",
    "promote",
    "detect_lineage_cycle",
    "assert_single_active",
    "build_closeout_index",
]

MANIFEST_VERSION = "ldi4-03-v1"
INTERVENTION_KINDS = ("causal_peft", "twotower_delta", "reft", "sae_diagnostic")
# Kinds whose artifacts are research-only and can never be a production intervention.
_DIAGNOSTIC_ONLY_KINDS = ("sae_diagnostic",)

# Promotion states (deterministic). ``expired``/``stopped``/``blocked_by_corpus``/
# ``no_safe_direction`` are *run outcomes*, not promotion states, and never evidence of
# model failure.
PromotionStatus = str
_STATUSES = ("wiring", "diagnostic", "rejected", "eligible", "promoted")
RUN_OUTCOMES = ("expired", "stopped", "blocked_by_corpus", "no_safe_direction")

PROMOTION_TRANSITIONS: dict[str, frozenset[str]] = {
    "wiring": frozenset({"diagnostic"}),
    "diagnostic": frozenset({"rejected", "eligible"}),
    "eligible": frozenset({"promoted", "rejected"}),
    "rejected": frozenset(),
    "promoted": frozenset(),
}


class InterventionError(ValueError):
    """Raised on an unknown kind/field/version or an out-of-contract manifest."""


@dataclass(frozen=True)
class BaseIdentity:
    """The base a manifest binds to. A load must match every field before mutating."""

    architecture: str
    base_model_id: str
    base_model_revision: str
    tokenizer_sha: str
    base_compatibility_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "tokenizer_sha": self.tokenizer_sha,
            "base_compatibility_fingerprint": self.base_compatibility_fingerprint,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> BaseIdentity:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - known
        if unknown:
            raise InterventionError(f"unknown base identity field(s): {sorted(unknown)}")
        return cls(**dict(data))


@dataclass(frozen=True)
class InterventionManifest:
    """Common identity for one intervention artifact of any kind. Kind-specific config is
    a versioned tagged ``kind_payload``; the common fields are stable across kinds."""

    intervention_id: str
    kind: str
    method: str
    status: str
    deployable: bool
    base: BaseIdentity
    module_site_map: tuple[tuple[str, str], ...]
    parameter_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    trainable_parameter_count: int
    artifact_files: tuple[FileArtifact, ...]
    config_fingerprint: str
    kind_payload: dict[str, Any] = field(default_factory=dict)
    parent_intervention_ids: tuple[str, ...] = ()
    corpus_manifest_sha: str = ""
    action_evidence_sha: str = ""
    objective_materializer_sha: str = ""
    objective_config_sha: str = ""
    training_run_identity: str = ""
    evaluation_bundle_sha: str = ""
    dependency_versions: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    version: str = MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.kind not in INTERVENTION_KINDS:
            raise InterventionError(f"unknown intervention kind {self.kind!r}")
        if self.status not in _STATUSES:
            raise InterventionError(f"unknown promotion status {self.status!r}")
        if self.version != MANIFEST_VERSION:
            raise InterventionError(f"unknown manifest version {self.version!r}")
        if self.kind in _DIAGNOSTIC_ONLY_KINDS and self.deployable:
            raise InterventionError(
                f"{self.kind} is diagnostic-only and cannot be a deployable intervention"
            )
        if self.intervention_id in self.parent_intervention_ids:
            raise InterventionError("intervention lineage must be acyclic (self-parent)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "kind": self.kind,
            "method": self.method,
            "status": self.status,
            "deployable": self.deployable,
            "base": self.base.to_dict(),
            "module_site_map": [list(pair) for pair in self.module_site_map],
            "parameter_shapes": [[n, list(shp)] for n, shp in self.parameter_shapes],
            "trainable_parameter_count": self.trainable_parameter_count,
            "artifact_files": [
                {"name": a.name, "size_bytes": a.size_bytes, "sha256": a.sha256}
                for a in self.artifact_files
            ],
            "config_fingerprint": self.config_fingerprint,
            "kind_payload": self.kind_payload,
            "parent_intervention_ids": list(self.parent_intervention_ids),
            "corpus_manifest_sha": self.corpus_manifest_sha,
            "action_evidence_sha": self.action_evidence_sha,
            "objective_materializer_sha": self.objective_materializer_sha,
            "objective_config_sha": self.objective_config_sha,
            "training_run_identity": self.training_run_identity,
            "evaluation_bundle_sha": self.evaluation_bundle_sha,
            "dependency_versions": self.dependency_versions,
            "created_at": self.created_at,
            "version": self.version,
        }

    def fingerprint(self) -> str:
        return content_sha(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InterventionManifest:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - known
        if unknown:
            raise InterventionError(f"unknown manifest field(s): {sorted(unknown)}")
        kw = dict(data)
        kw["base"] = BaseIdentity.from_mapping(kw["base"]) if not isinstance(kw.get("base"), BaseIdentity) else kw["base"]
        kw["module_site_map"] = tuple(tuple(p) for p in kw.get("module_site_map", ()))
        kw["parameter_shapes"] = tuple((n, tuple(shp)) for n, shp in kw.get("parameter_shapes", ()))
        kw["artifact_files"] = tuple(
            a if isinstance(a, FileArtifact) else FileArtifact(a["name"], a["size_bytes"], a["sha256"])
            for a in kw.get("artifact_files", ())
        )
        kw["parent_intervention_ids"] = tuple(kw.get("parent_intervention_ids", ()))
        return cls(**kw)


# --------------------------------------------------------------------------- #
# Registry: one discoverable validate/inspect surface over the existing owners.
# --------------------------------------------------------------------------- #
Validator = Callable[["InterventionManifest"], list[str]]


def _common_failures(m: InterventionManifest) -> list[str]:
    fails: list[str] = []
    if not m.base.base_compatibility_fingerprint:
        fails.append("missing base_compatibility_fingerprint")
    if not m.config_fingerprint:
        fails.append("missing config_fingerprint")
    if not m.artifact_files:
        fails.append("no content-addressed artifact files")
    for a in m.artifact_files:
        if not a.sha256 or a.sha256 == "UNKNOWN":
            fails.append(f"artifact file {a.name!r} is not content-addressed")
    return fails


def _requires_module_map(m: InterventionManifest) -> list[str]:
    return [] if m.module_site_map else ["missing module/site map"]


def _sae_diagnostic_validator(m: InterventionManifest) -> list[str]:
    fails = _requires_module_map(m)
    if m.deployable:
        fails.append("sae_diagnostic cannot be deployable")
    return fails


_DEFAULT_VALIDATORS: dict[str, Validator] = {
    "causal_peft": _requires_module_map,
    "twotower_delta": _requires_module_map,
    "reft": _requires_module_map,
    "sae_diagnostic": _sae_diagnostic_validator,
}


class InterventionRegistry:
    """A ``kind -> validator`` registry that validates/inspects a manifest without loading
    the base model. It wraps the existing per-kind save/load; it adds no scheduler."""

    def __init__(self) -> None:
        self._validators: dict[str, Validator] = dict(_DEFAULT_VALIDATORS)

    def register(self, kind: str, validator: Validator) -> None:
        if kind not in INTERVENTION_KINDS:
            raise InterventionError(f"cannot register unknown kind {kind!r}")
        self._validators[kind] = validator

    def validate(self, m: InterventionManifest) -> list[str]:
        """Return every validation failure (empty == valid). Fails closed on kinds with no
        registered validator."""
        if m.kind not in self._validators:
            return [f"no validator registered for kind {m.kind!r}"]
        return _common_failures(m) + self._validators[m.kind](m)

    def is_valid(self, m: InterventionManifest) -> bool:
        return not self.validate(m)

    def inspect(self, m: InterventionManifest) -> dict[str, Any]:
        """Inspect an artifact without loading the base model."""
        return {
            "intervention_id": m.intervention_id,
            "kind": m.kind,
            "method": m.method,
            "status": m.status,
            "deployable": m.deployable and m.kind not in _DIAGNOSTIC_ONLY_KINDS,
            "base_compatibility_fingerprint": m.base.base_compatibility_fingerprint,
            "trainable_parameter_count": m.trainable_parameter_count,
            "artifact_shas": [a.sha256 for a in m.artifact_files],
            "parent_intervention_ids": list(m.parent_intervention_ids),
            "valid": self.is_valid(m),
            "failures": self.validate(m),
        }


# --------------------------------------------------------------------------- #
# Standard evaluation bundle (missing required field => ineligible, not pass).
# --------------------------------------------------------------------------- #
_REQUIRED_BUNDLE_FIELDS = {
    "identity": ("base_sha", "intervention_sha", "corpus_sha", "seed", "commit_sha"),
    "event": ("support_summary", "local_objective_metrics"),
    "locality": ("legal_space_drift", "preservation", "disabled_parity"),
    "end_to_end": ("ship_gates", "adversarial", "ood", "agentv"),
}


@dataclass(frozen=True)
class EvaluationBundle:
    """One standard result schema shared by every intervention experiment. Every required
    field must be present; a missing field makes the artifact **ineligible**, never a
    default pass/zero. ``end_to_end.ship_gates`` reuses ``evaluate_ship_gates`` output
    (a ``{"pass": bool, "failures": [...]}`` mapping)."""

    identity: dict[str, Any]
    event: dict[str, Any]
    locality: dict[str, Any]
    end_to_end: dict[str, Any]

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        for group, keys in _REQUIRED_BUNDLE_FIELDS.items():
            present = getattr(self, group)
            for key in keys:
                if key not in present or present[key] is None:
                    missing.append(f"{group}.{key}")
        return missing

    def ship_gates_pass(self) -> bool:
        gates = self.end_to_end.get("ship_gates")
        return bool(gates) and bool(gates.get("pass"))

    def eligible(self) -> bool:
        """Eligible only when complete AND the protected ship gates pass."""
        return not self.missing_fields() and self.ship_gates_pass()

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "event": self.event,
            "locality": self.locality,
            "end_to_end": self.end_to_end,
            "missing_fields": self.missing_fields(),
            "eligible": self.eligible(),
        }

    def sha(self) -> str:
        return content_sha({"identity": self.identity, "event": self.event,
                            "locality": self.locality, "end_to_end": self.end_to_end})


# --------------------------------------------------------------------------- #
# Deterministic promotion state machine.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PromotionResult:
    ok: bool
    status: str
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "status": self.status, "failures": list(self.failures)}


def promote(
    manifest: InterventionManifest,
    to: str,
    *,
    evidence: EvaluationBundle | None = None,
    registry: InterventionRegistry | None = None,
) -> PromotionResult:
    """Attempt a promotion transition. Deterministic and fail-closed: the target must be
    reachable from the current status; ``diagnostic -> eligible`` requires a complete,
    ship-gate-passing evaluation bundle; ``eligible -> promoted`` requires a deployable
    artifact whose ship gates pass. No scalar score overrides a failed protected gate, and
    a run outcome (expired/no_safe_direction/...) is never a promotion target."""
    registry = registry or InterventionRegistry()
    failures: list[str] = []
    if to in RUN_OUTCOMES:
        return PromotionResult(False, manifest.status, (f"{to} is a run outcome, not a promotion status",))
    if to not in _STATUSES:
        return PromotionResult(False, manifest.status, (f"unknown target status {to!r}",))
    allowed = PROMOTION_TRANSITIONS.get(manifest.status, frozenset())
    if to not in allowed:
        return PromotionResult(False, manifest.status, (f"illegal transition {manifest.status} -> {to}",))

    reg_failures = registry.validate(manifest)
    if reg_failures:
        failures.extend(reg_failures)

    if to == "eligible":
        if evidence is None:
            failures.append("diagnostic -> eligible requires an evaluation bundle")
        else:
            failures.extend(f"missing evidence: {m}" for m in evidence.missing_fields())
            if not evidence.ship_gates_pass():
                failures.append("protected ship gate failed; cannot become eligible")
    if to == "promoted":
        if not manifest.deployable or manifest.kind in _DIAGNOSTIC_ONLY_KINDS:
            failures.append("only a deployable (non-diagnostic) artifact can be promoted")
        if evidence is None or not evidence.ship_gates_pass():
            failures.append("promotion requires passing protected ship gates")

    if failures:
        return PromotionResult(False, manifest.status, tuple(failures))
    return PromotionResult(True, to, ())


# --------------------------------------------------------------------------- #
# Lineage-cycle detection + one-active-intervention runtime rule.
# --------------------------------------------------------------------------- #
def detect_lineage_cycle(manifests: Sequence[InterventionManifest]) -> list[str] | None:
    """Return one cycle (as an id list) in the intervention parent graph, or ``None``.
    Parent lineage is provenance and must be acyclic."""
    parents = {m.intervention_id: set(m.parent_intervention_ids) for m in manifests}
    WHITE, GREY, BLACK = 0, 1, 2
    color = {node: WHITE for node in parents}

    def visit(node: str, stack: list[str]) -> list[str] | None:
        color[node] = GREY
        stack.append(node)
        for parent in parents.get(node, ()):
            if parent not in color:
                continue  # external/base parent
            if color[parent] == GREY:
                return stack[stack.index(parent):] + [parent]
            if color[parent] == WHITE:
                found = visit(parent, stack)
                if found:
                    return found
        color[node] = BLACK
        stack.pop()
        return None

    for node in parents:
        if color[node] == WHITE:
            found = visit(node, [])
            if found:
                return found
    return None


def assert_single_active(manifests: Sequence[InterventionManifest]) -> None:
    """The runtime supports exactly one active intervention. Multiple parent ids are
    provenance only; accidental multi-intervention composition is rejected."""
    if len(manifests) > 1:
        raise InterventionError(
            f"one active intervention only; got {len(manifests)} "
            "(adapter routing/merging/composition is a separate future question)"
        )


# --------------------------------------------------------------------------- #
# Program-closeout index.
# --------------------------------------------------------------------------- #
def build_closeout_index(manifests: Sequence[InterventionManifest]) -> dict[str, Any]:
    """A canonical index of every intervention artifact and its status, with the current
    best eligible/promoted artifact -- or an explicit statement that none qualifies."""
    by_status: dict[str, list[str]] = {s: [] for s in _STATUSES}
    for m in manifests:
        by_status[m.status].append(m.intervention_id)
    promoted = by_status["promoted"]
    eligible = by_status["eligible"]
    best = promoted[0] if promoted else (eligible[0] if eligible else None)
    return {
        "version": MANIFEST_VERSION,
        "artifact_count": len(manifests),
        "by_kind": {k: [m.intervention_id for m in manifests if m.kind == k] for k in INTERVENTION_KINDS},
        "by_status": by_status,
        "best_deployable": best,
        "best_deployable_statement": (
            f"best deployable intervention: {best}" if best
            else "no intervention currently qualifies as eligible or promoted"
        ),
        "lineage_cycle": detect_lineage_cycle(manifests),
    }
