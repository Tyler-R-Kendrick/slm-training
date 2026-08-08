"""VerifiedSynthesisProblemV1 (SGS-007 / SLM-444): the pack-neutral synthesis
request envelope.

Per ``docs/design/repository-ownership-map.md``, no existing contract shapes
a pack-supplied, verifier-checked synthesis problem, so this is a genuinely
new owner layered on three existing ones it must never duplicate:

* ``semantic_plan`` (:mod:`slm_training.data.progspec.semantic_plan``) —
  program-side, AST-derived output. This envelope carries the *request*, not
  a candidate's contents.
* ``verifier_gate_stack`` (:mod:`slm_training.data.verify.stack`) — this
  envelope declares which gates/evaluators a solution must satisfy by
  identity string; it never re-implements gate logic or embeds a verdict.
* ``dsl_pack_registry_canonical`` (:mod:`slm_training.dsl.pack`) — pack
  identity fields here are descriptive strings, not a live registry lookup,
  so this module stays import-light; callers that want registry validation
  do it themselves (see ``tests/test_data/test_progspec/test_synthesis_problem.py``
  for a round trip through both the ``openui`` and ``symbolic_regression``
  packs).

Design invariants (see SLM-444 acceptance criteria):

* ``schema_version`` is explicit; :meth:`VerifiedSynthesisProblemV1.from_dict`
  rejects any other value instead of silently reinterpreting it.
* :data:`SynthesisEvidenceAuthority` excludes ``"compiler-hard"`` and
  ``"verifier-hard"`` (the same authority-tier vocabulary
  :mod:`slm_training.data.progspec.prompt_requirements` uses) -- nothing in
  this envelope can declare itself compiler-certified merely by existing.
  Legality is still proven only by the live compiler/verifier path.
* Prompt-side requirements are carried verbatim as
  :class:`~slm_training.data.progspec.prompt_requirements.PromptSemanticRequirementsV1`
  -- hard/advisory/unknown facts are distinguishable exactly as that
  contract defines them, never re-collapsed here.
* ``search_budget`` bounds resources only; declaring a budget grants no
  search authority and changes no legal-candidate domain by itself.
* Runtime symbols use a pack-neutral ``surface``/``role`` pair rather than
  :class:`slm_training.data.contract.RuntimeSymbol`, whose ``:``/``$``
  surface-prefix rules are OpenUI-specific and would reject, e.g., a bare
  symbolic-regression variable name like ``x``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slm_training.data.progspec.prompt_requirements import (
    PromptSemanticRequirementsV1,
)

__all__ = [
    "SYNTHESIS_PROBLEM_SCHEMA_VERSION",
    "SynthesisEvidenceAuthority",
    "PackIdentityV1",
    "RuntimeSymbolDeclarationV1",
    "VerificationRequirementV1",
    "SearchBudgetV1",
    "ObjectiveTermV1",
    "EvidenceProvenanceV1",
    "VerifiedSynthesisProblemV1",
]

SYNTHESIS_PROBLEM_SCHEMA_VERSION = "1"

# Reserved tiers ("compiler-hard", "verifier-hard") are intentionally absent:
# pydantic rejects them on every from_dict/model_validate call, so nothing
# built from this envelope can smuggle in compiler-certified authority.
SynthesisEvidenceAuthority = Literal[
    "advisory-learned", "oracle-diagnostic", "evaluation-only"
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PackIdentityV1(_StrictModel):
    """DSL pack/contract/grammar/tokenizer/canonicalizer/artifact identities.

    Field names deliberately mirror
    ``slm_training.models.decode_stats.DecodeIdentityV1`` so problem
    authoring and decode telemetry share one identity vocabulary end to end.
    """

    pack_id: str = Field(min_length=1)
    contract_version: int | None = None
    grammar_sha256: str | None = None
    tokenizer_id: str | None = None
    canonicalizer_id: str | None = None
    artifact_schema: str | None = None
    artifact_sha256: str | None = None


class RuntimeSymbolDeclarationV1(_StrictModel):
    """One authored/declared runtime symbol available to the request."""

    surface: str = Field(min_length=1)
    role: str = Field(min_length=1)


class VerificationRequirementV1(_StrictModel):
    """One required verifier gate or evaluator identity a solution must meet."""

    requirement_id: str = Field(min_length=1)
    kind: Literal["gate", "evaluator"]
    gate: str | None = None
    evaluator_version: str | None = None
    evaluator_hash: str | None = None
    mandatory: bool = True

    @model_validator(mode="after")
    def _check_identity_present(self) -> "VerificationRequirementV1":
        if self.kind == "gate" and not self.gate:
            raise ValueError("a 'gate' requirement must set `gate`")
        if self.kind == "evaluator" and not (
            self.evaluator_version or self.evaluator_hash
        ):
            raise ValueError(
                "an 'evaluator' requirement must set evaluator_version or evaluator_hash"
            )
        return self


class SearchBudgetV1(_StrictModel):
    """Bounded resource budget. Declaring a budget grants no search authority
    and changes no legal-candidate domain by itself."""

    max_forwards: int | None = Field(default=None, ge=0)
    max_wall_seconds: float | None = Field(default=None, ge=0.0)
    max_states: int | None = Field(default=None, ge=0)
    max_verifier_calls: int | None = Field(default=None, ge=0)


class ObjectiveTermV1(_StrictModel):
    """One weighted term of the multi-objective vector.

    The objective is always a vector of named terms, never collapsed into a
    single hidden scalar -- callers that want a scalar compute it themselves
    from the declared terms/weights.
    """

    name: str = Field(min_length=1)
    weight: float
    direction: Literal["maximize", "minimize"]


class EvidenceProvenanceV1(_StrictModel):
    """One piece of evidence backing this problem, with its authority tier."""

    source: str = Field(min_length=1)
    authority: SynthesisEvidenceAuthority
    identity: str | None = None


class VerifiedSynthesisProblemV1(_StrictModel):
    """One pack-neutral, versioned synthesis-request envelope."""

    schema_version: str = Field(default=SYNTHESIS_PROBLEM_SCHEMA_VERSION, pattern=r"^1$")
    problem_id: str = Field(min_length=1)
    pack_identity: PackIdentityV1
    authored_context: str = ""
    runtime_symbols: tuple[RuntimeSymbolDeclarationV1, ...] = ()
    requirements: PromptSemanticRequirementsV1 = Field(
        default_factory=PromptSemanticRequirementsV1
    )
    verification_requirements: tuple[VerificationRequirementV1, ...] = ()
    evidence_provenance: tuple[EvidenceProvenanceV1, ...] = ()
    search_budget: SearchBudgetV1 = Field(default_factory=SearchBudgetV1)
    objective: tuple[ObjectiveTermV1, ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_identifiers_unique(self) -> "VerifiedSynthesisProblemV1":
        surfaces = [symbol.surface for symbol in self.runtime_symbols]
        if len(surfaces) != len(set(surfaces)):
            raise ValueError("runtime_symbols surfaces must be unique")
        requirement_ids = [req.requirement_id for req in self.verification_requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("verification_requirements requirement_id values must be unique")
        objective_names = [term.name for term in self.objective]
        if len(objective_names) != len(set(objective_names)):
            raise ValueError("objective term names must be unique")
        return self

    @property
    def digest(self) -> str:
        """Deterministic sha256 identity over the full canonical payload."""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VerifiedSynthesisProblemV1":
        if str(value.get("schema_version")) != SYNTHESIS_PROBLEM_SCHEMA_VERSION:
            raise ValueError("unsupported VerifiedSynthesisProblemV1 version")
        return cls.model_validate(value)
