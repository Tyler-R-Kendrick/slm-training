"""Prompt-side partial semantic requirements (SGS-003 / SLM-437).

``SemanticPlanV1`` (``slm_training.data.progspec.semantic_plan``) is the
program-side output contract: it describes what an AST-derived candidate
*contains*. This module is the adjacent, intentionally separate contract for
what a natural-language request *appears to require* before any candidate
exists. Merging the two would let a compiler-derived gold plan masquerade as
an unverified prompt-side prediction (a leakage risk), so
``docs/design/repository-ownership-map.md`` authorizes this as a new,
adjacent owner rather than an extension of ``SemanticPlanV1``.

Design invariants (see SLM-437 acceptance criteria):

* Unknown/unspecified requirement facts never collapse to false/forbidden --
  ``"unspecified"`` is a distinct, first-class disposition.
* Conflicting interpretations of the same request coexist as an
  :class:`AmbiguityGroup` rather than being silently resolved.
* ``RequirementProvenance`` has no ``"gold"`` value: an AST-derived fact
  cannot be represented as a prompt-derived one at all.
* ``RequirementAuthority`` excludes ``"compiler-hard"`` and
  ``"verifier-hard"`` (see the ownership map's authority-tier vocabulary) --
  predicted/retrieved/oracle evidence can never deserialize as either, since
  pydantic rejects unknown ``Literal`` values on every ``from_dict``/
  ``model_validate`` call.
* :func:`combine_authority` and :meth:`PromptSemanticRequirementsV1.merged_with`
  only ever pick the *weaker* of two input tiers, so copying or merging
  requirement facts can never escalate authority.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "REQUIREMENTS_SCHEMA_VERSION",
    "RequirementAuthority",
    "RequirementDisposition",
    "RequirementProvenance",
    "RequirementFact",
    "AmbiguityGroup",
    "PromptSemanticRequirementsV1",
    "combine_authority",
]

REQUIREMENTS_SCHEMA_VERSION = "1"

RequirementDisposition = Literal[
    "required", "forbidden", "preferred", "optional", "unspecified"
]

# Provenance for a prompt-side fact. There is deliberately no "gold" value:
# gold plans are AST-derived (SemanticPlanV1.identity.provenance) and must
# never be representable as a prompt-derived requirement.
RequirementProvenance = Literal["predicted", "retrieved", "oracle_override", "merged"]

# The authority tiers this contract may ever emit. "compiler-hard" and
# "verifier-hard" (docs/design/repository-ownership-map.md) are reserved for
# compiled/verified output and are intentionally absent from this Literal --
# any dict claiming one fails pydantic validation on from_dict/model_validate.
RequirementAuthority = Literal["advisory-learned", "oracle-diagnostic", "evaluation-only"]

_AUTHORITY_RANK: dict[str, int] = {
    "evaluation-only": 0,
    "oracle-diagnostic": 1,
    "advisory-learned": 2,
}


def combine_authority(a: str, b: str) -> str:
    """Return the weaker (lower-ranked) of two requirement authorities.

    Merging or copying facts from two sources must never let the result
    claim a stronger tier than either source alone justified.
    """
    return a if _AUTHORITY_RANK[a] <= _AUTHORITY_RANK[b] else b


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RequirementFact(_StrictModel):
    """One prompt-derived fact about a single factor of the requested program."""

    fact_id: str = Field(min_length=1)
    # Free-text factor family. Where the fact concerns a SemanticPlanV1
    # program factor, reuse slm_training.models.semantic_plan_factors
    # .PROGRAM_FACTOR_FAMILIES values so downstream projections align; this
    # field stays a plain string (not that Literal) because prompt-side
    # requirements can name factors SemanticPlanV1 has no slot for yet.
    factor_family: str = Field(min_length=1)
    disposition: RequirementDisposition
    statement: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: RequirementProvenance
    authority: RequirementAuthority
    source_span: str | None = None


class AmbiguityGroup(_StrictModel):
    """Two or more facts that are mutually exclusive interpretations.

    Membership alone records the ambiguity; nothing here resolves it, so a
    caller cannot lose one interpretation by silently overwriting another.
    """

    group_id: str = Field(min_length=1)
    member_fact_ids: tuple[str, ...] = Field(min_length=2)
    note: str | None = None


class PromptSemanticRequirementsV1(_StrictModel):
    """Pack-neutral partial requirements extracted from a request prompt."""

    requirements_version: str = Field(default="1", pattern=r"^1$")
    prompt_context_hash: str | None = None
    facts: tuple[RequirementFact, ...] = ()
    ambiguity_groups: tuple[AmbiguityGroup, ...] = ()

    @model_validator(mode="after")
    def check_fact_and_group_integrity(self) -> "PromptSemanticRequirementsV1":
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("fact_id values must be unique")
        known = set(fact_ids)
        group_ids = [group.group_id for group in self.ambiguity_groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("ambiguity group_id values must be unique")
        for group in self.ambiguity_groups:
            unknown = set(group.member_fact_ids) - known
            if unknown:
                raise ValueError(
                    f"ambiguity group references unknown fact ids: {sorted(unknown)}"
                )
        return self

    def facts_by_disposition(
        self, disposition: RequirementDisposition
    ) -> tuple[RequirementFact, ...]:
        return tuple(fact for fact in self.facts if fact.disposition == disposition)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PromptSemanticRequirementsV1":
        if str(value.get("requirements_version")) != REQUIREMENTS_SCHEMA_VERSION:
            raise ValueError("unsupported PromptSemanticRequirementsV1 version")
        return cls.model_validate(value)

    def merged_with(
        self, other: "PromptSemanticRequirementsV1"
    ) -> "PromptSemanticRequirementsV1":
        """Combine two requirement sets, never escalating any fact's authority.

        Facts present in both keep ``self``'s disposition/statement/span but
        take ``provenance="merged"`` and the weaker of the two authorities.
        Facts unique to either side are carried over with their original
        provenance/authority untouched -- only a ``fact_id`` present on both
        sides is rewritten to ``provenance="merged"`` with the weaker of the
        two authorities, since only that fact's claim now depends on both
        sources agreeing not to contradict. Ambiguity groups from both sides
        are unioned by ``group_id``.
        """
        other_by_id = {fact.fact_id: fact for fact in other.facts}
        merged_facts: list[RequirementFact] = []
        seen: set[str] = set()
        for fact in self.facts:
            seen.add(fact.fact_id)
            partner = other_by_id.get(fact.fact_id)
            if partner is None:
                merged_facts.append(fact)
                continue
            merged_facts.append(
                fact.model_copy(
                    update={
                        "provenance": "merged",
                        "authority": combine_authority(fact.authority, partner.authority),
                        "confidence": min(fact.confidence, partner.confidence),
                    }
                )
            )
        for fact in other.facts:
            if fact.fact_id in seen:
                continue
            merged_facts.append(fact)

        merged_groups: dict[str, AmbiguityGroup] = {
            group.group_id: group for group in self.ambiguity_groups
        }
        for group in other.ambiguity_groups:
            merged_groups.setdefault(group.group_id, group)

        return PromptSemanticRequirementsV1(
            requirements_version=self.requirements_version,
            prompt_context_hash=self.prompt_context_hash,
            facts=tuple(merged_facts),
            ambiguity_groups=tuple(merged_groups.values()),
        )
