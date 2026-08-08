"""Canonicalize ``PromptSemanticRequirementsV1`` for stable fingerprints.

Mirrors ``canonicalize.py``'s treatment of ``SemanticPlanV1``: declaration
order is structural authority, so ``fact_id``/``group_id`` values are
opaque and get remapped to canonical indices before hashing. Two
requirement sets that differ only by fact/group naming (alpha-renaming)
must fingerprint identically.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from slm_training.data.progspec.prompt_requirements import (
    AmbiguityGroup,
    PromptSemanticRequirementsV1,
    RequirementFact,
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fact_sort_key(fact: RequirementFact) -> tuple[str, str, str, str]:
    return (fact.factor_family, fact.disposition, fact.statement, fact.provenance)


def canonicalize_requirements(
    requirements: PromptSemanticRequirementsV1,
) -> PromptSemanticRequirementsV1:
    """Return requirements with fact/group ids normalized to canonical indices.

    Normalization makes the fingerprint invariant to alpha-renaming of
    ``fact_id``/``group_id`` where the underlying facts are unchanged.
    """
    ordered_facts = sorted(requirements.facts, key=_fact_sort_key)
    fact_renames = {
        fact.fact_id: f"fact_{index:04d}" for index, fact in enumerate(ordered_facts)
    }

    new_facts = tuple(
        fact.model_copy(update={"fact_id": fact_renames[fact.fact_id]})
        for fact in ordered_facts
    )

    renamed_groups = []
    for group in requirements.ambiguity_groups:
        renamed_groups.append(
            (
                tuple(sorted(fact_renames[fid] for fid in group.member_fact_ids)),
                group.note,
            )
        )
    renamed_groups.sort(key=_stable_json)
    new_groups = tuple(
        AmbiguityGroup(
            group_id=f"group_{index:04d}",
            member_fact_ids=members,
            note=note,
        )
        for index, (members, note) in enumerate(renamed_groups)
    )

    return requirements.model_copy(update={"facts": new_facts, "ambiguity_groups": new_groups})


def requirement_fact_fingerprints(
    requirements: PromptSemanticRequirementsV1,
) -> dict[str, str]:
    """SHA-256 fingerprints for the whole set and each constituent part."""
    canonical = canonicalize_requirements(requirements)
    parts = {
        "exact": canonical.model_dump(mode="json"),
        "fact_set": tuple(
            sorted(
                (fact.factor_family, fact.disposition, fact.statement)
                for fact in canonical.facts
            )
        ),
        "authority": tuple(
            sorted((fact.fact_id, fact.authority) for fact in canonical.facts)
        ),
        "ambiguity_groups": tuple(
            sorted(
                _stable_json({"members": group.member_fact_ids, "note": group.note})
                for group in canonical.ambiguity_groups
            )
        ),
    }
    return {name: _sha256(_stable_json(value)) for name, value in parts.items()}
