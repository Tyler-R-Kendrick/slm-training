"""Metamorphic prompt/program, alpha-renaming, and positive-equivalence generators.

SLM-459 (VCE-007). Every generator here declares which facts a transform is
expected to change and which it is expected to leave invariant, and tags its
output with the same root-family split-lineage scheme
``harnesses.train_data.semantic_counterfactuals`` (SLM-366/DSH2-06) already
established, so derivatives from these generators cannot cross train/eval
splits independently of that owner's cases. This module deliberately reuses
existing certified equivalence primitives rather than inventing new ones:

* alpha-renaming reuses the real ``canonicalize_plan`` renaming logic (only
  live for non-gold-provenance plans -- see ``generate_alpha_rename_case``);
* declaration reordering reuses the ``positive_control_sibling_reorder``
  transform from :mod:`slm_training.data.semantic_contrast.transforms`
  (SLM-448/VCE-006);
* AST rewrite equivalence reuses ``dsl.analysis.optimize.optimize``, which
  already self-certifies via ``semantic_fingerprint`` equality;
* prompt-fact generators reuse ``extract_prompt_requirements`` (SGS-004).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

from slm_training.data.contract import GenerationRequest
from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.semantic_contrast.transforms import generate_transforms
from slm_training.data.semantic_plan.canonicalize import plan_factor_fingerprints
from slm_training.data.semantic_plan.requirements_canonicalize import (
    requirement_fact_fingerprints,
)
from slm_training.data.semantic_plan.requirements_extract import (
    extract_prompt_requirements,
)
from slm_training.data.semantic_plan.seed import PlanSeedBuilder
from slm_training.dsl.analysis.optimize import (
    OptimizeOptions,
    optimize,
    semantic_fingerprint,
)
from slm_training.dsl.pack import DslPack
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

MetamorphicFamily = Literal[
    "alpha_rename_plan_identity",
    "reorder_declarations",
    "prompt_single_fact_edit",
    "prompt_paraphrase_invariant",
    "ast_rewrite_equivalence",
]

_SPLIT_POLICY = RootFamilySplitPolicyV1()


def root_family_for(canonical_source: str, dsl: str) -> str:
    """Root split family for a metamorphic case.

    Identical scheme to
    :func:`slm_training.harnesses.train_data.semantic_counterfactuals.root_family_for`
    (SLM-366/DSH2-06) so cases from both generators land in compatible
    lineage families instead of a second, incompatible identity scheme.
    """
    return content_sha({"dsl": dsl, "canonical_source": canonical_source})


def _split_for(root_family_id: str) -> str:
    return _SPLIT_POLICY.assign(root_family_id)


@dataclass(frozen=True)
class MetamorphicCase:
    """One metamorphic generator's declared change/invariance contract.

    ``expected_changed``/``expected_invariant`` name the facts a consumer
    (typically a test) should assert about the before/after pair recorded in
    ``meta`` -- this class only declares the contract and carries the
    lineage/provenance a generator computed; it does not itself assert
    anything, since assertion requires re-running the pack's own verifier/
    evaluator, which is the caller's responsibility (see
    ``tests/test_data/test_semantic_contrast_metamorphic.py``).
    """

    case_id: str
    family: MetamorphicFamily
    description: str
    expected_changed: tuple[str, ...]
    expected_invariant: tuple[str, ...]
    root_family_id: str
    split: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "description": self.description,
            "expected_changed": list(self.expected_changed),
            "expected_invariant": list(self.expected_invariant),
            "root_family_id": self.root_family_id,
            "split": self.split,
            "meta": self.meta,
        }


def _relabel_plan_payload(
    payload: dict[str, Any],
    role_map: dict[str, str],
    symbol_map: dict[str, str],
) -> dict[str, Any]:
    payload = copy.deepcopy(payload)
    for slot in payload.get("role_slots") or []:
        slot["role_id"] = role_map.get(str(slot.get("role_id")), slot.get("role_id"))
    topology = payload.get("topology") or {}
    for edge in topology.get("parent_relation_candidates") or []:
        edge["parent_role_id"] = role_map.get(
            str(edge.get("parent_role_id")), edge.get("parent_role_id")
        )
        edge["child_role_id"] = role_map.get(
            str(edge.get("child_role_id")), edge.get("child_role_id")
        )
    if topology.get("sibling_order_groups"):
        topology["sibling_order_groups"] = [
            [role_map.get(str(role_id), role_id) for role_id in group]
            for group in topology["sibling_order_groups"]
        ]
    for symbol in payload.get("symbols") or []:
        symbol["symbol_id"] = symbol_map.get(
            str(symbol.get("symbol_id")), symbol.get("symbol_id")
        )
    for binding in payload.get("bindings") or []:
        binding["role_slot_id"] = role_map.get(
            str(binding.get("role_slot_id")), binding.get("role_slot_id")
        )
        if binding.get("candidate_symbols"):
            binding["candidate_symbols"] = [
                symbol_map.get(str(symbol_id), symbol_id)
                for symbol_id in binding["candidate_symbols"]
            ]
    return payload


def generate_alpha_rename_case(
    plan: SemanticPlanV1, pack: DslPack
) -> MetamorphicCase | None:
    """Consistently relabel every role/symbol id; nothing observable may move.

    Only meaningful on a non-gold-provenance plan: ``compile_to_baseline()``
    is true for gold/oracle-only plans, so ``canonicalize_plan`` short-
    circuits and never exercises the real alpha-renaming logic in
    ``data.semantic_plan.canonicalize`` -- see that module's
    ``_symbol_renames``/``_role_renames``. A gold plan is rejected here
    (``None``) rather than silently producing a case that cannot actually
    prove canonical-fingerprint invariance.

    New spellings are derived from each id's declaration *position*, not a
    scramble of the original spelling, so the relative order within any
    ``(component_family, role_id)`` tie group that ``_role_renames`` sorts by
    is preserved and canonical fingerprints still agree after the rename.
    """
    if plan.identity.provenance in ("gold", "oracle_override"):
        return None
    payload = plan.to_dict()
    role_ids = [str(slot["role_id"]) for slot in payload.get("role_slots") or ()]
    symbol_ids = [str(sym["symbol_id"]) for sym in payload.get("symbols") or ()]
    if not role_ids and not symbol_ids:
        return None
    role_map = {rid: f"alt_role_{i:04d}" for i, rid in enumerate(role_ids)}
    symbol_map = {sid: f"alt_sym_{i:04d}" for i, sid in enumerate(symbol_ids)}
    renamed_payload = _relabel_plan_payload(payload, role_map, symbol_map)
    renamed_plan = SemanticPlanV1.from_dict(renamed_payload)

    seed_builder = PlanSeedBuilder(pack)
    before = seed_builder.build(plan)
    after = seed_builder.build(renamed_plan)

    root_family_id = root_family_for(before.seed or "", pack.pack_id)
    return MetamorphicCase(
        case_id=content_sha({"family": "alpha_rename_plan_identity", "plan": payload}),
        family="alpha_rename_plan_identity",
        description=(
            "Relabeled every role_id/symbol_id with fresh, position-derived "
            "spellings. Neither identifier is ever rendered into the "
            "compiled surface (PlanSeedBuilder keys on them but only emits "
            "component_family/content), so the surface and every declared "
            "semantic factor must be unchanged."
        ),
        expected_changed=(),
        expected_invariant=(
            "compiled_surface",
            "meaningful_verdict",
            *plan_factor_fingerprints(plan).keys(),
        ),
        root_family_id=root_family_id,
        split=_split_for(root_family_id),
        meta={
            "role_map": role_map,
            "symbol_map": symbol_map,
            "before_seed": before.seed,
            "after_seed": after.seed,
            "before_plan": payload,
            "after_plan": renamed_payload,
            "before_fingerprints": plan_factor_fingerprints(plan),
            "after_fingerprints": plan_factor_fingerprints(renamed_plan),
        },
    )


def generate_reorder_case(
    plan: SemanticPlanV1, pack: DslPack
) -> MetamorphicCase | None:
    """Wrap the VCE-006 sibling-reorder positive control as a metamorphic case.

    Reuses ``positive_control_sibling_reorder`` from
    :mod:`slm_training.data.semantic_contrast.transforms` instead of adding a
    second reorder implementation. Sibling order is not alpha/order-
    normalized for gold-provenance plans (see that transform's docstring),
    so this changes the declared ``topology`` factor and the compiled
    surface but must still be evaluated as meaningful/valid.
    """
    reorder = next(
        (
            candidate
            for candidate in generate_transforms(plan)
            if candidate.transform_id == "positive_control_sibling_reorder"
        ),
        None,
    )
    if reorder is None:
        return None
    seed_builder = PlanSeedBuilder(pack)
    before = seed_builder.build(plan)
    after = seed_builder.build(reorder.plan)
    root_family_id = root_family_for(before.seed or "", pack.pack_id)
    return MetamorphicCase(
        case_id=content_sha(
            {"family": "reorder_declarations", "plan": plan.to_dict()}
        ),
        family="reorder_declarations",
        description=reorder.description,
        expected_changed=("compiled_surface", "topology"),
        expected_invariant=(
            "meaningful_verdict",
            "role_set",
            "component_family",
            "symbol_role",
            "bindings",
            "coverage",
            "archetype",
        ),
        root_family_id=root_family_id,
        split=_split_for(root_family_id),
        meta={
            "transform_id": reorder.transform_id,
            "before_seed": before.seed,
            "after_seed": after.seed,
        },
    )


def _requirement_fact_triples(requirements: Any) -> set[tuple[str, str, str]]:
    return {
        (fact.factor_family, fact.disposition, fact.statement)
        for fact in requirements.facts
    }


def generate_prompt_single_fact_edit_case(
    base_prompt: str,
    edited_prompt: str,
    *,
    edited_component: str,
    dsl: str = "openui",
) -> MetamorphicCase:
    """Edit a prompt so exactly one explicit requirement fact changes.

    Extracts :class:`PromptSemanticRequirementsV1` for both prompts
    (gold-blind, SGS-004) and records the symmetric difference of facts so a
    consumer can assert every changed fact names ``edited_component`` and no
    other component's facts moved.
    """
    base_reqs = extract_prompt_requirements(GenerationRequest(prompt=base_prompt))
    edited_reqs = extract_prompt_requirements(GenerationRequest(prompt=edited_prompt))
    base_facts = _requirement_fact_triples(base_reqs)
    edited_facts = _requirement_fact_triples(edited_reqs)
    changed = base_facts ^ edited_facts
    changed_mentions_only_target = all(
        edited_component in statement for (_family, _disposition, statement) in changed
    )
    root_family_id = root_family_for(
        content_sha({"base_prompt": base_prompt, "edited_prompt": edited_prompt}), dsl
    )
    return MetamorphicCase(
        case_id=content_sha(
            {
                "family": "prompt_single_fact_edit",
                "base_prompt": base_prompt,
                "edited_prompt": edited_prompt,
            }
        ),
        family="prompt_single_fact_edit",
        description=(
            f"Edited the prompt's {edited_component!r} requirement; every "
            "other extracted requirement fact must be unchanged."
        ),
        expected_changed=(f"prompt_requirements:{edited_component}",),
        expected_invariant=("prompt_requirements:unrelated_facts",),
        root_family_id=root_family_id,
        split=_split_for(root_family_id),
        meta={
            "base_prompt": base_prompt,
            "edited_prompt": edited_prompt,
            "edited_component": edited_component,
            "changed_facts": [list(fact) for fact in sorted(changed)],
            "changed_mentions_only_target": changed_mentions_only_target,
            "base_fact_fingerprints": requirement_fact_fingerprints(base_reqs),
            "edited_fact_fingerprints": requirement_fact_fingerprints(edited_reqs),
        },
    )


def generate_prompt_paraphrase_case(
    base_prompt: str, paraphrase_prompt: str, *, dsl: str = "openui"
) -> MetamorphicCase:
    """Reword a prompt without changing its extracted structured facts.

    ``extract_prompt_requirements`` is a deterministic, rule-based extractor
    (SGS-004), not an NLU system: it is provably invariant to whitespace and
    declaration order, not to arbitrary natural-language rewording. This
    generator is honestly scoped to that guarantee -- callers should pass
    prompts that differ only in ways the extractor's own rules are known to
    be insensitive to (see ``EXTRACTOR_RULES`` in
    ``data.semantic_plan.requirements_extract``), not claim general
    paraphrase understanding.
    """
    base_reqs = extract_prompt_requirements(GenerationRequest(prompt=base_prompt))
    paraphrase_reqs = extract_prompt_requirements(
        GenerationRequest(prompt=paraphrase_prompt)
    )
    root_family_id = root_family_for(
        content_sha({"base_prompt": base_prompt, "paraphrase_prompt": paraphrase_prompt}),
        dsl,
    )
    return MetamorphicCase(
        case_id=content_sha(
            {
                "family": "prompt_paraphrase_invariant",
                "base_prompt": base_prompt,
                "paraphrase_prompt": paraphrase_prompt,
            }
        ),
        family="prompt_paraphrase_invariant",
        description=(
            "Reworded the prompt in a way the extractor's own rules are "
            "insensitive to (whitespace/declaration order); extracted "
            "requirement facts must be unchanged."
        ),
        expected_changed=(),
        expected_invariant=("prompt_requirements_fact_set",),
        root_family_id=root_family_id,
        split=_split_for(root_family_id),
        meta={
            "base_prompt": base_prompt,
            "paraphrase_prompt": paraphrase_prompt,
            "base_fact_fingerprints": requirement_fact_fingerprints(base_reqs),
            "paraphrase_fact_fingerprints": requirement_fact_fingerprints(
                paraphrase_reqs
            ),
        },
    )


def generate_ast_rewrite_equivalence_case(
    source: str, *, dsl: str | None = None
) -> MetamorphicCase | None:
    """Apply the pack-owned default-elision/dead-binding optimizer.

    Reuses :func:`slm_training.dsl.analysis.optimize.optimize` with
    ``flatten_single_child=False``: that option combination is already
    self-certifying (``optimize`` raises ``OptimizeError`` internally if the
    non-flattening rewrite sweep changes ``semantic_fingerprint``), so this
    generator does not invent a new rewrite-certificate scheme -- it is
    exactly the "positive AST/program rewrite certified equivalent" family
    the issue asks for, scoped to the one pack-owned certified rewrite set
    that exists today (SRP-005's broader rewrite-certificate schema is not
    yet implemented).
    """
    result = optimize(source, options=OptimizeOptions(flatten_single_child=False), dsl=dsl)
    if not result.changed:
        return None
    root_family_id = root_family_for(source, dsl or "openui")
    return MetamorphicCase(
        case_id=content_sha({"family": "ast_rewrite_equivalence", "source": source}),
        family="ast_rewrite_equivalence",
        description=(
            "Applied the pack-owned default-elision/dead-binding optimizer "
            f"({dict(result.rewrites)}); semantic_fingerprint equality is "
            "self-certified by optimize()'s own contract."
        ),
        expected_changed=("surface_text",),
        expected_invariant=("semantic_fingerprint", "meaningful_verdict"),
        root_family_id=root_family_id,
        split=_split_for(root_family_id),
        meta={
            "before": source,
            "after": result.source,
            "rewrites": dict(result.rewrites),
            "semantic_fingerprint_before": semantic_fingerprint(source, dsl=dsl),
            "semantic_fingerprint_after": semantic_fingerprint(result.source, dsl=dsl),
        },
    )


__all__ = [
    "MetamorphicCase",
    "MetamorphicFamily",
    "root_family_for",
    "generate_alpha_rename_case",
    "generate_reorder_case",
    "generate_prompt_single_fact_edit_case",
    "generate_prompt_paraphrase_case",
    "generate_ast_rewrite_equivalence_case",
]
