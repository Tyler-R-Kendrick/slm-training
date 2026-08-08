"""Conservative deterministic request → ``PromptSemanticRequirementsV1`` (SGS-004).

Extracts only semantically explicit, authored facts from a
:class:`~slm_training.data.contract.GenerationRequest`. Ambiguous natural-
language interpretation abstains (no hard fact) rather than inventing layout
or style intent. Gold AST / target / reference outputs are never consulted —
the public entrypoint accepts only a ``GenerationRequest`` (plus an optional
component-name inventory for phrase matching).

Facts are advisory (``authority="advisory-learned"``) and must not be wired
into legal-support shrink paths; integration that could affect decode support
is owned by later SGS issues.

Rule IDs are stable and versioned; every emitted fact's ``fact_id`` is
prefixed with its rule id so provenance is recoverable without a parallel
sidecar schema.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping

from slm_training.data.contract import GenerationRequest
from slm_training.data.progspec.prompt_requirements import (
    AmbiguityGroup,
    PromptSemanticRequirementsV1,
    RequirementFact,
)
from slm_training.data.quality import (
    _official_component_names,
    _prompt_component_requirements,
)
from slm_training.data.semantic_plan.requirements_canonicalize import (
    canonicalize_requirements,
)

__all__ = [
    "EXTRACTOR_RULES",
    "ExtractorRule",
    "GoldTargetAccessError",
    "extract_prompt_requirements",
]

# Forbidden kwargs / nested keys that would leak gold/target into extraction.
_GOLD_LEAK_KEYS = frozenset(
    {
        "openui",
        "gold",
        "gold_ast",
        "gold_openui",
        "target",
        "target_ast",
        "reference",
        "reference_ast",
        "program_spec",
        "ast",
        "example_record",
        "record",
    }
)

_EITHER_OR_RE = re.compile(
    r"\beither\s+(?:a |an |the )?"
    r"([A-Za-z][A-Za-z0-9]*)\s+or\s+(?:a |an |the )?"
    r"([A-Za-z][A-Za-z0-9]*)\b",
    re.IGNORECASE,
)

_FORBIDDEN_COMPONENT_RE = re.compile(
    r"(?:\bnot|\bno|\bwithout|\bexclude|\bomit|\bavoid|"
    r"\bdo not (?:use|include|add))\s+(?:a |an |the |any )?"
    r"([A-Za-z][A-Za-z0-9]*)\b",
    re.IGNORECASE,
)


class GoldTargetAccessError(ValueError):
    """Raised when a caller tries to feed gold/target material into the extractor."""


class ExtractorRule:
    """Stable rule identity carried on every fact the rule emits."""

    __slots__ = ("rule_id", "version", "description")

    def __init__(self, rule_id: str, version: str, description: str) -> None:
        self.rule_id = rule_id
        self.version = version
        self.description = description

    @property
    def qualified_id(self) -> str:
        return f"{self.rule_id}.{self.version}"


EXTRACTOR_RULES: Mapping[str, ExtractorRule] = {
    "slot_contract": ExtractorRule(
        "sgs004.slot_contract",
        "v1",
        "Declared slot_contract / template-marker inventory → required binder refs.",
    ),
    "runtime_symbol": ExtractorRule(
        "sgs004.runtime_symbol",
        "v1",
        "Authored RuntimeSymbol declarations → required binder/role facts.",
    ),
    "output_kind": ExtractorRule(
        "sgs004.output_kind",
        "v1",
        "Explicit output_kind / output_category on the request.",
    ),
    "prompt_component_required": ExtractorRule(
        "sgs004.prompt_component_required",
        "v1",
        "Positive component inventory from prompt prose (existing quality helper).",
    ),
    "prompt_component_forbidden": ExtractorRule(
        "sgs004.prompt_component_forbidden",
        "v1",
        "Explicit negated component mentions in prompt prose.",
    ),
    "prompt_either_or": ExtractorRule(
        "sgs004.prompt_either_or",
        "v1",
        "Explicit either/or component alternatives → ambiguity group, not a pick.",
    ),
}


def _prompt_context_hash(request: GenerationRequest) -> str:
    payload = {
        "prompt": request.prompt,
        "slot_contract": list(request.slot_contract),
        "schema": request.schema,
        "design_md": request.design_md,
        "runtime_symbols": [symbol.to_dict() for symbol in request.runtime_symbols],
        "output_kind": request.output_kind,
        "output_category": request.output_category,
    }
    blob = repr(payload).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _reject_gold_kwargs(kwargs: Mapping[str, Any]) -> None:
    leaked = sorted(_GOLD_LEAK_KEYS.intersection(kwargs))
    if leaked:
        raise GoldTargetAccessError(
            f"prompt-requirement extraction refuses gold/target inputs: {leaked}"
        )


def _fact(
    *,
    rule: ExtractorRule,
    suffix: str,
    factor_family: str,
    disposition: str,
    statement: str,
    source_span: str | None,
    confidence: float = 1.0,
) -> RequirementFact:
    return RequirementFact(
        fact_id=f"{rule.qualified_id}:{suffix}",
        factor_family=factor_family,
        disposition=disposition,  # type: ignore[arg-type]
        statement=statement,
        confidence=confidence,
        provenance="retrieved",
        authority="advisory-learned",
        source_span=source_span,
    )


def _slot_contract_facts(request: GenerationRequest) -> list[RequirementFact]:
    rule = EXTRACTOR_RULES["slot_contract"]
    facts: list[RequirementFact] = []
    for surface in request.slot_contract:
        facts.append(
            _fact(
                rule=rule,
                suffix=surface,
                factor_family="binder_reference",
                disposition="required",
                statement=f"declared template marker {surface}",
                source_span=f"slot_contract:{surface}",
            )
        )
    return facts


def _runtime_symbol_facts(request: GenerationRequest) -> list[RequirementFact]:
    rule = EXTRACTOR_RULES["runtime_symbol"]
    facts: list[RequirementFact] = []
    for symbol in request.runtime_symbols:
        facts.append(
            _fact(
                rule=rule,
                suffix=f"{symbol.surface}:{symbol.role}",
                factor_family="binder_reference",
                disposition="required",
                statement=(
                    f"declared runtime symbol {symbol.surface} role={symbol.role}"
                ),
                source_span=f"runtime_symbols:{symbol.surface}",
            )
        )
    return facts


def _output_kind_facts(request: GenerationRequest) -> list[RequirementFact]:
    rule = EXTRACTOR_RULES["output_kind"]
    facts: list[RequirementFact] = []
    if request.output_kind != "document":
        facts.append(
            _fact(
                rule=rule,
                suffix=f"kind:{request.output_kind}",
                factor_family="inventory",
                disposition="required",
                statement=f"output_kind={request.output_kind}",
                source_span="output_kind",
            )
        )
    if request.output_category is not None:
        facts.append(
            _fact(
                rule=rule,
                suffix=f"category:{request.output_category}",
                factor_family="inventory",
                disposition="required",
                statement=f"output_category={request.output_category}",
                source_span="output_category",
            )
        )
    return facts


def _component_inventory(
    component_names: Iterable[str] | None,
) -> frozenset[str]:
    if component_names is None:
        return _official_component_names()
    return frozenset(str(name) for name in component_names)


def _prompt_required_component_facts(
    prompt: str, components: frozenset[str]
) -> list[RequirementFact]:
    """Reuse the existing multiplicity-aware inventory helper (preserve behavior)."""
    rule = EXTRACTOR_RULES["prompt_component_required"]
    # The helper already skips negated/replaced mentions and keeps counts.
    required = _prompt_component_requirements(prompt, preserve_repeated_mentions=False)
    counts: dict[str, int] = {}
    for name in required:
        if name not in components:
            continue
        counts[name] = counts.get(name, 0) + 1
    facts: list[RequirementFact] = []
    for name in sorted(counts):
        count = counts[name]
        statement = (
            f"prompt requires component {name}"
            if count == 1
            else f"prompt requires component {name} count>={count}"
        )
        facts.append(
            _fact(
                rule=rule,
                suffix=f"{name}:ge{count}",
                factor_family="inventory" if count == 1 else "cardinality",
                disposition="required",
                statement=statement,
                source_span=f"prompt:{name}",
            )
        )
    return facts


def _prompt_forbidden_component_facts(
    prompt: str, components: frozenset[str]
) -> list[RequirementFact]:
    rule = EXTRACTOR_RULES["prompt_component_forbidden"]
    facts: list[RequirementFact] = []
    seen: set[str] = set()
    for match in _FORBIDDEN_COMPONENT_RE.finditer(prompt):
        raw = match.group(1)
        # Match inventory names case-sensitively when present; also accept
        # exact PascalCase hits against the official set.
        name = raw if raw in components else None
        if name is None:
            lowered = {c.lower(): c for c in components}
            name = lowered.get(raw.lower())
        if name is None or name in seen:
            continue
        seen.add(name)
        facts.append(
            _fact(
                rule=rule,
                suffix=name,
                factor_family="inventory",
                disposition="forbidden",
                statement=f"prompt forbids component {name}",
                source_span=f"prompt:{match.group(0)}",
            )
        )
    return facts


def _either_or_ambiguity(
    prompt: str, components: frozenset[str]
) -> tuple[list[RequirementFact], list[AmbiguityGroup]]:
    """Record explicit either/or alternatives without resolving them."""
    rule = EXTRACTOR_RULES["prompt_either_or"]
    facts: list[RequirementFact] = []
    groups: list[AmbiguityGroup] = []
    lowered = {c.lower(): c for c in components}
    for index, match in enumerate(_EITHER_OR_RE.finditer(prompt)):
        left_raw, right_raw = match.group(1), match.group(2)
        left = left_raw if left_raw in components else lowered.get(left_raw.lower())
        right = right_raw if right_raw in components else lowered.get(right_raw.lower())
        if left is None or right is None or left == right:
            continue
        left_fact = _fact(
            rule=rule,
            suffix=f"{index}:{left}",
            factor_family="inventory",
            disposition="optional",
            statement=f"prompt offers component alternative {left}",
            source_span=f"prompt:{match.group(0)}",
            confidence=0.5,
        )
        right_fact = _fact(
            rule=rule,
            suffix=f"{index}:{right}",
            factor_family="inventory",
            disposition="optional",
            statement=f"prompt offers component alternative {right}",
            source_span=f"prompt:{match.group(0)}",
            confidence=0.5,
        )
        facts.extend((left_fact, right_fact))
        groups.append(
            AmbiguityGroup(
                group_id=f"{rule.qualified_id}:group:{index}",
                member_fact_ids=(left_fact.fact_id, right_fact.fact_id),
                note="explicit either/or; extractor abstains from choosing",
            )
        )
    return facts, groups


def extract_prompt_requirements(
    request: GenerationRequest,
    *,
    component_names: Iterable[str] | None = None,
    canonicalize: bool = True,
    **kwargs: Any,
) -> PromptSemanticRequirementsV1:
    """Extract advisory prompt-side requirements from a generation request.

    Parameters
    ----------
    request:
        Production-facing inputs only. Must not carry gold AST/target fields.
    component_names:
        Optional closed component inventory for phrase matching. Defaults to
        the official OpenUI component set used by the existing prompt inventory
        helper (behavior preserved).
    canonicalize:
        When True (default), alpha-rename fact/group ids for stable fingerprints.
    """
    _reject_gold_kwargs(kwargs)
    if not isinstance(request, GenerationRequest):
        raise TypeError("extract_prompt_requirements expects a GenerationRequest")

    components = _component_inventory(component_names)
    facts: list[RequirementFact] = []
    facts.extend(_slot_contract_facts(request))
    facts.extend(_runtime_symbol_facts(request))
    facts.extend(_output_kind_facts(request))
    either_facts, groups = _either_or_ambiguity(request.prompt, components)
    ambiguous_components = {
        fact.statement.removeprefix("prompt offers component alternative ").strip()
        for fact in either_facts
        if fact.statement.startswith("prompt offers component alternative ")
    }
    required = _prompt_required_component_facts(request.prompt, components)

    def _required_component_name(fact: RequirementFact) -> str | None:
        # Statements are "prompt requires component NAME" or
        # "prompt requires component NAME count>=N".
        prefix = "prompt requires component "
        if not fact.statement.startswith(prefix):
            return None
        rest = fact.statement[len(prefix) :]
        name = rest.split(" count>=", 1)[0].strip()
        return name or None

    # Either/or alternatives abstain from hard required inventory claims for
    # those exact component names only (never substring-suppress neighbors
    # like Form vs FormControl).
    facts.extend(
        fact
        for fact in required
        if _required_component_name(fact) not in ambiguous_components
    )
    facts.extend(_prompt_forbidden_component_facts(request.prompt, components))
    facts.extend(either_facts)

    # Deduplicate by fact_id (structured + prose can overlap on the same claim).
    by_id: dict[str, RequirementFact] = {}
    for fact in facts:
        by_id.setdefault(fact.fact_id, fact)

    requirements = PromptSemanticRequirementsV1(
        requirements_version="1",
        prompt_context_hash=_prompt_context_hash(request),
        facts=tuple(by_id.values()),
        ambiguity_groups=tuple(groups),
    )
    if canonicalize:
        return canonicalize_requirements(requirements)
    return requirements
