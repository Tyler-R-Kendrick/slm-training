"""Constraint-based L0-L5 abstraction ladder and grounding checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from slm_training.dsl.lang_core import ParseError, validate


class AbstractionLevel(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"

    @property
    def rank(self) -> int:
        return int(self.value[1:])

    @property
    def family(self) -> str:
        if self.rank <= 2:
            return "frontier_semantic"
        return {
            AbstractionLevel.L3: "frontier_product",
            AbstractionLevel.L4: "frontier_user",
            AbstractionLevel.L5: "frontier_simplified",
        }[self]


_LEVEL_ALIASES = {
    "exact": AbstractionLevel.L0,
    "dsl": AbstractionLevel.L0,
    "ast": AbstractionLevel.L0,
    "semantic": AbstractionLevel.L1,
    "graph": AbstractionLevel.L1,
    "detailed": AbstractionLevel.L2,
    "spec": AbstractionLevel.L2,
    "product": AbstractionLevel.L3,
    "requirements": AbstractionLevel.L3,
    "user": AbstractionLevel.L4,
    "story": AbstractionLevel.L4,
    "simplified": AbstractionLevel.L5,
    "vague": AbstractionLevel.L5,
}


class TargetDeterminacy(str, Enum):
    EXACT = "exact"
    STRUCTURAL = "structural"
    HOUSE_STYLE = "house_style"


def resolve_level(value: str | AbstractionLevel) -> AbstractionLevel:
    if isinstance(value, AbstractionLevel):
        return value
    normalized = str(value).strip()
    try:
        return AbstractionLevel(normalized.upper())
    except ValueError:
        try:
            return _LEVEL_ALIASES[normalized.lower()]
        except KeyError as exc:
            raise ValueError(f"unknown abstraction level: {value!r}") from exc


def _normalized(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


@dataclass(frozen=True)
class FactContract:
    required_facts: tuple[str, ...] = ()
    optional_facts: tuple[str, ...] = ()
    forbidden_facts: tuple[str, ...] = ()
    unspecified_dimensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "required_facts",
            "optional_facts",
            "forbidden_facts",
            "unspecified_dimensions",
        ):
            object.__setattr__(self, field_name, _normalized(getattr(self, field_name)))
        required = set(self.required_facts)
        optional = set(self.optional_facts)
        forbidden = set(self.forbidden_facts)
        if required & optional or required & forbidden or optional & forbidden:
            raise ValueError("required, optional, and forbidden facts must be disjoint")

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "required_facts": list(self.required_facts),
            "optional_facts": list(self.optional_facts),
            "forbidden_facts": list(self.forbidden_facts),
            "unspecified_dimensions": list(self.unspecified_dimensions),
        }


@dataclass(frozen=True)
class GroundingIssue:
    code: str
    fact: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        data = {"code": self.code}
        if self.fact is not None:
            data["fact"] = self.fact
        if self.detail:
            data["detail"] = self.detail
        return data


@dataclass(frozen=True)
class GroundingReport:
    issues: tuple[GroundingIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues


class GroundingError(ValueError):
    def __init__(self, report: GroundingReport) -> None:
        self.report = report
        super().__init__("; ".join(issue.code for issue in report.issues))


@dataclass(frozen=True)
class LadderRung:
    level: AbstractionLevel
    description: str
    contract: FactContract
    constraint_coverage: float
    target_determinacy: TargetDeterminacy
    grounding: GroundingReport

    @property
    def family(self) -> str:
        return self.level.family

    def to_meta(self) -> dict[str, object]:
        return {
            "abstraction_level": self.level.value,
            **self.contract.to_dict(),
            "constraint_coverage": self.constraint_coverage,
            "target_determinacy": self.target_determinacy.value,
            "grounding": {
                "ok": self.grounding.ok,
                "issues": [issue.to_dict() for issue in self.grounding.issues],
            },
        }


_COMPONENT_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*)\s*\(")
_DIRECTION_RE = re.compile(r'\bStack\([^\n]*?,\s*"(row|column)"')
_GAP_RE = re.compile(
    r'\bStack\([^\n]*?,\s*"(?:row|column)"\s*,\s*"(none|xs|s|m|l|xl|2xl)"'
)
_DSL_LEAK_RE = re.compile(
    r"(?m)\broot\s*=|^[a-z_][A-Za-z0-9_]*\s*=|\b[A-Z][A-Za-z0-9]*\s*\(|"
    r":[A-Za-z_][A-Za-z0-9_.]*"
)


def infer_target_facts(target: str) -> tuple[str, ...]:
    """Extract stable component/layout/spacing facts from a valid target."""
    program = validate(target)
    canonical = program.serialized or target
    facts = {f"component:{name}" for name in _COMPONENT_RE.findall(canonical)}
    direction = _DIRECTION_RE.search(canonical)
    if direction:
        facts.add(f"layout:{direction.group(1)}")
    gap = _GAP_RE.search(canonical)
    if gap:
        facts.add(f"spacing:{gap.group(1)}")
    elif "Stack(" in canonical:
        facts.add("spacing:m")
    return tuple(sorted(facts))


def _camel_words(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", value).lower()


def _fact_terms(fact: str) -> tuple[str, ...]:
    kind, _, value = fact.partition(":")
    base = _camel_words(value.replace("_", " ")).strip()
    terms = {base}
    if kind == "layout" and value == "row":
        terms.update({"horizontal", "beside", "side by side"})
    elif kind == "layout" and value == "column":
        terms.update({"vertical", "stacked", "below"})
    elif kind == "component":
        terms.add(value.lower())
        if value.endswith("Chart"):
            terms.add(_camel_words(value.removesuffix("Chart")) + " chart")
    return tuple(sorted(term for term in terms if term))


def _mentions(description: str, fact: str) -> bool:
    lowered = description.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in _fact_terms(fact))


def _excludes(description: str, fact: str) -> bool:
    lowered = description.lower()
    return any(
        re.search(rf"\b(?:without|no|exclude|avoid)\b[^.]*\b{re.escape(term)}\b", lowered)
        for term in _fact_terms(fact)
    )


def _target_has(target_facts: set[str], fact: str) -> bool:
    return fact in target_facts


def check_grounding(
    description: str,
    target: str,
    contract: FactContract,
    *,
    allow_dsl: bool = False,
) -> GroundingReport:
    """Compare fact record, natural-language description, and valid target."""
    issues: list[GroundingIssue] = []
    if not allow_dsl and _DSL_LEAK_RE.search(description):
        issues.append(GroundingIssue("dsl_leak", detail="description exposes DSL or placeholder syntax"))
    try:
        target_facts = set(infer_target_facts(target))
    except (ParseError, RuntimeError, ValueError) as exc:
        return GroundingReport(
            tuple(issues + [GroundingIssue("invalid_target", detail=str(exc).splitlines()[0][:200])])
        )
    for fact in contract.required_facts:
        if not _mentions(description, fact) and not allow_dsl:
            issues.append(GroundingIssue("required_fact_omitted", fact))
        if not _target_has(target_facts, fact):
            issues.append(GroundingIssue("target_missing_required_fact", fact))
    for fact in contract.forbidden_facts:
        if _mentions(description, fact) and not _excludes(description, fact):
            issues.append(GroundingIssue("forbidden_fact_invented", fact))
        if _target_has(target_facts, fact):
            issues.append(GroundingIssue("target_contains_forbidden_fact", fact))
    return GroundingReport(tuple(issues))


def _determinacy(level: AbstractionLevel) -> TargetDeterminacy:
    if level is AbstractionLevel.L0:
        return TargetDeterminacy.EXACT
    if level.rank <= 2:
        return TargetDeterminacy.STRUCTURAL
    return TargetDeterminacy.HOUSE_STYLE


def _infer_contract(
    level: AbstractionLevel, description: str, target_facts: tuple[str, ...]
) -> FactContract:
    if level is AbstractionLevel.L0:
        return FactContract(required_facts=target_facts)
    mentioned = tuple(fact for fact in target_facts if _mentions(description, fact))
    unmentioned = tuple(fact for fact in target_facts if fact not in mentioned)
    dimensions = tuple(fact.partition(":")[0] for fact in unmentioned)
    return FactContract(
        required_facts=mentioned,
        optional_facts=unmentioned if level.rank <= 2 else (),
        unspecified_dimensions=dimensions,
    )


def build_rung(
    level: str | AbstractionLevel,
    description: str,
    target: str,
    *,
    contract: FactContract | None = None,
) -> LadderRung:
    resolved = resolve_level(level)
    target_facts = infer_target_facts(target)
    active_contract = contract or _infer_contract(resolved, description, target_facts)
    covered = sum(_mentions(description, fact) for fact in target_facts)
    coverage = (
        1.0
        if resolved is AbstractionLevel.L0 or not target_facts
        else round(covered / len(target_facts), 6)
    )
    report = check_grounding(
        description,
        target,
        active_contract,
        allow_dsl=resolved is AbstractionLevel.L0,
    )
    if not report.ok:
        raise GroundingError(report)
    return LadderRung(
        level=resolved,
        description=description,
        contract=active_contract,
        constraint_coverage=coverage,
        target_determinacy=_determinacy(resolved),
        grounding=report,
    )


@dataclass(frozen=True)
class CounterfactualPair:
    left: LadderRung
    right: LadderRung
    changed_from: str
    changed_to: str


def make_counterfactual_pair(left: LadderRung, right: LadderRung) -> CounterfactualPair:
    """Accept only a one-fact substitution with all other constraints unchanged."""
    if left.level is not right.level:
        raise ValueError("counterfactual levels must match")
    left_required = set(left.contract.required_facts)
    right_required = set(right.contract.required_facts)
    removed = left_required - right_required
    added = right_required - left_required
    if len(removed) != 1 or len(added) != 1:
        raise ValueError("counterfactual pair must substitute exactly one required fact")
    if (
        left.contract.optional_facts != right.contract.optional_facts
        or left.contract.forbidden_facts != right.contract.forbidden_facts
        or left.contract.unspecified_dimensions != right.contract.unspecified_dimensions
    ):
        raise ValueError("counterfactual pair changed more than one constraint axis")
    return CounterfactualPair(left, right, removed.pop(), added.pop())
