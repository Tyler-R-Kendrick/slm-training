"""Stable design-system defaults for underdetermined L3-L5 prompts."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from slm_training.data.ladder import AbstractionLevel, infer_target_facts, resolve_level
from slm_training.dsl.lang_core import validate


@dataclass(frozen=True)
class HouseStylePolicy:
    layout_direction: str = "column"
    spacing: str = "m"
    responsive_policy: str = "stack_on_narrow"
    loading_policy: str = "skeleton_preserves_layout"
    error_policy: str = "inline_recoverable"
    content_policy: str = "placeholder_only"
    preferred_components: tuple[str, ...] = ("Stack", "Card", "TextContent", "Button")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_HOUSE_STYLE = HouseStylePolicy()


@dataclass(frozen=True)
class CandidateScore:
    target: str
    violations: tuple[str, ...]
    component_rank: int

    @property
    def rank(self) -> tuple[int, int, str]:
        return len(self.violations), self.component_rank, self.target


@dataclass(frozen=True)
class HouseStyleResolution:
    prompt: str
    level: AbstractionLevel
    target: str
    scores: tuple[CandidateScore, ...]
    policy: HouseStylePolicy

    def to_meta(self) -> dict[str, object]:
        winner = next(score for score in self.scores if score.target == self.target)
        return {
            "level": self.level.value,
            "candidate_count": len(self.scores),
            "winning_violations": list(winner.violations),
            "policy": self.policy.to_dict(),
        }


_INTENT_COMPONENTS = {
    "bar chart": "BarChart",
    "line chart": "LineChart",
    "table": "Table",
    "form": "Form",
    "modal": "Modal",
    "tabs": "Tabs",
}


def _canonical(target: str) -> str:
    program = validate(target)
    return program.serialized or target.strip()


def _score(prompt: str, target: str, policy: HouseStylePolicy) -> CandidateScore:
    lowered = prompt.lower()
    facts = set(infer_target_facts(target))
    violations: list[str] = []
    direction_is_specified = bool(
        re.search(r"\b(row|column|horizontal|vertical|beside|stacked)\b", lowered)
    )
    if not direction_is_specified and f"layout:{policy.layout_direction}" not in facts:
        violations.append("layout_default")
    if f"spacing:{policy.spacing}" not in facts:
        violations.append("spacing_default")
    for phrase, component in _INTENT_COMPONENTS.items():
        if phrase in lowered and f"component:{component}" not in facts:
            violations.append(f"missing_intent_component:{component}")
    component_rank = min(
        (
            index
            for index, component in enumerate(policy.preferred_components)
            if f"component:{component}" in facts and component != "Stack"
        ),
        default=len(policy.preferred_components),
    )
    return CandidateScore(target=target, violations=tuple(violations), component_rank=component_rank)


def resolve_target(
    prompt: str,
    candidates: tuple[str, ...],
    level: str | AbstractionLevel,
    *,
    policy: HouseStylePolicy = DEFAULT_HOUSE_STYLE,
) -> HouseStyleResolution:
    """Resolve one canonical target; exact/structural levels forbid ambiguity."""
    resolved_level = resolve_level(level)
    canonical = tuple(sorted({_canonical(candidate) for candidate in candidates}))
    if not canonical:
        raise ValueError("at least one candidate target is required")
    if resolved_level.rank <= 2 and len(canonical) != 1:
        raise ValueError("L0-L2 require one exact/structural target")
    scores = tuple(_score(prompt, target, policy) for target in canonical)
    winner = min(scores, key=lambda score: score.rank)
    return HouseStyleResolution(prompt, resolved_level, winner.target, scores, policy)
