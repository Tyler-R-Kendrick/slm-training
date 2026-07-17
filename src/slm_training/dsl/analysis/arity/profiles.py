"""Versioned bounded-analysis profiles for the OpenUI state graph (CAP1-01)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnalysisProfile:
    """Finite frame under which a state graph is explored.

    The profile intentionally limits the language so the graph can exhaust on
    CPU. Any bound that is hit during exploration makes the report ``UNKNOWN``;
    it never invents exact counts for the unrestricted language.
    """

    profile_id: str
    representation: str
    dsl: str
    max_semantic_decisions: int
    max_components: int
    max_live_bindings: int
    max_list_items: int
    max_object_members: int
    max_literal_slots: int
    allowed_component_subset: tuple[str, ...] = ()
    required_coverage: str = "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "representation": self.representation,
            "dsl": self.dsl,
            "max_semantic_decisions": self.max_semantic_decisions,
            "max_components": self.max_components,
            "max_live_bindings": self.max_live_bindings,
            "max_list_items": self.max_list_items,
            "max_object_members": self.max_object_members,
            "max_literal_slots": self.max_literal_slots,
            "allowed_component_subset": list(self.allowed_component_subset),
            "required_coverage": self.required_coverage,
        }


# Intentionally tiny fixture profile. Exhausts on CPU for programs like
# ``root = Stack([blurb], "column") / blurb = TextContent(":page.blurb")``.
# Bounds are deliberately tight so the graph remains small and exact; larger
# profiles that hit a bound report UNKNOWN rather than invent counts.
OPENVUI_CAP_V1 = AnalysisProfile(
    profile_id="openui-cap-v1",
    representation="choice",
    dsl="openui",
    max_semantic_decisions=6,
    max_components=2,
    max_live_bindings=3,
    max_list_items=3,
    max_object_members=3,
    max_literal_slots=1,
    allowed_component_subset=("Card", "TextContent", "Button", "Stack"),
    required_coverage="complete",
)


_PROFILES: dict[str, AnalysisProfile] = {
    OPENVUI_CAP_V1.profile_id: OPENVUI_CAP_V1,
}


def get_profile(profile_id: str) -> AnalysisProfile:
    """Return a built-in profile by id."""
    if profile_id not in _PROFILES:
        raise KeyError(f"unknown analysis profile {profile_id!r}")
    return _PROFILES[profile_id]


def register_profile(profile: AnalysisProfile) -> None:
    """Register a custom profile for CLI use."""
    _PROFILES[profile.profile_id] = profile


__all__ = ["AnalysisProfile", "OPENVUI_CAP_V1", "get_profile", "register_profile"]
