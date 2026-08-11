"""Typed reasoning harness profiles (HARN-01 / SLM-520).

Registers ``reasoning/revmath`` as an opt-in profile of the existing G4
reasoning package. Default callers that omit a profile keep the historical
bench / warmup path — this module adds discovery only, not task semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_PROFILE_ID = "reasoning/g4"
REVMATH_PROFILE_ID = "reasoning/revmath"
PROFILE_PARENT = "src/slm_training/harnesses/reasoning/"

# Owner surface ids from revmath_owner_map.json that every revmath run must bind.
REVMATH_BIND_OWNERS: tuple[str, ...] = (
    "experiment_campaign",
    "formal_preflight_execution",
    "formal_object_loop",
    "decode_stats_record",
    "mechanism_disposition",
    "verifier_witness",
    "replay_bundle",
    "bounded_process",
    "semantic_repair",
    "version_registry",
    "agent_surface_parity",
)


@dataclass(frozen=True)
class ReasoningProfileV1:
    """Discoverable reasoning profile descriptor (no execution authority)."""

    profile_id: str
    description: str
    opt_in: bool
    task_semantics_ready: bool
    binds_owners: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "description": self.description,
            "opt_in": self.opt_in,
            "task_semantics_ready": self.task_semantics_ready,
            "binds_owners": list(self.binds_owners),
            "parent_package": PROFILE_PARENT,
        }


PROFILES: Mapping[str, ReasoningProfileV1] = {
    DEFAULT_PROFILE_ID: ReasoningProfileV1(
        profile_id=DEFAULT_PROFILE_ID,
        description="Historical G4 sketch-vs-direct / abstract-CoT reasoning default.",
        opt_in=False,
        task_semantics_ready=True,
        binds_owners=(),
    ),
    REVMATH_PROFILE_ID: ReasoningProfileV1(
        profile_id=REVMATH_PROFILE_ID,
        description=(
            "Reverse-mathematics / computability typed profile over the G4 "
            "reasoning harness; HARN-02 schemas present, runner remains HARN-03."
        ),
        opt_in=True,
        task_semantics_ready=False,
        binds_owners=REVMATH_BIND_OWNERS,
    ),
}


class ReasoningProfileError(ValueError):
    """Unknown or misconfigured reasoning profile (fail closed)."""


def list_profile_ids() -> tuple[str, ...]:
    return tuple(sorted(PROFILES))


def get_profile(profile_id: str) -> ReasoningProfileV1:
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise ReasoningProfileError(
            f"unknown reasoning profile {profile_id!r}; "
            f"expected one of {sorted(PROFILES)}"
        ) from exc


def resolve_profile(profile_id: str | None = None) -> ReasoningProfileV1:
    """Resolve a profile id; ``None`` / empty keeps the historical default."""
    if profile_id is None or profile_id == "":
        return get_profile(DEFAULT_PROFILE_ID)
    return get_profile(profile_id)


def assert_default_unchanged() -> None:
    """Fail closed if revmath ever becomes the implicit default."""
    default = resolve_profile(None)
    if default.profile_id != DEFAULT_PROFILE_ID:
        raise ReasoningProfileError(
            f"default reasoning profile drifted to {default.profile_id!r}"
        )
    if default.opt_in:
        raise ReasoningProfileError("default reasoning profile must not be opt-in")
    revmath = get_profile(REVMATH_PROFILE_ID)
    if not revmath.opt_in:
        raise ReasoningProfileError(f"{REVMATH_PROFILE_ID!r} must remain opt-in")
    if revmath.task_semantics_ready:
        raise ReasoningProfileError(
            f"{REVMATH_PROFILE_ID!r} must stay task_semantics_ready=False "
            "until HARN-03 lands the deterministic runner"
        )


__all__ = [
    "DEFAULT_PROFILE_ID",
    "PROFILES",
    "PROFILE_PARENT",
    "REVMATH_BIND_OWNERS",
    "REVMATH_PROFILE_ID",
    "ReasoningProfileError",
    "ReasoningProfileV1",
    "assert_default_unchanged",
    "get_profile",
    "list_profile_ids",
    "resolve_profile",
]
