"""Regression coverage for the cross-harness law parity certificate."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import verify_agent_surfaces
from scripts.verify_agent_surfaces import (
    HOOK_CONFIGS,
    OBLIGATIONS,
    PRIMARY_SURFACES,
    AgentSurfaceError,
    check,
)

ROOT = Path(__file__).resolve().parents[2]


def test_every_configured_harness_carries_every_law() -> None:
    checked = check()
    assert set(checked) == {obligation.id for obligation in OBLIGATIONS}


def test_the_matrix_covers_every_primary_surface() -> None:
    """Adding a surface without adding it to the laws is the drift being fixed.

    Every primary instruction surface must appear in the shared obligations,
    not only in the one law that happened to be enforced before.
    """
    shared = {
        obligation.id: set(obligation.surfaces())
        for obligation in OBLIGATIONS
        if obligation.scope == "instructions"
        and obligation.id not in {"decode.invariants", "canonical.agents-md"}
    }
    for obligation_id, surfaces in shared.items():
        missing = set(PRIMARY_SURFACES) - surfaces
        assert not missing, f"{obligation_id} does not reach {sorted(missing)}"


def test_every_hook_capable_harness_gets_the_same_hooks() -> None:
    """Automated feedback must not be one harness's privilege.

    The post-edit dashboard-parity and version-stamp hooks were Claude Code
    only, so a Codex or Copilot agent editing a dashboard page got no signal.
    """
    hook_obligations = [o for o in OBLIGATIONS if o.scope == "hooks"]
    assert hook_obligations
    for obligation in hook_obligations:
        assert set(HOOK_CONFIGS) <= set(obligation.surfaces()), obligation.id


def test_every_surface_file_exists() -> None:
    for obligation in OBLIGATIONS:
        for relative in obligation.surfaces():
            assert (ROOT / relative).is_file(), relative


def test_a_missing_marker_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    surface = tmp_path / "AGENTS.md"
    surface.write_text("carries nothing\n", encoding="utf-8")
    monkeypatch.setattr(verify_agent_surfaces, "ROOT", tmp_path)
    with pytest.raises(AgentSurfaceError, match="does not carry"):
        check(obligation_id="run.cap")

    surface.write_text("obey MAX_RUN_MINUTES\n", encoding="utf-8")
    # The other surfaces are still absent, so it must keep failing rather than
    # passing on a partial matrix.
    with pytest.raises(AgentSurfaceError, match="missing agent surface"):
        check(obligation_id="run.cap")


def test_unknown_obligation_is_rejected() -> None:
    with pytest.raises(AgentSurfaceError, match="unknown obligation id"):
        check(obligation_id="not.a.law")


def test_decode_invariants_delegates_to_this_matrix() -> None:
    """One owner for 'every surface carries the law', not two."""
    from scripts.verify_decode_invariants import (
        AGENT_SURFACE_OBLIGATION,
        check_agent_surfaces,
    )

    assert AGENT_SURFACE_OBLIGATION == "decode.invariants"
    assert set(check_agent_surfaces()) == set(
        check(obligation_id="decode.invariants")["decode.invariants"]
    )
