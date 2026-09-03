"""A certified substitution must never be recorded as a successful generation.

`TwoTowerModel` may satisfy decode invariant I6 by returning a certified
deterministic program instead of an invalid one. That keeps the output legal
but it is *not* a successful generation, and the serving harness persists every
attempt as annotation evidence — so it must be able to tell the two apart.

The substitution flag used to be written only on the choice-constrained lane.
The grammar/LTR lane (the one production serving runs) substituted silently, so
`_raise_on_substituted_generation` never fired there. These tests pin the
evidence channel on both lanes.
"""

from __future__ import annotations

import pytest

from slm_training.web.service import (
    SubstitutedGeneration,
    _raise_on_substituted_generation,
)


class _EvidenceOnly:
    """Minimal stand-in exposing only the evidence channel the guard reads."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.consumed = 0

    def consume_generation_evidence(self) -> list[dict]:
        self.consumed += 1
        rows, self._rows = self._rows, []
        return rows


def test_guard_raises_when_any_row_reports_a_substitution() -> None:
    model = _EvidenceOnly([{"fallback_used": False}, {"fallback_used": True}])
    with pytest.raises(SubstitutedGeneration):
        _raise_on_substituted_generation(model)


def test_guard_passes_when_no_row_substituted() -> None:
    model = _EvidenceOnly([{"fallback_used": False}, {"fallback_used": False}])
    _raise_on_substituted_generation(model)
    assert model.consumed == 1


def test_guard_ignores_backends_without_an_evidence_channel() -> None:
    _raise_on_substituted_generation(object())


# ---------------------------------------------------------------------------
# The real model's evidence channel, exercised without building a network.
# ---------------------------------------------------------------------------


class _FlagRecorder:
    """Drives the real drain/record logic bound to a bare object.

    `consume_generation_evidence` and the finalize wrapper only touch instance
    state, so they can be bound directly. That keeps the test at unit speed
    while still exercising the shipped code rather than a copy of it.
    """

    def __init__(self) -> None:
        from slm_training.models.twotower import TwoTowerModel

        self._last_generation_evidence: list[dict[str, object]] = []
        self._certified_substitution_flags: list[bool] = []
        self._last_ensure_substituted = False
        self._consume = TwoTowerModel.consume_generation_evidence.__get__(self)
        self._ensure = TwoTowerModel._ensure_valid_openui.__get__(self)

    # Stands in for the finalize body: returns text, optionally substituting.
    def _ensure_valid_openui_inner(self, text: str, *, substitute: bool) -> str:
        if substitute:
            self._last_ensure_substituted = True
            return "root = Stack([])"
        return text

    def finalize(self, text: str, *, substitute: bool) -> str:
        return self._ensure(text, substitute=substitute)

    def consume(self) -> list[dict]:
        return self._consume()


def test_grammar_lane_substitution_is_visible_with_no_prior_evidence() -> None:
    """The regression: this lane records no evidence of its own.

    Before the fix the drain returned `[]`, so the guard saw nothing and a
    substituted program was persisted as a genuine generation.
    """
    recorder = _FlagRecorder()
    recorder.finalize("junk", substitute=True)

    rows = recorder.consume()
    assert rows == [{"fallback_used": True}]
    with pytest.raises(SubstitutedGeneration):
        _raise_on_substituted_generation(_EvidenceOnly(rows))


def test_clean_finalize_reports_no_substitution() -> None:
    recorder = _FlagRecorder()
    assert (
        recorder.finalize("root = Stack([b1])", substitute=False)
        == "root = Stack([b1])"
    )
    rows = recorder.consume()
    assert rows == [{"fallback_used": False}]
    _raise_on_substituted_generation(_EvidenceOnly(rows))


def test_flags_align_positionally_with_rows_in_a_batch() -> None:
    """Both call sites finalize once per row, in row order."""
    recorder = _FlagRecorder()
    for index, substitute in enumerate([False, True, False]):
        recorder.finalize(f"row{index}", substitute=substitute)

    rows = recorder.consume()
    assert [row["fallback_used"] for row in rows] == [False, True, False]


def test_existing_evidence_rows_are_preserved_and_stamped() -> None:
    """A lane that already builds rows keeps them; the flag is merged in."""
    recorder = _FlagRecorder()
    recorder._last_generation_evidence = [{"terminal_binding": {"a": 1}}]
    recorder.finalize("junk", substitute=True)

    rows = recorder.consume()
    assert rows == [{"terminal_binding": {"a": 1}, "fallback_used": True}]


def test_consume_clears_state_so_the_next_batch_starts_clean() -> None:
    recorder = _FlagRecorder()
    recorder.finalize("junk", substitute=True)
    assert recorder.consume() == [{"fallback_used": True}]
    # A second batch that substitutes nothing must not inherit the flag.
    recorder.finalize("ok", substitute=False)
    assert recorder.consume() == [{"fallback_used": False}]


def test_a_raising_finalize_still_records_its_row() -> None:
    """The flag is recorded in a finally block, so an exception cannot skip it."""
    recorder = _FlagRecorder()

    def _boom(text: str, *, substitute: bool) -> str:
        recorder._last_ensure_substituted = True
        raise RuntimeError("finalize failed")

    recorder._ensure_valid_openui_inner = _boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        recorder.finalize("junk", substitute=True)
    assert recorder.consume() == [{"fallback_used": True}]
