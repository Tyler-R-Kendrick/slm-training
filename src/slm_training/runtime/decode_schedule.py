"""Symbol-table-driven prefill scheduling for constrained decode.

Decode invariant I4 (`docs/design/decode-invariants.md`): the forward-calculated
symbol table is not only a legality filter — it is a *schedule*. Before every
denoiser call the planner knows, from the grammar alone:

* which batch rows still have an ambiguous next symbol (the rest are already
  committed by the deterministic bypass and must not be forwarded);
* how far ahead the grammar stays ambiguous, i.e. where the next **grammar
  checkpoint** is — the first position at which the domain collapses to a
  singleton and the model's opinion stops mattering;
* which device the forward will land on, so the read-ahead window can be sized
  for the accelerator instead of for the canvas.

The planner is pure: it returns a :class:`PrefillPlanV1` and never runs a
forward itself. With ``mode="off"`` it reproduces the caller's legacy budget
exactly, so turning the lever on is the only thing that can change a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

PREFILL_SCHEDULE_IMPL_VERSION = "prefill_schedule/v1"

# Devices whose forwards are dominated by kernel-launch overhead prefer fewer,
# wider prefills; CPU prefers the smallest window that covers the ambiguity.
_WIDE_PREFILL_BACKENDS = frozenset({"cuda", "npu", "dml"})


@dataclass(frozen=True)
class PrefillPlanV1:
    """One planned denoiser call over a constrained decode canvas."""

    rows: tuple[int, ...]
    prefill_end: int
    checkpoint: int | None
    backend: str
    reason: str
    rows_skipped: int = 0
    tokens_saved: int = 0
    impl_version: str = PREFILL_SCHEDULE_IMPL_VERSION

    @property
    def needs_forward(self) -> bool:
        return bool(self.rows)

    def as_dict(self) -> dict[str, Any]:
        return {
            "impl_version": self.impl_version,
            "rows": list(self.rows),
            "prefill_end": self.prefill_end,
            "checkpoint": self.checkpoint,
            "backend": self.backend,
            "reason": self.reason,
            "rows_skipped": self.rows_skipped,
            "tokens_saved": self.tokens_saved,
        }


def next_grammar_checkpoint(
    position: int,
    canvas: int,
    forced_run_lengths: Mapping[int, int] | None = None,
) -> int | None:
    """Return the next position whose symbol is already grammar-determined.

    ``forced_run_lengths`` maps a canvas position to the number of consecutive
    tokens the grammar forces from there. The first such position at or after
    ``position`` is where a forward stops buying information, so it is the
    natural end of a prefill window.
    """
    if not forced_run_lengths:
        return None
    candidates = [
        int(index)
        for index, run in forced_run_lengths.items()
        if int(run) > 0 and position <= int(index) < canvas
    ]
    if not candidates:
        return None
    return min(candidates)


def plan_prefill(
    *,
    position: int,
    canvas: int,
    active_rows: Sequence[int],
    committed_rows: Sequence[int],
    mode: str = "off",
    legacy_lookahead: int = 0,
    max_lookahead: int = 0,
    backend: str = "cpu",
    forced_run_lengths: Mapping[int, int] | None = None,
) -> PrefillPlanV1:
    """Plan the next forward from the symbol table.

    ``committed_rows`` are rows the deterministic bypass already resolved at
    this position (I2). They are never forwarded, in any mode — dropping them
    is a correctness consequence of the bypass, not a scheduling choice.
    """
    committed = {int(row) for row in committed_rows}
    rows = tuple(int(row) for row in active_rows if int(row) not in committed)
    skipped = len(committed & {int(row) for row in active_rows})
    normalized = str(mode or "off").lower()
    legacy_end = (
        min(canvas, position + int(legacy_lookahead))
        if int(legacy_lookahead) > 0
        else canvas
    )

    if normalized in {"off", ""} or not rows:
        return PrefillPlanV1(
            rows=rows,
            prefill_end=legacy_end,
            checkpoint=None,
            backend=backend,
            reason="legacy_budget" if rows else "fully_committed",
            rows_skipped=skipped,
            tokens_saved=0,
        )
    if normalized != "checkpoint":
        raise ValueError(
            f"unsupported prefill_schedule {normalized!r}; expected 'off' or 'checkpoint'"
        )

    checkpoint = next_grammar_checkpoint(position + 1, canvas, forced_run_lengths)
    window = max(1, int(max_lookahead)) if int(max_lookahead) > 0 else canvas - position
    if backend in _WIDE_PREFILL_BACKENDS:
        # A launch is expensive on an accelerator: never plan a window narrower
        # than the legacy one, only wider up to the checkpoint.
        window = max(window, int(legacy_lookahead) or window)
    planned_end = min(canvas, position + window)
    reason = "device_window"
    if checkpoint is not None and checkpoint < planned_end:
        # Reading past a forced lexeme adds canvas without adding a decision.
        planned_end = max(position + 1, checkpoint)
        reason = "grammar_checkpoint"
    planned_end = max(position + 1, min(planned_end, canvas))
    return PrefillPlanV1(
        rows=rows,
        prefill_end=planned_end,
        checkpoint=checkpoint,
        backend=backend,
        reason=reason,
        rows_skipped=skipped,
        tokens_saved=max(0, legacy_end - planned_end),
    )


def record_plan(stats: Any | None, plan: PrefillPlanV1) -> None:
    """Fold one plan into ``DecodeStats`` so utilization stays measurable."""
    if stats is None or not plan.needs_forward:
        return
    stats.scheduled_prefills += 1
    stats.scheduled_rows_skipped += plan.rows_skipped
    stats.scheduled_prefill_tokens_saved += plan.tokens_saved
    if plan.reason == "grammar_checkpoint":
        stats.schedule_checkpoint_hits += 1


def schedule_backend(preferred: str | None = None) -> str:
    """Best-effort accelerator id for planning, without importing torch eagerly."""
    try:
        from slm_training.runtime.accel import detect_device

        return detect_device(preferred).backend
    except Exception:  # noqa: BLE001 - planning must never break a decode
        return "cpu"


__all__ = [
    "PREFILL_SCHEDULE_IMPL_VERSION",
    "PrefillPlanV1",
    "next_grammar_checkpoint",
    "plan_prefill",
    "record_plan",
    "schedule_backend",
]
