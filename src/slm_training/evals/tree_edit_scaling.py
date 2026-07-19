"""EFS2-01 wiring: X22 beam-width × edit-depth scaling over valid program states.

This module provides a torch-free, replayable harness for the factorial grid
requested by SLM-111: ``beam_width ∈ {1,4,16}`` × ``max_edit_depth ∈ {1,2,4}``
over seeded tree-edit search trajectories.  It reuses the existing
``TreeEditSpace`` from ``slm_training.models.tree_edit_diffusion`` so every
applied edit is re-verified by the real parser; invalid edits are counted but
never enter the live beam.

The harness is eval/scaffolding wiring only.  It loads no checkpoint, runs no
learned policy, and makes no quality or ship claim.  A real EFS2-01 run still
requires the trained X22 ``TreeEditDiffusionModel`` and durable checkpoints.

Invariants enforced here:

1. Every state admitted to the beam round-trips through the parser.
2. Edit depth counts successful non-STOP edits along a path.
3. Duplicate canonical AST states are removed before beam truncation.
4. ``max_edit_depth`` is a hard limit, not overloaded onto loop iterations.
5. Invalid edit attempts are telemetry only; they never change candidate membership.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Protocol

from slm_training.models.tree_edit_diffusion import (
    ACTION_ADD,
    ACTION_REMOVE,
    ACTION_REPLACE,
    ACTION_STOP,
    Edit,
    Statement,
    TreeEditSpace,
    parse_statements,
    render_statements,
)
from slm_training.versioning import UNKNOWN, build_version_stamp

SCALING_SCHEMA_VERSION = 1


class InferenceMode(str, Enum):
    """Soft ordering used to expand the live beam."""

    RANDOM_VALUE = "random_value"
    DETERMINISTIC = "deterministic"


@dataclass(frozen=True)
class TreeEditScalingConfig:
    """One decode cell in the SLM-111 grid."""

    beam_width: int
    max_edit_depth: int
    expand_per_state: int = 4
    max_search_steps: int = 12
    seed: int = 0
    mode: InferenceMode = InferenceMode.RANDOM_VALUE

    def to_dict(self) -> dict[str, Any]:
        return {
            "beam_width": self.beam_width,
            "max_edit_depth": self.max_edit_depth,
            "expand_per_state": self.expand_per_state,
            "max_search_steps": self.max_search_steps,
            "seed": self.seed,
            "mode": self.mode.value,
        }


@dataclass(frozen=True)
class BeamState:
    """One search state carried by the beam."""

    program_text: str
    fingerprint: str
    edit_depth: int
    parent_fingerprint: str | None
    edit: dict[str, Any] | None
    value_score: float
    frozen: bool
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_text": self.program_text,
            "fingerprint": self.fingerprint,
            "edit_depth": self.edit_depth,
            "parent_fingerprint": self.parent_fingerprint,
            "edit": self.edit,
            "value_score": _safe_float(self.value_score),
            "frozen": self.frozen,
            "valid": self.valid,
        }


@dataclass
class SearchTelemetry:
    """Counters and evidence for one beam search run."""

    steps: int = 0
    invalid_attempts: int = 0
    duplicate_prunes: int = 0
    visited_states: int = 0
    expanded_states: int = 0
    max_live_beam: int = 0
    mean_live_beam: float = 0.0
    final_frozen: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "invalid_attempts": self.invalid_attempts,
            "duplicate_prunes": self.duplicate_prunes,
            "visited_states": self.visited_states,
            "expanded_states": self.expanded_states,
            "max_live_beam": self.max_live_beam,
            "mean_live_beam": _safe_float(self.mean_live_beam),
            "final_frozen": self.final_frozen,
        }


@dataclass(frozen=True)
class BeamSearchResult:
    """Outcome of one cell (one seed × config)."""

    config: TreeEditScalingConfig
    seed_program: str
    final_beam: tuple[BeamState, ...]
    telemetry: SearchTelemetry

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "seed_program": self.seed_program,
            "final_beam": [s.to_dict() for s in self.final_beam],
            "telemetry": self.telemetry.to_dict(),
        }


@dataclass(frozen=True)
class GridCell:
    """All seeds for one beam × depth cell."""

    beam_width: int
    max_edit_depth: int
    seeds: tuple[int, ...]
    results: tuple[BeamSearchResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "beam_width": self.beam_width,
            "max_edit_depth": self.max_edit_depth,
            "seeds": list(self.seeds),
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class ScalingGridResult:
    """Full SLM-111 factorial grid."""

    beam_widths: tuple[int, ...]
    max_edit_depths: tuple[int, ...]
    seeds: tuple[int, ...]
    cells: list[GridCell] = field(default_factory=list)
    version_stamp: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "beam_widths": list(self.beam_widths),
            "max_edit_depths": list(self.max_edit_depths),
            "seeds": list(self.seeds),
            "cells": [c.to_dict() for c in self.cells],
            "version_stamp": self.version_stamp,
        }


def _safe_float(x: float) -> float | None:
    return None if not (isinstance(x, float) and x == x) else float(x)


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EditRanker(Protocol):
    """Score a batch of candidate edits/states for beam ordering."""

    def score(
        self,
        state: list[Statement],
        candidates: list[tuple[Edit, list[Statement] | None]],
        rng: random.Random,
    ) -> list[tuple[float, Edit, list[Statement]]]: ...


@dataclass
class RandomValueRanker:
    """Deterministic random ranker stand-in for a learned value head."""

    seed: int = 0

    def score(
        self,
        state: list[Statement],
        candidates: list[tuple[Edit, list[Statement] | None]],
        rng: random.Random,
    ) -> list[tuple[float, Edit, list[Statement]]]:
        valid = [
            (float(rng.random()), edit, child)
            for edit, child in candidates
            if child is not None
        ]
        valid.sort(key=lambda t: t[0], reverse=True)
        return valid


@dataclass
class DeterministicRanker:
    """Canonical edit ordering: STOP, then REPLACE, ADD, REMOVE by indices."""

    def score(
        self,
        state: list[Statement],
        candidates: list[tuple[Edit, list[Statement] | None]],
        rng: random.Random,
    ) -> list[tuple[float, Edit, list[Statement]]]:
        def _key(item: tuple[Edit, list[Statement] | None]) -> tuple[int, int, int, int]:
            edit, child = item
            if child is None:
                return (10, 0, 0, 0)
            return (edit.action, edit.stmt, edit.comp, edit.slot)

        return [
            (float(i), edit, child)
            for i, (edit, child) in enumerate(sorted(candidates, key=_key))
            if child is not None
        ]


def _enumerate_edits(
    statements: list[Statement],
    inventory: list[str],
    space: TreeEditSpace,
) -> list[tuple[Edit, list[Statement] | None]]:
    """Enumerate all bounded edits for a state; apply re-validates each one."""
    candidates: list[tuple[Edit, list[Statement] | None]] = [
        (Edit(ACTION_STOP), list(statements))
    ]
    n_comp = len(space.components)
    n_slots = min(len(inventory), 16)
    for stmt in range(min(len(statements), 24)):
        candidates.append((Edit(ACTION_REMOVE, stmt), space.apply(statements, Edit(ACTION_REMOVE, stmt), inventory)))
        for comp in range(n_comp):
            edit = Edit(ACTION_REPLACE, stmt, comp)
            candidates.append((edit, space.apply(statements, edit, inventory)))
            for slot in range(n_slots):
                add_edit = Edit(ACTION_ADD, stmt, comp, slot)
                candidates.append((add_edit, space.apply(statements, add_edit, inventory)))
    return candidates


def _make_beam_state(
    statements: list[Statement],
    edit_depth: int,
    parent_fingerprint: str | None,
    edit: Edit | None,
    value_score: float,
    frozen: bool,
) -> BeamState:
    text = render_statements(statements)
    return BeamState(
        program_text=text,
        fingerprint=_fingerprint(text),
        edit_depth=edit_depth,
        parent_fingerprint=parent_fingerprint,
        edit=({
            "action": edit.action,
            "stmt": edit.stmt,
            "comp": edit.comp,
            "slot": edit.slot,
        } if edit else None),
        value_score=value_score,
        frozen=frozen,
        valid=True,
    )


def run_tree_edit_scaling_cell(
    seed_program: str,
    inventory: list[str],
    config: TreeEditScalingConfig,
    *,
    ranker: EditRanker | None = None,
) -> BeamSearchResult:
    """Run one decode cell with explicit edit-depth tracking."""
    space = TreeEditSpace()
    seed_statements = parse_statements(seed_program)
    if seed_statements is None:
        raise ValueError(f"seed program does not parse: {seed_program!r}")
    seed_text = render_statements(seed_statements)
    if not seed_text:
        raise ValueError("empty seed program")

    rng = random.Random(config.seed)
    ranker = ranker or (
        RandomValueRanker(seed=config.seed)
        if config.mode is InferenceMode.RANDOM_VALUE
        else DeterministicRanker()
    )

    seed_beam = _make_beam_state(seed_statements, 0, None, None, 0.0, False)
    beam: list[BeamState] = [seed_beam]
    seen: set[str] = {seed_beam.fingerprint}
    telemetry = SearchTelemetry()

    for _ in range(config.max_search_steps):
        live = [b for b in beam if not b.frozen]
        if not live:
            break
        telemetry.steps += 1
        telemetry.max_live_beam = max(telemetry.max_live_beam, len(live))
        telemetry.mean_live_beam += len(live)

        next_beam: list[BeamState] = [b for b in beam if b.frozen]
        for state in live:
            statements = parse_statements(state.program_text)
            if statements is None:
                telemetry.invalid_attempts += 1
                continue
            telemetry.expanded_states += 1
            raw_candidates = _enumerate_edits(statements, inventory, space)
            scored = ranker.score(statements, raw_candidates, rng)
            expanded = 0
            for value_score, edit, child in scored:
                if expanded >= config.expand_per_state:
                    break
                if edit.action == ACTION_STOP:
                    if state.edit_depth == 0:
                        # Skip trivial STOP at the seed; the seed is already represented.
                        continue
                    frozen_state = _make_beam_state(
                        parse_statements(state.program_text) or statements,
                        state.edit_depth,
                        state.fingerprint,
                        edit,
                        value_score,
                        True,
                    )
                    if frozen_state.fingerprint not in seen:
                        seen.add(frozen_state.fingerprint)
                        next_beam.append(frozen_state)
                        expanded += 1
                    else:
                        telemetry.duplicate_prunes += 1
                    continue

                if state.edit_depth >= config.max_edit_depth:
                    continue
                child_text = render_statements(child)
                child_fp = _fingerprint(child_text)
                if child_fp in seen:
                    telemetry.duplicate_prunes += 1
                    continue
                seen.add(child_fp)
                telemetry.visited_states += 1
                next_beam.append(
                    _make_beam_state(
                        child,
                        state.edit_depth + 1,
                        state.fingerprint,
                        edit,
                        value_score,
                        False,
                    )
                )
                expanded += 1

        # Count invalid attempts from raw candidates that produced None.
        for _, edit, child in scored:
            if child is None:
                telemetry.invalid_attempts += 1

        if not next_beam:
            break
        beam = sorted(next_beam, key=lambda s: s.value_score, reverse=True)[: config.beam_width]
        if all(b.frozen for b in beam):
            break

    telemetry.final_frozen = sum(1 for b in beam if b.frozen)
    if telemetry.steps > 0:
        telemetry.mean_live_beam /= telemetry.steps
    return BeamSearchResult(
        config=config,
        seed_program=seed_program,
        final_beam=tuple(beam),
        telemetry=telemetry,
    )


def run_scaling_grid(
    seed_programs: Iterable[str],
    inventory: list[str],
    *,
    beam_widths: tuple[int, ...] = (1, 4, 16),
    max_edit_depths: tuple[int, ...] = (1, 2, 4),
    seeds: tuple[int, ...] = (0, 1, 2),
    expand_per_state: int = 4,
    max_search_steps: int = 12,
    mode: InferenceMode = InferenceMode.RANDOM_VALUE,
    stamp_components: tuple[str, ...] = ("evals.scoring",),
) -> ScalingGridResult:
    """Run the full SLM-111 factorial grid over seeds and programs."""
    result = ScalingGridResult(
        beam_widths=beam_widths,
        max_edit_depths=max_edit_depths,
        seeds=seeds,
    )
    programs = list(seed_programs)
    for beam_width in beam_widths:
        for max_edit_depth in max_edit_depths:
            cell_results: list[BeamSearchResult] = []
            for seed in seeds:
                for program in programs:
                    config = TreeEditScalingConfig(
                        beam_width=beam_width,
                        max_edit_depth=max_edit_depth,
                        expand_per_state=expand_per_state,
                        max_search_steps=max_search_steps,
                        seed=seed,
                        mode=mode,
                    )
                    cell_results.append(
                        run_tree_edit_scaling_cell(program, inventory, config)
                    )
            result.cells.append(
                GridCell(
                    beam_width=beam_width,
                    max_edit_depth=max_edit_depth,
                    seeds=seeds,
                    results=tuple(cell_results),
                )
            )
    try:
        result.version_stamp = build_version_stamp(*stamp_components)
    except KeyError:
        result.version_stamp = {
            "stamp_schema": UNKNOWN,
            "components": {cid: UNKNOWN for cid in stamp_components},
            "note": "version stamp unavailable",
        }
    return result
