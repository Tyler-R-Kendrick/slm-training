"""CAP4-03 wiring: quantize local legal-action energies and compare inference modes.

Provides a torch-free harness that takes the same small lattice of legal actions,
quantizes the local energies to a fixed set of low-bit formats, and compares
**greedy local** selection with **exact global** selection (additive path
objective).  The goal is to expose tie/collision pathology: if exponentially many
AST paths collapse into a small number of quantized cumulative scores, global
inference cannot recover ranking resolution lost by coarse local energies.

This module is eval-only wiring.  It loads no checkpoint, runs no model, and
makes no quality or ship claim.  It reuses the existing
``slm_training.models.quantization.formats`` descriptors for level sets and bit
accounting, but does not depend on a real learned energy scorer.

Hard invariants:

1. Candidate membership is supplied by the caller; quantization only changes
   scores, never adds or drops legal actions.
2. ``UNKNOWN`` scores are preserved as ``None`` and excluded from aggregation,
   never relabeled as best or worst.
3. Exact global inference is only claimed for acyclic additive path problems;
   cyclic/incomplete problems are reported as ``exact=False``.
4. The final selection still passes through the caller's verifier; this module
   emits only ranking evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from slm_training.models.quantization.formats import (
    QuantFormat,
    binary_format,
    fp16_format,
    int4_format,
    learned_four_level_zero_format,
    symmetric_four_level_format,
    ternary_format,
)
from slm_training.versioning import UNKNOWN, build_version_stamp

ENERGY_SCHEMA_VERSION = 1


class ScoreSemantics(str, Enum):
    """Interpretation of the scalar attached to a local legal action."""

    ADDITIVE_EDGE = "additive_edge"
    COST_TO_GO = "cost_to_go"


class InferenceMode(str, Enum):
    """How a decision is produced from local scores."""

    GREEDY_LOCAL = "greedy_local"
    EXACT_VITERBI = "exact_viterbi"


@dataclass(frozen=True)
class LegalAction:
    """One legal action at one decision stage with a scalar local energy."""

    action_id: str
    local_energy: float
    known: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "local_energy": _safe_float(self.local_energy),
            "known": self.known,
        }


@dataclass(frozen=True)
class EnergyStage:
    """A decision stage: choose one legal action."""

    stage_id: str
    actions: tuple[LegalAction, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "actions": [a.to_dict() for a in self.actions],
        }


@dataclass(frozen=True)
class EnergyProblem:
    """Acyclic additive lattice defined by a sequence of independent stages.

    A path is one action per stage.  The global score is the sum of the selected
    local energies.  This is intentionally small and replayable; it stands in for
    a compiler forest where every next action is supplied by the hard oracle.
    """

    problem_id: str
    stages: tuple[EnergyStage, ...]
    semantics: ScoreSemantics = ScoreSemantics.ADDITIVE_EDGE

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "stages": [s.to_dict() for s in self.stages],
            "semantics": self.semantics.value,
        }

    @property
    def path_count(self) -> int:
        total = 1
        for stage in self.stages:
            n = len(stage.actions)
            if n == 0:
                return 0
            total *= n
        return total


@dataclass(frozen=True)
class QuantizedAction:
    """Local action after score quantization."""

    action_id: str
    original_energy: float
    quantized_energy: float
    known: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "original_energy": _safe_float(self.original_energy),
            "quantized_energy": _safe_float(self.quantized_energy),
            "known": self.known,
        }


@dataclass(frozen=True)
class EnergyQuantizer:
    """Calibrate and quantize local energies to a fixed QuantFormat level grid.

    For non-learned formats the normalized grid is symmetric around zero; the
    scale is ``max_abs_original / max_abs_grid`` so the largest magnitude maps
    to the extreme level.  Learned formats use their declared level set as the
    grid.  ``fp16`` is treated as a reference passthrough.
    """

    fmt: QuantFormat
    scale: float
    schema_version: int = ENERGY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.fmt.format_id,
            "scale": _safe_float(self.scale),
            "nominal_bits": self.fmt.nominal_symbol_bits,
            "physical_bits": self.fmt.physical_slot_bits,
            "level_count": self.fmt.level_count,
            "schema_version": self.schema_version,
        }

    @classmethod
    def calibrate(
        cls,
        fmt: QuantFormat,
        energies: Iterable[float],
    ) -> "EnergyQuantizer":
        finite = [float(e) for e in energies if _is_finite(e)]
        if not finite or fmt.format_id in ("fp16", "bf16"):
            return cls(fmt=fmt, scale=1.0)
        max_abs = max(abs(e) for e in finite)
        grid_max = _grid_max_abs(fmt)
        if grid_max <= 0 or max_abs <= 0:
            return cls(fmt=fmt, scale=1.0)
        return cls(fmt=fmt, scale=max_abs / grid_max)

    def quantize(self, energy: float) -> float:
        if not _is_finite(energy):
            return float(energy)
        if self.fmt.format_id in ("fp16", "bf16"):
            return float(energy)
        levels = _normalized_levels(self.fmt)
        if not levels:
            return float(energy)
        idx = min(
            range(len(levels)),
            key=lambda i: abs(levels[i] * self.scale - energy),
        )
        return float(levels[idx] * self.scale)


def _is_finite(x: float) -> bool:
    return math.isfinite(x)


def _safe_float(x: float) -> float | None:
    return None if not math.isfinite(x) else float(x)


def _normalized_levels(fmt: QuantFormat) -> tuple[float, ...]:
    if fmt.is_learned:
        return tuple(fmt.learned_levels) if fmt.learned_levels else ()
    return tuple(fmt.weight_levels)


def _grid_max_abs(fmt: QuantFormat) -> float:
    levels = _normalized_levels(fmt)
    if not levels:
        return 1.0
    return max(abs(v) for v in levels)


@dataclass(frozen=True)
class PathSelection:
    """Result of one inference mode on one quantized problem."""

    mode: InferenceMode
    format_id: str
    path: tuple[str, ...]
    total_original_energy: float
    total_quantized_energy: float
    exact: bool
    tie_class_size: int
    path_count_considered: int
    score_distribution: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "format_id": self.format_id,
            "path": list(self.path),
            "total_original_energy": _safe_float(self.total_original_energy),
            "total_quantized_energy": _safe_float(self.total_quantized_energy),
            "exact": self.exact,
            "tie_class_size": self.tie_class_size,
            "path_count_considered": self.path_count_considered,
            "score_distribution": [_safe_float(v) for v in self.score_distribution],
        }


@dataclass(frozen=True)
class FormatResult:
    """All selections for one quantization format."""

    quantizer: EnergyQuantizer
    greedy: PathSelection
    exact: PathSelection

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantizer": self.quantizer.to_dict(),
            "greedy": self.greedy.to_dict(),
            "exact": self.exact.to_dict(),
        }


@dataclass
class CompareResult:
    """Container for the CAP4-03 comparison across formats and inference modes."""

    problem: EnergyProblem
    format_results: list[FormatResult] = field(default_factory=list)
    version_stamp: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem.to_dict(),
            "format_results": [fr.to_dict() for fr in self.format_results],
            "version_stamp": self.version_stamp,
        }


def _enumerate_paths(
    problem: EnergyProblem,
    score_fn: Callable[[LegalAction], float],
) -> list[tuple[tuple[str, ...], float, float]]:
    """Return (path action ids, total quantized score, total original score)."""
    partials: list[tuple[tuple[str, ...], float, float]] = [(
        (), 0.0, 0.0
    )]
    for stage in problem.stages:
        next_partials: list[tuple[tuple[str, ...], float, float]] = []
        for prefix, q_sum, o_sum in partials:
            for action in stage.actions:
                if not action.known:
                    continue
                q = score_fn(action)
                if not math.isfinite(q):
                    continue
                next_partials.append((
                    (*prefix, action.action_id),
                    q_sum + q,
                    o_sum + action.local_energy,
                ))
        partials = next_partials
    return partials


def _tie_distribution(scores: list[float]) -> tuple[int, tuple[float, ...]]:
    """Return (max tie class size, sorted distinct scores)."""
    if not scores:
        return 0, ()
    counter: dict[float, int] = {}
    for s in scores:
        counter[s] = counter.get(s, 0) + 1
    return max(counter.values()), tuple(sorted(counter))


def _greedy_select(
    problem: EnergyProblem,
    score_fn: Callable[[LegalAction], float],
    format_id: str,
) -> PathSelection:
    path: list[str] = []
    total_q = 0.0
    total_o = 0.0
    considered = 1
    for stage in problem.stages:
        known = [a for a in stage.actions if a.known and math.isfinite(score_fn(a))]
        considered *= max(len(known), 1)
        if not known:
            continue
        best = min(known, key=lambda a: (score_fn(a), a.action_id))
        path.append(best.action_id)
        total_q += score_fn(best)
        total_o += best.local_energy
    # Greedy is not exact unless stages are independent and local minima compose.
    return PathSelection(
        mode=InferenceMode.GREEDY_LOCAL,
        format_id=format_id,
        path=tuple(path),
        total_original_energy=total_o,
        total_quantized_energy=total_q,
        exact=False,
        tie_class_size=1,
        path_count_considered=considered,
        score_distribution=(total_q,),
    )


def _exact_select(
    problem: EnergyProblem,
    score_fn: Callable[[LegalAction], float],
    format_id: str,
) -> PathSelection:
    paths = _enumerate_paths(problem, score_fn)
    if not paths:
        return PathSelection(
            mode=InferenceMode.EXACT_VITERBI,
            format_id=format_id,
            path=(),
            total_original_energy=0.0,
            total_quantized_energy=0.0,
            exact=True,
            tie_class_size=0,
            path_count_considered=0,
            score_distribution=(),
        )
    # Lower energy is better (earlier under solver convention).
    min_q = min(s for _, s, _ in paths)
    best_paths = [p for p in paths if p[1] == min_q]
    best_paths.sort(key=lambda p: p[0])  # deterministic tie-break by ids
    selected = best_paths[0]
    tie_size, dist = _tie_distribution([s for _, s, _ in paths])
    return PathSelection(
        mode=InferenceMode.EXACT_VITERBI,
        format_id=format_id,
        path=selected[0],
        total_original_energy=selected[2],
        total_quantized_energy=selected[1],
        exact=True,
        tie_class_size=tie_size,
        path_count_considered=len(paths),
        score_distribution=dist,
    )


def evaluate_format(
    problem: EnergyProblem,
    quantizer: EnergyQuantizer,
) -> FormatResult:
    """Quantize the problem and run both greedy and exact inference."""
    def score_fn(action: LegalAction) -> float:
        if not action.known:
            return float("inf")
        return quantizer.quantize(action.local_energy)

    return FormatResult(
        quantizer=quantizer,
        greedy=_greedy_select(problem, score_fn, quantizer.fmt.format_id),
        exact=_exact_select(problem, score_fn, quantizer.fmt.format_id),
    )


def compare_quantized_energy_inference(
    problem: EnergyProblem,
    formats: Iterable[QuantFormat] | None = None,
    *,
    stamp_components: tuple[str, ...] = ("evals.scoring",),
) -> CompareResult:
    """Build the CAP4-03 comparison for one energy problem across formats.

    Default formats mirror the issue's requested energy formats:
    FP16 reference, binary, ternary, symmetric four-level, learned four-level
    with zero, and INT4.
    """
    if formats is None:
        formats = (
            fp16_format(),
            binary_format(),
            ternary_format(),
            symmetric_four_level_format(),
            learned_four_level_zero_format(),
            int4_format(),
        )
    result = CompareResult(problem=problem)
    all_energies = [
        a.local_energy
        for stage in problem.stages
        for a in stage.actions
        if a.known
    ]
    for fmt in formats:
        quantizer = EnergyQuantizer.calibrate(fmt, all_energies)
        result.format_results.append(evaluate_format(problem, quantizer))

    try:
        result.version_stamp = build_version_stamp(*stamp_components)
    except KeyError:
        result.version_stamp = {
            "stamp_schema": UNKNOWN,
            "components": {cid: UNKNOWN for cid in stamp_components},
            "note": "version stamp unavailable",
        }
    return result
