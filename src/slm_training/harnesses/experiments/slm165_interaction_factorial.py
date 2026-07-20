"""SLM-165 (SDE1-03): 2×2×2 interaction factorial fixture harness.

Combines description initialization (SLM-163), type balancing (E396), and
targeted legal-sibling margin (SLM-164) in a deterministic, CPU-only factorial.
No GPU model is trained and no ship-gate claim is made.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "EXPERIMENT_ID",
    "ACTION_INIT_LEVELS",
    "TYPE_BALANCE_LEVELS",
    "SIBLING_MARGIN_LEVELS",
    "InteractionCell",
    "InteractionMetrics",
    "InteractionFactorialReport",
    "build_cells",
    "validate_manifest",
    "resolve_action_init_winner",
    "resolve_sibling_margin_winner",
    "run_fixture_campaign",
    "render_markdown",
]

MATRIX_VERSION = "sde1-03-v1"
MATRIX_SET = "slm165_interaction_factorial"
EXPERIMENT_ID = "slm165-interaction-factorial"

_DEFAULT_SEEDS = (0, 1, 2)
_DEFAULT_TARGET_DECISIONS = 1000

ACTION_INIT_LEVELS = ("current_stub", "schema_description")
TYPE_BALANCE_LEVELS = ("neutral", "e396_balanced")
SIBLING_MARGIN_LEVELS = ("none", "targeted_weighted")

_SLM163_RESULT_JSON = "docs/design/iter-slm163-schema-action-embedding-20260720.json"
_SLM164_RESULT_JSON = "docs/design/iter-slm164-targeted-margin-20260720.json"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InteractionCell:
    """One 2×2×2 factorial cell plus derived recipe fields."""

    cell_id: str
    action_init: str
    type_balance: str
    sibling_margin: str
    seed: int
    action_embedding_init: str
    slot_component_loss_weight: float
    slot_component_class_balance_power: float
    slot_component_owner_rare_threshold: int
    slot_component_owner_rare_multiplier: int
    legal_margin_mode: str
    targeted_margin_value: float
    target_decisions: int

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionCell":
        return cls(
            cell_id=str(data["cell_id"]),
            action_init=str(data["action_init"]),
            type_balance=str(data["type_balance"]),
            sibling_margin=str(data["sibling_margin"]),
            seed=int(data["seed"]),
            action_embedding_init=str(data["action_embedding_init"]),
            slot_component_loss_weight=float(data["slot_component_loss_weight"]),
            slot_component_class_balance_power=float(data["slot_component_class_balance_power"]),
            slot_component_owner_rare_threshold=int(data["slot_component_owner_rare_threshold"]),
            slot_component_owner_rare_multiplier=int(data["slot_component_owner_rare_multiplier"]),
            legal_margin_mode=str(data["legal_margin_mode"]),
            targeted_margin_value=float(data["targeted_margin_value"]),
            target_decisions=int(data["target_decisions"]),
        )


@dataclass(frozen=True)
class InteractionMetrics:
    """Per-cell, per-seed synthetic fixture metrics."""

    cell_id: str
    action_init: str
    type_balance: str
    sibling_margin: str
    seed: int
    rare_component_recall: float
    meaningful_program_rate: float
    common_component_recall: float
    parse_validity_rate: float
    first_attempt_quality: float
    target_decisions: int
    wall_seconds: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionMetrics":
        return cls(
            cell_id=str(data["cell_id"]),
            action_init=str(data["action_init"]),
            type_balance=str(data["type_balance"]),
            sibling_margin=str(data["sibling_margin"]),
            seed=int(data["seed"]),
            rare_component_recall=float(data["rare_component_recall"]),
            meaningful_program_rate=float(data["meaningful_program_rate"]),
            common_component_recall=float(data["common_component_recall"]),
            parse_validity_rate=float(data["parse_validity_rate"]),
            first_attempt_quality=float(data["first_attempt_quality"]),
            target_decisions=int(data["target_decisions"]),
            wall_seconds=float(data["wall_seconds"]),
            notes=list(data.get("notes", [])),
        )


@dataclass(frozen=True)
class InteractionFactorialReport:
    """Full fixture report for SLM-165."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    cells: tuple[InteractionCell, ...]
    rows: list[InteractionMetrics]
    factorial_analysis: dict[str, Any]
    dependency_caveats: list[str]
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "cells": [cell.to_dict() for cell in self.cells],
            "rows": [row.to_dict() for row in self.rows],
            "factorial_analysis": dict(self.factorial_analysis),
            "dependency_caveats": list(self.dependency_caveats),
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionFactorialReport":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", EXPERIMENT_ID),
            run_id=data.get("run_id", "slm165_fixture"),
            status=data.get("status", "fixture"),
            claim_class=data.get("claim_class", "wiring"),
            hypothesis=data.get(
                "hypothesis",
                "The combination of schema-description action initialization, E396 type balancing, "
                "and targeted legal-sibling margin produces a positive three-way interaction on "
                "rare-component recall.",
            ),
            falsifier=data.get(
                "falsifier",
                "The full three-lever cell does not outperform the best two-lever cell, or the "
                "three-way interaction is non-positive.",
            ),
            cells=tuple(InteractionCell.from_dict(c) for c in data.get("cells", [])),
            rows=[InteractionMetrics.from_dict(r) for r in data.get("rows", [])],
            factorial_analysis=dict(data.get("factorial_analysis", {})),
            dependency_caveats=list(data.get("dependency_caveats", [])),
            version_stamp=data.get("version_stamp", {}),
        )


def _project_root() -> Path:
    """Return the repository root relative to this module."""
    return Path(__file__).resolve().parents[4]


def _read_winner(
    result_path: Path | None,
    default_path: str,
    expected_levels: tuple[str, ...],
    fallback: str,
    strict: bool,
) -> tuple[str, str | None]:
    """Resolve a machine-readable winner from a predecessor result JSON.

    Returns ``(value, caveat)``.  In strict mode a missing/invalid winner raises
    ``ValueError``.  In non-strict mode a failure returns ``fallback`` and a
    caveat string.
    """
    path = result_path or (_project_root() / default_path)
    caveat: str | None = None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if strict:
            raise ValueError(
                f"strict winner resolution failed: result file not found: {path}"
            ) from None
        caveat = f"fallback used: {path} not found; using {fallback!r}"
        logger.warning(caveat)
        return fallback, caveat
    except json.JSONDecodeError as exc:
        if strict:
            raise ValueError(
                f"strict winner resolution failed: invalid JSON in {path}: {exc}"
            ) from exc
        caveat = f"fallback used: {path} has invalid JSON; using {fallback!r}"
        logger.warning(caveat)
        return fallback, caveat

    candidate = None
    for key in ("disposition", "winner", "recommended_source"):
        if key in data:
            if key == "disposition" and isinstance(data[key], dict):
                candidate = data[key].get("recommended_source")
            else:
                candidate = data[key]
            if candidate is not None:
                break

    if candidate is None:
        if strict:
            raise ValueError(
                f"strict winner resolution failed: no winner field "
                f"(disposition.recommended_source, winner, or recommended_source) in {path}"
            )
        caveat = (
            f"fallback used: no winner field in {path}; using {fallback!r}"
        )
        logger.warning(caveat)
        return fallback, caveat

    if candidate not in expected_levels:
        if strict:
            raise ValueError(
                f"strict winner resolution failed: winner {candidate!r} in {path} "
                f"is not one of {expected_levels}"
            )
        caveat = (
            f"fallback used: winner {candidate!r} is not one of {expected_levels}; "
            f"using {fallback!r}"
        )
        logger.warning(caveat)
        return fallback, caveat

    return candidate, None


def resolve_action_init_winner(
    result_path: Path | None = None, *, strict: bool = True
) -> str:
    """Resolve the SLM-163 action-initialization winner.

    In strict mode the referenced JSON must contain a machine-readable winner
    field whose value is ``current_stub`` or ``schema_description``.  In
    non-strict mode a failure falls back to ``schema_description`` and logs a
    caveat.
    """
    value, _ = _read_winner(
        result_path,
        _SLM163_RESULT_JSON,
        ACTION_INIT_LEVELS,
        "schema_description",
        strict,
    )
    return value


def resolve_sibling_margin_winner(
    result_path: Path | None = None, *, strict: bool = True
) -> str:
    """Resolve the SLM-164 targeted-sibling-margin winner.

    In strict mode the referenced JSON must contain a machine-readable winner
    field whose value is ``none`` or ``targeted_weighted``.  In non-strict mode
    a failure falls back to ``targeted_weighted`` and logs a caveat.
    """
    value, _ = _read_winner(
        result_path,
        _SLM164_RESULT_JSON,
        SIBLING_MARGIN_LEVELS,
        "targeted_weighted",
        strict,
    )
    return value


def _level_value(level: str, levels: tuple[str, ...]) -> int:
    """Return 0/1 coded value for a factor level."""
    return levels.index(level)


def _cell_label(action_init: str, type_balance: str, sibling_margin: str, seed: int) -> str:
    return f"A_{action_init}__{type_balance}__{sibling_margin}__s{seed}"


def build_cells(seeds: tuple[int, ...] = _DEFAULT_SEEDS) -> tuple[InteractionCell, ...]:
    """Build the 2×2×2 factorial cells for the requested seeds."""
    cells: list[InteractionCell] = []
    for seed in seeds:
        for action_init, type_balance, sibling_margin in product(
            ACTION_INIT_LEVELS, TYPE_BALANCE_LEVELS, SIBLING_MARGIN_LEVELS
        ):
            is_balanced = type_balance == "e396_balanced"
            is_targeted = sibling_margin == "targeted_weighted"
            cell_id = _cell_label(action_init, type_balance, sibling_margin, seed)
            cells.append(
                InteractionCell(
                    cell_id=cell_id,
                    action_init=action_init,
                    type_balance=type_balance,
                    sibling_margin=sibling_margin,
                    seed=seed,
                    action_embedding_init=action_init,
                    slot_component_loss_weight=1.0,
                    slot_component_class_balance_power=1.0 if is_balanced else 0.0,
                    slot_component_owner_rare_threshold=3 if is_balanced else 0,
                    slot_component_owner_rare_multiplier=4 if is_balanced else 1,
                    legal_margin_mode=sibling_margin,
                    targeted_margin_value=1.0 if is_targeted else 0.0,
                    target_decisions=_DEFAULT_TARGET_DECISIONS,
                )
            )
    return tuple(cells)


def validate_manifest(cells: tuple[InteractionCell, ...]) -> list[str]:
    """Validate the factorial manifest."""
    errors: list[str] = []
    if not cells:
        errors.append("cells must not be empty")
    seen: set[str] = set()
    expected_counts = {
        "action_init": set(ACTION_INIT_LEVELS),
        "type_balance": set(TYPE_BALANCE_LEVELS),
        "sibling_margin": set(SIBLING_MARGIN_LEVELS),
    }
    for cell in cells:
        if cell.cell_id in seen:
            errors.append(f"duplicate cell_id: {cell.cell_id}")
        seen.add(cell.cell_id)
        if cell.action_init not in expected_counts["action_init"]:
            errors.append(f"{cell.cell_id}: invalid action_init {cell.action_init!r}")
        if cell.type_balance not in expected_counts["type_balance"]:
            errors.append(f"{cell.cell_id}: invalid type_balance {cell.type_balance!r}")
        if cell.sibling_margin not in expected_counts["sibling_margin"]:
            errors.append(
                f"{cell.cell_id}: invalid sibling_margin {cell.sibling_margin!r}"
            )
    return errors


def _hash_noise(payload: str, span: float = 0.01) -> float:
    """Deterministic noise in ``[-span, span]`` from ``payload``."""
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    normalized = int(digest[:16], 16) / (2 ** 64)
    return (normalized * 2.0 - 1.0) * span


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _simulate_cell(cell: InteractionCell) -> InteractionMetrics:
    """Deterministic, CPU-only simulator with known main effects and interaction."""
    base = 0.35
    description_gain = 0.05 if cell.action_init == "schema_description" else 0.0
    balance_gain = 0.04 if cell.type_balance == "e396_balanced" else 0.0
    margin_gain = 0.03 if cell.sibling_margin == "targeted_weighted" else 0.0
    interaction_gain = (
        0.06
        if (
            cell.action_init == "schema_description"
            and cell.type_balance == "e396_balanced"
            and cell.sibling_margin == "targeted_weighted"
        )
        else 0.0
    )
    seed_noise = _hash_noise(f"{cell.cell_id}:{MATRIX_VERSION}", span=0.01)
    rare_recall = _clamp(
        base + description_gain + balance_gain + margin_gain + interaction_gain + seed_noise
    )

    active_levers = sum(
        (
            cell.action_init == "schema_description",
            cell.type_balance == "e396_balanced",
            cell.sibling_margin == "targeted_weighted",
        )
    )
    wall_seconds = _clamp(
        0.5 + 0.1 * active_levers + _hash_noise(f"wall:{cell.cell_id}", span=0.02),
        low=0.01,
        high=10.0,
    )

    notes = [
        f"action_init={cell.action_init}",
        f"type_balance={cell.type_balance}",
        f"sibling_margin={cell.sibling_margin}",
        "fixture-only: deterministic main-effect + interaction simulator",
    ]

    return InteractionMetrics(
        cell_id=cell.cell_id,
        action_init=cell.action_init,
        type_balance=cell.type_balance,
        sibling_margin=cell.sibling_margin,
        seed=cell.seed,
        rare_component_recall=rare_recall,
        meaningful_program_rate=_clamp(rare_recall - 0.02 + _hash_noise(f"mp:{cell.cell_id}")),
        common_component_recall=_clamp(rare_recall + 0.08 + _hash_noise(f"cc:{cell.cell_id}")),
        parse_validity_rate=_clamp(rare_recall + 0.05 + _hash_noise(f"pv:{cell.cell_id}")),
        first_attempt_quality=_clamp(rare_recall - 0.01 + _hash_noise(f"faq:{cell.cell_id}")),
        target_decisions=cell.target_decisions,
        wall_seconds=wall_seconds,
        notes=notes,
    )


def _cell_means(
    cells: tuple[InteractionCell, ...], rows: list[InteractionMetrics]
) -> dict[tuple[int, int, int], float]:
    grouped: dict[tuple[int, int, int], list[float]] = {}
    for cell in cells:
        key = (
            _level_value(cell.action_init, ACTION_INIT_LEVELS),
            _level_value(cell.type_balance, TYPE_BALANCE_LEVELS),
            _level_value(cell.sibling_margin, SIBLING_MARGIN_LEVELS),
        )
        grouped.setdefault(key, [])
    for row in rows:
        key = (
            _level_value(row.action_init, ACTION_INIT_LEVELS),
            _level_value(row.type_balance, TYPE_BALANCE_LEVELS),
            _level_value(row.sibling_margin, SIBLING_MARGIN_LEVELS),
        )
        grouped[key].append(row.rare_component_recall)
    return {key: statistics.mean(values) for key, values in grouped.items()}


def _effect(
    means: dict[tuple[int, int, int], float],
    sign_fn: callable,
) -> float:
    """Signed contrast divided by 4 for a 2³ factorial effect estimate."""
    total = 0.0
    for key, mean in means.items():
        total += sign_fn(key) * mean
    return total / 4.0


def _key_to_label(key: tuple[int, int, int]) -> str:
    a = ACTION_INIT_LEVELS[key[0]]
    t = TYPE_BALANCE_LEVELS[key[1]]
    s = SIBLING_MARGIN_LEVELS[key[2]]
    return f"{a} × {t} × {s}"


def _analyze_factorial(
    cells: tuple[InteractionCell, ...], rows: list[InteractionMetrics]
) -> dict[str, Any]:
    means = _cell_means(cells, rows)

    main_action = _effect(means, lambda k: 1 if k[0] == 1 else -1)
    main_balance = _effect(means, lambda k: 1 if k[1] == 1 else -1)
    main_margin = _effect(means, lambda k: 1 if k[2] == 1 else -1)

    interaction_ab = _effect(means, lambda k: (1 if k[0] == 1 else -1) * (1 if k[1] == 1 else -1))
    interaction_am = _effect(means, lambda k: (1 if k[0] == 1 else -1) * (1 if k[2] == 1 else -1))
    interaction_bm = _effect(means, lambda k: (1 if k[1] == 1 else -1) * (1 if k[2] == 1 else -1))
    interaction_abc = _effect(
        means,
        lambda k: (1 if k[0] == 1 else -1)
        * (1 if k[1] == 1 else -1)
        * (1 if k[2] == 1 else -1),
    )

    full_key = (1, 1, 1)
    two_way_keys = [(1, 1, 0), (1, 0, 1), (0, 1, 1)]
    best_two_key = max(two_way_keys, key=lambda k: means[k])
    full_mean = means[full_key]
    best_two_mean = means[best_two_key]
    diff = full_mean - best_two_mean

    # Seed-level difference for a simple normal CI.
    full_rows = [r for r in rows if (r.action_init, r.type_balance, r.sibling_margin) == (ACTION_INIT_LEVELS[1], TYPE_BALANCE_LEVELS[1], SIBLING_MARGIN_LEVELS[1])]
    best_rows = [r for r in rows if (r.action_init, r.type_balance, r.sibling_margin) == (ACTION_INIT_LEVELS[best_two_key[0]], TYPE_BALANCE_LEVELS[best_two_key[1]], SIBLING_MARGIN_LEVELS[best_two_key[2]])]
    full_by_seed = {r.seed: r.rare_component_recall for r in full_rows}
    best_by_seed = {r.seed: r.rare_component_recall for r in best_rows}
    common_seeds = sorted(set(full_by_seed) & set(best_by_seed))
    seed_diffs = [full_by_seed[s] - best_by_seed[s] for s in common_seeds]
    n = len(seed_diffs)
    if n > 1:
        sd = statistics.stdev(seed_diffs) if n > 1 else 0.0
        se = sd / math.sqrt(n)
        ci_low = diff - 1.96 * se
        ci_high = diff + 1.96 * se
    else:
        ci_low = diff
        ci_high = diff

    equivalence_margin = 0.02
    if diff >= 0.05 and interaction_abc > 0:
        verdict = "synergistic"
    elif interaction_abc <= -equivalence_margin:
        verdict = "antagonistic"
    else:
        mains = {
            "description": main_action,
            "balance": main_balance,
            "margin": main_margin,
        }
        sorted_mains = sorted(mains.items(), key=lambda kv: kv[1], reverse=True)
        largest = sorted_mains[0][1]
        second_largest = sorted_mains[1][1]
        if largest - second_largest < equivalence_margin:
            verdict = "redundant"
        elif sorted_mains[0][0] == "description":
            verdict = "description_dominant"
        elif sorted_mains[0][0] == "balance":
            verdict = "balance_dominant"
        else:
            verdict = "margin_dominant"

    return {
        "cell_means": {
            _key_to_label(key): round(value, 6) for key, value in means.items()
        },
        "main_effects": {
            "action_init": round(main_action, 6),
            "type_balance": round(main_balance, 6),
            "sibling_margin": round(main_margin, 6),
        },
        "two_way_interactions": {
            "action_init_x_type_balance": round(interaction_ab, 6),
            "action_init_x_sibling_margin": round(interaction_am, 6),
            "type_balance_x_sibling_margin": round(interaction_bm, 6),
        },
        "three_way_interaction": round(interaction_abc, 6),
        "best_two_way_cell": _key_to_label(best_two_key),
        "best_two_way_mean": round(best_two_mean, 6),
        "full_cell_mean": round(full_mean, 6),
        "full_minus_best_two_way": round(diff, 6),
        "full_minus_best_two_way_ci_95": [round(ci_low, 6), round(ci_high, 6)],
        "verdict": verdict,
        "verdict_rule": (
            "synergistic if full-cell rare recall exceeds the best two-lever cell by >= 0.05 "
            "and the three-way interaction is positive; otherwise antagonistic if the three-way "
            "interaction is <= -0.02, redundant if the largest main effect is within 0.02 of the "
            "second largest, and otherwise dominated by the largest main effect."
        ),
    }


def run_fixture_campaign(
    cells: tuple[InteractionCell, ...] | None = None,
    *,
    run_id: str = "slm165-interaction-factorial",
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    strict_winners: bool = False,
) -> InteractionFactorialReport:
    """Run the SLM-165 2×2×2 interaction factorial fixture campaign."""
    cells = cells or build_cells(seeds)
    errors = validate_manifest(cells)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))

    caveats: list[str] = []
    try:
        resolve_action_init_winner(strict=strict_winners)
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        if strict_winners:
            raise
        caveats.append(f"action_init winner resolution fallback: {exc}")

    try:
        resolve_sibling_margin_winner(strict=strict_winners)
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        if strict_winners:
            raise
        caveats.append(f"sibling_margin winner resolution fallback: {exc}")

    rows = [_simulate_cell(cell) for cell in cells]
    factorial_analysis = _analyze_factorial(cells, rows)

    hypothesis = (
        "The combination of schema-description action initialization (SLM-163), "
        "E396 type balancing, and targeted legal-sibling margin (SLM-164) produces "
        "a positive three-way interaction on rare-component recall, exceeding any "
        "two-lever combination."
    )
    falsifier = (
        "The full three-lever cell does not outperform the best two-lever cell by "
        "at least 0.05 rare-component recall, or the three-way interaction is "
        "non-positive."
    )

    report = InteractionFactorialReport(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        status="fixture",
        claim_class="wiring",
        hypothesis=hypothesis,
        falsifier=falsifier,
        cells=cells,
        rows=rows,
        factorial_analysis=factorial_analysis,
        dependency_caveats=caveats,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm165_interaction_factorial",
        ),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm165_interaction_factorial_report.json")
    return report


def render_markdown(report: InteractionFactorialReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-165 (SDE1-03): 2×2×2 interaction factorial fixture ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`",
        "",
        f"Version: `{report.matrix_version}`",
        "",
        f"Status: **{report.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no production "
        "TwoTower wiring was touched, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        report.hypothesis,
        "",
        "## Falsifier",
        "",
        report.falsifier,
        "",
        "## Factorial cells",
        "",
        "| Cell | action_init | type_balance | sibling_margin | seed | action_embedding_init | balance_power | rare_threshold | rare_multiplier | legal_margin_mode | targeted_margin_value | target_decisions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in report.cells:
        lines.append(
            f"| {cell.cell_id} | {cell.action_init} | {cell.type_balance} | {cell.sibling_margin} | "
            f"{cell.seed} | {cell.action_embedding_init} | {cell.slot_component_class_balance_power} | "
            f"{cell.slot_component_owner_rare_threshold} | {cell.slot_component_owner_rare_multiplier} | "
            f"{cell.legal_margin_mode} | {cell.targeted_margin_value} | {cell.target_decisions} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Cell | Seed | rare_recall | meaningful_program_rate | common_recall | parse_validity | first_attempt_quality | wall_seconds |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        lines.append(
            f"| {row.cell_id} | {row.seed} | {row.rare_component_recall:.3f} | "
            f"{row.meaningful_program_rate:.3f} | {row.common_component_recall:.3f} | "
            f"{row.parse_validity_rate:.3f} | {row.first_attempt_quality:.3f} | {row.wall_seconds:.3f} |"
        )

    fa = report.factorial_analysis
    lines.extend(
        [
            "",
            "## Factorial analysis",
            "",
            "### Main effects",
            "",
            "| Factor | Effect |",
            "| --- | --- |",
        ]
    )
    for name, value in fa["main_effects"].items():
        lines.append(f"| {name} | {value:.4f} |")

    lines.extend(
        [
            "",
            "### Two-way interactions",
            "",
            "| Interaction | Effect |",
            "| --- | --- |",
        ]
    )
    for name, value in fa["two_way_interactions"].items():
        lines.append(f"| {name} | {value:.4f} |")

    lines.extend(
        [
            "",
            f"### Three-way interaction: {fa['three_way_interaction']:.4f}",
            "",
            f"Full-cell mean: {fa['full_cell_mean']:.4f}",
            "",
            f"Best two-way cell: {fa['best_two_way_cell']} (mean {fa['best_two_way_mean']:.4f})",
            "",
            f"Full − best two-way: {fa['full_minus_best_two_way']:.4f} "
            f"(95% CI: {fa['full_minus_best_two_way_ci_95']})",
            "",
            f"Verdict: **{fa['verdict']}**",
            "",
            f"Verdict rule: {fa['verdict_rule']}",
            "",
        ]
    )
    if report.dependency_caveats:
        lines.extend(
            [
                "## Dependency-resolution caveats",
                "",
            ]
        )
        for caveat in report.dependency_caveats:
            lines.append(f"- {caveat}")
        lines.append("")

    lines.extend(
        [
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This is a wiring fixture. The factorial cells, "
            "metrics, and analysis plumbing are exercised over a deterministic simulator, "
            "but no real model was trained or evaluated. The mechanism remains "
            "``retain_diagnostic`` / ``blocked_pending_real_model`` until a trained "
            "scorer and AgentV evaluation are available.",
            "",
            "## Honest caveats",
            "",
            "- Metrics are generated by a deterministic main-effect + interaction simulator, "
            "  not a trained model.",
            "- Seed noise is bounded to ±0.01 so the factorial structure dominates.",
            "- The synthetic simulator is tuned to make the full three-lever cell best on "
            "  rare-component recall; real measurements could differ.",
            "- No Pareto or ship-gate claim is made; this is wiring evidence only.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm165_interaction_factorial_fixture --mode plan-only",
            "python -m scripts.run_slm165_interaction_factorial_fixture --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
