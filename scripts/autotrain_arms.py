"""Describing and ordering the arms of one autotrain cycle.

One responsibility: what an arm *is* -- its extras, eval version, trainable
parameter count, whether it is a process arm or latency-only -- and the
counterbalanced order the cycle runs them in. Nothing here executes an arm.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.autotrain_io import (
    read_json,
)
from scripts.autotrain_levers import (
    bank_lever_categories,
)
from scripts.autotrain_metrics import (
    find_nested_key,
)
from slm_training.autoresearch.thrash_regime import (
    is_latency_only_arm,
)
from slm_training.levers import (
    CAPACITY_SCALING_LEVERS,
    require_size_matched_arms,
)


def counterbalanced_arm_order(
    control_id: str,
    candidate_id: str,
    *,
    cycle_index: int,
    seed: int,
    promotion_replicate_index: int | None = None,
) -> list[str]:
    """Serialize matched arms in deterministic AB/BA order without relabeling."""

    if cycle_index < 1:
        raise ValueError("cycle_index must be positive")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    order = [control_id, candidate_id]
    reverse = (
        promotion_replicate_index % 2 == 1
        if promotion_replicate_index is not None
        else cycle_index % 2 == 0
    )
    if reverse:
        order.reverse()
    return order


def capacity_view(knobs: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            name: knobs[name] if name in knobs else spec["baseline_value"]
            for name, spec in CAPACITY_SCALING_LEVERS.items()
        }
    )


def size_match_skip_reason(
    control_knobs: Mapping[str, Any], candidate_knobs: Mapping[str, Any]
) -> str | None:
    try:
        require_size_matched_arms(
            capacity_view(control_knobs),
            capacity_view(candidate_knobs),
            context="screening-multi-arm",
        )
    except ValueError as exc:
        return f"capacity_unmatched:{exc}"
    return None


def arm_trainable_params(camp_dir: Path, run_id: str) -> int:
    summary = read_json(camp_dir / "runs" / run_id / "train_summary.json")
    for key in ("trainable_params", "n_params", "parameter_count"):
        raw = summary.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return int(raw)
    return 0


def arm_eval_version(run_dir: Path) -> str | None:
    """Eval snapshot the arm was assigned (train summary, manifest, experiment)."""

    run_dir = Path(run_dir)
    camp_dir = run_dir.parent.parent
    candidates = (
        run_dir / "train_summary.json",
        camp_dir / "manifests" / f"{run_dir.name}.json",
        camp_dir / "experiments" / f"{run_dir.name}.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        found = find_nested_key(read_json(path), "eval_version")
        if isinstance(found, str) and found.strip():
            return found.strip()
    return None


def arm_swaps_train_corpus(
    extras: Mapping[str, Any] | None, *, control_train_version: str
) -> bool:
    """True when the arm's ``train_version`` differs from the control corpus.

    Warm-start cycles fork the champion for both arms on identical data, so a
    corpus-swap arm cannot be paired there (``warm_start:unequal_train_data``).
    """
    public = {k: v for k, v in (extras or {}).items() if not str(k).startswith("_")}
    if "train_version" not in public:
        return False
    return str(public["train_version"] or "") != str(control_train_version or "")


def latency_only_arm(extras: Mapping[str, Any] | None) -> bool:
    """Bank-row classifier: every public knob is a decode/run cost lever."""
    return is_latency_only_arm(extras, lever_categories=bank_lever_categories())


PROCESS_ARM_FLAG_KEYS = ("heal_resume", "_heal_resume", "process_arm")

PROCESS_ROLES = frozenset({"heal_resume", "rebuild_data", "first_snapshot"})


def is_process_arm(extras: Mapping[str, Any] | None) -> bool:
    """True for heal/resume/first-snapshot arms — execution, not confirmatory climb."""
    extras = extras or {}
    if any(extras.get(key) for key in PROCESS_ARM_FLAG_KEYS):
        return True
    role = extras.get("process_role")
    return isinstance(role, str) and role in PROCESS_ROLES


def current_rung_label() -> str:
    """Current uncertified rung from climb policy (I10 — never skip ahead)."""
    try:
        from slm_training.autoresearch.climb_policy import load_climb_policy

        return str(
            load_climb_policy().rung_gates.get("current_rung") or "grammar_2_ast"
        )
    except Exception:  # noqa: BLE001 — park prose must stay computable
        return "grammar_2_ast"


def compose_atom_extras(key: str, value: Any) -> dict[str, Any]:
    extras: dict[str, Any] = {key: value}
    if key == "semantic_contrast_loss_weight":
        extras.update(
            {
                "semantic_contrast_dir": (
                    "src/slm_training/resources/data/eval/openui_hard_valid_v1"
                ),
                "semantic_contrast_margin": 1.0,
                "semantic_contrast_fraction": 0.5,
                "batch_size": 3,
            }
        )
    return extras


def finalize_compose_extras(extras: dict[str, Any], *, slug: str) -> dict[str, Any]:
    out = dict(extras)
    out["_thrash_slug"] = slug
    if out.get("semantic_contrast_loss_weight") and int(out.get("batch_size") or 0) < 3:
        out["batch_size"] = 3
    return out


def apply_arm_extras(base_steps: int, extras: dict[str, Any]) -> dict[str, Any]:
    """Materialize arm knob extras (handles _steps_factor).

    Depth arms (factor >= 2) keep the ``base + 10`` depth-confound minimum; a
    floor-fill factor (< 2, ``steps-fill``) gets ``base + 1`` so the arm never
    spends past the fitted train floor it is charged against.
    """
    out = {k: v for k, v in extras.items() if not str(k).startswith("_")}
    factor = extras.get("_steps_factor")
    if factor is not None:
        bump = 10 if float(factor) >= 2 else 1
        out["steps"] = max(int(round(base_steps * float(factor))), base_steps + bump)
    return out


def arm_completed_n(metrics: Mapping[str, Any] | None) -> int | None:
    if not isinstance(metrics, Mapping):
        return None
    for key in (
        "completed_document_n",
        "smoke.completed_document_n",
        "n",
        "smoke.n",
    ):
        raw = metrics.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None
