"""Tier-split scoreboard: generation metrics × formal preflight, core vs library.

Consumes an ordinary eval metrics dict (or per-row list) plus optional formal
preflight statuses and emits a scoreboard where **production core** and
**ecosystem library** never share a pass/fail bit.

This is a measurement seam only. It does not weaken ship gates, invent quality
scores, or let ecosystem inventory growth green a core formal preflight.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from slm_training.dsl.ecosystem_tier import (
    CORE_FORMAL_MODULES,
    ECOSYSTEM_FORMAL_MODULES,
    ECOSYSTEM_TIER_SCHEMA,
    assert_core_independent_of_library,
    formal_module_partition,
    inventory_snapshot,
    split_generation_metrics,
)
from slm_training.formal.checkers import (
    FormalProjectLock,
    ProcessOutcome,
    formal_process_budget,
    run_formal_process,
)
from slm_training.levers import MAX_RUN_SECONDS
from slm_training.versioning import build_version_stamp

SCOREBOARD_SCHEMA = "ecosystem_tier_scoreboard/v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
LEAN_ROOT = REPO_ROOT / "src" / "leverproof_lean"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_generation_rows(
    rows: list[Mapping[str, Any]],
    *,
    metric_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Mean numeric metrics across eval rows (None values skipped)."""

    if not rows:
        return {}
    keys: set[str] = set()
    if metric_keys is not None:
        keys.update(metric_keys)
    else:
        for row in rows:
            keys.update(str(k) for k in row)
    out: dict[str, Any] = {}
    for key in sorted(keys):
        vals: list[float] = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, bool):
                vals.append(1.0 if value else 0.0)
                continue
            if isinstance(value, (int, float)):
                vals.append(float(value))
        if vals:
            out[key] = _mean(vals)
            out[f"{key}__n"] = len(vals)
    return out


def _formal_status_from_mapping(
    statuses: Mapping[str, str] | None,
    modules: tuple[str, ...],
) -> dict[str, Any]:
    """Reduce per-module statuses to a tier rollup.

    Missing modules are ``unknown`` (fail closed for required core reporting).
    """

    statuses = dict(statuses or {})
    per_module: dict[str, str] = {}
    for module in modules:
        per_module[module] = str(statuses.get(module, "unknown"))
    values = list(per_module.values())
    if values and all(v in {"proved", "pass", "passed", "ok"} for v in values):
        rollup = "proved"
    elif any(v in {"refuted", "fail", "failed", "error"} for v in values):
        rollup = "failed"
    elif any(v == "unknown" for v in values):
        rollup = "unknown"
    else:
        rollup = "conditional"
    return {
        "rollup": rollup,
        "modules": per_module,
        "module_count": len(modules),
        "proved_count": sum(
            1 for v in values if v in {"proved", "pass", "passed", "ok"}
        ),
    }


def probe_lean_formal_statuses(
    *,
    timeout_s: float | None = None,
) -> dict[str, str]:
    """Build LeverProofLean and mark known modules ``proved`` when the build succeeds.

    Individual theorem-level audit remains ``make -C src/leverproof_lean proofs``;
    this probe only separates tier *modules* after a successful package build.
    """

    total, _interrupt_after, _grace = formal_process_budget(
        timeout_s if timeout_s is not None else min(120.0, MAX_RUN_SECONDS - 1)
    )
    try:
        with FormalProjectLock(LEAN_ROOT, timeout_seconds=total) as lock:
            result = run_formal_process(
                ["lake", "build", "LeverProofLean"],
                cwd=LEAN_ROOT,
                timeout_seconds=max(0.001, total - lock.wait_seconds),
            )
    except TimeoutError as exc:
        status = f"error:{type(exc).__name__}"
        return {
            **{m: status for m in CORE_FORMAL_MODULES},
            **{m: status for m in ECOSYSTEM_FORMAL_MODULES},
        }
    if result.timed_out:
        status = "error:TimeoutExpired"
        return {
            **{m: status for m in CORE_FORMAL_MODULES},
            **{m: status for m in ECOSYSTEM_FORMAL_MODULES},
        }
    if result.outcome is ProcessOutcome.LAUNCH_FAILED or result.returncode:
        status = "error:CalledProcessError"
        return {
            **{m: status for m in CORE_FORMAL_MODULES},
            **{m: status for m in ECOSYSTEM_FORMAL_MODULES},
        }
    # Build success ⇒ every registered formal module is currently proved in-tree.
    return {
        **{m: "proved" for m in CORE_FORMAL_MODULES},
        **{m: "proved" for m in ECOSYSTEM_FORMAL_MODULES},
    }


def build_ecosystem_tier_scoreboard(
    *,
    generation_metrics: Mapping[str, Any] | None = None,
    generation_rows: list[Mapping[str, Any]] | None = None,
    formal_statuses: Mapping[str, str] | None = None,
    probe_lean: bool = False,
    extra_components: Iterable[str] | None = None,
    extra_roles: Iterable[str] | None = None,
    extra_operators: Iterable[str] | None = None,
    previous_ecosystem_library_size: int | None = None,
    recipe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit the tier-split scoreboard.

    Parameters
    ----------
    generation_metrics:
        Already-aggregated metric dict (eval scoreboard fragment).
    generation_rows:
        Optional per-row metrics; aggregated then merged under generation_metrics.
    formal_statuses:
        Map from formal module name → status string. If omitted and
        ``probe_lean`` is true, runs a bounded lake build probe.
    """

    metrics: dict[str, Any] = dict(generation_metrics or {})
    if generation_rows:
        aggregated = aggregate_generation_rows(list(generation_rows))
        # Explicit aggregated metrics win over row means when both supplied.
        for key, value in aggregated.items():
            metrics.setdefault(key, value)

    if formal_statuses is None and probe_lean:
        formal_statuses = probe_lean_formal_statuses()
    formal_statuses = dict(formal_statuses or {})

    split = split_generation_metrics(metrics)
    inventory = inventory_snapshot(
        extra_components=extra_components or (),
        extra_roles=extra_roles or (),
        extra_operators=extra_operators or (),
    )
    eco_size = int(inventory["counts"]["components_ecosystem"]) + int(
        inventory["counts"]["operators_ecosystem"]
    ) + int(inventory["counts"]["roles_ecosystem"])

    core_formal = _formal_status_from_mapping(formal_statuses, CORE_FORMAL_MODULES)
    eco_formal = _formal_status_from_mapping(formal_statuses, ECOSYSTEM_FORMAL_MODULES)
    separation = assert_core_independent_of_library(
        core_formal_status=core_formal["rollup"],
        ecosystem_library_size=eco_size,
        previous_ecosystem_library_size=previous_ecosystem_library_size,
    )

    scoreboard = {
        "schema": SCOREBOARD_SCHEMA,
        "tier_schema": ECOSYSTEM_TIER_SCHEMA,
        "inventory": inventory,
        "formal_module_partition": {
            k: list(v) for k, v in formal_module_partition().items()
        },
        "production_core": {
            "generation_metrics": split["production_core"],
            "formal_preflight": core_formal,
        },
        "ecosystem_library": {
            "generation_metrics": split["ecosystem_library"],
            "formal_preflight": eco_formal,
        },
        "unassigned_generation_metrics": split["unassigned"],
        "separation": separation,
        "recipe": dict(recipe or {}),
        "honesty": {
            "fixture_demo": bool((recipe or {}).get("fixture_demo", False)),
            "ship_claim": False,
            "note": (
                "Tier split is diagnostic/measurement only. Ship gates remain "
                "the multi-suite policy in ship_gates.py; ecosystem growth "
                "never greens production-core formal preflight."
            ),
        },
    }
    scoreboard["version_stamp"] = build_version_stamp(
        "dsl.ecosystem_tier",
        "evals.ecosystem_tier_scoreboard",
    )
    return scoreboard


def write_scoreboard(path: Path, scoreboard: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(scoreboard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def scoreboard_markdown(scoreboard: Mapping[str, Any]) -> str:
    """Human scoreboard for docs/design measured-results."""

    inv = scoreboard.get("inventory", {}).get("counts", {})
    core_gen = scoreboard.get("production_core", {}).get("generation_metrics", {})
    eco_gen = scoreboard.get("ecosystem_library", {}).get("generation_metrics", {})
    core_f = scoreboard.get("production_core", {}).get("formal_preflight", {})
    eco_f = scoreboard.get("ecosystem_library", {}).get("formal_preflight", {})
    sep = scoreboard.get("separation", {})

    lines = [
        "# Ecosystem tier scoreboard",
        "",
        f"Schema: `{scoreboard.get('schema')}`",
        "",
        "## Inventory",
        "",
        "| Surface | Core | Ecosystem |",
        "| --- | ---: | ---: |",
        f"| Components | {inv.get('components_core', 0)} | {inv.get('components_ecosystem', 0)} |",
        f"| Roles | {inv.get('roles_core', 0)} | {inv.get('roles_ecosystem', 0)} |",
        f"| Operators | {inv.get('operators_core', 0)} | {inv.get('operators_ecosystem', 0)} |",
        f"| Formal modules | {inv.get('formal_modules_core', 0)} | {inv.get('formal_modules_ecosystem', 0)} |",
        "",
        "## Generation metrics (split)",
        "",
        "### Production core",
        "",
    ]
    if core_gen:
        for key, value in sorted(core_gen.items()):
            if key.endswith("__n"):
                continue
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- _(none supplied)_")
    lines.extend(["", "### Ecosystem library", ""])
    if eco_gen:
        for key, value in sorted(eco_gen.items()):
            if key.endswith("__n"):
                continue
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- _(none supplied)_")

    lines.extend(
        [
            "",
            "## Formal preflight (split)",
            "",
            (
                f"- **Production core rollup:** `{core_f.get('rollup')}` "
                f"({core_f.get('proved_count')}/{core_f.get('module_count')} modules)"
            ),
            (
                f"- **Ecosystem library rollup:** `{eco_f.get('rollup')}` "
                f"({eco_f.get('proved_count')}/{eco_f.get('module_count')} modules)"
            ),
            "",
            "## Separation",
            "",
            f"- Core formal ok: `{sep.get('core_formal_ok')}`",
            f"- Ecosystem library size: `{sep.get('ecosystem_library_size')}`",
            f"- Separation holds: `{sep.get('separation_holds')}`",
            f"- Note: {sep.get('note')}",
            "",
            (
                "Honesty: not a ship claim; production-core formal preflight is "
                "independent of library growth."
            ),
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "SCOREBOARD_SCHEMA",
    "aggregate_generation_rows",
    "build_ecosystem_tier_scoreboard",
    "probe_lean_formal_statuses",
    "scoreboard_markdown",
    "write_scoreboard",
]
