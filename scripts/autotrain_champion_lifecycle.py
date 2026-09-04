"""Establishing and refreshing the climbing champion.

One responsibility: making sure a champion exists and its recipe is current --
seeding a baseline from the first complete control run, and refreshing source
recipes when the champion's provenance moves.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.autotrain_campaign import warm_start_policy
from scripts.autotrain_levers import (
    knobs_fingerprint,
    lever_knobs,
    load_experiment_knobs,
)
from scripts.autotrain_paths import loop_campaign_dirs, loop_champion_dir
from scripts.autotrain_provenance import checkpoint_path_for_candidate
from slm_training.autoresearch.hillclimb import (
    CLIMB_CHAMPION_ADVANCE_STATUSES,
    ClimbChampionSidecar,
    assert_champion_eval_disjoint,
    climb_champion_checkpoint_path,
    load_climb_champion,
    seed_climb_champion,
    train_manifest_record_count,
)


def refresh_champion_source_recipes(root: Path, entries: list[dict[str, Any]]) -> bool:
    """Restore lever-complete queue recipes from immutable source experiments.

    Older queue writers projected only a subset of registered levers. Any such
    confirmation or promotion changed the treatment and is not evidence for the
    source winner, so reopen it at confirmation with the exact source recipes.
    """

    changed = False
    open_statuses = {
        "queued",
        "confirming",
        "confirmation_inconclusive",
        "confirmed",
        "promoting",
        "promotion_inconclusive",
        "harness_failure",
    }
    for row in entries:
        if row.get("status") not in open_statuses:
            continue
        source_dir = root / str(row.get("source_campaign_id") or "")
        candidate = lever_knobs(
            load_experiment_knobs(source_dir, str(row.get("source_candidate_id") or ""))
        )
        control = lever_knobs(
            load_experiment_knobs(source_dir, str(row.get("source_control_id") or ""))
        )
        if not candidate:
            continue
        if candidate == lever_knobs(row.get("knobs") or {}) and control == lever_knobs(
            row.get("control_knobs") or {}
        ):
            continue
        prior_status = str(row.get("status") or "")
        row.update(
            knobs=candidate,
            control_knobs=control,
            knobs_fingerprint=knobs_fingerprint(candidate),
            status="queued",
            confirm_attempts=0,
            promote_attempts=0,
            confirm_campaign_id=None,
            confirm_cycle_index=None,
            promotion_campaign_id=None,
            promotion_cycle_index=None,
            resolved_at=None,
            resolve_reasons=[
                "champion_recipe_repaired_from_source",
                f"invalidated_phase_status:{prior_status}",
            ],
        )
        row.pop("cert_policy", None)
        row.pop("formal_preflight_status", None)
        row.pop("last_harness_failure", None)
        row.pop("last_harness_failure_at", None)
        changed = True
        print(
            "CHAMPION_RECIPE_REPAIRED "
            f"entry_id={row.get('entry_id')} prior_status={prior_status} "
            f"source={row.get('source_campaign_id')}",
            flush=True,
        )
    return changed


CONTROL_RUN_SUFFIXES = ("-control", "_control")


def first_complete_control_run(
    root: Path, loop_id: str
) -> tuple[Path, Path, dict[str, Any]] | None:
    """First complete control run found scanning the newest cycle backwards.

    In a fresh loop this is the loop's first complete control; in a long
    history it is the latest one, so the seed matches the current recipe and
    corpus. Returns ``(campaign_dir, run_dir, train_summary)`` or None. A run
    counts as complete only when ``train_summary.json`` reports
    ``stopped_on == "steps"`` with ``steps > 0`` and ``checkpoints/last.pt``
    exists; wall or token-budget truncations are never a warm-start seed.
    """

    for camp_dir in loop_campaign_dirs(root, loop_id):
        runs_dir = camp_dir / "runs"
        if not runs_dir.is_dir():
            continue
        try:
            run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir())
        except OSError:
            continue
        for run_dir in run_dirs:
            if not run_dir.name.endswith(CONTROL_RUN_SUFFIXES):
                continue
            summary_path = run_dir / "train_summary.json"
            ckpt = run_dir / "checkpoints" / "last.pt"
            if not summary_path.is_file() or not ckpt.is_file():
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(summary, dict):
                continue
            try:
                steps = int(summary.get("steps") or 0)
            except (TypeError, ValueError):
                steps = 0
            if steps <= 0 or str(summary.get("stopped_on") or "") != "steps":
                continue
            return camp_dir, run_dir, summary
    return None


def seed_baseline_champion(root: Path, loop_id: str) -> ClimbChampionSidecar | None:
    """Seed a ``baseline_seed`` champion from the first complete control run."""

    loop_dir = loop_champion_dir(root, loop_id)
    found = first_complete_control_run(root, loop_id)
    if found is None:
        return None
    camp_dir, run_dir, summary = found
    try:
        steps = int(summary.get("steps") or 0)
    except (TypeError, ValueError):
        steps = 0
    train_dir = str(summary.get("train_dir") or "")
    record_count = train_manifest_record_count(train_dir) if train_dir else None
    if record_count is None:
        try:
            record_count = int(summary.get("record_count") or 0) or None
        except (TypeError, ValueError):
            record_count = None
    params = summary.get("trainable_params")
    if params is None:
        params = (summary.get("track") or {}).get("trainable_params")
    try:
        trainable = int(params) if params is not None else None
    except (TypeError, ValueError):
        trainable = None
    sidecar = seed_climb_champion(
        loop_dir,
        baseline_checkpoint=run_dir / "checkpoints" / "last.pt",
        source_campaign=camp_dir.name,
        extra_steps=steps,
        train_data_manifest_sha=str(
            summary.get("train_data_manifest_sha")
            or summary.get("data_manifest_sha")
            or ""
        ),
        record_count=int(record_count or 1),
        # Baseline knobs stay empty: the control keeps the baseline recipe and
        # every treatment is baseline + one residual (directive §2.6).
        knobs={},
        trainable_params=trainable,
        train_dir=train_dir or None,
    )
    if sidecar is not None:
        print(
            "CLIMB_CHAMPION_SEEDED "
            f"status={sidecar.status} campaign={camp_dir.name} run={run_dir.name} "
            f"steps={steps} record_count={record_count} "
            f"checkpoint={climb_champion_checkpoint_path(loop_dir)}",
            flush=True,
        )
    return sidecar


def ensure_climb_champion(
    *,
    root: Path,
    loop_id: str,
    queue_entries: Sequence[Mapping[str, Any]] | None,
    eval_data_manifest_sha: str | None,
    policy: Any | None = None,
) -> str | None:
    """Seed champion from a confirmed artifact, else the first complete control.

    Confirmed / climb_accepted / promoted queue rows seed a ``confirmed``
    champion. When none exists, the loop's first complete control run seeds a
    ``baseline_seed`` champion (warm start only; never confirmed or promoted).
    Returns a park reason or None to continue.
    """
    loop_dir = loop_champion_dir(root, loop_id)
    artifacts: list[dict[str, Any]] = []
    for row in queue_entries or ():
        if str(row.get("status") or "") not in CLIMB_CHAMPION_ADVANCE_STATUSES:
            continue
        item = dict(row)
        camp = str(row.get("confirm_campaign_id") or row.get("campaign_id") or "")
        cand = str(row.get("candidate_id") or "")
        ckpt = checkpoint_path_for_candidate(root, camp, cand) if camp else None
        if ckpt is not None:
            item["checkpoint"] = str(ckpt)
        artifacts.append(item)
    seed_climb_champion(loop_dir, confirmed_artifacts=artifacts)
    sidecar = load_climb_champion(loop_dir)
    if sidecar is None:
        warm = warm_start_policy(policy) if policy is not None else {}
        if bool(warm.get("enabled", True)) and bool(
            warm.get("seed_from_baseline_control", True)
        ):
            sidecar = seed_baseline_champion(root, loop_id)
    if (
        sidecar is not None
        and sidecar.train_data_manifest_sha
        and eval_data_manifest_sha
    ):
        assert_champion_eval_disjoint(
            train_data_manifest_sha=sidecar.train_data_manifest_sha,
            eval_data_manifest_sha=eval_data_manifest_sha,
        )
    return None
