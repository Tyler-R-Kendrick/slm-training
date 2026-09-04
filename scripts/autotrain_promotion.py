"""Planning and scoring a promotion attempt.

One responsibility: what a promotion run consists of -- its chunked eval
commands, decode p95, scoreboard state, power feasibility, and the reasons a
measurement is judged incomplete.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.autotrain_io import read_json
from scripts.autotrain_paths import screening_expectations_path
from slm_training.autoresearch.storage import (
    CampaignStore,
)


def measured_promotion_decode_p95_seconds(
    root: Path | None,
) -> tuple[float | None, str | None]:
    """Newest measured eval ``latency_ms_p95`` (seconds) under the campaigns root.

    Promotion suites are preferred (``eval_held_out.json``) and the smoke suite
    is the fallback; a run whose p95 is null (all timeouts) is skipped.
    """

    if root is None or not Path(root).is_dir():
        return None, None
    candidates: list[tuple[float, Path]] = []
    for name in ("eval_held_out.json", "eval_smoke.json"):
        for path in Path(root).glob(f"*/runs/*/{name}"):
            try:
                candidates.append((path.stat().st_mtime, path))
            except OSError:
                continue
    for _mtime, path in sorted(candidates, key=lambda row: row[0], reverse=True):
        payload = read_json(path)
        value = payload.get("latency_ms_p95") if isinstance(payload, dict) else None
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        ):
            return float(value) / 1000.0, str(path)
    return None, None


def set_command_flag(cmd: list[str], flag: str, value: str) -> list[str]:
    """Return ``cmd`` with ``flag value`` set exactly once."""

    out = list(cmd)
    if flag in out:
        index = out.index(flag)
        if index + 1 < len(out):
            out[index + 1] = value
            return out
        out.append(value)
        return out
    out.extend([flag, value])
    return out


def command_flag_value(cmd: Sequence[str], flag: str) -> str | None:
    if flag in cmd:
        index = list(cmd).index(flag)
        if index + 1 < len(cmd):
            return str(cmd[index + 1])
    return None


def promotion_chunk_eval_command(
    *,
    root: Path,
    campaign_id: str,
    experiment_path: Path,
    run_dir: Path,
    plan: Mapping[str, Any],
) -> list[str]:
    """The locked arm's evaluate command, re-armed as one resumable chunk.

    The command is compiled from the same typed knobs the arm ran with (same
    checkpoint, suites, timeout, eval limit); only the resume flags and the
    chunk wall are (re)set, so no chunk can drift from the locked measurement.
    """

    from slm_training.autoresearch.engine import (
        compile_commands,
        is_latency_probe_command,
    )
    from slm_training.autoresearch.schemas import ExperimentSpec

    store = CampaignStore(campaign_id, root)
    campaign = store.load_campaign()
    experiment = ExperimentSpec.model_validate_json(
        Path(experiment_path).read_text(encoding="utf-8")
    )
    evaluates = [
        list(command)
        for command in compile_commands(campaign, experiment, output_root=root)
        if "scripts.evaluate_model" in command and not is_latency_probe_command(command)
    ]
    if not evaluates:
        raise ValueError(f"no evaluate_model command compiled for {experiment_path}")
    cmd = evaluates[-1]
    if "--partial-scoreboard" not in cmd:
        cmd.append("--partial-scoreboard")
    cmd = set_command_flag(cmd, "--resume-run", str(run_dir))
    cmd = set_command_flag(
        cmd, "--max-records-this-run", str(int(plan["records_per_run"]))
    )
    return set_command_flag(
        cmd, "--evaluation-wall-seconds", f"{float(plan['chunk_wall_seconds']):.6f}"
    )


def promotion_scoreboard_state(run_dir: Path) -> dict[str, Any]:
    """Completion state of an arm's (possibly partial) ``scoreboard.json``."""

    path = Path(run_dir) / "scoreboard.json"
    if not path.is_file():
        return {"exists": False, "complete": False, "pending": None, "decoded": None}
    board = read_json(path)
    resume = board.get("resume") if isinstance(board, dict) else None
    resume = resume if isinstance(resume, dict) else {}

    def _total(key: str) -> int | None:
        rows = resume.get(key)
        if not isinstance(rows, dict):
            return None
        return sum(int(value) for value in rows.values() if isinstance(value, int))

    return {
        "exists": True,
        # A scoreboard without the key predates resumable evals: complete.
        "complete": board.get("measurement_complete") is not False,
        "pending": _total("pending_record_n"),
        "decoded": _total("decoded_this_run_n"),
    }


def attach_promotion_chunks(
    delivery: dict[str, Any], ledger: Mapping[str, Any]
) -> dict[str, Any]:
    """Fold chunk-stage outcomes into the delivery as typed reasons.

    An exhausted chunk budget or a chunk harness failure marks the measurement
    incomplete (retryable, refunded); it is never a model reject.
    """

    reasons = list(delivery.get("reasons") or [])
    incomplete = False
    for eid, arm in (ledger.get("arms") or {}).items():
        status = str(arm.get("status") or "")
        if status == "chunk_budget_exhausted":
            reasons.append(
                f"measurement_incomplete:{eid}:chunk_budget_exhausted:"
                f"runs={arm.get('runs_used')}/{arm.get('run_budget')}"
            )
            incomplete = True
        elif status == "harness_failure":
            reasons.append(
                f"harness_failure:{eid}:promotion_chunk:{arm.get('error') or 'exit'}"
            )
            incomplete = True
        elif status == "no_checkpoint":
            reasons.append(f"measurement_incomplete:{eid}:no_checkpoint_for_chunks")
            incomplete = True
    out = {**delivery, "promotion_chunks": dict(ledger), "reasons": reasons}
    if incomplete:
        out["measurement_complete"] = False
    return out


def promotion_measurement_incomplete_reasons(
    camp_dir: Path,
    *,
    control_id: str,
    candidate_id: str,
    delivery: Mapping[str, Any],
) -> list[str]:
    """Typed reasons a promotion measurement is still partial evidence.

    A partial scoreboard (``measurement_complete: false``) or a spent chunk
    budget can never be disposed as a model verdict: the disposition is
    ``promotion_inconclusive`` (retryable) until a later run merges the suite
    to completion or the locked chunk budget is exhausted for good.
    """

    reasons: list[str] = []
    ledger = delivery.get("promotion_chunks")
    if isinstance(ledger, Mapping):
        for eid, arm in (ledger.get("arms") or {}).items():
            status = str(arm.get("status") or "")
            if status == "chunk_budget_exhausted":
                reasons.append(
                    f"measurement_incomplete:{eid}:chunk_budget_exhausted:"
                    f"runs={arm.get('runs_used')}/{arm.get('run_budget')}"
                )
    for run_id in (control_id, candidate_id):
        if not run_id:
            continue
        state = promotion_scoreboard_state(camp_dir / "runs" / run_id)
        if state["exists"] and not state["complete"]:
            reasons.append(
                f"measurement_incomplete:{run_id}:partial_scoreboard:"
                f"pending={state['pending']}"
            )
    return list(dict.fromkeys(reasons))


def merged_promotion_power_feasibility(
    camp_dir: Path,
    *,
    control_id: str,
    candidate_id: str,
    locked: dict[str, Any] | None,
    primary_metric: str,
) -> dict[str, Any] | None:
    """Power feasibility at the final merged n, not the planned n.

    The locked report admits the planned geometry; the disposition must judge
    the records actually completed in *both* arms of the primary suite (the
    paired sign test cannot use more pairs than the smaller arm completed).
    When either scoreboard is missing or still partial the locked report is
    returned unchanged (the incomplete path decides, never this gate).
    """

    if not isinstance(locked, dict):
        return None
    suite = primary_metric.rsplit(".", 1)[0] if "." in primary_metric else "held_out"
    merged_n: int | None = None
    for run_id in (control_id, candidate_id):
        if not run_id:
            return dict(locked)
        path = camp_dir / "runs" / run_id / "scoreboard.json"
        if not path.is_file():
            return dict(locked)
        board = read_json(path)
        if not isinstance(board, dict) or board.get("measurement_complete") is False:
            return dict(locked)
        suites = board.get("suites")
        row = suites.get(suite) if isinstance(suites, dict) else None
        completed = row.get("completed_document_n") if isinstance(row, dict) else None
        if type(completed) is not int or completed < 0:
            return dict(locked)
        merged_n = completed if merged_n is None else min(merged_n, completed)
    if merged_n is None:
        return dict(locked)
    from slm_training.autoresearch import evidence_ledger as _ev

    report = _ev.power_feasibility_report(
        max(1, int(merged_n)), _ev.parse_alpha(locked.get("alpha"))
    )
    if merged_n < 1:
        report["decisive"] = False
    return {
        **report,
        "locked_n": locked.get("n"),
        "locked_decisive": locked.get("decisive"),
        "merged_n": int(merged_n),
        "merged_suite": suite,
        "source": "merged_scoreboard",
    }


def stamp_promote_authority(row: dict[str, Any], authority: dict[str, str]) -> None:
    row["promote_authority_sha256"] = authority["sha256"]
    row["promote_authority"] = {
        "schema": authority.get("schema"),
        "climb_policy_sha256": authority.get("climb_policy_sha256"),
        "locked_expectations_sha256": authority.get("locked_expectations_sha256"),
        "harness_component": authority.get("harness_component"),
        "harness_component_version": authority.get("harness_component_version"),
    }
    row["promote_authority_stamped_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )


def locked_screening_expectations_sha256() -> str:
    """SHA-256 of the locked continuous screening expectation manifest."""
    return hashlib.sha256(screening_expectations_path().read_bytes()).hexdigest()


def load_promote_certificate(camp_dir: Path) -> dict[str, Any] | None:
    """Load campaign-local metric certificate if present (JSON object)."""
    candidates = [
        camp_dir / "metric-certificate.json",
        camp_dir / "artifacts" / "metric-certificate.json",
        camp_dir / "promote" / "metric-certificate.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        data = read_json(path)
        if data:
            return data
    return None


def empty_promotion_slot_falls_back(
    *,
    cadence_role: str,
    replay: object | None,
    promotion_target_available: bool,
    prior_screening_win_required: bool,
) -> bool:
    """Keep fresh hypotheses out of promotion suites and held-out selection."""

    return (
        replay is None
        and cadence_role == "promotion"
        and not promotion_target_available
        and prior_screening_win_required
    )
