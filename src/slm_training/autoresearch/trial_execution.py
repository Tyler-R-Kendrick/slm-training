"""Validated trial cursors schedule continuation; the trainer certifies exact state."""

from __future__ import annotations

from pathlib import Path

from slm_training.harness_core.checkpoint_bundle import validate_bundle


def command_value(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def incomplete_train_reason(command, parsed, *, cwd) -> str | None:
    """A nominally successful process still needs its declared training output."""
    if not isinstance(parsed, dict):
        return "training produced no typed summary"
    stopped_on = str(parsed.get("stopped_on") or "")
    requested_raw = command_value(command, "--steps")
    try:
        requested = int(requested_raw) if requested_raw is not None else None
    except ValueError:
        return f"training declared invalid --steps value {requested_raw!r}"
    completed = parsed.get("steps")
    if stopped_on != "steps":
        return f"training stopped_on={stopped_on or 'missing'} before declared steps"
    if requested is not None and (type(completed) is not int or completed != requested):
        return f"training completed {completed!r}/{requested} declared steps"
    checkpoint = parsed.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        return "training summary omitted checkpoint"
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(cwd) / checkpoint_path
    return None if checkpoint_path.is_file() else f"training checkpoint is missing: {checkpoint}"


def training_resume(command, parsed, *, cwd) -> dict:
    """Schedule verified operational yields, never token endpoints or killed work."""
    if not isinstance(parsed, dict) or parsed.get("stopped_on") not in {
        "wall_time_budget", "invocation_update_budget",
    }:
        return {}
    try:
        requested = int(command_value(command, "--steps") or "")
        completed = parsed.get("steps")
        if type(completed) is not int or not 0 < completed < requested:
            raise ValueError("training yield has no positive unfinished training cursor")
        if parsed["stopped_on"] == "invocation_update_budget":
            cap = int(command_value(command, "--max-updates-this-invocation") or "")
            start = parsed.get("invocation_start_step")
            if type(start) is not int or start < 0 or cap <= 0 or completed - start != cap:
                raise ValueError("invocation update yield does not match its locked cap")
        path = Path(parsed["checkpoint"])
        if not path.is_absolute():
            path = Path(cwd) / path
        directory = path.parent
        bundle, manifest = validate_bundle(directory.parent.parent, directory.name)
        meta = manifest["metadata"]
        if (path.name != "last.pt" or not manifest["resume_state_present"]
                or meta.get("role") != "trial_cursor"
                or meta.get("run_id") != command_value(command, "--run-id")
                or meta.get("optimizer_updates") != completed
                or not meta.get("data_manifest_sha")):
            raise ValueError("training yield does not bind the current resumable trial")
        resumed = list(command)
        for flag in ("--initialize-from", "--resume-from", "--max-wall-minutes"):
            if flag in resumed:
                index = resumed.index(flag)
                del resumed[index:index + 2]
        resumed.extend(("--resume-from", str(bundle / "last_full_state.pt")))
        return {"resume_pending": True, "resume_kind": "training",
                "resume_command": resumed, "progress_identity": directory.name,
                "completed_optimizer_updates": completed,
                "requested_optimizer_updates": requested,
                "resume_validation": "required_by_trainer_before_updates"}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"resume_refused": str(exc)}
