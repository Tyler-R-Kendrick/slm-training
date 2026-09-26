"""Derive evaluator continuation only from a certified, committed command prefix."""

def _resumable_cursor_outcome(cursor):
    from scripts.autoresearch_continuation import _canonical, _pending_stage, _resume_commands
    from slm_training.autoresearch.engine import is_resumable_eval_command

    outcome, position, commands = cursor.outcome, cursor.position, cursor.inputs["commands"]
    if outcome is None or position >= len(commands) or not is_resumable_eval_command(commands[position]):
        return None
    pending = _pending_stage(outcome)
    expected = _resume_commands(commands, position, {})[0]
    if pending is not None and _canonical(pending["command"]) in (
        _canonical(commands[position]), _canonical(expected)
    ):
        return outcome
    # Replay the certified prefix, including trainer-validated resume commands.
    if not (cursor.unresolved and position and _completed_cursor_prefix(outcome, commands, position)):
        return None
    from slm_training.autoresearch.engine import _stage_artifact_path

    command = list(commands[position])
    if "--resume-run" in command:
        return outcome.model_copy(update={"status": "stopped", "error": "continuation_budget_pending"})
    artifact = _stage_artifact_path(command, cwd=cursor.inputs["cwd"])
    if artifact is None:
        return None  # No authenticated run destination; never guess one.
    stage = {"command": command, "exit_code": None, "interrupted": True,
             "measurement_complete": False, "resume_pending": True,
             "resume_validation": "required_by_evaluator_before_decode",
             "resume_command": [*command, "--resume-run", str(artifact.parent)]}
    return outcome.model_copy(update={"status": "stopped", "error": "continuation_budget_pending",
                                      "stage_telemetry": (*outcome.stage_telemetry, stage)})


def _completed_cursor_prefix(outcome, commands, position):
    from scripts.autoresearch_continuation import _canonical, _pending_stage, _resume_commands, _stage_complete

    index, pending = 0, {}
    for stage in outcome.stage_telemetry:
        if index >= position or _canonical(stage.get("command", ())) != _canonical(
                _resume_commands(commands, index, pending)[0]):
            return False
        if _stage_complete(stage):
            index, pending = index + 1, {}
        else:
            pending = _pending_stage(outcome.model_copy(update={"status": "stopped", "stage_telemetry": (stage,)}))
            if pending is None:
                return False
    return index == position and not pending
