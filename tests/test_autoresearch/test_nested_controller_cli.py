"""Nested runner consumes actual generated locked CLI, including global options."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import autoresearch, autotrain_cycle_prepare as prepare
from scripts.autoresearch_command_cursor import ContinuationGrant
from scripts.autotrain_nested_execution import _nested_argv
from tests.test_autoresearch.test_locked_prereg_supervisor import _fixture, _selection


def test_nested_argv_accepts_actual_locked_command_and_rejects_runtime_change(tmp_path, monkeypatch):
    plan, path, root, store, _ = _fixture(tmp_path, monkeypatch)
    selected = _selection(plan, path, root, store)
    monkeypatch.setattr(prepare, "resolved_continuation_grant", lambda *_: ContinuationGrant("fixture", 300))
    value = prepare.prepare_recorded_cycle(Path(plan["source_path"]), root, SimpleNamespace(), selected)
    command = value["arms"][value["order"][0]]["cmd"]
    argv = _nested_argv(command, store)
    args = autoresearch.build_parser().parse_args(argv)
    assert args.command == "run" and args.root == store.root.parent
    assert args.campaign_id == store.campaign_id and args.execute
    assert argv == command[3:]
    with pytest.raises(ValueError, match="interpreter_differs"):
        _nested_argv(["/unapproved/python", *command[1:]], store)
    changed = list(command)
    changed[4] = str(tmp_path / "another-campaign")
    with pytest.raises(ValueError, match="canonical_run_cli"):
        _nested_argv(changed, store)


def test_diagnostic_controller_stamp_cannot_claim_workload_membership(tmp_path, monkeypatch):
    from scripts import autotrain_nested_execution as nested
    from scripts.autotrain_cycle_lock import load_context

    plan, path, root, store, _ = _fixture(tmp_path, monkeypatch)
    selected = _selection(plan, path, root, store)
    monkeypatch.setattr(prepare, "resolved_continuation_grant", lambda *_: ContinuationGrant("fixture", 300))
    prepare.prepare_recorded_cycle(Path(plan["source_path"]), root, SimpleNamespace(), selected)
    # Formatting-only unit: transition authorization has separate live-worker tests.
    monkeypatch.setattr(nested, "logical_source", lambda _: Path(plan["source_path"]))
    binding = {"cwd": plan["source_path"], "source_digest": plan["source_digest"], "runtime_digest": "a" * 64}
    stamp = {"code_commit": "old-measurement-commit", "code_dirty": True,
             "components": {"fixture": "1"}, "source_authority": {"kind": "initial_release"}}
    with nested.execution_scope({"controller_execution": binding}):
        result = nested.diagnostic_provenance(store, stamp)
    assert result["version_stamp"]["code_commit"] == plan["source_commit"]
    assert result["version_stamp"]["code_dirty"] is False
    assert "source_authority" not in result["version_stamp"]
    assert result["measurement_source"] == load_context(store)["publication_source"]
    assert result["controller_execution"] == binding
    assert stamp["code_commit"] == "old-measurement-commit"
