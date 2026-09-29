import json
from types import SimpleNamespace

from scripts import merge_verification_controller as controller
from scripts.merge_verification_cli import cmd_verify_release


def test_canonical_parser_exposes_finite_release_command(tmp_path):
    from scripts.autoresearch import build_parser

    args = build_parser().parse_args(
        ["--root", str(tmp_path), "verify-release", "--source", str(tmp_path),
         "--state-dir", str(tmp_path / "cache"), "--total-seconds", "600",
         "--max-invocations", "8", "--max-step-seconds", "15", "--identity",
         "d" * 64, "--activity-id", "source-verification-" + "d" * 64]
    )
    assert args.func is cmd_verify_release
    assert args.total_seconds == 600 and args.max_invocations == 8
    assert args.max_step_seconds == 15 and not args.local_feedback
    assert args.identity == "d" * 64
    assert args.activity_id == "source-verification-" + args.identity


def test_stale_release_lock_exits_with_terminal_status(monkeypatch, capsys):
    def reject_stale_plan(_args):
        raise ValueError("locked verification source/config/grant changed; use an explicit successor job")

    monkeypatch.setattr(controller, "verify_release", reject_stale_plan)
    assert cmd_verify_release(SimpleNamespace()) == 20
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "invalid_evidence"
    assert result["verification_complete"] is False
    assert result["release_authorized"] is False
