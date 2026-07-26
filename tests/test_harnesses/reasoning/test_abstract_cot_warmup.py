import json
from pathlib import Path

from scripts.run_abstract_cot_warmup import main
from slm_training.harnesses.reasoning.abstract_cot_warmup import AbstractWarmupCampaignV1, run_abstract_cot_warmup


def _runner_payload(iteration: int, parent: str | None, out: Path) -> dict:
    out.mkdir(parents=True)
    config_sha = AbstractWarmupCampaignV1("slm311-fixture", source_commit="1" * 40).fingerprint
    result = out / "result.json"
    result.write_text("{}\n", encoding="utf-8")
    phase = {
        "command": f"fixture --iteration {iteration}",
        "config_sha256": config_sha,
        "input_sha256": f"input-{iteration}",
        "output_sha256": f"output-{iteration}",
        "token_count": 10 * iteration,
        "result_ref": str(result),
    }
    return {
        "policy": f"policy-{iteration}",
        "command": f"fixture --iteration {iteration}",
        "token_count": 10 * iteration,
        "update_count": iteration,
        "result_ref": str(result),
        "phases": {"ap020_collection": phase, "ap019_segment_masked_sft": dict(phase)},
    }


def test_fixture_warmup_has_three_addressable_policies_and_resumes(tmp_path: Path) -> None:
    calls: list[tuple[int, str | None]] = []

    def runner(iteration: int, parent: str | None, out: Path) -> dict:
        calls.append((iteration, parent))
        return _runner_payload(iteration, parent, out)

    campaign = AbstractWarmupCampaignV1("slm311-fixture", source_commit="1" * 40)
    first = run_abstract_cot_warmup(campaign, root=tmp_path, runner=runner)
    assert [r.policy for r in first.iterations] == ["policy-1", "policy-2", "policy-3"]
    assert [r.parent_policy for r in first.iterations] == [None, "policy-1", "policy-2"]
    assert first.ran == 3 and calls == [(1, None), (2, "policy-1"), (3, "policy-2")]
    second = run_abstract_cot_warmup(campaign, root=tmp_path, runner=lambda *_: (_ for _ in ()).throw(AssertionError("must reuse")))
    assert second.reused == 3 and second.ran == 0


def test_fixture_cli_runs_three_iterations_and_resumes(tmp_path: Path, capsys) -> None:
    args = ["--mode", "fixture", "--no-evidence", "--root", str(tmp_path), "--campaign-id", "slm311-cli"]
    assert main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["ran"] == 3
    assert main([*args, "--resume"]) == 0
    assert json.loads(capsys.readouterr().out)["reused"] == 3


def test_incomplete_iteration_directory_is_never_overwritten(tmp_path: Path) -> None:
    campaign = AbstractWarmupCampaignV1("slm311-fixture", source_commit="1" * 40)
    incomplete = tmp_path / campaign.campaign_id / "iterations" / "001"
    incomplete.mkdir(parents=True)
    (incomplete / "policy.json").write_text('{"preserve": true}\n', encoding="utf-8")
    try:
        run_abstract_cot_warmup(campaign, root=tmp_path, runner=_runner_payload)
    except RuntimeError as exc:
        assert "without a completed ledger event" in str(exc)
    else:
        raise AssertionError("incomplete iteration must fail closed")
    assert (incomplete / "policy.json").read_text(encoding="utf-8") == '{"preserve": true}\n'
