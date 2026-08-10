"""CLI spend-safety tests for scripts.hf_jobs_screen (HF API fully mocked)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import hf_jobs_screen, hf_jobs_train


def _no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)


def _forbid_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(cmd: list[str], **kwargs: object) -> int:
        raise AssertionError(f"submission attempted: {cmd[:6]}")

    monkeypatch.setattr(hf_jobs_screen, "submit_jobs_command", _boom)


def test_dry_run_is_the_default_and_never_submits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    _forbid_submit(monkeypatch)
    rc = hf_jobs_screen.main(["--screen-id", "unit_screen"])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["job_count"] == 8  # --seeds default
    assert plan["flavor"] == hf_jobs_train.DEFAULT_FLAVOR
    assert plan["per_job_timeout"] == "30m"  # --max-minutes default
    assert plan["estimated_gpu_minutes_total"] == 8 * 30
    assert "--i-understand-this-costs-money" in plan["submit_requires"]
    seeds = [job["seed"] for job in plan["jobs"]]
    assert seeds == list(range(8))
    # Each job spec is a full hf jobs argv with the per-job cap and seed.
    for job in plan["jobs"]:
        cmd = job["command"]
        assert cmd[:3] == ["hf", "jobs", "run"]
        assert cmd[cmd.index("--timeout") + 1] == "30m"
        assert f"--seed {job['seed']}" in job["entrypoint"]
        assert "SLM_MAX_WALL_MINUTES=30" in job["entrypoint"]


def test_no_dry_run_without_money_flag_refuses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    _forbid_submit(monkeypatch)
    rc = hf_jobs_screen.main(["--screen-id", "unit_screen", "--no-dry-run"])
    assert rc == 2
    err = json.loads(capsys.readouterr().out)
    assert err["missing"] == "--i-understand-this-costs-money"


def test_no_dry_run_without_hf_token_refuses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_token(monkeypatch)
    _forbid_submit(monkeypatch)
    rc = hf_jobs_screen.main(
        [
            "--screen-id",
            "unit_screen",
            "--no-dry-run",
            "--i-understand-this-costs-money",
        ]
    )
    assert rc == 2
    err = json.loads(capsys.readouterr().out)
    assert "HF_TOKEN" in err["missing"]


def test_submit_requires_all_guards_and_prints_flavor_first(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    submitted: list[list[str]] = []

    def _fake_submit(cmd: list[str], **kwargs: object) -> int:
        submitted.append(list(cmd))
        return 0

    monkeypatch.setattr(hf_jobs_screen, "submit_jobs_command", _fake_submit)
    out = tmp_path / "submission.json"
    rc = hf_jobs_screen.main(
        [
            "--screen-id",
            "unit_screen",
            "--seeds",
            "3",
            "--max-minutes",
            "12",
            "--no-dry-run",
            "--i-understand-this-costs-money",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert len(submitted) == 3
    for cmd in submitted:
        assert cmd[cmd.index("--timeout") + 1] == "12m"
    lines = capsys.readouterr().out.strip().splitlines()
    # Spend transparency: flavor + job count precede any submission output.
    header = json.loads(lines[0])
    assert header["submitting"] is True
    assert header["flavor"] == hf_jobs_train.DEFAULT_FLAVOR
    assert header["job_count"] == 3
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert [s["seed"] for s in manifest["submissions"]] == [0, 1, 2]
    assert all(s["returncode"] == 0 for s in manifest["submissions"])


def test_hugging_face_hub_token_only_still_forwards_hf_token_secret(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """A caller with only HUGGING_FACE_HUB_TOKEN set must still submit with
    the token actually reaching the container: --secrets HF_TOKEN forwards
    the *local* env var named HF_TOKEN, so it must be aliased in-process
    before hf jobs run is invoked — never by putting the value on argv."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hub_test_token")
    submitted: list[list[str]] = []

    def _fake_submit(cmd: list[str], **kwargs: object) -> int:
        submitted.append(list(cmd))
        assert os.environ.get("HF_TOKEN") == "hub_test_token"
        return 0

    monkeypatch.setattr(hf_jobs_screen, "submit_jobs_command", _fake_submit)
    out = tmp_path / "submission.json"
    rc = hf_jobs_screen.main(
        [
            "--screen-id",
            "unit_screen",
            "--seeds",
            "2",
            "--no-dry-run",
            "--i-understand-this-costs-money",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert len(submitted) == 2
    for cmd in submitted:
        # Name-only form: the secret value itself never appears on argv.
        assert cmd[cmd.index("--secrets") + 1] == "HF_TOKEN"
        assert "hub_test_token" not in cmd


def test_config_file_merges_and_cli_overrides(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _forbid_submit(monkeypatch)
    config = tmp_path / "candidate.json"
    config.write_text(
        json.dumps(
            {
                "screen_id": "from_file",
                "seeds": 9,
                "steps": 50,
                "metric": "structural_similarity",
                "extra_train_args": "--ltr-loss-weight 0.5",
            }
        ),
        encoding="utf-8",
    )
    rc = hf_jobs_screen.main(
        ["--config", str(config), "--seeds", "8", "--suite", "held_out"]
    )
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["screen_id"] == "from_file"
    assert plan["job_count"] == 8  # CLI override wins
    assert plan["metric"] == "structural_similarity"  # file field survives
    assert plan["suite"] == "held_out"
    assert "--ltr-loss-weight 0.5 --seed 0" in plan["jobs"][0]["entrypoint"]


def test_trackio_mirror_is_wired_into_the_job_payload() -> None:
    config = hf_jobs_screen.ScreenConfig(screen_id="unit_screen", seeds=2)
    job = hf_jobs_screen.build_seed_job(config, 1)
    # Follows telemetry.TrackioSink: init(project=..., name=run) then log.
    assert "import trackio" in job["entrypoint"]
    assert "trackio.init(project='openui-hf-screening'" in job["entrypoint"]
    assert "trackio.log(" in job["entrypoint"]
    assert "TRACKIO_PROJECT=openui-hf-screening" in job["command"]
    # Per-seed evidence lands on the durable bucket mount.
    assert (
        f"{hf_jobs_train.BUCKET_MOUNT}/screening/unit_screen/seed_1.json"
        in job["entrypoint"]
    )


def test_hf_jobs_train_launcher_stays_byte_compatible(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The shared refactor must not change the training launcher's behavior."""
    # Default entrypoint still pins the canonical local wall-clock.
    entrypoint = hf_jobs_train.build_entrypoint_script(
        repo_url=hf_jobs_train.DEFAULT_REPO,
        branch="main",
        run_id="unit",
        steps=1,
        context_backend="hf",
        checkpoint_bucket=hf_jobs_train.DEFAULT_BUCKET,
        sync_checkpoints=True,
        skip_eval=True,
        extra_train_args="",
    )
    assert "export SLM_MAX_WALL_MINUTES=3" in entrypoint
    # The launcher wrapper still enforces the canonical Jobs timeout.
    with pytest.raises(ValueError, match="timeout must be"):
        hf_jobs_train.build_hf_jobs_command(
            flavor="a10g-large",
            timeout="30m",
            image=hf_jobs_train.DEFAULT_IMAGE,
            entrypoint="true",
            checkpoint_bucket=hf_jobs_train.DEFAULT_BUCKET,
            mount_bucket=False,
        )
    # And its --dry-run CLI shape is unchanged (prints the plan, no submit).
    rc = hf_jobs_train.main(["--dry-run", "--run-id", "unit", "--steps", "1"])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["timeout"] == hf_jobs_train.JOB_TIMEOUT
    assert plan["command"][:3] == ["hf", "jobs", "run"]
