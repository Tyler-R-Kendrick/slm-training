"""Statistics + collection tests for scripts.hf_jobs_screen."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import hf_jobs_screen


def _write_seed(
    directory: Path,
    seed: int,
    *,
    value: float | None,
    status: str = "complete",
    screen_id: str = "unit_screen",
    steps: int = 200,
) -> None:
    payload = {
        "schema": hf_jobs_screen.SEED_SCHEMA,
        "screen_id": screen_id,
        "seed": seed,
        "run_id": f"{screen_id}_s{seed}",
        "metric": "meaningful_program_rate",
        "suite": "smoke",
        "steps": steps,
        "value": value,
        "status": status,
    }
    (directory / f"seed_{seed}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_inline_permutation_p_value_separated_and_identical() -> None:
    low = [0.0, 0.01, 0.02, 0.0, 0.01, 0.02, 0.0, 0.01]
    high = [0.9, 0.91, 0.92, 0.9, 0.91, 0.92, 0.9, 0.91]
    separated = hf_jobs_screen.inline_permutation_p_value(high, low)
    assert separated["method"] == "exact_permutation"
    # Fully separated groups: only the observed split and its mirror match.
    assert separated["p_value"] == pytest.approx(2 / 12870)
    identical = hf_jobs_screen.inline_permutation_p_value(low, list(low))
    assert identical["p_value"] == pytest.approx(1.0)


def test_min_attainable_p_matches_rc1_arithmetic() -> None:
    # RC1: n=3 screening is undecidable at alpha=0.05 — arithmetic, not data.
    assert hf_jobs_screen.inline_min_attainable_p(3, 3) == pytest.approx(0.1)
    assert hf_jobs_screen.inline_min_attainable_p(3, 3) > 0.05
    # n=8 fanout is decidable.
    assert hf_jobs_screen.inline_min_attainable_p(8, 8) == pytest.approx(
        2 / 12870
    )
    assert hf_jobs_screen.inline_min_attainable_p(8, 8) < 0.05


def test_permutation_report_survives_missing_power_module() -> None:
    # The power module import is guarded; the report must always compute.
    report = hf_jobs_screen.permutation_report(
        [0.5, 0.6, 0.55, 0.65, 0.5, 0.6, 0.55, 0.65],
        [0.1, 0.2, 0.15, 0.25, 0.1, 0.2, 0.15, 0.25],
    )
    assert report["p_value"] is not None and report["p_value"] < 0.05
    assert report["decidable"] is True
    assert report["min_attainable_p"] <= 0.05
    assert report["source"] in (
        "inline_fallback",
        "slm_training.autoresearch.power",
    )


def test_collect_excludes_incomplete_seeds_from_evidence(
    tmp_path: Path,
) -> None:
    config = hf_jobs_screen.ScreenConfig(screen_id="unit_screen", seeds=8)
    collect = tmp_path / "screening" / "unit_screen"
    collect.mkdir(parents=True)
    complete_values = [0.4, 0.5, 0.45, 0.55, 0.4, 0.5]
    for seed, value in enumerate(complete_values):
        _write_seed(collect, seed, value=value)
    # Seed 6 hit the cap mid-eval and self-reported incomplete; seed 7 was
    # killed by the platform timeout and never wrote a file at all.
    _write_seed(collect, 6, value=None, status="incomplete")

    baseline = {"source": "unit", "values": [0.1, 0.12, 0.11, 0.13, 0.1, 0.12]}
    result = hf_jobs_screen.collect_results(
        config, collect_dir=collect, baseline=baseline
    )

    assert result["schema"] == "hf_jobs_screening/v1"
    assert result["n_seeds"] == 8
    assert result["n_obs"] == 8
    assert result["n_complete"] == 6
    assert result["n_incomplete_excluded"] == 2
    # Killed/timed-out seeds are never evidence: excluded everywhere.
    assert result["values"] == complete_values
    assert result["sum"] == pytest.approx(sum(complete_values))
    assert result["sumsq"] == pytest.approx(
        sum(v * v for v in complete_values)
    )
    assert result["mean"] == pytest.approx(sum(complete_values) / 6)
    statuses = {row["seed"]: row["status"] for row in result["per_seed"]}
    assert statuses[6] == "incomplete" and statuses[7] == "incomplete"
    assert "never evidence" in result["per_seed"][7]["reason"]
    assert "never evidence" in result["incomplete_policy"]
    # Permutation evidence computed on complete seeds only (6 vs 6).
    assert result["permutation"]["p_value"] is not None
    assert result["permutation"]["p_value"] < 0.05
    # version_stamp follows the quality-matrix-results convention.
    stamp = result["version_stamp"]
    assert stamp["stamp_schema"] == "version_stamp/v1"
    assert "code_commit" in stamp and "stamped_at" in stamp


def test_collect_rejects_mismatched_seed_identity(tmp_path: Path) -> None:
    config = hf_jobs_screen.ScreenConfig(screen_id="unit_screen", seeds=2)
    collect = tmp_path / "screening" / "unit_screen"
    collect.mkdir(parents=True)
    _write_seed(collect, 0, value=0.4)
    # A stale seed_1.json from a different screen (wrong screen_id) must
    # never silently enter this screen's aggregate.
    _write_seed(collect, 1, value=0.5, screen_id="some_other_screen")

    result = hf_jobs_screen.collect_results(config, collect_dir=collect, baseline=None)

    assert result["n_complete"] == 1
    assert result["values"] == [0.4]
    row1 = next(row for row in result["per_seed"] if row["seed"] == 1)
    assert row1["status"] == "incomplete"
    assert "identity mismatch" in row1["reason"]
    assert "screen_id" in row1["reason"]


def test_collect_rejects_non_finite_value(tmp_path: Path) -> None:
    config = hf_jobs_screen.ScreenConfig(screen_id="unit_screen", seeds=2)
    collect = tmp_path / "screening" / "unit_screen"
    collect.mkdir(parents=True)
    _write_seed(collect, 0, value=0.4)
    (collect / "seed_1.json").write_text(
        json.dumps(
            {
                "schema": hf_jobs_screen.SEED_SCHEMA,
                "screen_id": "unit_screen",
                "seed": 1,
                "run_id": "unit_screen_s1",
                "metric": "meaningful_program_rate",
                "suite": "smoke",
                "steps": 200,
                "value": float("nan"),
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )

    result = hf_jobs_screen.collect_results(config, collect_dir=collect, baseline=None)

    assert result["n_complete"] == 1
    row1 = next(row for row in result["per_seed"] if row["seed"] == 1)
    assert row1["status"] == "incomplete"
    assert "finite" in row1["reason"]


def test_collect_rejects_malformed_seed_json(tmp_path: Path) -> None:
    config = hf_jobs_screen.ScreenConfig(screen_id="unit_screen", seeds=2)
    collect = tmp_path / "screening" / "unit_screen"
    collect.mkdir(parents=True)
    (collect / "seed_0.json").write_text("{not json", encoding="utf-8")

    result = hf_jobs_screen.collect_results(config, collect_dir=collect, baseline=None)

    assert result["n_complete"] == 0
    assert "unreadable" in result["per_seed"][0]["reason"]


@pytest.mark.parametrize("body", ["[1, 2, 3]", "42", "null", '"a string"'])
def test_collect_rejects_non_object_seed_json(tmp_path: Path, body: str) -> None:
    """Valid JSON that isn't an object must degrade to incomplete, not crash
    (`payload.get(...)` on a list/scalar/None raises AttributeError)."""
    config = hf_jobs_screen.ScreenConfig(screen_id="unit_screen", seeds=2)
    collect = tmp_path / "screening" / "unit_screen"
    collect.mkdir(parents=True)
    (collect / "seed_0.json").write_text(body, encoding="utf-8")
    _write_seed(collect, 1, value=0.4)

    result = hf_jobs_screen.collect_results(config, collect_dir=collect, baseline=None)

    assert result["n_complete"] == 1
    assert result["per_seed"][0]["status"] == "incomplete"
    assert "JSON object" in result["per_seed"][0]["reason"]


def test_collect_rejects_seed_file_with_mismatched_steps(tmp_path: Path) -> None:
    """A stale seed_<n>.json from an earlier run with a different step count
    is a different training condition, not the same evidence — the identity
    check must catch it even though schema/screen_id/seed/suite/metric match."""
    config = hf_jobs_screen.ScreenConfig(screen_id="unit_screen", seeds=2, steps=400)
    collect = tmp_path / "screening" / "unit_screen"
    collect.mkdir(parents=True)
    _write_seed(collect, 0, value=0.4, steps=200)
    _write_seed(collect, 1, value=0.4, steps=400)

    result = hf_jobs_screen.collect_results(config, collect_dir=collect, baseline=None)

    assert result["n_complete"] == 1
    row0 = result["per_seed"][0]
    assert row0["status"] == "incomplete"
    assert "steps" in row0["reason"]


class TestScreenConfigValidation:
    def test_rejects_path_unsafe_screen_id(self) -> None:
        with pytest.raises(ValueError, match="screen_id"):
            hf_jobs_screen.ScreenConfig(screen_id="../../etc/passwd")

    def test_rejects_shell_metacharacters_in_screen_id(self) -> None:
        with pytest.raises(ValueError):
            hf_jobs_screen.ScreenConfig(screen_id="x$(rm -rf /)")

    def test_rejects_shell_metacharacters_in_metric(self) -> None:
        with pytest.raises(ValueError):
            hf_jobs_screen.ScreenConfig(metric='x"; import os; os.system("rm -rf /"); y="')

    def test_rejects_shell_metacharacters_in_suite(self) -> None:
        with pytest.raises(ValueError):
            hf_jobs_screen.ScreenConfig(suite="smoke`echo pwned`")

    def test_accepts_dotted_metric_names(self) -> None:
        config = hf_jobs_screen.ScreenConfig(metric="smoke.structural_similarity")
        assert config.metric == "smoke.structural_similarity"

    def test_max_minutes_capped_at_repo_hard_run_cap(self) -> None:
        from slm_training.levers import MAX_RUN_MINUTES

        hf_jobs_screen.ScreenConfig(max_minutes=MAX_RUN_MINUTES)
        with pytest.raises(ValueError):
            hf_jobs_screen.ScreenConfig(max_minutes=MAX_RUN_MINUTES + 1)


def test_collect_without_baseline_reports_no_p_value(tmp_path: Path) -> None:
    config = hf_jobs_screen.ScreenConfig(screen_id="unit_screen", seeds=2)
    collect = tmp_path / "screening" / "unit_screen"
    collect.mkdir(parents=True)
    _write_seed(collect, 0, value=0.4)
    _write_seed(collect, 1, value=0.5)
    result = hf_jobs_screen.collect_results(
        config, collect_dir=collect, baseline=None
    )
    assert result["n_complete"] == 2
    assert result["permutation"]["p_value"] is None
    assert "baseline" in result["permutation"]["reason"]


def test_collect_cli_writes_result_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collect = tmp_path / "mirror"
    collect.mkdir()
    for seed in range(4):
        _write_seed(collect, seed, value=0.3 + 0.01 * seed)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps([0.1, 0.11, 0.12, 0.13]), encoding="utf-8")
    out = tmp_path / "screening_result.json"
    rc = hf_jobs_screen.main(
        [
            "--screen-id",
            "unit_screen",
            "--seeds",
            "4",
            "--collect",
            str(collect),
            "--baseline",
            str(baseline),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_seeds"] == 4
    assert payload["baseline"]["values"] == [0.1, 0.11, 0.12, 0.13]
    assert payload["permutation"]["min_attainable_p"] == pytest.approx(2 / 70)
    printed = json.loads(capsys.readouterr().out)
    assert printed["output"] == str(out)


def test_baseline_accepts_prior_screening_result(tmp_path: Path) -> None:
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps({"schema": "hf_jobs_screening/v1", "values": [0.2, 0.3]}),
        encoding="utf-8",
    )
    loaded = hf_jobs_screen.load_baseline(prior)
    assert loaded["values"] == [0.2, 0.3]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"nope": 1}), encoding="utf-8")
    with pytest.raises(SystemExit):
        hf_jobs_screen.load_baseline(bad)
