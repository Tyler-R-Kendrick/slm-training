"""Tests for the SLM-294 frontier path of scripts/run_external_ceiling.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.run_external_ceiling as runner
from slm_training.evals.meaningful_program import binding_aware_meaningful_v2
from slm_training.evals.power_protocol import wilson_interval
from slm_training.harnesses.experiments.external_ceiling_matrix import (
    FRONTIER_SYSTEM_PROMPT,
    frontier_prompt_template_sha256,
    load_frozen_requests,
    run_frontier_chunk,
    run_score_pass,
)
from slm_training.models.external_scorer import (
    ExternalScorerConfig,
    FakeExternalScorer,
)

_GOLD = 'Stack([Text("GOLDSECRETXYZ")])'
_PLACEHOLDER = ":gold_secret_ph"


def _frozen_rows(n: int = 3) -> list[dict]:
    return [
        {
            "id": f"req_{i}",
            "prompt": f"Create a screen number {i} with a title.",
            "openui": _GOLD,
            "placeholders": [_PLACEHOLDER],
            "split": "held_out",
            "source": "fixture",
        }
        for i in range(n)
    ]


def _write_frozen(tmp_path: Path, rows: list[dict]) -> tuple[Path, str]:
    path = tmp_path / "frozen.jsonl"
    content = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")
    return path, hashlib.sha256(content.encode("utf-8")).hexdigest()


def _fake_scorer() -> FakeExternalScorer:
    return FakeExternalScorer(
        ExternalScorerConfig(model_id="fake/model", revision="main")
    )


class _RecordingScorer(FakeExternalScorer):
    def __init__(self, config: ExternalScorerConfig) -> None:
        super().__init__(config)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, max_new_tokens: int = 384) -> str:
        self.prompts.append(prompt)
        return super().generate(prompt, max_new_tokens=max_new_tokens)


class _ExplodingScorer(FakeExternalScorer):
    def __init__(self, config: ExternalScorerConfig, fail_after: int) -> None:
        super().__init__(config)
        self._fail_after = fail_after
        self._calls = 0

    def generate(self, prompt: str, *, max_new_tokens: int = 384) -> str:
        self._calls += 1
        if self._calls > self._fail_after:
            raise RuntimeError("simulated chunk kill")
        return super().generate(prompt, max_new_tokens=max_new_tokens)


def test_chunked_resume_unions_without_dupes(tmp_path: Path) -> None:
    frozen, sha = _write_frozen(tmp_path, _frozen_rows(3))
    out = tmp_path / "out"

    first = run_frontier_chunk(
        arm="B",
        output_dir=out,
        requests_file=frozen,
        expected_sha256=sha,
        chunk_size=2,
        scorer=_fake_scorer(),
    )
    assert first["processed"] == ["req_0", "req_1"]
    assert first["remaining"] == 1

    second = run_frontier_chunk(
        arm="B",
        output_dir=out,
        requests_file=frozen,
        expected_sha256=sha,
        chunk_size=2,
        scorer=_fake_scorer(),
    )
    assert second["processed"] == ["req_2"]
    assert second["remaining"] == 0

    third = run_frontier_chunk(
        arm="B",
        output_dir=out,
        requests_file=frozen,
        expected_sha256=sha,
        chunk_size=2,
        scorer=_fake_scorer(),
    )
    assert third["processed"] == []

    rows = [
        json.loads(line)
        for line in (out / "raw" / "B.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in rows] == ["req_0", "req_1", "req_2"]
    for row in rows:
        assert row["arm"] == "B"
        assert row["model_id"] == "fake/model"
        assert row["prompt_template_sha256"] == frontier_prompt_template_sha256()
        assert hashlib.sha256(row["raw_output"].encode()).hexdigest() == (
            row["raw_output_sha256"]
        )
        assert row["latency_ms"] >= 0.0
        assert row["started_at"] and row["finished_at"]


def test_killed_chunk_leaves_only_complete_records(tmp_path: Path) -> None:
    frozen, sha = _write_frozen(tmp_path, _frozen_rows(3))
    out = tmp_path / "out"
    scorer = _ExplodingScorer(
        ExternalScorerConfig(model_id="fake/model", revision="main"),
        fail_after=1,
    )
    with pytest.raises(RuntimeError, match="simulated chunk kill"):
        run_frontier_chunk(
            arm="B",
            output_dir=out,
            requests_file=frozen,
            expected_sha256=sha,
            chunk_size=3,
            scorer=scorer,
        )
    raw_path = out / "raw" / "B.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert [row["id"] for row in rows] == ["req_0"]
    # No partial temp file left behind.
    assert not list(raw_path.parent.glob("*.tmp"))
    # Resume skips the completed record.
    summary = run_frontier_chunk(
        arm="B",
        output_dir=out,
        requests_file=frozen,
        expected_sha256=sha,
        chunk_size=5,
        scorer=_fake_scorer(),
    )
    assert summary["processed"] == ["req_1", "req_2"]


def test_prompt_audit_excludes_gold_fields(tmp_path: Path) -> None:
    frozen, sha = _write_frozen(tmp_path, _frozen_rows(2))
    scorer = _RecordingScorer(
        ExternalScorerConfig(model_id="fake/model", revision="main")
    )
    run_frontier_chunk(
        arm="B",
        output_dir=tmp_path / "out",
        requests_file=frozen,
        expected_sha256=sha,
        chunk_size=5,
        scorer=scorer,
    )
    assert len(scorer.prompts) == 2
    for prompt in scorer.prompts:
        assert _GOLD not in prompt
        assert "GOLDSECRETXYZ" not in prompt
        assert _PLACEHOLDER not in prompt
        assert "openui" not in prompt
    assert "placeholder content" in FRONTIER_SYSTEM_PROMPT


def test_score_pass_aggregates_and_wilson(tmp_path: Path) -> None:
    rows = _frozen_rows(2)
    frozen, sha = _write_frozen(tmp_path, rows)
    out = tmp_path / "out"
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True)
    preds = {
        "req_0": 'root = Stack([v0])\nv0 = TextContent(":alpha")',
        "req_1": "this is not a program",
    }
    with raw_dir.joinpath("B.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "prompt": row["prompt"],
                        "arm": "B",
                        "raw_output": preds[row["id"]],
                    }
                )
                + "\n"
            )

    scoreboard = run_score_pass(
        output_dir=out, requests_file=frozen, expected_sha256=sha
    )
    assert (out / "scoreboard.json").exists()
    arm = scoreboard["arms"]["B"]
    assert arm["n"] == 2
    assert arm["syntax"] == {
        "successes": 1,
        "n": 2,
        "rate": 0.5,
        "wilson_95": wilson_interval(1, 2),
    }

    # Independent recomputation of the v2 strict count.
    from slm_training.dsl.schema import ExampleRecord

    expected_v2 = 0
    expected_v1 = 0
    from slm_training.harnesses.model_build.eval_runner import (
        meaningful_program_v1,
    )

    for row in rows:
        record = ExampleRecord(
            id=row["id"],
            prompt=row["prompt"],
            openui=row["openui"],
            placeholders=row["placeholders"],
            split=row["split"],
            source=row["source"],
        )
        expected_v2 += int(
            binding_aware_meaningful_v2(preds[row["id"]], record=record).verdict
        )
        v1_ok, _, _ = meaningful_program_v1(preds[row["id"]], gold=record)
        expected_v1 += int(v1_ok)
    assert arm["v2_strict"]["successes"] == expected_v2
    assert arm["v2_strict"]["wilson_95"] == wilson_interval(expected_v2, 2)
    assert arm["v1_verdict"]["successes"] == expected_v1
    assert arm["v1_verdict"]["wilson_95"] == wilson_interval(expected_v1, 2)

    # Idempotent: a second pass produces the identical scoreboard payload
    # (version_stamp carries a wall-clock timestamp, so compare without it).
    again = run_score_pass(output_dir=out, requests_file=frozen, expected_sha256=sha)
    strip = lambda sb: {k: v for k, v in sb.items() if k != "version_stamp"}  # noqa: E731
    assert strip(again) == strip(scoreboard)


def test_frozen_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    frozen, _sha = _write_frozen(tmp_path, _frozen_rows(1))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_frozen_requests(frozen, expected_sha256="0" * 64)
    code = runner.main(
        [
            "--mode",
            "frontier",
            "--arm",
            "B",
            "--requests-file",
            str(frozen),
            "--requests-sha256",
            "0" * 64,
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1


def test_cli_frontier_requires_arm(tmp_path: Path) -> None:
    code = runner.main(
        ["--mode", "frontier", "--output-dir", str(tmp_path / "out")]
    )
    assert code == 1


def test_cli_score_mode(tmp_path: Path) -> None:
    frozen, _sha = _write_frozen(tmp_path, _frozen_rows(1))
    out = tmp_path / "out"
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True)
    raw_dir.joinpath("C.jsonl").write_text(
        json.dumps(
            {
                "id": "req_0",
                "prompt": "p",
                "arm": "C",
                "raw_output": 'root = Stack([v0])\nv0 = TextContent(":alpha")',
            }
        )
        + "\n",
        encoding="utf-8",
    )
    default_sha = hashlib.sha256(frozen.read_bytes()).hexdigest()
    code = runner.main(
        [
            "--mode",
            "score",
            "--requests-file",
            str(frozen),
            "--requests-sha256",
            default_sha,
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads((out / "scoreboard.json").read_text())
    assert set(payload["arms"]) == {"C"}


def _arm(n=20, low=0.0, high=0.1, syntax=0.5):
    return {
        "n": n,
        "v2_strict": {"wilson_95": {"low": low, "high": high}},
        "syntax": {"rate": syntax},
    }


def test_disposition_scale_binding():
    from slm_training.harnesses.experiments.external_ceiling_matrix import (
        compute_disposition,
    )

    sb = {
        "arms": {
            "A": _arm(),
            "B": _arm(high=0.2),
            "C": _arm(low=0.3, high=0.6, syntax=0.9),
        }
    }
    out = compute_disposition(sb)
    assert out["disposition"] == "scale_binding"


def test_disposition_task_or_metric_binding():
    from slm_training.harnesses.experiments.external_ceiling_matrix import (
        compute_disposition,
    )

    sb = {
        "arms": {
            "A": _arm(),
            "B": _arm(high=0.2),
            "C": _arm(low=0.0, high=0.05, syntax=0.9),
        }
    }
    assert compute_disposition(sb)["disposition"] == "task_or_metric_binding"


def test_disposition_inconclusive_when_arm_incomplete():
    from slm_training.harnesses.experiments.external_ceiling_matrix import (
        compute_disposition,
    )

    sb = {"arms": {"A": _arm(), "B": _arm(), "C": _arm(n=3)}}
    out = compute_disposition(sb)
    assert out["disposition"] == "inconclusive"
    assert out["incomplete_arms"] == ["C"]
    assert "resolving_evidence" in out
    out2 = compute_disposition(sb, not_run=("B", "C"))
    assert out2["not_run_arms"] == ["B", "C"]
