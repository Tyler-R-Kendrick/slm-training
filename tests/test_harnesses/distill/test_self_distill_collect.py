"""AP-020 (SLM-309) on-policy abstract self-distillation collector tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from slm_training.evals.verifier_cascade import (
    Verdict,
    VerifierCascade,
    VerifierResultV1,
    VerifierStage,
    VerifierStageSpec,
)
from slm_training.harnesses.distill.self_distill_collect import (
    SelfDistillCollectConfig,
    collect_on_policy_traces,
    segment_ids_for_capture,
    training_example_from_capture,
)
from slm_training.harnesses.distill.trace_store import TraceStore
from slm_training.models.abstract_decode import (
    AbstractPhaseRecord,
    AbstractTraceCapture,
    AbstractTracedGeneration,
    DecodePhase,
)

VALID_OPENUI = "root = Separator()"
VALID_OPENUI_B = "root = Stack([])"
INVALID_OPENUI = "!! not openui !!"


def _fixture_verifier() -> VerifierCascade:
    """A hermetic stand-in for ``default_openui_cascade`` (no npm bridge needed).

    The real G0-G12 gate stack requires the openui_bridge Node toolchain; this
    fixture exercises the same accept/reject wiring in ``collect_on_policy_traces``
    without that dependency, following the repo's fixture-verifier convention
    (e.g. ``abstract_decode.py``'s injected ``forward_logits``).
    """

    def _evaluate(source: str, _context: object) -> VerifierResultV1:
        ok = source.startswith("root = ") and "!!" not in source
        return VerifierResultV1(
            stage_id="fixture",
            status=Verdict.PASS if ok else Verdict.FAIL,
            sound=not ok,
            reason=None if ok else "fixture verifier: not a well-formed program",
        )

    spec = VerifierStageSpec(
        stage_id="fixture", version="1", name="fixture", sound_fail=True, cache_policy="none"
    )
    return VerifierCascade([VerifierStage(spec, _evaluate)])


def _capture(
    *,
    prompt_tokens: tuple[int, ...] = (1, 2),
    abstract_tokens: tuple[int, ...] = (100, 5, 101),
    answer_tokens: tuple[int, ...] = (9, 10),
    forced_end: bool = False,
    plan_fingerprint: str = "plan-v1",
) -> AbstractTraceCapture:
    return AbstractTraceCapture(
        plan_fingerprint=plan_fingerprint,
        prompt=AbstractPhaseRecord(
            DecodePhase.PROMPT, prompt_tokens, tuple(None for _ in prompt_tokens)
        ),
        abstract=AbstractPhaseRecord(
            DecodePhase.ABSTRACT,
            abstract_tokens,
            tuple(None if i == 0 else -0.1 for i in range(len(abstract_tokens))),
        ),
        answer=AbstractPhaseRecord(
            DecodePhase.ANSWER, answer_tokens, tuple(-0.2 for _ in answer_tokens)
        ),
        abstract_termination="forced_cap" if forced_end else "sampled_end",
        forced_end=forced_end,
        stop_reason="eos",
    )


def _generation(text: str, **capture_kwargs: object) -> AbstractTracedGeneration:
    return AbstractTracedGeneration(text=text, capture=_capture(**capture_kwargs), valid=True)


def _fixed_generator(mapping: dict[str, AbstractTracedGeneration]):
    def _generate(prompt: str) -> AbstractTracedGeneration:
        return mapping[prompt]

    return _generate


def test_collect_accepts_verified_and_writes_auditable_rejections(tmp_path: Path) -> None:
    prompts = ["hero card", "cta button", "malformed thing"]
    generations = {
        "hero card": _generation(VALID_OPENUI, prompt_tokens=(1, 2)),
        "cta button": _generation(VALID_OPENUI_B, prompt_tokens=(3, 4)),
        "malformed thing": _generation(INVALID_OPENUI, prompt_tokens=(5, 6)),
    }
    store = TraceStore(tmp_path / "store")

    summary = collect_on_policy_traces(
        generate=_fixed_generator(generations),
        policy_checkpoint_sha="policy-a",
        prompts=prompts,
        store=store,
        verifier=_fixture_verifier(),
    )

    assert summary.candidate_count == 3
    assert summary.processed == 3
    assert summary.accepted == 2
    assert summary.rejected == 1
    assert summary.skipped_resumed == 0
    assert summary.duplicate == 0
    assert summary.yield_rate == pytest.approx(2 / 3)

    rows = list(store.iter_traces())
    assert len(rows) == 3
    rejected_rows = [r for r in rows if not r["labels"]["accepted"]]
    assert len(rejected_rows) == 1
    # Rejected samples remain in the store (auditable), with a reason.
    assert rejected_rows[0]["meta"]["prompt"] == "malformed thing"
    assert rejected_rows[0]["labels"]["reject_reason"] is not None
    assert rejected_rows[0]["capture"]["answer"]["token_count"] == 2


def test_collection_is_resumable_and_prompt_duplicate_free(tmp_path: Path) -> None:
    prompts = ["hero card", "cta button"]
    generations = {
        "hero card": _generation(VALID_OPENUI, prompt_tokens=(1, 2)),
        "cta button": _generation(VALID_OPENUI_B, prompt_tokens=(3, 4)),
    }
    store = TraceStore(tmp_path / "store")

    first = collect_on_policy_traces(
        generate=_fixed_generator(generations),
        policy_checkpoint_sha="policy-a",
        prompts=prompts,
        store=store,
        verifier=_fixture_verifier(),
    )
    assert first.processed == 2

    def _fail_if_called(prompt: str) -> AbstractTracedGeneration:
        raise AssertionError(f"resume must not regenerate {prompt!r}")

    second = collect_on_policy_traces(
        generate=_fail_if_called,
        policy_checkpoint_sha="policy-a",
        prompts=prompts,
        store=store,
        verifier=_fixture_verifier(),
    )
    assert second.processed == 0
    assert second.skipped_resumed == 2
    assert len(list(store.iter_traces())) == 2  # no duplicate rows appended


def test_duplicate_answer_across_different_prompts_is_flagged(tmp_path: Path) -> None:
    prompts = ["prompt one", "prompt two"]
    generations = {
        "prompt one": _generation(VALID_OPENUI, prompt_tokens=(1, 2)),
        "prompt two": _generation(VALID_OPENUI, prompt_tokens=(3, 4)),
    }
    store = TraceStore(tmp_path / "store")

    summary = collect_on_policy_traces(
        generate=_fixed_generator(generations),
        policy_checkpoint_sha="policy-a",
        prompts=prompts,
        store=store,
        verifier=_fixture_verifier(),
    )

    assert summary.processed == 1
    assert summary.accepted == 1
    assert summary.duplicate == 1
    rows = list(store.iter_traces())
    assert len(rows) == 2  # accepted row + an auditable duplicate-rejection row
    duplicate_rows = [r for r in rows if r["labels"]["reject_reason"] == "duplicate_answer"]
    assert len(duplicate_rows) == 1
    assert duplicate_rows[0]["meta"]["prompt"] == "prompt two"
    assert duplicate_rows[0]["labels"]["accepted"] is False


def test_generate_exception_is_isolated_and_auditable_and_retriable(tmp_path: Path) -> None:
    prompts = ["flaky prompt", "hero card"]
    generations = {
        "hero card": _generation(VALID_OPENUI, prompt_tokens=(3, 4)),
    }

    def _flaky_generate(prompt: str) -> AbstractTracedGeneration:
        if prompt == "flaky prompt":
            raise RuntimeError("simulated model inference failure")
        return generations[prompt]

    store = TraceStore(tmp_path / "store")
    summary = collect_on_policy_traces(
        generate=_flaky_generate,
        policy_checkpoint_sha="policy-a",
        prompts=prompts,
        store=store,
        verifier=_fixture_verifier(),
    )

    # The flaky prompt doesn't abort collection of the remaining prompt.
    assert summary.errored == 1
    assert summary.accepted == 1
    assert summary.processed == 1

    rows = list(store.iter_traces())
    error_rows = [r for r in rows if r["labels"]["reject_reason"] == "generation_error"]
    assert len(error_rows) == 1
    assert error_rows[0]["meta"]["prompt"] == "flaky prompt"
    assert error_rows[0]["error"]["type"] == "RuntimeError"
    # An error row must not carry a prompt_fingerprint: it is presumed
    # transient and must remain eligible for a retry on the next run.
    assert "prompt_fingerprint" not in error_rows[0]["meta"]

    # A resumed run retries the errored prompt (not skipped) rather than
    # regenerating the already-accepted one.
    calls: list[str] = []

    def _tracking_generate(prompt: str) -> AbstractTracedGeneration:
        calls.append(prompt)
        if prompt == "flaky prompt":
            return _generation(VALID_OPENUI_B, prompt_tokens=(1, 2))
        raise AssertionError(f"resume must not regenerate {prompt!r}")

    second = collect_on_policy_traces(
        generate=_tracking_generate,
        policy_checkpoint_sha="policy-a",
        prompts=prompts,
        store=store,
        verifier=_fixture_verifier(),
    )
    assert calls == ["flaky prompt"]
    assert second.accepted == 1
    assert second.skipped_resumed == 1


def test_verifier_exception_is_isolated_and_auditable(tmp_path: Path) -> None:
    prompts = ["hero card"]
    generations = {"hero card": _generation(VALID_OPENUI, prompt_tokens=(1, 2))}

    class _FlakyCascade:
        def evaluate(self, *, candidate_id: str, source: str):
            raise RuntimeError("simulated verifier failure")

    store = TraceStore(tmp_path / "store")
    summary = collect_on_policy_traces(
        generate=_fixed_generator(generations),
        policy_checkpoint_sha="policy-a",
        prompts=prompts,
        store=store,
        verifier=_FlakyCascade(),
    )

    assert summary.errored == 1
    assert summary.accepted == 0
    rows = list(store.iter_traces())
    assert len(rows) == 1
    assert rows[0]["labels"]["reject_reason"] == "verification_error"
    assert "prompt_fingerprint" not in rows[0]["meta"]


def test_disabled_collector_is_a_pure_no_op(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "store")

    def _fail_if_called(prompt: str) -> AbstractTracedGeneration:
        raise AssertionError("disabled collector must never call generate")

    summary = collect_on_policy_traces(
        generate=_fail_if_called,
        policy_checkpoint_sha="policy-a",
        prompts=["hero card"],
        store=store,
        verifier=_fixture_verifier(),
        config=SelfDistillCollectConfig(enabled=False),
    )

    assert summary.processed == 0
    assert summary.accepted == 0
    assert summary.rejected == 0
    assert list(store.iter_traces()) == []


def test_deterministic_sharding_selects_disjoint_reproducible_subsets(tmp_path: Path) -> None:
    prompts = [f"prompt {i}" for i in range(6)]
    generations = {
        p: _generation(f'root = Button("choice {i}")', prompt_tokens=(i, i + 1))
        for i, p in enumerate(prompts)
    }

    store_a = TraceStore(tmp_path / "shard_a")
    summary_a = collect_on_policy_traces(
        generate=_fixed_generator(generations),
        policy_checkpoint_sha="policy-a",
        prompts=prompts,
        store=store_a,
        verifier=_fixture_verifier(),
        config=SelfDistillCollectConfig(shard_index=0, num_shards=2),
    )
    store_b = TraceStore(tmp_path / "shard_b")
    summary_b = collect_on_policy_traces(
        generate=_fixed_generator(generations),
        policy_checkpoint_sha="policy-a",
        prompts=prompts,
        store=store_b,
        verifier=_fixture_verifier(),
        config=SelfDistillCollectConfig(shard_index=1, num_shards=2),
    )

    assert summary_a.shard_count == 3
    assert summary_b.shard_count == 3
    prompts_a = {row["meta"]["prompt"] for row in store_a.iter_traces()}
    prompts_b = {row["meta"]["prompt"] for row in store_b.iter_traces()}
    assert prompts_a.isdisjoint(prompts_b)
    assert prompts_a | prompts_b == set(prompts)

    # Re-running shard A against a fresh store reproduces the same subset.
    store_a2 = TraceStore(tmp_path / "shard_a_rerun")
    collect_on_policy_traces(
        generate=_fixed_generator(generations),
        policy_checkpoint_sha="policy-a",
        prompts=prompts,
        store=store_a2,
        verifier=_fixture_verifier(),
        config=SelfDistillCollectConfig(shard_index=0, num_shards=2),
    )
    assert {row["meta"]["prompt"] for row in store_a2.iter_traces()} == prompts_a


def test_configurable_acceptance_filters_reject_forced_end_and_short_traces(
    tmp_path: Path,
) -> None:
    prompts = ["forced", "short"]
    generations = {
        "forced": _generation(VALID_OPENUI, prompt_tokens=(1,), forced_end=True),
        "short": _generation(
            VALID_OPENUI_B, prompt_tokens=(2,), abstract_tokens=(100, 101)
        ),
    }
    store = TraceStore(tmp_path / "store")

    summary = collect_on_policy_traces(
        generate=_fixed_generator(generations),
        policy_checkpoint_sha="policy-a",
        prompts=prompts,
        store=store,
        verifier=_fixture_verifier(),
        config=SelfDistillCollectConfig(reject_forced_end=True, min_abstract_tokens=3),
    )

    assert summary.accepted == 0
    assert summary.rejected == 2
    rows = {row["meta"]["prompt"]: row for row in store.iter_traces()}
    assert rows["forced"]["labels"]["reject_reason"] == "forced_end_cap_rejected_by_config"
    assert (
        rows["short"]["labels"]["reject_reason"]
        == "abstract_trace_shorter_than_min_abstract_tokens"
    )


def test_training_example_from_capture_targets_only_abstract_and_answer() -> None:
    torch = pytest.importorskip("torch")
    from slm_training.models.block_attention import SegmentKind, loss_position_mask

    capture = _capture(prompt_tokens=(1, 2), abstract_tokens=(100, 5, 101), answer_tokens=(9, 10))
    example = training_example_from_capture(capture)

    assert example["input_ids"].tolist() == [[1, 2, 100, 5, 101, 9, 10]]
    segment_ids = example["segment_ids"]
    assert segment_ids.tolist() == [
        [
            SegmentKind.PROMPT,
            SegmentKind.PROMPT,
            SegmentKind.ABSTRACT,
            SegmentKind.ABSTRACT,
            SegmentKind.ABSTRACT,
            SegmentKind.TARGET,
            SegmentKind.TARGET,
        ]
    ]
    mask = loss_position_mask(segment_ids)
    assert int(mask.sum().item()) == 5  # 3 abstract + 2 target positions
    assert torch.equal(segment_ids_for_capture(capture), segment_ids)


def test_importing_abstract_decode_does_not_hit_a_circular_import() -> None:
    """Regression: models.abstract_decode -> models.causal_trace ->
    harnesses.distill.trace_store re-enters harnesses.distill.__init__ mid-init
    the first time anything imports either model module in a fresh process.
    If __init__.py ever goes back to eagerly importing self_distill_collect
    (which imports models.abstract_decode) at package-init time, that import
    fails with an ImportError for a partially initialized module. A same-process
    pytest run can't reliably reproduce this (modules may already be cached by
    an earlier test), so this spawns a fresh interpreter.
    """
    repo_root = Path(__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONPATH": "src"}
    result = subprocess.run(
        [sys.executable, "-c", "import slm_training.models.abstract_decode"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
