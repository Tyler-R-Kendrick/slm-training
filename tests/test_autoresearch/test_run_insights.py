from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from slm_training.autoresearch.evidence import collect_evidence
from slm_training.autoresearch.run_insights import (
    GeneratedRunInsights,
    InsightCause,
    RunInsightSubmission,
    build_run_insights,
    enrich_with_openai,
    load_run_insights,
    save_enrichment,
)


def _write_run(root) -> None:
    run = root / "outputs" / "runs" / "smoke-1"
    run.mkdir(parents=True)
    losses = [1.0, 0.98, 1.01, 0.99, 1.0, 3.0, 3.2, 3.5, 3.9, 4.5]
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps({"step": i + 1, "loss": loss}) + "\n" for i, loss in enumerate(losses)),
        encoding="utf-8",
    )
    (run / "train_telemetry.json").write_text(
        json.dumps(
            {
                "spans": {
                    "forward": {"pct": 72.0, "mean_ms": 8.0, "total_ms": 80.0},
                    "batch_build": {"pct": 28.0, "mean_ms": 3.0, "total_ms": 30.0},
                }
            }
        ),
        encoding="utf-8",
    )


def test_run_insights_detect_collapse_and_persist(tmp_path) -> None:
    _write_run(tmp_path)
    run = tmp_path / "outputs" / "runs" / "smoke-1"

    report = load_run_insights(run, run_id="smoke-1")

    assert report["loss"]["status"] == "collapsed"
    assert {event["kind"] for event in report["loss"]["events"]} >= {"spike", "divergence"}
    assert report["phases"][0]["name"] == "forward"
    assert "AMP/compile" in report["phases"][0]["help"]
    assert report["persistence"]["persisted"] is True
    assert (run / "run_insights.json").is_file()

    submission = RunInsightSubmission(
        source_fingerprint=report["source_fingerprint"],
        provider="browser",
        runtime="prompt-api",
        generated=GeneratedRunInsights(
            summary="The loss curve diverged after step five.",
            causes=(
                InsightCause(
                    category="collapse",
                    title="Optimizer instability",
                    rationale="The loss spike precedes a sustained rise.",
                    evidence=("loss.events[0]",),
                    suggestion="Test a lower learning rate against the same data snapshot.",
                    confidence=0.7,
                    event_step=6,
                ),
            ),
        ),
    )
    saved = save_enrichment(run, run_id="smoke-1", submission=submission)
    assert saved["enrichment"]["provider"] == "browser"
    assert load_run_insights(run, run_id="smoke-1")["enrichment"] == saved["enrichment"]

    with pytest.raises(ValueError, match="evidence changed"):
        save_enrichment(
            run,
            run_id="smoke-1",
            submission=submission.model_copy(update={"source_fingerprint": "0" * 64}),
            scoreboard={"id": "changed"},
        )


def test_run_insights_become_prioritized_autoresearch_evidence(tmp_path) -> None:
    _write_run(tmp_path)
    run = tmp_path / "outputs" / "runs" / "smoke-1"
    load_run_insights(run, run_id="smoke-1")

    evidence = collect_evidence(("outputs",), repo_root=tmp_path)

    assert evidence.items[0].kind == "run_insight"
    assert "suggestion=" in evidence.items[0].summary


def test_openai_enrichment_is_structured_and_not_stored(tmp_path) -> None:
    _write_run(tmp_path)
    report = build_run_insights(
        tmp_path / "outputs" / "runs" / "smoke-1", run_id="smoke-1"
    )
    calls = []
    generated = GeneratedRunInsights(summary="A bounded, evidence-grounded explanation.")

    class Responses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_parsed=generated,
                id="resp-1",
                model="gpt-test",
                usage=SimpleNamespace(model_dump=lambda **_: {"total_tokens": 12}),
            )

    submission = enrich_with_openai(
        report,
        client=SimpleNamespace(responses=Responses()),
        model="gpt-test",
    )

    assert submission.provider == "openai"
    assert submission.generated == generated
    assert calls[0]["store"] is False
    assert calls[0]["text_format"] is GeneratedRunInsights
