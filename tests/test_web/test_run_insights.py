from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from slm_training.web.capabilities import Capabilities
from slm_training.web.observability import Readers
from slm_training.web.routes import _require_execution, capabilities
from slm_training.autoresearch.run_insights import RunInsightSubmission


def _write_metrics(root) -> None:
    run = root / "outputs" / "runs" / "smoke-1"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text(
        "".join(
            json.dumps({"step": step, "loss": loss}) + "\n"
            for step, loss in enumerate([1.0, 1.0, 1.01, 0.99, 1.0, 3.0], 1)
        ),
        encoding="utf-8",
    )


def test_run_detail_exposes_and_persists_browser_insights(tmp_path) -> None:
    _write_metrics(tmp_path)
    readers = Readers(tmp_path)
    detail = readers.run("smoke-1")
    report = detail["insights"]
    response = readers.save_run_insights(
        "smoke-1",
        RunInsightSubmission.model_validate({
            "source_fingerprint": report["source_fingerprint"],
            "provider": "browser",
            "runtime": "prompt-api",
            "generated": {
                "summary": "The loss spike is consistent with instability.",
                "causes": [
                    {
                        "category": "collapse",
                        "title": "Learning-rate instability",
                        "rationale": "The final point exceeds the rolling baseline.",
                        "evidence": ["loss.events[0]"],
                        "suggestion": "Test a lower learning rate on the same data.",
                        "confidence": 0.7,
                        "event_step": 6,
                    }
                ],
                "phase_suggestions": [],
            },
        }),
    )

    assert response["enrichment"]["runtime"] == "prompt-api"
    assert (tmp_path / "outputs" / "runs" / "smoke-1" / "run_insights.json").is_file()


def test_run_insight_writes_require_execution_and_current_fingerprint(tmp_path) -> None:
    _write_metrics(tmp_path)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(capabilities=Capabilities(False)))
    )
    with pytest.raises(HTTPException) as denied:
        _require_execution(request)
    assert denied.value.status_code == 403

    readers = Readers(tmp_path)
    with pytest.raises(ValueError, match="evidence changed"):
        readers.save_run_insights(
            "smoke-1",
            RunInsightSubmission.model_validate({
            "source_fingerprint": "0" * 64,
            "provider": "browser",
            "runtime": "prompt-api",
            "generated": {"summary": "Stale analysis."},
            }),
        )


def test_invalid_run_id_never_escapes_outputs(tmp_path) -> None:
    payload = Readers(tmp_path).run("../escape")
    assert payload["provenance"] == "missing"
    assert payload["insights"] is None

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(capabilities=Capabilities(False))
        )
    )
    assert capabilities(request)["run_insights"]["browser"] is True
