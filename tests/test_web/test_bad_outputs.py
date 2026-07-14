"""Bad output quarantine for invalid OpenUI DSL."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from slm_training.harnesses.annotations import (
    BadOutputRecord,
    append_bad_output,
    load_bad_outputs,
    load_generation_attempts,
    new_bad_output_id,
    utc_now_iso,
)
from slm_training.dsl.lang_core import ParseError
from slm_training.web.app import create_app
from slm_training.web.service import GenerationExhausted, PlaygroundService

CKPT = Path("outputs/runs/playground_demo/checkpoints/last.pt")


def test_append_and_load_bad_outputs(tmp_path: Path) -> None:
    path = tmp_path / "bad_outputs.jsonl"
    record = BadOutputRecord(
        id=new_bad_output_id(),
        ts=utc_now_iso(),
        prompt="Hero card",
        openui='root = TextContent(":broken")\n',
        error="unknown component",
        checkpoint="demo.pt",
        attempt=1,
        meta={"label": "invalid_openui"},
    )
    append_bad_output(path, record)
    rows = load_bad_outputs(path)
    assert len(rows) == 1
    assert rows[0].prompt == "Hero card"
    assert rows[0].meta["label"] == "invalid_openui"


def test_generate_quarantines_invalid_openui(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad_outputs.jsonl"
    service = PlaygroundService(
        checkpoint=CKPT if CKPT.exists() else Path("/nonexistent.pt"),
        device="cpu",
        bad_outputs_path=bad_path,
        generation_attempts_path=tmp_path / "attempts.jsonl",
    )
    mock_model = MagicMock()
    mock_model.config = MagicMock()
    service._model = mock_model  # noqa: SLF001

    invalid = 'root = TextContent(":x")\n'
    valid = (
        'root = Stack([hero], "column")\n'
        'hero_title = TextContent(":hero.title")\n'
        "hero = Card([hero_title])\n"
    )

    with patch.object(service, "load", return_value=mock_model):
        mock_model.generate.side_effect = [invalid, valid]
        with (
            patch(
                "slm_training.web.service.validate",
                side_effect=[ParseError("bad syntax"), MagicMock(serialized=valid)],
            ),
            patch(
                "slm_training.web.service.stream_check",
                return_value={"ok": True, "incomplete": False, "has_root": True, "errors": [], "unresolved": []},
            ),
        ):
            result = service.generate(
                "Hero card",
                grammar_constrained=True,
                max_attempts=3,
                session_id="unit",
            )

    assert result.valid is True
    assert result.openui == valid
    rows = load_bad_outputs(bad_path)
    assert len(rows) == 1
    assert rows[0].openui.strip() == invalid.strip()
    assert rows[0].session_id == "unit"
    assert rows[0].meta["failure"] == "parse_error"
    attempts = load_generation_attempts(tmp_path / "attempts.jsonl")
    assert [attempt.valid for attempt in attempts] == [False, True]
    assert attempts[1].prior_failures == ["bad syntax"]


def test_generate_exception_does_not_reuse_prior_openui(tmp_path: Path) -> None:
    """A generate_exception after a parse failure must not quarantine stale DSL."""
    bad_path = tmp_path / "bad_outputs.jsonl"
    service = PlaygroundService(
        checkpoint=CKPT if CKPT.exists() else Path("/nonexistent.pt"),
        device="cpu",
        bad_outputs_path=bad_path,
        generation_attempts_path=tmp_path / "attempts.jsonl",
    )
    mock_model = MagicMock()
    mock_model.config = MagicMock()
    service._model = mock_model  # noqa: SLF001

    invalid = 'root = TextContent(":x")\n'
    valid = (
        'root = Stack([hero], "column")\n'
        'hero_title = TextContent(":hero.title")\n'
        "hero = Card([hero_title])\n"
    )

    with patch.object(service, "load", return_value=mock_model):
        mock_model.generate.side_effect = [invalid, RuntimeError("boom"), valid]
        with (
            patch(
                "slm_training.web.service.validate",
                side_effect=[ParseError("bad syntax"), MagicMock(serialized=valid)],
            ),
            patch(
                "slm_training.web.service.stream_check",
                return_value={
                    "ok": True,
                    "incomplete": False,
                    "has_root": True,
                    "errors": [],
                    "unresolved": [],
                },
            ),
        ):
            result = service.generate(
                "Hero card",
                grammar_constrained=True,
                max_attempts=3,
                session_id="stale-check",
            )

    assert result.valid is True
    assert result.openui == valid
    rows = load_bad_outputs(bad_path)
    assert len(rows) == 2
    assert rows[0].meta["failure"] == "parse_error"
    assert rows[0].openui.strip() == invalid.strip()
    assert rows[1].meta["failure"] == "generate_exception"
    assert rows[1].openui == ""
    assert rows[1].error == "boom"
    attempts = load_generation_attempts(tmp_path / "attempts.jsonl")
    assert len(attempts) == 3
    assert attempts[1].prior_failures == ["bad syntax"]
    assert attempts[2].prior_failures == ["bad syntax", "boom"]


def test_three_server_failures_are_returned_for_browser_fallback(tmp_path: Path) -> None:
    service = PlaygroundService(
        checkpoint=Path("/nonexistent.pt"),
        bad_outputs_path=tmp_path / "bad.jsonl",
        generation_attempts_path=tmp_path / "attempts.jsonl",
    )
    mock_model = MagicMock()
    mock_model.config = MagicMock()
    mock_model.generate.side_effect = RuntimeError("real model failed")

    with patch.object(service, "load", return_value=mock_model):
        with pytest.raises(GenerationExhausted) as exc_info:
            service.generate("Hero card", session_id="fallback")

    assert len(exc_info.value.attempts) == 3
    assert [item["attempt"] for item in exc_info.value.attempts] == [1, 2, 3]
    attempts = load_generation_attempts(tmp_path / "attempts.jsonl")
    assert len(attempts) == 3
    assert attempts[0].prior_failures == []
    assert attempts[1].prior_failures == ["real model failed"]
    assert attempts[2].prior_failures == ["real model failed", "real model failed"]


def test_browser_attempt_validation_and_failure_context_are_stored(tmp_path: Path) -> None:
    service = PlaygroundService(
        checkpoint=Path("/nonexistent.pt"),
        generation_attempts_path=tmp_path / "attempts.jsonl",
    )
    result = service.record_browser_attempt(
        prompt="Hero card",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")',
        attempt=3,
        prior_failures=["server failed", "browser parse failed"],
        session_id="browser",
    )

    assert result["valid"] is True
    rows = load_generation_attempts(tmp_path / "attempts.jsonl")
    assert len(rows) == 1
    assert rows[0].source == "browser"
    assert rows[0].prior_failures == ["server failed", "browser parse failed"]
    assert rows[0].meta["label"] == "valid_openui"


def test_empty_browser_container_is_stored_as_failure(tmp_path: Path) -> None:
    service = PlaygroundService(
        checkpoint=Path("/nonexistent.pt"),
        generation_attempts_path=tmp_path / "attempts.jsonl",
    )

    result = service.record_browser_attempt(
        prompt="Hero card",
        openui='root = Stack([], null, "column", "m")',
        attempt=1,
    )

    assert result["valid"] is False
    assert result["error"] == "generated OpenUI contains an empty container"
    rows = load_generation_attempts(tmp_path / "attempts.jsonl")
    assert rows[0].valid is False
    assert rows[0].meta["label"] == "invalid_openui"


def test_sample_api_hands_three_failures_to_browser(tmp_path: Path) -> None:
    app = create_app(
        checkpoint=Path("/nonexistent.pt"),
        bad_outputs_path=tmp_path / "bad.jsonl",
        generation_attempts_path=tmp_path / "attempts.jsonl",
    )
    model = MagicMock()
    model.generate.side_effect = RuntimeError("decode exploded")
    app.state.service._model = model  # noqa: SLF001

    response = TestClient(app).post(
        "/api/sample",
        json={"prompt": "Hero card", "auto_prompt": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fallback_required"] is True
    assert payload["valid"] is False
    assert [item["attempt"] for item in payload["attempts"]] == [1, 2, 3]


def test_numbered_server_attempt_keeps_browser_rejection_context(tmp_path: Path) -> None:
    service = PlaygroundService(
        checkpoint=Path("/nonexistent.pt"),
        generation_attempts_path=tmp_path / "attempts.jsonl",
    )
    model = MagicMock()
    model.generate.return_value = (
        'root = Card([title])\ntitle = TextContent(":hero.title")'
    )
    service._model = model  # noqa: SLF001

    result = service.server_attempt(
        prompt="Hero card",
        attempt=2,
        prior_failures=["browser baseline rejected attempt 1"],
    )

    assert result["valid"] is True
    assert result["attempt"]["attempt"] == 2
    rows = load_generation_attempts(tmp_path / "attempts.jsonl")
    assert rows[0].attempt == 2
    assert rows[0].prior_failures == ["browser baseline rejected attempt 1"]
    assert rows[0].identities["output_generator"]["provider"] == "slm-training"
    assert result["identities"]["request_generator"]["kind"] == "user"
    inference_prompt = model.generate.call_args.args[0]
    assert "Hero card" in inference_prompt
    assert "browser baseline rejected attempt 1" in inference_prompt


def test_browser_baseline_review_is_linked_and_stored(tmp_path: Path) -> None:
    service = PlaygroundService(
        checkpoint=Path("/nonexistent.pt"),
        generation_attempts_path=tmp_path / "attempts.jsonl",
    )

    result = service.record_browser_review(
        generation_id="gen_srv_1_candidate",
        prompt="Hero card",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")',
        attempt=1,
        passed=False,
        score=0.42,
        reasons=["Missing the requested body content"],
        prior_failures=[],
        meta={"runtime": "prompt-api"},
    )

    assert result["passed"] is False
    assert result["score"] == 0.42
    rows = load_generation_attempts(tmp_path / "attempts.jsonl")
    assert rows[0].meta["kind"] == "browser_judgement"
    assert rows[0].meta["target_generation_id"] == "gen_srv_1_candidate"
    assert rows[0].meta["label"] == "browser_rejected"
    assert rows[0].identities["reviewer"]["provider"] == "browser-built-in-ai"


@pytest.mark.skipif(not CKPT.exists(), reason="playground demo checkpoint missing")
def test_sample_api_never_returns_invalid(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad_outputs.jsonl"
    app = create_app(
        checkpoint=CKPT,
        device="cpu",
        bad_outputs_path=bad_path,
        generation_attempts_path=tmp_path / "attempts.jsonl",
    )
    client = TestClient(app)
    for _ in range(3):
        res = client.post(
            "/api/sample",
            json={"session_id": "valid-only", "auto_prompt": True, "grammar_constrained": True},
        )
        assert res.status_code == 200
        payload = res.json()
        assert payload["valid"] is True
        assert "root" in payload["openui"]
