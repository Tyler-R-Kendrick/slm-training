"""Bad output quarantine for invalid OpenUI DSL."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from slm_training.annotations import (
    BadOutputRecord,
    append_bad_output,
    load_bad_outputs,
    new_bad_output_id,
    utc_now_iso,
)
from slm_training.dsl.lang_core import ParseError
from slm_training.web.app import create_app
from slm_training.web.service import PlaygroundService

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


@pytest.mark.skipif(not CKPT.exists(), reason="playground demo checkpoint missing")
def test_sample_api_never_returns_invalid(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad_outputs.jsonl"
    app = create_app(
        checkpoint=CKPT,
        device="cpu",
        bad_outputs_path=bad_path,
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
