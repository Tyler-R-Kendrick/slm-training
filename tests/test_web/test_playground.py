"""Playground API smoke tests."""

from __future__ import annotations


import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT
from slm_training.web.app import create_app

CKPT = PLAYGROUND_DEMO_CHECKPOINT


def test_playground_demo_checkpoint_committed() -> None:
    assert CKPT.is_file(), "playground demo checkpoint must be committed under fixtures/checkpoints/"
    assert CKPT.with_suffix(".tokenizer.json").is_file()
    assert CKPT.with_suffix(".meta.json").is_file()


def test_playground_health_and_generate(tmp_path) -> None:
    app = create_app(
        checkpoint=CKPT,
        device="cpu",
        generation_attempts_path=tmp_path / "generation_attempts.jsonl",
        bad_outputs_path=tmp_path / "bad_outputs.jsonl",
    )
    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["exists"] is True

    # Classic annotate playground lives at /playground/classic; "/" and
    # /playground are the SPA.
    page = client.get("/playground/classic")
    assert page.status_code == 200
    assert "TwoTower" in page.text

    res = client.post(
        "/api/generate",
        json={"prompt": "Hero card with title and body", "grammar_constrained": True},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["valid"] is True
    assert "root" in payload["openui"]
    # The API either returns a lint-clean server candidate or an explicit
    # three-attempt handoff for the mandatory browser baseline gate.
    for _ in range(2):
        again = client.post(
            "/api/sample",
            json={"auto_prompt": True, "grammar_constrained": True, "session_id": "test"},
        )
        assert again.status_code == 200
        sample = again.json()
        if sample["valid"]:
            assert "root" in sample["openui"]
        else:
            assert sample["fallback_required"] is True
            assert len(sample["attempts"]) == 3


def test_annotate_static_has_tab_toggle() -> None:
    app = create_app(checkpoint=CKPT, device="cpu")
    client = TestClient(app)
    js = client.get("/static/app.js")
    assert js.status_code == 200
    assert 'event.key === "Tab"' in js.text
    assert 'activeView === "render" ? "dsl" : "render"' in js.text
    # Tab must not steal focus from buttons / view tabs.
    assert "focus === cardEl" in js.text
    # Grade must not auto-advance after thumbs up/down.
    assert "await go(1)" not in js.text
    # Grading still gives swipe feedback while preserving the reviewed sample.
    assert "async function swipeAway" in js.text
    assert "function syncControls" in js.text
    assert "btnPrev.disabled = busy || editing || index === 0" in js.text
    assert "btnNext.disabled = busy || editing || index >= stack.length - 1" in js.text
    assert "previewRendering" in js.text
    assert "activityLog" in js.text
    assert 'event.key === "Enter" && !event.shiftKey && !event.isComposing' in js.text
    assert 'statusEl.textContent = "Note ready · use a grading hotkey"' in js.text
    html = client.get("/playground/classic")
    assert 'aria-label="Thumbs up"' in html.text
    assert 'aria-label="Thumbs down"' in html.text
    assert 'id="btnSaveCorrection"' in html.text
    assert 'id="btnDiscardCorrection"' in html.text
    assert 'aria-label="Editable OpenUI DSL"' in html.text
    assert 'id="annotatorIdentity"' in html.text
    assert 'id="activityLog"' in html.text
    assert 'role="log"' in html.text
    assert "app.js?v=20260713-12" in html.text
    assert "styles.css?v=20260713-4" in html.text
    assert 'id="dslHighlight"' in html.text
    assert 'id="dslAutocomplete"' in html.text
    assert 'id="dslDiagnostics"' in html.text
    assert 'id="dslLintMount"' in html.text
    assert "scheduleDslValidation" in js.text
    assert "completionItems" in js.text
    editor_js = client.get("/static/openui_editor.js")
    assert editor_js.status_code == 200
    assert "function highlightOpenUI" in editor_js.text
    assert "function lintOpenUI" in editor_js.text
    assert "function completionItems" in editor_js.text
    assert 'id="annotationToken"' in html.text
    assert "headers.Authorization" in js.text
    assert "browserFallback" in js.text
    assert 'mode: "shared"' in js.text
    assert "void preloadBrowserModel()" in js.text
    assert "Browser inference is preloaded once" in js.text
    assert js.text.count("createBrowserModelSession({") == 1
    assert "withBrowserModelSession" in js.text
    assert "persistHumanAnnotation" in js.text
    assert "human_corrected" in js.text
    assert "correction_author" in js.text
    browser_js = client.get("/static/browser_inference.js")
    assert browser_js.status_code == 200
    assert "LanguageModel" in browser_js.text
    assert "Previous server and browser attempts failed" in browser_js.text
    assert "OPENUI_REVIEW_SCHEMA" in browser_js.text
    assert "browser baseline reviewer" in browser_js.text
    assert "@huggingface/transformers@4.2.0" in browser_js.text
    assert 'devices.push("webnn-npu", "webnn-gpu")' in browser_js.text
    assert 'devices.push("webgpu")' in browser_js.text
    assert 'devices.push("wasm")' in browser_js.text
    assert "wasm.numThreads = capabilities.wasmThreads" in browser_js.text
    assert "browserAccelerationCapabilities" in browser_js.text
    assert 'id="modelSource"' in html.text
    assert "Training model · candidate under evaluation" in html.text
    assert "Browser baseline · on-device reference" in html.text
