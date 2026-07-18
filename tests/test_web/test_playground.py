"""Playground API smoke tests."""

from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")

from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT
from slm_training.web.app import create_app
from slm_training.web.service import PlaygroundService

CKPT = PLAYGROUND_DEMO_CHECKPOINT


def test_playground_demo_checkpoint_committed() -> None:
    assert CKPT.is_file(), "playground demo checkpoint must be committed under src/slm_training/resources/checkpoints/"
    assert CKPT.with_suffix(".tokenizer.json").is_file()
    assert CKPT.with_suffix(".meta.json").is_file()


def test_cpu_playground_falls_back_to_onnx_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from slm_training.models.onnx_inference import OnnxTwoTowerModel

    real_import = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "slm_training.models.twotower":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)

    assert isinstance(PlaygroundService(checkpoint=CKPT).load(), OnnxTwoTowerModel)


def test_playground_validation_falls_back_without_node_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slm_training.dsl import lang_core

    monkeypatch.setattr(lang_core, "bridge_available", lambda: False)
    service = PlaygroundService(
        checkpoint=Path("/nonexistent.pt"),
        generation_attempts_path=tmp_path / "attempts.jsonl",
    )
    model = MagicMock()
    model.config = SimpleNamespace(
        generate_max_attempts=1,
        grammar_finalize_on_last_attempt_only=False,
        grammar_finalize_validate=False,
    )
    model.generate.return_value = (
        'root = Card([title])\ntitle = TextContent(":hero.title")'
    )
    service._model = model  # noqa: SLF001

    result = service.generate("Hero card", max_attempts=1)

    assert result.valid is True
    assert result.stream["ok"] is True


def test_playground_health_and_generate(tmp_path) -> None:
    app = create_app(
        checkpoint=CKPT,
        device="cpu",
        generation_attempts_path=tmp_path / "generation_attempts.jsonl",
        bad_outputs_path=tmp_path / "bad_outputs.jsonl",
    )
    model = MagicMock()
    model.config = SimpleNamespace(
        generate_max_attempts=3,
        grammar_finalize_on_last_attempt_only=False,
        grammar_finalize_validate=False,
    )
    model.generate.return_value = (
        'root = Card([title])\ntitle = TextContent(":hero.title")'
    )
    app.state.service._model = model  # noqa: SLF001
    service = app.state.service
    assert service.info()["exists"] is True
    assert 'id="root"' in Path(
        "src/slm_training/web/static/app/index.html"
    ).read_text(encoding="utf-8")

    result = service.generate(
        "Hero card with title and body", grammar_constrained=True
    )
    assert result.valid is True
    assert "root" in result.openui
    for _ in range(2):
        sample = service.sample(
            auto_prompt=True, grammar_constrained=True, session_id="test"
        )
        if sample["valid"]:
            assert "root" in sample["openui"]
        else:
            assert sample["fallback_required"] is True
            assert len(sample["attempts"]) == 3


def test_react_playground_has_full_annotate_surface() -> None:
    app = create_app(checkpoint=CKPT, device="cpu")
    source = Path("src/apps/dashboard/src/pages/Playground.tsx").read_text(
        encoding="utf-8"
    )
    for marker in (
        'id="btnUp"',
        'id="btnDown"',
        'id="btnSaveCorrection"',
        'id="btnDiscardCorrection"',
        'id="annotatorIdentity"',
        'id="annotationToken"',
        'id="activityLog"',
        'id="dslHighlight"',
        'id="dslAutocomplete"',
        'id="dslDiagnostics"',
        'id="dslLintMount"',
        "trainingModelPipeline",
        "browserFallback",
        "persistHumanAnnotation",
        "correction_author",
        'mode: "shared"',
        "DiffusionCanvas",
        "AbortController",
        "activeControllerRef",
    ):
        assert marker in source
    assert 'event.key === "Tab"' in source
    assert "await go(1)" not in source
    assert "headers.Authorization" in source
    assert "completionItems" in source
    assert "human_corrected" in source
    assert not Path("src/slm_training/web/static/app.js").exists()
    classic_route = next(
        route for route in app.routes if getattr(route, "path", None) == "/playground/classic"
    )
    retired = classic_route.endpoint()
    assert retired.status_code == 308
    assert retired.headers["location"] == "/playground"
    editor_js = Path("src/slm_training/web/static/openui_editor.js").read_text(
        encoding="utf-8"
    )
    assert "function highlightOpenUI" in editor_js
    assert "function lintOpenUI" in editor_js
    assert "function completionItems" in editor_js
    browser_js = Path("src/slm_training/web/static/browser_inference.js").read_text(
        encoding="utf-8"
    )
    assert "LanguageModel" in browser_js
    assert "Previous server and browser attempts failed" in browser_js
    assert "OPENUI_REVIEW_SCHEMA" in browser_js
    assert "browser baseline reviewer" in browser_js
    assert "@huggingface/transformers@4.2.0" in browser_js
    # NPU first, WebGPU before webnn-gpu (same silicon, smaller q4f16 weights),
    # WASM always last; each execution provider gets a dtype it can execute.
    assert 'devices.push("webnn-npu")' in browser_js
    assert 'devices.push("webgpu")' in browser_js
    assert 'devices.push("webnn-gpu")' in browser_js
    assert 'devices.push("wasm")' in browser_js
    assert browser_js.index('devices.push("webnn-npu")') < browser_js.index(
        'devices.push("webgpu")'
    ) < browser_js.index('devices.push("webnn-gpu")') < browser_js.index(
        'devices.push("wasm")'
    )
    assert '"webnn-npu": "fp16"' in browser_js
    assert '"webnn-gpu": "fp16"' in browser_js
    assert 'webgpu: "q4f16"' in browser_js
    # The q4 variant needs GatherBlockQuantized, which the WASM execution
    # provider rejects; q8 keeps the ladder's terminal device functional.
    assert 'wasm: "q8"' in browser_js
    assert 'dtype: "q4"' not in browser_js
    assert "TRANSFORMERS_DEVICE_DTYPES" in browser_js
    assert "assertWebnnBackend" in browser_js
    assert "ml.createContext" in browser_js
    # The baseline model actually holds the DSL/review formats, and the working
    # device/dtype profile is remembered so a later visit initializes directly.
    assert "SmolLM2-360M-Instruct" in browser_js
    assert "twotower_browser_inference_profile_v1" in browser_js
    assert "wasm.numThreads = capabilities.wasmThreads" in browser_js
    assert "browserAccelerationCapabilities" in browser_js
    assert "RUN_INSIGHTS_SYSTEM_PROMPT" in browser_js
    assert "buildRunInsightsPrompt" in browser_js
    assert "parseRunInsightsResponse" in browser_js


def test_spa_playground_has_diffusion_progressive_render() -> None:
    source = Path("src/apps/dashboard/src/pages/Playground.tsx").read_text(encoding="utf-8")
    assert "DiffusionCanvas" in source
    assert "resolving changing blocks" in source
