"""Playwright subprocess adapter for renderer-first ProgramSpec captures."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from slm_training.bridge_utils import repo_root
from slm_training.data.progspec import ProgramSpec
from slm_training.data.render.schema import CaptureVariant, RenderCapture


@dataclass(frozen=True)
class CaptureConfig:
    viewports: tuple[tuple[int, int], ...] = ((390, 844), (1280, 720))
    themes: tuple[str, ...] = ("light", "dark")
    render_states: tuple[str, ...] = ("empty", "loading", "populated", "error")
    interaction_states: tuple[str, ...] = ("idle",)
    tile_overlap: int = 64

    def variants(self) -> tuple[CaptureVariant, ...]:
        return tuple(
            CaptureVariant(width, height, theme, render_state, interaction_state)
            for (width, height), theme, render_state, interaction_state in product(
                self.viewports,
                self.themes,
                self.render_states,
                self.interaction_states,
            )
        )


def capture_program(
    spec: ProgramSpec,
    *,
    output_dir: Path,
    config: CaptureConfig | None = None,
    state_sources: Mapping[str, str] | None = None,
    timeout_s: float = 120.0,
    root: Path | None = None,
) -> tuple[RenderCapture, ...]:
    """Capture the configured render matrix with the bundled preview and Chromium."""
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required for renderer-first capture")
    root = root or repo_root()
    script = root / "tools" / "openui_preview" / "capture.mjs"
    if not script.is_file():
        raise RuntimeError(f"OpenUI capture script not found: {script}")
    config = config or CaptureConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "program_spec": spec.to_dict(),
        "variants": [asdict(variant) for variant in config.variants()],
        "output_dir": str(output_dir.resolve()),
        "tile_overlap": config.tile_overlap,
        "state_sources": dict(state_sources or {}),
    }
    proc = subprocess.run(
        [node, str(script)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=root,
        timeout=timeout_s,
        check=False,
    )
    raw = (proc.stdout or "").strip()
    if proc.returncode or not raw:
        detail = (proc.stderr or raw or f"exit {proc.returncode}").strip()
        raise RuntimeError(f"OpenUI capture failed: {detail[:1000]}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"OpenUI capture returned invalid JSON: {raw[:500]}"
        ) from exc
    return tuple(RenderCapture.from_dict(item) for item in data.get("captures") or ())
