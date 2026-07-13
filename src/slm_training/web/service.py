"""Model loading + generation service for the web playground."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.dsl.lang_core import ParseError, stream_check, validate
from slm_training.models.twotower import TwoTowerModel

DEFAULT_CHECKPOINT = Path("outputs/runs/playground_demo/checkpoints/last.pt")

EXAMPLE_PROMPTS = [
    "Hero card with title and body",
    "Primary call to action button",
    "Two feature cards stacked vertically",
    "Text blurb above a button",
    "Horizontal row of two buttons",
    "Pricing card with subscribe button",
]


@dataclass
class GenerateResult:
    prompt: str
    openui: str
    valid: bool
    error: str | None
    stream: dict[str, Any]
    serialized: str | None


class PlaygroundService:
    """Thread-safe lazy loader for a TwoTower checkpoint."""

    def __init__(self, checkpoint: Path | None = None, device: str = "cpu") -> None:
        self.checkpoint = Path(checkpoint or DEFAULT_CHECKPOINT)
        self.device = device
        self._model: TwoTowerModel | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.checkpoint.exists()

    def info(self) -> dict[str, Any]:
        meta_path = self.checkpoint.with_suffix(".meta.json")
        meta: dict[str, Any] = {}
        if meta_path.exists():
            import json

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "checkpoint": str(self.checkpoint),
            "exists": self.ready,
            "loaded": self._model is not None,
            "device": self.device,
            "examples": EXAMPLE_PROMPTS,
            "meta": meta,
        }

    def load(self) -> TwoTowerModel:
        with self._lock:
            if self._model is not None:
                return self._model
            if not self.checkpoint.exists():
                raise FileNotFoundError(
                    f"checkpoint not found: {self.checkpoint}. "
                    "Train one with scripts/train_model.py or the playground bootstrap."
                )
            self._model = TwoTowerModel.from_checkpoint(
                self.checkpoint, device=self.device
            )
            self._model.eval()
            return self._model

    def generate(
        self,
        prompt: str,
        *,
        grammar_constrained: bool = True,
        design_md: str | None = None,
    ) -> GenerateResult:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("prompt must be non-empty")
        model = self.load()
        if design_md is None:
            try:
                from slm_training.design_md import load_default_design_md

                design_md = load_default_design_md()
            except Exception:  # noqa: BLE001
                design_md = None
        openui = model.generate(
            prompt,
            grammar_constrained=grammar_constrained,
            design_md=design_md,
        )
        valid = False
        error: str | None = None
        serialized: str | None = None
        try:
            program = validate(openui)
            valid = True
            serialized = program.serialized or openui
        except ParseError as exc:
            error = str(exc)
        stream = stream_check(openui)
        return GenerateResult(
            prompt=prompt,
            openui=openui,
            valid=valid,
            error=error,
            stream={
                "ok": stream.get("ok"),
                "incomplete": stream.get("incomplete"),
                "has_root": stream.get("has_root"),
                "errors": stream.get("errors") or [],
                "unresolved": stream.get("unresolved") or [],
            },
            serialized=serialized,
        )
