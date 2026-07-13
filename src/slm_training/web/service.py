"""Model loading + generation service for the web playground."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.annotations import (
    DEFAULT_BAD_OUTPUTS_PATH,
    DEFAULT_FEEDBACK_PATH,
    DEFAULT_HUMAN_PAIRS_PATH,
    DEFAULT_HUMAN_TRAIN_PATH,
    AnnotationRecord,
    BadOutputRecord,
    append_bad_output,
    count_annotations,
    persist_annotation,
    new_annotation_id,
    new_bad_output_id,
    recent_annotations,
    utc_now_iso,
)
from slm_training.dsl.lang_core import ParseError, stream_check, validate
from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT
from slm_training.models.twotower import TwoTowerModel
from slm_training.web.prompts import EXAMPLE_PROMPTS, PromptCursor, load_prompt_bank

DEFAULT_CHECKPOINT = PLAYGROUND_DEMO_CHECKPOINT


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

    def __init__(
        self,
        checkpoint: Path | None = None,
        device: str = "cpu",
        annotations_path: Path | None = None,
        human_train_path: Path | None = None,
        human_pairs_path: Path | None = None,
        bad_outputs_path: Path | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint or DEFAULT_CHECKPOINT)
        self.device = device
        self.annotations_path = Path(annotations_path or DEFAULT_FEEDBACK_PATH)
        self.human_train_path = Path(human_train_path or DEFAULT_HUMAN_TRAIN_PATH)
        self.human_pairs_path = Path(human_pairs_path or DEFAULT_HUMAN_PAIRS_PATH)
        self.bad_outputs_path = Path(bad_outputs_path or DEFAULT_BAD_OUTPUTS_PATH)
        self._model: TwoTowerModel | None = None
        self._lock = threading.Lock()
        self._prompt_bank = load_prompt_bank()
        self._cursors: OrderedDict[str, PromptCursor] = OrderedDict()
        self._max_sessions = 1024

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
            "prompt_bank_size": len(self._prompt_bank),
            "annotations_path": str(self.annotations_path),
            "bad_outputs_path": str(self.bad_outputs_path),
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
            # Playground contract: grammar-constrained samples must be valid OpenUI.
            # Enable the deterministic DFA / LTR speculative layer + finalize gate.
            cfg = self._model.config
            cfg.grammar_constrained = True
            cfg.grammar_ltr_primary = True
            cfg.grammar_ltr_repair = True
            cfg.grammar_finalize_validate = True
            cfg.grammar_fastpath = True
            if int(cfg.grammar_ltr_max_tokens or 0) < 128:
                cfg.grammar_ltr_max_tokens = 192
            return self._model

    def next_prompt(self, session_id: str | None = None) -> dict[str, str]:
        sid = (session_id or "default").strip() or "default"
        with self._lock:
            cursor = self._cursors.get(sid)
            if cursor is None:
                cursor = PromptCursor(self._prompt_bank, session_id=sid, vary=True)
                self._cursors[sid] = cursor
                while len(self._cursors) > self._max_sessions:
                    self._cursors.popitem(last=False)
            else:
                self._cursors.move_to_end(sid)
            prompt = cursor.next()
        return {"prompt": prompt, "session_id": sid}

    def _quarantine_bad_output(
        self,
        *,
        prompt: str,
        openui: str,
        error: str,
        attempt: int,
        session_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> BadOutputRecord:
        record = BadOutputRecord(
            id=new_bad_output_id(),
            ts=utc_now_iso(),
            prompt=prompt.strip(),
            openui=openui.strip(),
            error=error,
            checkpoint=str(self.checkpoint),
            session_id=session_id,
            attempt=attempt,
            meta={
                "source": "playground_generate",
                "usable_for_training": True,
                "label": "invalid_openui",
                **dict(meta or {}),
            },
        )
        append_bad_output(self.bad_outputs_path, record)
        return record

    def generate(
        self,
        prompt: str,
        *,
        grammar_constrained: bool = True,
        design_md: str | None = None,
        max_attempts: int = 3,
        session_id: str | None = None,
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

        last_error: str | None = None
        openui = ""
        serialized: str | None = None
        valid = False
        with self._lock:
            attempts = max(1, int(max_attempts)) if grammar_constrained else 1
            for attempt_no in range(1, attempts + 1):
                # Reset per attempt so a generate_exception never reuses the
                # previous attempt's OpenUI in the bad-output quarantine.
                openui = ""
                serialized = None
                valid = False
                try:
                    openui = model.generate(
                        prompt,
                        grammar_constrained=grammar_constrained,
                        design_md=design_md,
                    )
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    if grammar_constrained:
                        self._quarantine_bad_output(
                            prompt=prompt,
                            openui="",
                            error=last_error,
                            attempt=attempt_no,
                            session_id=session_id,
                            meta={"failure": "generate_exception"},
                        )
                    if not grammar_constrained:
                        raise
                    continue
                try:
                    program = validate(openui)
                    valid = True
                    serialized = program.serialized or openui
                    last_error = None
                    break
                except ParseError as exc:
                    last_error = str(exc)
                    if grammar_constrained:
                        self._quarantine_bad_output(
                            prompt=prompt,
                            openui=openui,
                            error=last_error,
                            attempt=attempt_no,
                            session_id=session_id,
                            meta={"failure": "parse_error"},
                        )
                    openui = ""
                    if not grammar_constrained:
                        break
            if grammar_constrained and not valid:
                # Absolute contract: never hand the UI an invalid grammar-constrained sample.
                raise RuntimeError(
                    last_error
                    or "grammar-constrained generate failed to produce valid OpenUI"
                )

        stream = stream_check(openui)
        return GenerateResult(
            prompt=prompt,
            openui=openui,
            valid=valid,
            error=last_error,
            stream={
                "ok": stream.get("ok"),
                "incomplete": stream.get("incomplete"),
                "has_root": stream.get("has_root"),
                "errors": stream.get("errors") or [],
                "unresolved": stream.get("unresolved") or [],
            },
            serialized=serialized,
        )

    def sample(
        self,
        *,
        prompt: str | None = None,
        session_id: str | None = None,
        grammar_constrained: bool = True,
        design_md: str | None = None,
        auto_prompt: bool = True,
        max_rounds: int = 8,
    ) -> dict[str, Any]:
        """Generate a sample guaranteed valid when grammar_constrained is True."""
        sid = (session_id or "default").strip() or "default"
        rounds = max(1, int(max_rounds)) if grammar_constrained else 1
        last_error: str | None = None
        for _ in range(rounds):
            current_prompt = (prompt or "").strip()
            if not current_prompt:
                if not auto_prompt:
                    raise ValueError("prompt must be non-empty (or enable auto_prompt)")
                nxt = self.next_prompt(sid)
                current_prompt = nxt["prompt"]
                sid = nxt["session_id"]
            try:
                result = self.generate(
                    current_prompt,
                    grammar_constrained=grammar_constrained,
                    design_md=design_md,
                    session_id=sid,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if not grammar_constrained:
                    raise
                prompt = None
                continue
            if grammar_constrained and not result.valid:
                last_error = result.error or "invalid OpenUI"
                prompt = None
                continue
            return {
                "prompt": result.prompt,
                "openui": result.openui,
                "valid": result.valid,
                "error": result.error,
                "stream": result.stream,
                "serialized": result.serialized,
                "session_id": sid,
            }
        raise RuntimeError(
            last_error
            or "failed to produce a valid grammar-constrained sample"
        )

    def annotate(
        self,
        *,
        prompt: str,
        openui: str,
        rating: str,
        description: str | None = None,
        design_md: str | None = None,
        valid: bool | None = None,
        session_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rating_norm = (rating or "").strip().lower()
        if rating_norm not in {"up", "down"}:
            raise ValueError("rating must be 'up' or 'down'")
        prompt = (prompt or "").strip()
        openui = (openui or "").strip()
        if not prompt or not openui:
            raise ValueError("prompt and openui are required")
        record = AnnotationRecord(
            id=new_annotation_id(),
            ts=utc_now_iso(),
            prompt=prompt,
            openui=openui,
            rating=rating_norm,  # type: ignore[arg-type]
            description=(description or "").strip() or None,
            design_md=(design_md or "").strip() or None,
            valid=valid,
            checkpoint=str(self.checkpoint),
            session_id=session_id,
            meta=dict(meta or {}),
        )
        path, human_path, pair = persist_annotation(
            record,
            feedback_path=self.annotations_path,
            human_train_path=self.human_train_path,
            pairs_path=self.human_pairs_path,
        )
        return {
            "ok": True,
            "id": record.id,
            "path": str(path),
            "rating": record.rating,
            "human_train_path": str(human_path) if human_path else None,
            "preference_pair": pair.to_dict() if pair else None,
        }

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return [
            r.to_dict() for r in recent_annotations(self.annotations_path, limit=limit)
        ]

    def annotation_count(self) -> int:
        return count_annotations(self.annotations_path)
