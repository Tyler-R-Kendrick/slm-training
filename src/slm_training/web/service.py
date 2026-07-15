"""Model loading + generation service for the web playground."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from slm_training.harnesses.annotations import (
    DEFAULT_BAD_OUTPUTS_PATH,
    DEFAULT_FEEDBACK_PATH,
    DEFAULT_GENERATION_ATTEMPTS_PATH,
    DEFAULT_HUMAN_PAIRS_PATH,
    DEFAULT_HUMAN_TRAIN_PATH,
    AnnotationRecord,
    BadOutputRecord,
    GenerationAttemptRecord,
    append_bad_output,
    new_annotation_id,
    new_bad_output_id,
    new_generation_attempt_id,
    utc_now_iso,
)
from slm_training.harnesses.annotations.store import AnnotationStore, FileAnnotationStore
from slm_training.dsl.parser import ParseError, stream_check, validate
from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT
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
    attempts: list[dict[str, Any]]


class GenerationExhausted(RuntimeError):
    """All real-model attempts failed validation or inference."""

    def __init__(
        self,
        message: str,
        *,
        prompt: str,
        attempts: list[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.prompt = prompt
        self.attempts = attempts


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
        generation_attempts_path: Path | None = None,
        annotation_store: AnnotationStore | None = None,
        model_factory: Callable[[Path, str], Any] | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint or DEFAULT_CHECKPOINT)
        self.device = device
        self.annotations_path = Path(annotations_path or DEFAULT_FEEDBACK_PATH)
        self.human_train_path = Path(human_train_path or DEFAULT_HUMAN_TRAIN_PATH)
        self.human_pairs_path = Path(human_pairs_path or DEFAULT_HUMAN_PAIRS_PATH)
        self.bad_outputs_path = Path(bad_outputs_path or DEFAULT_BAD_OUTPUTS_PATH)
        self.generation_attempts_path = Path(
            generation_attempts_path or DEFAULT_GENERATION_ATTEMPTS_PATH
        )
        self.annotation_store = annotation_store or FileAnnotationStore(
            feedback_path=self.annotations_path,
            human_train_path=self.human_train_path,
            pairs_path=self.human_pairs_path,
            generation_attempts_path=self.generation_attempts_path,
        )
        self._model_factory = model_factory
        self._model: Any | None = None
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
            "annotation_storage": self.annotation_store.backend,
            "bad_outputs_path": str(self.bad_outputs_path),
            "generation_attempts_path": str(self.generation_attempts_path),
            "meta": meta,
        }

    def _training_model_identity(self) -> dict[str, str]:
        meta = self.info().get("meta") or {}
        model = str(meta.get("kind") or "twotower")
        return {
            "kind": "model",
            "provider": "slm-training",
            "id": f"{model}:{self.checkpoint.name}",
            "model": model,
            "runtime": self.device,
        }

    @staticmethod
    def _browser_model_identity(runtime: str | None = None) -> dict[str, str]:
        return {
            "kind": "model",
            "provider": "browser-built-in-ai",
            "id": "prompt-api-default",
            "model": "prompt-api-default",
            "runtime": runtime or "browser",
        }

    @staticmethod
    def _prompt_bank_identity() -> dict[str, str]:
        return {
            "kind": "system",
            "provider": "slm-training",
            "id": "prompt-bank-composer:v1",
            "model": "prompt-bank-composer",
        }

    @staticmethod
    def _user_identity(session_id: str | None) -> dict[str, str]:
        return {
            "kind": "user",
            "provider": "playground-session",
            "id": (session_id or "anonymous").strip() or "anonymous",
        }

    def load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            if not self.checkpoint.exists():
                raise FileNotFoundError(
                    f"checkpoint not found: {self.checkpoint}. "
                    "Train one with scripts/train_model.py or the playground bootstrap."
                )
            if self._model_factory is not None:
                self._model = self._model_factory(self.checkpoint, self.device)
            else:
                try:
                    from slm_training.models.twotower import TwoTowerModel
                except ModuleNotFoundError as exc:
                    if exc.name != "torch" or self.device != "cpu":
                        raise
                    from slm_training.models.onnx_inference import OnnxTwoTowerModel

                    self._model = OnnxTwoTowerModel.from_checkpoint(
                        self.checkpoint, device=self.device
                    )
                else:
                    self._model = TwoTowerModel.from_checkpoint(
                        self.checkpoint, device=self.device
                    )
            self._model.eval()
            # Enable the deterministic DFA / LTR speculative layer. Validation
            # and fallback policy belong to this service. Q9 decode levers cut
            # repair-path latency without hiding failed model attempts.
            cfg = self._model.config
            cfg.grammar_constrained = True
            cfg.grammar_ltr_primary = True
            cfg.grammar_ltr_repair = True
            # The harness owns the retry/fallback policy. Do not let the model
            # hide a failed decode behind its canned minimal-program fallback.
            cfg.grammar_finalize_validate = False
            cfg.grammar_fastpath = True
            cfg.grammar_incremental_state = True
            cfg.grammar_verify_chosen_only = True
            cfg.grammar_skip_exact_stream_probe = True
            cfg.grammar_copy_probes = True
            cfg.grammar_early_exit_pick = True
            cfg.grammar_multitoken_accept = True
            if int(getattr(cfg, "grammar_multitoken_max", 0) or 0) < 4:
                cfg.grammar_multitoken_max = 8
            if int(getattr(cfg, "grammar_canvas_lookahead", 0) or 0) <= 0:
                cfg.grammar_canvas_lookahead = 32
            # P1 defaults on; P7 attempt budget is honored by generate().
            if not hasattr(cfg, "generate_max_attempts") or not isinstance(
                getattr(cfg, "generate_max_attempts", None), (int, float)
            ):
                cfg.generate_max_attempts = 3
            elif int(cfg.generate_max_attempts or 0) < 1:
                cfg.generate_max_attempts = 3
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

    def _persist_generation_attempt(
        self,
        *,
        prompt: str,
        openui: str,
        source: str,
        attempt: int,
        valid: bool,
        error: str | None,
        prior_failures: list[str],
        design_md: str | None,
        session_id: str | None,
        meta: dict[str, Any] | None = None,
    ) -> tuple[GenerationAttemptRecord, str]:
        record_meta = dict(meta or {})
        identities = {
            str(role): dict(identity)
            for role, identity in dict(record_meta.pop("identities", {}) or {}).items()
        }
        identities.setdefault("request_generator", self._user_identity(session_id))
        if str(record_meta.get("kind") or "") == "browser_judgement":
            identities.setdefault("reviewer", self._browser_model_identity())
        elif source == "browser":
            identities.setdefault("output_generator", self._browser_model_identity())
        else:
            identities.setdefault("output_generator", self._training_model_identity())
        record = GenerationAttemptRecord(
            id=new_generation_attempt_id(source, attempt),
            ts=utc_now_iso(),
            prompt=prompt.strip(),
            openui=openui.strip(),
            source=source,  # type: ignore[arg-type]
            attempt=attempt,
            valid=valid,
            error=(error or "").strip() or None,
            prior_failures=list(prior_failures),
            design_md=(design_md or "").strip() or None,
            checkpoint=str(self.checkpoint),
            session_id=session_id,
            identities=identities,
            meta={
                "usable_for_training": True,
                "label": "valid_openui" if valid else "invalid_openui",
                **record_meta,
            },
        )
        saved = self.annotation_store.persist_generation_attempt(record)
        return record, saved.path

    @staticmethod
    def _attempt_summary(
        record: GenerationAttemptRecord, path: str
    ) -> dict[str, Any]:
        return {
            "id": record.id,
            "source": record.source,
            "attempt": record.attempt,
            "valid": record.valid,
            "error": record.error,
            "openui": record.openui,
            "prior_failures": record.prior_failures,
            "identities": record.identities,
            "training_path": path,
        }

    @staticmethod
    def _validate_candidate(openui: str) -> str:
        """Return canonical, useful OpenUI or raise a training-grade failure."""
        program = validate(openui)
        serialized = (program.serialized or openui).strip()
        compact = "".join(serialized.split())
        if "root=" not in compact:
            raise ParseError("generated OpenUI is missing a root assignment")
        if "Stack([]" in compact or "Card([]" in compact:
            raise ParseError("generated OpenUI contains an empty container")
        return serialized

    def generate(
        self,
        prompt: str,
        *,
        grammar_constrained: bool = True,
        design_md: str | None = None,
        max_attempts: int = 3,
        session_id: str | None = None,
        prior_failures: list[str] | None = None,
        attempt_start: int = 1,
        request_identity: dict[str, Any] | None = None,
    ) -> GenerateResult:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("prompt must be non-empty")
        model = self.load()
        if design_md is None:
            try:
                from slm_training.dsl.design_md import load_default_design_md

                design_md = load_default_design_md()
            except Exception:  # noqa: BLE001
                design_md = None

        last_error: str | None = None
        openui = ""
        serialized: str | None = None
        valid = False
        accumulated_failures = [
            str(reason)[:2000] for reason in (prior_failures or [])
        ][-6:]
        attempt_summaries: list[dict[str, Any]] = []
        identities = {
            "request_generator": dict(
                request_identity or self._user_identity(session_id)
            ),
            "output_generator": self._training_model_identity(),
        }
        with self._lock:
            cfg = model.config

            def _cfg_int(name: str, default: int) -> int:
                raw = getattr(cfg, name, default)
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    return default
                return max(1, int(raw))

            def _cfg_bool(name: str, default: bool = False) -> bool:
                raw = getattr(cfg, name, default)
                if isinstance(raw, bool):
                    return raw
                return default

            cfg_attempts = _cfg_int("generate_max_attempts", 3)
            # Caller max_attempts wins when tighter; config can lower the budget (P7).
            attempts = (
                max(1, min(int(max_attempts), cfg_attempts))
                if grammar_constrained
                else 1
            )
            finalize_last_only = _cfg_bool(
                "grammar_finalize_on_last_attempt_only", False
            )
            saved_finalize = _cfg_bool("grammar_finalize_validate", False)
            for attempt_index in range(attempts):
                attempt_no = max(1, int(attempt_start)) + attempt_index
                # Reset per attempt so a generate_exception never reuses the
                # previous attempt's OpenUI in the bad-output quarantine.
                openui = ""
                serialized = None
                valid = False
                if finalize_last_only:
                    cfg.grammar_finalize_validate = attempt_index == attempts - 1
                if grammar_constrained:
                    # First try is deterministic; retries sample legal tokens
                    # at increasing temperature so three attempts are not the
                    # same failed decode repeated verbatim.
                    model.config.grammar_sample_decode = attempt_no > 1
                    model.config.grammar_sample_temperature = 0.65 + (
                        0.15 * (attempt_no - 1)
                    )
                inference_prompt = prompt
                if accumulated_failures:
                    feedback = "\n".join(
                        f"- {reason}" for reason in accumulated_failures[-6:]
                    )
                    inference_prompt = (
                        f"{prompt}\n\nPrevious generation and review failures to correct:\n"
                        f"{feedback}"
                    )
                try:
                    openui = model.generate(
                        inference_prompt,
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
                            meta={
                                "failure": "generate_exception",
                                "identities": identities,
                            },
                        )
                    record, path = self._persist_generation_attempt(
                        prompt=prompt,
                        openui="",
                        source="server",
                        attempt=attempt_no,
                        valid=False,
                        error=last_error,
                        prior_failures=accumulated_failures,
                        design_md=design_md,
                        session_id=session_id,
                        meta={
                            "failure": "generate_exception",
                            "identities": identities,
                        },
                    )
                    attempt_summaries.append(self._attempt_summary(record, path))
                    accumulated_failures.append(last_error)
                    if not grammar_constrained:
                        raise
                    continue
                try:
                    serialized = self._validate_candidate(openui)
                    valid = True
                    last_error = None
                    record, path = self._persist_generation_attempt(
                        prompt=prompt,
                        openui=serialized,
                        source="server",
                        attempt=attempt_no,
                        valid=True,
                        error=None,
                        prior_failures=accumulated_failures,
                        design_md=design_md,
                        session_id=session_id,
                        meta={"identities": identities},
                    )
                    attempt_summaries.append(self._attempt_summary(record, path))
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
                            meta={
                                "failure": "parse_error",
                                "identities": identities,
                            },
                        )
                    record, path = self._persist_generation_attempt(
                        prompt=prompt,
                        openui=openui,
                        source="server",
                        attempt=attempt_no,
                        valid=False,
                        error=last_error,
                        prior_failures=accumulated_failures,
                        design_md=design_md,
                        session_id=session_id,
                        meta={
                            "failure": "parse_error",
                            "identities": identities,
                        },
                    )
                    attempt_summaries.append(self._attempt_summary(record, path))
                    accumulated_failures.append(last_error)
                    openui = ""
                    if not grammar_constrained:
                        break
            if finalize_last_only and isinstance(
                getattr(cfg, "grammar_finalize_validate", None), bool
            ):
                cfg.grammar_finalize_validate = saved_finalize
            if grammar_constrained and not valid:
                # Absolute contract: never hand the UI an invalid grammar-constrained sample.
                raise GenerationExhausted(
                    last_error
                    or "grammar-constrained generate failed to produce valid OpenUI",
                    prompt=prompt,
                    attempts=attempt_summaries,
                )

        stream = stream_check(openui)
        if isinstance(stream, dict):
            stream_payload = stream
        else:
            stream_payload = {
                "ok": stream.ok,
                "incomplete": stream.incomplete,
                "has_root": stream.has_root,
                "errors": list(stream.error_codes),
                "unresolved": list(stream.unresolved),
            }
        return GenerateResult(
            prompt=prompt,
            openui=openui,
            valid=valid,
            error=last_error,
            stream={
                "ok": stream_payload.get("ok"),
                "incomplete": stream_payload.get("incomplete"),
                "has_root": stream_payload.get("has_root"),
                "errors": stream_payload.get("errors") or [],
                "unresolved": stream_payload.get("unresolved") or [],
            },
            serialized=serialized,
            attempts=attempt_summaries,
        )

    def server_attempt(
        self,
        *,
        prompt: str | None = None,
        session_id: str | None = None,
        grammar_constrained: bool = True,
        design_md: str | None = None,
        auto_prompt: bool = True,
        attempt: int = 1,
        prior_failures: list[str] | None = None,
        request_identity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run and persist exactly one numbered training-model attempt."""
        if attempt < 1 or attempt > 3:
            raise ValueError("server attempt must be between 1 and 3")
        sid = (session_id or "default").strip() or "default"
        current_prompt = (prompt or "").strip()
        auto_selected = not current_prompt
        if auto_selected:
            if not auto_prompt:
                raise ValueError("prompt must be non-empty (or enable auto_prompt)")
            nxt = self.next_prompt(sid)
            current_prompt = nxt["prompt"]
            sid = nxt["session_id"]
        resolved_request_identity = request_identity or (
            self._prompt_bank_identity()
            if auto_selected
            else self._user_identity(sid)
        )
        try:
            result = self.generate(
                current_prompt,
                grammar_constrained=grammar_constrained,
                design_md=design_md,
                max_attempts=1,
                session_id=sid,
                prior_failures=prior_failures,
                attempt_start=attempt,
                request_identity=resolved_request_identity,
            )
        except GenerationExhausted as exc:
            item = exc.attempts[-1]
            return {
                "prompt": current_prompt,
                "openui": item.get("openui") or "",
                "serialized": None,
                "valid": False,
                "error": item.get("error") or str(exc),
                "session_id": sid,
                "source": "server",
                "attempt": item,
                "identities": item.get("identities") or {},
            }
        return {
            "prompt": result.prompt,
            "openui": result.openui,
            "serialized": result.serialized,
            "valid": result.valid,
            "error": result.error,
            "session_id": sid,
            "source": "server",
            "attempt": result.attempts[-1],
            "identities": result.attempts[-1].get("identities") or {},
        }

    def sample(
        self,
        *,
        prompt: str | None = None,
        session_id: str | None = None,
        grammar_constrained: bool = True,
        design_md: str | None = None,
        auto_prompt: bool = True,
    ) -> dict[str, Any]:
        """Try the real model three times, then hand failures to the browser."""
        sid = (session_id or "default").strip() or "default"
        current_prompt = (prompt or "").strip()
        auto_selected = not current_prompt
        if auto_selected:
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
                max_attempts=3,
                session_id=sid,
                request_identity=(
                    self._prompt_bank_identity()
                    if auto_selected
                    else self._user_identity(sid)
                ),
            )
        except GenerationExhausted as exc:
            failures = [
                str(item.get("error") or "unknown server generation failure")
                for item in exc.attempts
            ]
            return {
                "prompt": current_prompt,
                "openui": "",
                "valid": False,
                "error": str(exc),
                "stream": None,
                "serialized": None,
                "session_id": sid,
                "source": None,
                "fallback_required": True,
                "failure_reasons": failures,
                "attempts": exc.attempts,
            }
        return {
            "prompt": result.prompt,
            "openui": result.openui,
            "valid": result.valid,
            "error": result.error,
            "stream": result.stream,
            "serialized": result.serialized,
            "session_id": sid,
            "source": "server",
            "fallback_required": False,
            "failure_reasons": [],
            "attempts": result.attempts,
        }

    def record_browser_attempt(
        self,
        *,
        prompt: str,
        openui: str,
        attempt: int,
        error: str | None = None,
        prior_failures: list[str] | None = None,
        design_md: str | None = None,
        session_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate and durably retain one browser-model attempt."""
        if attempt < 1 or attempt > 3:
            raise ValueError("browser attempt must be between 1 and 3")
        prompt = (prompt or "").strip()
        openui = (openui or "").strip()
        if not prompt:
            raise ValueError("prompt must be non-empty")
        serialized: str | None = None
        validation_error = (error or "").strip() or None
        valid = False
        if openui and validation_error is None:
            try:
                serialized = self._validate_candidate(openui)
                valid = True
            except Exception as exc:  # noqa: BLE001
                validation_error = str(exc)
        if not openui and validation_error is None:
            validation_error = "browser model returned an empty response"
        failures = [str(item)[:2000] for item in (prior_failures or [])][-6:]
        record, path = self._persist_generation_attempt(
            prompt=prompt,
            openui=serialized or openui,
            source="browser",
            attempt=attempt,
            valid=valid,
            error=validation_error,
            prior_failures=failures,
            design_md=design_md,
            session_id=session_id,
            meta=meta,
        )
        return {
            "ok": True,
            "id": record.id,
            "valid": valid,
            "error": validation_error,
            "serialized": serialized,
            "training_path": path,
            "storage": self.annotation_store.backend,
        }

    def record_browser_review(
        self,
        *,
        generation_id: str,
        prompt: str,
        openui: str,
        attempt: int,
        passed: bool,
        score: float,
        reasons: list[str] | None = None,
        prior_failures: list[str] | None = None,
        session_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist the browser baseline's judgement of a server candidate."""
        if attempt < 1 or attempt > 3:
            raise ValueError("reviewed server attempt must be between 1 and 3")
        if score < 0 or score > 1:
            raise ValueError("browser review score must be between 0 and 1")
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("prompt must be non-empty")
        reason_list = [str(reason)[:2000] for reason in (reasons or [])][-8:]
        try:
            serialized = self._validate_candidate(openui)
        except Exception as exc:  # noqa: BLE001
            passed = False
            serialized = (openui or "").strip()
            reason_list.insert(0, f"server lint failed during browser review: {exc}")
        if not reason_list:
            reason_list = [
                "browser baseline approved the candidate"
                if passed
                else "browser baseline rejected the candidate"
            ]
        error = None if passed else "; ".join(reason_list)
        record, path = self._persist_generation_attempt(
            prompt=prompt,
            openui=serialized,
            source="browser",
            attempt=attempt,
            valid=passed,
            error=error,
            prior_failures=[
                str(reason)[:2000] for reason in (prior_failures or [])
            ][-6:],
            design_md=None,
            session_id=session_id,
            meta={
                "kind": "browser_judgement",
                "target_generation_id": generation_id,
                "score": float(score),
                "reasons": reason_list,
                "label": "browser_approved" if passed else "browser_rejected",
                **dict(meta or {}),
            },
        )
        return {
            "ok": True,
            "id": record.id,
            "passed": passed,
            "score": float(score),
            "reasons": reason_list,
            "error": error,
            "training_path": path,
            "storage": self.annotation_store.backend,
        }

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
        generation_id: str | None = None,
        original_openui: str | None = None,
        human_corrected: bool = False,
        identities: dict[str, dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rating_norm = (rating or "").strip().lower()
        if rating_norm not in {"up", "down"}:
            raise ValueError("rating must be 'up' or 'down'")
        prompt = (prompt or "").strip()
        openui = (openui or "").strip()
        if not prompt or not openui:
            raise ValueError("prompt and openui are required")
        original = (original_openui or "").strip() or None
        corrected = bool(original and original != openui)
        if human_corrected and not corrected:
            raise ValueError(
                "human_corrected annotations require a distinct original_openui"
            )
        if corrected:
            openui = self._validate_candidate(openui)
            valid = True
        identity_map = {
            str(role): dict(identity)
            for role, identity in dict(identities or {}).items()
        }
        identity_map.setdefault("annotator", self._user_identity(session_id))
        if corrected:
            identity_map.setdefault("correction_author", identity_map["annotator"])
        identity_map.setdefault("output_generator", self._training_model_identity())
        identity_map.setdefault("request_generator", self._user_identity(session_id))
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
            generation_id=(generation_id or "").strip() or None,
            original_openui=original,
            human_corrected=corrected,
            identities=identity_map,
            meta=dict(meta or {}),
        )
        saved = self.annotation_store.persist(record)
        return {
            "ok": True,
            "id": record.id,
            "path": saved.path,
            "storage": saved.backend,
            "rating": record.rating,
            "openui": record.openui,
            "human_corrected": record.human_corrected,
            "identities": record.identities,
            "human_train_path": saved.human_train_path,
            "preference_pair": saved.preference_pair,
        }

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.annotation_store.recent(limit=limit)]

    def annotation_count(self) -> int:
        return self.annotation_store.count()
