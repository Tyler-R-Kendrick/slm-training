"""OpenUI via official @openuidev/lang-core (Node bridge)."""

from __future__ import annotations

from typing import Any

from slm_training.dsl import lang_core
from slm_training.dsl.lang_core import ParseError, Program
from slm_training.dsl.openui_tokens import STRUCTURAL_TOKENS
from slm_training.dsl.placeholders import CONTENT_PROPS
from slm_training.dsl.stream_types import StreamStatus
from slm_training.grammar_backends.types import GrammarInfo


class OpenUILangCoreBackend:
    """Official lexer/parser AST (ElementNode) through the Node bridge."""

    name = "openui-langcore"
    dsl_id = "openui-langcore"

    @property
    def info(self) -> GrammarInfo:
        return GrammarInfo(
            id="openui-langcore",
            kind="lang-core",
            description="Official @openuidev/lang-core ElementNode AST",
            grammar_path=None,
            root_component="root",
        )

    def available(self) -> bool:
        return lang_core.bridge_available()

    def is_available(self) -> bool:
        return self.available()

    def parse(self, source: str) -> Program:
        if not self.available():
            raise ParseError("openui-langcore bridge unavailable")
        program = lang_core.parse(source)
        program.meta = {**program.meta, "backend": self.info.id, "kind": "lang-core"}
        return program

    def validate(self, source: str) -> Program:
        if not self.available():
            raise ParseError("openui-langcore bridge unavailable")
        program = lang_core.validate(source)
        program.meta = {**program.meta, "backend": self.info.id, "kind": "lang-core"}
        return program

    def serialize(self, program: Program) -> str:
        return lang_core.serialize(program)

    def stream_check(self, source: str) -> StreamStatus:
        if not self.available():
            return StreamStatus(
                ok=False,
                incomplete=False,
                has_root=False,
                error_codes=("bridge-unavailable",),
                unresolved=(),
                serialized=None,
            )
        result = lang_core.stream_check(source)
        errors = result.get("errors") or []
        codes = tuple(
            str(e.get("code") or e.get("message") or "error")
            for e in errors
            if isinstance(e, dict)
        )
        return StreamStatus(
            ok=bool(result.get("ok")),
            incomplete=bool(result.get("incomplete")),
            has_root=bool(result.get("has_root")),
            error_codes=codes,
            unresolved=tuple(result.get("unresolved") or []),
            serialized=result.get("serialized"),
        )

    def structural_tokens(self) -> frozenset[str]:
        return STRUCTURAL_TOKENS

    def component_names(self) -> frozenset[str]:
        try:
            schema = self.library_schema()
            props = schema.get("properties") or {}
            return frozenset(str(k) for k in props.keys())
        except Exception:  # noqa: BLE001
            return frozenset(
                t for t in STRUCTURAL_TOKENS if t[:1].isupper() and t.isidentifier()
            )

    def content_props(self) -> frozenset[str]:
        return CONTENT_PROPS

    def library_schema(self) -> dict[str, Any]:
        return lang_core.library_schema()

    def generate_system_prompt(self, **options: Any) -> str:
        return lang_core.generate_system_prompt(**options)
