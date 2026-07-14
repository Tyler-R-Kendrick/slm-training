"""Hybrid OpenUI backend: official lang-core when available, else Lark."""

from __future__ import annotations

from typing import Any

from slm_training.dsl.lang_core import Program
from slm_training.dsl.stream_types import StreamStatus
from slm_training.dsl.grammar.backends.openui_langcore import OpenUILangCoreBackend
from slm_training.dsl.grammar.backends.openui_lark import OpenUILarkBackend
from slm_training.dsl.grammar.backends.types import GrammarInfo


class OpenUIHybridBackend:
    """Prefer official ElementNode AST; fall back to Lark for in-process training."""

    name = "openui"
    dsl_id = "openui"

    def __init__(self) -> None:
        self._langcore = OpenUILangCoreBackend()
        self._lark = OpenUILarkBackend()

    def _active(self) -> OpenUILangCoreBackend | OpenUILarkBackend:
        if self._langcore.available():
            return self._langcore
        return self._lark

    @property
    def info(self) -> GrammarInfo:
        active = self._active()
        return GrammarInfo(
            id="openui",
            kind="hybrid",
            description=(
                f"OpenUI hybrid → {active.info.id} "
                f"({'lang-core' if active is self._langcore else 'lark'})"
            ),
            grammar_path=self._lark.info.grammar_path,
            root_component="root",
        )

    def available(self) -> bool:
        return self._langcore.available() or self._lark.available()

    def is_available(self) -> bool:
        return self.available()

    def parse(self, source: str) -> Program:
        return self._active().parse(source)

    def validate(self, source: str) -> Program:
        return self._active().validate(source)

    def serialize(self, program: Program) -> str:
        return self._active().serialize(program)

    def stream_check(self, source: str) -> StreamStatus:
        return self._active().stream_check(source)

    def structural_tokens(self) -> frozenset[str]:
        return self._active().structural_tokens()

    def component_names(self) -> frozenset[str]:
        return self._active().component_names()

    def content_props(self) -> frozenset[str]:
        return self._active().content_props()

    def library_schema(self) -> dict[str, Any]:
        return self._active().library_schema()

    def generate_system_prompt(self, **options: Any) -> str:
        return self._active().generate_system_prompt(**options)
