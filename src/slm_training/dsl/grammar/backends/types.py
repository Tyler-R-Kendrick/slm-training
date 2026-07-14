"""Shared types for pluggable grammar backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from slm_training.bridge_utils import repo_root
from slm_training.dsl.lang_core import Program
from slm_training.dsl.stream_types import StreamStatus

REPO_ROOT = repo_root()
GRAMMARS_DIR = REPO_ROOT / "grammars"


@dataclass(frozen=True)
class GrammarInfo:
    id: str
    kind: str  # lang-core | lark | hybrid
    description: str
    grammar_path: Path | None = None
    root_component: str | None = None


@runtime_checkable
class GrammarBackend(Protocol):
    """Contract for parse → AST, stream checks, and train-time priors."""

    @property
    def info(self) -> GrammarInfo: ...

    def available(self) -> bool: ...

    def parse(self, source: str) -> Program: ...

    def validate(self, source: str) -> Program: ...

    def serialize(self, program: Program) -> str: ...

    def stream_check(self, source: str) -> StreamStatus: ...

    def structural_tokens(self) -> frozenset[str]: ...

    def component_names(self) -> frozenset[str]: ...

    def content_props(self) -> frozenset[str]: ...

    def library_schema(self) -> dict[str, Any]: ...

    def generate_system_prompt(self, **options: Any) -> str: ...
