"""Shared stream-check status for grammar backends and constrained decode."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamStatus:
    ok: bool
    incomplete: bool
    has_root: bool
    error_codes: tuple[str, ...]
    unresolved: tuple[str, ...]
    serialized: str | None = None

    @property
    def hard_error(self) -> bool:
        hard = {
            "unknown-component",
            "invalid-type",
            "unexpected-token",
            "placeholder_required",
            "parse-error",
        }
        return any(code in hard for code in self.error_codes)

    @property
    def complete_ok(self) -> bool:
        return (
            self.ok
            and self.has_root
            and not self.incomplete
            and not self.error_codes
            and not self.unresolved
        )
