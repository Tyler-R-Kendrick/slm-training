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
            "parse-error",
        }
        # placeholder_required is a typing hint while a string arg is still open —
        # do not treat it as fatal during incremental constrained decode.
        if not self.incomplete and "placeholder_required" in self.error_codes:
            return True
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
