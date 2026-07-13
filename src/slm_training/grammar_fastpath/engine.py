"""OpenUI incremental grammar engine for deterministic fast-path decode."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from lark import Lark, UnexpectedCharacters, UnexpectedToken
from lark.exceptions import UnexpectedEOF

from slm_training.grammar_backends.types import GRAMMARS_DIR

# Terminals that are never force-emitted alone (too broad / ignorable).
_BROAD = frozenset(
    {
        "$END",
        "NAME",
        "COMPONENT",
        "STRING",
        "NUMBER",
        "BOOL",
        "WS_INLINE",
        "COMMENT",
        "_NL",
    }
)

_TERM_TO_TEXT = {
    "EQUAL": "=",
    "LPAR": "(",
    "RPAR": ")",
    "LSQB": "[",
    "RSQB": "]",
    "COMMA": ",",
}


@lru_cache(maxsize=4)
def _load_parser(grammar_path: str) -> Lark:
    text = Path(grammar_path).read_text(encoding="utf-8")
    return Lark(
        text,
        start="start",
        parser="lalr",
        lexer="basic",
        maybe_placeholders=False,
        propagate_positions=False,
    )


class OpenUIIncrementalEngine:
    """
    Incremental OpenUI acceptor via Lark InteractiveParser.

    Deterministic force-emit when ``accepts()`` is a singleton structural
    terminal (e.g. NAME → EQUAL, COMPONENT → LPAR). Hole-admissibility for
    MaskGIT is adapted from Mündler et al. 2025 (arXiv:2508.10111 /
    constrained-diffusion.ai): CFG ∩ completion language non-empty,
    specialized to OpenUI via benign hole substitution + reparse.

    P1: ``set_prefix`` / ``advance`` reuse the InteractiveParser when the new
    prefix extends the previous one, feeding only newly completed lexemes
    instead of re-lexing+re-feeding the entire prefix from scratch.
    """

    def __init__(self, grammar_path: Path | None = None) -> None:
        path = Path(grammar_path) if grammar_path else GRAMMARS_DIR / "openui.lark"
        self.grammar_path = path
        self._parser = _load_parser(str(path.resolve()))
        self._prefix = ""
        self._accepts: frozenset[str] = frozenset()
        self._ip = None
        self._fed_token_count = 0
        self._full_syncs = 0
        self._incremental_advances = 0
        self._sync_ms = 0.0

    def reset(self) -> None:
        self._prefix = ""
        self._ip = self._parser.parse_interactive()
        self._fed_token_count = 0
        try:
            self._accepts = frozenset(str(x) for x in self._ip.accepts())
        except Exception:
            self._accepts = frozenset()

    def _lex_tokens(self, prefix: str) -> list | None:
        try:
            return list(self._parser.lex(prefix))
        except UnexpectedCharacters:
            trimmed = prefix.rstrip()
            cut = len(trimmed)
            while cut > 0 and trimmed[cut - 1] not in " \n\t=,()[]":
                cut -= 1
            try:
                return list(self._parser.lex(trimmed[:cut])) if cut else []
            except Exception:
                return None

    def _refresh_accepts(self) -> None:
        if self._ip is None:
            self._accepts = frozenset()
            return
        try:
            self._accepts = frozenset(str(x) for x in self._ip.accepts())
        except Exception:
            self._accepts = frozenset()

    def _full_sync(self, prefix: str) -> bool:
        """Lex+feed complete tokens from prefix; update accepts. False on hard error."""
        self._full_syncs += 1
        self._prefix = prefix
        self._ip = self._parser.parse_interactive()
        self._fed_token_count = 0
        tokens = self._lex_tokens(prefix)
        if tokens is None:
            self._accepts = frozenset()
            return False
        try:
            for tok in tokens:
                self._ip.feed_token(tok)
                self._fed_token_count += 1
        except UnexpectedToken:
            self._accepts = frozenset()
            return False
        except UnexpectedEOF:
            pass
        self._refresh_accepts()
        return True

    def _incremental_sync(self, prefix: str) -> bool:
        """Extend an existing InteractiveParser when ``prefix`` grows monotonically."""
        assert self._ip is not None
        tokens = self._lex_tokens(prefix)
        if tokens is None:
            return self._full_sync(prefix)
        if len(tokens) < self._fed_token_count:
            # Lexeme boundary shrank (e.g. incomplete → complete merge) — resync.
            return self._full_sync(prefix)
        prev_prefix = self._prefix
        try:
            for tok in tokens[self._fed_token_count :]:
                self._ip.feed_token(tok)
                self._fed_token_count += 1
        except UnexpectedToken:
            # InteractiveParser is poisoned after a rejected feed — full resync
            # back to the last good prefix so shared row state stays usable.
            return self._full_sync(prev_prefix)
        except UnexpectedEOF:
            pass
        self._prefix = prefix
        self._incremental_advances += 1
        self._refresh_accepts()
        return True

    def _sync(self, prefix: str) -> bool:
        t0 = time.perf_counter()
        try:
            if (
                self._ip is not None
                and prefix.startswith(self._prefix)
                and len(prefix) >= len(self._prefix)
            ):
                if prefix == self._prefix:
                    return True
                return self._incremental_sync(prefix)
            return self._full_sync(prefix)
        finally:
            self._sync_ms += (time.perf_counter() - t0) * 1000.0

    def feed_text(self, chunk: str) -> bool:
        if not chunk and not self._prefix:
            self.reset()
            return True
        return self._sync(self._prefix + chunk if chunk else self._prefix)

    def advance(self, chunk: str) -> bool:
        """Append ``chunk`` to the current prefix (P1 incremental path)."""
        if not chunk:
            return True
        return self._sync(self._prefix + chunk)

    def set_prefix(self, prefix: str) -> bool:
        return self._sync(prefix)

    def next_terminals(self) -> frozenset[str]:
        return self._accepts

    def terminals_are_exact(self) -> bool:
        """True when accepts() contains only non-broad structural terminals.

        Used by P2 to skip Node/Lark stream probes when the DFA already pins
        the legal set to exact punctuation / forced lexemes.
        """
        if not self._accepts:
            return False
        meaningful = self._accepts - {"$END", "COMMENT"}
        if not meaningful:
            return False
        return not bool(meaningful & _BROAD)

    def is_deterministic_next(self) -> str | None:
        # Ignore whitespace / end markers only. Broad content terminals
        # (NAME, COMPONENT, STRING, …) must block force-emit even when a
        # structural terminal like LSQB is also accepted (e.g. after `root=`).
        ignorable = frozenset({"$END", "WS_INLINE", "COMMENT", "_NL"})
        meaningful = self._accepts - ignorable
        if len(meaningful) != 1:
            return None
        term = next(iter(meaningful))
        if term in _BROAD:
            return None
        return _TERM_TO_TEXT.get(term)

    def can_complete_with_holes(self, text: str) -> bool:
        probe = text.replace("<mask>", "hole")
        while "hole hole" in probe:
            probe = probe.replace("hole hole", "hole")
        if not probe.endswith("\n"):
            probe = probe + "\n"
        try:
            self._parser.parse(probe)
            return True
        except UnexpectedEOF:
            return True
        except UnexpectedToken:
            open_d = probe.count("(") + probe.count("[")
            close_d = probe.count(")") + probe.count("]")
            return open_d > close_d
        except UnexpectedCharacters:
            return True
        except Exception:
            return False


def engine_for_dsl(dsl: str | None = None) -> OpenUIIncrementalEngine | None:
    key = (dsl or "openui").strip().lower()
    if key in {
        "openui",
        "openui-lark",
        "lark-openui",
        "default",
        "auto",
        "openui-langcore",
    }:
        return OpenUIIncrementalEngine()
    if key == "toy-layout":
        return OpenUIIncrementalEngine(GRAMMARS_DIR / "toy_layout.lark")
    return None
