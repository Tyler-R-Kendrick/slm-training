"""OpenUI incremental grammar engine for deterministic fast-path decode."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from lark import Lark, UnexpectedCharacters, UnexpectedToken
from lark.exceptions import UnexpectedEOF

from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR

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
    "LBRACE": "{",
    "RBRACE": "}",
    "DOT": ".",
    "COLON": ":",
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
        # (type, value) tuples for the tokens already fed into ``_ip``.
        # Used by probe_chunk to detect NAME/COMPONENT gluing that changes
        # an already-fed lexeme's identity without growing the token count.
        self._fed_tokens: list[tuple[str, str]] = []
        self._full_syncs = 0
        self._incremental_advances = 0
        self._copy_probes = 0
        self._copy_probe_fallbacks = 0
        self._sync_ms = 0.0

    def reset(self) -> None:
        self._prefix = ""
        self._ip = self._parser.parse_interactive()
        self._fed_token_count = 0
        self._fed_tokens = []
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

    def _token_keys(self, tokens: list) -> list[tuple[str, str]]:
        return [(str(tok.type), str(tok.value)) for tok in tokens]

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
        self._fed_tokens = []
        tokens = self._lex_tokens(prefix)
        if tokens is None:
            self._accepts = frozenset()
            return False
        try:
            for tok in tokens:
                self._ip.feed_token(tok)
                self._fed_token_count += 1
            self._fed_tokens = self._token_keys(tokens)
        except UnexpectedToken:
            self._accepts = frozenset()
            self._fed_tokens = []
            return False
        except UnexpectedEOF:
            self._fed_tokens = self._token_keys(tokens[: self._fed_token_count])
        self._refresh_accepts()
        return True

    def _incremental_sync(self, prefix: str) -> bool:
        """Extend an existing InteractiveParser when ``prefix`` grows monotonically."""
        assert self._ip is not None
        tokens = self._lex_tokens(prefix)
        if tokens is None:
            return self._full_sync(prefix)
        keys = self._token_keys(tokens)
        # Token count shrank OR an already-fed lexeme changed identity
        # (NAME/COMPONENT gluing: "Te" → "Text") — must resync.
        if len(tokens) < self._fed_token_count or keys[: self._fed_token_count] != self._fed_tokens:
            return self._full_sync(prefix)
        prev_prefix = self._prefix
        try:
            for tok in tokens[self._fed_token_count :]:
                self._ip.feed_token(tok)
                self._fed_token_count += 1
            self._fed_tokens = keys[: self._fed_token_count]
        except UnexpectedToken:
            # InteractiveParser is poisoned after a rejected feed — full resync
            # back to the last good prefix so shared row state stays usable.
            return self._full_sync(prev_prefix)
        except UnexpectedEOF:
            self._fed_tokens = keys[: self._fed_token_count]
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

    def probe_chunk(self, chunk: str) -> bool | None:
        """
        Q1: test whether ``prefix + chunk`` remains a legal prefix without
        mutating this engine.

        Uses ``InteractiveParser.copy()`` + delta-token feed when the already
        fed lexemes are unchanged. Returns ``None`` when the caller should fall
        back to a throwaway full sync (lexeme identity changed / unsynced).
        """
        if self._ip is None:
            return None
        if not chunk:
            return True
        new_prefix = self._prefix + chunk
        t0 = time.perf_counter()
        tokens = self._lex_tokens(new_prefix)
        if tokens is None:
            self._sync_ms += (time.perf_counter() - t0) * 1000.0
            return False
        keys = self._token_keys(tokens)
        if (
            len(tokens) < self._fed_token_count
            or keys[: self._fed_token_count] != self._fed_tokens
        ):
            # NAME/COMPONENT gluing or boundary shrink — caller falls back.
            self._copy_probe_fallbacks += 1
            self._sync_ms += (time.perf_counter() - t0) * 1000.0
            return None
        delta = tokens[self._fed_token_count :]
        if not delta:
            # Whitespace / incomplete interior that adds no complete tokens.
            self._copy_probes += 1
            self._sync_ms += (time.perf_counter() - t0) * 1000.0
            return True
        try:
            snap = self._ip.copy()
        except Exception:
            self._copy_probe_fallbacks += 1
            self._sync_ms += (time.perf_counter() - t0) * 1000.0
            return None
        try:
            for tok in delta:
                snap.feed_token(tok)
        except UnexpectedToken:
            self._copy_probes += 1
            self._sync_ms += (time.perf_counter() - t0) * 1000.0
            return False
        except UnexpectedEOF:
            pass
        self._copy_probes += 1
        self._sync_ms += (time.perf_counter() - t0) * 1000.0
        return True

    def next_terminals(self) -> frozenset[str]:
        return self._accepts

    def minimum_completion_tokens(
        self, prefix: str, *, max_steps: int = 32
    ) -> int | None:
        """Prove a deterministic lower bound, or return unknown on any branch."""
        probe = OpenUIIncrementalEngine(self.grammar_path)
        if not probe.set_prefix(prefix):
            return None
        for steps in range(max(0, max_steps) + 1):
            if "$END" in probe.next_terminals():
                return steps
            term = probe.is_deterministic_next()
            text = _TERM_TO_TEXT.get(term or "")
            if text is None or steps >= max_steps or not probe.advance(text):
                return None
        return None

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
    # F1: a registered DSL pack's incremental_engine slot wins; the alias
    # list below stays as the registry-free fallback.
    try:
        from slm_training.dsl.pack import get_pack

        pack = get_pack(key)
        if pack.incremental_engine is not None:
            return pack.incremental_engine()
    except KeyError:
        pass
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
    # F1 pack seam: any registered backend that carries a Lark grammar path
    # gets a fastpath engine without editing this function.
    try:
        from slm_training.dsl.grammar.backends import get_backend

        info = get_backend(key).info
        if info.grammar_path is not None:
            return OpenUIIncrementalEngine(info.grammar_path)
    except KeyError:
        pass
    return None
