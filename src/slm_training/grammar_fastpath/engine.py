"""OpenUI incremental grammar engine for deterministic fast-path decode."""

from __future__ import annotations

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
    MaskGIT follows constrained-diffusion.ai: CFG ∩ completion language
    non-empty, specialized to OpenUI via benign hole substitution + reparse.
    """

    def __init__(self, grammar_path: Path | None = None) -> None:
        path = Path(grammar_path) if grammar_path else GRAMMARS_DIR / "openui.lark"
        self.grammar_path = path
        self._parser = _load_parser(str(path.resolve()))
        self._prefix = ""
        self._accepts: frozenset[str] = frozenset()

    def reset(self) -> None:
        self._prefix = ""
        self._accepts = frozenset(self._parser.parse_interactive().accepts())

    def _sync(self, prefix: str) -> bool:
        """Lex+feed complete tokens from prefix; update accepts. False on hard error."""
        self._prefix = prefix
        ip = self._parser.parse_interactive()
        try:
            tokens = list(self._parser.lex(prefix))
        except UnexpectedCharacters:
            trimmed = prefix.rstrip()
            cut = len(trimmed)
            while cut > 0 and trimmed[cut - 1] not in " \n\t=,()[]":
                cut -= 1
            try:
                tokens = list(self._parser.lex(trimmed[:cut])) if cut else []
            except Exception:
                return False
        try:
            for tok in tokens:
                ip.feed_token(tok)
        except UnexpectedToken:
            return False
        except UnexpectedEOF:
            pass
        try:
            self._accepts = frozenset(str(x) for x in ip.accepts())
        except Exception:
            self._accepts = frozenset()
        return True

    def feed_text(self, chunk: str) -> bool:
        if not chunk and not self._prefix:
            self.reset()
            return True
        return self._sync(self._prefix + chunk if chunk else self._prefix)

    def set_prefix(self, prefix: str) -> bool:
        return self._sync(prefix)

    def next_terminals(self) -> frozenset[str]:
        return self._accepts

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
