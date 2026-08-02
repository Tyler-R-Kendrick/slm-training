"""OpenUI incremental grammar engine for deterministic fast-path decode."""

from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

from lark import Lark, Token, UnexpectedCharacters, UnexpectedToken
from lark.exceptions import UnexpectedEOF
from lark.lexer import BasicLexer, LexerThread

from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR
from slm_training.dsl.grammar.fastpath.token_map import (
    _NUMBER_COMPLETE,
    _NUMBER_PREFIX,
    dsl_direct_terminal_map,
    token_surface_piece,
)

_STATIC_ARTIFACT_CACHE: dict[tuple[str, int, int], dict | None] = {}

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

# sha256 of the grammar file, computed once per resolved path — the grammar
# content fingerprint used by parser-state identity keys.
_GRAMMAR_FINGERPRINTS: dict[str, str] = {}


def _grammar_fingerprint(path: Path) -> str:
    key = str(Path(path).resolve())
    fp = _GRAMMAR_FINGERPRINTS.get(key)
    if fp is None:
        fp = hashlib.sha256(Path(key).read_bytes()).hexdigest()
        _GRAMMAR_FINGERPRINTS[key] = fp
    return fp

# Lark's ``InteractiveParser.accepts()`` trial-feeds every terminal choice on
# a fresh parser copy (milliseconds per call).  The result is a pure function
# of the LALR state stack (plus the fixed grammar), and decode revisits the
# same stacks constantly across candidates, so memoize it process-wide.
# This is exact memoization of a deterministic computation — the returned
# terminal set is unchanged, so the legal-action authority is untouched.
_ACCEPTS_MEMO: dict[tuple[str, tuple[int, ...]], frozenset[str]] = {}
_ACCEPTS_MEMO_CAP = 16384


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


@lru_cache(maxsize=4)
def _load_lexer(grammar_path: str) -> BasicLexer:
    """Build (once per grammar path) the ``BasicLexer`` that ``Lark.lex()``
    would otherwise reconstruct from scratch on every call.

    ``Lark.lex(text)`` calls ``self._build_lexer(dont_ignore)`` whenever the
    ``Lark`` object has no cached ``self.lexer`` attribute -- true for any
    grammar built with ``parser="lalr"`` (as ``_load_parser`` does), since
    that path stores the compiled grammar on ``self.parser`` instead. Each
    rebuild reruns ``BasicLexer.__init__`` -> terminal sort/validate and,
    lazily on first ``next_token``, ``_build_scanner`` -> ``_create_unless``,
    which recompiles every terminal-disambiguation regex. The grammar and its
    terminal set never change within (or across) a decode session, so one
    ``BasicLexer`` per grammar path is reused for every ``_lex_tokens`` call
    instead -- same ``LexerConf`` as ``_load_parser``'s cached ``Lark``
    object, so the token stream stays byte-identical; only the *rebuild* is
    removed. See docs/design/decode-compiler-tree-lexer-rebuild-cost-finding.md.
    """
    return _load_parser(grammar_path)._build_lexer()


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
        # Resolved once: ``resolve()`` shows up in decode profiles at ~0.4ms
        # per call, and engines are forked thousands of times per record.
        self._resolved = str(path.resolve())
        self._parser = _load_parser(self._resolved)
        self._lexer = _load_lexer(self._resolved)
        self._prefix = ""
        # ``None`` means dirty: accepts are recomputed lazily on first query.
        # Lark's ``InteractiveParser.accepts()`` costs milliseconds, so the
        # sync paths mark dirty instead of refreshing eagerly — probe/advance
        # chains that never query the terminal set skip the cost entirely.
        self._accepts: frozenset[str] | None = None
        self._ip = None
        self._fed_token_count = 0
        # (type, value) tuples for the tokens already fed into ``_ip``.
        # Used by probe_chunk to detect NAME/COMPONENT gluing that changes
        # an already-fed lexeme's identity without growing the token count.
        self._fed_tokens: list[tuple[str, str]] = []
        # The fed Lark tokens themselves plus a flag that they cover the
        # entire lex of ``_prefix`` (no trim / EOF cut).  Enables the
        # suffix-lex fast path in ``advance_checked``.
        self._fed_lark: list | None = []
        self._fed_buffers_shared = False
        self._tail_clean = True
        # Open framed literal as (kind, body), kind in {"str", "num"}.
        # Explicit state: a pending frame determines all future feed
        # behavior, so it is part of ``parser_state_key``.
        self._frame: tuple[str, str] | None = None
        # Pending unlexed tail (whitespace / unlexable partial source after
        # the last fed lexeme). Explicit state: suffix-lexed tokens carry
        # suffix-relative offsets, so this is maintained, not derived.
        self._tail = ""
        self._fingerprint = _grammar_fingerprint(path)
        # id-keyed, but the tokenizer is stored beside its verified map so
        # (a) a hit re-verifies identity and (b) the strong reference keeps
        # the id from being recycled while the entry lives.
        self._direct_map_cache: dict[int, tuple[object, dict | None]] = {}
        self._direct_map_cache_shared = False
        self._full_syncs = 0
        self._incremental_advances = 0
        self._copy_probes = 0
        self._copy_probe_fallbacks = 0
        self._sync_ms = 0.0
        # Direct-feed instrumentation.
        self.direct_terminal_feeds = 0
        self.full_sync_fallbacks = 0
        self.full_prefix_lex_bytes = 0
        self.parser_forks = 0
        self.static_artifact_hits = 0
        self.static_artifact_fallbacks = 0
        self._control_only = False

    @property
    def stats(self) -> dict[str, int | float]:
        """Decode instrumentation snapshot (counters live per instance)."""
        return {
            "full_syncs": self._full_syncs,
            "incremental_advances": self._incremental_advances,
            "copy_probes": self._copy_probes,
            "copy_probe_fallbacks": self._copy_probe_fallbacks,
            "sync_ms": self._sync_ms,
            "direct_terminal_feeds": self.direct_terminal_feeds,
            "full_sync_fallbacks": self.full_sync_fallbacks,
            "full_prefix_lex_bytes": self.full_prefix_lex_bytes,
            "parser_forks": self.parser_forks,
            "static_artifact_hits": self.static_artifact_hits,
            "static_artifact_fallbacks": self.static_artifact_fallbacks,
        }

    def reset(self) -> None:
        self._prefix = ""
        self._ip = self._parser.parse_interactive()
        if self._control_only:
            self._ip = self._ip_control_copy()
        self._fed_token_count = 0
        self._fed_tokens = []
        self._fed_lark = []
        self._fed_buffers_shared = False
        self._tail_clean = True
        self._frame = None
        self._tail = ""
        self._accepts = None

    def _lex(self, text: str) -> list:
        """Lex ``text`` against the per-grammar cached ``BasicLexer``.

        Equivalent to ``self._parser.lex(text)`` (same ``LexerConf``, so the
        token stream is byte-identical) but reuses the lexer/scanner built
        once by ``_load_lexer`` instead of rebuilding it on every call --
        see docs/design/decode-compiler-tree-lexer-rebuild-cost-finding.md.
        """
        return list(LexerThread.from_text(self._lexer, text).lex(None))

    def _lex_tokens_report(self, prefix: str) -> tuple[list | None, bool]:
        """``_lex_tokens`` plus whether the text had to be trimmed.

        A trim means the returned tokens do NOT cover the full ``prefix``
        text (e.g. an unterminated literal tail was cut), which invalidates
        the ``_tail_clean`` invariant the suffix-lex fast path relies on.
        """
        # Every call lexes the entire prefix — suffix lexes use ``_lex``.
        self.full_prefix_lex_bytes += len(prefix.encode("utf-8"))
        try:
            return self._lex(prefix), False
        except UnexpectedCharacters:
            trimmed = prefix.rstrip()
            cut = len(trimmed)
            while cut > 0 and trimmed[cut - 1] not in " \n\t=,()[]":
                cut -= 1
            try:
                return (self._lex(trimmed[:cut]) if cut else []), True
            except (TimeoutError, KeyboardInterrupt):
                raise
            except Exception:
                return None, True

    def _lex_tokens(self, prefix: str) -> list | None:
        return self._lex_tokens_report(prefix)[0]

    def _token_keys(self, tokens: list) -> list[tuple[str, str]]:
        return [(str(tok.type), str(tok.value)) for tok in tokens]

    def _refresh_accepts(self) -> None:
        if self._ip is None:
            self._accepts = frozenset()
            return
        try:
            stack = tuple(
                int(state)
                for state in getattr(self._ip.parser_state, "state_stack", ())
            )
            key = (self._fingerprint, stack)
            cached = _ACCEPTS_MEMO.get(key)
            if cached is None:
                cached = frozenset(str(x) for x in self._ip.accepts())
                if len(_ACCEPTS_MEMO) >= _ACCEPTS_MEMO_CAP:
                    _ACCEPTS_MEMO.clear()
                _ACCEPTS_MEMO[key] = cached
            self._accepts = cached
        except (TimeoutError, KeyboardInterrupt):
            raise
        except Exception:
            self._accepts = frozenset()

    def _ensure_accepts(self) -> None:
        if self._accepts is None:
            self._refresh_accepts()

    def _full_sync(self, prefix: str) -> bool:
        """Lex+feed complete tokens from prefix; update accepts. False on hard error."""
        self._full_syncs += 1
        self._frame = None  # text route owns lexical state now
        self._prefix = prefix
        self._ip = self._parser.parse_interactive()
        if self._control_only:
            self._ip = self._ip_control_copy()
        self._fed_token_count = 0
        self._fed_tokens = []
        self._fed_buffers_shared = False
        tokens, trimmed = self._lex_tokens_report(prefix)
        if tokens is None:
            self._accepts = frozenset()
            self._fed_lark = None
            self._tail_clean = False
            self._tail = prefix
            return False
        try:
            for tok in tokens:
                self._ip.feed_token(tok)
                self._fed_token_count += 1
            self._fed_tokens = self._token_keys(tokens)
        except UnexpectedToken:
            self._accepts = frozenset()
            self._fed_tokens = []
            self._fed_lark = None
            self._tail_clean = False
            self._tail = prefix
            return False
        except UnexpectedEOF:
            self._fed_tokens = self._token_keys(tokens[: self._fed_token_count])
        self._fed_lark = list(tokens[: self._fed_token_count])
        self._fed_buffers_shared = False
        self._tail_clean = not trimmed and self._fed_token_count == len(tokens)
        self._tail = (
            prefix[tokens[self._fed_token_count - 1].end_pos :]
            if self._fed_token_count
            else prefix
        )
        self._accepts = None  # dirty — computed lazily on first query
        return True

    def _incremental_sync(self, prefix: str) -> bool:
        """Extend an existing InteractiveParser when ``prefix`` grows monotonically."""
        assert self._ip is not None
        self._frame = None  # text route owns lexical state now
        tokens, trimmed = self._lex_tokens_report(prefix)
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
        self._fed_lark = list(tokens[: self._fed_token_count])
        self._fed_buffers_shared = False
        self._tail_clean = not trimmed and self._fed_token_count == len(tokens)
        self._tail = (
            prefix[tokens[self._fed_token_count - 1].end_pos :]
            if self._fed_token_count
            else prefix
        )
        self._accepts = None  # dirty — computed lazily on first query
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

    def advance_checked(self, chunk: str) -> bool:
        """Test-and-commit ``chunk`` in a single lex+feed pass.

        Exactly equivalent to the ``probe_chunk`` + ``advance`` pair a
        speculative draft loop would otherwise run per token:

        - lexeme gluing / boundary change → full sync of the extended text
          (same as ``set_prefix(prefix + chunk)``), returning its result;
        - grammar rejection of the delta → resync to the previous prefix and
          return ``False`` (a ``probe_chunk`` ``False`` leaves the engine
          unmutated; the resync restores the identical observable state);
        - clean extension → feed only the delta lexemes and return ``True``.

        One lex of the extended text and at most one delta feed, instead of
        two of each.
        """
        if not chunk:
            return True
        t0 = time.perf_counter()
        try:
            self._frame = None  # text route owns lexical state now
            new_prefix = self._prefix + chunk
            if self._ip is None:
                return self._full_sync(new_prefix)
            # Suffix-lex fast path: when the fed tokens cover the whole
            # current prefix, only the junction can retokenize (basic lexer =
            # deterministic maximal munch).  Re-lex ``last_token + chunk``;
            # if the last token survives intact, the rest is the delta.
            delta = None
            full_tokens = None
            fed_lark = self._fed_lark
            if self._tail_clean and fed_lark is not None:
                try:
                    if fed_lark:
                        suffix = self._lex(fed_lark[-1].value + chunk)
                        if (
                            suffix
                            and str(suffix[0].type) == str(fed_lark[-1].type)
                            and str(suffix[0].value) == str(fed_lark[-1].value)
                        ):
                            delta = suffix[1:]
                    else:
                        delta = self._lex(chunk)
                except (TimeoutError, KeyboardInterrupt):
                    raise
                except Exception:
                    delta = None
            if delta is None:
                full_tokens, full_trimmed = self._lex_tokens_report(new_prefix)
                if full_tokens is None:
                    return False
                keys = self._token_keys(full_tokens)
                if (
                    len(full_tokens) < self._fed_token_count
                    or keys[: self._fed_token_count] != self._fed_tokens
                ):
                    return self._full_sync(new_prefix)
                delta = full_tokens[self._fed_token_count :]
            prev_prefix = self._prefix
            try:
                for tok in delta:
                    self._ip.feed_token(tok)
                    self._fed_token_count += 1
                tail_clean = True
            except UnexpectedToken:
                # InteractiveParser is poisoned after a rejected feed — full
                # resync back to the last good prefix, then report rejection.
                self._full_sync(prev_prefix)
                return False
            except UnexpectedEOF:
                tail_clean = False
            if full_tokens is not None:
                fed = list(full_tokens[: self._fed_token_count])
                tail_clean = (
                    tail_clean
                    and not full_trimmed
                    and self._fed_token_count == len(full_tokens)
                )
                tail = (
                    new_prefix[fed[-1].end_pos :]
                    if self._fed_token_count
                    else new_prefix
                )
            else:
                fed = (list(fed_lark or []) + list(delta))[
                    : self._fed_token_count
                ]
                # Fast path: the delta covers ``chunk`` by construction, so
                # the tail is clean exactly when no EOF cut the feed short.
                # Delta tokens carry suffix-relative offsets; the tail is the
                # unconsumed remainder of the re-lexed suffix text.
                suffix_text = (fed_lark[-1].value if fed_lark else "") + chunk
                fed_delta = self._fed_token_count - len(fed_lark or [])
                if fed_delta > 0:
                    tail = suffix_text[delta[fed_delta - 1].end_pos :]
                elif fed_lark:
                    tail = suffix_text[len(fed_lark[-1].value) :]
                else:
                    tail = suffix_text
            self._fed_tokens = self._token_keys(fed)
            self._fed_lark = fed
            self._fed_buffers_shared = False
            self._tail_clean = tail_clean
            self._tail = tail
            self._prefix = new_prefix
            self._incremental_advances += 1
            self._accepts = None  # dirty — computed lazily on first query
            return True
        finally:
            self._sync_ms += (time.perf_counter() - t0) * 1000.0

    def set_prefix(self, prefix: str) -> bool:
        return self._sync(prefix)

    def _pending_tail(self) -> str:
        """Source text after the last fed lexeme (whitespace / unlexable)."""
        return self._tail

    def parser_state_key(self) -> tuple:
        """Everything that determines future accepts/feed behavior.

        (grammar content fingerprint, LALR state stack, pending unlexed tail,
        open literal frame). The value stack is excludable: acceptance never
        depends on it. Whitespace-only tails are behaviorally inert and are
        normalized away so states reached via different spacing intern equal.
        """
        stack = (
            tuple(int(s) for s in self._ip.parser_state.state_stack)
            if self._ip is not None
            else ()
        )
        tail = self._pending_tail()
        tail = tail if tail.strip() else ""
        frame = tuple(self._frame) if self._frame is not None else None
        return (self._fingerprint, stack, tail, frame)

    # --- direct DSL-native token-id feeds --------------------------------

    def _direct_map(self, tokenizer) -> dict | None:
        """Verified token→terminal map for ``tokenizer`` (cached, fail closed)."""
        key = id(tokenizer)
        entry = self._direct_map_cache.get(key)
        if entry is None or entry[0] is not tokenizer:
            mapping = None
            if self.grammar_path.name == "openui.lark" and hasattr(
                tokenizer, "token_to_id"
            ):
                artifact_key = (
                    self._fingerprint,
                    int(getattr(tokenizer, "version", 0)),
                    hash(tuple(sorted(tokenizer.token_to_id.items()))),
                )
                if artifact_key not in _STATIC_ARTIFACT_CACHE:
                    try:
                        from slm_training.dsl.grammar.fastpath.completion_artifact import (
                            load_checked_completion_artifact,
                        )

                        checked = load_checked_completion_artifact(
                            tokenizer,
                            self._parser,
                            grammar_path=self.grammar_path,
                        )
                        _STATIC_ARTIFACT_CACHE[artifact_key] = checked.direct_map
                    except (OSError, ValueError, KeyError, json.JSONDecodeError):
                        # Exact reference construction remains the fail-closed
                        # path for absent, stale, or corrupt artifacts.
                        _STATIC_ARTIFACT_CACHE[artifact_key] = None
                mapping = _STATIC_ARTIFACT_CACHE[artifact_key]
                if mapping is None:
                    self.static_artifact_fallbacks += 1
                else:
                    self.static_artifact_hits += 1
            if mapping is None:
                mapping = dsl_direct_terminal_map(tokenizer, self._parser.terminals)
            entry = (tokenizer, mapping)
            if self._direct_map_cache_shared:
                self._direct_map_cache = dict(self._direct_map_cache)
                self._direct_map_cache_shared = False
            self._direct_map_cache[key] = entry
        return entry[1]

    def _snapshot(self) -> tuple:
        return (
            self._prefix,
            self._accepts,
            # Isolate the value stack: star-rule reductions alias child
            # ``children`` lists, so the feed below can mutate trees this
            # snapshot must restore exactly (see ``copy()``).
            (
                self._ip_control_copy()
                if self._control_only
                else self._ip_with_cloned_values()
            )
            if self._ip is not None
            else None,
            self._fed_token_count,
            list(self._fed_tokens),
            list(self._fed_lark) if self._fed_lark is not None else None,
            self._tail_clean,
            self._tail,
            self._frame,
        )

    def _restore(self, snap: tuple) -> None:
        (
            self._prefix,
            self._accepts,
            self._ip,
            self._fed_token_count,
            self._fed_tokens,
            self._fed_lark,
            self._tail_clean,
            self._tail,
            self._frame,
        ) = snap
        self._fed_buffers_shared = False

    def _append_piece(self, piece: str) -> None:
        """Append ``piece`` to ``_prefix``, separating against lexeme gluing.

        Whitespace is ``%ignore``d by the grammar, so an inserted space never
        changes the token stream — it only prevents two adjacent pieces from
        maximal-munching into one lexeme (``root`` + ``b1``, ``=`` + ``=``),
        matching how ``decode_prefix`` spaces the same pieces.
        """
        if not piece:
            return
        if (
            self._prefix
            and not self._prefix.endswith((" ", "\n"))
            and not piece.startswith("\n")
        ):
            self._prefix += " "
        self._prefix += piece

    def _feed_terminal_direct(
        self,
        term: str,
        value: str,
        piece: str,
        *,
        restore_on_reject: bool = True,
    ) -> bool:
        """Feed one verified terminal; optionally restore on rejection.

        Callers normally retain the transactional default.  A caller that has
        just forked this engine and will discard the fork on ``False`` may opt
        out of the redundant second control-state copy.
        """
        assert self._ip is not None
        snap = self._snapshot() if restore_on_reject else None
        token = Token(term, value)
        try:
            self._ip.feed_token(token)
            self._fed_token_count += 1
        except UnexpectedToken:
            # Mirror advance_checked: never leave a poisoned InteractiveParser.
            if snap is not None:
                self._restore(snap)
            return False
        except UnexpectedEOF:
            # Defensive (mirrors _full_sync): keep the fed token committed.
            self._fed_token_count += 1
        self._append_piece(piece)
        self._tail = ""  # a directly fed piece is consumed fully
        if self._fed_buffers_shared:
            self._fed_tokens = list(self._fed_tokens)
            if self._fed_lark is not None:
                self._fed_lark = list(self._fed_lark)
            self._fed_buffers_shared = False
        self._fed_tokens.append((term, value))
        if self._fed_lark is not None:
            self._fed_lark.append(token)
        self._accepts = None  # dirty — computed lazily on first query
        self.direct_terminal_feeds += 1
        return True

    def _text_fallback(self, tokenizer, token_id: int) -> bool:
        """Ambiguous junction: route one token through the canonical text path."""
        self.full_sync_fallbacks += 1
        return self.advance_checked(token_surface_piece(tokenizer, token_id))

    def feed_token_id(
        self,
        tokenizer,
        token_id: int,
        *,
        _macro_seen: frozenset[int] = frozenset(),
        _restore_on_reject: bool = True,
        _token_kind=None,
    ) -> bool | None:
        """Feed one DSL-native token id directly as its Lark terminal.

        Returns ``True`` when committed (terminal fed, frame advanced, or a
        no-op special), ``False`` when the grammar rejected the token and the
        pre-feed state was restored exactly, and ``None`` when the token kind
        is unsupported — the caller must use the canonical text path
        (``advance_checked(token_surface_piece(...))``) instead. Anything that
        cannot be verified against the live grammar fails closed.

        ``_restore_on_reject=False`` is reserved for an already-isolated fork
        that the caller discards when this method returns ``False``.  The
        default remains fully transactional, and macro expansion always keeps
        its own all-or-nothing snapshot.
        """
        mapping = self._direct_map(tokenizer)
        if mapping is None:
            return None
        tid = int(token_id)
        if tid in mapping["skip_ids"]:
            # BOS/EOS/PAD/MASK are never fed as source terminals (decode
            # drops them; EOS acceptability is queried via next_terminals).
            return True
        kind = _token_kind
        if kind is None:
            try:
                kind = tokenizer.kind_of(tid)
            except (TimeoutError, KeyboardInterrupt):
                raise
            except Exception:
                return None
        from slm_training.models.dsl_tokenizer import TokenKind

        if kind is TokenKind.SPECIAL:  # <unk> and friends: never fed
            return None
        if self._ip is None:
            self.reset()

        # Pending unlexable tail (e.g. an unterminated string left by the
        # text path) makes any junction ambiguous — canonical text route.
        if self._frame is None and self._pending_tail().strip():
            return self._text_fallback(tokenizer, tid)

        # --- inside an open framed literal ---
        if self._frame is not None:
            frame_kind, body = self._frame
            if tid == mapping["lit_end"]:
                if frame_kind == "num":
                    if not body or not _NUMBER_COMPLETE.fullmatch(body):
                        return False  # malformed frame: fail closed
                    self._frame = None
                    return self._feed_terminal_direct(
                        "NUMBER",
                        body,
                        body,
                        restore_on_reject=_restore_on_reject,
                    )
                self._frame = None
                quoted = '"' + body + '"'
                return self._feed_terminal_direct(
                    "STRING",
                    quoted,
                    quoted,
                    restore_on_reject=_restore_on_reject,
                )
            if kind is TokenKind.BYTE:
                raw = str(tokenizer.id_to_token.get(tid, ""))
                ch = chr(int(raw[2:], 16)) if raw.startswith("B:") else None
                if ch is None:
                    return None
                if frame_kind == "num":
                    if not _NUMBER_PREFIX.fullmatch(body + ch):
                        return False  # malformed frame: fail closed
                    self._frame = (frame_kind, body + ch)
                    return True
                if ch in {'"', "'", "\\", "\n", "\r"}:
                    # Would retokenize the frame text — flush through the
                    # canonical text route instead of guessing a lexeme.
                    self._frame = None
                    self.full_sync_fallbacks += 1
                    return self.advance_checked('"' + body + ch)
                self._frame = (frame_kind, body + ch)
                return True
            return False  # frame interrupted by a non-byte: fail closed

        # --- framed-literal openers / stray closer ---
        if tid == mapping["lit_num"]:
            self._frame = ("num", "")
            return True
        if mapping["lit_str"] is not None and tid == mapping["lit_str"]:
            self._frame = ("str", "")
            return True
        if tid == mapping["lit_end"]:
            return False  # closer without an opener: fail closed

        # --- punctuation / NL ---
        term = mapping["punct"].get(tid)
        if term is not None:
            text = str(tokenizer.id_to_token.get(tid, ""))
            piece = "\n" if text == "NL" else text
            if (
                term == "_NL"
                and self._fed_tokens
                and self._fed_tokens[-1][0] == "_NL"
            ):
                # The canonical lexer maximal-munches consecutive newlines
                # into one _NL lexeme (and decode_prefix collapses them), so
                # a second adjacent NL token is a no-op — never a second
                # _NL terminal, which the grammar never sees.
                return True
            return self._feed_terminal_direct(
                term,
                piece,
                piece,
                restore_on_reject=_restore_on_reject,
            )

        # --- bools / null ---
        term = mapping["bool"].get(tid) or mapping["null"].get(tid)
        if term is not None:
            text = str(tokenizer.id_to_token.get(tid, ""))
            return self._feed_terminal_direct(
                term,
                text,
                text,
                restore_on_reject=_restore_on_reject,
            )

        # --- fixed string rows ---
        if tid in mapping["str_lit_ids"]:
            body = str(tokenizer.id_to_token.get(tid, ""))[4:]
            quoted = json.dumps(body, ensure_ascii=False)
            return self._feed_terminal_direct(
                "STRING",
                quoted,
                quoted,
                restore_on_reject=_restore_on_reject,
            )

        # --- broad content kinds ---
        term = mapping["kind_terminals"].get(kind)
        if term is not None:
            piece = tokenizer.decode([tid]) or str(tokenizer.id_to_token.get(tid, ""))
            return self._feed_terminal_direct(
                term,
                piece,
                piece,
                restore_on_reject=_restore_on_reject,
            )

        # --- macros: feed the verified fixed expansion iteratively ---
        if kind is TokenKind.MACRO:
            if tid in _macro_seen:
                return None
            slot = tokenizer.macro_slot_of(tid)
            expansions = getattr(tokenizer, "macro_expansions", ())
            if slot is None or slot >= len(expansions):
                # No verified expansion — the caller uses the text path.
                return None
            snap = self._snapshot()
            for name in expansions[slot]:
                sub_id = tokenizer.token_to_id.get(name)
                if sub_id is None:
                    self._restore(snap)
                    return None
                result = self.feed_token_id(
                    tokenizer,
                    sub_id,
                    _macro_seen=_macro_seen | {tid},
                )
                if result is not True:
                    self._restore(snap)
                    return result
            return True

        # Bytes outside a frame (identifiers/keys/members) glue with their
        # neighbors — canonical text route. ABSTRACT and anything else
        # unverifiable stays on the text path too.
        if kind is TokenKind.BYTE:
            return self._text_fallback(tokenizer, tid)
        return None

    def _ip_with_cloned_values(self):
        """Copy ``_ip`` with fork-local parse trees.

        Lark's star-rule callback (``ChildFilterLALR_NoPlaceholders``,
        "Optimize for left-recursion") aliases a child tree's ``children``
        list into its parent reduction, so a real ``feed_token`` on a
        shallow-copied parser can append to a tree the source engine still
        holds — an in-place mutation that corrupts the source's value stack
        (and every forest built from it).  Lark's own ``accepts()`` trial
        feeds get away with ``deepcopy_values=False`` only because they run
        with empty callbacks (no tree building at all), and a full
        ``deepcopy_values=True`` pays for deep-copying immutable Tokens.
        Cloning only the Trees (fresh ``children`` lists, shared Tokens)
        breaks the aliasing at a fraction of the cost.
        """
        from copy import copy as _shallow_copy

        from lark import Tree

        def _clone(value):
            if isinstance(value, Tree):
                return Tree(value.data, [_clone(child) for child in value.children])
            return value  # Tokens/terminals are immutable

        interactive = self._ip
        parser_state = interactive.parser_state
        cloned_state = type(parser_state)(
            parser_state.parse_conf,
            parser_state.lexer,
            _shallow_copy(parser_state.state_stack),
            [_clone(value) for value in parser_state.value_stack],
        )
        return type(interactive)(
            interactive.parser,
            cloned_state,
            _shallow_copy(interactive.lexer_thread),
        )

    def _ip_control_copy(self):
        """Copy parser control state without syntax trees or callbacks."""
        from copy import copy as _shallow_copy

        interactive = self._ip
        parser_state = interactive.parser_state
        parse_conf = parser_state.parse_conf
        share_lexer_thread = not bool(parse_conf.callbacks)
        if parse_conf.callbacks:
            # The first semantic -> control fork must detach and suppress tree
            # callbacks. Descendant control-only forks can share that immutable
            # callback-free parser configuration; only their mutable stacks and
            # lexer threads need copying. Exact traversal and reductions are
            # unchanged, while the completion forest avoids one allocation per
            # parser fork.
            parse_conf = _shallow_copy(parse_conf)
            parse_conf.callbacks = {}
        # Reductions require one stack value per shifted state, but callbacks
        # are disabled, so immutable ``None`` placeholders are sufficient.
        cloned_state = type(parser_state)(
            parse_conf,
            parser_state.lexer,
            _shallow_copy(parser_state.state_stack),
            [None] * len(parser_state.value_stack),
        )
        return type(interactive)(
            interactive.parser,
            cloned_state,
            (
                interactive.lexer_thread
                if share_lexer_thread
                else _shallow_copy(interactive.lexer_thread)
            ),
        )

    def _copy(self, *, control_only: bool) -> "OpenUIIncrementalEngine":
        """Cheap structural fork: share parser/lexer, copy parser state.

        The returned engine is synced to the same prefix as this one, so a
        descendant can ``set_prefix`` an extension of this prefix and take the
        incremental path (feed only the delta lexemes) instead of a full
        re-lex+re-feed.  ``InteractiveParser.copy()`` semantics are the same
        ones ``probe_chunk`` already relies on.
        """
        self.parser_forks += 1
        fork = self.__class__.__new__(self.__class__)
        fork.grammar_path = self.grammar_path
        fork._resolved = self._resolved
        fork._parser = self._parser
        fork._lexer = self._lexer
        fork._prefix = self._prefix
        fork._accepts = self._accepts
        fork._ip = (
            self._ip_control_copy()
            if control_only and self._ip is not None
            else self._ip_with_cloned_values()
            if self._ip is not None
            else None
        )
        fork._fed_token_count = self._fed_token_count
        # Forks are overwhelmingly read-only probes. Share the append-only fed
        # history and detach it only when either engine commits a terminal;
        # parser stacks and lexer threads remain independently copied above.
        fork._fed_tokens = self._fed_tokens
        fork._fed_lark = self._fed_lark
        self._fed_buffers_shared = True
        fork._fed_buffers_shared = True
        fork._tail_clean = self._tail_clean
        fork._tail = self._tail
        fork._frame = self._frame
        fork._fingerprint = self._fingerprint
        # The verified tokenizer map is read-only on the hot fork path. Share
        # it until a fork sees a different tokenizer, then detach before the
        # cache insert. This avoids one dict allocation per parser fork.
        fork._direct_map_cache = self._direct_map_cache
        self._direct_map_cache_shared = True
        fork._direct_map_cache_shared = True
        fork._full_syncs = 0
        fork._incremental_advances = 0
        fork._copy_probes = 0
        fork._copy_probe_fallbacks = 0
        fork._sync_ms = 0.0
        fork.direct_terminal_feeds = 0
        fork.full_sync_fallbacks = 0
        fork.full_prefix_lex_bytes = 0
        fork.parser_forks = 0
        fork.static_artifact_hits = 0
        fork.static_artifact_fallbacks = 0
        fork._control_only = bool(control_only)
        return fork

    def copy(self) -> "OpenUIIncrementalEngine":
        """Fork parser state, preserving its semantic/control-only mode."""
        return self._copy(control_only=self._control_only)

    def copy_control(self) -> "OpenUIIncrementalEngine":
        """Fork only LALR control state; semantic authority lives elsewhere."""
        return self._copy(control_only=True)

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
        except (TimeoutError, KeyboardInterrupt):
            raise
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
        self._ensure_accepts()
        return self._accepts or frozenset()

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
            self._ensure_accepts()
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
        self._ensure_accepts()
        ignorable = frozenset({"$END", "WS_INLINE", "COMMENT", "_NL"})
        meaningful = (self._accepts or frozenset()) - ignorable
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
        except (TimeoutError, KeyboardInterrupt):
            raise
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
