"""Grammar-constrained decode helpers via pluggable GrammarBackend.

Default backend is OpenUI hybrid (official lang-core when available, else Lark).
Stream checks and structural priors come from the active DSL backend so other
grammars can drive the same MaskGIT / LTR decode path.

P1: ``GrammarDecodeState`` reuses one DFA engine + decoded prefix text across
token steps so we do not re-lex / re-decode the whole prefix each call.

P2: ``verify_chosen_only`` probes the model argmax first and only expands the
legal candidate set on rejection; exact (non-broad) DFA terminal sets skip
stream probes entirely.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Iterable

from slm_training.dsl.openui_tokens import STRUCTURAL_TOKENS
from slm_training.dsl.stream_types import StreamStatus
from slm_training.models.tokenizer import OpenUITokenizer

_STREAM_CACHE: dict[str, StreamStatus] = {}
_STREAM_CACHE_MAX = 2048
_STRUCT_ID_CACHE: dict[int, tuple[int, set[int]]] = {}
_BIAS_CACHE: dict[tuple[int, int, float, str, str], object] = {}
_ACTIVE_DSL: str | None = None


def require_constrained_generation(
    requested: bool | None, *, configured: bool = True
) -> None:
    """Reject generation settings that could permit invalid grammar."""
    if requested is False or (requested is None and not configured):
        raise ValueError(
            "grammar_constrained=False is unsafe for OpenUI generation; "
            "use constrained generation and shadow logits for diagnostics"
        )


def _token_surface_piece(tokenizer: OpenUITokenizer, token_id: int) -> str:
    """Decode one token into the partial source consumed by the grammar."""
    from slm_training.dsl.grammar.fastpath.token_map import token_surface_piece

    return token_surface_piece(tokenizer, token_id)


def _decode_prefix_text(
    tokenizer: OpenUITokenizer, prefix_ids: list[int]
) -> str:
    from slm_training.dsl.grammar.fastpath.token_map import decode_prefix

    return decode_prefix(tokenizer, prefix_ids)


def set_active_dsl(dsl: str | None) -> None:
    """Select grammar backend id used by stream_check / structural priors."""
    global _ACTIVE_DSL, _STREAM_CACHE, _STRUCT_ID_CACHE, _BIAS_CACHE
    _ACTIVE_DSL = dsl
    _STREAM_CACHE.clear()
    _STRUCT_ID_CACHE.clear()
    _BIAS_CACHE.clear()
    if dsl:
        from slm_training.dsl.grammar.backends import set_default_backend

        set_default_backend(dsl)


def active_dsl() -> str:
    return _ACTIVE_DSL or os.getenv("SLM_GRAMMAR_DSL") or "openui"


def _backend():
    from slm_training.dsl.grammar.backends import get_backend

    return get_backend(active_dsl())


def stream_check(source: str) -> StreamStatus:
    """Run the active DSL backend's streaming / incremental check."""
    from slm_training.models.decode_stats import get_active_stats

    key = hashlib.sha256(f"{active_dsl()}|{source}".encode("utf-8")).hexdigest()
    hit = _STREAM_CACHE.get(key)
    if hit is not None:
        return hit

    stats = get_active_stats()
    t0 = time.perf_counter()
    status = _backend().stream_check(source)
    if stats is not None:
        stats.stream_check_ms += (time.perf_counter() - t0) * 1000.0
        stats.probes_count += 1
    if len(_STREAM_CACHE) >= _STREAM_CACHE_MAX:
        _STREAM_CACHE.pop(next(iter(_STREAM_CACHE)))
    _STREAM_CACHE[key] = status
    return status


def structural_tokens() -> frozenset[str]:
    try:
        return _backend().structural_tokens()
    except Exception:  # noqa: BLE001
        return STRUCTURAL_TOKENS


def structural_token_ids(tokenizer: OpenUITokenizer) -> set[int]:
    # Key on id() for speed, but validate the cached vocab size: id() is reused
    # after a tokenizer is GC'd, so a stale entry from a larger-vocab tokenizer
    # must not be returned for a smaller one (its ids would index past the
    # logits width downstream and raise IndexError).
    vocab_len = len(tokenizer.token_to_id)
    cache_key = id(tokenizer.token_to_id)
    cached = _STRUCT_ID_CACHE.get(cache_key)
    if cached is not None and cached[0] == vocab_len and len(cached[1]) > 0:
        return cached[1]

    ids: set[int] = set()
    try:
        from slm_training.models.dsl_tokenizer import TokenKind, is_dsl_native_tokenizer

        if is_dsl_native_tokenizer(tokenizer):
            ids |= tokenizer.kind_ids(TokenKind.STRUCT)
            ids |= tokenizer.kind_ids(TokenKind.COMPONENT)
            ids |= tokenizer.kind_ids(TokenKind.BUILTIN)
            ids |= tokenizer.kind_ids(TokenKind.SYM)
            ids |= tokenizer.kind_ids(TokenKind.LIT)
            ids.update(
                {
                    tokenizer.bos_id,
                    tokenizer.eos_id,
                    tokenizer.mask_id,
                }
            )
            _STRUCT_ID_CACHE[cache_key] = (vocab_len, ids)
            return ids
    except Exception:  # noqa: BLE001
        pass

    for tok in structural_tokens():
        if tok in tokenizer.token_to_id:
            ids.add(tokenizer.token_to_id[tok])
    for tok, tid in tokenizer.token_to_id.items():
        if tok.startswith(":") or (tok.startswith('"') and tok.endswith('"')):
            ids.add(tid)
        if tok[:1].isupper() and tok.isidentifier():
            ids.add(tid)
    ids.update(
        {
            tokenizer.bos_id,
            tokenizer.eos_id,
            tokenizer.mask_id,
        }
    )
    _STRUCT_ID_CACHE[cache_key] = (vocab_len, ids)
    return ids


def apply_structural_bias(
    logits,  # torch.Tensor [B, T, V]
    tokenizer: OpenUITokenizer,
    *,
    bias: float = 1.5,
):
    """Boost known structural tokens (returns new tensor)."""
    import torch

    allowed = structural_token_ids(tokenizer)
    vocab = logits.size(-1)
    # Defensive: never bias an index past the logits width, even if a stale
    # structural-id set slips through (belt-and-suspenders with the vocab-size
    # validation in structural_token_ids).
    allowed = {i for i in allowed if 0 <= i < vocab}
    if not allowed:
        return logits
    cache_key = (
        id(tokenizer.token_to_id),
        vocab,
        float(bias),
        str(logits.device),
        str(logits.dtype),
    )
    boost = _BIAS_CACHE.get(cache_key)
    if boost is None or getattr(boost, "numel", lambda: 0)() != logits.size(-1):
        boost = torch.zeros(logits.size(-1), device=logits.device, dtype=logits.dtype)
        idx = torch.tensor(sorted(allowed), device=logits.device, dtype=torch.long)
        boost.index_fill_(0, idx, bias)
        _BIAS_CACHE[cache_key] = boost
    return logits + boost.view(1, 1, -1)  # type: ignore[union-attr]


class CompletionDomainCache(dict[tuple[object, ...], object]):
    """A batch-shareable immutable hard-domain cache."""

    def __init__(self, *, shared: bool = False) -> None:
        super().__init__()
        self.shared = bool(shared)

    def get(self, key: tuple[object, ...], default: object = None) -> object:
        if self.shared:
            from slm_training.models.decode_stats import get_active_stats

            stats = get_active_stats()
            if stats is not None:
                field = (
                    "completion_shared_domain_hits"
                    if key in self
                    else "completion_shared_domain_misses"
                )
                setattr(stats, field, int(getattr(stats, field)) + 1)
        return super().get(key, default)


@dataclass
class CompletionBatchCache:
    """Batch-lifetime sharing for identical packed hard states."""

    domains: dict[
        tuple[tuple[object, ...], tuple[int, ...], int],
        object,
    ] = field(default_factory=dict)


@dataclass
class GrammarDecodeState:
    """Per-row persistent grammar state for LTR decode (P1)."""

    engine: object | None = None
    prefix_ids: list[int] = field(default_factory=list)
    prefix_text: str = ""
    verify_chosen_only: bool = False
    skip_exact_stream_probe: bool = True
    # Q1: use InteractiveParser.copy() probes when possible.
    use_copy_probes: bool = True
    # Q2: early-exit descending-logit candidate scoring.
    early_exit_pick: bool = True
    # Per-position admit memo (token_id -> bool); cleared on advance/sync.
    admit_memo: dict[int, bool] = field(default_factory=dict)
    # Cached whitespace-admit result for the current position (Q2).
    whitespace_ok: bool | None = None
    # The caller sets this from its actual canvas horizon before each strict
    # choice.  It is part of the completion-domain cache key.
    remaining_tokens: int | None = None
    completion_domain_cache: dict[tuple[object, ...], object] = field(
        default_factory=CompletionDomainCache
    )
    # Packed completion ownership is row-local. State ids and the prefix index
    # are immutable request-local handles, so rollback restores one handle
    # instead of replaying the whole prefix through the parser.
    completion_session: object | None = None
    completion_state_id: int | None = None
    completion_authority_key: tuple[object, ...] | None = None
    completion_prefix_states: dict[tuple[int, ...], int] = field(default_factory=dict)
    completion_stats_snapshot: dict[str, int] = field(default_factory=dict)
    completion_batch_cache: CompletionBatchCache | None = None
    # P3 direct-feed sync marker: the number of leading ``prefix_ids`` the
    # engine committed via ``feed_token_id`` (DSL-native direct terminal
    # feeds).  ``None`` means the engine is not known to be id-synced (fresh,
    # text-route sync, or a rejected feed).  Direct feeds keep the engine's
    # internal ``_prefix`` text grammatically equivalent but not byte-identical
    # to ``prefix_text`` (whitespace is ``%ignore``d, framed literals commit at
    # close), so sync checks must consult this marker instead of comparing
    # text alone.  Cleared whenever ``sync_ids`` replaces (not extends) the
    # prefix, so a stale marker can never vouch for different ids.
    engine_ids_len: int | None = None

    def clear_position_memo(self) -> None:
        self.admit_memo.clear()
        self.whitespace_ok = None

    def _collect_completion_stats(self) -> None:
        if self.completion_session is None:
            return
        from slm_training.models.decode_stats import collect_completion_session_delta

        self.completion_stats_snapshot = collect_completion_session_delta(
            self.completion_session,
            self.completion_stats_snapshot,
        )

    def bind_completion_session(
        self,
        session: object,
        authority_key: tuple[object, ...],
        state_id: int,
        prefix_ids: tuple[int, ...] | list[int],
    ) -> None:
        """Bind a request-local packed session at the current prefix."""
        if session is not self.completion_session:
            self._collect_completion_stats()
            self.completion_stats_snapshot = {}
            self.completion_prefix_states = {}
        self.completion_session = session
        self.completion_authority_key = authority_key
        self.completion_state_id = int(state_id)
        self.completion_prefix_states[tuple(int(token_id) for token_id in prefix_ids)] = (
            int(state_id)
        )
        self._collect_completion_stats()

    def _restore_completion_prefix(self, prefix_ids: list[int]) -> None:
        session = self.completion_session
        if session is None:
            return
        state_id = self.completion_prefix_states.get(tuple(prefix_ids))
        if state_id is None:
            self.completion_state_id = None
            return
        restore = getattr(session, "restore", None)
        self.completion_state_id = (
            int(restore(state_id)) if callable(restore) else int(state_id)
        )

    def _advance_completion_token(
        self, token_id: int, *, require_advertised: bool = True
    ) -> None:
        session = self.completion_session
        state_id = self.completion_state_id
        if session is None or state_id is None:
            return
        child_id = session.advance(int(state_id), int(token_id))
        if child_id is None:
            if not require_advertised:
                self.completion_state_id = None
                self._collect_completion_stats()
                return
            raise RuntimeError(
                "packed completion session rejected an advertised decode token "
                f"{int(token_id)} at prefix {tuple(self.prefix_ids)}"
            )
        self.completion_state_id = int(child_id)
        self.completion_prefix_states[tuple(self.prefix_ids)] = int(child_id)
        self._collect_completion_stats()

    def completion_forced_closure(
        self, room: int
    ) -> tuple[tuple[int, ...], int, str] | None:
        """Return the session's exact forced run at the current row state."""
        if self.completion_session is None or self.completion_state_id is None:
            return None
        result = self.completion_session.forced_closure(
            int(self.completion_state_id), int(room)
        )
        self._collect_completion_stats()
        return result

    def engine_in_sync(self, prefix_ids: list[int], prefix_text: str) -> bool:
        """True when ``engine`` already represents exactly ``prefix_ids``.

        Either the engine's text matches byte-for-byte (the legacy text route)
        or the direct-feed marker vouches for this exact prefix length — the
        marker is only ever set to ``len(prefix_ids)`` after feeding those very
        ids and is cleared on any non-append ``sync_ids`` replacement, so
        equal length implies equal ids.  Never widens legality: a desynced
        engine simply takes the canonical ``set_prefix`` full sync as before.
        """
        eng = self.engine
        if eng is None or getattr(eng, "_ip", None) is None:
            return False
        if not getattr(eng, "_synced_ok", True):
            # A rejected sync leaves ``_prefix`` set for diagnostics; it must
            # never vouch for an in-sync engine (false-admit hazard).
            return False
        if getattr(eng, "_prefix", None) == prefix_text:
            return True
        return self.engine_ids_len is not None and self.engine_ids_len == len(
            prefix_ids
        )

    def sync_ids(self, tokenizer: OpenUITokenizer, prefix_ids: list[int]) -> str:
        """Update prefix_ids/text incrementally; return current prefix text."""
        from slm_training.models.decode_stats import get_active_stats

        stats = get_active_stats()
        from slm_training.dsl.grammar.fastpath.token_map import decode_prefix

        if prefix_ids == self.prefix_ids:
            return self.prefix_text
        t0 = time.perf_counter()
        old_prefix = list(self.prefix_ids)
        if (
            len(prefix_ids) >= len(self.prefix_ids)
            and prefix_ids[: len(self.prefix_ids)] == self.prefix_ids
        ):
            # Append-only growth — decode only the new suffix tokens.
            extra = prefix_ids[len(self.prefix_ids) :]
            if extra:
                chunk = decode_prefix(tokenizer, extra)
                self.prefix_text = self.prefix_text + chunk
            self.prefix_ids = list(prefix_ids)
            if extra and self.completion_session is not None:
                self.prefix_ids = old_prefix
                for token_id in extra:
                    self.prefix_ids.append(int(token_id))
                    if int(token_id) != int(tokenizer.eos_id):
                        self._advance_completion_token(
                            int(token_id), require_advertised=False
                        )
                self.prefix_ids = list(prefix_ids)
        else:
            self.prefix_ids = list(prefix_ids)
            self.prefix_text = decode_prefix(tokenizer, prefix_ids) if prefix_ids else ""
            # The engine (if any) no longer represents these ids.
            self.engine_ids_len = None
            self._restore_completion_prefix(self.prefix_ids)
        self.clear_position_memo()
        if stats is not None:
            stats.detok_ms += (time.perf_counter() - t0) * 1000.0
        return self.prefix_text

    def advance_token(
        self,
        tokenizer: OpenUITokenizer,
        token_id: int,
        *,
        require_completion_advertised: bool = True,
    ) -> str:
        """Append one emitted token to the cached prefix.

        DSL-native tokenizers advance the DFA engine by direct terminal feed
        (``feed_token_id``): zero full-prefix lexing after the initial sync.
        Anything the direct feed cannot verify (``None`` — compositional
        tokenizers, unsupported kinds, adapter mismatch) takes the canonical
        text route exactly as before, counted as a fallback on the engine.
        A grammar rejection (``False``) restores the pre-feed engine state
        exactly — the same observable state the legacy ``advance`` rejection
        left behind (engine resynced to the pre-append prefix).
        """
        self.prefix_ids.append(int(token_id))
        from slm_training.models.decode_stats import get_active_stats

        stats = get_active_stats()
        t0 = time.perf_counter()
        # Native lexer ids (e.g. <BIND_0>) are not surface text. Decode the
        # emitted id before advancing the DFA or its prefix diverges from the
        # text that will be parsed.
        chunk = _token_surface_piece(tokenizer, int(token_id))
        self.prefix_text = self.prefix_text + chunk
        if stats is not None:
            stats.detok_ms += (time.perf_counter() - t0) * 1000.0
        if self.engine is not None:
            fed: bool | None = None
            feed = getattr(self.engine, "feed_token_id", None)
            if callable(feed):
                try:
                    fed = feed(tokenizer, int(token_id))
                except (TimeoutError, KeyboardInterrupt):
                    raise
                except Exception:  # noqa: BLE001 - adapter mismatch: text fallback
                    fed = None
            if fed is not None:
                # Committed (or exactly restored) by the direct feed.
                self.engine_ids_len = (
                    len(self.prefix_ids) if fed else None
                )
            else:
                if callable(feed):
                    # Unsupported id/kind or adapter failure on a tokenizer the
                    # engine could otherwise direct-feed: counted fallback.
                    try:
                        from slm_training.models.dsl_tokenizer import (
                            is_dsl_native_tokenizer,
                        )

                        if is_dsl_native_tokenizer(tokenizer):
                            self.engine.full_sync_fallbacks += 1  # type: ignore[union-attr]
                    except (TimeoutError, KeyboardInterrupt):
                        raise
                    except Exception:  # noqa: BLE001
                        pass
                if chunk:
                    resynced = True
                    try:
                        self.engine.advance(chunk)  # type: ignore[union-attr]
                    except (TimeoutError, KeyboardInterrupt):
                        raise
                    except Exception:  # noqa: BLE001
                        try:
                            resynced = bool(
                                self.engine.set_prefix(self.prefix_text)  # type: ignore[union-attr]
                            )
                        except (TimeoutError, KeyboardInterrupt):
                            raise
                        except Exception:  # noqa: BLE001
                            resynced = False
                else:
                    resynced = True
                # The text route leaves the engine grammar-synced with the
                # same token stream the direct marker would vouch for.
                self.engine_ids_len = (
                    len(self.prefix_ids) if resynced else None
                )
        # EOS is a terminal certificate edge, not an incremental parser
        # state. The row is complete and will not query another domain.
        if int(token_id) != int(tokenizer.eos_id):
            self._advance_completion_token(
                int(token_id),
                require_advertised=require_completion_advertised,
            )
        self.clear_position_memo()
        return self.prefix_text


def make_grammar_state(
    *,
    grammar_dsl: str | None = None,
    verify_chosen_only: bool = False,
    skip_exact_stream_probe: bool = True,
    use_copy_probes: bool = True,
    early_exit_pick: bool = True,
) -> GrammarDecodeState:
    """Construct a fresh per-row grammar state with a reusable DFA engine."""
    engine = _dfa_engine(grammar_dsl)
    if engine is not None:
        engine.reset()
    return GrammarDecodeState(
        engine=engine,
        verify_chosen_only=verify_chosen_only,
        skip_exact_stream_probe=skip_exact_stream_probe,
        use_copy_probes=use_copy_probes,
        early_exit_pick=early_exit_pick,
    )


def force_emit_token_id(
    tokenizer: OpenUITokenizer,
    prefix_ids: list[int],
    *,
    grammar_dsl: str | None = None,
    state: GrammarDecodeState | None = None,
) -> int | None:
    """Return a forced next token id when the grammar DFA has a singleton structural emit."""
    from slm_training.models.decode_stats import get_active_stats

    dsl = grammar_dsl or active_dsl()
    try:
        from slm_training.dsl.grammar.fastpath import force_next_token_id
    except Exception:  # noqa: BLE001
        return None
    if state is not None:
        engine = state.engine
        if engine is None:
            return None
        prefix_text = state.sync_ids(tokenizer, prefix_ids)
    else:
        engine = _dfa_engine(dsl)
        if engine is None:
            return None
        stats = get_active_stats()
        t0 = time.perf_counter()
        prefix_text = _decode_prefix_text(tokenizer, prefix_ids)
        if stats is not None:
            stats.detok_ms += (time.perf_counter() - t0) * 1000.0
    stats = get_active_stats()
    already = (
        state.engine_in_sync(prefix_ids, prefix_text)
        if state is not None
        else (
            getattr(engine, "_prefix", None) == prefix_text
            and getattr(engine, "_ip", None) is not None
            and getattr(engine, "_synced_ok", True)
        )
    )
    t0 = time.perf_counter()
    tid = force_next_token_id(engine, tokenizer, prefix_text, assume_synced=already)
    if stats is not None and not already:
        # R2: only charge a sync when force_emit actually re-lexed.
        stats.dfa_sync_ms += (time.perf_counter() - t0) * 1000.0
        stats.dfa_sync_count += 1
    return tid


def exact_forced_token_id(
    tokenizer: OpenUITokenizer,
    prefix_ids: list[int],
    *,
    forced_token_id: int | None = None,
    slot_contract: list[str] | None = None,
    state: GrammarDecodeState | None = None,
    remaining_tokens: int | None = None,
    runtime_symbols: tuple[object, ...] = (),
) -> int | None:
    """Return an id only when exact authorities prove one legal continuation.

    DSL-native tokenizers use the active pack's complete completion domain, so
    scope-aware semantic singletons bypass inference just like structural
    singletons. Other tokenizers retain the stricter DFA/vocabulary proof.

    A *horizon-limited* completion domain is not a contradiction. When the
    compiler cannot enumerate a terminal witness inside the remaining budget
    (``coverage != "complete"``) it has proven nothing either way, so the DFA
    proof below is consulted instead of discarding a decision the grammar has
    already made (decode invariant I2, ``docs/design/decode-invariants.md``). A
    *complete* domain that names more than one candidate genuinely contradicts
    a singleton and still refuses.

    The whitespace veto at the end of the DFA proof does not apply to the
    DSL-native codec. That veto exists because a compositional tokenizer can
    legally emit insignificant whitespace that competes with the forced lexeme
    for the picker's argmax. The native completion domain already excludes
    insignificant whitespace from its candidate set, so applying the veto only
    in the horizon-limited fallback would make short-horizon repair disagree
    with the very same position under a complete domain. Whitespace is not a
    symbol under the symbol-only output contract, and emitting the forced
    lexeme in its place cannot make a program illegal.
    """
    if state is None or state.engine is None:
        return None
    prefix_text = state.sync_ids(tokenizer, prefix_ids)
    engine = state.engine
    from slm_training.models.dsl_tokenizer import is_dsl_native_tokenizer

    forced = (
        int(forced_token_id)
        if forced_token_id is not None
        else force_emit_token_id(tokenizer, prefix_ids, state=state)
    )
    native = is_dsl_native_tokenizer(tokenizer)
    if native:
        try:
            from slm_training.dsl.grammar.fastpath.compiler_draft import (
                build_completion_forest,
            )
            from slm_training.models.decode_stats import get_active_stats, timed_ms

            # This is the same exact grammar-authority computation the
            # compiler-tree decode path times as "compiler_ms" (twotower.py's
            # _compiler_ltr_decode_one/_compiler_ltr_decode_batch). The legacy
            # LTR-repair loop calls it once per token via this function; left
            # unwrapped, that real cost fell entirely into unattributed_ms,
            # making compiler_ms_mean incomparable across checkpoints that
            # decode through different mechanisms (see docs/design/
            # compiler-tree-forced-closure-decode-metering-gap.md).
            with timed_ms(get_active_stats(), "compiler_ms"):
                forest = build_completion_forest(
                    tokenizer,
                    prefix_ids,
                    state=state,
                    slot_contract=slot_contract,
                    remaining_tokens=(
                        remaining_tokens
                        if remaining_tokens is not None
                        else getattr(state, "remaining_tokens", None)
                    ),
                    runtime_symbols=runtime_symbols,
                )
            candidates = set(forest.candidate_ids)
            if forest.coverage == "complete":
                # A complete domain is authoritative in both directions: one
                # candidate proves the singleton, more than one refutes it.
                if len(candidates) != 1:
                    return None
                return next(iter(candidates))
            # Horizon-limited: fall through to the DFA proof below.
        except (TimeoutError, KeyboardInterrupt):
            raise
        except Exception:  # noqa: BLE001 - incomplete proof must fail closed
            return None

    if not bool(getattr(engine, "terminals_are_exact", lambda: False)()):
        return None

    if forced is None:
        return None

    try:
        from slm_training.dsl.grammar.fastpath.token_map import allowed_id_set

        dfa_allowed = allowed_id_set(tokenizer, engine.next_terminals())
    except (TimeoutError, KeyboardInterrupt):
        raise
    except Exception:  # noqa: BLE001 - incomplete proof must fail closed
        return None
    if dfa_allowed != {forced}:
        return None

    if not dfa_admits_token(
        tokenizer,
        prefix_ids,
        forced,
        engine=engine,
        prefix_text=prefix_text,
        state=state,
    ):
        return None

    if native:
        # Insignificant whitespace is not a candidate in the native completion
        # domain, so it does not get a veto here either. See the docstring.
        return forced

    # The picker honors a structural force ahead of every non-whitespace
    # candidate. Its one deliberate exception is a legal whitespace argmax,
    # so prove that no such tokenizer token can compete.
    current_line = prefix_text.rstrip().split("\n")[-1].strip()
    excluded = {
        tokenizer.pad_id,
        tokenizer.mask_id,
        tokenizer.bos_id,
        tokenizer.unk_id,
    }
    for token_id in range(int(tokenizer.vocab_size)):
        if token_id in excluded:
            continue
        token = tokenizer.id_to_token.get(token_id, "")
        if not _token_surface_piece(tokenizer, token_id).isspace():
            continue
        if token == "NL" and (
            re.fullmatch(r"(?:[A-Za-z_]\w*|b\d+)", current_line)
            or (current_line == "root" and "=" not in prefix_text)
        ):
            continue
        if dfa_admits_token(
            tokenizer,
            prefix_ids,
            token_id,
            engine=engine,
            prefix_text=prefix_text,
            state=state,
        ):
            return None
    return forced


def _incomplete_quoted_string(prefix_text: str) -> bool:
    """True when the prefix ends inside an unclosed double-quoted string."""
    return prefix_text.count('"') % 2 == 1


def contract_allowed_token_ids(
    tokenizer: OpenUITokenizer,
    prefix_ids: list[int],
    slot_contract: list[str] | None,
) -> set[int] | None:
    """
    When building a quoted placeholder, return allowed next token ids from the
    slot contract inventory. None means no contract filter applies.

    For lexer-native tokenizers with a symbol table, this returns the unused
    ``<SYM_i>`` ids corresponding to the inventory.
    """
    if not slot_contract:
        return None

    try:
        from slm_training.models.dsl_tokenizer import (
            SymbolTable,
            is_dsl_native_tokenizer,
        )

        if is_dsl_native_tokenizer(tokenizer):
            table = SymbolTable.from_placeholders(
                slot_contract, max_slots=tokenizer.sym_slots
            )
            prefix_id_set = set(prefix_ids)
            allowed: set[int] = set()
            for i, _ph in enumerate(table.placeholders):
                token_id = tokenizer.sym_id(i)
                if token_id not in prefix_id_set:
                    allowed.add(token_id)
            return allowed
    except Exception:  # noqa: BLE001
        pass

    prefix_text = _decode_prefix_text(tokenizer, prefix_ids)
    if not _incomplete_quoted_string(prefix_text):
        return None

    from slm_training.models.tokenizer import tokenize_text

    last_open = prefix_text.rfind('"')
    built = prefix_text[last_open + 1 :]
    # Ordinary string literals ("column", "row") must not use placeholder contract.
    if not built.startswith(":"):
        return None

    built_seq = tokenize_text(f'"{built}')
    allowed = set()
    for ph in slot_contract:
        target = ph if ph.startswith(":") else f":{ph}"
        target_seq = tokenize_text(f'"{target}"')
        if len(built_seq) > len(target_seq):
            continue
        if target_seq[: len(built_seq)] != built_seq:
            continue
        if len(built_seq) < len(target_seq):
            tok = target_seq[len(built_seq)]
            tid = tokenizer.token_to_id.get(tok)
            if tid is not None:
                allowed.add(tid)
        else:
            # Complete placeholder — allow closing quote if present in target
            if len(target_seq) > len(built_seq):
                tok = target_seq[len(built_seq)]
                tid = tokenizer.token_to_id.get(tok)
                if tid is not None:
                    allowed.add(tid)
    return allowed or None


def _dfa_engine(grammar_dsl: str | None = None):
    try:
        from slm_training.dsl.grammar.fastpath import engine_for_dsl
    except Exception:  # noqa: BLE001
        return None
    return engine_for_dsl(grammar_dsl or active_dsl())


def dfa_admits_token(
    tokenizer: OpenUITokenizer,
    prefix_ids: list[int],
    token_id: int,
    *,
    grammar_dsl: str | None = None,
    engine=None,
    prefix_text: str | None = None,
    state: GrammarDecodeState | None = None,
) -> bool:
    """True iff Lark incremental parse accepts ``prefix + token`` as a legal prefix."""
    from slm_training.models.decode_stats import get_active_stats

    tid = int(token_id)
    if state is not None and tid in state.admit_memo:
        return state.admit_memo[tid]

    chunk = _token_surface_piece(tokenizer, tid)

    if prefix_text is None:
        if state is not None:
            prefix_text = state.sync_ids(tokenizer, prefix_ids)
        else:
            stats = get_active_stats()
            t0 = time.perf_counter()
            prefix_text = _decode_prefix_text(tokenizer, prefix_ids)
            if stats is not None:
                stats.detok_ms += (time.perf_counter() - t0) * 1000.0

    # Q2: whitespace fast-admit — ignorable WS never changes DFA state.
    if chunk and chunk.isspace() and (state is not None or engine is not None):
        eng = (state.engine if state is not None else None) or engine
        if eng is not None and getattr(eng, "_ip", None) is not None:
            if state is not None and state.whitespace_ok is not None:
                ok = state.whitespace_ok
            else:
                # No lex/feed needed: Lark treats WS as insignificant.
                ok = True
                stats = get_active_stats()
                if stats is not None:
                    stats.dfa_sync_count += 1
                if state is not None:
                    state.whitespace_ok = ok
            if state is not None:
                state.admit_memo[tid] = ok
            return ok

    # Q1: copy-based probe on the shared synced engine.
    eng = (state.engine if state is not None else None) or engine
    use_copy = bool(state is not None and state.use_copy_probes and eng is not None)
    if use_copy and getattr(eng, "_ip", None) is not None:
        # R2: only re-sync when the shared engine drifted from prefix_text.
        # P3: a direct-fed engine is id-synced without textual equality.
        drifted = (
            not state.engine_in_sync(prefix_ids, prefix_text)
            if state is not None and eng is state.engine
            else getattr(eng, "_prefix", None) != prefix_text
        )
        if drifted:
            stats = get_active_stats()
            t0 = time.perf_counter()
            try:
                eng.set_prefix(prefix_text)  # type: ignore[union-attr]
            except (TimeoutError, KeyboardInterrupt):
                raise
            except Exception:  # noqa: BLE001
                pass
            if stats is not None:
                stats.dfa_sync_ms += (time.perf_counter() - t0) * 1000.0
                stats.dfa_sync_count += 1
        stats = get_active_stats()
        t0 = time.perf_counter()
        try:
            probed = eng.probe_chunk(chunk)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            probed = None
        if probed is not None:
            ok = bool(probed)
            if stats is not None:
                stats.dfa_sync_ms += (time.perf_counter() - t0) * 1000.0
                stats.dfa_sync_count += 1
            if state is not None:
                state.admit_memo[tid] = ok
            return ok
        if stats is not None:
            # Fallback path continues timing below.
            pass

    # Fallback: throwaway engine + full set_prefix (safe, O(|prefix|)).
    try:
        from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
    except Exception:  # noqa: BLE001
        return True
    base = eng if eng is not None else _dfa_engine(grammar_dsl)
    if base is None:
        probe_engine = _dfa_engine(grammar_dsl)
        if probe_engine is None:
            return True
    else:
        grammar_path = getattr(base, "grammar_path", None)
        probe_engine = OpenUIIncrementalEngine(grammar_path)
    text = prefix_text + chunk
    stats = get_active_stats()
    t0 = time.perf_counter()
    try:
        ok = bool(probe_engine.set_prefix(text))
    except Exception:  # noqa: BLE001
        ok = False
    if stats is not None:
        stats.dfa_sync_ms += (time.perf_counter() - t0) * 1000.0
        stats.dfa_sync_count += 1
    if state is not None:
        state.admit_memo[tid] = ok
    return ok


def _stream_probe_ok(
    tokenizer: OpenUITokenizer,
    prefix_ids: list[int],
    token_id: int,
    *,
    prefix_text: str | None = None,
) -> bool:
    """Reject unknown-component / typed hard errors via streaming semantic check."""
    if prefix_text is None:
        trial_ids = [*prefix_ids, int(token_id)]
        text = tokenizer.decode(trial_ids)
    else:
        chunk = tokenizer.id_to_token.get(int(token_id), "")
        if chunk == "":
            chunk = tokenizer.decode([int(token_id)])
        text = prefix_text + chunk
    token = tokenizer.id_to_token.get(int(token_id), "")
    # Incomplete quoted strings / closing delimiters: probe as-is (no synthetic '(').
    if (
        token in {")", "]", '"', ",", "=", ":", ".", " "}
        or _incomplete_quoted_string(text)
        or text.rstrip().endswith((")", "]", '"', ":"))
    ):
        probe = text
    elif text.endswith(("(", "[", ",", "=", " ", "\n")):
        probe = text
    else:
        probe = f"{text}("
    try:
        status = stream_check(probe)
    except Exception:  # noqa: BLE001
        return False
    return not status.hard_error


def _placeholder_interior_allowed_ids(
    tokenizer: OpenUITokenizer,
    prefix_ids: list[int],
    *,
    prefix_text: str | None = None,
) -> set[int] | None:
    """When inside a quoted `:placeholder`, allow compositional subtoken ids."""
    text = (
        prefix_text
        if prefix_text is not None
        else _decode_prefix_text(tokenizer, prefix_ids)
    )
    if not _incomplete_quoted_string(text):
        return None
    last_open = text.rfind('"')
    built = text[last_open + 1 :]
    if not built.startswith(":"):
        return None
    ids: set[int] = set()
    for tok, tid in tokenizer.token_to_id.items():
        if tok in {'"', ':', '.'}:
            ids.add(tid)
        elif tok and tok.isidentifier() and tok[0].islower():
            ids.add(tid)
    return ids or None


def pick_constrained_token(
    logits_1d,
    tokenizer: OpenUITokenizer,
    prefix_ids: list[int],
    *,
    top_k: int = 16,
    forced_token_id: int | None = None,
    slot_contract: list[str] | None = None,
    prefer_structural: bool = True,
    sample: bool = False,
    temperature: float = 0.8,
    state: GrammarDecodeState | None = None,
    verify_chosen_only: bool | None = None,
    grammar_equivalence_cache: bool = False,
    active_dynamic_ids: set[int] | None = None,
    remaining_tokens: int | None = None,
    runtime_symbols: tuple[object, ...] = (),
) -> int | None:
    """
    Speculative constrained pick: only tokens admitted by the grammar DFA
    (and not rejected by stream hard-errors) may be selected.

    This is *pseudo* speculative decoding (verify against the OpenUI acceptor),
    not draft-model speculative decoding — see docs/design/research-lineage.md.

    When ``forced_token_id`` is set (singleton DFA structural emit), that id is
    returned only if the DFA still admits it.

    When ``slot_contract`` is set and the prefix is inside a quoted placeholder,
    candidates are further restricted to the inventory continuation.

    Returns ``None`` when no legal candidate exists (never returns a DFA-illegal
    or hard-error token).
    """
    import torch

    from slm_training.models.decode_stats import get_active_stats

    stats = get_active_stats()
    pick_t0 = time.perf_counter()
    if stats is not None:
        # Early returns do not enumerate the full legal set; prevent a prior
        # pick's count from being reported for this token.
        stats.constrained_last_legal_candidates = -1

    if state is not None:
        prefix_text = state.sync_ids(tokenizer, prefix_ids)
        engine = state.engine
        vco = (
            bool(verify_chosen_only)
            if verify_chosen_only is not None
            else bool(state.verify_chosen_only)
        )
        skip_exact = bool(state.skip_exact_stream_probe)
    else:
        t0 = time.perf_counter()
        prefix_text = _decode_prefix_text(tokenizer, prefix_ids)
        if stats is not None:
            stats.detok_ms += (time.perf_counter() - t0) * 1000.0
        engine = _dfa_engine()
        vco = bool(verify_chosen_only) if verify_chosen_only is not None else False
        skip_exact = True

    # Live decoders carry an explicit horizon on their state. Direct analysis
    # callers do not; keep those on the immediate-frontier path instead of
    # inventing a 64-token production budget and proving terminal reachability
    # for every broad literal candidate.
    domain_budget = (
        remaining_tokens
        if remaining_tokens is not None
        else getattr(state, "remaining_tokens", None)
    )

    contract_allowed = contract_allowed_token_ids(
        tokenizer, prefix_ids, slot_contract
    )

    allowed: set[int] | None = None
    exact_terminals = False
    if engine is not None:
        t0 = time.perf_counter()
        try:
            # R2: skip re-sync when P1 advance_token already left the engine
            # at this prefix_text (text route) or committed these exact ids
            # via direct terminal feeds (P3).
            already = (
                state.engine_in_sync(prefix_ids, prefix_text)
                if state is not None
                else (
                    getattr(engine, "_prefix", None) == prefix_text
                    and getattr(engine, "_ip", None) is not None
                )
            )
            if already:
                synced = True
            else:
                synced = bool(engine.set_prefix(prefix_text))
        except (TimeoutError, KeyboardInterrupt):
            raise
        except Exception:  # noqa: BLE001
            synced = False
            already = False
        if stats is not None and not already:
            stats.dfa_sync_ms += (time.perf_counter() - t0) * 1000.0
            stats.dfa_sync_count += 1
        if not synced and prefix_text.strip():
            # Prefix already illegal — no legal continuation.
            if stats is not None:
                stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
            return None
        try:
            from slm_training.dsl.grammar.fastpath.token_map import allowed_id_set

            allowed = allowed_id_set(
                tokenizer,
                engine.next_terminals(),
                active_dynamic_ids=(
                    None if domain_budget is not None else active_dynamic_ids
                ),
                use_cache=grammar_equivalence_cache,
            )
            exact_terminals = bool(
                skip_exact and getattr(engine, "terminals_are_exact", lambda: False)()
            )
        except (TimeoutError, KeyboardInterrupt):
            raise
        except Exception:  # noqa: BLE001
            allowed = None

    native_contract_symbol_ids: set[int] | None = None
    if contract_allowed is not None:
        # Slot-contract inventory is authoritative inside a quoted placeholder.
        # Intersecting with broad Lark terminals can empty the set (e.g. '.') —
        # prefer the inventory, then union with DFA when both agree.
        try:
            from slm_training.models.dsl_tokenizer import (
                TokenKind,
                is_dsl_native_tokenizer,
            )

            if is_dsl_native_tokenizer(tokenizer):
                native_contract_symbol_ids = tokenizer.kind_ids(TokenKind.SYM)
        except Exception:  # noqa: BLE001
            native_contract_symbol_ids = None
        if allowed is None:
            if native_contract_symbol_ids is None:
                allowed = set(contract_allowed)
        else:
            inter = allowed & contract_allowed
            if native_contract_symbol_ids is not None:
                # Native <SYM_i> tokens are complete STRING atoms. Restrict
                # only that token class; punctuation/components remain legal.
                allowed = (allowed - native_contract_symbol_ids) | inter
            else:
                allowed = inter if inter else set(contract_allowed)
        if not allowed:
            if stats is not None:
                stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
            return None

    ph_allowed = _placeholder_interior_allowed_ids(
        tokenizer, prefix_ids, prefix_text=prefix_text
    )
    if ph_allowed is not None:
        if allowed is None:
            allowed = set(ph_allowed)
        else:
            allowed = allowed | ph_allowed
        # Placeholder interiors are compositional — not exact structural.
        exact_terminals = False

    # At an empty native prefix the semantic root invariant is stronger than
    # the broad NAME/STATE terminals exposed by the surface lexer.
    if not prefix_text.strip():
        try:
            root_id = tokenizer.token_to_id.get("root")
            if root_id is None:
                root_id = tokenizer.bind_id(0)
            if 0 <= int(root_id) < int(logits_1d.numel()):
                if stats is not None:
                    stats.root_invariant_bypass_count += 1
                    stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
                return int(root_id)
        except Exception:  # noqa: BLE001
            pass

    # Lexer-native quoted literals are framed as LIT_STR + BYTE* + LIT_END.
    # Restrict by token-frame state because quote-equivalent surface terminals
    # cannot distinguish openers, closers, bytes, and complete symbol tokens.
    try:
        from slm_training.dsl.grammar.fastpath.token_map import apply_literal_frame
        from slm_training.models.dsl_tokenizer import is_dsl_native_tokenizer

        if is_dsl_native_tokenizer(tokenizer):
            allowed = apply_literal_frame(tokenizer, prefix_ids, allowed)
            exact_terminals = False
    except Exception:  # noqa: BLE001
        pass

    singleton_allowed = (
        int(next(iter(allowed))) if allowed is not None and len(allowed) == 1 else None
    )
    if stats is not None and singleton_allowed is not None:
        stats.constrained_last_legal_candidates = 1

    compiler_candidates: set[int] | None = None

    def _compiler_admits(tid: int) -> bool:
        nonlocal compiler_candidates
        try:
            if compiler_candidates is None:
                from slm_training.dsl.grammar.fastpath.compiler_draft import (
                    build_completion_forest,
                )

                forest = build_completion_forest(
                    tokenizer,
                    prefix_ids,
                    state=state,
                    slot_contract=slot_contract,
                    remaining_tokens=domain_budget,
                    runtime_symbols=runtime_symbols,
                )
                if (
                    forest.coverage != "complete"
                    and not runtime_symbols
                    and _incomplete_quoted_string(prefix_text)
                ):
                    forest = build_completion_forest(
                        tokenizer,
                        prefix_ids,
                        state=state,
                        slot_contract=slot_contract,
                        remaining_tokens=None,
                    )
                if forest.coverage != "complete":
                    # Direct analysis callers have no terminal horizon, so a
                    # partial compiler frontier is advisory; the DFA below
                    # remains the exact authority. Live decoders always carry
                    # a budget and therefore still fail closed here.
                    return domain_budget is None
                compiler_candidates = set(forest.candidate_ids)
            return int(tid) in compiler_candidates
        except (TimeoutError, KeyboardInterrupt):
            raise
        except Exception:  # noqa: BLE001 - strict domain authority fails closed
            return domain_budget is None

    def _legal(token_id: int, *, stream: bool = True) -> bool:
        tid = int(token_id)
        if tid in {
            tokenizer.pad_id,
            tokenizer.mask_id,
            tokenizer.bos_id,
            tokenizer.unk_id,
        }:
            return False
        token = tokenizer.id_to_token.get(tid, "")
        line = prefix_text.rstrip().split("\n")[-1].strip()
        if token in {"[", "LIT_STR"} and prefix_text.rstrip().endswith('"'):
            return False
        if token == "NL" and (
            re.fullmatch(r"(?:[A-Za-z_]\w*|b\d+)", line)
            or (line == "root" and "=" not in prefix_text)
            or prefix_text.count("[") > prefix_text.count("]")
            or prefix_text.count("(") > prefix_text.count(")")
            or prefix_text.count('"') % 2 == 1
        ):
            return False
        # EOS is a structural completion marker, not merely another DFA
        # terminal.  The incremental grammar accepts incomplete prefixes such
        # as ``root``/``root =`` via UnexpectedEOF; allowing EOS there strands
        # constrained LTR on a partial program.  Require the actual parser to
        # certify the prefix before permitting termination.
        if tid == tokenizer.eos_id:
            try:
                from slm_training.dsl.parser import validate

                program = validate(prefix_text.strip())
                if not getattr(program, "serialized", None):
                    return False
            except Exception:  # noqa: BLE001
                return False
        if contract_allowed is not None:
            if native_contract_symbol_ids is not None:
                if tid in native_contract_symbol_ids and tid not in contract_allowed:
                    return False
            elif tid not in contract_allowed:
                return False
        if not _compiler_admits(tid):
            return False
        # A complete compiler forest is already an exact grammar-and-schema
        # certificate for this prefix. Semantic, special-token, and contract
        # guards above still run first; avoid replaying the slower broad lexer
        # probe for a candidate the compiler has just certified.
        if compiler_candidates is not None and tid in compiler_candidates:
            return True
        # A singleton admission already proves lexical legality. Apply this
        # only after special-token, semantic, and contract checks above.
        if singleton_allowed == tid:
            return True
        # R1: when the DFA already lists this id in an exact (non-broad) accept
        # set, skip the redundant copy-probe admit — set_prefix + allowed_id_set
        # already certified it.
        in_allowed = allowed is not None and tid in allowed
        if in_allowed and exact_terminals:
            pass
        elif allowed is not None and tid not in allowed:
            # DFA terminal set can lag placeholder interiors — still admit when
            # incremental parse accepts the extension.
            if not dfa_admits_token(
                tokenizer,
                prefix_ids,
                tid,
                engine=engine,
                prefix_text=prefix_text,
                state=state,
            ):
                return False
        elif engine is not None and not in_allowed:
            # tid not covered by allowed (allowed was None) — must probe.
            if not dfa_admits_token(
                tokenizer,
                prefix_ids,
                tid,
                engine=engine,
                prefix_text=prefix_text,
                state=state,
            ):
                return False
        elif in_allowed and engine is not None:
            # Broad terminals (NAME/COMPONENT/…): only probe when the chunk
            # could glue onto / change an incomplete lexeme at the frontier.
            chunk = tokenizer.id_to_token.get(tid, "")
            if chunk == "NL":
                chunk = "\n"
            elif chunk in {"LIT_STR", "LIT_END"}:
                chunk = '"'
            elif chunk.startswith("B:"):
                try:
                    chunk = chr(int(chunk[2:], 16))
                except ValueError:
                    pass
            elif chunk == "" or chunk.startswith(("<BIND_", "<SYM_", "<STATE_")):
                chunk = tokenizer.decode([tid])
            needs_probe = bool(chunk) and (
                chunk[:1].isalnum() or chunk[:1] in {":", ".", "_", '"'}
            )
            if needs_probe and not dfa_admits_token(
                tokenizer,
                prefix_ids,
                tid,
                engine=engine,
                prefix_text=prefix_text,
                state=state,
            ):
                return False
        if not stream:
            return True
        # A complete compiler certificate subsumes the semantic probe. When
        # the compiler is unavailable/partial, however, a broad identifier DFA
        # terminal does not prove component-schema membership. Never let the
        # performance-oriented skip widen that uncertain frontier.
        needs_semantic_probe = compiler_candidates is None and token.isidentifier()
        if (
            needs_semantic_probe or not (skip_exact or exact_terminals)
        ) and not _stream_probe_ok(tokenizer, prefix_ids, tid, prefix_text=prefix_text):
            return False
        return True

    # Do not let the broad whitespace terminal commit a newline after a bare
    # binding name; that prefix cannot continue to an assignment.
    current_line = prefix_text.rstrip().split("\n")[-1].strip()
    if re.fullmatch(r"(?:[A-Za-z_]\w*|b\d+)", current_line) or (
        current_line == "root" and "=" not in prefix_text
    ):
        nl_id = tokenizer.token_to_id.get("NL")
        if nl_id is not None:
            logits_1d[int(nl_id)] = -float("inf")

    if forced_token_id is not None:
        # Force-emit comes from significant-lexeme DFA and can skip whitespace
        # tokens that our OpenUI tokenizer models explicitly. Prefer a legal
        # whitespace argmax over a structural force that would drop spaces;
        # otherwise honor the forced structural emit when it remains legal.
        argmax_id = int(logits_1d.argmax().item())
        argmax_tok = tokenizer.id_to_token.get(argmax_id, "")
        if (
            argmax_id != int(forced_token_id)
            and (
                argmax_tok in {" ", "\n", "\t"}
                or (argmax_tok and argmax_tok.isspace())
            )
            and _legal(argmax_id)
        ):
            if stats is not None:
                stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
            return argmax_id
        if _legal(int(forced_token_id)):
            if stats is not None:
                stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
            return int(forced_token_id)
        forced_token_id = None

    # The post-filtered DFA set is authoritative. Do not let an invalid model
    # argmax or an empty probe expansion erase a proven singleton choice.
    if singleton_allowed is not None and _legal(singleton_allowed):
        if stats is not None:
            stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
        return singleton_allowed

    # P2: verify the model-chosen token first; only expand on rejection.
    if vco and not sample:
        argmax_id = int(logits_1d.argmax().item())
        if _legal(argmax_id):
            if stats is not None:
                stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
            return argmax_id

    # A legal model argmax is already the best constrained decision. Structural
    # preference only breaks ties among candidates; it must not replace it.
    if not sample:
        argmax_id = int(logits_1d.argmax().item())
        if _legal(argmax_id):
            if stats is not None:
                stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
            return argmax_id

    backend = _backend()
    vocab = int(logits_1d.numel())
    search_k = min(max(top_k, 1), vocab)
    early_exit = bool(state is not None and state.early_exit_pick and not sample)

    # Q2: materialize logits once (avoids per-candidate .item() syncs).
    logits_list = logits_1d.detach().tolist()

    # Score legal candidates: expand beyond the DFA terminal set so whitespace
    # and compositionally admitted tokens (placeholder interiors) compete with
    # the highest model logits.
    if allowed is not None and allowed:
        candidate_ids = set(allowed)
        # Always let the model vote: include top-k logits so whitespace etc.
        # that pass `_legal` via dfa_admits aren't dropped solely because the
        # Lark terminal set omits insignificant tokens.
        # Exact terminal sets are already complete legal inventories; adding
        # top-k candidates there only creates redundant broad-token probes.
        if not exact_terminals:
            _vals, top_idx = torch.topk(logits_1d, k=min(max(top_k, 1), vocab))
            candidate_ids.update(int(i) for i in top_idx.tolist())
        # Descending-logit order for early-exit (Q2).
        ordered = sorted(
            (tid for tid in candidate_ids if 0 <= tid < vocab),
            key=lambda tid: logits_list[tid],
            reverse=True,
        )
        preferred_names = frozenset()
        struct = structural_tokens() if prefer_structural else frozenset()
        scored: list[tuple[float, int]] = []
        best_score: float | None = None
        for tid in ordered:
            if not _legal(tid):
                continue
            score = float(logits_list[tid])
            if best_score is None:
                best_score = score
            scored.append((score, tid))
            if early_exit and not sample:
                if not prefer_structural:
                    # First legal (highest logit) wins.
                    if stats is not None:
                        stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
                    return tid
                # A structural token is only a tie-breaker. Never override a
                # clearly higher-scoring legal NAME/BIND or literal token.
                if best_score - score > 1.0:
                    break
                token = tokenizer.id_to_token.get(tid, "")
                if token in preferred_names or token in struct:
                    if stats is not None:
                        stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
                    return tid
        if scored:
            if sample and temperature > 0:
                scores = torch.tensor([s[0] for s in scored], dtype=logits_1d.dtype)
                probs = torch.softmax(scores / temperature, dim=0)
                idx = int(torch.multinomial(probs, 1).item())
                if stats is not None:
                    stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
                return scored[idx][1]
            if prefer_structural:
                # Prefer structural tokens only when they are near the top score
                # (within 1.0 logit) — never override a clearly better argmax.
                assert best_score is not None
                for score, tid in scored:
                    if best_score - score > 1.0:
                        break
                    token = tokenizer.id_to_token.get(tid, "")
                    if token in preferred_names or token in struct:
                        if stats is not None:
                            stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
                        return tid
            if stats is not None:
                stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
            return scored[0][1]

    # Escalate top-k search if no allowed-set hit (or allowed was broad/None).
    for k in (search_k, min(max(search_k * 4, 64), vocab), vocab):
        _values, indices = torch.topk(logits_1d, k=k)
        if not backend.available() and engine is None:
            # Cannot certify legality — refuse rather than emit unconstrained top-1.
            if stats is not None:
                stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
            return None

        preferred_names = frozenset()
        struct = structural_tokens()
        preferred: list[int] = []
        acceptable: list[int] = []

        for idx in indices.tolist():
            token_id = int(idx)
            if not _legal(token_id):
                continue
            token = tokenizer.id_to_token.get(token_id, "")
            if prefer_structural:
                text = prefix_text + (
                    tokenizer.id_to_token.get(token_id, "")
                    or tokenizer.decode([token_id])
                )
                if token in {")", "]", '"', ",", "="} or text.rstrip().endswith(
                    (")", "]", '"')
                ):
                    probe = text
                elif text.endswith(("(", "[", ",", "=", " ", "\n")):
                    probe = text
                else:
                    probe = f"{text}("
                status = None
                # `_legal` already applies the configured stream-probe policy.
                # Do not bypass `skip_exact_stream_probe` here: this fallback
                # ranking loop can inspect hundreds of candidates, and a
                # direct LangCore probe per candidate makes constrained decode
                # effectively unbounded on CPU.
                if not skip_exact and not exact_terminals:
                    try:
                        status = stream_check(probe)
                    except Exception:  # noqa: BLE001
                        status = None
                if (
                    token in preferred_names
                    or token in struct
                    or (
                        status is not None
                        and (status.has_root or status.incomplete or status.complete_ok)
                    )
                ):
                    preferred.append(token_id)
                    if early_exit:
                        if stats is not None:
                            stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
                        return token_id
                else:
                    acceptable.append(token_id)
                    if early_exit and not prefer_structural:
                        if stats is not None:
                            stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
                        return token_id
            else:
                acceptable.append(token_id)
                if early_exit:
                    if stats is not None:
                        stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
                    return token_id

        pool = preferred if (prefer_structural and preferred) else acceptable
        if stats is not None:
            stats.constrained_last_legal_candidates = len(pool)
        if pool:
            if sample and temperature > 0:
                scores = torch.tensor(
                    [float(logits_list[i]) for i in pool],
                    dtype=logits_1d.dtype,
                )
                probs = torch.softmax(scores / temperature, dim=0)
                pick = int(torch.multinomial(probs, 1).item())
                if stats is not None:
                    stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
                return pool[pick]
            if stats is not None:
                stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
            return pool[0]
        if k >= vocab:
            break
    if contract_allowed:
        # Last resort under an active placeholder contract: any inventory id
        # that still passes DFA/stream probes.
        for tid in contract_allowed:
            if _legal(tid):
                if stats is not None:
                    stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
                return tid
    if stats is not None:
        stats.constrained_last_legal_candidates = 0
        stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
    return None


def filter_ids_by_stream(
    tokenizer: OpenUITokenizer,
    token_ids: list[int],
    newly_filled: Iterable[int],
) -> list[int]:
    """
    If the current decode has hard errors, return positions among newly_filled
    that should be remasked.
    """
    text = tokenizer.decode(token_ids)
    try:
        status = stream_check(text)
    except TimeoutError:
        # A harness deadline is a decode failure, never a clean stream probe.
        raise
    except Exception:  # noqa: BLE001
        return []
    if not status.hard_error:
        # Also remask when the Lark DFA rejects the full prefix.
        engine = _dfa_engine()
        if engine is not None:
            try:
                if not engine.set_prefix(text):
                    return list(newly_filled)
            except (TimeoutError, KeyboardInterrupt):
                raise
            except Exception:  # noqa: BLE001
                pass
        return []
    return list(newly_filled)


__all__ = [
    "STRUCTURAL_TOKENS",
    "GrammarDecodeState",
    "StreamStatus",
    "active_dsl",
    "apply_structural_bias",
    "contract_allowed_token_ids",
    "dfa_admits_token",
    "filter_ids_by_stream",
    "exact_forced_token_id",
    "force_emit_token_id",
    "make_grammar_state",
    "pick_constrained_token",
    "set_active_dsl",
    "stream_check",
    "structural_token_ids",
    "structural_tokens",
]
