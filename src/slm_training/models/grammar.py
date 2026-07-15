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
import time
from dataclasses import dataclass, field
from typing import Iterable

from slm_training.dsl.openui_tokens import (
    PREFERRED_COMPONENT_NAMES,
    STRUCTURAL_TOKENS,
)
from slm_training.dsl.stream_types import StreamStatus
from slm_training.models.tokenizer import OpenUITokenizer

_STREAM_CACHE: dict[str, StreamStatus] = {}
_STREAM_CACHE_MAX = 2048
_STRUCT_ID_CACHE: dict[int, set[int]] = {}
_BIAS_CACHE: dict[tuple[int, float, str, str], object] = {}
_ACTIVE_DSL: str | None = None


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


def preferred_components() -> frozenset[str]:
    try:
        comps = _backend().component_names()
        if comps:
            return frozenset(c for c in comps if c in PREFERRED_COMPONENT_NAMES) or comps
    except Exception:  # noqa: BLE001
        pass
    return PREFERRED_COMPONENT_NAMES


def structural_token_ids(tokenizer: OpenUITokenizer) -> set[int]:
    cache_key = id(tokenizer.token_to_id)
    cached = _STRUCT_ID_CACHE.get(cache_key)
    if cached is not None and len(cached) > 0:
        return cached

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
            _STRUCT_ID_CACHE[cache_key] = ids
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
    _STRUCT_ID_CACHE[cache_key] = ids
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
    if not allowed:
        return logits
    cache_key = (
        id(tokenizer.token_to_id),
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

    def clear_position_memo(self) -> None:
        self.admit_memo.clear()
        self.whitespace_ok = None

    def sync_ids(self, tokenizer: OpenUITokenizer, prefix_ids: list[int]) -> str:
        """Update prefix_ids/text incrementally; return current prefix text."""
        from slm_training.models.decode_stats import get_active_stats

        stats = get_active_stats()
        if prefix_ids == self.prefix_ids:
            return self.prefix_text
        t0 = time.perf_counter()
        if (
            len(prefix_ids) >= len(self.prefix_ids)
            and prefix_ids[: len(self.prefix_ids)] == self.prefix_ids
        ):
            # Append-only growth — decode only the new suffix tokens.
            extra = prefix_ids[len(self.prefix_ids) :]
            if extra:
                chunk = tokenizer.decode(extra)
                self.prefix_text = self.prefix_text + chunk
            self.prefix_ids = list(prefix_ids)
        else:
            self.prefix_ids = list(prefix_ids)
            self.prefix_text = tokenizer.decode(prefix_ids) if prefix_ids else ""
        self.clear_position_memo()
        if stats is not None:
            stats.detok_ms += (time.perf_counter() - t0) * 1000.0
        return self.prefix_text

    def advance_token(self, tokenizer: OpenUITokenizer, token_id: int) -> str:
        """Append one emitted token to the cached prefix."""
        self.prefix_ids.append(int(token_id))
        from slm_training.models.decode_stats import get_active_stats

        stats = get_active_stats()
        t0 = time.perf_counter()
        chunk = tokenizer.id_to_token.get(int(token_id), "")
        if chunk == "":
            chunk = tokenizer.decode([int(token_id)])
        self.prefix_text = self.prefix_text + chunk
        if stats is not None:
            stats.detok_ms += (time.perf_counter() - t0) * 1000.0
        if self.engine is not None:
            try:
                self.engine.advance(chunk)  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                try:
                    self.engine.set_prefix(self.prefix_text)  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001
                    pass
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
        prefix_text = tokenizer.decode(prefix_ids)
        if stats is not None:
            stats.detok_ms += (time.perf_counter() - t0) * 1000.0
    stats = get_active_stats()
    already = (
        getattr(engine, "_prefix", None) == prefix_text
        and getattr(engine, "_ip", None) is not None
    )
    t0 = time.perf_counter()
    tid = force_next_token_id(engine, tokenizer, prefix_text)
    if stats is not None and not already:
        # R2: only charge a sync when force_emit actually re-lexed.
        stats.dfa_sync_ms += (time.perf_counter() - t0) * 1000.0
        stats.dfa_sync_count += 1
    return tid


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

    For lexer-native tokenizers with a symbol table, this returns the set of
    ``<SYM_i>`` ids corresponding to the inventory (prefix-independent).
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
            allowed: set[int] = set()
            for i, _ph in enumerate(table.placeholders):
                allowed.add(tokenizer.sym_id(i))
            return allowed or None
    except Exception:  # noqa: BLE001
        pass

    prefix_text = tokenizer.decode(prefix_ids)
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

    chunk = tokenizer.id_to_token.get(tid, "")
    if chunk == "":
        chunk = tokenizer.decode([tid])

    if prefix_text is None:
        if state is not None:
            prefix_text = state.sync_ids(tokenizer, prefix_ids)
        else:
            stats = get_active_stats()
            t0 = time.perf_counter()
            prefix_text = tokenizer.decode(prefix_ids)
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
        if getattr(eng, "_prefix", None) != prefix_text:
            stats = get_active_stats()
            t0 = time.perf_counter()
            try:
                eng.set_prefix(prefix_text)  # type: ignore[union-attr]
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
    text = prefix_text if prefix_text is not None else tokenizer.decode(prefix_ids)
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
        prefix_text = tokenizer.decode(prefix_ids)
        if stats is not None:
            stats.detok_ms += (time.perf_counter() - t0) * 1000.0
        engine = _dfa_engine()
        vco = bool(verify_chosen_only) if verify_chosen_only is not None else False
        skip_exact = True

    contract_allowed = contract_allowed_token_ids(
        tokenizer, prefix_ids, slot_contract
    )

    allowed: set[int] | None = None
    exact_terminals = False
    if engine is not None:
        t0 = time.perf_counter()
        try:
            # R2: skip re-sync when P1 advance_token already left the engine
            # at this prefix_text.
            already = getattr(engine, "_prefix", None) == prefix_text and getattr(
                engine, "_ip", None
            ) is not None
            if already:
                synced = True
            else:
                synced = bool(engine.set_prefix(prefix_text))
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

            allowed = allowed_id_set(tokenizer, engine.next_terminals())
            exact_terminals = bool(
                skip_exact and getattr(engine, "terminals_are_exact", lambda: False)()
            )
        except Exception:  # noqa: BLE001
            allowed = None

    if contract_allowed is not None:
        # Slot-contract inventory is authoritative inside a quoted placeholder.
        # Intersecting with broad Lark terminals can empty the set (e.g. '.') —
        # prefer the inventory, then union with DFA when both agree.
        if allowed is None:
            allowed = set(contract_allowed)
        else:
            inter = allowed & contract_allowed
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

    def _legal(token_id: int, *, stream: bool = True) -> bool:
        tid = int(token_id)
        if tid in {
            tokenizer.pad_id,
            tokenizer.mask_id,
            tokenizer.bos_id,
            tokenizer.unk_id,
        }:
            return False
        if contract_allowed is not None and tid not in contract_allowed:
            return False
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
            if chunk == "":
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
        if exact_terminals:
            return True
        return _stream_probe_ok(
            tokenizer, prefix_ids, tid, prefix_text=prefix_text
        )

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

    # P2: verify the model-chosen token first; only expand on rejection.
    if vco and not sample:
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
        _vals, top_idx = torch.topk(logits_1d, k=min(max(top_k, 1), vocab))
        candidate_ids.update(int(i) for i in top_idx.tolist())
        # Descending-logit order for early-exit (Q2).
        ordered = sorted(
            (tid for tid in candidate_ids if 0 <= tid < vocab),
            key=lambda tid: logits_list[tid],
            reverse=True,
        )
        preferred_names = preferred_components() if prefer_structural else frozenset()
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
                token = tokenizer.id_to_token.get(tid, "")
                if token in preferred_names or token in struct:
                    if stats is not None:
                        stats.pick_ms += (time.perf_counter() - pick_t0) * 1000.0
                    return tid
                # Outside the 1.0-logit structural window — best non-structural wins.
                if best_score - score > 1.0:
                    break
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

        preferred_names = preferred_components()
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
    except Exception:  # noqa: BLE001
        return []
    if not status.hard_error:
        # Also remask when the Lark DFA rejects the full prefix.
        engine = _dfa_engine()
        if engine is not None:
            try:
                if not engine.set_prefix(text):
                    return list(newly_filled)
            except Exception:  # noqa: BLE001
                pass
        return []
    return list(newly_filled)


__all__ = [
    "PREFERRED_COMPONENT_NAMES",
    "STRUCTURAL_TOKENS",
    "GrammarDecodeState",
    "StreamStatus",
    "active_dsl",
    "apply_structural_bias",
    "contract_allowed_token_ids",
    "dfa_admits_token",
    "filter_ids_by_stream",
    "force_emit_token_id",
    "make_grammar_state",
    "pick_constrained_token",
    "preferred_components",
    "set_active_dsl",
    "stream_check",
    "structural_token_ids",
    "structural_tokens",
]
