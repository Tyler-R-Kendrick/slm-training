"""Lexer-native OpenUI output tokenizer (V5 / Stage 1–2).

Replaces string-piece BPE-style output ids with a typed compiler-derived alphabet:

* fixed grammar terminals + component keywords
* dynamic per-example symbol table for placeholders (``<SYM_i>``)
* alpha-renamed binder pool (``<BIND_j>``)
* typed literal channel with byte/char fallback

Whitespace is not modeled. ``decode`` pretty-prints a deterministic OpenUI source.
Round-trip: ``decode(encode(canonicalize(src))) == canonicalize(src)`` when the
same ``SymbolTable`` is used.
"""

from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

from slm_training.data.contract import RuntimeSymbol
from slm_training.dsl.openui_tokens import STRUCTURAL_TOKENS

# Bump when serialization / vocab layout changes.
DSL_TOKENIZER_VERSION = 2
SYMBOL_TABLE_VERSION = 3

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
MASK = "<mask>"
UNK = "<unk>"
SPECIAL = [PAD, BOS, EOS, MASK, UNK]

# Reserved table sizes (kept small / fixed so vocabulary is corpus-independent).
DEFAULT_SYM_SLOTS = 64
DEFAULT_BIND_SLOTS = 64
DEFAULT_STATE_SLOTS = 64

LIT_STR = "LIT_STR"
LIT_NUM = "LIT_NUM"
LIT_END = "LIT_END"
NL = "NL"

_BYTE_PREFIX = "B:"


class TokenKind(str, Enum):
    SPECIAL = "special"
    STRUCT = "struct"
    COMPONENT = "component"
    SYM = "sym"
    BIND = "bind"
    STATE = "state"
    BUILTIN = "builtin"
    LIT = "lit"
    BYTE = "byte"


# Common closed string atoms that appear as OpenUI props (layout / size / tone).
_FIXED_STRING_BODIES: tuple[str, ...] = tuple(
    sorted(
        {
            s
            for s in STRUCTURAL_TOKENS
            if s
            and s[0].islower()
            and s.isidentifier()
            or s
            in {
                "2xl",
                "small",
                "default",
                "large",
                "small-heavy",
                "large-heavy",
                "column",
                "row",
                "none",
                "xs",
                "s",
                "m",
                "l",
                "xl",
                "primary",
                "secondary",
                "tertiary",
                "text",
                "info",
            }
        }
    )
)

_COMPONENT_NAMES: tuple[str, ...] = tuple(
    sorted(
        {
            s
            for s in STRUCTURAL_TOKENS
            if s[:1].isupper() and s.isidentifier()
        }
        | {
            "Stack",
            "Card",
            "CardHeader",
            "TextContent",
            "Button",
            "Buttons",
            "Input",
            "Form",
            "FormControl",
            "Label",
            "TextArea",
            "Select",
            "SelectItem",
            "CheckBoxGroup",
            "CheckBoxItem",
            "RadioGroup",
            "RadioItem",
            "SwitchGroup",
            "SwitchItem",
            "Slider",
            "DatePicker",
            "Image",
            "ImageBlock",
            "ImageGallery",
            "Modal",
            "Tabs",
            "TabItem",
            "Callout",
            "TextCallout",
            "Separator",
            "Table",
            "Col",
        }
    )
)

_STRUCT_PUNCT: tuple[str, ...] = (
    "=",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    ",",
    ".",
    ":",
    "?",
    "+",
    "-",
    "*",
    "/",
    "%",
    "!",
    "==",
    "!=",
    ">",
    "<",
    ">=",
    "<=",
    "&&",
    "||",
)
_BOOLS: tuple[str, ...] = ("true", "false", "null")
_BUILTIN_NAMES: tuple[str, ...] = (
    "Query",
    "Mutation",
    "Action",
    "@Run",
    "@Set",
    "@Reset",
    "@ToAssistant",
    "@OpenUrl",
    "@Count",
    "@First",
    "@Last",
    "@Sum",
    "@Avg",
    "@Min",
    "@Max",
    "@Sort",
    "@Filter",
    "@Round",
    "@Abs",
    "@Floor",
    "@Ceil",
    "@Each",
)

# OpenUI surface lexer used for serialization (aligned with src/slm_training/dsl/grammars/openui.lark).
_NUMBER_PATTERN = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NUMBER_RE = re.compile(_NUMBER_PATTERN)
_LEX_RE = re.compile(
    r"""
    (//[^\n]*|\#[^\n]*)
  | ("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')
  | (\$[A-Za-z_][A-Za-z0-9_]*)
  | (@[A-Z][A-Za-z0-9_]*)
  | ([A-Z][A-Za-z0-9_]*)
  | ([a-z_][A-Za-z0-9_]*)
  | ("""
    + _NUMBER_PATTERN
    + r""")
  | (==|!=|>=|<=|&&|\|\||=|\(|\)|\[|\]|\{|\}|,|\.|:|\?|\+|-|\*|/|%|!|>|<)
  | (\n+)
  | ([^\S\n]+)
    """,
    re.VERBOSE,
)

_PLACEHOLDER_BODY_RE = re.compile(
    r"^:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)


def _fixed_string_token(body: str) -> str:
    return f'STR:{body}'


def _byte_token(ch: str) -> str:
    return f"{_BYTE_PREFIX}{ord(ch):02x}"


def _sym_token(i: int) -> str:
    return f"<SYM_{i}>"


def _bind_token(i: int) -> str:
    return f"<BIND_{i}>"


def _state_token(i: int) -> str:
    return f"<STATE_{i}>"


def _bind_surface_name(i: int) -> str:
    """Deterministic OpenUI binder identifier for slot i."""
    return f"b{i}"


def _state_surface_name(i: int) -> str:
    return f"$s{i}"


@dataclass
class SymbolTable:
    """Per-example dynamic symbol table for placeholders (and optional binders)."""

    placeholders: list[str] = field(default_factory=list)
    # Maps surface binder name -> bind slot (first-occurrence order).
    binders: dict[str, int] = field(default_factory=dict)
    states: dict[str, int] = field(default_factory=dict)
    runtime_symbols: list[RuntimeSymbol] = field(default_factory=list)

    def _record(self, symbol: RuntimeSymbol) -> None:
        if not any(item.surface == symbol.surface for item in self.runtime_symbols):
            self.runtime_symbols.append(symbol)

    def placeholder_slot(self, ph: str) -> int | None:
        key = ph if ph.startswith(":") else f":{ph}"
        try:
            return self.placeholders.index(key)
        except ValueError:
            return None

    def ensure_placeholder(self, ph: str, *, max_slots: int) -> int | None:
        key = ph if ph.startswith(":") else f":{ph}"
        slot = self.placeholder_slot(key)
        if slot is not None:
            return slot
        if len(self.placeholders) >= max_slots:
            return None
        self.placeholders.append(key)
        self._record(RuntimeSymbol(surface=key, role="external_entity"))
        return len(self.placeholders) - 1

    def ensure_binder(self, name: str, *, max_slots: int) -> int | None:
        if name in self.binders:
            return self.binders[name]
        if max_slots <= 0:
            return None
        if name == "root":
            if 0 in self.binders.values():
                if max(self.binders.values(), default=-1) + 1 >= max_slots:
                    return None
                self.binders = {key: value + 1 for key, value in self.binders.items()}
            self.binders[name] = 0
            self._record(RuntimeSymbol(surface=name, role="alpha_binder"))
            return 0
        used = set(self.binders.values())
        slot = next((candidate for candidate in range(max_slots) if candidate not in used), None)
        if slot is None:
            return None
        self.binders[name] = slot
        self._record(RuntimeSymbol(surface=name, role="alpha_binder"))
        return slot

    def ensure_state(self, name: str, *, max_slots: int) -> int | None:
        key = name if name.startswith("$") else f"${name}"
        if key in self.states:
            return self.states[key]
        if len(self.states) >= max_slots:
            return None
        slot = len(self.states)
        self.states[key] = slot
        self._record(RuntimeSymbol(surface=key, role="state"))
        return slot

    def symbol_for_surface(self, surface: str) -> RuntimeSymbol | None:
        return next(
            (symbol for symbol in self.runtime_symbols if symbol.surface == surface),
            None,
        )

    def binder_name(self, slot: int) -> str:
        for name, s in self.binders.items():
            if s == slot:
                return name
        return _bind_surface_name(slot)

    def placeholder_at(self, slot: int) -> str | None:
        if 0 <= slot < len(self.placeholders):
            return self.placeholders[slot]
        return None

    def state_name(self, slot: int) -> str:
        for name, state_slot in self.states.items():
            if state_slot == slot:
                return name
        return _state_surface_name(slot)

    def to_dict(self) -> dict:
        return {
            "version": SYMBOL_TABLE_VERSION,
            "placeholders": list(self.placeholders),
            "binders": dict(self.binders),
            "states": dict(self.states),
            "runtime_symbols": [symbol.to_dict() for symbol in self.runtime_symbols],
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> SymbolTable:
        if not data:
            return cls()
        table = cls(
            placeholders=list(data.get("placeholders") or []),
            binders={str(k): int(v) for k, v in (data.get("binders") or {}).items()},
            states={str(k): int(v) for k, v in (data.get("states") or {}).items()},
            runtime_symbols=[
                RuntimeSymbol.from_dict(item)
                for item in data.get("runtime_symbols") or ()
            ],
        )
        # v2 tables had no typed metadata. Reconstruct it deterministically.
        for surface in table.placeholders:
            table._record(RuntimeSymbol(surface=surface, role="external_entity"))
        for surface in table.binders:
            table._record(RuntimeSymbol(surface=surface, role="alpha_binder"))
        for surface in table.states:
            table._record(RuntimeSymbol(surface=surface, role="state"))
        return table

    @classmethod
    def from_placeholders(
        cls,
        placeholders: Iterable[str] | None,
        *,
        max_slots: int = DEFAULT_SYM_SLOTS,
    ) -> SymbolTable:
        table = cls()
        for raw in placeholders or []:
            table.ensure_placeholder(raw, max_slots=max_slots)
        return table

    @classmethod
    def from_runtime_symbols(
        cls,
        symbols: Iterable[RuntimeSymbol],
        *,
        sym_slots: int = DEFAULT_SYM_SLOTS,
        bind_slots: int = DEFAULT_BIND_SLOTS,
        state_slots: int = DEFAULT_STATE_SLOTS,
    ) -> SymbolTable:
        table = cls()
        for symbol in symbols:
            table._record(symbol)
            if symbol.role == "external_entity":
                table.ensure_placeholder(symbol.surface, max_slots=sym_slots)
            elif symbol.role == "state":
                table.ensure_state(symbol.surface, max_slots=state_slots)
            else:
                table.ensure_binder(symbol.surface, max_slots=bind_slots)
        return table

    def active_token_ids(self, tokenizer: DSLNativeTokenizer) -> set[int]:
        """Reserved vocabulary rows that are meaningful for this request."""
        ids = {tokenizer.sym_id(i) for i in range(len(self.placeholders))}
        ids.update(tokenizer.bind_id(i) for i in self.binders.values())
        ids.update(tokenizer.state_id(i) for i in self.states.values())
        return ids

    def permuted(self, seed: int) -> SymbolTable:
        """Training-only slot permutation; root binder remains slot zero."""
        rng = random.Random(seed)
        placeholders = list(self.placeholders)
        rng.shuffle(placeholders)
        binder_names = [name for name in self.binders if name != "root"]
        rng.shuffle(binder_names)
        binders = {"root": 0} if "root" in self.binders else {}
        start = 1 if binders else 0
        binders.update({name: start + i for i, name in enumerate(binder_names)})
        state_names = list(self.states)
        rng.shuffle(state_names)
        return SymbolTable(
            placeholders=placeholders,
            binders=binders,
            states={name: i for i, name in enumerate(state_names)},
            runtime_symbols=list(self.runtime_symbols),
        )


@dataclass
class DSLNativeTokenizer:
    """Fixed lexer-native vocabulary for OpenUI output sequences."""

    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)
    id_to_kind: dict[int, str] = field(default_factory=dict)
    version: int = DSL_TOKENIZER_VERSION
    sym_slots: int = DEFAULT_SYM_SLOTS
    bind_slots: int = DEFAULT_BIND_SLOTS
    state_slots: int = DEFAULT_STATE_SLOTS
    # Overflow counter (byte-path used when symbol table is full).
    overflow_count: int = 0

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    @property
    def mask_id(self) -> int:
        return self.token_to_id[MASK]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def kind_of(self, token_id: int) -> TokenKind:
        raw = self.id_to_kind.get(int(token_id), TokenKind.SPECIAL.value)
        try:
            return TokenKind(raw)
        except ValueError:
            return TokenKind.SPECIAL

    def kind_ids(self, kind: TokenKind | str) -> set[int]:
        want = kind.value if isinstance(kind, TokenKind) else str(kind)
        return {i for i, k in self.id_to_kind.items() if k == want}

    def is_sym_id(self, tid: int) -> bool:
        return self.kind_of(tid) == TokenKind.SYM

    def is_bind_id(self, tid: int) -> bool:
        return self.kind_of(tid) == TokenKind.BIND

    def is_state_id(self, tid: int) -> bool:
        return self.kind_of(tid) == TokenKind.STATE

    def sym_id(self, slot: int) -> int:
        return self.token_to_id[_sym_token(slot)]

    def bind_id(self, slot: int) -> int:
        return self.token_to_id[_bind_token(slot)]

    def state_id(self, slot: int) -> int:
        return self.token_to_id[_state_token(slot)]

    def sym_slot_of(self, tid: int) -> int | None:
        tok = self.id_to_token.get(int(tid), "")
        if tok.startswith("<SYM_") and tok.endswith(">"):
            try:
                return int(tok[5:-1])
            except ValueError:
                return None
        return None

    def bind_slot_of(self, tid: int) -> int | None:
        tok = self.id_to_token.get(int(tid), "")
        if tok.startswith("<BIND_") and tok.endswith(">"):
            try:
                return int(tok[6:-1])
            except ValueError:
                return None
        return None

    def state_slot_of(self, tid: int) -> int | None:
        tok = self.id_to_token.get(int(tid), "")
        if tok.startswith("<STATE_") and tok.endswith(">"):
            try:
                return int(tok[7:-1])
            except ValueError:
                return None
        return None

    @classmethod
    def build(
        cls,
        *,
        sym_slots: int = DEFAULT_SYM_SLOTS,
        bind_slots: int = DEFAULT_BIND_SLOTS,
        state_slots: int = DEFAULT_STATE_SLOTS,
    ) -> DSLNativeTokenizer:
        """Build the fixed corpus-independent vocabulary."""
        vocab: list[str] = []
        kinds: list[str] = []

        def _add(tok: str, kind: TokenKind) -> None:
            if tok in vocab:
                return
            vocab.append(tok)
            kinds.append(kind.value)

        for tok in SPECIAL:
            _add(tok, TokenKind.SPECIAL)

        for tok in _STRUCT_PUNCT:
            _add(tok, TokenKind.STRUCT)
        _add(NL, TokenKind.STRUCT)

        for name in _COMPONENT_NAMES:
            _add(name, TokenKind.COMPONENT)
        for name in _BUILTIN_NAMES:
            _add(name, TokenKind.BUILTIN)

        for body in _FIXED_STRING_BODIES:
            _add(_fixed_string_token(body), TokenKind.LIT)
        for b in _BOOLS:
            _add(b, TokenKind.LIT)

        _add(LIT_STR, TokenKind.LIT)
        _add(LIT_NUM, TokenKind.LIT)
        _add(LIT_END, TokenKind.LIT)

        # Printable ASCII byte fallback (plus a few common codepoints).
        for code in range(32, 127):
            _add(_byte_token(chr(code)), TokenKind.BYTE)
        for ch in ("\t",):
            _add(_byte_token(ch), TokenKind.BYTE)

        for i in range(sym_slots):
            _add(_sym_token(i), TokenKind.SYM)
        for i in range(bind_slots):
            _add(_bind_token(i), TokenKind.BIND)
        for i in range(state_slots):
            _add(_state_token(i), TokenKind.STATE)

        token_to_id = {t: i for i, t in enumerate(vocab)}
        id_to_token = {i: t for t, i in token_to_id.items()}
        id_to_kind = {i: kinds[i] for i in range(len(vocab))}
        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            id_to_kind=id_to_kind,
            version=DSL_TOKENIZER_VERSION,
            sym_slots=sym_slots,
            bind_slots=bind_slots,
            state_slots=state_slots,
        )

    # --- surface lexing / canonicalize ---------------------------------

    @staticmethod
    def lex_surface(text: str) -> list[str]:
        """Lex OpenUI source into surface pieces (no whitespace / comments)."""
        tokens: list[str] = []
        pos = 0
        for m in _LEX_RE.finditer(text):
            if m.start() != pos:
                # Keep unexpected characters as singleton pieces.
                for ch in text[pos : m.start()]:
                    if not ch.isspace():
                        tokens.append(ch)
            raw = next(g for g in m.groups() if g is not None)
            if raw.startswith(("//", "#")):
                pos = m.end()
                continue
            if raw.isspace() and "\n" not in raw:
                pos = m.end()
                continue
            if "\n" in raw:
                tokens.append(NL)
            else:
                tokens.append(raw)
            pos = m.end()
        if pos < len(text):
            for ch in text[pos:]:
                if not ch.isspace():
                    tokens.append(ch)
        # Collapse consecutive NL.
        out: list[str] = []
        for tok in tokens:
            if tok == NL and out and out[-1] == NL:
                continue
            out.append(tok)
        while out and out[-1] == NL:
            out.pop()
        return out

    def canonicalize(self, text: str, table: SymbolTable | None = None) -> str:
        """Deterministic pretty-print of OpenUI source via encode→decode."""
        return self.decode(
            self.encode(text, add_special=False, table=table, use_symbol_table=True),
            skip_special=True,
            table=table,
        )

    # --- encode / decode -----------------------------------------------

    def encode(
        self,
        text: str,
        add_special: bool = True,
        *,
        table: SymbolTable | None = None,
        use_symbol_table: bool = True,
        placeholders: Iterable[str] | None = None,
    ) -> list[int]:
        """Encode OpenUI source into lexer-native ids.

        When ``use_symbol_table`` is True (E41+), placeholders become ``<SYM_i>``
        and binders become ``<BIND_j>``. When False (E40), placeholders go through
        the typed literal / byte channel; binders are still alpha-renamed to BIND
        slots (local names carry no cross-example semantics).
        """
        table = table or SymbolTable.from_placeholders(
            placeholders, max_slots=self.sym_slots
        )
        # Seed table from explicit placeholder list first.
        if placeholders:
            for ph in placeholders:
                table.ensure_placeholder(ph, max_slots=self.sym_slots)

        pieces = self.lex_surface(text)
        ids: list[int] = []
        # First pass: discover binder definitions (NAME = ...) in order.
        for i, piece in enumerate(pieces):
            if (
                piece[:1].islower()
                and piece.isidentifier()
                and i + 1 < len(pieces)
                and pieces[i + 1] == "="
            ):
                table.ensure_binder(piece, max_slots=self.bind_slots)
            if piece.startswith("$") and i + 1 < len(pieces) and pieces[i + 1] == "=":
                table.ensure_state(piece, max_slots=self.state_slots)

        for i, piece in enumerate(pieces):
            ids.extend(
                self._encode_piece(
                    piece,
                    table=table,
                    use_symbol_table=use_symbol_table,
                    preserve_identifier=(
                        (i + 1 < len(pieces) and pieces[i + 1] == ":")
                        or (i > 0 and pieces[i - 1] == ".")
                    ),
                )
            )

        if add_special:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def _encode_piece(
        self,
        piece: str,
        *,
        table: SymbolTable,
        use_symbol_table: bool,
        preserve_identifier: bool = False,
    ) -> list[int]:
        # Structural punctuation / NL
        if piece == NL:
            return [self.token_to_id[NL]]
        if piece in self.token_to_id and self.kind_of(self.token_to_id[piece]) in {
            TokenKind.STRUCT,
            TokenKind.COMPONENT,
            TokenKind.BUILTIN,
        }:
            return [self.token_to_id[piece]]
        if piece in _BOOLS:
            return [self.token_to_id[piece]]

        # String literal
        if len(piece) >= 2 and piece[0] in {'"', "'"} and piece[-1] == piece[0]:
            return self._encode_string_literal(
                piece, table=table, use_symbol_table=use_symbol_table
            )

        if piece.startswith("$"):
            slot = table.ensure_state(piece, max_slots=self.state_slots)
            if slot is not None:
                return [self.state_id(slot)]
            self.overflow_count += 1
            return self._encode_bytes(piece)

        # Component already handled; NAME (binder / ref)
        if piece[:1].islower() and piece.isidentifier():
            if preserve_identifier:
                return self._encode_bytes(piece)
            slot = table.ensure_binder(piece, max_slots=self.bind_slots)
            if slot is None:
                self.overflow_count += 1
                return self._encode_bytes(piece)
            return [self.bind_id(slot)]

        # Number
        if _NUMBER_RE.fullmatch(piece or ""):
            return self._encode_number(piece)

        # Fallback: bytes
        self.overflow_count += 1
        return self._encode_bytes(piece)

    def _encode_string_literal(
        self,
        quoted: str,
        *,
        table: SymbolTable,
        use_symbol_table: bool,
    ) -> list[int]:
        body = ast.literal_eval(quoted)
        if not isinstance(body, str):
            return self._encode_bytes(quoted)
        if _PLACEHOLDER_BODY_RE.fullmatch(body) and use_symbol_table:
            ph = body
            slot = table.ensure_placeholder(ph, max_slots=self.sym_slots)
            if slot is not None:
                return [self.sym_id(slot)]
            self.overflow_count += 1

        fixed = _fixed_string_token(body)
        if fixed in self.token_to_id:
            return [self.token_to_id[fixed]]

        # Typed literal channel: LIT_STR + bytes + LIT_END
        return [
            self.token_to_id[LIT_STR],
            *self._encode_bytes(body),
            self.token_to_id[LIT_END],
        ]

    def _encode_number(self, text: str) -> list[int]:
        return [self.token_to_id[LIT_NUM], *self._encode_bytes(text), self.token_to_id[LIT_END]]

    def _encode_bytes(self, text: str) -> list[int]:
        ids: list[int] = []
        for ch in text:
            tok = _byte_token(ch)
            if tok in self.token_to_id:
                ids.append(self.token_to_id[tok])
            else:
                # Non-ASCII: emit UNK bytes marker as '?' substitute.
                ids.append(self.token_to_id.get(_byte_token("?"), self.unk_id))
        return ids

    def decode(
        self,
        ids: list[int],
        skip_special: bool = True,
        *,
        table: SymbolTable | None = None,
    ) -> str:
        """Pretty-print lexer-native ids back to OpenUI source."""
        table = table or SymbolTable()
        special = {self.pad_id, self.bos_id, self.eos_id, self.mask_id}
        pieces: list[str] = []
        i = 0
        n = len(ids)
        while i < n:
            tid = int(ids[i])
            if skip_special and tid in special:
                i += 1
                continue
            tok = self.id_to_token.get(tid, UNK)
            kind = self.kind_of(tid)

            if kind == TokenKind.SYM:
                slot = self.sym_slot_of(tid)
                ph = table.placeholder_at(slot) if slot is not None else None
                if ph is None:
                    ph = f":sym{slot if slot is not None else 0}"
                pieces.append(json.dumps(ph, ensure_ascii=False))
                i += 1
                continue

            if kind == TokenKind.BIND:
                slot = self.bind_slot_of(tid)
                # Slot zero is reserved for the required OpenUI root binder.
                pieces.append("root" if (slot or 0) == 0 else _bind_surface_name(slot or 0))
                i += 1
                continue

            if kind == TokenKind.STATE:
                slot = self.state_slot_of(tid)
                pieces.append(_state_surface_name(slot or 0))
                i += 1
                continue

            if tok == NL:
                pieces.append("\n")
                i += 1
                continue

            if tok.startswith("STR:"):
                pieces.append(json.dumps(tok[4:], ensure_ascii=False))
                i += 1
                continue

            if tok == LIT_STR:
                body, i = self._consume_literal_body(ids, i + 1)
                pieces.append(json.dumps(body, ensure_ascii=False))
                continue

            if tok == LIT_NUM:
                body, i = self._consume_literal_body(ids, i + 1)
                pieces.append(body)
                continue

            if kind == TokenKind.BYTE:
                # Preserve an identifier/key/member as one surface piece rather
                # than spacing each byte independently in the pretty-printer.
                chars: list[str] = []
                while i < n:
                    raw = self.id_to_token.get(int(ids[i]), UNK)
                    if self.kind_of(int(ids[i])) != TokenKind.BYTE:
                        break
                    chars.append(
                        chr(int(raw[2:], 16)) if raw.startswith(_BYTE_PREFIX) else raw
                    )
                    i += 1
                pieces.append("".join(chars))
                continue

            if tok == LIT_END:
                i += 1
                continue

            pieces.append(tok)
            i += 1

        return self._pretty_print(pieces)

    def _consume_literal_body(self, ids: list[int], start: int) -> tuple[str, int]:
        chars: list[str] = []
        i = start
        while i < len(ids):
            tid = int(ids[i])
            if tid in {self.pad_id, self.bos_id, self.eos_id, self.mask_id}:
                i += 1
                continue
            tok = self.id_to_token.get(tid, UNK)
            if tok == LIT_END:
                return "".join(chars), i + 1
            if tok.startswith(_BYTE_PREFIX):
                chars.append(chr(int(tok[2:], 16)))
                i += 1
                continue
            # Stop if we hit a non-byte framed token (robustness).
            break
        return "".join(chars), i

    @staticmethod
    def _pretty_print(pieces: list[str]) -> str:
        """Join pieces with OpenUI-like spacing."""
        if not pieces:
            return ""
        out: list[str] = []
        for idx, tok in enumerate(pieces):
            if tok == "\n":
                out.append("\n")
                continue
            prev = out[-1] if out else ""
            if not out or prev == "\n":
                out.append(tok)
                continue
            # Spaces around '=' and after ',' for readability / fixture parity.
            if tok == "=":
                out.append(" = ")
                continue
            if prev.rstrip().endswith("="):
                # Already handled by " = "
                pass
            if tok == ",":
                out.append(", ")
                continue
            if prev.endswith(", "):
                out.append(tok)
                continue
            if tok in {")", "]", "}"}:
                out.append(tok)
                continue
            if prev.endswith("(") or prev.endswith("[") or prev.endswith("{"):
                out.append(tok)
                continue
            if tok in {"(", "[", "{"}:
                out.append(tok)
                continue
            if tok == ".":
                out.append(tok)
                continue
            if prev.endswith("."):
                out.append(tok)
                continue
            if tok == ":":
                out.append(": ")
                continue
            if tok in {"+", "-", "*", "/", "%", "==", "!=", ">", "<", ">=", "<=", "&&", "||", "?"}:
                out.append(f" {tok} ")
                continue
            if tok == "!":
                out.append(tok)
                continue
            # Default: no space (identifiers glued only when intended).
            # Insert space before a NAME/COMPONENT after a prior NAME/literal.
            if tok[:1].isalnum() or tok[:1] in {'"', "'"}:
                if prev[-1:].isalnum() or prev.endswith('"'):
                    out.append(" ")
            out.append(tok)
        text = "".join(out)
        # Normalize spaces introduced by " = " near newlines.
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n+$", "", text)
        return text

    def statement_spans(self, ids: list[int]) -> list[tuple[int, int]]:
        """Return half-open [start, end) spans for each statement (NL-delimited)."""
        spans: list[tuple[int, int]] = []
        start = 0
        # Skip leading BOS.
        if ids and ids[0] == self.bos_id:
            start = 1
        i = start
        n = len(ids)
        nl_id = self.token_to_id[NL]
        special = {self.pad_id, self.eos_id, self.mask_id}
        while i < n:
            if ids[i] in special:
                if i > start:
                    spans.append((start, i))
                break
            if ids[i] == nl_id:
                if i > start:
                    spans.append((start, i))
                start = i + 1
            i += 1
        else:
            if start < n and n > start:
                # Trailing statement without NL.
                end = n
                if ids and ids[-1] == self.eos_id:
                    end = n - 1
                if end > start:
                    spans.append((start, end))
        return spans

    def spanning_statement(self, ids: list[int], index: int) -> tuple[int, int] | None:
        for lo, hi in self.statement_spans(ids):
            if lo <= index < hi:
                return lo, hi
        return None

    # --- persistence ---------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "kind": "dsl_native",
                    "version": self.version,
                    "sym_slots": self.sym_slots,
                    "bind_slots": self.bind_slots,
                    "state_slots": self.state_slots,
                    "token_to_id": self.token_to_id,
                    "id_to_kind": {str(k): v for k, v in self.id_to_kind.items()},
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> DSLNativeTokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("kind") not in {None, "dsl_native"}:
            # Accept missing kind for forward compat only when version present.
            if "id_to_kind" not in data:
                raise ValueError(f"not a dsl_native tokenizer: {path}")
        token_to_id = {str(k): int(v) for k, v in data["token_to_id"].items()}
        id_to_token = {i: t for t, i in token_to_id.items()}
        raw_kinds = data.get("id_to_kind") or {}
        id_to_kind = {int(k): str(v) for k, v in raw_kinds.items()}
        # Backfill kinds if missing.
        if not id_to_kind:
            rebuilt = cls.build(
                sym_slots=int(data.get("sym_slots") or DEFAULT_SYM_SLOTS),
                bind_slots=int(data.get("bind_slots") or DEFAULT_BIND_SLOTS),
                state_slots=int(data.get("state_slots") or DEFAULT_STATE_SLOTS),
            )
            id_to_kind = {
                i: rebuilt.id_to_kind.get(i, TokenKind.SPECIAL.value)
                for i in id_to_token
            }
        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            id_to_kind=id_to_kind,
            version=int(data.get("version") or DSL_TOKENIZER_VERSION),
            sym_slots=int(data.get("sym_slots") or DEFAULT_SYM_SLOTS),
            bind_slots=int(data.get("bind_slots") or DEFAULT_BIND_SLOTS),
            state_slots=int(data.get("state_slots") or DEFAULT_STATE_SLOTS),
        )


def is_dsl_native_tokenizer(obj: object) -> bool:
    return isinstance(obj, DSLNativeTokenizer)
