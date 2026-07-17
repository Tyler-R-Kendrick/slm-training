"""Choice-codec OpenUI output tokenizer (B1 / SLM-42).

Wraps ``dsl.production_codec.encode_choices``/``decode_choices`` as a third
output tokenizer beside :class:`~slm_training.models.tokenizer.OpenUITokenizer`
(compositional) and :class:`~slm_training.models.dsl_tokenizer.DSLNativeTokenizer`
(lexer-native). The model predicts ONLY semantic decisions (which production,
which slot filler); all non-lexical surface syntax is reconstructed by the
deterministic detokenizer through the official lang-core serializer.

Vocabulary policy — grammar-closed, corpus-independent: the vocabulary is
built deterministically from the component library (``openui_prop_order.json``),
the builtin/operator alphabets, and fixed slot/ref/state pools. Tokens outside
that closure (unseen components, free-form object keys / member names, open
literals beyond the fixed pool) fail closed to ``<unk>`` at encode time and to
an empty decode (which the parse gate rejects). This trades open-vocabulary
coverage for a stable loss space; see
``docs/design/iter-b1-choice-sequence-codec-20260717.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from slm_training.dsl.lang_core import ParseError
from slm_training.dsl.production_codec import (
    BUILTIN_PREFIX,
    CHOICE_STMT_MARKERS,
    CLOSE,
    DIR_PREFIX,
    INDEX_OP,
    LIST_CLOSE,
    LIST_OPEN,
    LIT_PREFIX,
    MEMBER_PREFIX,
    NAME_PREFIX,
    NEG_OP,
    NOT_OP,
    OBJ_CLOSE,
    OBJ_OPEN,
    OPEN_PREFIX,
    OP_PREFIX,
    REF_PREFIX,
    SLOT_PREFIX,
    STATE_REF_PREFIX,
    TERNARY_OP,
    _prop_order,
    decode_choices,
    encode_choices,
)

# Bump when vocab layout / serialization changes.
CHOICE_TOKENIZER_VERSION = 1
CHOICE_TOKENIZER_KIND = "choice_codec"

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
MASK = "<mask>"
UNK = "<unk>"
SPECIAL = (PAD, BOS, EOS, MASK, UNK)

# Framed open-vocabulary channels (byte-spelled; grammar-closed alphabet).
# A literal's / key's CONTENT is a genuine semantic decision, so free-form
# strings, numbers, object keys, and member names outside the fixed pools are
# byte-spelled instead of collapsing to <unk>. Components stay fail-closed.
LIT_STR = "LIT_STR"
LIT_NUM = "LIT_NUM"
NAME_STR = "NAME_STR"
MEMBER_STR = "MEMBER_STR"
LIT_END = "LIT_END"
_BYTE_PREFIX = "B:"


def _byte_token(ch: str) -> str:
    return f"{_BYTE_PREFIX}{ord(ch):02x}"

DEFAULT_SYM_SLOTS = 64
DEFAULT_REF_SLOTS = 64
DEFAULT_STATE_SLOTS = 64
DEFAULT_MAX_INT_LITERAL = 128

_BINARY_OPS = ("||", "&&", "==", "!=", ">=", "<=", ">", "<", "+", "-", "*", "/", "%")

# Builtin action/aggregate names (mirrors dsl_tokenizer._BUILTIN_NAMES @-set).
_BUILTIN_NAMES: tuple[str, ...] = (
    "Run",
    "Set",
    "Reset",
    "ToAssistant",
    "OpenUrl",
    "Count",
    "First",
    "Last",
    "Sum",
    "Avg",
    "Min",
    "Max",
    "Sort",
    "Filter",
    "Round",
    "Abs",
    "Floor",
    "Ceil",
    "Each",
)

# Common closed string atoms (layout / size / tone enums) — grammar-derived.
_FIXED_STRING_BODIES: tuple[str, ...] = (
    "",
    "2xl",
    "all",
    "column",
    "date",
    "default",
    "email",
    "info",
    "l",
    "large",
    "large-heavy",
    "m",
    "none",
    "number",
    "password",
    "primary",
    "row",
    "s",
    "secondary",
    "small",
    "small-heavy",
    "tertiary",
    "text",
    "warning",
    "xl",
    "xs",
)


def _grammar_names() -> tuple[str, ...]:
    """Closed identifier pool for object keys / member names (grammar-derived)."""
    names: set[str] = set()
    for props in _prop_order().values():
        names.update(props)
    return tuple(sorted(names))


@dataclass
class ChoiceTokenizer:
    """Fixed grammar-closed vocabulary over the pure choice stream."""

    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)
    id_to_kind: dict[int, str] = field(default_factory=dict)
    version: int = CHOICE_TOKENIZER_VERSION
    sym_slots: int = DEFAULT_SYM_SLOTS
    ref_slots: int = DEFAULT_REF_SLOTS
    state_slots: int = DEFAULT_STATE_SLOTS
    max_int_literal: int = DEFAULT_MAX_INT_LITERAL
    # Fail-closed <unk> encodings observed (unseen component / literal / key).
    overflow_count: int = 0

    # --- special ids -----------------------------------------------------

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

    # --- kind helpers (parity with DSLNativeTokenizer) --------------------

    def kind_of(self, token_id: int) -> str:
        return self.id_to_kind.get(int(token_id), "special")

    def kind_ids(self, kind: object) -> set[int]:
        want = getattr(kind, "value", kind)
        return {i for i, k in self.id_to_kind.items() if k == str(want)}

    def sym_id(self, slot: int) -> int:
        return self.token_to_id[f"{SLOT_PREFIX}{slot}"]

    # --- build -----------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        sym_slots: int = DEFAULT_SYM_SLOTS,
        ref_slots: int = DEFAULT_REF_SLOTS,
        state_slots: int = DEFAULT_STATE_SLOTS,
        max_int_literal: int = DEFAULT_MAX_INT_LITERAL,
    ) -> ChoiceTokenizer:
        """Deterministic, corpus-independent vocabulary from the grammar."""
        vocab: list[str] = []
        kinds: list[str] = []

        def _add(tok: str, kind: str) -> None:
            if tok in vocab:
                return
            vocab.append(tok)
            kinds.append(kind)

        for tok in SPECIAL:
            _add(tok, "special")

        # Statement-production choices (v0.5) + arity/shape choices.
        for marker in CHOICE_STMT_MARKERS:
            _add(marker, "struct")
        for tok in (CLOSE, LIST_OPEN, LIST_CLOSE, OBJ_OPEN, OBJ_CLOSE):
            _add(tok, "struct")

        # Operator choices (which operator applies IS a decision).
        for tok in (TERNARY_OP, NOT_OP, NEG_OP, INDEX_OP):
            _add(tok, "builtin")
        for op in _BINARY_OPS:
            _add(f"{OP_PREFIX}{op}", "builtin")

        # Component productions from the library + runtime statement heads.
        components = sorted(set(_prop_order()) | {"Query", "Mutation", "Action"})
        for name in components:
            _add(f"{OPEN_PREFIX}{name}", "component")
        for name in _BUILTIN_NAMES:
            _add(f"{BUILTIN_PREFIX}{name}", "builtin")

        # Direction enum fillers.
        for direction in ("column", "row"):
            _add(f"{DIR_PREFIX}{direction}", "lit")

        # Fixed literal pool: keywords, common string atoms, small ints.
        for kw in ("null", "true", "false"):
            _add(f"{LIT_PREFIX}{kw}", "lit")
        for body in _FIXED_STRING_BODIES:
            _add(f"{LIT_PREFIX}{json.dumps(body)}", "lit")
        for value in range(0, max_int_literal + 1):
            _add(f"{LIT_PREFIX}{value}", "lit")

        # Closed identifier pool (prop names) for object keys / members.
        for name in _grammar_names():
            _add(f"{NAME_PREFIX}{name}", "lit")
            _add(f"{MEMBER_PREFIX}{name}", "lit")

        # Framed open-vocabulary channels + printable-ASCII byte alphabet.
        for marker in (LIT_STR, LIT_NUM, NAME_STR, MEMBER_STR, LIT_END):
            _add(marker, "lit")
        for code in range(32, 127):
            _add(_byte_token(chr(code)), "byte")

        # Slot / reference / state pointer pools.
        for i in range(sym_slots):
            _add(f"{SLOT_PREFIX}{i}", "sym")
        for i in range(ref_slots):
            _add(f"{REF_PREFIX}{i}", "bind")
        for i in range(state_slots):
            _add(f"{STATE_REF_PREFIX}{i}", "state")

        token_to_id = {t: i for i, t in enumerate(vocab)}
        return cls(
            token_to_id=token_to_id,
            id_to_token={i: t for t, i in token_to_id.items()},
            id_to_kind={i: kinds[i] for i in range(len(vocab))},
            version=CHOICE_TOKENIZER_VERSION,
            sym_slots=sym_slots,
            ref_slots=ref_slots,
            state_slots=state_slots,
            max_int_literal=max_int_literal,
        )

    # --- encode / decode ---------------------------------------------------

    def _contract_from(
        self,
        table: object | None,
        placeholders: Iterable[str] | None,
    ) -> list[str] | None:
        if placeholders is not None:
            return list(placeholders)
        table_placeholders = getattr(table, "placeholders", None)
        if table_placeholders:
            return list(table_placeholders)
        return None

    def encode(
        self,
        text: str,
        add_special: bool = True,
        *,
        table: object | None = None,
        use_symbol_table: bool = True,  # noqa: ARG002 - interface parity
        placeholders: Iterable[str] | None = None,
    ) -> list[int]:
        """Encode OpenUI source into choice-stream ids (fail-closed to <unk>).

        ``table`` may be a :class:`~slm_training.models.dsl_tokenizer.SymbolTable`
        — only its placeholder inventory is used (choice refs/states are
        positional, not named). Placeholders discovered during encoding are
        appended to the table so decode can share the same contract.
        """
        declared = self._contract_from(table, placeholders)
        program = encode_choices(text, slot_contract=declared)
        ensure = getattr(table, "ensure_placeholder", None)
        if callable(ensure):
            for placeholder in program.slot_contract:
                ensure(placeholder, max_slots=self.sym_slots)
        ids: list[int] = []
        for token in program.tokens:
            token_id = self.token_to_id.get(token)
            if token_id is not None:
                ids.append(token_id)
                continue
            framed = self._frame_open_token(token)
            if framed is None:
                self.overflow_count += 1
                ids.append(self.unk_id)
            else:
                ids.extend(framed)
        if add_special:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def _frame_open_token(self, token: str) -> list[int] | None:
        """Byte-spell an open-vocabulary codec token; None = fail closed."""
        if token.startswith(LIT_PREFIX):
            payload = token[len(LIT_PREFIX) :]
            if payload.startswith('"'):
                try:
                    body = json.loads(payload)
                except json.JSONDecodeError:
                    return None
                return self._frame(LIT_STR, str(body))
            return self._frame(LIT_NUM, payload)
        if token.startswith(NAME_PREFIX):
            return self._frame(NAME_STR, token[len(NAME_PREFIX) :])
        if token.startswith(MEMBER_PREFIX):
            return self._frame(MEMBER_STR, token[len(MEMBER_PREFIX) :])
        return None

    def _frame(self, marker: str, body: str) -> list[int] | None:
        ids = [self.token_to_id[marker]]
        for ch in body:
            byte_id = self.token_to_id.get(_byte_token(ch))
            if byte_id is None:
                return None  # outside the closed byte alphabet — fail closed
            ids.append(byte_id)
        ids.append(self.token_to_id[LIT_END])
        return ids

    def decode(
        self,
        ids: list[int],
        skip_special: bool = True,
        *,
        table: object | None = None,
        placeholders: Iterable[str] | None = None,
    ) -> str:
        """Deterministic detokenize through the lang-core serializer.

        Fail closed: any stream that does not reconstruct to valid canonical
        OpenUI (including one containing ``<unk>``/``<mask>``) decodes to ``""``
        so the parse gate rejects it — the detokenizer never invents syntax.
        """
        contract = self._contract_from(table, placeholders) or []
        skip = {self.pad_id, self.bos_id}
        frame_markers = {
            LIT_STR: lambda body: f"{LIT_PREFIX}{json.dumps(body)}",
            LIT_NUM: lambda body: f"{LIT_PREFIX}{body}",
            NAME_STR: lambda body: f"{NAME_PREFIX}{body}",
            MEMBER_STR: lambda body: f"{MEMBER_PREFIX}{body}",
        }
        tokens: list[str] = []
        i = 0
        n = len(ids)
        while i < n:
            token_id = int(ids[i])
            if skip_special and token_id in skip:
                i += 1
                continue
            if token_id == self.eos_id:
                break
            if token_id == self.mask_id or token_id == self.unk_id:
                return ""
            token = self.id_to_token.get(token_id, UNK)
            build = frame_markers.get(token)
            if build is not None:
                chars: list[str] = []
                i += 1
                while i < n:
                    raw = self.id_to_token.get(int(ids[i]), UNK)
                    if raw == LIT_END:
                        i += 1
                        break
                    if not raw.startswith(_BYTE_PREFIX):
                        return ""  # malformed frame — fail closed
                    chars.append(chr(int(raw[len(_BYTE_PREFIX) :], 16)))
                    i += 1
                else:
                    return ""  # unterminated frame — fail closed
                tokens.append(build("".join(chars)))
                continue
            tokens.append(token)
            i += 1
        if not tokens:
            return ""
        try:
            return decode_choices(tokens, contract)
        except ParseError:
            return ""

    # --- persistence -------------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "kind": CHOICE_TOKENIZER_KIND,
                    "version": self.version,
                    "sym_slots": self.sym_slots,
                    "ref_slots": self.ref_slots,
                    "state_slots": self.state_slots,
                    "max_int_literal": self.max_int_literal,
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
    def load(cls, path: Path | str) -> ChoiceTokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("kind") != CHOICE_TOKENIZER_KIND:
            raise ValueError(f"not a {CHOICE_TOKENIZER_KIND} tokenizer: {path}")
        token_to_id = {str(k): int(v) for k, v in data["token_to_id"].items()}
        return cls(
            token_to_id=token_to_id,
            id_to_token={i: t for t, i in token_to_id.items()},
            id_to_kind={
                int(k): str(v) for k, v in (data.get("id_to_kind") or {}).items()
            },
            version=int(data.get("version") or CHOICE_TOKENIZER_VERSION),
            sym_slots=int(data.get("sym_slots") or DEFAULT_SYM_SLOTS),
            ref_slots=int(data.get("ref_slots") or DEFAULT_REF_SLOTS),
            state_slots=int(data.get("state_slots") or DEFAULT_STATE_SLOTS),
            max_int_literal=int(
                data.get("max_int_literal") or DEFAULT_MAX_INT_LITERAL
            ),
        )


def is_choice_tokenizer(obj: object) -> bool:
    return isinstance(obj, ChoiceTokenizer)


__all__ = [
    "CHOICE_TOKENIZER_KIND",
    "CHOICE_TOKENIZER_VERSION",
    "ChoiceTokenizer",
    "is_choice_tokenizer",
]
