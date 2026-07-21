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
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

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

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from slm_training.dsl.grammar.fastpath.compiler_draft import ConstraintEvidence

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
_PLACEHOLDER_SCHEMA_KEY = "x-openui-placeholder"


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


@lru_cache(maxsize=1)
def _component_contracts() -> dict[str, tuple[tuple[dict[str, Any], ...], int]]:
    """Positional component contracts from the pinned OpenUI JSON schema."""
    from slm_training.dsl.lang_core import library_schema
    from slm_training.dsl.placeholders import CONTENT_PROPS

    schema = library_schema()
    definitions = dict(schema.get("$defs") or {})
    contracts: dict[str, tuple[tuple[dict[str, Any], ...], int]] = {}
    for component, props in _prop_order().items():
        definition = dict(definitions.get(component) or {})
        properties = dict(definition.get("properties") or {})
        positional = []
        for name in props:
            prop_schema = dict(properties.get(name) or {})
            if name in CONTENT_PROPS:
                prop_schema[_PLACEHOLDER_SCHEMA_KEY] = True
            positional.append(prop_schema)
        required = set(definition.get("required") or ())
        required_args = max(
            (index + 1 for index, name in enumerate(props) if name in required),
            default=0,
        )
        contracts[component] = (tuple(positional), required_args)
    return contracts


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
    allowed_cache: dict[tuple[object, ...], frozenset[int]] = field(
        default_factory=dict, repr=False, compare=False
    )
    allowed_cache_hits: int = field(default=0, repr=False, compare=False)
    allowed_cache_misses: int = field(default=0, repr=False, compare=False)
    candidate_partitions: dict[str, frozenset[int]] = field(
        default_factory=dict, repr=False, compare=False
    )
    expression_candidate_cache: dict[tuple[int, int], frozenset[int]] = field(
        default_factory=dict, repr=False, compare=False
    )
    completion_cache: dict[tuple[object, ...], int] = field(
        default_factory=dict, repr=False, compare=False
    )
    candidates_considered: int = field(default=0, repr=False, compare=False)
    vocab_candidates_avoided: int = field(default=0, repr=False, compare=False)
    completion_cache_hits: int = field(default=0, repr=False, compare=False)
    completion_cache_misses: int = field(default=0, repr=False, compare=False)

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

    def candidate_partition(self, name: str) -> frozenset[int]:
        """Return a lazily built production-category token partition."""
        cached = self.candidate_partitions.get(name)
        if cached is not None:
            return cached

        def _is_expression_start(token: str) -> bool:
            return (
                token.startswith(
                    (
                        OPEN_PREFIX,
                        BUILTIN_PREFIX,
                        OP_PREFIX,
                        MEMBER_PREFIX,
                        REF_PREFIX,
                        SLOT_PREFIX,
                        STATE_REF_PREFIX,
                        DIR_PREFIX,
                        LIT_PREFIX,
                    )
                )
                or token
                in {
                    LIST_OPEN,
                    OBJ_OPEN,
                    TERNARY_OP,
                    NOT_OP,
                    NEG_OP,
                    INDEX_OP,
                    LIT_STR,
                    LIT_NUM,
                    MEMBER_STR,
                }
            )

        predicates = {
            "byte": lambda token: token.startswith(_BYTE_PREFIX),
            "bind": lambda token: token.startswith(REF_PREFIX),
            "expression": _is_expression_start,
            "marker": lambda token: token in CHOICE_STMT_MARKERS,
            "object_key": lambda token: token.startswith(NAME_PREFIX)
            or token == NAME_STR,
            "slot": lambda token: token.startswith(SLOT_PREFIX),
        }
        predicate = predicates.get(name)
        if predicate is None:
            raise ValueError(f"unknown choice candidate partition: {name}")
        result = frozenset(
            token_id
            for token_id, token in self.id_to_token.items()
            if predicate(token)
        )
        self.candidate_partitions[name] = result
        return result

    def expression_candidates(
        self, *, slot_count: int, available_ref_count: int
    ) -> frozenset[int]:
        """Exact reusable expression-start superset for request-local counts."""
        key = (
            min(max(int(slot_count), 0), self.sym_slots),
            min(max(int(available_ref_count), 0), self.ref_slots),
        )
        cached = self.expression_candidate_cache.get(key)
        if cached is not None:
            return cached
        candidates = set(self.candidate_partition("expression"))
        candidates.difference_update(self.candidate_partition("bind"))
        candidates.update(
            self.token_to_id[f"{REF_PREFIX}{index}"] for index in range(key[1])
        )
        candidates.difference_update(self.candidate_partition("slot"))
        candidates.update(
            self.token_to_id[f"{SLOT_PREFIX}{index}"] for index in range(key[0])
        )
        result = frozenset(candidates)
        self.expression_candidate_cache[key] = result
        return result

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


@dataclass
class _ChoiceFrame:
    kind: str
    expr_type: str
    close: str | None = None
    remaining: int = 0
    phase: str = ""
    schemas: tuple[dict[str, Any], ...] = ()
    required_args: int = 0
    arg_index: int = 0
    reference_count: int = 0
    item_count: int = 0
    property_names: tuple[str, ...] = ()
    required_properties: tuple[str, ...] = ()
    seen_properties: tuple[str, ...] = ()
    active_property: str | None = None
    additional_properties: bool = True


@dataclass
class ChoiceDecodeState:
    """Grammar-derived pushdown state for choice-codec generation."""

    tokenizer: ChoiceTokenizer
    slot_count: int = 0
    mode: str | None = None
    frames: list[_ChoiceFrame] = field(default_factory=list)
    section_types: list[str] = field(default_factory=list)
    current_marker: str | None = None
    valid_root_seen: bool = False
    literal_frame: str | None = None
    literal_size: int = 0
    literal_is_object_key: bool = False

    def clone(self) -> ChoiceDecodeState:
        return ChoiceDecodeState(
            tokenizer=self.tokenizer,
            slot_count=self.slot_count,
            mode=self.mode,
            frames=deepcopy(self.frames),
            section_types=list(self.section_types),
            current_marker=self.current_marker,
            valid_root_seen=self.valid_root_seen,
            literal_frame=self.literal_frame,
            literal_size=self.literal_size,
            literal_is_object_key=self.literal_is_object_key,
        )

    def can_end(self) -> bool:
        if self.frames or self.literal_frame is not None:
            return False
        if self.mode == "structural":
            return bool(
                self.section_types
                and self.section_types[-1].startswith("element:")
            )
        return self.mode == "v05" and self.valid_root_seen

    @staticmethod
    def _schema_accepts(schema: dict[str, Any], expr_type: str) -> bool:
        if not schema:
            return True
        if schema.get(_PLACEHOLDER_SCHEMA_KEY):
            return expr_type == "placeholder"
        if "anyOf" in schema:
            return any(
                ChoiceDecodeState._schema_accepts(dict(option), expr_type)
                for option in schema["anyOf"]
            )
        ref = str(schema.get("$ref") or "")
        if ref.startswith("#/$defs/"):
            return expr_type == f"element:{ref.rsplit('/', 1)[-1]}"
        expected = schema.get("type")
        if isinstance(expected, list):
            return any(
                ChoiceDecodeState._schema_accepts({"type": item}, expr_type)
                for item in expected
            )
        if expected == "integer":
            expected = "number"
        if expected == "string" and expr_type == "placeholder":
            return True
        return expected is None or expr_type in {str(expected), "any"}

    def _active_schema(self) -> dict[str, Any]:
        if not self.frames:
            return {}
        frame = self.frames[-1]
        if frame.kind == "component" and frame.arg_index < len(frame.schemas):
            return dict(frame.schemas[frame.arg_index])
        if frame.kind == "variadic" and frame.schemas:
            return dict(frame.schemas[0])
        if (
            frame.kind == "object"
            and frame.phase == "value"
            and 0 <= frame.arg_index < len(frame.schemas)
        ):
            return dict(frame.schemas[frame.arg_index])
        return {}

    def _complete_expr(self, expr_type: str) -> bool:
        if not self.frames:
            self.section_types.append(expr_type)
            if self.mode == "v05":
                if self.current_marker == "r=" and expr_type.startswith("element:"):
                    self.valid_root_seen = True
                self.current_marker = None
            return True
        frame = self.frames[-1]
        if frame.kind == "fixed":
            frame.remaining -= 1
            if frame.remaining == 0:
                self.frames.pop()
                return self._complete_expr(frame.expr_type)
            return True
        elif frame.kind == "object":
            if 0 <= frame.arg_index < len(frame.schemas) and not self._schema_accepts(
                frame.schemas[frame.arg_index], expr_type
            ):
                return False
            if frame.active_property is not None:
                frame.seen_properties = (
                    *frame.seen_properties,
                    frame.active_property,
                )
            frame.active_property = None
            frame.arg_index = -1
            frame.phase = "key"
            return True
        elif frame.kind == "variadic" and frame.schemas:
            item_schema = frame.schemas[0] if frame.schemas else {}
            accepted = self._schema_accepts(item_schema, expr_type)
            if accepted:
                frame.item_count += 1
            return accepted
        elif frame.kind == "variadic":
            frame.item_count += 1
            return True
        elif frame.kind == "component":
            if frame.arg_index >= len(frame.schemas):
                return False
            if not self._schema_accepts(frame.schemas[frame.arg_index], expr_type):
                return False
            frame.arg_index += 1
            return True
        return True

    def _reference_type(self, token: str) -> str | None:
        try:
            index = int(token[len(REF_PREFIX) :])
        except ValueError:
            return None
        if index < 0 or index >= len(self.section_types):
            return None
        return self.section_types[index]

    def _accept_expression_token(self, token: str) -> bool:
        if token.startswith(OPEN_PREFIX):
            component = token[len(OPEN_PREFIX) :]
            contract = _component_contracts().get(component)
            if contract is None:
                return False
            schemas, required_args = contract
            self.frames.append(
                _ChoiceFrame(
                    "component",
                    f"element:{component}",
                    close=CLOSE,
                    schemas=schemas,
                    required_args=required_args,
                    property_names=tuple(_prop_order().get(component, ())),
                )
            )
            return True
        if token.startswith(BUILTIN_PREFIX):
            self.frames.append(_ChoiceFrame("variadic", "any", close=CLOSE))
            return True
        if token == LIST_OPEN:
            item_schema: dict[str, Any] | None = None
            if self.frames:
                active_schema = self._active_schema()
                if active_schema.get("type") == "array":
                    item_schema = dict(active_schema.get("items") or {})
            if item_schema is None:
                self.frames.append(
                    _ChoiceFrame("variadic", "array", close=LIST_CLOSE)
                )
                return True
            self.frames.append(
                _ChoiceFrame(
                    "variadic",
                    "array",
                    close=LIST_CLOSE,
                    schemas=(item_schema,),
                )
            )
            return True
        if token == OBJ_OPEN:
            schema = self._active_schema()
            properties = schema.get("properties")
            property_map = properties if isinstance(properties, dict) else {}
            property_names = tuple(str(name) for name in property_map)
            required = tuple(
                str(name)
                for name in schema.get("required", ())
                if str(name) in property_map
            )
            self.frames.append(
                _ChoiceFrame(
                    "object",
                    "object",
                    close=OBJ_CLOSE,
                    phase="key",
                    schemas=tuple(
                        dict(property_map[name]) for name in property_names
                    ),
                    arg_index=-1,
                    property_names=property_names,
                    required_properties=required,
                    additional_properties=schema.get("additionalProperties")
                    is not False,
                )
            )
            return True
        fixed_arity = {
            TERNARY_OP: 3,
            NOT_OP: 1,
            NEG_OP: 1,
            INDEX_OP: 2,
        }
        if token.startswith(OP_PREFIX):
            fixed_arity[token] = 2
        if token.startswith(MEMBER_PREFIX):
            fixed_arity[token] = 1
        if token in fixed_arity:
            self.frames.append(
                _ChoiceFrame("fixed", "other", remaining=fixed_arity[token])
            )
            return True
        if token.startswith(REF_PREFIX):
            expr_type = self._reference_type(token)
            if expr_type is None:
                return False
            if (
                self.frames
                and self.frames[-1].kind == "variadic"
                and self.frames[-1].expr_type == "array"
            ):
                self.frames[-1].reference_count += 1
            return self._complete_expr(expr_type)
        if token.startswith(SLOT_PREFIX):
            try:
                if int(token[len(SLOT_PREFIX) :]) >= self.slot_count:
                    return False
            except ValueError:
                return False
            return self._complete_expr("placeholder")
        if token.startswith(STATE_REF_PREFIX):
            return self._complete_expr("any")
        if token.startswith(DIR_PREFIX):
            return self._complete_expr("string")
        if token.startswith(LIT_PREFIX):
            payload = token[len(LIT_PREFIX) :]
            if payload.startswith('"'):
                expr_type = "string"
            elif payload in {"true", "false"}:
                expr_type = "boolean"
            elif payload == "null":
                expr_type = "null"
            else:
                expr_type = "number"
            return self._complete_expr(expr_type)
        if token in {LIT_STR, LIT_NUM, MEMBER_STR}:
            self.literal_frame = token
            self.literal_size = 0
            self.literal_is_object_key = False
            return True
        return False

    def advance_id(self, token_id: int) -> bool:
        tok = self.tokenizer
        token_id = int(token_id)
        if token_id in {tok.pad_id, tok.bos_id, tok.mask_id, tok.unk_id}:
            return False
        token = tok.id_to_token.get(token_id, UNK)

        if self.literal_frame is not None:
            if token == LIT_END:
                if self.literal_frame in {LIT_NUM, NAME_STR, MEMBER_STR} and not (
                    self.literal_size
                ):
                    return False
                marker = self.literal_frame
                object_key = self.literal_is_object_key
                self.literal_frame = None
                self.literal_size = 0
                self.literal_is_object_key = False
                if object_key:
                    self.frames[-1].phase = "value"
                elif marker == MEMBER_STR:
                    self.frames.append(_ChoiceFrame("fixed", "other", remaining=1))
                else:
                    return self._complete_expr(
                        "string" if marker in {LIT_STR, NAME_STR} else "number"
                    )
                return True
            if token.startswith(_BYTE_PREFIX):
                self.literal_size += 1
                return True
            return False

        if token_id == tok.eos_id:
            return self.can_end()

        if not self.frames:
            if self.mode == "v05" and self.current_marker is None:
                if token not in CHOICE_STMT_MARKERS:
                    return False
                self.current_marker = token
                return True
            if self.mode is None and token in CHOICE_STMT_MARKERS:
                self.mode = "v05"
                self.current_marker = token
                return True
            if self.mode is None:
                self.mode = "structural"
            return self._accept_expression_token(token)

        frame = self.frames[-1]
        if frame.kind in {"variadic", "component"} and token == frame.close:
            if frame.kind == "component" and frame.arg_index < frame.required_args:
                return False
            self.frames.pop()
            return self._complete_expr(frame.expr_type)
        if frame.kind == "object" and frame.phase == "key":
            if token == frame.close:
                if not set(frame.required_properties).issubset(
                    frame.seen_properties
                ):
                    return False
                self.frames.pop()
                return self._complete_expr(frame.expr_type)
            if token.startswith(NAME_PREFIX):
                name = token[len(NAME_PREFIX) :]
                if name in frame.seen_properties:
                    return False
                if name in frame.property_names:
                    frame.arg_index = frame.property_names.index(name)
                elif not frame.additional_properties:
                    return False
                else:
                    frame.arg_index = -1
                frame.active_property = name
                frame.phase = "value"
                return True
            if token == NAME_STR:
                if not frame.additional_properties:
                    return False
                frame.arg_index = -1
                frame.active_property = None
                self.literal_frame = token
                self.literal_size = 0
                self.literal_is_object_key = True
                return True
            return False
        return self._accept_expression_token(token)

    def _first_id(self, predicate: object) -> int:
        fn = predicate
        return min(
            token_id
            for token_id, token in self.tokenizer.id_to_token.items()
            if callable(fn) and fn(token)
        )

    def _completion_id(self) -> int:
        tok = self.tokenizer
        if self.literal_frame is not None:
            if self.literal_size or self.literal_frame == LIT_STR:
                return tok.token_to_id[LIT_END]
            return tok.token_to_id[_byte_token("0")]
        if self.frames:
            frame = self.frames[-1]
            if frame.kind == "variadic":
                return tok.token_to_id[str(frame.close)]
            if frame.kind == "component":
                if frame.arg_index >= frame.required_args:
                    return tok.token_to_id[str(frame.close)]
                schema = frame.schemas[frame.arg_index]
                return self._minimal_schema_id(schema)
            if frame.kind == "object" and frame.phase == "key":
                missing = next(
                    (
                        name
                        for name in frame.required_properties
                        if name not in frame.seen_properties
                    ),
                    None,
                )
                if missing is not None:
                    return tok.token_to_id[f"{NAME_PREFIX}{missing}"]
                return tok.token_to_id[str(frame.close)]
            if (
                frame.kind == "object"
                and 0 <= frame.arg_index < len(frame.schemas)
            ):
                return self._minimal_schema_id(frame.schemas[frame.arg_index])
            return tok.token_to_id[f"{LIT_PREFIX}null"]
        if self.can_end():
            return tok.eos_id
        if self.mode == "v05" and self.current_marker is None:
            return tok.token_to_id["r="]
        return self._first_id(lambda token: token.startswith(OPEN_PREFIX))

    def _minimal_schema_id(self, schema: dict[str, Any]) -> int:
        tok = self.tokenizer
        if schema.get(_PLACEHOLDER_SCHEMA_KEY):
            return tok.token_to_id[f"{SLOT_PREFIX}0"]
        if "anyOf" in schema:
            return self._minimal_schema_id(dict(schema["anyOf"][0]))
        expected = schema.get("type")
        if isinstance(expected, list):
            expected = expected[0] if expected else None
        if expected == "array":
            return tok.token_to_id[LIST_OPEN]
        if expected == "object":
            return tok.token_to_id[OBJ_OPEN]
        if expected == "string":
            return tok.token_to_id[f'{LIT_PREFIX}""']
        if expected in {"number", "integer"}:
            return tok.token_to_id[f"{LIT_PREFIX}0"]
        if expected == "boolean":
            return tok.token_to_id[f"{LIT_PREFIX}false"]
        ref = str(schema.get("$ref") or "")
        if ref.startswith("#/$defs/"):
            component = ref.rsplit("/", 1)[-1]
            return tok.token_to_id[f"{OPEN_PREFIX}{component}"]
        return tok.token_to_id[f"{LIT_PREFIX}null"]

    def minimal_completion_length(self) -> int:
        key = self.signature()
        cached = self.tokenizer.completion_cache.get(key)
        if cached is not None:
            self.tokenizer.completion_cache_hits += 1
            return cached
        self.tokenizer.completion_cache_misses += 1
        probe = self.clone()
        result = 1025
        for count in range(1, 1025):
            token_id = probe._completion_id()
            if not probe.advance_id(token_id):
                break
            if token_id == probe.tokenizer.eos_id:
                result = count
                break
        if len(self.tokenizer.completion_cache) >= 8192:
            self.tokenizer.completion_cache.clear()
        self.tokenizer.completion_cache[key] = result
        return result

    def signature(self) -> tuple[object, ...]:
        return (
            self.slot_count,
            self.mode,
            tuple(
                (
                    frame.kind,
                    frame.expr_type,
                    frame.close,
                    frame.remaining,
                    frame.phase,
                    frame.required_args,
                    frame.arg_index,
                    frame.reference_count,
                    frame.item_count,
                    frame.property_names,
                    frame.required_properties,
                    frame.seen_properties,
                    frame.active_property,
                    frame.additional_properties,
                    tuple(
                        json.dumps(schema, sort_keys=True, separators=(",", ":"))
                        for schema in frame.schemas
                    ),
                )
                for frame in self.frames
            ),
            tuple(self.section_types),
            self.current_marker,
            self.valid_root_seen,
            self.literal_frame,
            self.literal_size,
            self.literal_is_object_key,
        )

    def _candidate_ids(self) -> set[int]:
        """Grammar-derived superset of legal next IDs for the current frame."""
        tok = self.tokenizer

        def _expression_candidates() -> set[int]:
            return set(
                tok.expression_candidates(
                    slot_count=self.slot_count,
                    available_ref_count=len(self.section_types),
                )
            )

        if self.literal_frame is not None:
            return set(tok.candidate_partition("byte")) | {
                tok.token_to_id[LIT_END]
            }

        if self.frames:
            frame = self.frames[-1]
            if frame.kind == "object" and frame.phase == "key":
                if frame.additional_properties:
                    candidates = set(tok.candidate_partition("object_key"))
                else:
                    candidates = {
                        tok.token_to_id[f"{NAME_PREFIX}{name}"]
                        for name in frame.property_names
                        if name not in frame.seen_properties
                    }
                if set(frame.required_properties).issubset(frame.seen_properties):
                    candidates.add(tok.token_to_id[str(frame.close)])
                return candidates
            candidates = _expression_candidates()
            if frame.kind in {"variadic", "component"}:
                candidates.add(tok.token_to_id[str(frame.close)])
            return candidates

        candidates: set[int] = set()
        if self.mode == "v05" and self.current_marker is None:
            candidates.update(tok.candidate_partition("marker"))
        else:
            candidates.update(_expression_candidates())
            if self.mode is None:
                candidates.update(tok.candidate_partition("marker"))
        if self.can_end():
            candidates.add(tok.eos_id)
        return candidates

    def _filter_allowed(
        self, candidate_ids: Iterable[int], remaining_positions: int
    ) -> set[int]:
        allowed: set[int] = set()
        for token_id in candidate_ids:
            probe = self.clone()
            if not probe.advance_id(token_id):
                continue
            completion = (
                0
                if token_id == self.tokenizer.eos_id
                else probe.minimal_completion_length()
            )
            if completion <= remaining_positions - 1:
                allowed.add(token_id)
        return allowed

    def exhaustive_allowed_ids(self, remaining_positions: int) -> set[int]:
        """Reference implementation used to prove direct candidates exact."""
        return self._filter_allowed(
            self.tokenizer.id_to_token, int(remaining_positions)
        )

    def allowed_ids(self, remaining_positions: int) -> set[int]:
        key = (self.signature(), int(remaining_positions))
        cached = self.tokenizer.allowed_cache.get(key)
        if cached is not None:
            self.tokenizer.allowed_cache_hits += 1
            return set(cached)
        self.tokenizer.allowed_cache_misses += 1
        candidates = self._candidate_ids()
        self.tokenizer.candidates_considered += len(candidates)
        self.tokenizer.vocab_candidates_avoided += (
            self.tokenizer.vocab_size - len(candidates)
        )
        allowed = self._filter_allowed(candidates, int(remaining_positions))
        if len(self.tokenizer.allowed_cache) >= 4096:
            self.tokenizer.allowed_cache.clear()
        self.tokenizer.allowed_cache[key] = frozenset(allowed)
        return allowed

    def allowed_ids_with_evidence(
        self, remaining_positions: int
    ) -> tuple[set[int], tuple["ConstraintEvidence", ...]]:
        """Explain-mode companion to :meth:`allowed_ids` (VSS0-02).

        Returns ``(allowed, evidence)`` where ``allowed`` is byte-for-byte the
        set :meth:`allowed_ids` would return and ``evidence`` is reason-coded
        :class:`ConstraintEvidence` for every candidate the choice frame
        actually considered. Purely observational: it recomputes from
        ``_candidate_ids``/``_filter_allowed`` and deliberately does not touch
        the ``allowed_cache`` or the ``candidates_considered`` counters that the
        default hot path maintains. Evidence over considered candidates is not an
        exhaustive support proof (see ``verified-scope-solver.md``).
        """
        from slm_training.dsl.grammar.fastpath.compiler_draft import (
            ConstraintEvidence,
            ConstraintStage,
        )

        remaining = int(remaining_positions)
        considered = self._candidate_ids()
        allowed = self._filter_allowed(considered, remaining)
        evidence: list[ConstraintEvidence] = []
        for token_id in sorted(considered):
            token = int(token_id)
            if token in allowed:
                evidence.append(
                    ConstraintEvidence(
                        token, (token,), ConstraintStage.GRAMMAR, True, "choice_admitted"
                    )
                )
                continue
            probe = self.clone()
            # Not in ``allowed`` ⇒ either the frame rejected the edge or the
            # token cannot be completed within the remaining budget.
            if not probe.advance_id(token):
                reason = "choice_advance_rejected"
            else:
                reason = "choice_length_infeasible"
            evidence.append(
                ConstraintEvidence(
                    token, (token,), ConstraintStage.GRAMMAR, False, reason
                )
            )
        return allowed, tuple(evidence)


def is_choice_tokenizer(obj: object) -> bool:
    return isinstance(obj, ChoiceTokenizer)


def structural_root_reference_arity(
    tokenizer: ChoiceTokenizer, token_ids: Iterable[int], *, slot_count: int = 0
) -> int | None:
    """Return direct reference count in the final generated structural root list."""
    target = structural_root_reference_arity_target(
        tokenizer, token_ids, slot_count=slot_count
    )
    return target[0] if target is not None else None


def structural_root_reference_arity_target(
    tokenizer: ChoiceTokenizer, token_ids: Iterable[int], *, slot_count: int = 0
) -> tuple[int, int] | None:
    """Return root-reference count plus the available generated-section bound."""
    state = ChoiceDecodeState(tokenizer, slot_count=slot_count)
    completed: int | None = None
    completed_bound: int | None = None
    list_counts: dict[int, int] = {}
    for raw_token_id in token_ids:
        token_id = int(raw_token_id)
        if token_id in {tokenizer.pad_id, tokenizer.bos_id}:
            continue
        token = str(tokenizer.id_to_token.get(token_id, ""))
        active_list = bool(
            state.mode == "structural"
            and state.frames
            and state.frames[-1].kind == "variadic"
            and state.frames[-1].expr_type == "array"
        )
        depth = len(state.frames)
        if active_list and token.startswith(REF_PREFIX):
            list_counts[depth] = list_counts.get(depth, 0) + 1
        elif active_list and token == LIST_CLOSE:
            completed = list_counts.pop(depth, 0)
            completed_bound = len(state.section_types)
        if token_id == tokenizer.eos_id:
            break
        if not state.advance_id(token_id):
            return None
        if token == LIST_OPEN:
            list_counts[len(state.frames)] = 0
    if completed is None or completed_bound is None:
        return None
    return completed, completed_bound


def structural_root_reference_identity_target(
    tokenizer: ChoiceTokenizer, token_ids: Iterable[int], *, slot_count: int = 0
) -> tuple[frozenset[int], int] | None:
    """Return referenced section indices plus the generated-section bound."""
    state = ChoiceDecodeState(tokenizer, slot_count=slot_count)
    completed: frozenset[int] | None = None
    completed_bound: int | None = None
    list_references: dict[int, set[int]] = {}
    for raw_token_id in token_ids:
        token_id = int(raw_token_id)
        if token_id in {tokenizer.pad_id, tokenizer.bos_id}:
            continue
        token = str(tokenizer.id_to_token.get(token_id, ""))
        active_list = bool(
            state.mode == "structural"
            and state.frames
            and state.frames[-1].kind == "variadic"
            and state.frames[-1].expr_type == "array"
        )
        depth = len(state.frames)
        if active_list and token.startswith(REF_PREFIX):
            try:
                list_references.setdefault(depth, set()).add(
                    int(token[len(REF_PREFIX) :])
                )
            except ValueError:
                return None
        elif active_list and token == LIST_CLOSE:
            completed = frozenset(list_references.pop(depth, set()))
            completed_bound = len(state.section_types)
        if token_id == tokenizer.eos_id:
            break
        if not state.advance_id(token_id):
            return None
        if token == LIST_OPEN:
            list_references[len(state.frames)] = set()
    if completed is None or completed_bound is None:
        return None
    return completed, completed_bound


__all__ = [
    "CHOICE_TOKENIZER_KIND",
    "CHOICE_TOKENIZER_VERSION",
    "ChoiceTokenizer",
    "ChoiceDecodeState",
    "is_choice_tokenizer",
    "structural_root_reference_arity",
    "structural_root_reference_arity_target",
    "structural_root_reference_identity_target",
]
