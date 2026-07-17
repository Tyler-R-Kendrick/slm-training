"""B1 training half (SLM-42/SLM-23): choice-stream output tokenizer.

Targets are the choice-sequence codec's decision tokens — canonical-space by
construction (B2/SLM-22 alignment): ``encode`` derives targets through
``encode_choices`` (canonical statement order, De Bruijn binder pool, style
literals stripped), and ``decode`` reconstructs OpenUI through the
deterministic detokenizer + official serializer. The model never predicts a
token the decoder could have emitted itself.

Duck-types the ``OpenUITokenizer`` surface (`encode(text, add_special)`,
`decode(ids, skip_special)`, special ids, vocab maps) so the TwoTower
compositional code path runs unchanged; grammar-constrained decode is NOT
supported for choice targets yet (the OpenUI DFA speaks surface tokens), so
fixture rows run unconstrained — recorded in
docs/design/choice-sequence-codec.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from slm_training.dsl.production_codec import (
    CHOICE_STOP,
    decode_choices,
    encode_choices,
)

PAD = "<PAD>"
BOS = "<BOS>"
MASK = "<MASK>"
EOS = "<EOS>"
UNK = "<UNK>"
_SPECIALS = (PAD, BOS, MASK, EOS, UNK)
CHOICE_TOKENIZER_VERSION = 1
_SLOT_RE = re.compile(r"^@(\d+)$")


@dataclass
class ChoiceTokenizer:
    """Fixed vocabulary over choice-stream decision tokens."""

    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)
    version: int = CHOICE_TOKENIZER_VERSION

    # ── special ids (OpenUITokenizer-compatible) ─────────────────────────
    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def mask_id(self) -> int:
        return self.token_to_id[MASK]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    # ── build / persist ──────────────────────────────────────────────────
    @classmethod
    def build(
        cls,
        sources: list[str],
        *,
        slot_slots: int = 16,
        ref_slots: int = 16,
    ) -> "ChoiceTokenizer":
        """Vocabulary = specials + corpus decision tokens + slot/ref pools.

        Fixed ``@k``/``&k`` pools keep the vocabulary stable across corpora
        (mirroring the DSL-native tokenizer's reserved symbol rows).
        """
        tokens: dict[str, None] = {}
        for token in (CHOICE_STOP, "["):
            tokens[token] = None
        for k in range(max(1, slot_slots)):
            tokens[f"@{k}"] = None
        for k in range(max(1, ref_slots)):
            tokens[f"&{k}"] = None
        for source in sources:
            for token in encode_choices(source).tokens:
                tokens[token] = None
        token_to_id: dict[str, int] = {}
        for special in _SPECIALS:
            token_to_id[special] = len(token_to_id)
        for token in sorted(tokens):
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
        return cls(
            token_to_id=token_to_id,
            id_to_token={i: t for t, i in token_to_id.items()},
        )

    def to_json(self) -> str:
        return json.dumps(
            {"version": self.version, "token_to_id": self.token_to_id},
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> "ChoiceTokenizer":
        data = json.loads(payload)
        token_to_id = {str(k): int(v) for k, v in data["token_to_id"].items()}
        return cls(
            token_to_id=token_to_id,
            id_to_token={i: t for t, i in token_to_id.items()},
            version=int(data.get("version", CHOICE_TOKENIZER_VERSION)),
        )

    def save(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "ChoiceTokenizer":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # ── encode / decode ──────────────────────────────────────────────────
    def encode(self, text: str, add_special: bool = True) -> list[int]:
        """OpenUI source → canonical choice-decision ids.

        Non-program snippets (e.g. the trainer probing a bare quoted
        placeholder for fidelity spans) have no choice decomposition; they
        encode to an empty span rather than raising — placeholder identity in
        choice space is positional (``@k``), not lexical.
        """
        from slm_training.dsl.lang_core import ParseError

        try:
            choices = encode_choices(text)
        except ParseError:
            return [self.bos_id, self.eos_id] if add_special else []
        ids = [self.token_to_id.get(t, self.unk_id) for t in choices.tokens]
        if add_special:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Choice ids → OpenUI via the deterministic detokenizer.

        Without an external slot contract a generic ``:slot.k`` contract sized
        by the highest ``@k`` reference is used (fixture decode only; eval
        studies pass the record's real contract through ``decode_with_contract``).
        """
        from slm_training.dsl.lang_core import ParseError

        tokens = self._tokens(ids, skip_special=skip_special)
        try:
            return decode_choices(
                tokens, slot_contract=self._generic_contract(tokens)
            )
        except ParseError:
            # An illegal decision stream is an honest decode failure — the
            # detokenizer never invents a program the choices don't denote.
            return ""

    def decode_with_contract(
        self, ids: list[int], slot_contract: list[str] | tuple[str, ...]
    ) -> str:
        tokens = self._tokens(ids, skip_special=True)
        return decode_choices(tokens, slot_contract=tuple(slot_contract))

    def _tokens(self, ids: list[int], *, skip_special: bool) -> list[str]:
        special = {self.pad_id, self.bos_id, self.eos_id, self.mask_id}
        out: list[str] = []
        for i in ids:
            if skip_special and int(i) in special:
                continue
            out.append(self.id_to_token.get(int(i), UNK))
        return out

    @staticmethod
    def _generic_contract(tokens: list[str]) -> tuple[str, ...]:
        highest = -1
        for token in tokens:
            match = _SLOT_RE.match(token)
            if match:
                highest = max(highest, int(match.group(1)))
        return tuple(f":slot.{k}" for k in range(highest + 1))


def is_choice_tokenizer(tokenizer: object) -> bool:
    return isinstance(tokenizer, ChoiceTokenizer)


__all__ = ["ChoiceTokenizer", "is_choice_tokenizer"]
