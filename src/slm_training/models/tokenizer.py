"""OpenUI Lang tokenizer and vocabulary for TwoTower training."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_TOKEN_RE = re.compile(
    r"""
    (:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)
  | ("(?:\\.|[^"\\])*")
  | ([A-Za-z_][A-Za-z0-9_]*)
  | (\d+(?:\.\d+)?)
  | (\[|\]|\(|\)|,|=)
  | (\s+)
    """,
    re.VERBOSE,
)

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
MASK = "<mask>"
UNK = "<unk>"
SPECIAL = [PAD, BOS, EOS, MASK, UNK]


@dataclass
class OpenUITokenizer:
    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)

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

    @classmethod
    def build(cls, texts: Iterable[str]) -> OpenUITokenizer:
        vocab = list(SPECIAL)
        seen = set(vocab)
        for text in texts:
            for tok in tokenize_text(text):
                if tok not in seen:
                    seen.add(tok)
                    vocab.append(tok)
        token_to_id = {t: i for i, t in enumerate(vocab)}
        id_to_token = {i: t for t, i in token_to_id.items()}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokenize_text(text)]
        if add_special:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        pieces: list[str] = []
        special = {self.pad_id, self.bos_id, self.eos_id, self.mask_id}
        for i in ids:
            if skip_special and i in special:
                continue
            pieces.append(self.id_to_token.get(i, UNK))
        return "".join(pieces)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"token_to_id": self.token_to_id}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> OpenUITokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        token_to_id = {str(k): int(v) for k, v in data["token_to_id"].items()}
        id_to_token = {i: t for t, i in token_to_id.items()}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)


def tokenize_text(text: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    for m in _TOKEN_RE.finditer(text):
        if m.start() != pos:
            for ch in text[pos : m.start()]:
                tokens.append(ch)
        for g in m.groups():
            if g is not None:
                tokens.append(g)
                break
        pos = m.end()
    if pos < len(text):
        for ch in text[pos:]:
            tokens.append(ch)
    return tokens
