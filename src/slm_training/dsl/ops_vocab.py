"""Reserved compute-ops vocabulary shared by the encoder and the decoder.

Decode invariant I13 (`docs/design/decode-invariants.md`): the encoder
vocabulary MUST reserve a vocabulary of compute operations — AST / graph / set
/ topology / history — that is **known and shared by the decoder vocabulary**.
Grammar symbols layer on top of that shared base. Natural-language vocabulary
sits above grammar symbols and is strictly optional.

This module is that vocabulary. Two properties make it worth having rather than
being one more list to keep in sync:

* **It is derived, not authored.** Every entry comes from a live operator
  registry (`operators.local`, `operators.topology`, `operators.conversation`),
  so an op cannot exist in the model's vocabulary without an implementation, and
  cannot be implemented without appearing here.
* **There is exactly one mapping.** :func:`shared_token_ids` is the only source
  of op token ids. "Shared encoder↔decoder" is not a convention both towers are
  asked to honor — it is the same function called twice.

Ids live in the versioned ``ops`` namespace of
:mod:`slm_training.dsl.openui_tokens`, above every existing codec range, so
adding this vocabulary moves no stored embedding row. Drift is caught by
`scripts/verify_decode_invariants.py` against the fingerprint pinned in
`src/slm_training/resources/ops_vocab_registry.json`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from slm_training.dsl.openui_tokens import logical_token_id

OPS_VOCAB_SCHEMA = "ops_vocab/v1"
OPS_NAMESPACE = "ops"
REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "resources" / "ops_vocab_registry.json"
)


class OpFamily(str, Enum):
    """What kind of computation an op performs on the conversation artifact."""

    # Node-local edits: the value at one position changes.
    AST = "ast"
    # Graph rewiring: an edge moves, so parentage or containment changes.
    GRAPH = "graph"
    # Set/sequence mutation: the membership or order of a child collection.
    SET = "set"
    # Topology: the shape of a subtree changes (wrap/unwrap/expand/contract).
    TOPOLOGY = "topology"
    # History: operations on the conversation event store itself.
    HISTORY = "history"


@dataclass(frozen=True)
class OpSymbolV1:
    """One reserved op, shared verbatim by both towers."""

    op_id: str
    family: OpFamily
    local_id: int

    @property
    def token(self) -> str:
        return f"<|op:{self.op_id}|>"

    @property
    def token_id(self) -> int:
        return logical_token_id(OPS_NAMESPACE, self.local_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op_id,
            "family": self.family.value,
            "local_id": self.local_id,
            "token": self.token,
            "token_id": self.token_id,
        }


# How each live operator id is classified. Keeping this explicit (rather than
# guessing from the id) means adding an operator without deciding what kind of
# computation it is fails the build instead of silently landing in a default.
_FAMILY_BY_OP: Mapping[str, OpFamily] = {
    # operators.local
    "openui.add_child": OpFamily.SET,
    "openui.remove_node": OpFamily.SET,
    "openui.reorder_children": OpFamily.SET,
    "openui.replace_node": OpFamily.AST,
    "openui.set_property": OpFamily.AST,
    "openui.unset_property": OpFamily.AST,
    # operators.topology
    "openui.move_node": OpFamily.GRAPH,
    "openui.reparent_node": OpFamily.GRAPH,
    "openui.duplicate_subtree": OpFamily.GRAPH,
    "openui.wrap_node": OpFamily.TOPOLOGY,
    "openui.unwrap_node": OpFamily.TOPOLOGY,
    "openui.expand_template": OpFamily.TOPOLOGY,
    "openui.contract_subtree": OpFamily.TOPOLOGY,
}


class OpsVocabError(ValueError):
    """The reserved ops vocabulary and the live operator registries disagree."""


def _live_operator_ids() -> tuple[str, ...]:
    from slm_training.dsl.operators import local, topology

    ids: list[str] = []
    for module in (local, topology):
        for declaration in module._declarations():  # noqa: SLF001 - registry read
            ids.append(str(declaration.operator_id))
    if len(set(ids)) != len(ids):
        raise OpsVocabError("duplicate operator id across operator registries")
    return tuple(ids)


def _history_op_ids() -> tuple[str, ...]:
    from slm_training.dsl.operators.conversation import ConversationOperation

    return tuple(f"conversation.{item.value}" for item in ConversationOperation)


def build_ops_vocab() -> tuple[OpSymbolV1, ...]:
    """Derive the reserved ops vocabulary from the live operator registries.

    Ordering is ``(family, op_id)`` so a new op lands at a deterministic id and
    the fingerprint changes visibly rather than silently renumbering peers.
    """
    live = _live_operator_ids()
    unclassified = sorted(set(live) - set(_FAMILY_BY_OP))
    if unclassified:
        raise OpsVocabError(
            "operators are missing an OpFamily classification: "
            + ", ".join(unclassified)
            + "; add them to _FAMILY_BY_OP (decode invariant I13)"
        )
    stale = sorted(set(_FAMILY_BY_OP) - set(live))
    if stale:
        raise OpsVocabError(
            "reserved ops no longer exist in any operator registry: "
            + ", ".join(stale)
        )
    entries = [(op_id, _FAMILY_BY_OP[op_id]) for op_id in live]
    entries.extend((op_id, OpFamily.HISTORY) for op_id in _history_op_ids())
    entries.sort(key=lambda row: (row[1].value, row[0]))
    return tuple(
        OpSymbolV1(op_id=op_id, family=family, local_id=index)
        for index, (op_id, family) in enumerate(entries)
    )


OPS_VOCAB: tuple[OpSymbolV1, ...] = build_ops_vocab()


def shared_token_ids() -> dict[str, int]:
    """The one op-token mapping. Both towers call this; neither owns a copy."""
    return {symbol.token: symbol.token_id for symbol in OPS_VOCAB}


def ops_by_family(family: OpFamily) -> tuple[OpSymbolV1, ...]:
    return tuple(symbol for symbol in OPS_VOCAB if symbol.family is family)


def fingerprint() -> str:
    """Content address over the ordered vocabulary."""
    payload = json.dumps(
        [symbol.to_dict() for symbol in OPS_VOCAB],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest() -> dict[str, Any]:
    return {
        "schema": OPS_VOCAB_SCHEMA,
        "namespace": OPS_NAMESPACE,
        "fingerprint": fingerprint(),
        "count": len(OPS_VOCAB),
        "families": {
            family.value: [symbol.op_id for symbol in ops_by_family(family)]
            for family in OpFamily
        },
        "symbols": [symbol.to_dict() for symbol in OPS_VOCAB],
    }


def assert_layering(grammar_token_ids: Mapping[str, int]) -> None:
    """Grammar symbols layer *on top of* ops; the two never collide.

    NL vocabulary is not represented here at all — that is the point. It is
    optional fluff above the grammar layer and must never be load-bearing.
    """
    ops_ids = set(shared_token_ids().values())
    collisions = sorted(
        token for token, token_id in grammar_token_ids.items() if token_id in ops_ids
    )
    if collisions:
        raise OpsVocabError(
            "grammar symbols collide with the reserved ops namespace: "
            + ", ".join(collisions[:8])
        )


def assert_shared_across_towers(encoder: Mapping[str, int], decoder: Mapping[str, int]) -> None:
    """Both towers must carry the reserved ops at identical ids (I13)."""
    shared = shared_token_ids()
    for name, vocab in (("encoder", encoder), ("decoder", decoder)):
        missing = sorted(token for token in shared if token not in vocab)
        if missing:
            raise OpsVocabError(
                f"{name} vocabulary is missing reserved ops: {', '.join(missing[:8])}"
            )
        disagreeing = sorted(
            token for token, token_id in shared.items() if vocab[token] != token_id
        )
        if disagreeing:
            raise OpsVocabError(
                f"{name} vocabulary assigns different ids to reserved ops: "
                + ", ".join(disagreeing[:8])
            )


def load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def write_registry() -> Path:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return REGISTRY_PATH


__all__ = [
    "OPS_NAMESPACE",
    "OPS_VOCAB",
    "OPS_VOCAB_SCHEMA",
    "OpFamily",
    "OpSymbolV1",
    "OpsVocabError",
    "REGISTRY_PATH",
    "assert_layering",
    "assert_shared_across_towers",
    "build_ops_vocab",
    "fingerprint",
    "load_registry",
    "manifest",
    "ops_by_family",
    "shared_token_ids",
    "write_registry",
]
