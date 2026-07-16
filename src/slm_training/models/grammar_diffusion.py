"""Trans-dimensional diffusion over typed OpenUI production trees."""

from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from difflib import SequenceMatcher
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.plugin import GenerationRequest
from slm_training.models.blocks import RMSNorm, TransformerBlock
from slm_training.models.context import (
    ScratchContextEncoder,
    build_context_encoder,
    is_hf_context,
)
from slm_training.models.tokenizer import OpenUITokenizer, tokenize_text
from slm_training.models.twotower import format_context_text


def _load_production_codec(texts: list[str]):
    from slm_training.dsl.production_codec import ProductionCodec

    return ProductionCodec.build(texts)


def _codec_kind_from_vocab(id_to_production: dict[int, str]) -> str:
    for tok in id_to_production.values():
        if tok.startswith(("+", "@", "^", "&", "#")) or tok in {"=", "-", "[", "]"}:
            return "production"
        if tok in {"NEWLINE", "ASSIGN", "SLOT", "LPAREN", "RPAREN"} or tok.startswith(
            ("COMP:", "VAR:", "LIT:", "REF:", "TOK:")
        ):
            return "inline"
    return "production"


def _restore_codec(raw_codec: dict[str, Any]) -> InlineProductionCodec:
    id_to_prod = {
        int(k): v for k, v in (raw_codec.get("id_to_production") or {}).items()
    }
    kind = str(raw_codec.get("codec_kind") or _codec_kind_from_vocab(id_to_prod))
    common = dict(
        production_to_id=dict(raw_codec.get("production_to_id") or {}),
        id_to_production=id_to_prod,
        pad_id=int(raw_codec.get("pad_id", 0)),
        bos_id=int(raw_codec.get("bos_id", 1)),
        eos_id=int(raw_codec.get("eos_id", 2)),
        mask_id=int(raw_codec.get("mask_id", 3)),
        slot_none_id=int(raw_codec.get("slot_none_id", 0)),
    )
    if kind == "production":
        from slm_training.dsl.production_codec import ProductionCodec

        return ProductionCodec(
            **common,
            unk_id=int(raw_codec.get("unk_id", 4)),
        )
    return InlineProductionCodec(**common)


@dataclass
class InlineProductionCodec:
    """Minimal production linearization until dsl.production_codec ships."""

    production_to_id: dict[str, int] = field(default_factory=dict)
    id_to_production: dict[int, str] = field(default_factory=dict)
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    mask_id: int = 3
    slot_none_id: int = 0

    @classmethod
    def build(cls, texts: list[str]) -> InlineProductionCodec:
        specials = ["<pad>", "<bos>", "<eos>", "<mask>"]
        vocab: dict[str, int] = {tok: i for i, tok in enumerate(specials)}
        structural = [
            "NEWLINE",
            "ASSIGN",
            "LPAREN",
            "RPAREN",
            "LBRACK",
            "RBRACK",
            "COMMA",
            "SPACE",
            "SLOT",
        ]
        for tok in structural:
            if tok not in vocab:
                vocab[tok] = len(vocab)
        for text in texts:
            for prod in cls._extract_production_names(text):
                if prod not in vocab:
                    vocab[prod] = len(vocab)
        inv = {i: t for t, i in vocab.items()}
        return cls(
            production_to_id=vocab,
            id_to_production=inv,
            pad_id=vocab["<pad>"],
            bos_id=vocab["<bos>"],
            eos_id=vocab["<eos>"],
            mask_id=vocab["<mask>"],
        )

    @staticmethod
    def _extract_production_names(text: str) -> list[str]:
        names: list[str] = []
        for prod, _slot in InlineProductionCodec._walk(text, ["__noop__"]):
            names.append(prod)
        return names

    @staticmethod
    def _walk(text: str, slot_inventory: list[str]) -> list[tuple[str, int]]:
        inv = {
            ph if ph.startswith(":") else f":{ph}": i + 1
            for i, ph in enumerate(slot_inventory)
        }
        tokens = tokenize_text(text)
        out: list[tuple[str, int]] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "\n":
                out.append(("NEWLINE", 0))
                i += 1
                continue
            if tok == " ":
                out.append(("SPACE", 0))
                i += 1
                continue
            if tok in {"=", "(", ")", "[", "]", ","}:
                mapping = {
                    "=": "ASSIGN",
                    "(": "LPAREN",
                    ")": "RPAREN",
                    "[": "LBRACK",
                    "]": "RBRACK",
                    ",": "COMMA",
                }
                out.append((mapping[tok], 0))
                i += 1
                continue
            if tok == '"':
                placeholder = InlineProductionCodec._read_placeholder(tokens, i)
                if placeholder is not None:
                    ph, end = placeholder
                    slot_id = inv.get(ph, 0)
                    out.append(("SLOT", slot_id))
                    i = end + 1
                    continue
                literal, end = InlineProductionCodec._read_literal(tokens, i)
                out.append((f"LIT:{literal}", 0))
                i = end + 1
                continue
            if i + 1 < len(tokens) and tokens[i + 1] == "=":
                out.append((f"VAR:{tok}", 0))
                i += 1
                continue
            if tok and tok[0].isupper() and tok.isidentifier():
                out.append((f"COMP:{tok}", 0))
                i += 1
                continue
            if tok.isidentifier():
                out.append((f"REF:{tok}", 0))
                i += 1
                continue
            out.append((f"TOK:{tok}", 0))
            i += 1
        return out

    @staticmethod
    def _read_placeholder(tokens: list[str], start: int) -> tuple[str, int] | None:
        if start >= len(tokens) or tokens[start] != '"':
            return None
        if start + 1 >= len(tokens) or tokens[start + 1] != ":":
            return None
        parts = [":"]
        i = start + 2
        while i < len(tokens):
            if tokens[i] == '"':
                ph = "".join(parts)
                return ph, i
            parts.append(tokens[i])
            i += 1
        return None

    @staticmethod
    def _read_literal(tokens: list[str], start: int) -> tuple[str, int]:
        if start >= len(tokens) or tokens[start] != '"':
            return "", start
        parts = ['"']
        i = start + 1
        while i < len(tokens):
            parts.append(tokens[i])
            if tokens[i] == '"':
                return "".join(parts), i
            i += 1
        return "".join(parts), i

    @property
    def vocab_size(self) -> int:
        return len(self.production_to_id)

    def encode(
        self,
        openui: str,
        slot_inventory: list[str] | None = None,
        *,
        max_len: int = 256,
    ) -> tuple[list[int], list[int]]:
        inventory = list(slot_inventory or extract_placeholders(openui))
        steps = self._walk(openui, inventory)
        prod_ids = [self.bos_id]
        slot_ids = [self.slot_none_id]
        for prod, slot in steps:
            pid = self.production_to_id.get(prod)
            if pid is None:
                pid = self.production_to_id.setdefault(prod, len(self.production_to_id))
                self.id_to_production[pid] = prod
            prod_ids.append(pid)
            slot_ids.append(int(slot))
        prod_ids.append(self.eos_id)
        slot_ids.append(self.slot_none_id)
        prod_ids = prod_ids[:max_len]
        slot_ids = slot_ids[: len(prod_ids)]
        if len(slot_ids) < len(prod_ids):
            slot_ids.extend([self.slot_none_id] * (len(prod_ids) - len(slot_ids)))
        return prod_ids, slot_ids

    def decode(
        self,
        production_ids: list[int],
        slot_ids: list[int],
        slot_inventory: list[str],
        *,
        stop_at_mask: bool = False,
    ) -> str:
        pieces: list[str] = []
        for idx, (pid, sid) in enumerate(zip(production_ids, slot_ids)):
            if stop_at_mask and pid == self.mask_id:
                pieces.append("<mask>")
                break
            if pid in {self.pad_id, self.bos_id}:
                continue
            if pid == self.eos_id:
                break
            if pid == self.mask_id:
                pieces.append("<mask>")
                continue
            prod = self.id_to_production.get(pid, "")
            if prod == "NEWLINE":
                if pieces and not pieces[-1].endswith("\n"):
                    pieces.append("\n")
            elif prod == "SPACE":
                if pieces and not pieces[-1].endswith((" ", "\n", "(", "[", ",")):
                    pieces.append(" ")
            elif prod == "ASSIGN":
                pieces.append(" = ")
            elif prod == "LPAREN":
                pieces.append("(")
            elif prod == "RPAREN":
                pieces.append(")")
            elif prod == "LBRACK":
                pieces.append("[")
            elif prod == "RBRACK":
                pieces.append("]")
            elif prod == "COMMA":
                pieces.append(", ")
            elif prod == "SLOT":
                slot_idx = int(sid) - 1
                if 0 <= slot_idx < len(slot_inventory):
                    ph = slot_inventory[slot_idx]
                    if not ph.startswith(":"):
                        ph = f":{ph}"
                    pieces.append(f'"{ph}"')
                else:
                    pieces.append('":slot.unknown"')
            elif prod.startswith("VAR:"):
                name = prod.split(":", 1)[1]
                if pieces and not pieces[-1].endswith(("\n", " ")):
                    pieces.append(" ")
                pieces.append(name)
            elif prod.startswith("COMP:"):
                name = prod.split(":", 1)[1]
                pieces.append(name)
            elif prod.startswith("REF:"):
                name = prod.split(":", 1)[1]
                pieces.append(name)
            elif prod.startswith("LIT:"):
                pieces.append(prod.split(":", 1)[1])
            elif prod.startswith("TOK:"):
                pieces.append(prod.split(":", 1)[1])
            _ = idx
        text = "".join(pieces)
        return text.replace("  ", " ").replace(" \n", "\n").strip() + (
            "\n" if text.endswith("\n") else ""
        )


class TopologyAction(IntEnum):
    EXPAND = 0
    KEEP = 1
    DELETE = 2
    CONTRACT = 3
    STOP = 4


NODE_TYPES = ("document", "statement", "expression", "component", "list", "leaf")
NODE_TYPE_ID = {name: index for index, name in enumerate(NODE_TYPES)}
V05_MARKERS = {"r=", "$=", "q=", "m=", "a=", "="}


@dataclass
class TopologyNode:
    node_id: int
    node_type: str
    production_id: int
    slot_id: int = 0
    parent_id: int = -1
    depth: int = 0
    sibling_index: int = 0
    children: list[TopologyNode] = field(default_factory=list)
    active: bool = False
    target_action: int = int(TopologyAction.KEEP)
    target_production_id: int | None = None
    target_arity: int = 0
    target_slot_id: int = 0
    critic_target: float = 1.0
    scope_bucket: int = 0
    scope_noise: float = 0.0
    scope_summary_target: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    scope_gate_target: float = 1.0
    failure_cone_target: float = 0.0

    def clone(self) -> TopologyNode:
        return TopologyNode(
            node_id=self.node_id,
            node_type=self.node_type,
            production_id=self.production_id,
            slot_id=self.slot_id,
            parent_id=self.parent_id,
            depth=self.depth,
            sibling_index=self.sibling_index,
            children=[child.clone() for child in self.children],
            active=self.active,
            target_action=self.target_action,
            target_production_id=self.target_production_id,
            target_arity=self.target_arity,
            target_slot_id=self.target_slot_id,
            critic_target=self.critic_target,
            scope_bucket=self.scope_bucket,
            scope_noise=self.scope_noise,
            scope_summary_target=self.scope_summary_target,
            scope_gate_target=self.scope_gate_target,
            failure_cone_target=self.failure_cone_target,
        )


def _node_type(token: str) -> str:
    if token.startswith("+"):
        return "component"
    if token == "[":
        return "list"
    return "leaf"


def _refresh_layout(root: TopologyNode, *, preserve_ids: bool = False) -> None:
    """Refresh tree-relative coordinates without losing persistent runtime IDs."""
    next_id = 0

    def visit(node: TopologyNode, parent: int, depth: int, sibling: int) -> None:
        nonlocal next_id
        if not preserve_ids:
            node.node_id = next_id
            next_id += 1
        node.parent_id = parent
        node.depth = depth
        node.sibling_index = sibling
        for index, child in enumerate(node.children):
            visit(child, node.node_id, depth + 1, index)

    visit(root, -1, 0, 0)


def _flatten(root: TopologyNode) -> list[TopologyNode]:
    out: list[TopologyNode] = []

    def visit(node: TopologyNode) -> None:
        out.append(node)
        for child in node.children:
            visit(child)

    visit(root)
    return out


def _topology_from_ids(
    codec: InlineProductionCodec,
    production_ids: list[int],
    slot_ids: list[int],
) -> TopologyNode:
    pairs = [
        (pid, sid)
        for pid, sid in zip(production_ids, slot_ids)
        if pid not in {codec.pad_id, codec.bos_id, codec.eos_id}
    ]
    next_id = 1

    def make(node_type: str, pid: int, sid: int = 0) -> TopologyNode:
        nonlocal next_id
        node = TopologyNode(next_id, node_type, pid, sid)
        next_id += 1
        return node

    def parse_expr(index: int) -> tuple[TopologyNode, int]:
        if index >= len(pairs):
            raise ValueError("unexpected end of production tree")
        pid, sid = pairs[index]
        token = codec.id_to_production.get(pid, "<unk>")
        node = make(_node_type(token), pid, sid)
        index += 1
        if token.startswith("+"):
            while index < len(pairs):
                next_token = codec.id_to_production.get(pairs[index][0], "<unk>")
                if next_token == "-":
                    return node, index + 1
                child, index = parse_expr(index)
                node.children.append(child)
            raise ValueError("unterminated component production")
        if token == "[":
            while index < len(pairs):
                next_token = codec.id_to_production.get(pairs[index][0], "<unk>")
                if next_token == "]":
                    return node, index + 1
                child, index = parse_expr(index)
                node.children.append(child)
            raise ValueError("unterminated list production")
        return node, index

    v05_id = codec.production_to_id.get("!v0.5")
    root_pid = codec.bos_id
    index = 0
    is_v05 = bool(pairs and pairs[0][0] == v05_id)
    if is_v05:
        root_pid = pairs[0][0]
        index = 1
    root = TopologyNode(0, "document", root_pid)
    if is_v05:
        eol_id = codec.production_to_id.get(";")
        while index < len(pairs):
            marker_id, marker_slot = pairs[index]
            marker = codec.id_to_production.get(marker_id, "")
            if marker not in V05_MARKERS:
                raise ValueError(f"expected v0.5 statement marker, got {marker!r}")
            statement = make("statement", marker_id, marker_slot)
            index += 1
            while index < len(pairs) and pairs[index][0] != eol_id:
                pid, sid = pairs[index]
                statement.children.append(make("expression", pid, sid))
                index += 1
            index += int(index < len(pairs))
            root.children.append(statement)
    else:
        assign_id = codec.production_to_id.get("=")
        while index < len(pairs):
            if pairs[index][0] != assign_id:
                raise ValueError("expected statement production")
            statement = make("statement", pairs[index][0], pairs[index][1])
            child, index = parse_expr(index + 1)
            statement.children.append(child)
            root.children.append(statement)
    _refresh_layout(root)
    return root


def topology_from_openui(
    codec: InlineProductionCodec,
    openui: str,
    slot_inventory: list[str] | None = None,
    *,
    max_len: int = 0,
) -> TopologyNode:
    production_ids, slot_ids = codec.encode(openui, slot_inventory, max_len=max_len)
    return _topology_from_ids(codec, production_ids, slot_ids)


def _serialize_topology(
    codec: InlineProductionCodec, root: TopologyNode
) -> tuple[list[int], list[int]]:
    production_ids = [codec.bos_id]
    slot_ids = [codec.slot_none_id]
    root_token = codec.id_to_production.get(root.production_id, "")
    is_v05 = root_token == "!v0.5"
    if is_v05:
        production_ids.append(root.production_id)
        slot_ids.append(root.slot_id)

    def emit(node: TopologyNode) -> None:
        production_ids.append(node.production_id)
        slot_ids.append(node.slot_id)
        token = codec.id_to_production.get(node.production_id, "")
        for child in node.children:
            emit(child)
        if token.startswith("+"):
            production_ids.append(codec.production_to_id["-"])
            slot_ids.append(codec.slot_none_id)
        elif token == "[":
            production_ids.append(codec.production_to_id["]"])
            slot_ids.append(codec.slot_none_id)

    for statement in root.children:
        emit(statement)
        if is_v05:
            production_ids.append(codec.production_to_id[";"])
            slot_ids.append(codec.slot_none_id)
    production_ids.append(codec.eos_id)
    slot_ids.append(codec.slot_none_id)
    return production_ids, slot_ids


def topology_arity_accuracy(
    codec: InlineProductionCodec, prediction: str, gold: str
) -> float:
    try:
        predicted = _flatten(topology_from_openui(codec, prediction))
        expected = _flatten(topology_from_openui(codec, gold))
    except (ValueError, KeyError):
        return 0.0
    pairs = zip(predicted, expected)
    total = max(len(predicted), len(expected), 1)
    correct = sum(
        left.production_id == right.production_id
        and len(left.children) == len(right.children)
        for left, right in pairs
    )
    return correct / total


def production_sequence_accuracy(
    codec: InlineProductionCodec, prediction: str, gold: str
) -> float:
    try:
        left, _ = codec.encode(prediction, max_len=0)
        right, _ = codec.encode(gold, max_len=0)
    except (ValueError, KeyError):
        return 0.0
    return SequenceMatcher(a=left, b=right, autojunk=False).ratio()


@dataclass
class GrammarDiffusionConfig:
    d_model: int = 96
    n_heads: int = 4
    context_layers: int = 2
    denoiser_layers: int = 3
    max_prompt_len: int = 192
    max_target_len: int = 192
    dropout: float = 0.0
    block_size: int = 4  # retained for old config parsing; topology ignores it
    mask_min: float = 0.15
    mask_max: float = 0.85
    gen_steps: int = 8
    max_slots: int = 16
    context_backend: str = "scratch"
    hf_model_name: str = "HuggingFaceTB/SmolLM2-135M"
    freeze_context: bool = False
    local_files_only: bool = False
    grammar_dsl: str = "openui"
    parallel_unmask: str = "adaptive"
    grammar_top_k: int = 8
    production_loss_weight: float = 1.0
    slot_loss_weight: float = 0.5
    confidence_loss_weight: float = 0.25
    action_loss_weight: float = 1.0
    arity_loss_weight: float = 0.5
    critic_loss_weight: float = 0.25
    design_md_in_context: bool = False
    design_md_budget: int = 1200
    schema_in_context: bool = False
    slot_contract_in_context: bool = True
    slot_contract_constrained_decode: bool = True
    honest_slot_contract: bool = True
    topology_actions: bool = True
    topology_structural_embeddings: bool = True
    topology_heterogeneous_noise: bool = True
    topology_critic_decode: bool = True
    topology_bounded_buffer: bool = True
    topology_max_nodes: int = 256
    topology_max_active: int = 64
    topology_max_arity: int = 8
    topology_max_depth: int = 32
    topology_max_phases: int = 32
    topology_global_sync_interval: int = 4
    topology_accept_threshold: float = 0.5
    topology_contract_threshold: float = 0.25
    scope_contracts: bool = False
    scope_independent_noise: bool = False
    scope_local_oracle: bool = False
    scope_contract_negatives: bool = False
    seed: int = 0
    eval_mode_no_fallback: bool = True


class _TopologyCore(nn.Module):
    def __init__(
        self,
        n_productions: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        max_depth: int,
        max_arity: int,
        dropout: float,
        scope_contracts: bool = False,
    ) -> None:
        super().__init__()
        self.tok = nn.Embedding(n_productions, d_model)
        self.node_type = nn.Embedding(len(NODE_TYPES), d_model)
        self.parent_type = nn.Embedding(len(NODE_TYPES) + 1, d_model)
        self.depth = nn.Embedding(max_depth + 1, d_model)
        self.sibling = nn.Embedding(max_arity + 1, d_model)
        self.scope_contract = nn.Embedding(257, d_model) if scope_contracts else None
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, dropout=dropout, cross_attn=True)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, n_productions, bias=False)
        self.lm_head.weight = self.tok.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        node_types: torch.Tensor,
        parent_types: torch.Tensor,
        depths: torch.Tensor,
        siblings: torch.Tensor,
        scope_buckets: torch.Tensor | None,
        context: torch.Tensor,
        *,
        pad_id: int,
        structural: bool,
        ctx_pad_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.tok(input_ids)
        if structural:
            hidden = (
                hidden
                + self.node_type(node_types)
                + self.parent_type(parent_types)
                + self.depth(depths.clamp_max(self.depth.num_embeddings - 1))
                + self.sibling(siblings.clamp_max(self.sibling.num_embeddings - 1))
            )
        if self.scope_contract is not None and scope_buckets is not None:
            hidden = hidden + self.scope_contract(scope_buckets.clamp(0, 256))
        pad_mask = input_ids.eq(pad_id)
        for layer in self.layers:
            hidden = layer(
                hidden,
                self_pad_mask=pad_mask,
                ctx=context,
                ctx_pad_mask=ctx_pad_mask,
            )
        hidden = self.norm(hidden)
        return self.lm_head(hidden), hidden


class GrammarDenoiser(nn.Module):
    """Tree-structural denoiser with expansion, arity, slot, and critic heads."""

    def __init__(
        self,
        n_productions: int,
        max_slots: int,
        d_model: int = 96,
        n_layers: int = 3,
        n_heads: int = 4,
        max_depth: int = 32,
        max_arity: int = 8,
        dropout: float = 0.0,
        pad_id: int = 0,
        scope_contracts: bool = False,
    ) -> None:
        super().__init__()
        self.core = _TopologyCore(
            n_productions,
            d_model,
            n_layers,
            n_heads,
            max_depth,
            max_arity,
            dropout,
            scope_contracts,
        )
        self.pad_id = pad_id
        self.slot_head = nn.Linear(d_model, max_slots + 1)
        self.action_head = nn.Linear(d_model, len(TopologyAction))
        self.arity_head = nn.Linear(d_model, max_arity + 1)
        self.critic_head = nn.Linear(d_model, 1)
        self.confidence_head = nn.Linear(d_model, 1)
        self.scope_summary_head = nn.Linear(d_model, 4) if scope_contracts else None
        self.scope_gate_head = nn.Linear(d_model, 1) if scope_contracts else None
        self.failure_cone_head = nn.Linear(d_model, 1) if scope_contracts else None

    def forward(
        self,
        input_ids: torch.Tensor,
        node_types: torch.Tensor,
        parent_types: torch.Tensor,
        depths: torch.Tensor,
        siblings: torch.Tensor,
        scope_buckets: torch.Tensor | None,
        context: torch.Tensor,
        *,
        structural: bool,
        ctx_pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        production_logits, hidden = self.core(
            input_ids,
            node_types,
            parent_types,
            depths,
            siblings,
            scope_buckets,
            context,
            pad_id=self.pad_id,
            structural=structural,
            ctx_pad_mask=ctx_pad_mask,
        )
        return (
            production_logits,
            self.slot_head(hidden),
            self.action_head(hidden),
            self.arity_head(hidden),
            torch.sigmoid(self.critic_head(hidden).squeeze(-1)),
            torch.sigmoid(self.confidence_head(hidden).squeeze(-1)),
            self.scope_summary_head(hidden)
            if self.scope_summary_head is not None
            else None,
            (
                torch.sigmoid(self.scope_gate_head(hidden).squeeze(-1))
                if self.scope_gate_head is not None
                else None
            ),
            (
                torch.sigmoid(self.failure_cone_head(hidden).squeeze(-1))
                if self.failure_cone_head is not None
                else None
            ),
        )


def _pad_rows(
    rows: list[list[int | float]], value: int | float, device: str | torch.device
) -> torch.Tensor:
    width = max((len(row) for row in rows), default=1)
    dtype = torch.float32 if isinstance(value, float) else torch.long
    out = torch.full((len(rows), width), value, dtype=dtype, device=device)
    for index, row in enumerate(rows):
        if row:
            out[index, : len(row)] = torch.as_tensor(row, dtype=dtype, device=device)
    return out


def _pad_vector_rows(
    rows: list[list[tuple[float, ...]]], width: int, device: str | torch.device
) -> torch.Tensor:
    length = max((len(row) for row in rows), default=1)
    out = torch.zeros((len(rows), length, width), device=device)
    for index, row in enumerate(rows):
        if row:
            out[index, : len(row)] = torch.tensor(row, device=device)
    return out


def _parent_type_ids(nodes: list[TopologyNode]) -> list[int]:
    by_id = {node.node_id: node for node in nodes}
    missing = len(NODE_TYPES)
    return [
        NODE_TYPE_ID.get(by_id[node.parent_id].node_type, missing)
        if node.parent_id in by_id
        else missing
        for node in nodes
    ]


def _apply_scope_contract(
    root: TopologyNode,
    record: ExampleRecord,
    config: GrammarDiffusionConfig,
    rng: random.Random,
) -> None:
    if not config.scope_contracts:
        return
    contract = (record.meta or {}).get("scope_contract")
    if not isinstance(contract, dict):
        return
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    bucket = int(hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:8], 16) % 256 + 1
    summary = (
        min(1.0, len(contract.get("definitions") or ()) / 8.0),
        min(1.0, len(contract.get("uses") or ()) / 8.0),
        min(1.0, len(contract.get("synthesized_slots") or ()) / 8.0),
        min(1.0, float(contract.get("realized_size") or 0) / 64.0),
    )
    gate = float((record.meta or {}).get("scope_gate_target", 1.0))
    if gate == 0.0 and not config.scope_contract_negatives:
        gate = 1.0
    cone_depth = (
        len((record.meta or {}).get("failure_cone") or ())
        if config.scope_contract_negatives
        else 0
    )
    for node in _flatten(root):
        node.scope_bucket = bucket
        node.scope_noise = rng.random()
        node.scope_summary_target = summary
        node.scope_gate_target = gate
        node.failure_cone_target = float(cone_depth > 0 and node.depth >= cone_depth)


def _corrupt_topology(
    gold: TopologyNode,
    codec: InlineProductionCodec,
    config: GrammarDiffusionConfig,
    rng: random.Random,
) -> TopologyNode:
    root = gold.clone()
    candidates = [node for node in _flatten(root) if node.depth > 0]
    rate = rng.uniform(config.mask_min, config.mask_max)
    selected: set[int] = set()
    # Decode starts from one document mask, so train on that exact state too.
    if rng.random() < 0.35:
        selected.add(root.node_id)
    for node in candidates:
        probability = rate
        if config.topology_heterogeneous_noise:
            probability *= 0.5 + rng.random()
            probability *= 1.0 + min(node.depth, 8) / 16.0
        if config.scope_independent_noise and node.scope_bucket:
            probability *= 0.5 + node.scope_noise
        if root.node_id not in selected and rng.random() < min(1.0, probability):
            selected.add(node.node_id)
    if not selected and candidates:
        selected.add(rng.choice(candidates).node_id)

    def visit(node: TopologyNode, ancestor_selected: bool = False) -> None:
        collapse = not ancestor_selected and node.node_id in selected
        if collapse:
            node.target_action = int(TopologyAction.EXPAND)
            node.target_production_id = node.production_id
            node.target_arity = min(len(node.children), config.topology_max_arity)
            node.target_slot_id = node.slot_id
            if node.node_type not in {"document", "statement"}:
                node.node_type = "expression"
            node.production_id = codec.mask_id
            node.slot_id = codec.slot_none_id
            node.active = True
            node.children = []
            return
        node.target_production_id = node.production_id
        node.target_arity = min(len(node.children), config.topology_max_arity)
        node.target_slot_id = node.slot_id
        node.target_action = int(
            TopologyAction.STOP if not node.children else TopologyAction.KEEP
        )
        for child in node.children:
            visit(child, ancestor_selected or collapse)

    visit(root)
    if config.topology_actions:
        visible = [
            node for node in _flatten(root) if not node.active and node.depth > 0
        ]
        if visible and rng.random() < 0.5:
            node = rng.choice(visible)
            node.target_action = int(TopologyAction.CONTRACT)
            node.critic_target = 0.0
            legal = [
                pid
                for pid in codec.id_to_production
                if pid not in {codec.pad_id, codec.bos_id, codec.eos_id, codec.mask_id}
            ]
            if legal:
                node.production_id = rng.choice(legal)
        parents = [node for node in _flatten(root) if node.children]
        if parents and rng.random() < 0.35:
            parent = rng.choice(parents)
            extra = TopologyNode(
                node_id=max(item.node_id for item in _flatten(root)) + 1,
                node_type="expression",
                production_id=codec.mask_id,
                parent_id=parent.node_id,
                depth=parent.depth + 1,
                sibling_index=len(parent.children),
                active=True,
                target_action=int(TopologyAction.DELETE),
                target_production_id=codec.mask_id,
                critic_target=0.0,
            )
            parent.children.append(extra)
    _refresh_layout(root)
    return root


class GrammarDiffusionModel(nn.Module):
    """Trans-dimensional diffusion over a bounded OpenUI production tree."""

    CHECKPOINT_FORMAT = 2

    def __init__(
        self,
        tokenizer: OpenUITokenizer,
        codec: InlineProductionCodec,
        config: GrammarDiffusionConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.codec = codec
        self.config = config or GrammarDiffusionConfig()
        self.device_name = str(device)
        backend = (self.config.context_backend or "scratch").lower()
        self.context = build_context_encoder(
            backend=backend,
            vocab_size=tokenizer.vocab_size,
            d_model=self.config.d_model,
            n_layers=self.config.context_layers,
            n_heads=self.config.n_heads,
            max_len=self.config.max_prompt_len,
            dropout=self.config.dropout,
            freeze=self.config.freeze_context,
            hf_model_name=self.config.hf_model_name,
            local_files_only=self.config.local_files_only,
        )
        self.denoiser = GrammarDenoiser(
            n_productions=max(8, codec.vocab_size),
            max_slots=self.config.max_slots,
            d_model=self.config.d_model,
            n_layers=self.config.denoiser_layers,
            n_heads=self.config.n_heads,
            max_depth=self.config.topology_max_depth,
            max_arity=self.config.topology_max_arity,
            dropout=self.config.dropout,
            pad_id=codec.pad_id,
            scope_contracts=self.config.scope_contracts,
        )
        self._rng = random.Random(self.config.seed)
        self.last_training_metrics: dict[str, float] = {}
        self._generation_evidence: list[dict[str, Any]] = []
        self.to(device)

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def consume_generation_evidence(self) -> list[dict[str, Any]]:
        evidence, self._generation_evidence = self._generation_evidence, []
        return evidence

    def _encode_context(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if is_hf_context(self.context):
            return self.context.forward_prompts(
                prompts,
                max_len=self.config.max_prompt_len,
                device=self.device_name,
            )
        assert isinstance(self.context, ScratchContextEncoder)
        with torch.set_grad_enabled((not self.config.freeze_context) and self.training):
            return self.context.forward_prompts(
                prompts,
                encode_fn=self.tokenizer.encode,
                max_len=self.config.max_prompt_len,
                pad_id=self.tokenizer.pad_id,
                device=self.device_name,
            )

    def _format_context(
        self,
        prompt: str,
        *,
        design_md: str | None = None,
        slot_contract: list[str] | None = None,
        schema: str | None = None,
    ) -> str:
        if schema is None and self.config.schema_in_context:
            from slm_training.harnesses.quality import compact_schema_snippet

            schema = compact_schema_snippet(
                budget=min(600, self.config.design_md_budget)
            )
        return format_context_text(
            prompt,
            design_md if self.config.design_md_in_context else None,
            budget=self.config.design_md_budget,
            schema=schema,
            slot_contract=(
                slot_contract if self.config.slot_contract_in_context else None
            ),
        )

    def forward(self, batch: list[ExampleRecord]) -> float:
        return float(self.training_loss(batch).detach().cpu())

    def _state_rows(
        self, batch: list[ExampleRecord]
    ) -> tuple[list[str], list[list[TopologyNode]]]:
        prompts: list[str] = []
        rows: list[list[TopologyNode]] = []
        for record in batch:
            inventory = list(record.placeholders or extract_placeholders(record.openui))
            prompts.append(
                self._format_context(
                    record.prompt,
                    design_md=record.design_md,
                    slot_contract=inventory,
                )
            )
            gold = topology_from_openui(self.codec, record.openui, inventory, max_len=0)
            _apply_scope_contract(gold, record, self.config, self._rng)
            state = _corrupt_topology(gold, self.codec, self.config, self._rng)
            rows.append(_flatten(state)[: self.config.topology_max_nodes])
        return prompts, rows

    def training_loss(self, batch: list[ExampleRecord]) -> torch.Tensor:
        self.train()
        prompts, rows = self._state_rows(batch)
        ctx, ctx_pad = self._encode_context(prompts)
        ids = _pad_rows(
            [[node.production_id for node in row] for row in rows],
            self.codec.pad_id,
            self.device_name,
        )
        types = _pad_rows(
            [[NODE_TYPE_ID[node.node_type] for node in row] for row in rows],
            0,
            self.device_name,
        )
        parents = _pad_rows(
            [_parent_type_ids(row) for row in rows], len(NODE_TYPES), self.device_name
        )
        depths = _pad_rows(
            [[node.depth for node in row] for row in rows], 0, self.device_name
        )
        siblings = _pad_rows(
            [[node.sibling_index for node in row] for row in rows], 0, self.device_name
        )
        scope_buckets = _pad_rows(
            [[node.scope_bucket for node in row] for row in rows],
            0,
            self.device_name,
        )
        actions = _pad_rows(
            [[node.target_action for node in row] for row in rows],
            int(TopologyAction.KEEP),
            self.device_name,
        )
        prod_targets = _pad_rows(
            [
                [
                    node.target_production_id
                    if node.target_production_id is not None
                    else node.production_id
                    for node in row
                ]
                for row in rows
            ],
            self.codec.pad_id,
            self.device_name,
        )
        arities = _pad_rows(
            [[node.target_arity for node in row] for row in rows], 0, self.device_name
        )
        slots = _pad_rows(
            [[node.target_slot_id for node in row] for row in rows],
            self.codec.slot_none_id,
            self.device_name,
        )
        critic_targets = _pad_rows(
            [[node.critic_target for node in row] for row in rows],
            1.0,
            self.device_name,
        )
        valid = ids.ne(self.codec.pad_id)
        outputs = self.denoiser(
            ids,
            types,
            parents,
            depths,
            siblings,
            scope_buckets,
            ctx,
            structural=self.config.topology_structural_embeddings,
            ctx_pad_mask=ctx_pad,
        )
        (
            prod_logits,
            slot_logits,
            action_logits,
            arity_logits,
            critic,
            confidence,
            scope_summary,
            scope_gate,
            failure_cone,
        ) = outputs
        action_loss = F.cross_entropy(action_logits[valid], actions[valid])
        critic_loss = F.binary_cross_entropy(critic[valid], critic_targets[valid])
        expand = valid & actions.eq(int(TopologyAction.EXPAND))
        total = (
            self.config.action_loss_weight * action_loss
            + self.config.critic_loss_weight * critic_loss
        )
        prod_loss = total * 0.0
        arity_loss = total * 0.0
        slot_loss = total * 0.0
        confidence_loss = total * 0.0
        scope_summary_loss = total * 0.0
        scope_gate_loss = total * 0.0
        failure_cone_loss = total * 0.0
        production_accuracy = 0.0
        arity_accuracy = 0.0
        if expand.any():
            prod_loss = F.cross_entropy(prod_logits[expand], prod_targets[expand])
            arity_loss = F.cross_entropy(arity_logits[expand], arities[expand])
            slot_mask = expand & slots.ne(self.codec.slot_none_id)
            if slot_mask.any():
                slot_loss = F.cross_entropy(slot_logits[slot_mask], slots[slot_mask])
            correct = prod_logits.argmax(-1).eq(prod_targets) & arity_logits.argmax(
                -1
            ).eq(arities)
            confidence_loss = F.binary_cross_entropy(
                confidence[expand], correct[expand].float()
            )
            production_accuracy = float(
                prod_logits.argmax(-1)[expand]
                .eq(prod_targets[expand])
                .float()
                .mean()
                .detach()
                .cpu()
            )
            arity_accuracy = float(
                arity_logits.argmax(-1)[expand]
                .eq(arities[expand])
                .float()
                .mean()
                .detach()
                .cpu()
            )
            total = (
                total
                + self.config.production_loss_weight * prod_loss
                + self.config.arity_loss_weight * arity_loss
                + self.config.slot_loss_weight * slot_loss
                + self.config.confidence_loss_weight * confidence_loss
            )
        scoped = valid & scope_buckets.gt(0)
        if scoped.any() and scope_summary is not None and scope_gate is not None:
            summary_targets = _pad_vector_rows(
                [[node.scope_summary_target for node in row] for row in rows],
                4,
                self.device_name,
            )
            gate_targets = _pad_rows(
                [[node.scope_gate_target for node in row] for row in rows],
                1.0,
                self.device_name,
            )
            scope_summary_loss = F.mse_loss(
                scope_summary[scoped], summary_targets[scoped]
            )
            scope_gate_loss = F.binary_cross_entropy(
                scope_gate[scoped], gate_targets[scoped]
            )
            total = total + 0.25 * (scope_summary_loss + scope_gate_loss)
            if self.config.scope_local_oracle and failure_cone is not None:
                cone_targets = _pad_rows(
                    [[node.failure_cone_target for node in row] for row in rows],
                    0.0,
                    self.device_name,
                )
                failure_cone_loss = F.binary_cross_entropy(
                    failure_cone[scoped], cone_targets[scoped]
                )
                total = total + 0.25 * failure_cone_loss
        self.last_training_metrics = {
            "action_loss": float(action_loss.detach().cpu()),
            "production_loss": float(prod_loss.detach().cpu()),
            "arity_loss": float(arity_loss.detach().cpu()),
            "slot_loss": float(slot_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "confidence_loss": float(confidence_loss.detach().cpu()),
            "scope_summary_loss": float(scope_summary_loss.detach().cpu()),
            "scope_gate_loss": float(scope_gate_loss.detach().cpu()),
            "failure_cone_loss": float(failure_cone_loss.detach().cpu()),
            "production_accuracy": production_accuracy,
            "arity_accuracy": arity_accuracy,
            "active_nodes": float(expand.sum().detach().cpu()),
            "materialized_nodes": float(valid.sum().detach().cpu()),
        }
        return total

    @torch.inference_mode()
    def score_topology_targets(
        self, records: list[ExampleRecord]
    ) -> list[dict[str, float]]:
        """Teacher-forced topology-head metrics, kept separate from generation."""
        if not records:
            return []
        self.eval()
        prompts: list[str] = []
        rows: list[list[TopologyNode]] = []
        oov_rates: list[float] = []
        vocab_size = self.denoiser.core.tok.num_embeddings
        for index, record in enumerate(records):
            inventory = list(record.placeholders or extract_placeholders(record.openui))
            prompts.append(
                self._format_context(
                    record.prompt,
                    design_md=record.design_md,
                    slot_contract=inventory,
                )
            )
            # ProductionCodec.encode learns unseen productions. Evaluation must not
            # resize the checkpoint vocabulary, so retain their syntax in a throwaway
            # codec and map only model-facing IDs to the checkpoint's <unk> row.
            eval_codec = deepcopy(self.codec)
            gold = topology_from_openui(eval_codec, record.openui, inventory, max_len=0)
            _apply_scope_contract(
                gold,
                record,
                self.config,
                random.Random(self.config.seed + index + 9_000),
            )
            gold_ids = [node.production_id for node in _flatten(gold)]
            oov_rates.append(
                sum(not 0 <= pid < vocab_size for pid in gold_ids)
                / max(1, len(gold_ids))
            )
            row = _flatten(
                _corrupt_topology(
                    gold,
                    eval_codec,
                    self.config,
                    random.Random(self.config.seed + index + 10_000),
                )
            )[: self.config.topology_max_nodes]
            unk_id = int(getattr(self.codec, "unk_id", self.codec.mask_id))
            for node in row:
                if not 0 <= node.production_id < vocab_size:
                    node.production_id = unk_id
                if (
                    node.target_production_id is not None
                    and not 0 <= node.target_production_id < vocab_size
                ):
                    node.target_production_id = unk_id
            rows.append(row)
        ctx, ctx_pad = self._encode_context(prompts)
        ids = _pad_rows(
            [[node.production_id for node in row] for row in rows],
            self.codec.pad_id,
            self.device_name,
        )
        types = _pad_rows(
            [[NODE_TYPE_ID[node.node_type] for node in row] for row in rows],
            0,
            self.device_name,
        )
        parents = _pad_rows(
            [_parent_type_ids(row) for row in rows],
            len(NODE_TYPES),
            self.device_name,
        )
        depths = _pad_rows(
            [[node.depth for node in row] for row in rows], 0, self.device_name
        )
        siblings = _pad_rows(
            [[node.sibling_index for node in row] for row in rows],
            0,
            self.device_name,
        )
        scope_buckets = _pad_rows(
            [[node.scope_bucket for node in row] for row in rows],
            0,
            self.device_name,
        )
        outputs = self.denoiser(
            ids,
            types,
            parents,
            depths,
            siblings,
            scope_buckets,
            ctx,
            structural=self.config.topology_structural_embeddings,
            ctx_pad_mask=ctx_pad,
        )
        (
            prod_logits,
            _slot_logits,
            action_logits,
            arity_logits,
            critic,
            _confidence,
            _scope_summary,
            _scope_gate,
            _failure_cone,
        ) = outputs
        evidence: list[dict[str, float]] = []
        for batch_index, row in enumerate(rows):
            width = len(row)
            action_gold = torch.tensor(
                [node.target_action for node in row], device=self.device_name
            )
            action_pred = action_logits[batch_index, :width].argmax(-1)
            class_f1: list[float] = []
            for action in sorted(set(action_gold.tolist()) | set(action_pred.tolist())):
                predicted = action_pred.eq(action)
                expected = action_gold.eq(action)
                tp = int((predicted & expected).sum().item())
                fp = int((predicted & ~expected).sum().item())
                fn = int((~predicted & expected).sum().item())
                class_f1.append((2.0 * tp) / max(1, 2 * tp + fp + fn))
            expand = action_gold.eq(int(TopologyAction.EXPAND))
            if expand.any():
                prod_gold = torch.tensor(
                    [
                        node.target_production_id
                        if node.target_production_id is not None
                        else node.production_id
                        for node in row
                    ],
                    device=self.device_name,
                )
                arity_gold = torch.tensor(
                    [node.target_arity for node in row], device=self.device_name
                )
                production_accuracy = float(
                    prod_logits[batch_index, :width]
                    .argmax(-1)[expand]
                    .eq(prod_gold[expand])
                    .float()
                    .mean()
                    .item()
                )
                arity_accuracy = float(
                    arity_logits[batch_index, :width]
                    .argmax(-1)[expand]
                    .eq(arity_gold[expand])
                    .float()
                    .mean()
                    .item()
                )
            else:
                production_accuracy = 1.0
                arity_accuracy = 1.0
            critic_gold = torch.tensor(
                [node.critic_target for node in row], device=self.device_name
            )
            critic_ece = float(
                (critic[batch_index, :width] - critic_gold).abs().mean().item()
            )
            evidence.append(
                {
                    "action_macro_f1": sum(class_f1) / max(1, len(class_f1)),
                    "production_head_accuracy": production_accuracy,
                    "arity_head_accuracy": arity_accuracy,
                    "critic_ece": critic_ece,
                    "production_oov_rate": oov_rates[batch_index],
                }
            )
        return evidence

    def _legal_ids(self, node_type: str, *, leaf_only: bool = False) -> list[int]:
        specials = {self.codec.pad_id, self.codec.eos_id, self.codec.mask_id}
        result: list[int] = []
        for pid, token in self.codec.id_to_production.items():
            if pid in specials or token in {"-", "]", ";"}:
                continue
            if node_type == "document" and token not in {"<bos>", "!v0.5"}:
                continue
            if node_type == "statement" and token not in V05_MARKERS:
                continue
            if node_type == "expression" and token in V05_MARKERS | {"!v0.5", "<bos>"}:
                continue
            if leaf_only and _node_type(token) != "leaf":
                continue
            result.append(pid)
        return result or [
            self.codec.unk_id if hasattr(self.codec, "unk_id") else self.codec.mask_id
        ]

    def _selected_nodes(self, root: TopologyNode, phase: int) -> list[TopologyNode]:
        nodes = _flatten(root)[: self.config.topology_max_nodes]
        if (
            not self.config.topology_bounded_buffer
            or phase % max(1, self.config.topology_global_sync_interval) == 0
        ):
            return nodes
        active = [node for node in nodes if node.active][
            : self.config.topology_max_active
        ]
        wanted = {node.node_id for node in active}
        by_id = {node.node_id: node for node in nodes}
        for node in active:
            current = node
            while current.parent_id in by_id:
                current = by_id[current.parent_id]
                wanted.add(current.node_id)
                for sibling in current.children:
                    wanted.add(sibling.node_id)
        return [node for node in nodes if node.node_id in wanted][
            : self.config.topology_max_nodes
        ]

    def _decode_one(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        *,
        slot_inventory: list[str],
    ) -> tuple[str, dict[str, Any]]:
        root = TopologyNode(0, "document", self.codec.mask_id, active=True)
        next_id = 1
        stats: dict[str, Any] = {
            "phases": 0,
            "node_passes": 0,
            "active_peak": 1,
            "expansions": 0,
            "deletions": 0,
            "contractions": 0,
            "deferred": 0,
            "invalid_actions": 0,
            "critic_confidences": [],
            "budget_failure": None,
            "steps_to_first_valid": None,
        }
        for phase in range(self.config.topology_max_phases):
            all_nodes = _flatten(root)
            active_all = [node for node in all_nodes if node.active]
            stats["active_peak"] = max(stats["active_peak"], len(active_all))
            if not active_all:
                break
            if len(all_nodes) >= self.config.topology_max_nodes:
                stats["budget_failure"] = "node_budget_exhausted"
                break
            selected = self._selected_nodes(root, phase)
            active_selected = [node for node in selected if node.active][
                : self.config.topology_max_active
            ]
            if not active_selected:
                stats["budget_failure"] = "active_buffer_stalled"
                break
            ids = torch.tensor(
                [[node.production_id for node in selected]], device=self.device_name
            )
            types = torch.tensor(
                [[NODE_TYPE_ID[node.node_type] for node in selected]],
                device=self.device_name,
            )
            parents = torch.tensor(
                [_parent_type_ids(selected)], device=self.device_name
            )
            depths = torch.tensor(
                [[node.depth for node in selected]], device=self.device_name
            )
            siblings = torch.tensor(
                [[node.sibling_index for node in selected]], device=self.device_name
            )
            outputs = self.denoiser(
                ids,
                types,
                parents,
                depths,
                siblings,
                None,
                ctx,
                structural=self.config.topology_structural_embeddings,
                ctx_pad_mask=ctx_pad,
            )
            (
                prod_logits,
                slot_logits,
                action_logits,
                arity_logits,
                critic,
                confidence,
                _scope_summary,
                _scope_gate,
                _failure_cone,
            ) = outputs
            index_by_id = {node.node_id: index for index, node in enumerate(selected)}
            proposals: list[tuple[TopologyNode, int, int, int, float, float]] = []
            proposal_nodes = list(active_selected)
            if (
                self.config.topology_actions
                and phase % max(1, self.config.topology_global_sync_interval) == 0
            ):
                proposal_nodes.extend(
                    node for node in selected if not node.active and node.parent_id >= 0
                )
            for node in proposal_nodes:
                index = index_by_id[node.node_id]
                action = int(action_logits[0, index].argmax().item())
                if not node.active:
                    if (
                        action == int(TopologyAction.CONTRACT)
                        and float(critic[0, index].item())
                        < self.config.topology_contract_threshold
                        and not any(child.active for child in _flatten(node)[1:])
                    ):
                        proposals.append((node, -2, 0, 0, 1.0, 1.0))
                    continue
                if not self.config.topology_actions:
                    action = int(TopologyAction.EXPAND)
                if action == int(TopologyAction.DELETE) and node.parent_id >= 0:
                    proposals.append((node, -1, 0, 0, 1.0, 1.0))
                    continue
                if action not in {int(TopologyAction.EXPAND), int(TopologyAction.KEEP)}:
                    stats["invalid_actions"] += 1
                    stats["deferred"] += 1
                    continue
                legal = self._legal_ids(
                    node.node_type,
                    leaf_only=node.depth >= self.config.topology_max_depth,
                )
                legal_logits = prod_logits[0, index, legal]
                production_id = legal[int(legal_logits.argmax().item())]
                token = self.codec.id_to_production.get(production_id, "")
                arity = int(arity_logits[0, index].argmax().item())
                if node.node_type == "document":
                    arity = max(1, arity)
                elif node.node_type == "statement" and token == "=":
                    arity = 1
                elif _node_type(token) == "leaf":
                    arity = 0
                arity = min(arity, self.config.topology_max_arity)
                slot_id = int(slot_logits[0, index].argmax().item())
                validity = float(critic[0, index].item())
                conf = float(confidence[0, index].item())
                stats["critic_confidences"].append(validity)
                phase_fraction = phase / max(1, self.config.topology_max_phases - 1)
                accept_threshold = self.config.topology_accept_threshold * (
                    1.0 - phase_fraction
                )
                if (
                    self.config.topology_critic_decode
                    and min(validity, conf) < accept_threshold
                ):
                    stats["deferred"] += 1
                    continue
                proposals.append((node, production_id, arity, slot_id, validity, conf))
            proposed_children = sum(
                arity
                for _node, production_id, arity, *_rest in proposals
                if production_id >= 0
            )
            if len(all_nodes) + proposed_children > self.config.topology_max_nodes:
                stats["budget_failure"] = "node_budget_exhausted"
                break
            for (
                node,
                production_id,
                arity,
                slot_id,
                _validity,
                _confidence,
            ) in proposals:
                if production_id == -2:
                    node.production_id = self.codec.mask_id
                    node.slot_id = self.codec.slot_none_id
                    node.children = []
                    node.active = True
                    stats["contractions"] += 1
                    continue
                if production_id < 0:
                    parent = next(
                        (
                            item
                            for item in _flatten(root)
                            if item.node_id == node.parent_id
                        ),
                        None,
                    )
                    if parent is not None:
                        parent.children = [
                            child
                            for child in parent.children
                            if child.node_id != node.node_id
                        ]
                        stats["deletions"] += 1
                    continue
                node.production_id = production_id
                node.slot_id = slot_id
                token = self.codec.id_to_production.get(production_id, "")
                if node.node_type == "expression":
                    node.node_type = _node_type(token)
                node.active = False
                node.children = []
                for child_index in range(arity):
                    if node.node_type == "document":
                        child_type = "statement"
                    else:
                        child_type = "expression"
                    node.children.append(
                        TopologyNode(
                            next_id,
                            child_type,
                            self.codec.mask_id,
                            parent_id=node.node_id,
                            depth=node.depth + 1,
                            sibling_index=child_index,
                            active=True,
                        )
                    )
                    next_id += 1
                stats["expansions"] += 1
            _refresh_layout(root, preserve_ids=True)
            stats["node_passes"] += len(selected)
            stats["phases"] = phase + 1
        production_ids, slot_ids = _serialize_topology(self.codec, root)
        stats["candidate_productions"] = [
            self.codec.id_to_production.get(production_id, "<unknown>")
            for production_id in production_ids[:128]
        ]
        text = self.codec.decode(production_ids, slot_ids, slot_inventory).strip()
        valid = False
        validation_error: str | None = None
        if text:
            try:
                from slm_training.dsl.parser import validate

                validate(text)
                valid = True
            except Exception as exc:  # noqa: BLE001 - parser bridge exposes backend errors
                validation_error = str(exc)[:300]
                valid = False
        stats["candidate_preview"] = text[:500]
        stats["validation_error"] = validation_error
        confidences = list(stats.pop("critic_confidences"))
        label = 1.0 if valid else 0.0
        stats["critic_ece"] = (
            sum(abs(value - label) for value in confidences) / len(confidences)
            if confidences
            else 1.0
        )
        stats["action_macro_f1"] = max(
            0.0,
            1.0
            - stats["invalid_actions"]
            / max(1, stats["expansions"] + stats["invalid_actions"]),
        )
        stats["expand_contract_success"] = float(valid and not stats["budget_failure"])
        stats["steps_to_first_valid"] = stats["phases"] if valid else None
        stats["fixed_canvas_node_passes"] = self.config.max_target_len * max(
            1, stats["phases"]
        )
        stats["efficiency_score"] = min(
            1.0,
            stats["fixed_canvas_node_passes"] / (2.0 * max(1, stats["node_passes"])),
        )
        return (text if valid else ""), stats

    @torch.inference_mode()
    def generate_batch_requests(self, requests: list[GenerationRequest]) -> list[str]:
        self.eval()
        if not requests:
            return []
        prompts = [
            self._format_context(
                request.prompt,
                design_md=request.design_md,
                slot_contract=list(request.slot_contract or ()),
                schema=request.schema,
            )
            for request in requests
        ]
        ctx, ctx_pad = self._encode_context(prompts)
        outputs: list[str] = []
        self._generation_evidence = []
        for index, request in enumerate(requests):
            inventory = [
                value if value.startswith(":") else f":{value}"
                for value in (request.slot_contract or ())
            ]
            if not inventory:
                from slm_training.models.template_fill import inventory_from_prompt

                inventory = inventory_from_prompt(
                    request.prompt, request.design_md, heuristic=True
                )
            text, evidence = self._decode_one(
                ctx[index : index + 1],
                ctx_pad[index : index + 1],
                slot_inventory=inventory,
            )
            outputs.append(text)
            self._generation_evidence.append(evidence)
        return outputs

    def generate(self, prompt: str, gold: ExampleRecord | None = None) -> str:
        from slm_training.models.template_fill import inventory_from_prompt

        design_md = gold.design_md if gold is not None else None
        contract = tuple(inventory_from_prompt(prompt, design_md, heuristic=True))
        return self.generate_batch_requests(
            [
                GenerationRequest(
                    prompt=prompt, slot_contract=contract, design_md=design_md
                )
            ]
        )[0]

    def _codec_payload(self) -> dict[str, Any]:
        return {
            "codec_kind": "production"
            if type(self.codec).__name__ == "ProductionCodec"
            else "inline",
            "production_to_id": self.codec.production_to_id,
            "id_to_production": {
                str(key): value for key, value in self.codec.id_to_production.items()
            },
            "pad_id": self.codec.pad_id,
            "bos_id": self.codec.bos_id,
            "eos_id": self.codec.eos_id,
            "mask_id": self.codec.mask_id,
            "slot_none_id": self.codec.slot_none_id,
            "unk_id": getattr(self.codec, "unk_id", 4),
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "grammar_diffusion",
            "format_version": self.CHECKPOINT_FORMAT,
            "config": asdict(self.config),
            "codec": self._codec_payload(),
            "state_dict": {
                key: value.cpu() for key, value in self.state_dict().items()
            },
        }
        tokenizer_path = path.with_suffix(".tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "kind": "grammar_diffusion",
                    "format_version": self.CHECKPOINT_FORMAT,
                    "topology": True,
                    "tokenizer": tokenizer_path.name,
                    "vocab_size": self.tokenizer.vocab_size,
                    "production_vocab": self.codec.vocab_size,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        torch.save(payload, path)

    def load(self, path: Path | str) -> None:
        loaded = self.from_checkpoint(path, device=self.device_name)
        self.load_state_dict(loaded.state_dict(), strict=True)

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str,
        device: str | torch.device = "cpu",
    ) -> GrammarDiffusionModel:
        path = Path(path)
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("kind") != "grammar_diffusion":
            raise ValueError(
                f"checkpoint kind {payload.get('kind')!r} is not grammar_diffusion"
            )
        if int(payload.get("format_version") or 1) < cls.CHECKPOINT_FORMAT:
            raise ValueError(
                "fixed-canvas grammar checkpoint requires: "
                f"python -m scripts.migrate_checkpoint --checkpoint {path} --output <v2.pt>"
            )
        tokenizer_path = path.with_suffix(".tokenizer.json")
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"missing tokenizer next to checkpoint: {tokenizer_path}"
            )
        tokenizer = OpenUITokenizer.load(tokenizer_path)
        raw_config = dict(payload.get("config") or {})
        valid = set(GrammarDiffusionConfig.__dataclass_fields__)
        config = GrammarDiffusionConfig(
            **{key: value for key, value in raw_config.items() if key in valid}
        )
        codec = _restore_codec(payload.get("codec") or {})
        model = cls(tokenizer, codec, config, device)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model

    @classmethod
    def from_records(
        cls,
        records: list[ExampleRecord],
        config: GrammarDiffusionConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> GrammarDiffusionModel:
        codec = _load_production_codec([record.openui for record in records])
        tokenizer = OpenUITokenizer.build(
            [record.prompt for record in records]
            + [record.openui for record in records]
        )
        cfg = config or GrammarDiffusionConfig()
        cfg.max_prompt_len = max(
            cfg.max_prompt_len,
            max(
                (len(tokenizer.encode(record.prompt)) for record in records), default=16
            )
            + 4,
        )
        return cls(tokenizer, codec, cfg, device)


__all__ = [
    "GrammarDiffusionConfig",
    "GrammarDiffusionModel",
    "GrammarDenoiser",
    "InlineProductionCodec",
    "TopologyAction",
    "TopologyNode",
    "production_sequence_accuracy",
    "topology_arity_accuracy",
    "topology_from_openui",
]
