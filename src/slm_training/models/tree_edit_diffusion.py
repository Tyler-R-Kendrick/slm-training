"""Kapur-style tree-edit diffusion baseline (D3 / SLM-31, X22).

Faithful to the *mechanism* of "Diffusion On Syntax Trees For Program
Synthesis" (Kapur, Jenner, Russell; NeurIPS 2024, arXiv:2405.20519):

- forward noise = a chain of small **validity-preserving** program edits —
  every intermediate state parses (unlike the X-series' typed mask nodes);
- reverse = a policy network supervised on the **inverse edit** of the last
  mutation in the chain;
- decode = **value-guided beam search** over edit sequences, starting from a
  minimal valid program, so every emitted candidate is valid by construction.

Stated boundary (research-lineage.md): the paper's observation channel is a
rendered image compared against the target render; this domain has no target
render at generation time, so the policy/value nets condition on the prompt
context instead. Everything else (all-valid state space, inverse-edit
supervision, value-guided search) follows the paper.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_training.dsl.parser import validate, validate_output
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.plugin import GenerationRequest
from slm_training.models.blocks import RMSNorm, TransformerBlock
from slm_training.models.context import (
    ScratchContextEncoder,
    build_context_encoder,
    is_hf_context,
)
from slm_training.models.tokenizer import OpenUITokenizer
from slm_training.models.twotower import format_context_text

# Bounded edit-action space (Kapur's "small tree edits", specialized to the
# OpenUI statement grammar). STOP freezes the current state as the output.
ACTION_STOP = 0
ACTION_REPLACE = 1  # swap the component type of one statement
ACTION_ADD = 2  # add a fresh leaf statement + reference it from a container
ACTION_REMOVE = 3  # remove one leaf statement and its references
# SLM-305 (LAR2-01) extended valid-state edit language. Every new action is
# bounded, deterministic, validity-preserving (re-validated through the real
# parser before acceptance), and invertible; preconditions are documented at
# each branch of ``TreeEditSpace.apply``.
ACTION_ADD_CONTAINER = 4  # insert an empty container into a container's children
ACTION_REMOVE_CONTAINER = 5  # safe inverse: remove an empty/leaf-only container subtree
ACTION_INSERT_SUBTREE = 6  # transactional: declare container+leaf subtree and reference it
ACTION_REPLACE_SUBTREE = 7  # replace a canonical one-leaf container subtree's leaf
ACTION_INSERT_STATEMENT = 8  # insert a canonical V0.5 state/query/mutation statement
ACTION_REPLACE_STATEMENT = 9  # swap one canonical V0.5 statement for another
ACTION_BIND_PLACEHOLDER = 10  # (re)bind a leaf's slot to an inventory placeholder
N_ACTIONS = 11

MAX_STMTS = 24
MAX_SLOTS = 16

# Leaf components take a single placeholder argument; containers hold a
# child-reference list. Derived from the fixed grammar rather than hardcoded
# beyond this split so the action space stays grammar-coupled.
LEAF_COMPONENTS = ("TextContent", "Button", "Image", "TextInput")
CONTAINER_COMPONENTS = ("Stack", "Card", "Form")

# Canonical rest texts for containers minted by ADD_CONTAINER / INSERT_SUBTREE.
# Fixed candidate set (indexed by ``Edit.target``) so minted containers are
# deterministic and the reachability invariants can reason about them exactly.
CONTAINER_RESTS: tuple[str, ...] = (', "column"', "")
CONTAINER_REST = CONTAINER_RESTS[0]

# V0.5 statement component names (state/query/mutation/action pack forms).
V05_COMPONENTS = ("Query", "Mutation", "Action", "State", "Resource")

# Bounded canonical V0.5 statement templates: (component, canonical arg text).
# Construction is canonical-AST-backed: each inserted/replaced line is built
# from this structured spec and fragment-validated through the canonical
# grammar (``validate_output(..., kind="statement")``) before acceptance —
# never regex string surgery on existing program text.
V05_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("Query", '"tool", {arg: $x}, {default: []}, 15'),
    ("Mutation", '"tool", {arg: $x}'),
)


def v05_template_index(stmt: Statement) -> int | None:
    """Index of the canonical V0.5 template ``stmt`` instantiates, else None.

    Only canonical-template statements are REPLACE_STATEMENT-editable, so the
    inverse edit (restore the old template) is always expressible.
    """
    if stmt.has_list:
        return None
    rest = stmt.rest.strip()
    for index, (comp, args) in enumerate(V05_TEMPLATES):
        if stmt.comp == comp and rest == args:
            return index
    return None

_STMT_RE = re.compile(r"^(?P<name>\w+)\s*=\s*(?P<comp>\w+)\((?P<args>.*)\)\s*$")


def _grammar_components() -> tuple[str, ...]:
    """Component inventory from the fixed lexer grammar vocabulary — deterministic
    and corpus-independent, so checkpoint round-trips keep head sizes stable."""
    try:
        from slm_training.models.dsl_tokenizer import DSLNativeTokenizer, TokenKind

        tok = DSLNativeTokenizer.build()
        comps = sorted(
            token
            for token, tid in tok.token_to_id.items()
            if tok.kind_of(tid) == TokenKind.COMPONENT
        )
    except Exception:  # noqa: BLE001
        comps = []
    merged = list(dict.fromkeys([*LEAF_COMPONENTS, *CONTAINER_COMPONENTS, *comps]))
    return tuple(merged)


@dataclass
class Statement:
    """One `name = Component(args)` line in structural form."""

    name: str
    comp: str
    children: list[str]
    rest: str  # raw arg text after the child list (or the full args for leaves)
    has_list: bool

    def render(self) -> str:
        if self.has_list:
            inner = ", ".join(self.children)
            rest = self.rest
            return f"{self.name} = {self.comp}([{inner}]{rest})"
        return f"{self.name} = {self.comp}({self.rest})"


def parse_statements(source: str) -> list[Statement] | None:
    """Structural parse of a canonical program; None when a line defies the
    `name = Comp(...)` shape (those programs are skipped, never mutated).

    V0.5 statement lines (Query/Mutation/Action/State/Resource) are owned by
    the canonical grammar: the line is fragment-validated through
    ``validate_output(..., kind="statement")`` rather than the regex alone.
    UI statement lines keep the legacy structural split so
    ``Statement.render()`` stays byte-stable for existing fixture programs.
    """
    statements: list[Statement] = []
    for line in source.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _STMT_RE.match(line)
        if match is None:
            return None
        if match.group("comp") in V05_COMPONENTS:
            try:
                validate_output(line, kind="statement")
            except Exception:  # noqa: BLE001
                return None
        args = match.group("args")
        if args.startswith("["):
            depth = 0
            end = -1
            for index, ch in enumerate(args):
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        end = index
                        break
            if end < 0:
                return None
            inner = args[1:end].strip()
            children = [c.strip() for c in inner.split(",") if c.strip()]
            statements.append(
                Statement(
                    name=match.group("name"),
                    comp=match.group("comp"),
                    children=children,
                    rest=args[end + 1 :],
                    has_list=True,
                )
            )
        else:
            statements.append(
                Statement(
                    name=match.group("name"),
                    comp=match.group("comp"),
                    children=[],
                    rest=args,
                    has_list=False,
                )
            )
    return statements or None


def render_statements(statements: list[Statement]) -> str:
    return "\n".join(stmt.render() for stmt in statements)


@lru_cache(maxsize=65536)
def _is_valid(source: str) -> bool:
    try:
        validate(source)
        return True
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class Edit:
    """One bounded edit: action + statement index + component + slot.

    SLM-305: ``target``/``payload`` are NEW DEFAULTED fields only, so old
    pickles and comparisons keep working. ``payload`` carries the leaf
    component index (INSERT_SUBTREE / REPLACE_SUBTREE) or the canonical V0.5
    template index (INSERT_STATEMENT / REPLACE_STATEMENT); ``target`` is
    reserved for secondary statement addressing.
    """

    action: int
    stmt: int = 0
    comp: int = 0
    slot: int = 0
    target: int = 0
    payload: int = 0


class TreeEditSpace:
    """Applies and inverts bounded edits on valid statement lists.

    Every application is re-verified through the real parser; an edit that
    produces an invalid program is reported as inapplicable (fail closed) —
    the all-valid-states invariant is the point of the baseline.
    """

    def __init__(self, components: tuple[str, ...] | None = None) -> None:
        if components is None:
            components = _grammar_components()
        self.components: tuple[str, ...] = tuple(components)
        self.comp_index = {c: i for i, c in enumerate(self.components)}

    def fresh_name(self, statements: list[Statement]) -> str:
        taken = {s.name for s in statements}
        for i in range(len(statements) + 8):
            name = f"n{i}"
            if name not in taken:
                return name
        return f"n{len(statements)}x"

    def fresh_v05_name(self, statements: list[Statement], comp: str) -> str:
        """Fresh V0.5 statement name with the conventional pack prefix."""
        prefix = {"Query": "q", "Mutation": "m"}.get(comp, "r")
        taken = {s.name for s in statements}
        for i in range(len(statements) + 8):
            name = f"{prefix}{i}"
            if name not in taken:
                return name
        return f"{prefix}{len(statements)}x"

    @staticmethod
    def _placeholder(inventory: list[str], slot: int) -> str:
        placeholder = inventory[slot]
        if not placeholder.startswith(":"):
            placeholder = f":{placeholder}"
        return placeholder

    def apply(
        self,
        statements: list[Statement],
        edit: Edit,
        inventory: list[str],
        pre_validate: Callable[[list[Statement]], bool] | None = None,
    ) -> list[Statement] | None:
        """Apply one edit; None when inapplicable or invalid (fail closed).

        ``pre_validate`` is an optional cheap rejection hook invoked on the
        mutated statement list just before parser re-validation (used by the
        reachability analyzer to skip already-visited states). It can only
        reject, never accept: every accepted state is still re-validated
        through the real parser.
        """
        if edit.action == ACTION_STOP:
            return [Statement(**vars(s)) for s in statements]
        working = [
            Statement(s.name, s.comp, list(s.children), s.rest, s.has_list)
            for s in statements
        ]
        if edit.action == ACTION_REPLACE:
            if not (0 <= edit.stmt < len(working) and 0 <= edit.comp < len(self.components)):
                return None
            target = working[edit.stmt]
            new_comp = self.components[edit.comp]
            leaf_like = not target.has_list
            if leaf_like != (new_comp in LEAF_COMPONENTS):
                return None
            if target.comp == new_comp:
                return None
            target.comp = new_comp
        elif edit.action == ACTION_ADD:
            if not (0 <= edit.stmt < len(working) and 0 <= edit.comp < len(self.components)):
                return None
            if len(working) >= MAX_STMTS:
                return None
            parent = working[edit.stmt]
            comp = self.components[edit.comp]
            if not parent.has_list or comp not in LEAF_COMPONENTS:
                return None
            if not inventory or not (0 <= edit.slot < len(inventory)):
                return None
            placeholder = inventory[edit.slot]
            if not placeholder.startswith(":"):
                placeholder = f":{placeholder}"
            name = self.fresh_name(working)
            parent.children.append(name)
            working.append(
                Statement(
                    name=name,
                    comp=comp,
                    children=[],
                    rest=json.dumps(placeholder, ensure_ascii=False),
                    has_list=False,
                )
            )
        elif edit.action == ACTION_REMOVE:
            if not (0 <= edit.stmt < len(working)):
                return None
            target = working[edit.stmt]
            if target.has_list or target.name == "root":
                return None
            referenced = False
            for other in working:
                if target.name in other.children:
                    other.children = [c for c in other.children if c != target.name]
                    referenced = True
            if not referenced and target.comp not in V05_COMPONENTS:
                # Unreferenced UI leaves stay immutable (old behavior); V0.5
                # pack statements are unreferenced by construction and are
                # removable (inverse of INSERT_STATEMENT).
                return None
            working = [s for s in working if s.name != target.name]
        elif edit.action == ACTION_ADD_CONTAINER:
            # Preconditions: parent is a container, MAX_STMTS bound, comp is a
            # container. The minted container starts EMPTY (leaf-only subtree)
            # so REMOVE_CONTAINER is an exact safe inverse.
            if not (0 <= edit.stmt < len(working) and 0 <= edit.comp < len(self.components)):
                return None
            if len(working) >= MAX_STMTS:
                return None
            parent = working[edit.stmt]
            comp = self.components[edit.comp]
            if not parent.has_list or comp not in CONTAINER_COMPONENTS:
                return None
            if not (0 <= edit.target < len(CONTAINER_RESTS)):
                return None
            name = self.fresh_name(working)
            parent.children.append(name)
            working.append(
                Statement(
                    name=name, comp=comp, children=[],
                    rest=CONTAINER_RESTS[edit.target], has_list=True,
                )
            )
        elif edit.action == ACTION_REMOVE_CONTAINER:
            # Safe inverse of ADD_CONTAINER / INSERT_SUBTREE. Preconditions:
            # target is a non-root container, referenced from some parent, and
            # its subtree is leaf-only (no nested containers) — exactly the
            # shapes the container-creating actions mint, so removal restores
            # the prior state exactly. Leaf children are dropped with it.
            if not (0 <= edit.stmt < len(working)):
                return None
            target = working[edit.stmt]
            if not target.has_list or target.name == "root":
                return None
            by_name = {s.name: s for s in working}
            if any(
                by_name.get(child) is None or by_name[child].has_list
                for child in target.children
            ):
                return None
            if not any(target.name in other.children for other in working):
                return None
            drop = {target.name, *target.children}
            working = [
                Statement(
                    s.name,
                    s.comp,
                    [c for c in s.children if c not in drop],
                    s.rest,
                    s.has_list,
                )
                for s in working
                if s.name not in drop
            ]
        elif edit.action == ACTION_INSERT_SUBTREE:
            # Transactional declare-plus-reference: mint a small canonical
            # subtree (container root + one leaf child bound to an inventory
            # slot) and reference the root from an existing container, all
            # re-validated as one step. Preconditions: parent is a container,
            # comp is a container, payload indexes a leaf component, slot is
            # in inventory, MAX_STMTS bound. Inverse: REMOVE_CONTAINER.
            if not (0 <= edit.stmt < len(working) and 0 <= edit.comp < len(self.components)):
                return None
            if len(working) + 2 > MAX_STMTS:
                return None
            parent = working[edit.stmt]
            root_comp = self.components[edit.comp]
            if not parent.has_list or root_comp not in CONTAINER_COMPONENTS:
                return None
            if not (0 <= edit.payload < len(self.components)):
                return None
            leaf_comp = self.components[edit.payload]
            if leaf_comp not in LEAF_COMPONENTS:
                return None
            if not inventory or not (0 <= edit.slot < min(len(inventory), MAX_SLOTS)):
                return None
            if not (0 <= edit.target < len(CONTAINER_RESTS)):
                return None
            placeholder = self._placeholder(inventory, edit.slot)
            cname = self.fresh_name(working)
            lname = self.fresh_name(
                [*working, Statement(cname, root_comp, [], CONTAINER_REST, True)]
            )
            parent.children.append(cname)
            working.append(
                Statement(cname, root_comp, [lname], CONTAINER_RESTS[edit.target], True)
            )
            working.append(
                Statement(
                    lname, leaf_comp, [],
                    json.dumps(placeholder, ensure_ascii=False), False,
                )
            )
        elif edit.action == ACTION_REPLACE_SUBTREE:
            # Replace the leaf of a canonical one-leaf container subtree
            # (same root kind, so the subtree shape is preserved).
            # Preconditions: target container has exactly one child which is a
            # leaf, payload indexes a leaf component, slot is in inventory.
            # The small-canonical-subtree precondition keeps the inverse
            # (restore old leaf comp + slot) expressible as the same action.
            if not (0 <= edit.stmt < len(working)):
                return None
            target = working[edit.stmt]
            if not target.has_list or len(target.children) != 1:
                return None
            by_name = {s.name: s for s in working}
            leaf = by_name.get(target.children[0])
            if leaf is None or leaf.has_list:
                return None
            if not (0 <= edit.payload < len(self.components)):
                return None
            leaf_comp = self.components[edit.payload]
            if leaf_comp not in LEAF_COMPONENTS:
                return None
            if not inventory or not (0 <= edit.slot < min(len(inventory), MAX_SLOTS)):
                return None
            placeholder = self._placeholder(inventory, edit.slot)
            leaf.comp = leaf_comp
            leaf.rest = json.dumps(placeholder, ensure_ascii=False)
        elif edit.action == ACTION_INSERT_STATEMENT:
            # Insert a canonical V0.5 state/query/mutation statement built
            # from the structured template spec and fragment-validated through
            # the canonical grammar (never regex string surgery).
            # Preconditions: MAX_STMTS bound, payload indexes V05_TEMPLATES.
            # Inverse: REMOVE (pack statements are unreferenced).
            if len(working) >= MAX_STMTS:
                return None
            if not (0 <= edit.payload < len(V05_TEMPLATES)):
                return None
            comp, args = V05_TEMPLATES[edit.payload]
            candidate = Statement(self.fresh_v05_name(working, comp), comp, [], args, False)
            try:
                validate_output(candidate.render(), kind="statement")
            except Exception:  # noqa: BLE001
                return None
            working.append(candidate)
        elif edit.action == ACTION_REPLACE_STATEMENT:
            # Swap one canonical V0.5 statement for another template.
            # Preconditions: target instantiates a known canonical template
            # (so the inverse — restore the old template — is expressible),
            # payload indexes V05_TEMPLATES, and the swap is a real change.
            if not (0 <= edit.stmt < len(working)):
                return None
            target = working[edit.stmt]
            if v05_template_index(target) is None:
                return None
            if not (0 <= edit.payload < len(V05_TEMPLATES)):
                return None
            comp, args = V05_TEMPLATES[edit.payload]
            if target.comp == comp and target.rest.strip() == args:
                return None
            candidate = Statement(target.name, comp, [], args, False)
            try:
                validate_output(candidate.render(), kind="statement")
            except Exception:  # noqa: BLE001
                return None
            working[edit.stmt] = candidate
        elif edit.action == ACTION_BIND_PLACEHOLDER:
            # Transactional declaration-plus-reference: (re)bind a leaf's slot
            # to an inventory placeholder. Preconditions: target is a UI leaf
            # (non-root, non-container, leaf component), slot in inventory.
            # Inverse: BIND_PLACEHOLDER with the old slot index.
            if not (0 <= edit.stmt < len(working)):
                return None
            target = working[edit.stmt]
            if target.has_list or target.name == "root":
                return None
            if target.comp not in LEAF_COMPONENTS:
                return None
            if not inventory or not (0 <= edit.slot < min(len(inventory), MAX_SLOTS)):
                return None
            placeholder = self._placeholder(inventory, edit.slot)
            target.rest = json.dumps(placeholder, ensure_ascii=False)
        else:
            return None
        if pre_validate is not None and not pre_validate(working):
            return None
        rendered = render_statements(working)
        if not _is_valid(rendered):
            return None
        return working

    def sample_mutation(
        self,
        statements: list[Statement],
        inventory: list[str],
        rng: random.Random,
    ) -> tuple[list[Statement], Edit] | None:
        """One random validity-preserving mutation and the *inverse* edit
        (the supervised repair step) — Kapur's forward process."""
        for _ in range(12):
            kind = rng.choice(
                (
                    ACTION_REPLACE,
                    ACTION_ADD,
                    ACTION_REMOVE,
                    ACTION_ADD_CONTAINER,
                    ACTION_REMOVE_CONTAINER,
                    ACTION_INSERT_SUBTREE,
                    ACTION_REPLACE_SUBTREE,
                    ACTION_INSERT_STATEMENT,
                    ACTION_REPLACE_STATEMENT,
                    ACTION_BIND_PLACEHOLDER,
                )
            )
            if kind == ACTION_REPLACE:
                idx = rng.randrange(len(statements))
                stmt = statements[idx]
                if stmt.comp not in self.comp_index:
                    # Unknown surface (e.g. runtime builtin): never mutated,
                    # so the inverse edit is always expressible.
                    continue
                pool = LEAF_COMPONENTS if not stmt.has_list else CONTAINER_COMPONENTS
                choices = [c for c in pool if c != stmt.comp and c in self.comp_index]
                if not choices:
                    continue
                new_comp = rng.choice(choices)
                mutation = Edit(ACTION_REPLACE, idx, self.comp_index[new_comp])
                mutated = self.apply(statements, mutation, inventory)
                if mutated is None:
                    continue
                inverse = Edit(ACTION_REPLACE, idx, self.comp_index[stmt.comp])
                return mutated, inverse
            if kind == ACTION_ADD:
                # Mutation = spurious leaf; inverse = REMOVE it.
                parents = [
                    i for i, s in enumerate(statements) if s.has_list
                ]
                if not parents or not inventory or len(statements) >= MAX_STMTS:
                    continue
                parent_idx = rng.choice(parents)
                comp = rng.choice(
                    [c for c in LEAF_COMPONENTS if c in self.comp_index]
                )
                slot = rng.randrange(min(len(inventory), MAX_SLOTS))
                mutation = Edit(ACTION_ADD, parent_idx, self.comp_index[comp], slot)
                mutated = self.apply(statements, mutation, inventory)
                if mutated is None:
                    continue
                inverse = Edit(ACTION_REMOVE, len(mutated) - 1)
                return mutated, inverse
            if kind == ACTION_ADD_CONTAINER:
                # Mutation = empty container under a container; inverse =
                # REMOVE_CONTAINER (exact, the minted subtree is empty).
                parents = [i for i, s in enumerate(statements) if s.has_list]
                if not parents or len(statements) >= MAX_STMTS:
                    continue
                parent_idx = rng.choice(parents)
                comp = rng.choice(
                    [c for c in CONTAINER_COMPONENTS if c in self.comp_index]
                )
                mutation = Edit(
                    ACTION_ADD_CONTAINER, parent_idx, self.comp_index[comp],
                    target=rng.randrange(len(CONTAINER_RESTS)),
                )
                mutated = self.apply(statements, mutation, inventory)
                if mutated is None:
                    continue
                inverse = Edit(ACTION_REMOVE_CONTAINER, len(mutated) - 1)
                return mutated, inverse
            if kind == ACTION_REMOVE_CONTAINER:
                # Mutation = remove an empty/leaf-only container subtree;
                # inverse = ADD_CONTAINER (empty) or INSERT_SUBTREE (one leaf).
                by_name = {s.name: s for s in statements}
                removable = [
                    i
                    for i, s in enumerate(statements)
                    if s.has_list
                    and s.name != "root"
                    and any(s.name in o.children for o in statements)
                    and all(
                        by_name.get(c) is not None and not by_name[c].has_list
                        for c in s.children
                    )
                ]
                if not removable:
                    continue
                idx = rng.choice(removable)
                victim = statements[idx]
                if victim.comp not in self.comp_index:
                    continue
                if victim.rest not in CONTAINER_RESTS:
                    # The inverse (re-minting this subtree) is only expressible
                    # for canonically-rested containers; skip, never fake it.
                    continue
                rest_idx = CONTAINER_RESTS.index(victim.rest)
                parent_name = next(
                    o.name for o in statements if victim.name in o.children
                )
                inverse: Edit | None = None
                if not victim.children:
                    inverse = Edit(
                        ACTION_ADD_CONTAINER, 0, self.comp_index[victim.comp],
                        target=rest_idx,
                    )
                elif len(victim.children) == 1:
                    leaf = by_name[victim.children[0]]
                    slot = self._leaf_slot_index(leaf, inventory)
                    if leaf.comp in self.comp_index and slot is not None:
                        inverse = Edit(
                            ACTION_INSERT_SUBTREE,
                            0,
                            self.comp_index[victim.comp],
                            slot,
                            target=rest_idx,
                            payload=self.comp_index[leaf.comp],
                        )
                if inverse is None:
                    continue
                mutation = Edit(ACTION_REMOVE_CONTAINER, idx)
                mutated = self.apply(statements, mutation, inventory)
                if mutated is None:
                    continue
                parent_idx = next(
                    i for i, s in enumerate(mutated) if s.name == parent_name
                )
                inverse = Edit(
                    inverse.action, parent_idx, inverse.comp, inverse.slot,
                    target=inverse.target, payload=inverse.payload,
                )
                return mutated, inverse
            if kind == ACTION_INSERT_SUBTREE:
                # Mutation = transactional container+leaf subtree; inverse =
                # REMOVE_CONTAINER of the minted root.
                parents = [i for i, s in enumerate(statements) if s.has_list]
                if not parents or not inventory or len(statements) + 2 > MAX_STMTS:
                    continue
                parent_idx = rng.choice(parents)
                root_comp = rng.choice(
                    [c for c in CONTAINER_COMPONENTS if c in self.comp_index]
                )
                leaf_comp = rng.choice(
                    [c for c in LEAF_COMPONENTS if c in self.comp_index]
                )
                slot = rng.randrange(min(len(inventory), MAX_SLOTS))
                mutation = Edit(
                    ACTION_INSERT_SUBTREE,
                    parent_idx,
                    self.comp_index[root_comp],
                    slot,
                    target=rng.randrange(len(CONTAINER_RESTS)),
                    payload=self.comp_index[leaf_comp],
                )
                mutated = self.apply(statements, mutation, inventory)
                if mutated is None:
                    continue
                inverse = Edit(ACTION_REMOVE_CONTAINER, len(mutated) - 2)
                return mutated, inverse
            if kind == ACTION_REPLACE_SUBTREE:
                # Mutation = replace the leaf of a canonical one-leaf container
                # subtree; inverse = REPLACE_SUBTREE restoring old comp+slot.
                by_name = {s.name: s for s in statements}
                candidates = [
                    i
                    for i, s in enumerate(statements)
                    if s.has_list
                    and len(s.children) == 1
                    and by_name.get(s.children[0]) is not None
                    and not by_name[s.children[0]].has_list
                ]
                if not candidates or not inventory:
                    continue
                idx = rng.choice(candidates)
                leaf = by_name[statements[idx].children[0]]
                old_slot = self._leaf_slot_index(leaf, inventory)
                if leaf.comp not in self.comp_index or old_slot is None:
                    continue
                choices = [
                    c for c in LEAF_COMPONENTS if c in self.comp_index
                ]
                slot = rng.randrange(min(len(inventory), MAX_SLOTS))
                new_comp = rng.choice(choices)
                if new_comp == leaf.comp and slot == old_slot:
                    continue
                mutation = Edit(
                    ACTION_REPLACE_SUBTREE, idx, slot=slot,
                    payload=self.comp_index[new_comp],
                )
                mutated = self.apply(statements, mutation, inventory)
                if mutated is None:
                    continue
                inverse = Edit(
                    ACTION_REPLACE_SUBTREE, idx, slot=old_slot,
                    payload=self.comp_index[leaf.comp],
                )
                return mutated, inverse
            if kind == ACTION_INSERT_STATEMENT:
                # Mutation = canonical V0.5 statement; inverse = REMOVE it.
                if len(statements) >= MAX_STMTS:
                    continue
                payload = rng.randrange(len(V05_TEMPLATES))
                mutation = Edit(ACTION_INSERT_STATEMENT, payload=payload)
                mutated = self.apply(statements, mutation, inventory)
                if mutated is None:
                    continue
                inverse = Edit(ACTION_REMOVE, len(mutated) - 1)
                return mutated, inverse
            if kind == ACTION_REPLACE_STATEMENT:
                # Mutation = swap canonical V0.5 template; inverse = swap back.
                candidates = [
                    i for i, s in enumerate(statements)
                    if v05_template_index(s) is not None
                ]
                if not candidates:
                    continue
                idx = rng.choice(candidates)
                old_payload = v05_template_index(statements[idx])
                assert old_payload is not None
                choices = [t for t in range(len(V05_TEMPLATES)) if t != old_payload]
                if not choices:
                    continue
                mutation = Edit(
                    ACTION_REPLACE_STATEMENT, idx, payload=rng.choice(choices)
                )
                mutated = self.apply(statements, mutation, inventory)
                if mutated is None:
                    continue
                inverse = Edit(ACTION_REPLACE_STATEMENT, idx, payload=old_payload)
                return mutated, inverse
            if kind == ACTION_BIND_PLACEHOLDER:
                # Mutation = rebind a leaf's slot; inverse = bind the old slot.
                bindable = [
                    i
                    for i, s in enumerate(statements)
                    if not s.has_list
                    and s.name != "root"
                    and s.comp in LEAF_COMPONENTS
                ]
                if not bindable or not inventory:
                    continue
                idx = rng.choice(bindable)
                old_slot = self._leaf_slot_index(statements[idx], inventory)
                if old_slot is None or len(inventory) < 2:
                    continue
                choices = [
                    s for s in range(min(len(inventory), MAX_SLOTS)) if s != old_slot
                ]
                if not choices:
                    continue
                mutation = Edit(ACTION_BIND_PLACEHOLDER, idx, slot=rng.choice(choices))
                mutated = self.apply(statements, mutation, inventory)
                if mutated is None:
                    continue
                inverse = Edit(ACTION_BIND_PLACEHOLDER, idx, slot=old_slot)
                return mutated, inverse
            # Mutation = remove a leaf; inverse = ADD it back.
            removable = [
                i
                for i, s in enumerate(statements)
                if not s.has_list
                and s.name != "root"
                and any(s.name in o.children for o in statements)
            ]
            if not removable:
                continue
            idx = rng.choice(removable)
            victim = statements[idx]
            parent_idx = next(
                (
                    i
                    for i, o in enumerate(statements)
                    if victim.name in o.children
                ),
                None,
            )
            if parent_idx is None or victim.comp not in self.comp_index:
                continue
            body = victim.rest.strip()
            slot = None
            if body.startswith('"') or body.startswith("'"):
                try:
                    literal = json.loads(body) if body.startswith('"') else body[1:-1]
                except Exception:  # noqa: BLE001
                    literal = None
                if isinstance(literal, str) and literal in inventory:
                    slot = inventory.index(literal)
            if slot is None or slot >= MAX_SLOTS:
                continue
            mutation = Edit(ACTION_REMOVE, idx)
            mutated = self.apply(statements, mutation, inventory)
            if mutated is None:
                continue
            adjusted_parent = parent_idx if parent_idx < idx else parent_idx - 1
            inverse = Edit(
                ACTION_ADD, adjusted_parent, self.comp_index[victim.comp], slot
            )
            return mutated, inverse
        return None

    def _leaf_slot_index(self, stmt: Statement, inventory: list[str]) -> int | None:
        """Inventory index of the placeholder bound by a leaf, else None."""
        body = stmt.rest.strip()
        if not (body.startswith('"') or body.startswith("'")):
            return None
        try:
            literal = json.loads(body) if body.startswith('"') else body[1:-1]
        except Exception:  # noqa: BLE001
            return None
        if isinstance(literal, str) and literal in inventory:
            index = inventory.index(literal)
            return index if index < MAX_SLOTS else None
        return None


@dataclass
class TreeEditDiffusionConfig:
    d_model: int = 96
    n_heads: int = 4
    context_layers: int = 2
    denoiser_layers: int = 3
    dropout: float = 0.1
    max_prompt_len: int = 192
    max_state_len: int = 256
    max_chain: int = 4
    beam_width: int = 4
    expand_per_state: int = 4
    max_search_steps: int = 12
    context_backend: str = "scratch"
    hf_model_name: str | None = None
    freeze_context: bool = True
    local_files_only: bool = False
    design_md_in_context: bool = True
    design_md_budget: int = 1200
    schema_in_context: bool = False
    slot_contract_in_context: bool = True
    seed: int = 0


class TreeEditPolicy(nn.Module):
    """Transformer over program tokens with prompt cross-attention; policy
    heads factorize the bounded edit and a value head scores the state."""

    def __init__(
        self, vocab_size: int, cfg: TreeEditDiffusionConfig, n_components: int
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_state_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                cfg.d_model, cfg.n_heads, dropout=cfg.dropout, cross_attn=True
            )
            for _ in range(cfg.denoiser_layers)
        )
        self.norm = RMSNorm(cfg.d_model)
        self.action_head = nn.Linear(cfg.d_model, N_ACTIONS)
        self.stmt_head = nn.Linear(cfg.d_model, MAX_STMTS)
        self.comp_head = nn.Linear(cfg.d_model, n_components)
        self.slot_head = nn.Linear(cfg.d_model, MAX_SLOTS)
        self.value_head = nn.Linear(cfg.d_model, 1)

    def forward(
        self,
        state_ids: torch.Tensor,
        pad_id: int,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pad_mask = state_ids.eq(pad_id)
        positions = torch.arange(state_ids.shape[1], device=state_ids.device)
        hidden = self.embed(state_ids) + self.pos(positions)[None, :, :]
        for block in self.blocks:
            hidden = block(hidden, pad_mask, ctx=ctx, ctx_pad_mask=ctx_pad)
        hidden = self.norm(hidden)
        keep = (~pad_mask).float().unsqueeze(-1)
        pooled = (hidden * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        return {
            "action": self.action_head(pooled),
            "stmt": self.stmt_head(pooled),
            "comp": self.comp_head(pooled),
            "slot": self.slot_head(pooled),
            "value": torch.sigmoid(self.value_head(pooled)).squeeze(-1),
        }


class TreeEditDiffusionModel(nn.Module):
    """Prompt-conditioned Kapur-style edit policy + value search (X22)."""

    # Format 2 (SLM-305): action_head grew to N_ACTIONS=11 with the extended
    # edit language. Format-1 checkpoints fail closed here; warm-start them
    # via ``checkpoint_migrate.migrate_tree_edit_checkpoint``.
    CHECKPOINT_FORMAT = 2

    def __init__(
        self,
        tokenizer: OpenUITokenizer,
        config: TreeEditDiffusionConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config or TreeEditDiffusionConfig()
        self.device_name = str(device)
        self.space = TreeEditSpace()
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
        self.policy = TreeEditPolicy(
            tokenizer.vocab_size, self.config, len(self.space.components)
        )
        self._rng = random.Random(self.config.seed)
        self.last_training_metrics: dict[str, float] = {}
        self._generation_evidence: list[dict[str, Any]] = []
        self.to(device)

    # --- shared plumbing -------------------------------------------------

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def consume_generation_evidence(self) -> list[dict[str, Any]]:
        evidence, self._generation_evidence = self._generation_evidence, []
        return evidence

    def _encode_context(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if is_hf_context(self.context):
            return self.context.forward_prompts(
                prompts, max_len=self.config.max_prompt_len, device=self.device_name
            )
        assert isinstance(self.context, ScratchContextEncoder)
        with torch.set_grad_enabled(
            (not self.config.freeze_context) and self.training
        ):
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
    ) -> str:
        return format_context_text(
            prompt,
            design_md if self.config.design_md_in_context else None,
            budget=self.config.design_md_budget,
            schema=None,
            slot_contract=(
                slot_contract if self.config.slot_contract_in_context else None
            ),
        )

    def _state_batch(self, sources: list[str]) -> torch.Tensor:
        rows = [
            self.tokenizer.encode(text)[: self.config.max_state_len]
            for text in sources
        ]
        width = max((len(r) for r in rows), default=1)
        batch = torch.full(
            (len(rows), width),
            self.tokenizer.pad_id,
            dtype=torch.long,
            device=self.device_name,
        )
        for i, row in enumerate(rows):
            if row:
                batch[i, : len(row)] = torch.tensor(
                    row, dtype=torch.long, device=self.device_name
                )
        return batch

    # --- training ---------------------------------------------------------

    def forward(self, batch: list[ExampleRecord]) -> float:
        return float(self.training_loss(batch).detach().cpu())

    def training_loss(self, batch: list[ExampleRecord]) -> torch.Tensor:
        prompts: list[str] = []
        states: list[str] = []
        targets: list[Edit] = []
        values: list[float] = []
        skipped = 0
        for record in batch:
            source = (record.openui or "").strip()
            statements = parse_statements(source) if source else None
            if statements is None or not _is_valid(source):
                skipped += 1
                continue
            inventory = [
                p if p.startswith(":") else f":{p}"
                for p in (record.placeholders or extract_placeholders(source))
            ][:MAX_SLOTS]
            prompt = self._format_context(
                record.prompt,
                design_md=record.design_md,
                slot_contract=inventory,
            )
            if self._rng.random() < 0.2:
                # Clean state: the correct move is STOP with full value.
                prompts.append(prompt)
                states.append(source)
                targets.append(Edit(ACTION_STOP))
                values.append(1.0)
                continue
            k = self._rng.randint(1, self.config.max_chain)
            current = statements
            inverse: Edit | None = None
            applied = 0
            for _ in range(k):
                step = self.space.sample_mutation(current, inventory, self._rng)
                if step is None:
                    break
                current, inverse = step
                applied += 1
            if inverse is None:
                skipped += 1
                continue
            prompts.append(prompt)
            states.append(render_statements(current))
            targets.append(inverse)
            values.append(1.0 - applied / float(self.config.max_chain + 1))
        if not states:
            return torch.zeros((), device=self.device_name, requires_grad=True)
        ctx, ctx_pad = self._encode_context(prompts)
        out = self.policy(
            self._state_batch(states), self.tokenizer.pad_id, ctx, ctx_pad
        )
        device = self.device_name
        action_t = torch.tensor([e.action for e in targets], device=device)
        loss = F.cross_entropy(out["action"], action_t)
        losses = {"action": float(loss.detach().cpu())}
        stmt_rows = [i for i, e in enumerate(targets) if e.action != ACTION_STOP]
        if stmt_rows:
            idx = torch.tensor(stmt_rows, device=device)
            stmt_t = torch.tensor(
                [min(targets[i].stmt, MAX_STMTS - 1) for i in stmt_rows],
                device=device,
            )
            stmt_loss = F.cross_entropy(out["stmt"][idx], stmt_t)
            loss = loss + stmt_loss
            losses["stmt"] = float(stmt_loss.detach().cpu())
        comp_rows = [
            i
            for i, e in enumerate(targets)
            if e.action
            in {ACTION_REPLACE, ACTION_ADD, ACTION_ADD_CONTAINER, ACTION_INSERT_SUBTREE}
        ]
        if comp_rows:
            idx = torch.tensor(comp_rows, device=device)
            comp_t = torch.tensor([targets[i].comp for i in comp_rows], device=device)
            comp_loss = F.cross_entropy(out["comp"][idx], comp_t)
            loss = loss + comp_loss
            losses["comp"] = float(comp_loss.detach().cpu())
        slot_rows = [
            i
            for i, e in enumerate(targets)
            if e.action
            in {
                ACTION_ADD,
                ACTION_INSERT_SUBTREE,
                ACTION_REPLACE_SUBTREE,
                ACTION_BIND_PLACEHOLDER,
            }
        ]
        if slot_rows:
            idx = torch.tensor(slot_rows, device=device)
            slot_t = torch.tensor(
                [min(targets[i].slot, MAX_SLOTS - 1) for i in slot_rows],
                device=device,
            )
            slot_loss = F.cross_entropy(out["slot"][idx], slot_t)
            loss = loss + slot_loss
            losses["slot"] = float(slot_loss.detach().cpu())
        value_t = torch.tensor(values, device=device, dtype=out["value"].dtype)
        value_loss = F.mse_loss(out["value"], value_t)
        loss = loss + value_loss
        losses["value"] = float(value_loss.detach().cpu())
        losses["skipped"] = float(skipped)
        self.last_training_metrics = losses
        return loss

    # --- decode: value-guided beam search over edits ----------------------

    def _seed_state(self, inventory: list[str]) -> list[Statement] | None:
        """Minimal valid program to start the search from."""
        slot = inventory[0] if inventory else ":content.body"
        if not slot.startswith(":"):
            slot = f":{slot}"
        candidates = [
            (
                'root = Stack([n0], "column")\n'
                f"n0 = TextContent({json.dumps(slot, ensure_ascii=False)})"
            ),
            'root = Stack([], "column")',
        ]
        for text in candidates:
            statements = parse_statements(text)
            if statements is not None and _is_valid(text):
                return statements
        return None

    def _enumerate_edits(
        self, out: dict[str, torch.Tensor], row: int, n_stmts: int, n_slots: int
    ) -> list[tuple[float, Edit]]:
        action_lp = F.log_softmax(out["action"][row], dim=-1)
        stmt_lp = F.log_softmax(out["stmt"][row][: max(n_stmts, 1)], dim=-1)
        comp_lp = F.log_softmax(out["comp"][row], dim=-1)
        slot_lp = F.log_softmax(out["slot"][row][: max(n_slots, 1)], dim=-1)
        scored: list[tuple[float, Edit]] = [
            (float(action_lp[ACTION_STOP]), Edit(ACTION_STOP))
        ]
        n_comp = comp_lp.shape[0]
        leaf_comps = [
            i for i, c in enumerate(self.space.components) if c in LEAF_COMPONENTS
        ]
        for stmt in range(min(n_stmts, MAX_STMTS)):
            base = float(stmt_lp[stmt])
            for comp in range(n_comp):
                scored.append(
                    (
                        float(action_lp[ACTION_REPLACE]) + base + float(comp_lp[comp]),
                        Edit(ACTION_REPLACE, stmt, comp),
                    )
                )
                for rest_idx in range(len(CONTAINER_RESTS)):
                    scored.append(
                        (
                            float(action_lp[ACTION_ADD_CONTAINER])
                            + base
                            + float(comp_lp[comp]),
                            Edit(ACTION_ADD_CONTAINER, stmt, comp, target=rest_idx),
                        )
                    )
                for slot in range(min(n_slots, MAX_SLOTS)):
                    slot_score = float(slot_lp[slot])
                    scored.append(
                        (
                            float(action_lp[ACTION_ADD])
                            + base
                            + float(comp_lp[comp])
                            + slot_score,
                            Edit(ACTION_ADD, stmt, comp, slot),
                        )
                    )
                    scored.append(
                        (
                            float(action_lp[ACTION_BIND_PLACEHOLDER])
                            + base
                            + slot_score,
                            Edit(ACTION_BIND_PLACEHOLDER, stmt, slot=slot),
                        )
                    )
                    for payload in leaf_comps:
                        for rest_idx in range(len(CONTAINER_RESTS)):
                            scored.append(
                                (
                                    float(action_lp[ACTION_INSERT_SUBTREE])
                                    + base
                                    + float(comp_lp[comp])
                                    + slot_score,
                                    Edit(ACTION_INSERT_SUBTREE, stmt, comp, slot,
                                         target=rest_idx, payload=payload),
                                )
                            )
                        scored.append(
                            (
                                float(action_lp[ACTION_REPLACE_SUBTREE])
                                + base
                                + slot_score,
                                Edit(ACTION_REPLACE_SUBTREE, stmt, slot=slot,
                                     payload=payload),
                            )
                        )
            scored.append(
                (float(action_lp[ACTION_REMOVE]) + base, Edit(ACTION_REMOVE, stmt))
            )
            scored.append(
                (
                    float(action_lp[ACTION_REMOVE_CONTAINER]) + base,
                    Edit(ACTION_REMOVE_CONTAINER, stmt),
                )
            )
            for payload in range(len(V05_TEMPLATES)):
                scored.append(
                    (
                        float(action_lp[ACTION_REPLACE_STATEMENT]) + base,
                        Edit(ACTION_REPLACE_STATEMENT, stmt, payload=payload),
                    )
                )
        for payload in range(len(V05_TEMPLATES)):
            scored.append(
                (
                    float(action_lp[ACTION_INSERT_STATEMENT]),
                    Edit(ACTION_INSERT_STATEMENT, payload=payload),
                )
            )
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored

    @torch.no_grad()
    def _decode_one(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        inventory: list[str],
    ) -> tuple[str, dict[str, Any]]:
        seed = self._seed_state(inventory)
        if seed is None:
            return "", {"failure": "no_valid_seed"}
        beam: list[tuple[float, list[Statement], bool]] = [(0.0, seed, False)]
        evidence: dict[str, Any] = {"steps": 0, "expansions": 0, "kind": "tree_edit"}
        for _ in range(self.config.max_search_steps):
            live = [entry for entry in beam if not entry[2]]
            if not live:
                break
            sources = [render_statements(s) for _, s, _ in live]
            out = self.policy(
                self._state_batch(sources),
                self.tokenizer.pad_id,
                ctx.expand(len(sources), -1, -1),
                ctx_pad.expand(len(sources), -1),
            )
            next_beam: list[tuple[float, list[Statement], bool]] = [
                entry for entry in beam if entry[2]
            ]
            seen: set[str] = {
                render_statements(s) for _, s, frozen in next_beam if frozen
            }
            for row, (_, statements, _) in enumerate(live):
                candidates = self._enumerate_edits(
                    out, row, len(statements), len(inventory)
                )
                expanded = 0
                for _, edit in candidates:
                    if expanded >= self.config.expand_per_state:
                        break
                    if edit.action == ACTION_STOP:
                        text = render_statements(statements)
                        if text not in seen:
                            seen.add(text)
                            next_beam.append(
                                (float(out["value"][row]), statements, True)
                            )
                        expanded += 1
                        continue
                    child = self.space.apply(statements, edit, inventory)
                    if child is None:
                        continue
                    text = render_statements(child)
                    if text in seen:
                        continue
                    seen.add(text)
                    next_beam.append((float(out["value"][row]), child, False))
                    expanded += 1
                    evidence["expansions"] += 1
            if not next_beam:
                break
            # Re-score unfrozen children by the value head (Kapur's search
            # signal) and keep the top beam_width states.
            unfrozen = [entry for entry in next_beam if not entry[2]]
            if unfrozen:
                sources = [render_statements(s) for _, s, _ in unfrozen]
                rescore = self.policy(
                    self._state_batch(sources),
                    self.tokenizer.pad_id,
                    ctx.expand(len(sources), -1, -1),
                    ctx_pad.expand(len(sources), -1),
                )
                rescored = [
                    (float(rescore["value"][i]), entry[1], False)
                    for i, entry in enumerate(unfrozen)
                ]
            else:
                rescored = []
            frozen = [entry for entry in next_beam if entry[2]]
            beam = sorted(
                frozen + rescored, key=lambda entry: entry[0], reverse=True
            )[: self.config.beam_width]
            evidence["steps"] += 1
            if all(entry[2] for entry in beam):
                break
        best = max(beam, key=lambda entry: entry[0])
        evidence["value"] = float(best[0])
        evidence["frozen"] = bool(best[2])
        return render_statements(best[1]), evidence

    def generate_batch_requests(self, requests: list[GenerationRequest]) -> list[str]:
        self.eval()
        if not requests:
            return []
        prompts = [
            self._format_context(
                request.prompt,
                design_md=request.design_md,
                slot_contract=list(request.slot_contract or ()),
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
                inventory[:MAX_SLOTS],
            )
            try:
                program = validate(text)
                text = (program.serialized or text).strip()
                evidence["fallback_used"] = False
            except Exception:  # noqa: BLE001
                fallback = "root = Separator()"
                program = validate(fallback)
                text = (program.serialized or fallback).strip()
                evidence["fallback_used"] = True
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

    # --- persistence -------------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_path = path.with_suffix(".tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        payload = {
            "kind": "tree_edit_diffusion",
            "format_version": self.CHECKPOINT_FORMAT,
            "config": asdict(self.config),
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
        }
        parameter_count = int(sum(p.numel() for p in self.parameters()))
        path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "kind": "tree_edit_diffusion",
                    "format_version": self.CHECKPOINT_FORMAT,
                    "tokenizer": tokenizer_path.name,
                    "vocab_size": self.tokenizer.vocab_size,
                    "parameter_count": parameter_count,
                    "serialized_weight_bytes": parameter_count * 4,
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
        cls, path: Path | str, device: str | torch.device = "cpu"
    ) -> TreeEditDiffusionModel:
        path = Path(path)
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("kind") != "tree_edit_diffusion":
            raise ValueError(
                f"checkpoint kind {payload.get('kind')!r} is not tree_edit_diffusion"
            )
        format_version = int(payload.get("format_version") or 1)
        if format_version != cls.CHECKPOINT_FORMAT:
            raise ValueError(
                f"tree_edit_diffusion checkpoint format_version={format_version} "
                f"is not supported (expected {cls.CHECKPOINT_FORMAT}); warm-start "
                "older checkpoints via "
                "slm_training.models.checkpoint_migrate.migrate_tree_edit_checkpoint"
            )
        tokenizer = OpenUITokenizer.load(path.with_suffix(".tokenizer.json"))
        config = TreeEditDiffusionConfig(**payload["config"])
        model = cls(tokenizer, config=config, device=device)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model

    @classmethod
    def from_records(
        cls,
        records: list[ExampleRecord],
        config: TreeEditDiffusionConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> TreeEditDiffusionModel:
        texts = [r.prompt for r in records] + [r.openui for r in records if r.openui]
        tokenizer = OpenUITokenizer.build(texts)
        return cls(tokenizer, config=config, device=device)
