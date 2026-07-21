"""Concrete state adapters for the exact CTMC reference domains."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from slm_training.flow.reference.adapter import ActionRef, StateAdapter, StateRef


@dataclass(frozen=True)
class _ToyLayoutState:
    program_text: str
    depth: int


class ToyLayoutAdapter(StateAdapter):
    """Small bounded OpenUI layout AST with tree-edit actions."""

    def __init__(
        self,
        seed_programs: list[str],
        inventory: list[str],
        max_depth: int = 4,
        max_states: int = 1_000,
    ) -> None:
        from slm_training.models.tree_edit_diffusion import TreeEditSpace

        self.seed_programs = seed_programs
        self.inventory = inventory
        self.max_depth = max_depth
        self.max_states = max_states
        self.space = TreeEditSpace()
        self._cache: dict[str, list[tuple[ActionRef, StateRef]]] = {}

    @property
    def domain_id(self) -> str:
        return "toy_layout"

    def _fingerprint(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _state(self, text: str, depth: int) -> StateRef:
        return StateRef(
            _ToyLayoutState(text, depth),
            self._fingerprint(text),
        )

    def initial_states(self) -> list[StateRef]:
        return [self._state(p, 0) for p in self.seed_programs]

    def is_terminal(self, state: StateRef) -> bool:
        return self.legal_actions(state) == []

    def terminal_class(self, state: StateRef) -> str:
        return state.fingerprint

    def legal_actions(self, state: StateRef) -> list[ActionRef]:
        if state.fingerprint in self._cache:
            return [a for a, _ in self._cache[state.fingerprint]]
        from slm_training.dsl.parser import validate
        from slm_training.evals.tree_edit_scaling import _enumerate_edits
        from slm_training.models.tree_edit_diffusion import (
            ACTION_STOP,
            parse_statements,
            render_statements,
        )

        value = state.value
        text = value.program_text
        depth = value.depth
        if depth >= self.max_depth:
            return []
        try:
            validate(text)
        except Exception:  # noqa: BLE001
            return []
        statements = parse_statements(text)
        if statements is None:
            return []
        candidates = _enumerate_edits(statements, self.inventory, self.space)
        actions: list[ActionRef] = []
        seen: set[str] = set()
        for edit, result in candidates:
            if result is None:
                continue
            if edit.action == ACTION_STOP:
                continue
            next_text = render_statements(result)
            fp = self._fingerprint(next_text)
            if fp == state.fingerprint or fp in seen:
                continue
            seen.add(fp)
            actions.append(
                ActionRef(
                    {"action": edit.action, "stmt": edit.stmt, "comp": edit.comp, "slot": edit.slot},
                    f"edit_{edit.action}_{edit.stmt}_{edit.comp}_{edit.slot}",
                )
            )
        return actions

    def apply(self, state: StateRef, action: ActionRef) -> StateRef | None:
        if state.fingerprint in self._cache:
            for a, nxt in self._cache[state.fingerprint]:
                if a == action:
                    return nxt
        from slm_training.models.tree_edit_diffusion import (
            Edit,
            parse_statements,
            render_statements,
        )

        value = state.value
        statements = parse_statements(value.program_text)
        if statements is None:
            return None
        av = action.value
        edit = Edit(av["action"], av["stmt"], av["comp"], av["slot"])
        result = self.space.apply(statements, edit, self.inventory)
        if result is None:
            return None
        nxt = self._state(render_statements(result), value.depth + 1)
        self._cache.setdefault(state.fingerprint, []).append((action, nxt))
        return nxt

    def state_fingerprint(self, state_value: Any) -> str:
        return self._fingerprint(state_value.program_text)


@dataclass(frozen=True)
class _ChoiceState:
    stack: tuple[str, ...]
    emitted: tuple[str, ...]


class ChoiceSequenceAdapter(StateAdapter):
    """Bounded choice-sequence grammar with dynamic live actions."""

    def __init__(
        self,
        productions: dict[str, list[list[str]]] | None = None,
        max_length: int = 6,
        max_states: int = 1_000,
    ) -> None:
        if productions is None:
            productions = {
                "S": [["A", "B"], ["B", "A"], ["a"]],
                "A": [["a"], ["A", "A"]],
                "B": [["b"], ["B", "B"]],
            }
        self.productions = productions
        self.max_length = max_length
        self.max_states = max_states

    @property
    def domain_id(self) -> str:
        return "choice_sequence"

    def _fingerprint(self, stack: tuple[str, ...], emitted: tuple[str, ...]) -> str:
        return hashlib.sha256(
            json.dumps({"stack": stack, "emitted": emitted}, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _state(self, stack: tuple[str, ...], emitted: tuple[str, ...]) -> StateRef:
        return StateRef(_ChoiceState(stack, emitted), self._fingerprint(stack, emitted))

    def initial_states(self) -> list[StateRef]:
        return [self._state(("S",), ())]

    def is_terminal(self, state: StateRef) -> bool:
        return len(state.value.stack) == 0

    def terminal_class(self, state: StateRef) -> str:
        return "terminal_" + "".join(state.value.emitted)

    def legal_actions(self, state: StateRef) -> list[ActionRef]:
        stack = state.value.stack
        if not stack:
            return []
        top = stack[0]
        if top not in self.productions:
            # Terminal token: emit and pop.
            return [
                ActionRef(
                    {"emit": top},
                    f"emit_{top}",
                )
            ]
        actions: list[ActionRef] = []
        for idx, rhs in enumerate(self.productions[top]):
            actions.append(ActionRef({"lhs": top, "rhs": rhs}, f"expand_{top}_{idx}"))
        return actions

    def apply(self, state: StateRef, action: ActionRef) -> StateRef | None:
        stack = list(state.value.stack)
        emitted = list(state.value.emitted)
        if not stack:
            return None
        av = action.value
        if "emit" in av:
            token = av["emit"]
            if stack[0] != token:
                return None
            if len(emitted) + 1 > self.max_length:
                return None
            emitted.append(token)
            stack.pop(0)
            return self._state(tuple(stack), tuple(emitted))
        rhs = av["rhs"]
        lhs = av["lhs"]
        if stack[0] != lhs:
            return None
        if len(emitted) + len([x for x in rhs if x not in self.productions]) > self.max_length:
            return None
        stack.pop(0)
        stack = list(rhs) + stack
        return self._state(tuple(stack), tuple(emitted))

    def state_fingerprint(self, state_value: Any) -> str:
        return self._fingerprint(state_value.stack, state_value.emitted)


class CanonicalEditGraphAdapter(StateAdapter):
    """Canonical edit graph from a sketch seed to a target program.

    Uses SLM-188's edit algebra to enumerate legal canonical edits and replay
    them through ``apply_canonical_edit``.  Terminal states are the canonical
    target program (and any state with no forward edits under the budget).
    """

    def __init__(
        self,
        source_program: str,
        target_program: str,
        max_edits: int = 6,
        max_states: int = 2_000,
    ) -> None:
        from slm_training.dsl.canonicalize import canonical_fingerprint, canonicalize

        self.source_program = source_program
        self.target_program = target_program
        self.max_edits = max_edits
        self.max_states = max_states
        try:
            self.target_canonical = canonicalize(target_program, validate=True)
        except Exception:  # noqa: BLE001
            self.target_canonical = target_program
        self.target_fp = canonical_fingerprint(self.target_canonical)
        self._cache: dict[str, list[tuple[ActionRef, StateRef]]] = {}

    @property
    def domain_id(self) -> str:
        return "canonical_edit_graph"

    def _fingerprint(self, text: str) -> str:
        from slm_training.dsl.canonicalize import canonical_fingerprint

        return canonical_fingerprint(text)

    def _state(self, text: str, edit_count: int) -> StateRef:
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _EditState:
            program_text: str
            edit_count: int

        return StateRef(_EditState(text, edit_count), self._fingerprint(text))

    def initial_states(self) -> list[StateRef]:
        return [self._state(self.source_program, 0)]

    def is_terminal(self, state: StateRef) -> bool:
        return state.fingerprint == self.target_fp

    def terminal_class(self, state: StateRef) -> str:
        if state.fingerprint == self.target_fp:
            return self.target_fp
        return "nonterminal"

    def legal_actions(self, state: StateRef) -> list[ActionRef]:
        if state.value.edit_count >= self.max_edits or self.is_terminal(state):
            return []
        if state.fingerprint in self._cache:
            return [a for a, _ in self._cache[state.fingerprint]]
        from slm_training.dsl.placeholders import extract_placeholders
        from slm_training.harnesses.experiments.slm188_edit_algebra import (
            CanonicalEdit,
            apply_canonical_edit,
        )
        from slm_training.models.tree_edit_diffusion import (
            TreeEditSpace,
            parse_statements,
        )

        statements = parse_statements(state.value.program_text)
        if statements is None:
            return []
        inventory = [
            p if p.startswith(":") else f":{p}"
            for p in extract_placeholders(self.target_canonical)
        ]
        if not inventory:
            inventory = [":slot"]
        space = TreeEditSpace()
        components = list(space.components)
        actions: list[ActionRef] = []
        seen: set[str] = set()

        def add(edit: CanonicalEdit) -> None:
            nxt = apply_canonical_edit(state.value.program_text, edit)
            if nxt is None:
                return
            fp = self._fingerprint(nxt)
            if fp == state.fingerprint or fp in seen:
                return
            seen.add(fp)
            actions.append(
                ActionRef(
                    edit.to_dict(),
                    edit.edit_id or f"edit_{edit.target_name}_{edit.action}_{len(actions)}",
                )
            )

        for stmt in statements:
            # Production replacement.
            for comp in components:
                if comp == stmt.comp:
                    continue
                add(
                    CanonicalEdit(
                        edit_id=f"replace-{stmt.name}-{comp}",
                        action="ReplaceProduction",
                        target_name=stmt.name,
                        production=comp,
                    )
                )
            # Slot binding for leaves.
            if not stmt.has_list:
                for slot in inventory:
                    add(
                        CanonicalEdit(
                            edit_id=f"bind-{stmt.name}-{slot}",
                            action="BindSlotPointer",
                            target_name=stmt.name,
                            slot=slot,
                        )
                    )
            else:
                # Child list edits.
                candidates = [s.name for s in statements if s.name != stmt.name and s.name != "root"]
                for child in candidates:
                    if child not in stmt.children:
                        add(
                            CanonicalEdit(
                                edit_id=f"insert-child-{stmt.name}-{child}",
                                action="InsertChild",
                                target_name=stmt.name,
                                child_name=child,
                            )
                        )
                for child in stmt.children:
                    add(
                        CanonicalEdit(
                            edit_id=f"delete-child-{stmt.name}-{child}",
                            action="DeleteChild",
                            target_name=stmt.name,
                            child_name=child,
                        )
                    )
        return actions

    def apply(self, state: StateRef, action: ActionRef) -> StateRef | None:
        if state.fingerprint in self._cache:
            for a, nxt in self._cache[state.fingerprint]:
                if a == action:
                    return nxt
        from slm_training.harnesses.experiments.slm188_edit_algebra import (
            CanonicalEdit,
            apply_canonical_edit,
        )

        edit = CanonicalEdit.from_dict(action.value)
        nxt_text = apply_canonical_edit(state.value.program_text, edit)
        if nxt_text is None:
            return None
        nxt = self._state(nxt_text, state.value.edit_count + 1)
        self._cache.setdefault(state.fingerprint, []).append((action, nxt))
        return nxt

    def state_fingerprint(self, state_value: Any) -> str:
        return self._fingerprint(state_value.program_text)
