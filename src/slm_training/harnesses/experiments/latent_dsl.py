"""G3 (SLM-47): latent-DSL generator — task → grammar → instantiated pack.

Grammar Prompting (Wang et al., NeurIPS 2023) synthesizes a minimal grammar
per task and constrains a frozen large model with it. This harness goes the
program's way instead: the synthesized grammar instantiates a **full DSL
pack** (F1 contract) — grammar → backend/oracle → typed corpus generator →
tiny scratch model — so the reasoning substrate for G4 is a trained model
under verifier constraints, not a prompt.

Fixture-grade: grammar synthesis is deterministic template instantiation
over a typed :class:`TaskSpec` (no model in the loop yet); the end-goal
meta-model that *proposes* TaskSpecs trains later on G5 traces. Everything
downstream of the spec reuses existing owners (``LarkFileBackend``,
``DslPack``, ``TwoTowerModel``).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slm_training.dsl.grammar.backends import get_backend, register_backend
from slm_training.dsl.grammar.backends.lark_backend import LarkFileBackend
from slm_training.dsl.pack import DslPack, PlaceholderPolicy, ScopeRules, register_pack
from slm_training.dsl.schema import ExampleRecord

_GRAMMAR_TEMPLATE = """\
// Latent DSL synthesized for task: {task_id}
// {description}

start: statement*

statement: NAME "=" expr (_NL)*

?expr: call
     | list
     | STRING
     | NAME -> ref

call: NAME "(" [arg_list] ")"
arg_list: expr ("," expr)*
list: "[" [expr ("," expr)*] "]"

NAME: /[A-Za-z_][A-Za-z0-9_]*/
STRING: /"(?:\\\\.|[^"\\\\])*"/

_NL: /(\\r?\\n)+/
%import common.WS_INLINE
%ignore WS_INLINE
"""


@dataclass(frozen=True)
class TaskSpec:
    """Typed task description — the latent DSL's entire input."""

    task_id: str
    description: str
    # Component name -> ordered prop names; the first list-typed prop (by
    # convention "children") nests, string props route content slots.
    components: dict[str, tuple[str, ...]] = field(default_factory=dict)
    content_slots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.task_id):
            raise ValueError("task_id must be a short lowercase slug")
        if not self.components:
            raise ValueError("a latent DSL needs at least one component")
        for name in self.components:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"component name {name!r} must be a lower slug")
        for slot in self.content_slots:
            if not slot.startswith(":"):
                raise ValueError(f"content slot {slot!r} must start with ':'")


def synthesize_grammar(task: TaskSpec) -> str:
    """Deterministic minimal grammar for the task (Lark surface)."""
    return _GRAMMAR_TEMPLATE.format(
        task_id=task.task_id, description=task.description
    )


def instantiate_pack(task: TaskSpec, root: Path | str) -> DslPack:
    """task → grammar file → registered backend → registered DslPack."""
    root_dir = Path(root)
    root_dir.mkdir(parents=True, exist_ok=True)
    grammar_path = root_dir / f"latent_{task.task_id}.lark"
    grammar_text = synthesize_grammar(task)
    grammar_path.write_text(grammar_text, encoding="utf-8")

    dsl_id = f"latent-{task.task_id}"
    backend = LarkFileBackend(
        dsl_id=dsl_id,
        grammar_path=grammar_path,
        description=f"Latent DSL for task {task.task_id}",
        prop_order={name: list(props) for name, props in task.components.items()},
        structural_extras=frozenset(
            {"root", *task.components, "(", ")", "[", "]", ",", "=", '"'}
        ),
    )
    register_backend(backend)

    grammar_sha = hashlib.sha256(grammar_text.encode("utf-8")).hexdigest()

    def _canonicalize(source: str) -> str:
        program = backend.validate(source)
        return backend.serialize(program)

    def _fingerprint(source: str) -> str:
        return hashlib.sha256(_canonicalize(source).encode("utf-8")).hexdigest()

    pack = DslPack(
        id=dsl_id,
        grammar=dsl_id,
        canonicalize=_canonicalize,
        canonical_fingerprint=_fingerprint,
        validity_oracle=backend.validate,
        scope_rules=ScopeRules(
            bind_encodings=("name",),
            reference_legality=(
                "lark parse: refs must resolve to bound statement names"
            ),
        ),
        placeholder_policy=PlaceholderPolicy(
            is_placeholder=lambda value: value.startswith(":"),
            extract=lambda source: sorted(set(re.findall(r":[a-z][a-z0-9_.]*", source))),
            merge=lambda *groups: sorted({item for g in groups for item in g}),
        ),
        contract_id=lambda: f"{dsl_id}-{grammar_sha[:12]}",
    )
    register_pack(pack)
    return pack


def generate_corpus(task: TaskSpec, *, per_component: int = 4) -> list[ExampleRecord]:
    """Deterministic typed corpus: every component exercised with content slots."""
    records: list[ExampleRecord] = []
    slots = list(task.content_slots) or [":content.value"]
    names = sorted(task.components)
    container = next(
        (n for n in names if (task.components[n] or ("",))[0] == "children"),
        None,
    )
    for index, name in enumerate(names):
        for variant in range(per_component):
            slot = slots[(index + variant) % len(slots)]
            if name == container:
                # The container is exercised as the root wrapping a leaf.
                leaf_name = next(n for n in names if n != container) if len(names) > 1 else None
                if leaf_name is None:
                    program = f"root = {name}([])"
                else:
                    program = (
                        f"root = {name}([item])\n"
                        f'item = {leaf_name}("{slot}")'
                    )
            elif container is not None:
                program = (
                    f"root = {container}([item])\n"
                    f'item = {name}("{slot}")'
                )
            else:
                program = f'root = {name}("{slot}")'
            records.append(
                ExampleRecord(
                    id=f"{task.task_id}-{name}-{variant}",
                    prompt=f"{task.description} using {name} for {slot}",
                    openui=program,
                    placeholders=[slot],
                    source="latent-dsl-generator",
                    meta={"dsl": f"latent-{task.task_id}", "component": name},
                )
            )
    return records


def run_fixture(
    task: TaskSpec,
    root: Path | str,
    *,
    train_steps: int = 3,
) -> dict[str, Any]:
    """The G3 verify gate: task in → pack out → scratch model → oracle-valid.

    Returns an honest summary (counts, oracle verdicts); makes no ship claim.
    """
    import torch

    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    pack = instantiate_pack(task, root)
    records = generate_corpus(task)
    for record in records:
        pack.validity_oracle(record.openui)  # generator outputs must be legal

    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            seed=0,
            gen_steps=2,
        ),
        device="cpu",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses: list[float] = []
    for _ in range(max(1, train_steps)):
        optimizer.zero_grad()
        loss = model.training_loss(records[: min(4, len(records))])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    decoded = model.generate(records[0].prompt, grammar_constrained=False)
    oracle_valid = False
    if isinstance(decoded, str) and decoded.strip():
        try:
            pack.validity_oracle(decoded)
            oracle_valid = True
        except Exception:  # noqa: BLE001
            oracle_valid = False
    return {
        "task_id": task.task_id,
        "pack_id": pack.id,
        "contract_id": pack.contract_id(),
        "records": len(records),
        "train_losses": losses,
        "decoded_nonempty": bool(decoded and decoded.strip()),
        "decoded_oracle_valid": oracle_valid,
        "backend_available": get_backend(pack.grammar).available(),
    }


__all__ = [
    "TaskSpec",
    "generate_corpus",
    "instantiate_pack",
    "run_fixture",
    "synthesize_grammar",
]
