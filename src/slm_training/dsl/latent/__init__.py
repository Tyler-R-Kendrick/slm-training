"""G3 (SLM-47): the latent-DSL generator — task → grammar → instantiated pack.

Grammar-Prompting-style (Wang et al., NeurIPS 2023, `arXiv:2305.19234
<https://arxiv.org/abs/2305.19234>`_) but with the "an LLM synthesizes a
grammar for the task" step **deterministically stubbed**: given a typed task
inventory (:class:`LatentTaskSpec`), we mechanically emit a minimal ``.lark``
and assemble a full :class:`~slm_training.dsl.pack.DslPack`. No frozen large
model is prompted; no per-task model is trained. What this proves is that pack
*instantiation* is mechanical — which is exactly the substrate Track G4 needs.

Honest scope (full accounting in ``docs/design/latent-dsl-generator.md``):

* **FILLED** — ``backend`` (a :class:`LarkFileBackend` over the synthesized
  grammar), ``scope_extractor``, ``placeholder_policy``, ``prop_order``,
  ``incremental_engine``.
* **HONEST-None** — ``canonicalize``, ``oracle``, ``corpus_generator``. These
  are ``None`` for the same reason ``toy-layout`` leaves them ``None``: the
  production codec's lexical layer is still OpenUI-shaped (uppercase-call), so
  a real canonicalizer/oracle is blocked on F2's codec generalization
  (``docs/design/dsl-pack-contract.md`` ~99-104). ``reward_label`` is
  ``"parse_only"``.
* **DEFERRED** — the real LLM task→grammar step; a synthesized component JSON
  schema for :class:`ProgramGenerator` (needed for oracle + generation, not for
  round-trip); a per-task trained model (the "tiny trained model per task"
  end-goal claim). None of these are faked here.

The synthesized grammar reuses the ``toy_layout`` skeleton
(``start``/``statement``/``call``/``list``/``STRING``/``NAME``): component names
ride the generic ``NAME`` call rule rather than becoming per-name terminals, so
the generic Lark transformer yields ElementNode ASTs and the grammar-generic
scope extractor works with no extra wiring. Restricting ``call`` to only the
declared component names (a terminal alternation) is a deferred refinement — it
would drop the free generic-transformer fastpath.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from slm_training.data.contract import canonical_slot_contract
from slm_training.dsl.grammar.backends import (
    GRAMMARS_DIR,
    available_backends,
    register_backend,
)
from slm_training.dsl.grammar.backends.lark_backend import LarkFileBackend
from slm_training.dsl.pack import (
    DslPack,
    PlaceholderPolicy,
    list_packs,
    register_pack,
)
from slm_training.dsl.placeholders import CONTENT_PROPS, PLACEHOLDER_RE

# Default home for generated grammars. A dedicated subdir keeps them clearly
# machine-written and out of the way of the hand-authored grammars. The file
# must persist for the backend's lazy read (``LarkFileBackend`` reads the path
# on first ``parse``); callers that want an ephemeral lifetime pass a
# ``grammars_dir`` under a temp path (e.g. pytest ``tmp_path``).
_DEFAULT_GRAMMARS_DIR = GRAMMARS_DIR / "latent"

# The toy_layout CFG skeleton, verbatim. A raw string so the ``.lark`` file
# gets the exact backslash escapes Lark expects in the STRING terminal.
_SKELETON = r"""start: statement*

statement: NAME "=" expr (_NL)*

?expr: call
     | list
     | STRING
     | NAME -> ref

call: NAME "(" [arg_list] ")"
arg_list: expr ("," expr)*
list: "[" [expr ("," expr)*] "]"

NAME: /[A-Za-z_][A-Za-z0-9_]*/
STRING: /"(?:\\.|[^"\\])*"/

_NL: /(\r?\n)+/
%import common.WS_INLINE
%ignore WS_INLINE
"""


def _slug(text: str) -> str:
    """Deterministic lowercase-kebab identifier fragment."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "task"


@dataclass(frozen=True)
class LatentComponent:
    """One production in a task's inventory: a call head and its prop order.

    ``props`` doubles as the positional-argument names (its length is the
    component's arity) and as the pack's ``prop_order`` entry for the head.
    """

    name: str
    props: tuple[str, ...] = ()

    @classmethod
    def coerce(cls, value: Any) -> "LatentComponent":
        if isinstance(value, LatentComponent):
            return value
        if isinstance(value, Mapping):
            return cls(
                name=str(value["name"]),
                props=tuple(str(p) for p in value.get("props", ())),
            )
        # (name, [props]) or (name, arity) tuples.
        if isinstance(value, (list, tuple)):
            name, rest = value[0], value[1:]
            if rest and isinstance(rest[0], int):
                return cls(name=str(name), props=tuple(f"arg{i}" for i in range(rest[0])))
            props = rest[0] if rest else ()
            return cls(name=str(name), props=tuple(str(p) for p in props))
        return cls(name=str(value))


@dataclass(frozen=True)
class LatentTaskSpec:
    """A task's deterministic component/production inventory.

    ``task_id`` + ``description`` name the task; ``components`` is the ordered
    production inventory the synthesized grammar/pack is built from.
    """

    task_id: str
    description: str
    components: tuple[LatentComponent, ...]
    root_name: str = "root"

    @property
    def dsl_id(self) -> str:
        """Registry id for the synthesized backend + pack (deterministic)."""
        return f"latent-{_slug(self.task_id)}"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LatentTaskSpec":
        return cls(
            task_id=str(data["task_id"]),
            description=str(data.get("description", "")),
            components=tuple(
                LatentComponent.coerce(c) for c in data.get("components", ())
            ),
            root_name=str(data.get("root_name", "root")),
        )


def synthesize_grammar(spec: LatentTaskSpec) -> tuple[str, dict[str, list[str]]]:
    """Deterministically emit ``(lark_text, prop_order)`` for ``spec``.

    Grammar-Prompting's LLM step is stubbed: the CFG is the fixed toy_layout
    skeleton with a task-derived header comment; the per-task signal lives in
    the ``prop_order`` map (component heads → positional prop names).
    """
    comp_names = ", ".join(c.name for c in spec.components) or "(none declared)"
    header = (
        f"// Latent DSL synthesized for task {spec.task_id!r} (G3, deterministic stub).\n"
        f"// {spec.description or 'no description'}\n"
        f"// components: {comp_names}\n"
        "// Grammar-Prompting (arXiv:2305.19234) task->grammar step is STUBBED:\n"
        "// component names ride the generic NAME call rule (toy_layout skeleton),\n"
        "// so the generic Lark transformer yields ElementNode ASTs for free.\n\n"
    )
    prop_order = {c.name: list(c.props) for c in spec.components}
    return header + _SKELETON, prop_order


def _make_scope_extractor(dsl_id: str):
    def scope_extractor(source: str, **kwargs: Any) -> list[Any]:
        from slm_training.data.scope_extract import extract_scope_slices

        return extract_scope_slices(source, dsl=dsl_id, **kwargs)

    return scope_extractor


def _make_engine(grammar_path: Path):
    def engine() -> Any:
        from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine

        return OpenUIIncrementalEngine(grammar_path)

    return engine


def synthesize_pack(spec: LatentTaskSpec, *, grammars_dir: Path | None = None) -> DslPack:
    """Instantiate + register a PARTIAL :class:`DslPack` for ``spec``.

    Writes the synthesized ``.lark`` to ``grammars_dir`` (default
    ``<grammars>/latent/``), builds and registers a :class:`LarkFileBackend`
    over it, then assembles and registers a partial pack (backend + scope rules
    + placeholder policy + prop order + incremental engine filled;
    ``canonicalize``/``oracle``/``corpus_generator`` honest-``None``).

    Idempotent / re-registration-safe: the grammar file is rewritten and the
    backend + pack overwrite their registry entries under ``spec.dsl_id``. The
    grammar file must outlive pack use (the backend reads it lazily); the
    default location persists, temp dirs give an ephemeral lifetime.
    """
    # Force the builtin backend + pack registries to load BEFORE we register,
    # so our insertion doesn't short-circuit their lazy "empty registry" guards.
    list_packs()
    available_backends()

    lark_text, prop_order = synthesize_grammar(spec)
    target_dir = Path(grammars_dir) if grammars_dir is not None else _DEFAULT_GRAMMARS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    grammar_path = target_dir / f"{spec.dsl_id}.lark"
    grammar_path.write_text(lark_text, encoding="utf-8")

    backend = LarkFileBackend(
        dsl_id=spec.dsl_id,
        grammar_path=grammar_path,
        description=f"Latent DSL for task {spec.task_id!r} (G3 synthesized)",
        root_name=spec.root_name,
        call_as_component=True,
        prop_order={k: list(v) for k, v in prop_order.items()},
    )
    register_backend(backend)

    policy = PlaceholderPolicy(
        placeholder_re=PLACEHOLDER_RE,
        content_props=CONTENT_PROPS,
        slot_contract=canonical_slot_contract,
    )
    pack = DslPack(
        pack_id=spec.dsl_id,
        backend=backend,
        placeholder_policy=policy,
        # Same honesty as toy-layout: the grammar/scope/engine are real, but the
        # oracle is only a parse, not a behavioral verdict.
        reward_label="parse_only",
        scope_extractor=_make_scope_extractor(spec.dsl_id),
        prop_order=backend.prop_order,
        incremental_engine=_make_engine(grammar_path),
    )
    register_pack(pack)
    return pack


__all__ = [
    "LatentComponent",
    "LatentTaskSpec",
    "synthesize_grammar",
    "synthesize_pack",
]
