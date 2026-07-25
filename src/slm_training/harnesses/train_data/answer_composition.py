"""Typed answer composition with scope-safe AST combinators.

SLM-358 (DSH1-06). Verified answers are composed into larger objects, lists,
scopes, and documents **through typed AST operations only** — never through
raw string concatenation of program text. Every combinator:

* is generation-only, with a declared typed domain/codomain and explicit
  pre/postconditions (:class:`CombinatorV1`);
* operates on parsed statement-AST objects (``production_codec`` statement
  bindings: ``element`` / ``ref`` / ``list`` / literal nodes) — answer source
  text is only ever *parsed*, never spliced; the composite surface is
  produced by a deterministic AST renderer, the same discipline as
  ``production_codec.decode_productions``;
* resolves binder collisions by **deterministic alpha-renaming** following
  the marker-table authority (``marker_tables.permute_marker_surfaces`` /
  ``_opaque_surface`` namespace scheme): the full rename map is computed up
  front and applied to every definition and reference in one pass, so a
  rename is atomic — no partially-renamed intermediate is ever observable;
* re-parses the composite through the official parser
  (``slm_training.dsl.parser.validate``) and canonicalizes it
  (``slm_training.dsl.canonicalize.canonicalize``), failing closed with a
  stable :class:`CompositionCode` reason when any check fails;
* validates ordering, cardinality, scope (unresolved references), schema
  (parser re-validation), canonicalization round-trip, and source-split
  compatibility (``split_policy.RootFamilySplitPolicyV1``);
* persists source answer ids, combinator version, before/after AST digests,
  and a proof/rejection report, and can persist lineage as
  content-addressed nodes in the artifact graph.

Stop rule: any combinator application that would emit an illegal
intermediate (e.g. inserting a child into a leaf component, dropping an
existing child, or leaving a dangling reference) is rejected — never
admitted, never repaired by string surgery.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Literal

from slm_training.dsl.canonicalize import canonical_fingerprint, canonicalize
from slm_training.dsl.lang_core import ParseError
from slm_training.dsl.parser import validate as official_validate
from slm_training.dsl.production_codec import (
    _expr_refs,
    _ordered_prop_values,
    _parse_bindings,
    _parse_rhs_value,
    _prop_order,
    _requires_v05_codec,
    _statement_order,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.train_data.artifact_graph import (
    ArtifactGraphStore,
    ArtifactNodeV1,
)
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

SCHEMA_VERSION = "answer_composition/v1"
COMBINATOR_VERSION = "v1"
REGISTRY_VERSION = "answer_composition_registry/v1"
ARTIFACT_TYPE = "composite_answer"

CombinatorName = Literal[
    "APPEND_STATEMENT", "INSERT_CHILD", "MAKE_LIST", "WRAP_NODE", "COMPOSE_DOCUMENT"
]
COMBINATOR_NAMES: tuple[CombinatorName, ...] = (
    "APPEND_STATEMENT",
    "INSERT_CHILD",
    "MAKE_LIST",
    "WRAP_NODE",
    "COMPOSE_DOCUMENT",
)


class CompositionCode(str, Enum):
    """Stable fail-closed rejection codes."""

    PRECONDITION = "precondition_violation"
    CARDINALITY = "cardinality_violation"
    UNRESOLVED_REF = "unresolved_ref"
    INVALID_TARGET = "invalid_target"
    INCOMPATIBLE_ROOTS = "incompatible_roots"
    CROSS_SPLIT = "cross_split_source"
    ILLEGAL_INTERMEDIATE = "illegal_intermediate"
    PARSE_FAILURE = "composite_parse_failure"
    VALIDATION_FAILURE = "composite_validation_failure"
    CANONICAL_MISMATCH = "canonical_roundtrip_mismatch"


@dataclass(frozen=True)
class CompositionRejectionV1:
    code: CompositionCode
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "reason": self.reason}


@dataclass(frozen=True)
class CompositionResultV1:
    """Outcome of one combinator application (admitted proof or rejection)."""

    ok: bool
    combinator: str
    combinator_version: str
    source_answer_ids: tuple[str, ...]
    before_ast_digest: str
    composite_source: str | None = None
    canonical_source: str | None = None
    after_ast_digest: str | None = None
    proof: dict[str, Any] | None = None
    rejection: CompositionRejectionV1 | None = None
    split: str | None = None
    root_family_id: str | None = None
    lineage_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "combinator": self.combinator,
            "combinator_version": self.combinator_version,
            "source_answer_ids": list(self.source_answer_ids),
            "before_ast_digest": self.before_ast_digest,
            "composite_source": self.composite_source,
            "canonical_source": self.canonical_source,
            "after_ast_digest": self.after_ast_digest,
            "proof": self.proof,
            "rejection": self.rejection.to_dict() if self.rejection else None,
            "split": self.split,
            "root_family_id": self.root_family_id,
            "lineage_id": self.lineage_id,
        }


@dataclass(frozen=True)
class CombinatorV1:
    """One generation-only composition combinator with a typed contract."""

    name: CombinatorName
    version: str
    domain: str
    codomain: str
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    apply: Callable[..., CompositionResultV1] = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "domain": self.domain,
            "codomain": self.codomain,
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
        }


# --------------------------------------------------------------------------- #
# Verified answers (parsed statement-AST objects, never raw text operands)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AnswerV1:
    """A verified answer: source admitted only after parse + canonical proof."""

    answer_id: str
    source: str
    canonical_source: str
    canonical_digest: str
    split: str = "train"
    root_family_id: str = ""
    schema_version: str = SCHEMA_VERSION

    def bindings(self, *, dsl: str | None = None) -> dict[str, Any]:
        """Parsed statement ASTs keyed by binder name (fresh deep copy)."""
        return copy.deepcopy(_parse_bindings(self.canonical_source, dsl))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "answer_id": self.answer_id,
            "source": self.source,
            "canonical_source": self.canonical_source,
            "canonical_digest": self.canonical_digest,
            "split": self.split,
            "root_family_id": self.root_family_id,
        }


def admit_answer(
    source: str,
    *,
    split: str = "train",
    root_family_id: str = "",
    dsl: str | None = None,
) -> AnswerV1:
    """Admit a source program as a verified answer (fail closed).

    The source must parse through the official parser, canonicalize, and
    round-trip canonically; its id is the content digest of the canonical
    form (one id per equivalence class).
    """
    canonical = canonicalize(source, dsl=dsl)  # raises unless parse+roundtrip ok
    official_validate(canonical, dsl=dsl)
    digest = canonical_fingerprint(canonical, dsl=dsl)
    return AnswerV1(
        answer_id=digest,
        source=source,
        canonical_source=canonical,
        canonical_digest=digest,
        split=split,
        root_family_id=root_family_id,
    )


def parse_rhs(rhs_source: str, *, dsl: str | None = None) -> Any:
    """Parse an RHS fragment into a statement AST (typed combinator input)."""
    return _parse_rhs_value(rhs_source, dsl)


# --------------------------------------------------------------------------- #
# Deterministic alpha-renaming (marker-table authority), applied atomically
# --------------------------------------------------------------------------- #


def _rename_namespace(*digests: str) -> str:
    """Deterministic opaque namespace, mirroring marker_tables' ns scheme."""
    return hashlib.sha256(f"ns|{'|'.join(sorted(digests))}".encode()).hexdigest()[:4]


def _alpha_rename(
    bindings: dict[str, Any],
    rename_map: dict[str, str],
) -> dict[str, Any]:
    """Apply ``rename_map`` to every definition and reference in one pass.

    The map is computed before any mutation; definitions and refs move
    atomically, so no partially-renamed intermediate exists.
    """
    def rename_refs(node: Any) -> Any:
        if isinstance(node, list):
            return [rename_refs(item) for item in node]
        if isinstance(node, dict):
            if node.get("type") == "ref":
                name = str(node.get("name"))
                return {"type": "ref", "name": rename_map.get(name, name)}
            return {key: rename_refs(value) for key, value in node.items()}
        return node

    renamed: dict[str, Any] = {}
    for name, rhs in bindings.items():
        renamed[rename_map.get(name, name)] = rename_refs(rhs)
    return renamed


def _collision_free_renames(
    incoming: Iterable[str],
    taken: set[str],
    namespace: str,
) -> dict[str, str]:
    """Deterministic rename map for colliding binders (defs + refs together)."""
    renames: dict[str, str] = {}
    reserved = set(taken)
    for name in incoming:
        if name in reserved:
            candidate = f"{name}__{namespace}"
            suffix = 0
            while candidate in reserved:
                suffix += 1
                candidate = f"{name}__{namespace}_{suffix}"
            renames[name] = candidate
            reserved.add(candidate)
        else:
            reserved.add(name)
    return renames


# --------------------------------------------------------------------------- #
# Deterministic AST renderer (AST -> surface; the inverse of parsing)
# --------------------------------------------------------------------------- #


def _render_expr(node: Any, *, dsl: str | None = None) -> str:
    if isinstance(node, list):
        return "[" + ", ".join(_render_expr(item, dsl=dsl) for item in node) + "]"
    if isinstance(node, dict):
        kind = node.get("type")
        if kind == "ref":
            return str(node["name"])
        if kind == "element":
            type_name = str(node.get("typeName"))
            props = dict(node.get("props") or {})
            values = [
                value
                for value in _ordered_prop_values(type_name, props, dsl)
                if value is not None
            ]
            args = ", ".join(_render_expr(value, dsl=dsl) for value in values)
            return f"{type_name}({args})"
        raise ParseError(f"cannot render AST node kind {kind!r}")
    if isinstance(node, bool):
        return "true" if node else "false"
    if node is None:
        return "null"
    if isinstance(node, str):
        return json.dumps(node)
    if isinstance(node, (int, float)):
        return repr(node)
    raise ParseError(f"cannot render literal {node!r}")


def _render_document(bindings: dict[str, Any], *, dsl: str | None = None) -> str:
    lines = [
        f"{name} = {_render_expr(bindings[name], dsl=dsl)}"
        for name in _statement_order(bindings, dsl)
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Admission: re-parse + official validation + canonical round-trip
# --------------------------------------------------------------------------- #


def _list_prop_key(type_name: str, props: dict[str, Any], dsl: str | None) -> str | None:
    """First list-valued positional prop (the children slot), if any."""
    for key in _prop_order(dsl).get(type_name) or []:
        if isinstance(props.get(key), list):
            return key
    return None


def _element_children(node: Any, dsl: str | None) -> tuple[str, list[Any]] | None:
    if not isinstance(node, dict) or node.get("type") != "element":
        return None
    props = dict(node.get("props") or {})
    key = _list_prop_key(str(node.get("typeName") or ""), props, dsl)
    if key is None:
        return None
    return key, props[key]


def _unresolved_refs(bindings: dict[str, Any], dsl: str | None) -> list[str]:
    known = set(bindings)
    missing: list[str] = []
    for name in bindings:
        for ref in _expr_refs(bindings[name], dsl):
            if ref not in known and ref not in missing:
                missing.append(ref)
    return missing


def _before_digest(answers: tuple[AnswerV1, ...]) -> str:
    return content_sha(
        {"answers": sorted(answer.canonical_digest for answer in answers)}
    )


def _finalize(
    *,
    combinator: CombinatorName,
    answers: tuple[AnswerV1, ...],
    bindings: dict[str, Any],
    rename_map: dict[str, str],
    split: str,
    root_family_id: str,
    dsl: str | None,
    extra_proof: dict[str, Any] | None = None,
) -> CompositionResultV1:
    """Render, re-parse, validate, canonicalize — fail closed with stable codes."""

    def reject(code: CompositionCode, reason: str) -> CompositionResultV1:
        return CompositionResultV1(
            ok=False,
            combinator=combinator,
            combinator_version=COMBINATOR_VERSION,
            source_answer_ids=tuple(answer.answer_id for answer in answers),
            before_ast_digest=_before_digest(answers),
            rejection=CompositionRejectionV1(code=code, reason=reason),
            split=split,
            root_family_id=root_family_id,
        )

    missing = _unresolved_refs(bindings, dsl)
    if missing:
        return reject(
            CompositionCode.UNRESOLVED_REF,
            f"composite references undefined binders: {missing}",
        )
    try:
        composite = _render_document(bindings, dsl=dsl)
    except (ParseError, ValueError, TypeError) as exc:
        return reject(
            CompositionCode.ILLEGAL_INTERMEDIATE,
            f"composite AST cannot be rendered: {exc}",
        )
    try:
        official_validate(composite, dsl=dsl)
    except (ParseError, RuntimeError, ValueError) as exc:
        return reject(
            CompositionCode.VALIDATION_FAILURE,
            f"composite fails official parser validation: {exc}",
        )
    try:
        canonical = canonicalize(composite, dsl=dsl)
    except Exception as exc:  # noqa: BLE001 - admission stays fail-closed
        return reject(
            CompositionCode.PARSE_FAILURE,
            f"composite does not canonicalize: {exc}",
        )
    if canonicalize(canonical, dsl=dsl) != canonical:
        return reject(
            CompositionCode.CANONICAL_MISMATCH,
            "canonical form is not a fixed point",
        )
    after = canonical_fingerprint(canonical, dsl=dsl)
    proof: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "official_parser": "slm_training.dsl.parser.validate",
        "canonicalizer": "slm_training.dsl.canonicalize.canonicalize",
        "round_trip": "parse -> compose -> render -> validate -> canonicalize",
        "alpha_rename_map": dict(sorted(rename_map.items())),
        "source_canonical_digests": {
            answer.answer_id: answer.canonical_digest for answer in answers
        },
        "canonical_digest": after,
    }
    if extra_proof:
        proof.update(extra_proof)
    return CompositionResultV1(
        ok=True,
        combinator=combinator,
        combinator_version=COMBINATOR_VERSION,
        source_answer_ids=tuple(answer.answer_id for answer in answers),
        before_ast_digest=_before_digest(answers),
        composite_source=composite,
        canonical_source=canonical,
        after_ast_digest=after,
        proof=proof,
        split=split,
        root_family_id=root_family_id,
        lineage_id=content_sha(
            {
                "schema": SCHEMA_VERSION,
                "combinator": combinator,
                "version": COMBINATOR_VERSION,
                "sources": sorted(answer.answer_id for answer in answers),
                "after": after,
            }
        ),
    )


def _check_split_compatibility(answers: tuple[AnswerV1, ...]) -> str | None:
    splits = {answer.split for answer in answers}
    if len(splits) != 1:
        return (
            f"composition has incompatible source splits: {sorted(splits)} "
            "(RootFamilySplitPolicyV1.require_inherited)"
        )
    return None


def _precondition_reject(
    combinator: CombinatorName,
    answers: tuple[AnswerV1, ...],
    code: CompositionCode,
    reason: str,
    *,
    split: str | None = None,
    root_family_id: str | None = None,
) -> CompositionResultV1:
    return CompositionResultV1(
        ok=False,
        combinator=combinator,
        combinator_version=COMBINATOR_VERSION,
        source_answer_ids=tuple(answer.answer_id for answer in answers),
        before_ast_digest=_before_digest(answers),
        rejection=CompositionRejectionV1(code=code, reason=reason),
        split=split or (answers[0].split if answers else None),
        root_family_id=(
            root_family_id if root_family_id is not None
            else (answers[0].root_family_id if answers else None)
        ),
    )


# --------------------------------------------------------------------------- #
# Combinator bodies (AST-only; no answer text is ever concatenated)
# --------------------------------------------------------------------------- #

_POSTCONDITIONS = (
    "composite re-parses through the official parser",
    "composite validates and canonicalizes to a canonical fixed point",
    "binder/reference semantics preserved (alpha-rename map total and atomic)",
    "source lineage (answer ids, digests, combinator version) recorded",
)


def _append_statement(
    answer: AnswerV1,
    *,
    binder: str,
    rhs: Any,
    dsl: str | None = None,
) -> CompositionResultV1:
    """APPEND_STATEMENT: add one typed statement AST under a fresh binder."""
    if not binder or not binder.isidentifier() or binder.startswith((":", "$")):
        return _precondition_reject(
            "APPEND_STATEMENT", (answer,), CompositionCode.PRECONDITION,
            f"illegal binder name {binder!r}",
        )
    bindings = answer.bindings(dsl=dsl)
    namespace = _rename_namespace(answer.canonical_digest, binder)
    rename_map = _collision_free_renames((binder,), set(bindings), namespace)
    binder = rename_map.get(binder, binder)
    bindings[binder] = copy.deepcopy(rhs)
    return _finalize(
        combinator="APPEND_STATEMENT",
        answers=(answer,),
        bindings=bindings,
        rename_map=rename_map,
        split=answer.split,
        root_family_id=answer.root_family_id,
        dsl=dsl,
        extra_proof={"appended_binder": binder},
    )


def _insert_child(
    answer: AnswerV1,
    *,
    target: str,
    child: Any,
    dsl: str | None = None,
) -> CompositionResultV1:
    """INSERT_CHILD: append a typed child AST into a container's children slot."""
    bindings = answer.bindings(dsl=dsl)
    if target not in bindings:
        return _precondition_reject(
            "INSERT_CHILD", (answer,), CompositionCode.INVALID_TARGET,
            f"target binder {target!r} is not defined",
        )
    node = bindings[target]
    slot = _element_children(node, dsl)
    if slot is None:
        type_name = (
            str(node.get("typeName")) if isinstance(node, dict) else type(node).__name__
        )
        return _precondition_reject(
            "INSERT_CHILD", (answer,), CompositionCode.ILLEGAL_INTERMEDIATE,
            f"inserting a child into leaf {type_name!r} would emit an "
            "illegal intermediate (no children slot)",
        )
    key, children = slot
    props = dict(node.get("props") or {})
    props[key] = [*children, copy.deepcopy(child)]
    bindings[target] = {"type": "element", "typeName": node["typeName"], "props": props}
    return _finalize(
        combinator="INSERT_CHILD",
        answers=(answer,),
        bindings=bindings,
        rename_map={},
        split=answer.split,
        root_family_id=answer.root_family_id,
        dsl=dsl,
        extra_proof={"target": target, "children_cardinality": len(props[key])},
    )


def _make_list(
    answer: AnswerV1,
    *,
    binder: str,
    items: tuple[Any, ...],
    dsl: str | None = None,
) -> CompositionResultV1:
    """MAKE_LIST: bind a typed list AST of item nodes under a fresh binder."""
    if not items:
        return _precondition_reject(
            "MAKE_LIST", (answer,), CompositionCode.CARDINALITY,
            "MAKE_LIST requires at least one item",
        )
    if not binder or not binder.isidentifier() or binder.startswith((":", "$")):
        return _precondition_reject(
            "MAKE_LIST", (answer,), CompositionCode.PRECONDITION,
            f"illegal binder name {binder!r}",
        )
    bindings = answer.bindings(dsl=dsl)
    namespace = _rename_namespace(answer.canonical_digest, binder, "list")
    rename_map = _collision_free_renames((binder,), set(bindings), namespace)
    binder = rename_map.get(binder, binder)
    bindings[binder] = [copy.deepcopy(item) for item in items]
    return _finalize(
        combinator="MAKE_LIST",
        answers=(answer,),
        bindings=bindings,
        rename_map=rename_map,
        split=answer.split,
        root_family_id=answer.root_family_id,
        dsl=dsl,
        extra_proof={"list_binder": binder, "cardinality": len(items)},
    )


def _wrap_node(
    answer: AnswerV1,
    *,
    target: str,
    wrapper_type: str,
    wrapper_args: tuple[Any, ...] = (),
    dsl: str | None = None,
) -> CompositionResultV1:
    """WRAP_NODE: rebind ``target`` to ``wrapper_type([old_target], *args)``.

    The original subtree moves intact under a fresh binder — wrapping never
    drops or reorders the existing children (parent meaning preserved).
    Wrapping a binder that other statements still reference would silently
    change the parents' meaning, so such applications are rejected (stop
    rule); ``root`` is always safe to wrap.
    """
    bindings = answer.bindings(dsl=dsl)
    if target not in bindings:
        return _precondition_reject(
            "WRAP_NODE", (answer,), CompositionCode.INVALID_TARGET,
            f"target binder {target!r} is not defined",
        )
    inbound = sorted(
        name
        for name in bindings
        if name != target and target in _expr_refs(bindings[name], dsl)
    )
    if inbound:
        return _precondition_reject(
            "WRAP_NODE", (answer,), CompositionCode.ILLEGAL_INTERMEDIATE,
            f"wrapping {target!r} would silently change the meaning of "
            f"referencing statements {inbound} (stop rule)",
        )
    namespace = _rename_namespace(answer.canonical_digest, target, wrapper_type)
    taken = set(bindings)
    inner = f"{target}__{namespace}"
    suffix = 0
    while inner in taken:
        suffix += 1
        inner = f"{target}__{namespace}_{suffix}"
    # Move the original subtree under the fresh inner binder, atomically.
    bindings[inner] = bindings[target]
    wrapper = {
        "type": "element",
        "typeName": wrapper_type,
        "props": {
            "_args": [[{"type": "ref", "name": inner}], *copy.deepcopy(list(wrapper_args))]
        },
    }
    bindings[target] = wrapper
    return _finalize(
        combinator="WRAP_NODE",
        answers=(answer,),
        bindings=bindings,
        rename_map={},
        split=answer.split,
        root_family_id=answer.root_family_id,
        dsl=dsl,
        extra_proof={"target": target, "wrapper_type": wrapper_type, "inner_binder": inner},
    )


def _compose_document(
    answers: tuple[AnswerV1, ...],
    *,
    container: str = "Stack",
    container_args: tuple[Any, ...] = ("column",),
    dsl: str | None = None,
) -> CompositionResultV1:
    """COMPOSE_DOCUMENT: merge 2+ verified answers under one container root.

    Every source keeps its own subtree; colliding binders (including each
    source's ``root``) are alpha-renamed deterministically with all
    references updated atomically; the new root references each renamed
    source root exactly once, in argument order.
    """
    if len(answers) < 2:
        return _precondition_reject(
            "COMPOSE_DOCUMENT", answers, CompositionCode.CARDINALITY,
            "COMPOSE_DOCUMENT requires at least two source answers",
        )
    split_error = _check_split_compatibility(answers)
    if split_error:
        return _precondition_reject(
            "COMPOSE_DOCUMENT", answers, CompositionCode.CROSS_SPLIT, split_error,
        )
    typed = [a.answer_id for a in answers if _requires_v05_codec(a.canonical_source)]
    families = {a.root_family_id for a in answers if a.root_family_id}
    if typed:
        return _precondition_reject(
            "COMPOSE_DOCUMENT", answers, CompositionCode.INCOMPATIBLE_ROOTS,
            f"typed v0.5 roots are not document-composable: {sorted(typed)}",
        )
    merged: dict[str, Any] = {}
    rename_map: dict[str, str] = {}
    root_refs: list[Any] = []
    for index, answer in enumerate(answers):
        bindings = answer.bindings(dsl=dsl)
        namespace = _rename_namespace(answer.canonical_digest, f"a{index}")
        local = _collision_free_renames(bindings.keys(), set(merged), namespace)
        # Every source root is always renamed aside (scope isolation): the
        # composite document gets exactly one fresh root of its own.
        if "root" not in local:
            alias = f"root__{namespace}"
            suffix = 0
            while alias in merged:
                suffix += 1
                alias = f"root__{namespace}_{suffix}"
            local["root"] = alias
        merged.update(_alpha_rename(bindings, local))
        rename_map.update(local)
        root_refs.append({"type": "ref", "name": local["root"]})
    wrapper = {
        "type": "element",
        "typeName": container,
        "props": {"_args": [root_refs, *copy.deepcopy(list(container_args))]},
    }
    if "root" in merged:
        return _precondition_reject(
            "COMPOSE_DOCUMENT", answers, CompositionCode.INCOMPATIBLE_ROOTS,
            "a source root survived renaming; composite root would be ambiguous",
        )
    merged["root"] = wrapper
    root_family_id = (
        families.pop() if len(families) == 1
        else _composite_family_id(answers, next(iter(families), ""), answers[0].split)
    )
    return _finalize(
        combinator="COMPOSE_DOCUMENT",
        answers=answers,
        bindings=merged,
        rename_map=rename_map,
        split=answers[0].split,
        root_family_id=root_family_id,
        dsl=dsl,
        extra_proof={"container": container, "root_arity": len(root_refs)},
    )


def _composite_family_id(
    answers: tuple[AnswerV1, ...], seed_family: str, split: str
) -> str:
    """Deterministic composite family id assigned to the sources' split."""
    policy = RootFamilySplitPolicyV1()
    base = content_sha(
        {"composite_of": sorted(a.root_family_id or a.answer_id for a in answers)}
    )[:16]
    for index in range(10_000):
        candidate = f"composite:{seed_family}:{base}:{index}"
        if policy.assign(candidate) == split:
            return candidate
    raise ValueError("no composite family id assigns to the required split")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

COMBINATORS: tuple[CombinatorV1, ...] = (
    CombinatorV1(
        name="APPEND_STATEMENT",
        version=COMBINATOR_VERSION,
        domain="AnswerV1 x Binder x StatementAST",
        codomain="AnswerV1 (document)",
        preconditions=(
            "binder is a legal identifier (no external/state codec)",
            "rhs is a parsed statement AST whose refs resolve after append",
        ),
        postconditions=_POSTCONDITIONS,
        apply=_append_statement,
    ),
    CombinatorV1(
        name="INSERT_CHILD",
        version=COMBINATOR_VERSION,
        domain="AnswerV1 x TargetBinder x ChildAST",
        codomain="AnswerV1 (object)",
        preconditions=(
            "target binder is defined and is an element with a children slot",
            "child is a parsed AST whose refs resolve after insertion",
        ),
        postconditions=_POSTCONDITIONS,
        apply=_insert_child,
    ),
    CombinatorV1(
        name="MAKE_LIST",
        version=COMBINATOR_VERSION,
        domain="AnswerV1 x Binder x NonemptyTuple[AST]",
        codomain="AnswerV1 (list scope)",
        preconditions=(
            "at least one item (cardinality >= 1)",
            "every item is a parsed AST whose refs resolve after binding",
        ),
        postconditions=_POSTCONDITIONS,
        apply=_make_list,
    ),
    CombinatorV1(
        name="WRAP_NODE",
        version=COMBINATOR_VERSION,
        domain="AnswerV1 x TargetBinder x WrapperType x Args",
        codomain="AnswerV1 (scope)",
        preconditions=(
            "target binder is defined",
            "original subtree moves intact (never dropped or reordered)",
        ),
        postconditions=_POSTCONDITIONS,
        apply=_wrap_node,
    ),
    CombinatorV1(
        name="COMPOSE_DOCUMENT",
        version=COMBINATOR_VERSION,
        domain="Tuple[AnswerV1, >=2] x Container x Args",
        codomain="AnswerV1 (document)",
        preconditions=(
            "at least two sources (cardinality >= 2)",
            "all sources share one split (RootFamilySplitPolicyV1)",
            "no typed v0.5 roots among sources (incompatible root kinds)",
        ),
        postconditions=_POSTCONDITIONS,
        apply=_compose_document,
    ),
)

_REGISTRY = {combinator.name: combinator for combinator in COMBINATORS}


def apply_combinator(name: CombinatorName, *args: Any, **kwargs: Any) -> CompositionResultV1:
    """Dispatch a registered combinator by name (fail closed on unknown name)."""
    combinator = _REGISTRY.get(name)
    if combinator is None:
        raise KeyError(f"unknown combinator {name!r}")
    return combinator.apply(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Content-addressed lineage persistence (artifact graph)
# --------------------------------------------------------------------------- #


def persist_lineage(result: CompositionResultV1, store: ArtifactGraphStore) -> str:
    """Persist an admitted composition as a content-addressed lineage node.

    Source ids, combinator version, and before/after AST digests ride in the
    node payload; the artifact id is the result's deterministic lineage id.
    Rejected compositions have no lineage and are refused.
    """
    if not result.ok:
        raise ValueError("rejected compositions have no lineage to persist")
    if not result.root_family_id:
        raise ValueError("lineage requires a root family id on the result")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "combinator": result.combinator,
        "combinator_version": result.combinator_version,
        "source_answer_ids": sorted(result.source_answer_ids),
        "before_ast_digest": result.before_ast_digest,
        "after_ast_digest": result.after_ast_digest,
        "canonical_source": result.canonical_source,
        "proof": result.proof,
    }
    node = ArtifactNodeV1(
        artifact_id=result.lineage_id or "",
        artifact_type=ARTIFACT_TYPE,
        root_family_id=result.root_family_id,
        split_group_id=result.root_family_id,
        split=result.split or "train",
        parent_ids=tuple(sorted(result.source_answer_ids)),
        surface_sha256=hashlib.sha256(
            (result.composite_source or "").encode("utf-8")
        ).hexdigest(),
        alpha_sha256=result.after_ast_digest or "",
        canonical_ast_sha256=result.after_ast_digest or "",
        near_template_sha256=content_sha(
            sorted((result.canonical_source or "").split())
        ),
        payload=payload,
    )
    store.append(node)
    return node.artifact_id


__all__ = [
    "ARTIFACT_TYPE",
    "COMBINATOR_NAMES",
    "COMBINATOR_VERSION",
    "COMBINATORS",
    "REGISTRY_VERSION",
    "SCHEMA_VERSION",
    "AnswerV1",
    "CombinatorName",
    "CombinatorV1",
    "CompositionCode",
    "CompositionRejectionV1",
    "CompositionResultV1",
    "admit_answer",
    "apply_combinator",
    "parse_rhs",
    "persist_lineage",
]
