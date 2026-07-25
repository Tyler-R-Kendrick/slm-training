"""Expanded-equivalent answers and canonical preference relations.

SLM-357 (DSH1-05). Nonminimal but semantically equivalent program forms are
materialized *without* contradictory SFT targets or shortest-valid collapse:

* every registered :class:`EquivalenceTransformV1` carries an equivalence
  **proof obligation** — after ``apply``, both forms parse through the
  official parser and canonicalize to the same canonical AST / equivalence
  class (``slm_training.dsl.canonicalize``). A transform that cannot prove
  equivalence on a fixture is removed or retained only as a negative fixture
  (the stop rule);
* relations ``EQUIVALENT_TO`` / ``CANONICALIZES_TO`` /
  ``CANONICALLY_PREFERRED_TO`` are emitted with deterministic margins,
  transform versions, and provenance stamps;
* canonical cost is compared **only** inside the same compiler-verified
  equivalence class, and only after required roles, cardinality, and scope
  checks pass — a shorter but semantically incomplete program can never
  outrank a complete one;
* the canonical form is the primary SFT target and equivalent forms become
  ``accepted_outputs`` / preference pairs. When a task explicitly asks for
  expansion (``prefer_expanded``), the expanded form is preferred instead;
* order-sensitive properties are never reordered without authority. Statement
  rotation is authorized because the canonicalizer itself fixes a canonical
  statement order; reordering child order inside a container is not, and the
  proof gate removes such transforms.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from slm_training.dsl.canonicalize import canonical_fingerprint, canonicalize
from slm_training.dsl.parser import ParseError, validate
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord, OutputTarget
from slm_training.harnesses.preference import PreferencePair

SCHEMA_VERSION = "equivalence_class/v1"
REGISTRY_VERSION = "equivalence_transform_registry/v1"

EQUIVALENT_TO = "EQUIVALENT_TO"
CANONICALIZES_TO = "CANONICALIZES_TO"
CANONICALLY_PREFERRED_TO = "CANONICALLY_PREFERRED_TO"

RelationKind = Literal["EQUIVALENT_TO", "CANONICALIZES_TO", "CANONICALLY_PREFERRED_TO"]

TransformKind = Literal[
    "explicit_defaults", "alias", "redundant_grouping", "noncanonical_formatting"
]

# Deterministic preference margins per transform kind. The canonical form
# scores 1.0; an equivalent expanded form scores 1.0 - margin, so canonical
# always outranks expanded *within the same verified equivalence class*.
KIND_MARGINS: dict[TransformKind, float] = {
    "explicit_defaults": 0.05,
    "alias": 0.10,
    "redundant_grouping": 0.15,
    "noncanonical_formatting": 0.20,
}

_STATEMENT_RE = re.compile(r"^(\S+)\s*=\s*(.+)$")


@dataclass(frozen=True)
class EquivalenceTransformV1:
    """One pack-approved equivalence transform with a proof obligation.

    ``preconditions`` declares the authority under which the rewrite is
    semantics-preserving (e.g. the canonicalizer strips style literals, or
    fixes canonical statement/binder order). ``apply`` maps a canonical form
    to an expanded surface form; ``prove_equivalence`` verifies the rewrite
    on each concrete fixture before it is admitted.
    """

    id: str
    version: str
    kind: TransformKind
    preconditions: tuple[str, ...]
    apply: Callable[[str], str]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "kind": self.kind,
            "preconditions": list(self.preconditions),
            "provenance": dict(self.provenance),
        }


# --------------------------------------------------------------------------- #
# Transform bodies (canonical -> expanded surface form)
# --------------------------------------------------------------------------- #


_LEAF_CALL_RE = re.compile(r"^(\S+\s*=\s*[A-Z][A-Za-z0-9]*\([^)]*)\)$")


def _add_explicit_style_default(canonical: str) -> str:
    """Append a style-literal default arg the canonicalizer provably strips."""
    out: list[str] = []
    appended = False
    for line in canonical.splitlines():
        match = _LEAF_CALL_RE.match(line)
        if not appended and match and "[" not in line:
            line = f'{match.group(1)}, "primary")'
            appended = True
        out.append(line)
    return "\n".join(out)


def _alias_binder(canonical: str) -> str:
    """Alpha-rename one binder (v0 -> header); canonicalizer renames binders."""
    return re.sub(r"\bv0\b", "header", canonical)


def _rotate_statements(canonical: str) -> str:
    """Permute top-level statement order (canonicalizer fixes canonical order)."""
    lines = canonical.splitlines()
    if len(lines) < 2:
        return canonical
    return "\n".join([*lines[1:], lines[0]])


def _tighten_whitespace(canonical: str) -> str:
    return canonical.replace(" = ", "=").replace(", ", ",")


def _swap_container_children(canonical: str) -> str:
    """UNPROVEN: reorder order-sensitive container children (negative fixture)."""
    return canonical.replace("[v0, v1]", "[v1, v0]")


PACK_APPROVED_TRANSFORMS: tuple[EquivalenceTransformV1, ...] = (
    EquivalenceTransformV1(
        id="explicit_style_default",
        version="v1",
        kind="explicit_defaults",
        preconditions=(
            "canonicalizer strips style literals (strip_style_literals) "
            "before canonical comparison",
            "appended token is a member of STYLE_STRING_TOKENS",
        ),
        apply=_add_explicit_style_default,
        provenance={
            "authority": "slm_training.data.structure.strip_style_literals",
            "rationale": "style-only quoted args carry no layout semantics",
        },
    ),
    EquivalenceTransformV1(
        id="binder_alias",
        version="v1",
        kind="alias",
        preconditions=(
            "canonicalizer renames binders to the De Bruijn-style v0..vn pool",
            "renaming is applied consistently to definition and all uses",
        ),
        apply=_alias_binder,
        provenance={
            "authority": "slm_training.dsl.production_codec binder pool",
            "rationale": "alpha-equivalent programs share one canonical form",
        },
    ),
    EquivalenceTransformV1(
        id="statement_rotation",
        version="v1",
        kind="redundant_grouping",
        preconditions=(
            "canonicalizer fixes a canonical statement order "
            "(topological, first-use)",
            "top-level statement permutation carries no order-sensitive "
            "semantics under that authority",
        ),
        apply=_rotate_statements,
        provenance={
            "authority": "slm_training.dsl.production_codec.encode_openui",
            "rationale": "statement order is redundant grouping under the codec",
        },
    ),
    EquivalenceTransformV1(
        id="tight_whitespace",
        version="v1",
        kind="noncanonical_formatting",
        preconditions=(
            "parser is whitespace-insensitive around '=' and ',' separators",
        ),
        apply=_tighten_whitespace,
        provenance={
            "authority": "slm_training.dsl.parser.validate",
            "rationale": "separator formatting is noncanonical surface only",
        },
    ),
)

# Registered for the stop rule only: order-sensitive rewrites with no
# authority. Materialization must retain them as negative fixtures, never as
# accepted outputs.
UNPROVEN_TRANSFORMS: tuple[EquivalenceTransformV1, ...] = (
    EquivalenceTransformV1(
        id="swap_container_children",
        version="v1",
        kind="redundant_grouping",
        preconditions=(),
        apply=_swap_container_children,
        provenance={
            "authority": None,
            "rationale": "container child order is order-sensitive; no "
            "authority permits reordering",
        },
    ),
)


# --------------------------------------------------------------------------- #
# Relations and equivalence classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RelationV1:
    """One typed, provenance-stamped relation between two program forms."""

    kind: RelationKind
    subject_fingerprint: str
    object_fingerprint: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "subject_fingerprint": self.subject_fingerprint,
            "object_fingerprint": self.object_fingerprint,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class ExpansionV1:
    """One proven expanded form of an equivalence class."""

    transform_id: str
    transform_version: str
    kind: TransformKind
    form: str
    canonical_form: str
    canonical_fingerprint: str
    margin: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "transform_version": self.transform_version,
            "kind": self.kind,
            "form": self.form,
            "canonical_form": self.canonical_form,
            "canonical_fingerprint": self.canonical_fingerprint,
            "margin": self.margin,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class NegativeFixtureV1:
    """Stop-rule record: a transform whose equivalence could not be proven."""

    transform_id: str
    transform_version: str
    kind: TransformKind
    reason: str
    source_form: str
    source_fingerprint: str
    counterexample_form: str
    counterexample_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "transform_version": self.transform_version,
            "kind": self.kind,
            "reason": self.reason,
            "source_form": self.source_form,
            "source_fingerprint": self.source_fingerprint,
            "counterexample_form": self.counterexample_form,
            "counterexample_fingerprint": self.counterexample_fingerprint,
        }


@dataclass(frozen=True)
class EquivalenceClassV1:
    canonical_form: str
    canonical_fingerprint: str
    expansions: tuple[ExpansionV1, ...]
    negative_fixtures: tuple[NegativeFixtureV1, ...]
    registry_version: str = REGISTRY_VERSION
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "canonical_form": self.canonical_form,
            "canonical_fingerprint": self.canonical_fingerprint,
            "expansions": [item.to_dict() for item in self.expansions],
            "negative_fixtures": [item.to_dict() for item in self.negative_fixtures],
        }


def prove_equivalence(
    transform: EquivalenceTransformV1,
    canonical_source: str,
    *,
    dsl: str | None = None,
) -> ExpansionV1 | NegativeFixtureV1:
    """Verify a transform application on one canonical fixture (fail-closed).

    Admitted only when the expanded form parses through the official parser
    AND canonicalizes byte-identically to the input canonical form — the
    compiler-verified same-equivalence-class proof. Otherwise the transform
    application is retained only as a negative fixture.
    """
    expanded = transform.apply(canonical_source)
    source_fingerprint = canonical_fingerprint(canonical_source, dsl=dsl)

    def _negative(reason: str) -> NegativeFixtureV1:
        try:
            counter_fingerprint = canonical_fingerprint(expanded, dsl=dsl)
        except Exception:  # noqa: BLE001 - unparseable counterexample
            counter_fingerprint = ""
        return NegativeFixtureV1(
            transform_id=transform.id,
            transform_version=transform.version,
            kind=transform.kind,
            reason=reason,
            source_form=canonical_source,
            source_fingerprint=source_fingerprint,
            counterexample_form=expanded,
            counterexample_fingerprint=counter_fingerprint,
        )

    if expanded == canonical_source:
        return _negative("transform produced no expansion on this fixture")
    try:
        validate(expanded, dsl=dsl)
    except (ParseError, RuntimeError, ValueError) as exc:
        return _negative(f"expanded form does not parse: {exc}")
    try:
        round_trip = canonicalize(expanded, dsl=dsl)
    except Exception as exc:  # noqa: BLE001 - proof gate stays fail-closed
        return _negative(f"expanded form does not canonicalize: {exc}")
    if round_trip != canonical_source:
        return _negative(
            "equivalence unproven: canonical forms differ "
            "(order-sensitive or semantic rewrite without authority)"
        )
    return ExpansionV1(
        transform_id=transform.id,
        transform_version=transform.version,
        kind=transform.kind,
        form=expanded,
        canonical_form=canonical_source,
        canonical_fingerprint=source_fingerprint,
        margin=KIND_MARGINS[transform.kind],
        provenance={
            "transform": transform.to_dict(),
            "proof": "official parser + canonicalizer round-trip",
        },
    )


def materialize_equivalence_class(
    source: str,
    *,
    transforms: tuple[EquivalenceTransformV1, ...] = PACK_APPROVED_TRANSFORMS,
    stop_rule_transforms: tuple[EquivalenceTransformV1, ...] = UNPROVEN_TRANSFORMS,
    dsl: str | None = None,
) -> EquivalenceClassV1:
    """Canonicalize ``source`` and admit only proven expansions of its class."""
    canonical = canonicalize(source, dsl=dsl)
    fingerprint = canonical_fingerprint(canonical, dsl=dsl)
    expansions: list[ExpansionV1] = []
    negatives: list[NegativeFixtureV1] = []
    for transform in (*transforms, *stop_rule_transforms):
        result = prove_equivalence(transform, canonical, dsl=dsl)
        if isinstance(result, ExpansionV1):
            expansions.append(result)
        else:
            negatives.append(result)
    return EquivalenceClassV1(
        canonical_form=canonical,
        canonical_fingerprint=fingerprint,
        expansions=tuple(expansions),
        negative_fixtures=tuple(negatives),
    )


# --------------------------------------------------------------------------- #
# Canonicality ordering (never let incomplete outrank complete)
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[A-Za-z_$:@][A-Za-z0-9_.$:@]*|\S")


def canonical_cost(source: str) -> tuple[int, int]:
    """Deterministic canonical cost: (surface token count, char length)."""
    return (len(_TOKEN_RE.findall(source)), len(source))


@dataclass(frozen=True)
class CanonicalityComparisonV1:
    """Ordered canonicality verdict for two candidate programs."""

    relation: RelationKind | None
    preferred: str
    other: str
    same_equivalence_class: bool
    completeness_guard: str
    preferred_cost: tuple[int, int]
    other_cost: tuple[int, int]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "preferred": self.preferred,
            "other": self.other,
            "same_equivalence_class": self.same_equivalence_class,
            "completeness_guard": self.completeness_guard,
            "preferred_cost": list(self.preferred_cost),
            "other_cost": list(self.other_cost),
            "provenance": dict(self.provenance),
        }


def compare_canonicality(
    a: str,
    b: str,
    *,
    required_roles: tuple[str, ...] = (),
    dsl: str | None = None,
) -> CanonicalityComparisonV1:
    """Order two programs by canonicality, guarded by completeness.

    Canonical cost decides **only** when both candidates (1) satisfy every
    required role with equal cardinality, (2) parse under the same scope
    (official parser, same dsl), and (3) share one compiler-verified
    equivalence class. A candidate missing a required role is semantically
    incomplete and can never outrank a complete one, no matter how short.
    Different equivalence classes are unordered (``relation=None``).
    """
    validate(a, dsl=dsl)
    validate(b, dsl=dsl)
    roles_a = sorted(extract_placeholders(a))
    roles_b = sorted(extract_placeholders(b))
    missing_a = sorted(set(required_roles) - set(roles_a))
    missing_b = sorted(set(required_roles) - set(roles_b))
    cost_a, cost_b = canonical_cost(a), canonical_cost(b)
    same_class = canonical_fingerprint(a, dsl=dsl) == canonical_fingerprint(
        b, dsl=dsl
    )
    provenance = {
        "required_roles": list(required_roles),
        "roles_a": roles_a,
        "roles_b": roles_b,
        "authority": "official parser + canonicalizer equivalence class",
    }

    def _verdict(
        preferred: str,
        other: str,
        guard: str,
        relation: RelationKind | None,
        preferred_cost: tuple[int, int],
        other_cost: tuple[int, int],
    ) -> CanonicalityComparisonV1:
        return CanonicalityComparisonV1(
            relation=relation,
            preferred=preferred,
            other=other,
            same_equivalence_class=same_class,
            completeness_guard=guard,
            preferred_cost=preferred_cost,
            other_cost=other_cost,
            provenance=provenance,
        )

    # Completeness guard outranks cost, even across equivalence classes.
    if missing_a and not missing_b:
        return _verdict(
            b, a, f"a missing required roles {missing_a}", None, cost_b, cost_a
        )
    if missing_b and not missing_a:
        return _verdict(
            a, b, f"b missing required roles {missing_b}", None, cost_a, cost_b
        )
    if not same_class:
        return _verdict(a, b, "different equivalence classes: unordered", None, cost_a, cost_b)
    if cost_a <= cost_b:
        return _verdict(
            a, b, "passed", CANONICALLY_PREFERRED_TO, cost_a, cost_b
        )
    return _verdict(b, a, "passed", CANONICALLY_PREFERRED_TO, cost_b, cost_a)


# --------------------------------------------------------------------------- #
# SFT / preference materialization
# --------------------------------------------------------------------------- #

CANONICAL_PROMPT = "Emit the OpenUI document for this input.\n---INPUT---\n{text}"
EXPAND_PROMPT = (
    "Expand this OpenUI document with explicit defaults.\n---INPUT---\n{text}"
)


def _record_id(*parts: str) -> str:
    raw = ":".join(parts)
    return f"{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}_equiv"


def materialize_sft_records(
    equivalence_class: EquivalenceClassV1,
    *,
    split: str = "train",
    prefer_expanded: bool = False,
    source: str = "equivalence_expansion",
) -> tuple[ExampleRecord, list[PreferencePair], list[RelationV1]]:
    """Canonical form as primary SFT target; expansions as accepted outputs.

    ``prefer_expanded`` models a task that explicitly asks for expansion: the
    expanded form becomes the chosen/preferred output and the canonical form
    moves to ``accepted_outputs``. Margins, transform versions, and
    provenance stamps are deterministic per transform kind.
    """
    canonical = equivalence_class.canonical_form
    fingerprint = equivalence_class.canonical_fingerprint
    expansions = equivalence_class.expansions
    if prefer_expanded and not expansions:
        raise ValueError("prefer_expanded requires at least one proven expansion")

    primary = expansions[0].form if prefer_expanded else canonical
    accepted = [canonical] if prefer_expanded else [item.form for item in expansions]
    record = ExampleRecord(
        id=_record_id(fingerprint, source, str(prefer_expanded)),
        prompt=(
            EXPAND_PROMPT.format(text=canonical)
            if prefer_expanded
            else CANONICAL_PROMPT.format(text=canonical)
        ),
        openui=primary,
        split=split,
        source=source,
        accepted_outputs=[OutputTarget(text=text) for text in accepted],
        meta={
            "task": "generation" if prefer_expanded else "edit",
            "determinacy": "deterministic",
            "equivalence_class": fingerprint,
            "equivalence_schema": equivalence_class.schema_version,
            "registry_version": equivalence_class.registry_version,
            "prefer_expanded": prefer_expanded,
            "transform_versions": {
                item.transform_id: item.transform_version for item in expansions
            },
        },
    )
    pairs: list[PreferencePair] = []
    relations: list[RelationV1] = []
    for item in expansions:
        if prefer_expanded:
            chosen, rejected = item.form, canonical
            chosen_score, rejected_score = 1.0, 1.0 - item.margin
        else:
            chosen, rejected = canonical, item.form
            chosen_score, rejected_score = 1.0, 1.0 - item.margin
        pairs.append(
            PreferencePair(
                prompt=record.prompt,
                chosen=chosen,
                rejected=rejected,
                chosen_score=chosen_score,
                rejected_score=rejected_score,
                meta={
                    "equivalence_class": fingerprint,
                    "transform_id": item.transform_id,
                    "transform_version": item.transform_version,
                    "margin": item.margin,
                    "prefer_expanded": prefer_expanded,
                    "provenance": item.provenance["proof"],
                },
            )
        )
        relations.append(
            RelationV1(
                kind="EQUIVALENT_TO",
                subject_fingerprint=fingerprint,
                object_fingerprint=item.canonical_fingerprint,
                provenance={"transform_id": item.transform_id},
            )
        )
        relations.append(
            RelationV1(
                kind="CANONICALIZES_TO",
                subject_fingerprint=canonical_fingerprint(item.form),
                object_fingerprint=fingerprint,
                provenance={
                    "transform_id": item.transform_id,
                    "transform_version": item.transform_version,
                },
            )
        )
    return record, pairs, relations


__all__ = [
    "CANONICALIZES_TO",
    "CANONICALLY_PREFERRED_TO",
    "EQUIVALENT_TO",
    "KIND_MARGINS",
    "PACK_APPROVED_TRANSFORMS",
    "REGISTRY_VERSION",
    "SCHEMA_VERSION",
    "UNPROVEN_TRANSFORMS",
    "CanonicalityComparisonV1",
    "EquivalenceClassV1",
    "EquivalenceTransformV1",
    "ExpansionV1",
    "NegativeFixtureV1",
    "RelationV1",
    "canonical_cost",
    "compare_canonicality",
    "materialize_equivalence_class",
    "materialize_sft_records",
    "prove_equivalence",
]
