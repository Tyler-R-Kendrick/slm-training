"""CAP0 eval-suite freeze and anti-collapse gates.

SLM-361 (DSH1-09). Freezes six immutable eval subsuites built deterministically
from the DSH1 artifact pipeline:

* ``atomic_coverage`` — every atomic production of an eval answer, checked
  through canonical-AST equivalence (DSH1-01/02 authority + witnesses);
* ``expanded_canonicalization`` — expanded surface → canonical pairs from the
  DSH1-05 proven equivalence transforms;
* ``compositional_held_out`` — DSH1-06 typed compositions of eval-split
  answers, held out from every train family;
* ``prefix_completion`` — statement-boundary prefixes carrying a DSH1-07
  compiler certificate, scored against the full verified accepted set;
* ``marker_permutation`` — DSH1-04 authority derangements: matched structure,
  permuted marker surfaces, different canonical fingerprint (must reject);
* ``anti_identity`` — same-symbol-multiset / different-structure hard
  negatives from DSH1-08 (must reject).

Split discipline: rows are admitted only for root families the
:class:`~slm_training.harnesses.train_data.split_policy.RootFamilySplitPolicyV1`
assigns to an eval split; train families are disjoint by construction and any
leak is a zero-tolerance invariant violation.

Zero tolerance: :func:`audit_suite` returns a
:class:`RejectionArtifactV1` for **every** finding — free-form targets,
undeclared markers, parse/static/canonical mismatches, unresolved
scope/binders, missing provenance, split leakage. Any finding invalidates the
suite; nothing is silently dropped.

Gates: :class:`Cap0GateV1` versions its thresholds and its prior-level
retention semantics. Identity-only, unchanged-input, and empty/trivial
baselines are constructed scorers that must FAIL the gate. Sample sizes are
preregistered per subsuite with Wilson confidence bounds; fixture ``n``
reports ``insufficient_n`` and can never issue ``CERT_CAP0``.

Stop rules: any split leakage, any prefix row without a verified compiler
certificate, or any accepted-set target that rewards an incomplete (shortest)
output invalidates the suite.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Literal, Mapping

from slm_training.dsl.canonicalize import canonical_fingerprint
from slm_training.dsl.grammar_capabilities import (
    GrammarCapabilityAdapterV1,
    production_id,
)
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.production_codec import _expr_refs, _parse_bindings
from slm_training.evals.power_protocol import wilson_interval
from slm_training.harnesses.train_data.answer_composition import (
    admit_answer,
    apply_combinator,
)
from slm_training.harnesses.train_data.cap0_qa import (
    _accepted_completions,
    _permuted_markers_negative,
    _same_symbols_negative,
    _statement_prefixes,
)
from slm_training.harnesses.train_data.equivalence_transforms import (
    CANONICALLY_PREFERRED_TO,
    materialize_equivalence_class,
)
from slm_training.harnesses.train_data.partial_states import (
    PartialStateClassV1,
    classify_partial_state,
)
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

SCHEMA_VERSION = "cap0_eval_freeze/v1"
GENERATOR_VERSION = "v1"
GATE_VERSION = "cap0_gate/v1"
RETENTION_POLICY_VERSION = "cap0_gate_retention/v1"
CERT_CAP0 = "CERT_CAP0"

Subsuite = Literal[
    "atomic_coverage",
    "expanded_canonicalization",
    "compositional_held_out",
    "prefix_completion",
    "marker_permutation",
    "anti_identity",
]
SUBSUITES: tuple[Subsuite, ...] = (
    "atomic_coverage",
    "expanded_canonicalization",
    "compositional_held_out",
    "prefix_completion",
    "marker_permutation",
    "anti_identity",
)

# Declared structured target kinds; anything else is a free-form target.
TARGET_KINDS = frozenset(
    {
        "canonical_ast_equivalence",
        "canonicalize",
        "compose",
        "accepted_set",
        "reject_near_miss",
    }
)

_PROVENANCE_REQUIRED = (
    "root_family_id",
    "split",
    "source_sha256",
    "generator_version",
    "dsh_refs",
)


class SuiteIntegrityError(ValueError):
    """A frozen suite failed its tamper-evident integrity check."""


class RejectionCodeV1(str, Enum):
    FREE_FORM_TARGET = "free_form_target"
    UNDECLARED_MARKER = "undeclared_marker"
    PARSE_STATIC_MISMATCH = "parse_static_mismatch"
    CANONICAL_MISMATCH = "canonical_mismatch"
    UNRESOLVED_SCOPE = "unresolved_scope"
    MISSING_PROVENANCE = "missing_provenance"
    SPLIT_LEAKAGE = "split_leakage"
    UNPROVEN_PREFIX = "unproven_prefix"
    INCOMPLETE_SHORTEST_REWARD = "incomplete_shortest_reward"


@dataclass(frozen=True)
class RejectionArtifactV1:
    """One zero-tolerance finding; any artifact invalidates the suite."""

    code: RejectionCodeV1
    subsuite: str
    row_id: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "subsuite": self.subsuite,
            "row_id": self.row_id,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


# --------------------------------------------------------------------------- #
# Rows + frozen suite
# --------------------------------------------------------------------------- #


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_digest(payload: Mapping[str, Any]) -> str:
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


@dataclass(frozen=True)
class Cap0EvalRowV1:
    """One frozen eval row; ``sha256`` binds every field but itself."""

    row_id: str
    subsuite: Subsuite
    root_family_id: str
    split: str
    payload: dict[str, Any]
    target: dict[str, Any]
    provenance: dict[str, Any]
    sha256: str = ""

    def body(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "subsuite": self.subsuite,
            "root_family_id": self.root_family_id,
            "split": self.split,
            "payload": self.payload,
            "target": self.target,
            "provenance": self.provenance,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "sha256": self.sha256}


def _make_row(
    subsuite: Subsuite,
    *,
    root_family_id: str,
    split: str,
    payload: dict[str, Any],
    target: dict[str, Any],
    provenance: dict[str, Any],
    id_parts: Iterable[str],
) -> Cap0EvalRowV1:
    raw = ":".join((subsuite, *id_parts))
    row_id = f"{_sha256_text(raw)[:16]}_cap0eval"
    # Declared markers cover every program text the row carries (payload +
    # target), so the undeclared-marker invariant is self-consistent.
    provenance = dict(provenance)
    declared = set(provenance.get("declared_markers") or ())

    def _collect(value: Any) -> None:
        if isinstance(value, str):
            declared.update(extract_placeholders(value))
        elif isinstance(value, dict):
            for item in value.values():
                _collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _collect(item)

    _collect(payload)
    _collect(target)
    provenance["declared_markers"] = sorted(declared)
    base = Cap0EvalRowV1(
        row_id=row_id,
        subsuite=subsuite,
        root_family_id=root_family_id,
        split=split,
        payload=payload,
        target=target,
        provenance=provenance,
    )
    return Cap0EvalRowV1(**{**base.body(), "sha256": _row_digest(base.body())})


@dataclass(frozen=True)
class Cap0EvalSuiteV1:
    """The six frozen CAP0 eval subsuites plus a tamper-evident manifest."""

    rows: tuple[Cap0EvalRowV1, ...]
    manifest: dict[str, Any]
    seed: int
    eval_splits: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION
    generator_version: str = GENERATOR_VERSION

    def rows_for(self, subsuite: Subsuite) -> tuple[Cap0EvalRowV1, ...]:
        return tuple(row for row in self.rows if row.subsuite == subsuite)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "eval_splits": list(self.eval_splits),
            "manifest": dict(self.manifest),
            "rows": [row.to_dict() for row in self.rows],
        }


def verify_suite_integrity(suite: Cap0EvalSuiteV1) -> bool:
    """Fail-closed tamper check: recompute every row and manifest digest."""
    for row in suite.rows:
        if row.sha256 != _row_digest(row.body()):
            raise SuiteIntegrityError(f"tampered row payload: {row.row_id}")
    expected = _manifest(suite.rows, seed=suite.seed, eval_splits=suite.eval_splits)
    if suite.manifest != expected:
        raise SuiteIntegrityError("tamper-evident manifest mismatch")
    return True


def _manifest(
    rows: tuple[Cap0EvalRowV1, ...], *, seed: int, eval_splits: tuple[str, ...]
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row.row_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "eval_splits": list(eval_splits),
        "split_policy": RootFamilySplitPolicyV1().schema_version,
        "rows_total": len(ordered),
        "rows_by_subsuite": {
            name: sum(1 for row in ordered if row.subsuite == name)
            for name in SUBSUITES
        },
        "root_families": sorted({row.root_family_id for row in ordered}),
        "rows_sha256": _row_digest([row.sha256 for row in ordered]),
        "subsuite_sha256": {
            name: _row_digest(
                [row.sha256 for row in ordered if row.subsuite == name]
            )
            for name in SUBSUITES
        },
    }


# --------------------------------------------------------------------------- #
# Builder (deterministic, eval-split families only)
# --------------------------------------------------------------------------- #


def _provenance(
    answer: str,
    *,
    root_family_id: str,
    split: str,
    dsh_refs: tuple[str, ...],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "root_family_id": root_family_id,
        "split": split,
        "source_sha256": _sha256_text(answer),
        "generator_version": GENERATOR_VERSION,
        "dsh_refs": list(dsh_refs),
        "declared_markers": sorted(set(extract_placeholders(answer))),
    }
    if extra:
        provenance.update(extra)
    return provenance


def _production_ids(adapter: GrammarCapabilityAdapterV1, source: str) -> list[str]:
    starts = adapter.start_symbols
    primary = str(starts[0]) if starts else "start"
    trace = adapter.authority.production_trace
    if trace is None:
        return []
    try:
        occurrences = trace(primary, source if source.endswith("\n") else source + "\n")
    except Exception:  # noqa: BLE001 - unparseable source: no productions
        return []
    return sorted({production_id(item.production) for item in occurrences})


def build_cap0_eval_suite(
    answers: Iterable[tuple[str, str]],
    *,
    adapter: GrammarCapabilityAdapterV1,
    seed: int = 0,
    dsl: str | None = None,
    eval_splits: tuple[str, ...] = ("test",),
) -> Cap0EvalSuiteV1:
    """Freeze the six CAP0 eval subsuites from ``(source, root_family_id)`` pairs.

    Only root families the split policy assigns to an eval split are admitted;
    train/held families are disjoint by construction. The build is
    deterministic in (answers, seed, dsl).
    """
    policy = RootFamilySplitPolicyV1()
    staged: dict[str, tuple[str, str]] = {}
    for source, family in answers:
        source = str(source).strip("\n")
        if not source.strip():
            continue
        split = policy.assign(str(family))
        if split not in eval_splits:
            continue
        staged[str(family)] = (source, split)
    if not staged:
        raise ValueError("no eval-split root families among the answers")

    witnesses = [
        str(item.source).strip("\n")
        for item in (adapter.authority.witness_candidates or (lambda: ()))()
    ]

    rows: list[Cap0EvalRowV1] = []
    eval_answers: list[tuple[str, str, str]] = []  # (source, family, split)
    for family in sorted(staged):
        source, split = staged[family]
        report = classify_partial_state(source, adapter=adapter)
        if report.state_class is not PartialStateClassV1.COMPLETE:
            raise ValueError(
                f"eval answer is not a verified terminal: {source!r} "
                f"({report.state_class.value}:{report.reason})"
            )
        eval_answers.append((source, family, split))
        equivalence = materialize_equivalence_class(source, dsl=dsl)
        canonical = equivalence.canonical_form
        fingerprint = canonical_fingerprint(source, dsl=dsl)

        # 1. atomic_coverage
        rows.append(
            _make_row(
                "atomic_coverage",
                root_family_id=family,
                split=split,
                payload={"operation": "produce", "canonical": canonical},
                target={
                    "kind": "canonical_ast_equivalence",
                    "canonical_fingerprint": fingerprint,
                    "productions": _production_ids(adapter, source),
                },
                provenance=_provenance(
                    source,
                    root_family_id=family,
                    split=split,
                    dsh_refs=("DSH1-01", "DSH1-02"),
                ),
                id_parts=(family, canonical),
            )
        )

        # 2. expanded_canonicalization (DSH1-05)
        for expansion in equivalence.expansions:
            rows.append(
                _make_row(
                    "expanded_canonicalization",
                    root_family_id=family,
                    split=split,
                    payload={"operation": "canonicalize", "expanded": expansion.form},
                    target={
                        "kind": "canonicalize",
                        "canonical": canonical,
                        "canonical_fingerprint": fingerprint,
                        "relations": [CANONICALLY_PREFERRED_TO],
                    },
                    provenance=_provenance(
                        expansion.form,
                        root_family_id=family,
                        split=split,
                        dsh_refs=("DSH1-05",),
                        extra={
                            "transform_id": expansion.transform_id,
                            "transform_version": expansion.transform_version,
                        },
                    ),
                    id_parts=(family, expansion.form, canonical),
                )
            )

        # 4. prefix_completion (DSH1-07 certificates + verified accepted set)
        completion_domain = (source, *witnesses)
        for prefix in _statement_prefixes(source):
            prefix_report = classify_partial_state(prefix, adapter=adapter)
            if prefix_report.state_class not in (
                PartialStateClassV1.COMPLETE,
                PartialStateClassV1.PREFIX,
            ):
                continue
            accepted = _accepted_completions(
                prefix, candidates=completion_domain, adapter=adapter, dsl=dsl
            )
            if not accepted:
                continue
            # The prefix's proof of completion: the DSH1-07 certificate when
            # the classifier emitted one, else the minimum-cost accepted
            # completion replayed to a verified terminal (same machinery).
            certificate = (
                prefix_report.certificate.to_dict()
                if prefix_report.certificate is not None
                else {
                    "witness_source": accepted[0].source,
                    "completion_suffix": accepted[0].suffix,
                    "terminal_verified": accepted[0].terminal_verified,
                    "prefix_state_class": prefix_report.state_class.value,
                }
            )
            rows.append(
                _make_row(
                    "prefix_completion",
                    root_family_id=family,
                    split=split,
                    payload={"operation": "complete_suffix", "prefix": prefix},
                    target={
                        "kind": "accepted_set",
                        "accepted": [item.to_dict() for item in accepted],
                    },
                    provenance=_provenance(
                        prefix,
                        root_family_id=family,
                        split=split,
                        dsh_refs=("DSH1-07", "DSH1-08"),
                        extra={
                            "prefix_state_class": prefix_report.state_class.value,
                            "prefix_certificate": certificate,
                        },
                    ),
                    id_parts=(family, prefix, *(item.source for item in accepted)),
                )
            )

        # 5. marker_permutation (DSH1-04 authority derangement)
        permuted = _permuted_markers_negative(source, adapter=adapter, dsl=dsl, seed=seed)
        if permuted is not None:
            rows.append(
                _make_row(
                    "marker_permutation",
                    root_family_id=family,
                    split=split,
                    payload={"operation": "verify", "candidate": permuted},
                    target={
                        "kind": "reject_near_miss",
                        "negative_kind": "matched_structure_permuted_markers",
                        "source_fingerprint": fingerprint,
                        "negative_fingerprint": canonical_fingerprint(permuted, dsl=dsl),
                    },
                    provenance=_provenance(
                        permuted,
                        root_family_id=family,
                        split=split,
                        dsh_refs=("DSH1-04",),
                        extra={"source_answer": source},
                    ),
                    id_parts=(family, source, permuted),
                )
            )

        # 6. anti_identity (DSH1-08 same-symbols different-structure negative)
        same_symbols = _same_symbols_negative(source, adapter=adapter, dsl=dsl)
        if same_symbols is not None:
            rows.append(
                _make_row(
                    "anti_identity",
                    root_family_id=family,
                    split=split,
                    payload={"operation": "verify", "candidate": same_symbols},
                    target={
                        "kind": "reject_near_miss",
                        "negative_kind": "same_symbols_different_structure",
                        "source_fingerprint": fingerprint,
                        "negative_fingerprint": canonical_fingerprint(
                            same_symbols, dsl=dsl
                        ),
                    },
                    provenance=_provenance(
                        same_symbols,
                        root_family_id=family,
                        split=split,
                        dsh_refs=("DSH1-08",),
                        extra={"source_answer": source},
                    ),
                    id_parts=(family, source, same_symbols),
                )
            )

    # 3. compositional_held_out (DSH1-06), pairs of eval answers, held out.
    for index in range(len(eval_answers) - 1):
        (left_source, left_family, split), (right_source, right_family, _) = (
            eval_answers[index],
            eval_answers[index + 1],
        )
        left = admit_answer(left_source, split=split, root_family_id=left_family, dsl=dsl)
        right = admit_answer(
            right_source, split=split, root_family_id=right_family, dsl=dsl
        )
        result = apply_combinator("COMPOSE_DOCUMENT", (left, right), dsl=dsl)
        if not result.ok or result.canonical_source is None:
            continue
        rows.append(
            _make_row(
                "compositional_held_out",
                root_family_id=result.root_family_id or left_family,
                split=split,
                payload={
                    "operation": "compose",
                    "source_answer_ids": sorted(result.source_answer_ids),
                },
                target={
                    "kind": "compose",
                    "canonical": result.canonical_source,
                    "canonical_fingerprint": result.after_ast_digest,
                },
                provenance=_provenance(
                    result.canonical_source,
                    root_family_id=result.root_family_id or left_family,
                    split=split,
                    dsh_refs=("DSH1-06",),
                    extra={
                        "held_out": True,
                        "combinator": result.combinator,
                        "combinator_version": result.combinator_version,
                        "lineage_id": result.lineage_id,
                        "source_families": sorted({left_family, right_family}),
                    },
                ),
                id_parts=(result.lineage_id or result.after_ast_digest or ""),
            )
        )

    frozen = tuple(sorted(rows, key=lambda row: row.row_id))
    return Cap0EvalSuiteV1(
        rows=frozen,
        manifest=_manifest(frozen, seed=seed, eval_splits=eval_splits),
        seed=seed,
        eval_splits=tuple(eval_splits),
    )


# --------------------------------------------------------------------------- #
# Zero-tolerance invariants
# --------------------------------------------------------------------------- #


def _row_texts(row: Cap0EvalRowV1) -> list[str]:
    texts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(row.payload)
    if isinstance(row.target, dict):
        walk(row.target)
    return texts


def _row_programs(row: Cap0EvalRowV1) -> list[tuple[str, bool]]:
    """``(program, is_partial)`` texts the row depends on.

    Partial texts (prefixes) are parsed and static-checked but are legitimately
    open scopes — the unresolved-binder invariant applies to full programs only.
    """
    programs: list[tuple[str, bool]] = []
    target = row.target if isinstance(row.target, dict) else {}
    for key in ("canonical", "expanded", "candidate"):
        value = row.payload.get(key) or target.get(key)
        if isinstance(value, str) and value.strip():
            programs.append((value, False))
    prefix = row.payload.get("prefix")
    if isinstance(prefix, str) and prefix.strip():
        programs.append((prefix, True))
    for item in target.get("accepted") or ():
        if isinstance(item, dict) and isinstance(item.get("source"), str):
            programs.append((item["source"], False))
    return programs


def audit_suite(
    suite: Cap0EvalSuiteV1,
    *,
    adapter: GrammarCapabilityAdapterV1,
    dsl: str | None = None,
) -> list[RejectionArtifactV1]:
    """Zero-tolerance invariants; ANY artifact invalidates the suite."""
    policy = RootFamilySplitPolicyV1()
    findings: list[RejectionArtifactV1] = []

    def add(code: RejectionCodeV1, row: Cap0EvalRowV1, reason: str, **evidence: Any) -> None:
        findings.append(
            RejectionArtifactV1(
                code=code,
                subsuite=row.subsuite,
                row_id=row.row_id,
                reason=reason,
                evidence=evidence,
            )
        )

    for row in suite.rows:
        target = row.target if isinstance(row.target, dict) else {}
        # Structured targets only: a declared kind, never free-form text.
        if not isinstance(row.target, dict) or row.target.get("kind") not in TARGET_KINDS:
            add(
                RejectionCodeV1.FREE_FORM_TARGET,
                row,
                "target is not a declared structured kind",
                target=row.target if not isinstance(row.target, dict) else None,
            )
        # Missing provenance.
        missing = [key for key in _PROVENANCE_REQUIRED if not row.provenance.get(key)]
        if missing:
            add(
                RejectionCodeV1.MISSING_PROVENANCE,
                row,
                f"provenance missing required keys: {missing}",
                missing=missing,
            )
        # Split leakage: every row family must assign to an eval split.
        assigned = policy.assign(row.root_family_id) if row.root_family_id else None
        if assigned not in suite.eval_splits or row.split not in suite.eval_splits:
            add(
                RejectionCodeV1.SPLIT_LEAKAGE,
                row,
                f"root family {row.root_family_id!r} assigns to {assigned!r}, "
                f"not an eval split {suite.eval_splits}",
                assigned_split=assigned,
            )
        # Undeclared markers.
        declared = set(row.provenance.get("declared_markers") or ())
        used = set()
        for text in _row_texts(row):
            used.update(extract_placeholders(text))
        undeclared = sorted(used - declared)
        if undeclared:
            add(
                RejectionCodeV1.UNDECLARED_MARKER,
                row,
                f"markers not declared in provenance: {undeclared}",
                undeclared=undeclared,
            )
        # Parse / static / canonical agreement on every program text.
        for program, is_partial in _row_programs(row):
            try:
                adapter.static_validate(program)
            except Exception as exc:  # noqa: BLE001 - fail closed
                add(
                    RejectionCodeV1.PARSE_STATIC_MISMATCH,
                    row,
                    f"program fails static validation: {exc}",
                    program=program,
                )
                continue
            # Unresolved scope / binders (full programs only; prefixes are
            # legitimately open scopes).
            if not is_partial:
                try:
                    bindings = _parse_bindings(program, dsl)
                except Exception:  # noqa: BLE001 - parse failure already reported
                    bindings = {}
                known = set(bindings)
                unresolved = sorted(
                    {
                        ref
                        for name in bindings
                        for ref in _expr_refs(bindings[name], dsl)
                        if ref not in known
                    }
                )
                if unresolved:
                    add(
                        RejectionCodeV1.UNRESOLVED_SCOPE,
                        row,
                        f"unresolved binders: {unresolved}",
                        unresolved=unresolved,
                    )
            # Canonical mismatch against the stored fingerprint.
            expected = target.get("canonical_fingerprint")
            negative = target.get("negative_fingerprint")
            try:
                actual = canonical_fingerprint(program, dsl=dsl)
            except Exception:  # noqa: BLE001 - reported by static check
                continue
            if (
                target.get("kind") in {"canonical_ast_equivalence", "canonicalize", "compose"}
                and expected
                and program in (target.get("canonical"), row.payload.get("canonical"))
                and actual != expected
            ):
                add(
                    RejectionCodeV1.CANONICAL_MISMATCH,
                    row,
                    "canonical fingerprint does not match the stored target",
                    expected=expected,
                    actual=actual,
                )
            if target.get("kind") == "reject_near_miss" and negative:
                if actual == target.get("source_fingerprint") and program in (
                    row.payload.get("candidate"),
                ):
                    add(
                        RejectionCodeV1.CANONICAL_MISMATCH,
                        row,
                        "near-miss negative shares the source canonical fingerprint",
                        fingerprint=actual,
                    )
    return findings


def stop_rule_violations(
    suite: Cap0EvalSuiteV1,
    *,
    adapter: GrammarCapabilityAdapterV1,
) -> list[RejectionArtifactV1]:
    """Stop rules: leakage, unproven prefixes, incomplete-shortest rewards."""
    findings: list[RejectionArtifactV1] = []

    def add(code: RejectionCodeV1, row: Cap0EvalRowV1, reason: str, **evidence: Any) -> None:
        findings.append(
            RejectionArtifactV1(
                code=code,
                subsuite=row.subsuite,
                row_id=row.row_id,
                reason=reason,
                evidence=evidence,
            )
        )

    policy = RootFamilySplitPolicyV1()
    for row in suite.rows:
        if policy.assign(row.root_family_id) not in suite.eval_splits:
            add(
                RejectionCodeV1.SPLIT_LEAKAGE,
                row,
                "row family outside the eval splits invalidates the suite",
            )
        if row.subsuite != "prefix_completion":
            continue
        certificate = row.provenance.get("prefix_certificate")
        if not certificate or not certificate.get("terminal_verified"):
            add(
                RejectionCodeV1.UNPROVEN_PREFIX,
                row,
                "prefix row lacks a verified compiler certificate",
            )
        target = row.target if isinstance(row.target, dict) else {}
        for item in target.get("accepted") or ():
            source = item.get("source") if isinstance(item, dict) else None
            if not source:
                continue
            report = classify_partial_state(source, adapter=adapter)
            if report.state_class is not PartialStateClassV1.COMPLETE:
                add(
                    RejectionCodeV1.INCOMPLETE_SHORTEST_REWARD,
                    row,
                    "accepted-set target rewards an incomplete (shortest) output",
                    source=source,
                    state_class=report.state_class.value,
                )
    return findings


def suite_valid(
    suite: Cap0EvalSuiteV1,
    *,
    adapter: GrammarCapabilityAdapterV1,
    dsl: str | None = None,
) -> bool:
    """True only when integrity holds and no invariant/stop-rule fires."""
    verify_suite_integrity(suite)
    return not audit_suite(suite, adapter=adapter, dsl=dsl) and not stop_rule_violations(
        suite, adapter=adapter
    )


# --------------------------------------------------------------------------- #
# Anti-collapse gates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cap0MetricsV1:
    """Scorer output over the frozen suite (rates in [0, 1])."""

    canonical_equivalence: float
    accepted_set_mass: float
    production_coverage: float
    structure: float
    binding_aware: float
    empty_trivial_rate: float
    diversity: float
    identity_baseline_score: float
    n_by_subsuite: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_equivalence": self.canonical_equivalence,
            "accepted_set_mass": self.accepted_set_mass,
            "production_coverage": self.production_coverage,
            "structure": self.structure,
            "binding_aware": self.binding_aware,
            "empty_trivial_rate": self.empty_trivial_rate,
            "diversity": self.diversity,
            "identity_baseline_score": self.identity_baseline_score,
            "n_by_subsuite": dict(self.n_by_subsuite),
        }


# Preregistered per-subsuite sample sizes for a certifying run (fixture builds
# are far below these and must report insufficient_n).
PREREGISTERED_N: dict[str, int] = {
    "atomic_coverage": 200,
    "expanded_canonicalization": 400,
    "compositional_held_out": 100,
    "prefix_completion": 200,
    "marker_permutation": 100,
    "anti_identity": 100,
}


@dataclass(frozen=True)
class Cap0GateV1:
    """Versioned anti-collapse thresholds + prior-level retention semantics.

    Retention (``cap0_gate_retention/v1``): a later gate version may only
    tighten thresholds or raise preregistered n; loosening any prior level
    requires a new retention policy version and a governance note.
    """

    min_canonical_equivalence: float = 0.95
    min_accepted_set_mass: float = 0.90
    min_production_coverage: float = 1.0
    min_structure: float = 0.90
    min_binding_aware: float = 0.90
    max_empty_trivial_rate: float = 0.05
    min_diversity: float = 0.50
    max_identity_baseline_score: float = 0.50
    preregistered_n: dict[str, int] = field(
        default_factory=lambda: dict(PREREGISTERED_N)
    )
    confidence_level: float = 0.95
    retention_policy: str = RETENTION_POLICY_VERSION
    gate_version: str = GATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_version": self.gate_version,
            "retention_policy": self.retention_policy,
            "min_canonical_equivalence": self.min_canonical_equivalence,
            "min_accepted_set_mass": self.min_accepted_set_mass,
            "min_production_coverage": self.min_production_coverage,
            "min_structure": self.min_structure,
            "min_binding_aware": self.min_binding_aware,
            "max_empty_trivial_rate": self.max_empty_trivial_rate,
            "min_diversity": self.min_diversity,
            "max_identity_baseline_score": self.max_identity_baseline_score,
            "preregistered_n": dict(self.preregistered_n),
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True)
class Cap0GateReportV1:
    passed: bool
    certificate: str | None
    status: str  # "certified" | "failed" | "insufficient_n"
    checks: dict[str, bool]
    insufficient_n: dict[str, dict[str, int]]
    confidence_bounds: dict[str, dict[str, Any]]
    gate: Cap0GateV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "certificate": self.certificate,
            "status": self.status,
            "checks": dict(self.checks),
            "insufficient_n": {k: dict(v) for k, v in self.insufficient_n.items()},
            "confidence_bounds": self.confidence_bounds,
            "gate": self.gate.to_dict(),
        }


def evaluate_gate(
    metrics: Cap0MetricsV1, *, gate: Cap0GateV1 | None = None
) -> Cap0GateReportV1:
    """Evaluate the anti-collapse gate; fixture n reports ``insufficient_n``."""
    gate = gate or Cap0GateV1()
    checks = {
        "canonical_equivalence": metrics.canonical_equivalence
        >= gate.min_canonical_equivalence,
        "accepted_set_mass": metrics.accepted_set_mass >= gate.min_accepted_set_mass,
        "production_coverage": metrics.production_coverage
        >= gate.min_production_coverage,
        "structure": metrics.structure >= gate.min_structure,
        "binding_aware": metrics.binding_aware >= gate.min_binding_aware,
        "empty_trivial_rate": metrics.empty_trivial_rate <= gate.max_empty_trivial_rate,
        "diversity": metrics.diversity >= gate.min_diversity,
        "identity_baseline_score": metrics.identity_baseline_score
        <= gate.max_identity_baseline_score,
    }
    insufficient = {
        name: {"required": required, "actual": metrics.n_by_subsuite.get(name, 0)}
        for name, required in sorted(gate.preregistered_n.items())
        if metrics.n_by_subsuite.get(name, 0) < required
    }
    # Wilson bounds on the threshold-gated rates (evidence, never a loosening).
    bounds = {
        name: wilson_interval(
            round(metrics.n_by_subsuite.get(name, 0) * getattr(metrics, metric)),
            metrics.n_by_subsuite.get(name, 0),
            confidence_level=gate.confidence_level,
        )
        for name, metric in (
            ("atomic_coverage", "canonical_equivalence"),
            ("prefix_completion", "accepted_set_mass"),
            ("anti_identity", "structure"),
        )
    }
    metrics_pass = all(checks.values())
    if insufficient:
        status = "insufficient_n"
    elif metrics_pass:
        status = "certified"
    else:
        status = "failed"
    certificate = CERT_CAP0 if status == "certified" else None
    return Cap0GateReportV1(
        passed=status == "certified",
        certificate=certificate,
        status=status,
        checks=checks,
        insufficient_n=insufficient,
        confidence_bounds=bounds,
        gate=gate,
    )


# --------------------------------------------------------------------------- #
# Constructed collapse baselines (must all FAIL the gate)
# --------------------------------------------------------------------------- #


def _n_by_subsuite(suite: Cap0EvalSuiteV1) -> dict[str, int]:
    return {name: len(suite.rows_for(name)) for name in SUBSUITES}


def _fingerprint_of(text: str, *, dsl: str | None) -> str | None:
    try:
        return canonical_fingerprint(text, dsl=dsl)
    except Exception:  # noqa: BLE001 - unparsable prediction: no match
        return None


def _canonical_hits(suite: Cap0EvalSuiteV1, predict: Any, *, dsl: str | None) -> float:
    """Fraction of canonical-ast rows whose target the prediction matches."""
    rows = suite.rows_for("atomic_coverage") + suite.rows_for(
        "expanded_canonicalization"
    )
    if not rows:
        return 0.0
    hits = 0
    for row in rows:
        prediction = predict(row)
        if prediction is None:
            continue
        fingerprint = _fingerprint_of(prediction, dsl=dsl)
        if fingerprint and fingerprint == row.target.get("canonical_fingerprint"):
            hits += 1
    return hits / len(rows)


def _accepted_mass(suite: Cap0EvalSuiteV1, predict: Any) -> float:
    rows = suite.rows_for("prefix_completion")
    if not rows:
        return 0.0
    hits = 0
    for row in rows:
        prediction = predict(row)
        accepted = {
            item.get("source")
            for item in row.target.get("accepted") or ()
            if isinstance(item, dict)
        }
        if prediction in accepted:
            hits += 1
    return hits / len(rows)


def _collapse_metrics(
    suite: Cap0EvalSuiteV1,
    predict: Any,
    *,
    empty: bool,
    dsl: str | None,
) -> Cap0MetricsV1:
    predictions = [predict(row) for row in suite.rows]
    distinct = {json.dumps(p, sort_keys=True, default=str) for p in predictions}
    n = len(suite.rows) or 1
    # Near-miss control rows: collapse baselines accept everything, so their
    # structure/binding-aware discrimination is zero by construction.
    return Cap0MetricsV1(
        canonical_equivalence=_canonical_hits(suite, predict, dsl=dsl),
        accepted_set_mass=_accepted_mass(suite, predict),
        production_coverage=0.0,
        structure=0.0,
        binding_aware=0.0,
        empty_trivial_rate=1.0 if empty else 0.0,
        diversity=len(distinct) / n,
        identity_baseline_score=1.0,
        n_by_subsuite=_n_by_subsuite(suite),
    )


def identity_baseline_metrics(
    suite: Cap0EvalSuiteV1, *, dsl: str | None = None
) -> Cap0MetricsV1:
    """Identity-only baseline: echo the canonical/prefix payload verbatim."""

    def predict(row: Cap0EvalRowV1) -> str | None:
        return row.payload.get("canonical") or row.payload.get("prefix")

    return _collapse_metrics(suite, predict, empty=False, dsl=dsl)


def unchanged_input_baseline_metrics(
    suite: Cap0EvalSuiteV1, *, dsl: str | None = None
) -> Cap0MetricsV1:
    """Unchanged-input baseline: return whatever the payload carries."""

    def predict(row: Cap0EvalRowV1) -> str | None:
        for key in ("expanded", "candidate", "canonical", "prefix"):
            value = row.payload.get(key)
            if isinstance(value, str):
                return value
        return None

    return _collapse_metrics(suite, predict, empty=False, dsl=dsl)


def empty_trivial_baseline_metrics(suite: Cap0EvalSuiteV1) -> Cap0MetricsV1:
    """Empty/trivial baseline: always predict the empty program."""
    return _collapse_metrics(suite, lambda row: "", empty=True, dsl=None)


__all__ = [
    "CERT_CAP0",
    "GATE_VERSION",
    "GENERATOR_VERSION",
    "PREREGISTERED_N",
    "RETENTION_POLICY_VERSION",
    "SCHEMA_VERSION",
    "SUBSUITES",
    "TARGET_KINDS",
    "Cap0EvalRowV1",
    "Cap0EvalSuiteV1",
    "Cap0GateReportV1",
    "Cap0GateV1",
    "Cap0MetricsV1",
    "RejectionArtifactV1",
    "RejectionCodeV1",
    "Subsuite",
    "SuiteIntegrityError",
    "audit_suite",
    "build_cap0_eval_suite",
    "empty_trivial_baseline_metrics",
    "evaluate_gate",
    "identity_baseline_metrics",
    "stop_rule_violations",
    "suite_valid",
    "unchanged_input_baseline_metrics",
    "verify_suite_integrity",
]
