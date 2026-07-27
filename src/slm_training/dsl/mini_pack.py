"""DSH2-07 (SLM-367): mini-flow — a complete, generic, test-only DSL pack.

The two-pack CAP1 freeze needs a second pack that exercises the *generic*
grammar/schema-grounding code paths without importing OpenUI.  This module
ships one: **mini-flow**, a tiny task-flow language
(``src/slm_training/dsl/grammars/mini_flow.lark``) with

* a Lark grammar authority (DSH1-01 ``lark_authority``),
* a deterministic canonicalizer (codec round-trip, confluent by construction),
* a static validator (parse + closed-domain check),
* a schema contract (:class:`CAP1SchemaV1`, constructed directly — the
  dataclass is DSL-agnostic; only ``from_pack`` is OpenUI-flavoured),
* a :class:`SemanticFrameV1` provider (DSH2-02) walking the Lark tree,
* a marker (placeholder) policy, and
* fragment support (``prop_stmt`` start symbol via the authority).

The pack is **not** registered in the builtin pack registry — it is
test/fixture authority only.  Nothing in this module imports an OpenUI module;
the static import audit in the SLM-367 tests enforces that.

Stop rule (fail-closed): :func:`require_complete_for_generic_claim` refuses to
let a pack support a generic grammar-grounding claim when any required
capability is missing or its smoke probe fails — an incomplete second pack
means the claim stays OpenUI-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from lark import Lark, Token, Tree

from slm_training.dsl.grammar.backends.lark_backend import LarkFileBackend
from slm_training.dsl.grammar_capabilities import (
    GrammarCapabilityAuthorityV1,
    GrammarWitnessCandidateV1,
    lark_authority,
)
from slm_training.dsl.pack import DslPack, PlaceholderPolicy
from slm_training.dsl.placeholders import PLACEHOLDER_RE, extract_placeholders
from slm_training.dsl.semantic_frame import (
    CAP1SchemaV1,
    SemanticFactV1,
    SemanticFrameNodeV1,
    SemanticFrameV1,
    SemanticRoleV1,
)

DSL_ID = "mini-flow"
GRAMMAR_PATH = Path(__file__).resolve().parent / "grammars" / "mini_flow.lark"
START_SYMBOLS = ("start", "prop_stmt")
MODE_DOMAIN = ("fast", "safe")
CONTENT_PROPS = frozenset({"step"})
COMPONENT_PROP_ORDER = {"task": ("step", "mode", "retry")}

WITNESS_SOURCE = 'let owner = "ops"\ntask build { step ":f.desc" mode fast retry 2 }\n'


class PackIncompleteError(ValueError):
    """A pack was asked to support a generic claim without complete authority."""


# --------------------------------------------------------------------------
# Canonicalizer + static validator
# --------------------------------------------------------------------------


def _parser(*, propagate_positions: bool = False) -> Lark:
    return Lark(
        GRAMMAR_PATH.read_text(encoding="utf-8"),
        start=list(START_SYMBOLS),
        parser="lalr",
        maybe_placeholders=False,
        propagate_positions=propagate_positions,
    )


def _literal_text(token: Token) -> str:
    if token.type == "ESCAPED_STRING":
        return json.dumps(json.loads(str(token)))
    return str(token)


def mini_canonicalize(source: str) -> str:
    """Deterministic re-serialization: one statement per line, normalized
    spacing, strings re-escaped via JSON, trailing newline.  Idempotent."""
    tree = _parser().parse(source if source.endswith("\n") else source + "\n", start="start")
    lines: list[str] = []
    for stmt in tree.children:
        (node,) = stmt.children
        if node.data == "let_stmt":
            name, literal = node.children
            lines.append(f"let {name} = {_literal_text(literal.children[0])}")
        elif node.data == "task_stmt":
            name = str(node.children[0])
            props: list[str] = []
            for prop in node.children[1:]:
                token = prop.children[0]
                if token.type == "ESCAPED_STRING":
                    props.append(f"step {_literal_text(token)}")
                elif token.type == "MODE":
                    props.append(f"mode {token}")
                else:
                    props.append(f"retry {token}")
            body = " " + " ".join(props) + " " if props else ""
            lines.append(f"task {name} {{{body}}}")
        else:  # pragma: no cover - grammar has exactly two stmt alternatives
            raise ValueError(f"unexpected stmt node {node.data!r}")
    return "\n".join(lines) + "\n"


def mini_static_validate(source: str) -> Tree:
    """Parse (position-preserving) or raise. Static validity = grammar
    validity; the closed mode domain is enforced by the grammar itself."""
    text = source if source.endswith("\n") else source + "\n"
    return _parser(propagate_positions=True).parse(text, start="start")


def _mini_oracle(record: Any, context: Any = None) -> Mapping[str, Any]:
    """Source-level validity verdict. Reward honesty: parse-only."""
    source = record if isinstance(record, str) else getattr(record, "openui", "")
    try:
        mini_static_validate(source)
    except Exception as exc:  # noqa: BLE001 - oracle reports, never raises
        return {"valid": False, "dsl": DSL_ID, "reward_label": "syntax_only", "error": str(exc)}
    return {"valid": True, "dsl": DSL_ID, "reward_label": "syntax_only"}


def _mini_slot_contract(source: str, *, declared: Iterable[str] | None = None) -> tuple[str, ...]:
    found = tuple(sorted(set(extract_placeholders(source))))
    if declared is None:
        return found
    extras = tuple(sorted(set(str(d) for d in declared) - set(found)))
    return found + extras


# --------------------------------------------------------------------------
# Schema contract + SemanticFrame provider (DSH2-02, DSL-agnostic dataclasses)
# --------------------------------------------------------------------------


def mini_schema() -> CAP1SchemaV1:
    return CAP1SchemaV1(
        dsl=DSL_ID,
        component_names=frozenset({"task"}),
        component_prop_order=COMPONENT_PROP_ORDER,
        content_props=CONTENT_PROPS,
        closed_value_domains={"mode": MODE_DOMAIN},
    )


def _unquote(token: Token) -> str:
    return json.loads(str(token)) if token.type == "ESCAPED_STRING" else str(token)


def derive_mini_frame(source: str, schema: CAP1SchemaV1 | None = None) -> SemanticFrameV1:
    """Derive a :class:`SemanticFrameV1` from the mini-flow parse tree.

    Fact provenance mirrors the OpenUI provider: closed-domain values emit a
    ``required closed_value`` fact plus a ``forbidden excluded_value`` sibling
    per declared domain alternative; placeholder content emits a
    ``required content_placeholder_id`` fact.  Anything outside the declared
    schema raises — no guessing (stop rule)."""
    schema = schema or mini_schema()
    canonical = mini_canonicalize(source)
    tree = _parser().parse(canonical, start="start")
    nodes: list[SemanticFrameNodeV1] = []
    roles: list[SemanticRoleV1] = []
    facts: list[SemanticFactV1] = []
    task_index = 0
    let_index = 0
    for stmt in tree.children:
        (node,) = stmt.children
        if node.data == "let_stmt":
            name, literal = node.children
            facts.append(
                SemanticFactV1(
                    category="required",
                    predicate="let_binding",
                    ast_path=("lets", let_index),
                    schema_field=str(name),
                    value=_unquote(literal.children[0]),
                )
            )
            let_index += 1
            continue
        name = str(node.children[0])
        ast_path = ("tasks", task_index)
        node_id = f"task:{name}"
        nodes.append(
            SemanticFrameNodeV1(node_id=node_id, ast_path=ast_path, kind="task", type_name="task")
        )
        roles.append(
            SemanticRoleV1(parent_id="root", child_id=node_id, role="task_order", order=task_index)
        )
        for prop in node.children[1:]:
            token = prop.children[0]
            if token.type == "ESCAPED_STRING":
                text = _unquote(token)
                if PLACEHOLDER_RE.fullmatch(text):
                    facts.append(
                        SemanticFactV1(
                            category="required",
                            predicate="content_placeholder_id",
                            ast_path=("tasks", task_index, "step"),
                            schema_field="step",
                            value=text,
                        )
                    )
                else:
                    facts.append(
                        SemanticFactV1(
                            category="required",
                            predicate="content_literal_value",
                            ast_path=("tasks", task_index, "step"),
                            schema_field="step",
                            value=text,
                        )
                    )
            elif token.type == "MODE":
                value = str(token)
                facts.append(
                    SemanticFactV1(
                        category="required",
                        predicate="closed_value",
                        ast_path=("tasks", task_index, "mode"),
                        schema_field="mode",
                        value=value,
                    )
                )
                for alternative in schema.closed_value_domains["mode"]:
                    if alternative != value:
                        facts.append(
                            SemanticFactV1(
                                category="forbidden",
                                predicate="excluded_value",
                                ast_path=("tasks", task_index, "mode"),
                                schema_field="mode",
                                value=alternative,
                            )
                        )
            else:
                facts.append(
                    SemanticFactV1(
                        category="required",
                        predicate="literal_value",
                        ast_path=("tasks", task_index, "retry"),
                        schema_field="retry",
                        value=int(str(token)),
                    )
                )
        task_index += 1
    return SemanticFrameV1(
        dsl=DSL_ID,
        schema_fingerprint=schema.fingerprint(),
        canonical_source=canonical,
        nodes=tuple(nodes),
        roles=tuple(roles),
        relations=(),
        effects=(),
        facts=tuple(facts),
    )


# --------------------------------------------------------------------------
# Pack assembly + completeness conformance (fail-closed)
# --------------------------------------------------------------------------


def _mini_fragment_parser(source: str, kind: str = "fragment", grammar_category: str | None = None) -> str:
    authority = _mini_authority()
    fragment_parse = authority.fragment_parse
    if fragment_parse is None:  # pragma: no cover - authority declares it
        raise PackIncompleteError("mini-flow authority lacks fragment_parse")
    start = grammar_category if grammar_category in START_SYMBOLS else "prop_stmt"
    fragment_parse(start, source)
    return source.strip()


def _mini_witness_candidates() -> tuple[GrammarWitnessCandidateV1, ...]:
    return (
        GrammarWitnessCandidateV1(source=WITNESS_SOURCE, runtime_symbols=()),
        GrammarWitnessCandidateV1(
            source="task deploy { step \":f.note\" mode safe retry 1 }\n",
            runtime_symbols=(),
        ),
    )


def _mini_authority() -> GrammarCapabilityAuthorityV1:
    return lark_authority(
        grammar_path=GRAMMAR_PATH,
        start_symbols=START_SYMBOLS,
        canonical_serialize=mini_canonicalize,
        static_validate=mini_static_validate,
        scope_policy=lambda source: (("document", mini_canonicalize(source)),),
        completion_frontier=lambda prefix: (
            frozenset({"let", "task"}) if not prefix.strip() else frozenset({"$END"})
        ),
        witness_candidates=_mini_witness_candidates,
    )


@dataclass(frozen=True)
class MiniPackBundleV1:
    """The complete pack plus the slots ``DslPack`` has no field for
    (schema contract, SemanticFrame provider)."""

    pack: DslPack
    schema: CAP1SchemaV1
    derive_frame: Callable[[str], SemanticFrameV1]
    canonicalize: Callable[[str], str]


def build_mini_pack() -> MiniPackBundleV1:
    backend = LarkFileBackend(
        dsl_id=DSL_ID,
        grammar_path=GRAMMAR_PATH,
        description="mini-flow generic test-only DSL (SLM-367 two-pack freeze)",
        root_name="task",
        call_as_component=False,
        start="start",
    )
    pack = DslPack(
        pack_id=DSL_ID,
        backend=backend,
        placeholder_policy=PlaceholderPolicy(
            placeholder_re=PLACEHOLDER_RE,
            content_props=CONTENT_PROPS,
            slot_contract=_mini_slot_contract,
        ),
        # F3 honesty: the oracle proves syntax-level well-formedness only.
        reward_label="syntax_only",
        canonicalize=mini_canonicalize,
        oracle=_mini_oracle,
        fragment_parser=_mini_fragment_parser,
        grammar_capability_authority=_mini_authority(),
    )
    return MiniPackBundleV1(
        pack=pack,
        schema=mini_schema(),
        derive_frame=derive_mini_frame,
        canonicalize=mini_canonicalize,
    )


#: Capabilities a pack must demonstrate before it may support a *generic*
#: grammar/schema-grounding claim (DSH2-07 stop rule).
GENERIC_CLAIM_REQUIREMENTS = (
    "grammar_backend",
    "canonicalizer",
    "static_validator",
    "schema_contract",
    "semantic_frame_provider",
    "marker_policy",
    "fragment_parse",
    "grammar_capability_authority",
)


@dataclass(frozen=True)
class PackCompletenessReportV1:
    pack_id: str
    missing: tuple[str, ...]
    probe_failures: Mapping[str, str]
    complete: bool


def completeness_report(bundle: MiniPackBundleV1) -> PackCompletenessReportV1:
    """Fail-closed conformance check, modeled on the DSH1-01 adapter: every
    required capability must be present *and* pass a smoke probe."""
    pack = bundle.pack
    authority = pack.grammar_capability_authority
    present: dict[str, bool] = {
        "grammar_backend": pack.backend is not None and pack.backend.available(),
        "canonicalizer": pack.canonicalize is not None and bundle.canonicalize is not None,
        "static_validator": authority is not None and authority.static_validate is not None,
        "schema_contract": bundle.schema is not None,
        "semantic_frame_provider": callable(bundle.derive_frame),
        "marker_policy": pack.placeholder_policy is not None,
        "fragment_parse": pack.fragment_parser is not None
        and authority is not None
        and authority.fragment_parse is not None,
        "grammar_capability_authority": authority is not None,
    }
    missing = tuple(name for name in GENERIC_CLAIM_REQUIREMENTS if not present.get(name))
    probes: dict[str, str] = {}
    if not missing:
        try:
            canon = bundle.canonicalize(WITNESS_SOURCE)
            if bundle.canonicalize(canon) != canon:
                probes["canonicalizer"] = "round-trip not idempotent"
        except Exception as exc:  # noqa: BLE001 - probe failure is data
            probes["canonicalizer"] = str(exc)
        try:
            mini_static_validate(WITNESS_SOURCE)
            try:
                mini_static_validate("task { mode fast }")
                probes["static_validator"] = "accepted invalid source"
            except Exception:  # noqa: BLE001 - rejection is the expected path
                pass
        except Exception as exc:  # noqa: BLE001
            probes["static_validator"] = str(exc)
        try:
            frame = bundle.derive_frame(WITNESS_SOURCE)
            if frame.schema_fingerprint != bundle.schema.fingerprint():
                probes["semantic_frame_provider"] = "frame/schema fingerprint mismatch"
        except Exception as exc:  # noqa: BLE001
            probes["semantic_frame_provider"] = str(exc)
        try:
            pack.require("fragment_parser")("mode safe", "fragment", "prop_stmt")
        except Exception as exc:  # noqa: BLE001
            probes["fragment_parse"] = str(exc)
    return PackCompletenessReportV1(
        pack_id=pack.pack_id,
        missing=missing,
        probe_failures=probes,
        complete=not missing and not probes,
    )


def require_complete_for_generic_claim(bundle: MiniPackBundleV1) -> PackCompletenessReportV1:
    """Return the report when complete; raise :class:`PackIncompleteError`
    otherwise. A pack that fails here must NOT be used to support a generic
    claim (DSH2-07 stop rule)."""
    report = completeness_report(bundle)
    if not report.complete:
        raise PackIncompleteError(
            f"pack {report.pack_id!r} cannot support a generic claim; "
            f"missing={report.missing!r} probe_failures={dict(report.probe_failures)!r}"
        )
    return report


__all__ = [
    "CONTENT_PROPS",
    "DSL_ID",
    "GENERIC_CLAIM_REQUIREMENTS",
    "GRAMMAR_PATH",
    "MODE_DOMAIN",
    "MiniPackBundleV1",
    "PackCompletenessReportV1",
    "PackIncompleteError",
    "START_SYMBOLS",
    "WITNESS_SOURCE",
    "build_mini_pack",
    "completeness_report",
    "derive_mini_frame",
    "mini_canonicalize",
    "mini_schema",
    "mini_static_validate",
    "require_complete_for_generic_claim",
]
