"""DSH2-07 (SLM-367): immutable two-pack generic CAP1 eval + schema-sensitivity suite.

Freezes a pair of CAP1 suites — OpenUI and the generic test-only mini-flow
pack (``dsl/mini_pack.py``) — with **disjoint semantic root families**, so a
generic grammar/schema-grounding claim rests on two independent languages
instead of OpenUI (or DESIGN.md) memorization.  The mini pack never imports
OpenUI; the suite-level stop rule refuses to let an incomplete second pack
support a generic claim (``require_complete_for_generic_claim``).

Strata per base case (all fixture scale):

* ``original`` — prompt + exact schema contract;
* ``schema_empty`` / ``schema_contradictory`` — *relevant* perturbations: the
  intended decision must change (abstain / flip);
* ``schema_reordered`` / ``schema_irrelevant`` — perturbations that must leave
  the intended decision approximately invariant;
* ``marker_permutation`` — DSH1-04 marker-surface rotation (SLM-366
  ``marker_permutation`` authority), semantics invariant;
* ``paraphrase`` — SLM-365 frame paraphrase, same answer;
* ``counterfactual`` — SLM-366 one-fact pair (OpenUI) / DSH2-02
  single-fact flip (mini-flow), decision must change;
* ``ambiguity`` — accepted-set-scored rows, plus explicitly excluded rows with
  a recorded reason (no SLM-344 accepted-set evidence exists in-repo yet, so
  accepted sets are local fixture evidence and that is recorded);
* ``cap0_retention`` — identity round-trip rows; retention is a mandatory gate.

Every row is sha256-addressed; the suite carries a tamper-evident manifest and
per-pack suite cards with exact generation/adjudication provenance.
:func:`verify_suite_integrity` is fail-closed.  :func:`freeze_suite` writes the
suite create-once (claim-manifest freeze pattern); rewriting different content
under the same path raises.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from slm_training.dsl.canonicalize import canonicalize as openui_canonicalize
from slm_training.dsl.mini_pack import (
    DSL_ID as MINI_DSL_ID,
    MiniPackBundleV1,
    build_mini_pack,
    require_complete_for_generic_claim,
)
from slm_training.dsl.semantic_frame import (
    CAP1SchemaV1,
    SemanticFrameV1,
    SemanticFactV1,
    derive_semantic_frame,
    verify_single_fact_change,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.train_data.frame_paraphrases import (
    FrameParaphraseFixtureProviderV1,
    generate_frame_paraphrases,
)
from slm_training.harnesses.train_data.semantic_counterfactuals import (
    CounterfactualPairV1,
    CounterfactualRejectionV1,
    generate_counterfactual_pair,
    marker_permutation,
    permute_marker_surfaces,
    root_family_for,
)
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

SCHEMA_VERSION = "cap1_two_pack_freeze/v1"

STRATUM_ORIGINAL = "original"
STRATUM_SCHEMA_EMPTY = "schema_empty"
STRATUM_SCHEMA_CONTRADICTORY = "schema_contradictory"
STRATUM_SCHEMA_REORDERED = "schema_reordered"
STRATUM_SCHEMA_IRRELEVANT = "schema_irrelevant"
STRATUM_MARKER_PERMUTATION = "marker_permutation"
STRATUM_PARAPHRASE = "paraphrase"
STRATUM_COUNTERFACTUAL = "counterfactual"
STRATUM_AMBIGUITY = "ambiguity"
STRATUM_CAP0_RETENTION = "cap0_retention"
STRATA = (
    STRATUM_ORIGINAL,
    STRATUM_SCHEMA_EMPTY,
    STRATUM_SCHEMA_CONTRADICTORY,
    STRATUM_SCHEMA_REORDERED,
    STRATUM_SCHEMA_IRRELEVANT,
    STRATUM_MARKER_PERMUTATION,
    STRATUM_PARAPHRASE,
    STRATUM_COUNTERFACTUAL,
    STRATUM_AMBIGUITY,
    STRATUM_CAP0_RETENTION,
)
#: Perturbations that MUST change the intended decision.
RELEVANT_STRATA = (STRATUM_SCHEMA_EMPTY, STRATUM_SCHEMA_CONTRADICTORY, STRATUM_COUNTERFACTUAL)
#: Perturbations that MUST leave the intended decision approximately invariant.
INVARIANT_STRATA = (STRATUM_SCHEMA_REORDERED, STRATUM_SCHEMA_IRRELEVANT)
#: Prompt-surface strata that must not change the intended answer.
PROMPT_INVARIANT_STRATA = (STRATUM_PARAPHRASE, STRATUM_MARKER_PERMUTATION)

DISPOSITION_DETERMINATE = "determinate"
DISPOSITION_ACCEPTED_SET = "accepted_set"
DISPOSITION_EXCLUDED = "excluded"

OPENUI_PACK_ID = "openui"
PACK_IDS = (OPENUI_PACK_ID, MINI_DSL_ID)

ABSTAIN = "abstain"

#: No SLM-344 accepted-set evidence exists in-repo; ambiguous rows either carry
#: a local fixture accepted set (recorded as such) or are excluded with reason.
AMBIGUITY_PROVENANCE_NOTE = (
    "no SLM-344 accepted-set evidence in-repo; accepted sets are local "
    "fixture evidence, exclusions are recorded with reasons"
)


class TwoPackFreezeError(ValueError):
    """Suite construction, integrity, or gate failure."""


# --------------------------------------------------------------------------
# Suite rows / cards / manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SuiteRowV1:
    row_id: str
    pack_id: str
    group_id: str
    root_family_id: str
    split: str
    stratum: str
    prompt: str
    prompt_sha256: str
    schema_text: str
    schema_sha256: str
    targets: tuple[str, ...]
    target_sha256s: tuple[str, ...]
    candidates: tuple[str, ...]
    required_facts: tuple[str, ...]
    forbidden_facts: tuple[str, ...]
    ambiguity_disposition: str
    exclusion_reason: str
    provenance: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def row_sha256(self) -> str:
        return content_sha(self.to_dict())


@dataclass(frozen=True)
class SuiteCardV1:
    pack_id: str
    rows_per_stratum: Mapping[str, int]
    root_family_ids: tuple[str, ...]
    generation_provenance: Mapping[str, Any]
    adjudication_provenance: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def card_sha256(self) -> str:
        return content_sha(self.to_dict())


@dataclass(frozen=True)
class Cap1TwoPackSuiteV1:
    schema_version: str
    rows: tuple[SuiteRowV1, ...]
    cards: tuple[SuiteCardV1, ...]
    manifest: Mapping[str, Any]
    suite_sha256: str
    provenance: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rows": [row.to_dict() for row in self.rows],
            "cards": [card.to_dict() for card in self.cards],
            "manifest": dict(self.manifest),
            "suite_sha256": self.suite_sha256,
            "provenance": dict(self.provenance),
        }

    def rows_for(self, pack_id: str, stratum: str | None = None) -> tuple[SuiteRowV1, ...]:
        return tuple(
            row
            for row in self.rows
            if row.pack_id == pack_id and (stratum is None or row.stratum == stratum)
        )


@dataclass(frozen=True)
class IntegrityReportV1:
    ok: bool
    violations: tuple[str, ...]
    checked_rows: int
    manifest_sha256: str


# --------------------------------------------------------------------------
# Fact signatures + closed-domain usage (schema contract projection)
# --------------------------------------------------------------------------


def fact_signature(fact: SemanticFactV1) -> str:
    value = json.dumps(fact.value, sort_keys=True, default=str)
    return f"{fact.category}:{fact.predicate}:{fact.schema_field}:{value}"


def _frame_fact_signatures(frame: SemanticFrameV1) -> tuple[tuple[str, ...], tuple[str, ...]]:
    required = tuple(sorted(fact_signature(f) for f in frame.facts if f.category == "required"))
    forbidden = tuple(sorted(fact_signature(f) for f in frame.facts if f.category == "forbidden"))
    return required, forbidden


def closed_value_domains_from_frame(frame: SemanticFrameV1) -> dict[str, tuple[Any, ...]]:
    """Reconstruct each closed domain from a frame's required ``closed_value``
    fact plus its schema-entailed forbidden complements (DSH2-02 contract)."""
    used: dict[str, Any] = {}
    excluded: dict[str, set[Any]] = {}
    for fact in frame.facts:
        if fact.schema_field == "":
            continue
        if fact.category == "required" and fact.predicate == "closed_value":
            used[fact.schema_field] = fact.value
        elif fact.category == "forbidden" and fact.predicate == "excluded_value":
            excluded.setdefault(fact.schema_field, set()).add(fact.value)
    return {
        field: tuple(sorted({used[field], *excluded.get(field, set())}, key=repr))
        for field in sorted(used)
    }


def closed_value_usage(pack_id: str, source: str) -> dict[str, Any]:
    """Closed-domain values asserted by ``source`` under the pack's frame
    provider. Shared by the builder and by adjudication ``decide_fn`` code."""
    frame = _derive_frame(pack_id, source)
    return {
        fact.schema_field: fact.value
        for fact in frame.facts
        if fact.category == "required" and fact.predicate == "closed_value"
    }


def component_usage(pack_id: str, source: str) -> frozenset[str]:
    """Component type names used by ``source`` under the pack's frame provider."""
    return frozenset(node.type_name for node in _derive_frame(pack_id, source).nodes)


def _derive_frame(pack_id: str, source: str) -> SemanticFrameV1:
    if pack_id == OPENUI_PACK_ID:
        return derive_semantic_frame(source, CAP1SchemaV1.from_pack(OPENUI_PACK_ID))
    if pack_id == MINI_DSL_ID:
        return build_mini_pack().derive_frame(source)
    raise TwoPackFreezeError(f"unknown pack {pack_id!r}")


def canonical_form(pack_id: str, source: str) -> str:
    if pack_id == OPENUI_PACK_ID:
        return openui_canonicalize(source, dsl=OPENUI_PACK_ID)
    if pack_id == MINI_DSL_ID:
        return build_mini_pack().canonicalize(source)
    raise TwoPackFreezeError(f"unknown pack {pack_id!r}")


def _schema_projection(pack_id: str, schema: CAP1SchemaV1, frames: Sequence[SemanticFrameV1]) -> dict[str, Any]:
    domains: dict[str, set[Any]] = {}
    for frame in frames:
        for field_name, values in closed_value_domains_from_frame(frame).items():
            domains.setdefault(field_name, set()).update(values)
    return {
        "dsl": schema.dsl,
        "component_names": sorted(schema.component_names),
        "content_props": sorted(schema.content_props),
        "closed_value_domains": {
            name: sorted(values, key=repr) for name, values in sorted(domains.items())
        },
    }


def _schema_text(projection: Mapping[str, Any]) -> str:
    return json.dumps(projection, sort_keys=True, default=str)


def _contradict_schema(
    projection: Mapping[str, Any],
    used: Mapping[str, Any],
    used_components: set[str],
) -> str:
    """Remove every closed-domain value the original target asserts and every
    component the target uses — the schema now contradicts the intended answer
    (a relevant perturbation), whether the case is closed-value- or
    role/topology-shaped."""
    mutated = json.loads(json.dumps(projection, default=str))
    domains = mutated.get("closed_value_domains", {})
    for field_name, value in used.items():
        if field_name in domains:
            domains[field_name] = [v for v in domains[field_name] if v != value]
    if "component_names" in mutated:
        mutated["component_names"] = [
            name for name in mutated["component_names"] if name not in used_components
        ]
    return json.dumps(mutated, sort_keys=True, default=str)


def _reorder_schema(schema_text: str) -> str:
    """Same mapping, reversed key order — semantically identical schema."""
    payload = json.loads(schema_text)
    reversed_keys = {key: payload[key] for key in reversed(list(payload))}
    return json.dumps(reversed_keys, default=str)


def _irrelevant_schema(schema_text: str) -> str:
    """Original schema plus an unrelated extension the decision never reads."""
    payload = json.loads(schema_text)
    payload["unrelated_extension"] = {"release_notes": ["n/a"], "owner": "fixture"}
    return json.dumps(payload, sort_keys=True, default=str)


# --------------------------------------------------------------------------
# Base-case construction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _BaseCase:
    pack_id: str
    group_id: str
    canonical_target: str
    prompt: str
    paraphrase_prompt: str
    counterfactual_target: str
    counterfactual_prompt: str
    ambiguous_prompt: str
    frame: SemanticFrameV1
    counterfactual_frame: SemanticFrameV1
    generation: Mapping[str, Any]


_OPENUI_CASES = (
    (
        "openui-stack",
        'a = TextContent(":tp7.a", 14)\n'
        'b = Separator("horizontal", true)\n'
        'c = TextContent(":tp7.c")\n'
        'root = Stack([a, b, c], "column", 8, "start", "start", "nowrap")\n',
        "closed_value",
        "Lay out the three items in a stack.",
    ),
    (
        "openui-form",
        'bt = Button(":tp7.bl", null, "primary", "button", "md")\n'
        'in_ = Input("email", ":tp7.ph", "text")\n'
        'root = Form("f", [bt], [in_])',
        "role",
        "Build the form with its controls.",
    ),
)

_MINI_CASES = (
    (
        "mini-build",
        'let tag = ":f.tag"\ntask build { step ":f.desc" mode fast retry 2 }\n',
        "Run task build with step :f.desc and retry 2.",
    ),
    (
        "mini-deploy",
        'let owner = ":f.owner"\ntask deploy { step ":f.note" mode safe retry 1 }\n',
        "Deploy with owner :f.owner, step :f.note, retry 1.",
    ),
)


def _mini_prompt(source: str, *, style: str) -> str:
    frame = build_mini_pack().derive_frame(source)
    tasks: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    lets: dict[str, Any] = {}
    for fact in frame.facts:
        if fact.predicate == "let_binding":
            lets[str(fact.schema_field)] = fact.value
        if len(fact.ast_path) >= 2 and fact.ast_path[0] == "tasks":
            key = str(fact.ast_path[1])
            tasks.setdefault(key, {})[fact.schema_field] = fact.value
            if key not in order:
                order.append(key)
    names = [node.node_id.split(":", 1)[1] for node in frame.nodes]
    parts: list[str] = []
    for index, key in enumerate(order):
        props = tasks[key]
        name = names[index] if index < len(names) else key
        if style == "concise":
            parts.append(
                f"task {name}; step {props.get('step')}; mode {props.get('mode')}; "
                f"retry {props.get('retry')}"
            )
        else:
            parts.append(
                f"run task {name}: set retry {props.get('retry')}, use mode "
                f"{props.get('mode')}, step {props.get('step')}"
            )
    prefix = "".join(f"{name} is {value}; " for name, value in lets.items())
    body = "; ".join(parts)
    return f"Flow: {prefix}{body}."


def _build_openui_cases(now: Callable[[], str]) -> tuple[_BaseCase, ...]:
    schema = CAP1SchemaV1.from_pack(OPENUI_PACK_ID)
    provider = FrameParaphraseFixtureProviderV1()
    cases: list[_BaseCase] = []
    for group_id, source, dimension, ambiguous_prompt in _OPENUI_CASES:
        frame = derive_semantic_frame(source, schema)
        pair = generate_counterfactual_pair(frame, dimension, schema=schema, now=now)
        if isinstance(pair, CounterfactualRejectionV1):  # fail closed at build time
            raise TwoPackFreezeError(
                f"counterfactual rejected for {group_id}: {pair.code} {pair.detail}"
            )
        assert isinstance(pair, CounterfactualPairV1)
        paraphrases = generate_frame_paraphrases(frame, provider=provider, now=now)
        paraphrase_row = next(row for row in paraphrases.rows if row.style == "imperative")
        cf_frame = derive_semantic_frame(pair.counterfactual_target, schema)
        cases.append(
            _BaseCase(
                pack_id=OPENUI_PACK_ID,
                group_id=group_id,
                canonical_target=frame.canonical_source,
                prompt=pair.source_prompt,
                paraphrase_prompt=paraphrase_row.prompt_text,
                counterfactual_target=pair.counterfactual_target,
                counterfactual_prompt=pair.counterfactual_prompt,
                ambiguous_prompt=ambiguous_prompt,
                frame=frame,
                counterfactual_frame=cf_frame,
                generation={
                    "counterfactual": {
                        "module": "semantic_counterfactuals (SLM-366)",
                        "dimension": dimension,
                        "pair_id": pair.pair_id,
                        "single_dimension_change": pair.proof.single_dimension_change,
                    },
                    "paraphrase": {
                        "module": "frame_paraphrases (SLM-365)",
                        "provider_id": provider.provider_id,
                        "model_id": provider.model_id,
                        "model_revision": provider.model_revision,
                        "network": False,
                    },
                },
            )
        )
    return tuple(cases)


def _build_mini_cases(bundle: MiniPackBundleV1) -> tuple[_BaseCase, ...]:
    cases: list[_BaseCase] = []
    for group_id, source, ambiguous_prompt in _MINI_CASES:
        canonical = bundle.canonicalize(source)
        frame = bundle.derive_frame(canonical)
        usage = {
            fact.schema_field: fact.value
            for fact in frame.facts
            if fact.category == "required" and fact.predicate == "closed_value"
        }
        if "mode" not in usage:  # fail closed: fixture must carry the closed domain
            raise TwoPackFreezeError(f"mini case {group_id} lacks a mode fact")
        alternative = next(v for v in bundle.schema.closed_value_domains["mode"] if v != usage["mode"])
        cf_source = canonical.replace(f"mode {usage['mode']}", f"mode {alternative}")
        cf_frame = bundle.derive_frame(cf_source)
        proof = verify_single_fact_change(frame, cf_frame)
        cases.append(
            _BaseCase(
                pack_id=MINI_DSL_ID,
                group_id=group_id,
                canonical_target=canonical,
                prompt=_mini_prompt(canonical, style="concise"),
                paraphrase_prompt=_mini_prompt(canonical, style="imperative"),
                counterfactual_target=cf_frame.canonical_source,
                counterfactual_prompt=_mini_prompt(cf_frame.canonical_source, style="concise"),
                ambiguous_prompt=ambiguous_prompt,
                frame=frame,
                counterfactual_frame=cf_frame,
                generation={
                    "counterfactual": {
                        "module": "dsl.semantic_frame.verify_single_fact_change (DSH2-02)",
                        "schema_field": proof.schema_field,
                        "old_value": proof.old_value,
                        "new_value": proof.new_value,
                        "single_fact_change": proof.single_fact_change,
                    },
                    "paraphrase": {
                        "module": "cap1_two_pack_freeze._mini_prompt",
                        "style": "imperative",
                        "network": False,
                    },
                },
            )
        )
    return tuple(cases)


# --------------------------------------------------------------------------
# Suite builder
# --------------------------------------------------------------------------


def _make_row(
    *,
    case: _BaseCase,
    family_id: str,
    split: str,
    stratum: str,
    prompt: str,
    schema_text: str,
    targets: tuple[str, ...],
    candidates: tuple[str, ...],
    disposition: str = DISPOSITION_DETERMINATE,
    exclusion_reason: str = "",
    provenance: Mapping[str, Any],
) -> SuiteRowV1:
    # Required/forbidden facts describe THIS row's own intended target (the
    # counterfactual and marker-permutation strata have different frames).
    fact_frame = _derive_frame(case.pack_id, targets[0]) if targets else case.frame
    required, forbidden = _frame_fact_signatures(fact_frame)
    payload = {
        "pack_id": case.pack_id,
        "group_id": case.group_id,
        "stratum": stratum,
        "prompt_sha256": content_sha(prompt),
        "target_sha256s": tuple(content_sha(t) for t in targets),
    }
    return SuiteRowV1(
        row_id=content_sha(payload),
        pack_id=case.pack_id,
        group_id=case.group_id,
        root_family_id=family_id,
        split=split,
        stratum=stratum,
        prompt=prompt,
        prompt_sha256=payload["prompt_sha256"],
        schema_text=schema_text,
        schema_sha256=content_sha(schema_text),
        targets=targets,
        target_sha256s=payload["target_sha256s"],
        candidates=candidates,
        required_facts=required,
        forbidden_facts=forbidden,
        ambiguity_disposition=disposition,
        exclusion_reason=exclusion_reason,
        provenance=dict(provenance),
    )


def build_cap1_two_pack_suite(
    *,
    seed: int = 0,
    now: Callable[[], str] | None = None,
    include_cap0: bool = True,
    include_ambiguity: bool = True,
) -> Cap1TwoPackSuiteV1:
    """Build (deterministically) the frozen two-pack suite. The mini pack must
    pass the generic-claim completeness check first — the stop rule."""
    now = now or (lambda: "2026-07-25T00:00:00+00:00")
    bundle = build_mini_pack()
    completeness = require_complete_for_generic_claim(bundle)

    cases = _build_openui_cases(now) + _build_mini_cases(bundle)
    policy = RootFamilySplitPolicyV1()

    frames_by_pack: dict[str, list[SemanticFrameV1]] = {OPENUI_PACK_ID: [], MINI_DSL_ID: []}
    for case in cases:
        frames_by_pack[case.pack_id].append(case.frame)
    schemas = {
        OPENUI_PACK_ID: CAP1SchemaV1.from_pack(OPENUI_PACK_ID),
        MINI_DSL_ID: bundle.schema,
    }
    schema_texts = {
        pack_id: _schema_text(_schema_projection(pack_id, schemas[pack_id], frames_by_pack[pack_id]))
        for pack_id in PACK_IDS
    }

    rows: list[SuiteRowV1] = []
    for case_index, case in enumerate(cases):
        family_id = root_family_for(case.canonical_target, case.pack_id)
        split = policy.assign(family_id)
        schema_text = schema_texts[case.pack_id]
        used = closed_value_usage(case.pack_id, case.canonical_target)
        used_components = {node.type_name for node in case.frame.nodes}
        pair_targets = (case.canonical_target, case.counterfactual_target)
        prov = {"case": dict(case.generation), "built_at": now(), "seed": seed}

        permutation = marker_permutation(case.canonical_target, seed=seed + case_index + 1)
        permuted_prompt = permute_marker_surfaces(case.prompt, permutation)
        permuted_targets = tuple(permute_marker_surfaces(t, permutation) for t in pair_targets)

        rows.append(
            _make_row(case=case, family_id=family_id, split=split, stratum=STRATUM_ORIGINAL,
                      prompt=case.prompt, schema_text=schema_text, targets=(case.canonical_target,),
                      candidates=pair_targets, provenance=prov)
        )
        rows.append(
            _make_row(case=case, family_id=family_id, split=split, stratum=STRATUM_SCHEMA_EMPTY,
                      prompt=case.prompt, schema_text="{}", targets=(case.canonical_target,),
                      candidates=pair_targets, provenance=prov)
        )
        rows.append(
            _make_row(case=case, family_id=family_id, split=split, stratum=STRATUM_SCHEMA_CONTRADICTORY,
                      prompt=case.prompt,
                      schema_text=_contradict_schema(json.loads(schema_text), used, used_components),
                      targets=(case.canonical_target,), candidates=pair_targets, provenance=prov)
        )
        rows.append(
            _make_row(case=case, family_id=family_id, split=split, stratum=STRATUM_SCHEMA_REORDERED,
                      prompt=case.prompt, schema_text=_reorder_schema(schema_text),
                      targets=(case.canonical_target,), candidates=pair_targets, provenance=prov)
        )
        rows.append(
            _make_row(case=case, family_id=family_id, split=split, stratum=STRATUM_SCHEMA_IRRELEVANT,
                      prompt=case.prompt, schema_text=_irrelevant_schema(schema_text),
                      targets=(case.canonical_target,), candidates=pair_targets, provenance=prov)
        )
        rows.append(
            _make_row(case=case, family_id=family_id, split=split, stratum=STRATUM_MARKER_PERMUTATION,
                      prompt=permuted_prompt, schema_text=schema_text, targets=(permuted_targets[0],),
                      candidates=permuted_targets,
                      provenance={**prov, "marker_permutation": dict(permutation)})
        )
        rows.append(
            _make_row(case=case, family_id=family_id, split=split, stratum=STRATUM_PARAPHRASE,
                      prompt=case.paraphrase_prompt, schema_text=schema_text,
                      targets=(case.canonical_target,), candidates=pair_targets, provenance=prov)
        )
        rows.append(
            _make_row(case=case, family_id=family_id, split=split, stratum=STRATUM_COUNTERFACTUAL,
                      prompt=case.counterfactual_prompt, schema_text=schema_text,
                      targets=(case.counterfactual_target,),
                      candidates=(case.counterfactual_target, case.canonical_target), provenance=prov)
        )
        if include_ambiguity:
            rows.append(
                _make_row(case=case, family_id=family_id, split=split, stratum=STRATUM_AMBIGUITY,
                          prompt=case.ambiguous_prompt, schema_text=schema_text,
                          targets=pair_targets, candidates=pair_targets,
                          disposition=DISPOSITION_ACCEPTED_SET,
                          provenance={**prov, "ambiguity_evidence": AMBIGUITY_PROVENANCE_NOTE})
            )
        if include_cap0:
            rows.append(
                _make_row(case=case, family_id=family_id, split=split,
                          stratum=STRATUM_CAP0_RETENTION,
                          prompt=f"Reproduce exactly:\n{case.canonical_target}",
                          schema_text=schema_text, targets=(case.canonical_target,),
                          candidates=pair_targets,
                          provenance={**prov, "cap0": "identity round-trip retention"})
            )

    if include_ambiguity:
        # One explicitly excluded ambiguous row (no accepted-set license).
        case = cases[0]
        family_id = root_family_for(case.canonical_target, case.pack_id)
        rows.append(
            _make_row(case=case, family_id=family_id, split=policy.assign(family_id),
                      stratum=STRATUM_AMBIGUITY,
                      prompt="Make something nice.",
                      schema_text=schema_texts[case.pack_id], targets=(), candidates=(),
                      disposition=DISPOSITION_EXCLUDED,
                      exclusion_reason=(
                          "prompt underdetermines the component choice beyond fixture "
                          "evidence; no SLM-344 accepted-set license — excluded from scoring"
                      ),
                      provenance={"ambiguity_evidence": AMBIGUITY_PROVENANCE_NOTE, "built_at": now()})
        )

    cards = _build_cards(tuple(rows), cases, completeness_report=completeness)
    manifest = _build_manifest(tuple(rows), cards, now)
    return Cap1TwoPackSuiteV1(
        schema_version=SCHEMA_VERSION,
        rows=tuple(rows),
        cards=cards,
        manifest=manifest,
        suite_sha256=manifest["manifest_sha256"],
        provenance={
            "builder": "cap1_two_pack_freeze.build_cap1_two_pack_suite",
            "built_at": now(),
            "seed": seed,
            "packs": PACK_IDS,
            "strata": STRATA,
            "ambiguity_evidence": AMBIGUITY_PROVENANCE_NOTE,
            "stop_rule": "mini pack must pass require_complete_for_generic_claim",
        },
    )


def _build_cards(
    rows: tuple[SuiteRowV1, ...],
    cases: tuple[_BaseCase, ...],
    *,
    completeness_report: Any,
) -> tuple[SuiteCardV1, ...]:
    cards: list[SuiteCardV1] = []
    for pack_id in PACK_IDS:
        pack_rows = [row for row in rows if row.pack_id == pack_id]
        per_stratum: dict[str, int] = {}
        for row in pack_rows:
            per_stratum[row.stratum] = per_stratum.get(row.stratum, 0) + 1
        generation: dict[str, Any] = {
            "cases": {
                case.group_id: dict(case.generation) for case in cases if case.pack_id == pack_id
            },
            "schema_contract": "CAP1SchemaV1 projection + frame-derived closed domains",
        }
        if pack_id == MINI_DSL_ID:
            generation["completeness"] = {
                "complete": completeness_report.complete,
                "missing": list(completeness_report.missing),
                "probe_failures": dict(completeness_report.probe_failures),
            }
        cards.append(
            SuiteCardV1(
                pack_id=pack_id,
                rows_per_stratum=per_stratum,
                root_family_ids=tuple(sorted({row.root_family_id for row in pack_rows})),
                generation_provenance=generation,
                adjudication_provenance={
                    "decide_contract": "decide_fn(row) -> canonical target text or 'abstain'",
                    "scorers": (
                        "score_canonical_equivalence",
                        "score_fact_satisfaction",
                        "score_schema_sensitivity",
                        "score_prompt_invariance",
                        "determinacy calibration",
                        "cap0 retention (mandatory)",
                    ),
                    "relevant_strata": RELEVANT_STRATA,
                    "invariant_strata": INVARIANT_STRATA,
                },
            )
        )
    return tuple(cards)


def _build_manifest(
    rows: tuple[SuiteRowV1, ...], cards: tuple[SuiteCardV1, ...], now: Callable[[], str]
) -> dict[str, Any]:
    payload = {
        "schema": f"{SCHEMA_VERSION}/manifest",
        "created_at": now(),
        "rows": {row.row_id: row.row_sha256() for row in rows},
        "cards": {card.pack_id: card.card_sha256() for card in cards},
    }
    payload["manifest_sha256"] = content_sha(payload)
    return payload


# --------------------------------------------------------------------------
# Integrity (fail-closed) + create-once freeze
# --------------------------------------------------------------------------


def verify_suite_integrity(suite: Cap1TwoPackSuiteV1) -> IntegrityReportV1:
    """Recompute every row/card/manifest hash plus the structural invariants.
    Fail-closed: any violation makes ``ok`` False; nothing is silently passed."""
    violations: list[str] = []
    manifest_rows = suite.manifest.get("rows", {})
    for row in suite.rows:
        if row.row_sha256() != manifest_rows.get(row.row_id):
            violations.append(f"row {row.row_id} hash mismatch (tampered)")
        if row.prompt_sha256 != content_sha(row.prompt):
            violations.append(f"row {row.row_id} prompt hash mismatch")
        if row.schema_sha256 != content_sha(row.schema_text):
            violations.append(f"row {row.row_id} schema hash mismatch")
        if row.target_sha256s != tuple(content_sha(t) for t in row.targets):
            violations.append(f"row {row.row_id} target hash mismatch")
        if row.stratum not in STRATA:
            violations.append(f"row {row.row_id} undeclared stratum {row.stratum!r}")
        if row.ambiguity_disposition == DISPOSITION_EXCLUDED and not row.exclusion_reason:
            violations.append(f"row {row.row_id} excluded without a recorded reason")
    manifest_cards = suite.manifest.get("cards", {})
    for card in suite.cards:
        if card.card_sha256() != manifest_cards.get(card.pack_id):
            violations.append(f"card {card.pack_id} hash mismatch (tampered)")
    expected_manifest = _build_manifest(suite.rows, suite.cards, lambda: suite.manifest.get("created_at", ""))
    if expected_manifest["manifest_sha256"] != suite.manifest.get("manifest_sha256"):
        violations.append("manifest hash mismatch (tampered)")
    if suite.suite_sha256 != suite.manifest.get("manifest_sha256"):
        violations.append("suite hash does not match manifest")
    families_by_pack: dict[str, set[str]] = {pack_id: set() for pack_id in PACK_IDS}
    for row in suite.rows:
        if row.pack_id not in families_by_pack:
            violations.append(f"row {row.row_id} undeclared pack {row.pack_id!r}")
            continue
        families_by_pack[row.pack_id].add(row.root_family_id)
    overlap = families_by_pack[OPENUI_PACK_ID] & families_by_pack[MINI_DSL_ID]
    if overlap:
        violations.append(f"non-disjoint root families across packs: {sorted(overlap)!r}")
    for pack_id in PACK_IDS:
        present = {row.stratum for row in suite.rows if row.pack_id == pack_id}
        missing = [s for s in STRATA if s not in present]
        if missing:
            violations.append(f"pack {pack_id} missing strata {missing!r}")
    return IntegrityReportV1(
        ok=not violations,
        violations=tuple(violations),
        checked_rows=len(suite.rows),
        manifest_sha256=str(suite.manifest.get("manifest_sha256", "")),
    )


def freeze_suite(suite: Cap1TwoPackSuiteV1, output_dir: Path) -> Path:
    """Write the frozen suite create-once (claim-manifest freeze pattern).
    Rewriting identical content is a no-op; different content raises."""
    integrity = verify_suite_integrity(suite)
    if not integrity.ok:
        raise TwoPackFreezeError(f"refusing to freeze suite with violations: {integrity.violations!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cap1_two_pack_suite.json"
    payload = json.dumps(suite.to_dict(), indent=2, sort_keys=True, default=str) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("suite_sha256") != suite.suite_sha256 or existing != json.loads(payload):
            raise FileExistsError(f"frozen suite already exists with different content: {path}")
        return path
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return path


# --------------------------------------------------------------------------
# Scoring + gates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaSensitivityReportV1:
    per_group: Mapping[str, Mapping[str, Any]]
    relevant_sensitivity: float
    invariant_rate: float
    tolerance: float
    passed: bool


def score_schema_sensitivity(
    suite: Cap1TwoPackSuiteV1,
    decide_fn: Callable[[SuiteRowV1], str],
    *,
    tolerance: float = 0.0,
) -> SchemaSensitivityReportV1:
    """Relevant perturbations must change the intended decision; irrelevant
    perturbations must keep it invariant within ``tolerance`` (fraction of
    invariant rows allowed to flip)."""
    groups: dict[str, dict[str, Any]] = {}
    originals = {
        (row.pack_id, row.group_id): row
        for row in suite.rows
        if row.stratum == STRATUM_ORIGINAL
    }
    relevant_changed = 0
    relevant_total = 0
    invariant_kept = 0
    invariant_total = 0
    for (pack_id, group_id), original in sorted(originals.items()):
        key = f"{pack_id}/{group_id}"
        d0 = decide_fn(original)
        detail: dict[str, Any] = {"original_decision_sha": content_sha(d0), "strata": {}}
        for row in suite.rows:
            if (row.pack_id, row.group_id) != (pack_id, group_id):
                continue
            if row.stratum not in RELEVANT_STRATA + INVARIANT_STRATA:
                continue
            if row.ambiguity_disposition != DISPOSITION_DETERMINATE:
                continue
            decision = decide_fn(row)
            changed = decision != d0
            detail["strata"][row.stratum] = {"changed": changed}
            if row.stratum in RELEVANT_STRATA:
                relevant_total += 1
                relevant_changed += int(changed)
            else:
                invariant_total += 1
                invariant_kept += int(not changed)
        groups[key] = detail
    relevant_sensitivity = relevant_changed / relevant_total if relevant_total else 0.0
    invariant_rate = invariant_kept / invariant_total if invariant_total else 1.0
    passed = relevant_total > 0 and relevant_sensitivity == 1.0 and invariant_rate >= 1.0 - tolerance
    return SchemaSensitivityReportV1(
        per_group=groups,
        relevant_sensitivity=relevant_sensitivity,
        invariant_rate=invariant_rate,
        tolerance=tolerance,
        passed=passed,
    )


@dataclass(frozen=True)
class RowDecisionV1:
    row_id: str
    stratum: str
    decision_sha: str
    canonical_match: bool
    facts_satisfied: bool | None


@dataclass(frozen=True)
class SuiteAdjudicationV1:
    decisions: tuple[RowDecisionV1, ...]
    canonical_accuracy: float
    facts_accuracy: float
    schema_sensitivity: SchemaSensitivityReportV1
    prompt_invariance_rate: float
    determinacy: Mapping[str, Any]
    cap0_retention: Mapping[str, Any]


def _canonical_match(pack_id: str, decision: str, targets: tuple[str, ...]) -> bool:
    if decision == ABSTAIN or not targets:
        return False
    try:
        normalized = canonical_form(pack_id, decision)
    except Exception:  # noqa: BLE001 - unparseable decision never matches
        return False
    return any(canonical_form(pack_id, target) == normalized for target in targets)


def _facts_satisfied(pack_id: str, decision: str, row: SuiteRowV1) -> bool | None:
    if decision == ABSTAIN:
        return None
    try:
        frame = _derive_frame(pack_id, decision)
    except Exception:  # noqa: BLE001
        return None
    signatures = {fact_signature(fact) for fact in frame.facts}
    required_ok = all(sig in signatures for sig in row.required_facts)
    # A forbidden fact is violated when the decision *asserts* the excluded
    # value (a required fact with the same field + value) — forbidden facts in
    # the decision's own frame are the schema-entailed complement, not a hit.
    asserted = {
        (fact.schema_field, json.dumps(fact.value, sort_keys=True, default=str))
        for fact in frame.facts
        if fact.category in {"required", "optional"}
    }
    forbidden_ok = True
    for sig in row.forbidden_facts:
        _, _, field_name, value = sig.split(":", 3)
        if (field_name, value) in asserted:
            forbidden_ok = False
            break
    return required_ok and forbidden_ok


def adjudicate_suite(
    suite: Cap1TwoPackSuiteV1,
    decide_fn: Callable[[SuiteRowV1], str],
    *,
    tolerance: float = 0.0,
) -> SuiteAdjudicationV1:
    decisions: list[RowDecisionV1] = []
    scored = 0
    canonical_hits = 0
    facts_scored = 0
    facts_hits = 0
    invariance_scored = 0
    invariance_hits = 0
    for row in suite.rows:
        if row.ambiguity_disposition == DISPOSITION_EXCLUDED:
            continue
        decision = decide_fn(row)
        match = _canonical_match(row.pack_id, decision, row.targets)
        facts = _facts_satisfied(row.pack_id, decision, row)
        decisions.append(
            RowDecisionV1(
                row_id=row.row_id,
                stratum=row.stratum,
                decision_sha=content_sha(decision),
                canonical_match=match,
                facts_satisfied=facts,
            )
        )
        scored += 1
        canonical_hits += int(match)
        if facts is not None:
            facts_scored += 1
            facts_hits += int(facts)
        if row.stratum in PROMPT_INVARIANT_STRATA:
            invariance_scored += 1
            invariance_hits += int(match)
    determinate = [row for row in suite.rows if row.ambiguity_disposition == DISPOSITION_DETERMINATE]
    accepted_set = [row for row in suite.rows if row.ambiguity_disposition == DISPOSITION_ACCEPTED_SET]
    excluded = [row for row in suite.rows if row.ambiguity_disposition == DISPOSITION_EXCLUDED]
    determinacy_calibrated = (
        all(len(row.targets) == 1 for row in determinate)
        and all(len(row.targets) >= 2 for row in accepted_set)
        and all(bool(row.exclusion_reason) for row in excluded)
    )
    determinacy = {
        "determinate_rows": len(determinate),
        "single_target_rows": sum(1 for row in determinate if len(row.targets) == 1),
        "accepted_set_rows": len(accepted_set),
        "excluded_rows": len(excluded),
        "exclusion_reasons": tuple(row.exclusion_reason for row in excluded),
        "calibrated": determinacy_calibrated,
    }
    cap0_rows = [row for row in suite.rows if row.stratum == STRATUM_CAP0_RETENTION]
    cap0_by_id = {decision.row_id: decision for decision in decisions}
    retained = sum(1 for row in cap0_rows if cap0_by_id[row.row_id].canonical_match)
    per_pack = {
        pack_id: sum(1 for row in cap0_rows if row.pack_id == pack_id) for pack_id in PACK_IDS
    }
    cap0_retention = {
        "rows": len(cap0_rows),
        "rows_per_pack": per_pack,
        "retained": retained,
        "accuracy": retained / len(cap0_rows) if cap0_rows else 0.0,
    }
    return SuiteAdjudicationV1(
        decisions=tuple(decisions),
        canonical_accuracy=canonical_hits / scored if scored else 0.0,
        facts_accuracy=facts_hits / facts_scored if facts_scored else 0.0,
        schema_sensitivity=score_schema_sensitivity(suite, decide_fn, tolerance=tolerance),
        prompt_invariance_rate=(
            invariance_hits / invariance_scored if invariance_scored else 1.0
        ),
        determinacy=determinacy,
        cap0_retention=cap0_retention,
    )


@dataclass(frozen=True)
class GateReportV1:
    passed: bool
    checks: Mapping[str, Mapping[str, Any]]
    failures: tuple[str, ...]


def evaluate_gates(suite: Cap1TwoPackSuiteV1, adjudication: SuiteAdjudicationV1) -> GateReportV1:
    """Mandatory gates: CAP0 retention present *and* perfect, determinacy
    calibrated, schema sensitivity passing. A suite without CAP0 retention
    rows fails closed."""
    checks: dict[str, Mapping[str, Any]] = {
        "cap0_retention_present": {
            "passed": all(
                adjudication.cap0_retention["rows_per_pack"].get(pack_id, 0) >= 1
                for pack_id in PACK_IDS
            ),
            "rows_per_pack": adjudication.cap0_retention["rows_per_pack"],
        },
        "cap0_retention_perfect": {
            "passed": adjudication.cap0_retention["rows"] > 0
            and adjudication.cap0_retention["accuracy"] == 1.0,
            "accuracy": adjudication.cap0_retention["accuracy"],
        },
        "determinacy_calibrated": {
            "passed": bool(adjudication.determinacy["calibrated"]),
            "detail": dict(adjudication.determinacy),
        },
        "schema_sensitivity": {
            "passed": adjudication.schema_sensitivity.passed,
            "relevant_sensitivity": adjudication.schema_sensitivity.relevant_sensitivity,
            "invariant_rate": adjudication.schema_sensitivity.invariant_rate,
        },
    }
    failures = tuple(name for name, check in checks.items() if not check["passed"])
    return GateReportV1(passed=not failures, checks=checks, failures=failures)


__all__ = [
    "ABSTAIN",
    "AMBIGUITY_PROVENANCE_NOTE",
    "Cap1TwoPackSuiteV1",
    "DISPOSITION_ACCEPTED_SET",
    "DISPOSITION_DETERMINATE",
    "DISPOSITION_EXCLUDED",
    "GateReportV1",
    "INVARIANT_STRATA",
    "IntegrityReportV1",
    "PACK_IDS",
    "PROMPT_INVARIANT_STRATA",
    "RELEVANT_STRATA",
    "RowDecisionV1",
    "SCHEMA_VERSION",
    "STRATA",
    "SchemaSensitivityReportV1",
    "SuiteAdjudicationV1",
    "SuiteCardV1",
    "SuiteRowV1",
    "TwoPackFreezeError",
    "adjudicate_suite",
    "build_cap1_two_pack_suite",
    "canonical_form",
    "closed_value_usage",
    "component_usage",
    "evaluate_gates",
    "fact_signature",
    "freeze_suite",
    "score_schema_sensitivity",
    "verify_suite_integrity",
]
