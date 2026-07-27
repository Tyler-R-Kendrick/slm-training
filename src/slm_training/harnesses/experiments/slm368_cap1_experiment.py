"""SLM-368 (DSH2-08): matched CAP1 grounding experiment + CERT_CAP1 decision.

Tests whether structured schema plus ambiguity-filtered NL paraphrases teach
portable semantic grounding beyond style and schema-absent controls. Four
matched arms are built from the same two-pack roots with equal source-family
exposure and equal target/compute exposure; model architecture, compiler,
optimizer, decoder, and evaluation are held fixed — only the declared data
lever changes:

* ``NL_NO_SCHEMA`` — NL paraphrase prompts only (schema-absent control);
* ``SCHEMA_UNFILTERED`` — schema side-channel + prompts with NO ambiguity
  filtering (style/unfiltered control);
* ``SCHEMA_FILTERED_SINGLE`` — schema + one accepted (determinate) prompt;
* ``SCHEMA_FILTERED_MULTI`` — schema + multi-style accepted paraphrases.

Evaluation runs the SLM-367 frozen two-pack suite
(:mod:`~slm_training.harnesses.train_data.cap1_two_pack_freeze`) end to end:
OpenUI and the complete mini-flow pack, ambiguity/determinacy, one-fact
counterfactuals, marker permutations, prompt invariance, and the mandatory
CAP0 retention rows. The suite's schema strata (original / empty /
contradictory / reordered / irrelevant) double as the causal-consumption
replay probe: relevant schema perturbations must change the intended
decision; irrelevant perturbations must stay approximately invariant.

``CERT_CAP1`` is issued only when a filtered schema-grounded arm beats both
the schema-absent and the unfiltered/style control on held-out
AST/equivalence rows with paired confidence evidence (Wilson bounds + exact
paired McNemar), relevant schema contradictions change decisions while
irrelevant ones stay invariant, CAP0 retention stays green, and every target
carries only grammar symbols plus declared markers. Fixture ``n`` honestly
reports ``underpowered`` and rejects (mirroring the SLM-362 pattern). Stop
rules — ignores_schema, fails_hard_contrasts, cap0_regression, underpowered
(plus contaminated / prediction_identical wiring stops) — reject the
certificate and keep natural-language CAP2 closed.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from slm_training.data.contract import GenerationRequest
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.power_protocol import (
    exact_paired_binary_test,
    wilson_interval,
)
from slm_training.harnesses.capability_artifacts import (
    CapabilityCertificateV1,
    ProcessProvenanceV1,
)
from slm_training.harnesses.staged import Capability
from slm_training.harnesses.train_data.cap1_two_pack_freeze import (
    ABSTAIN,
    Cap1TwoPackSuiteV1,
    GateReportV1,
    STRATUM_COUNTERFACTUAL,
    STRATUM_MARKER_PERMUTATION,
    STRATUM_ORIGINAL,
    STRATUM_PARAPHRASE,
    SuiteRowV1,
    adjudicate_suite,
    build_cap1_two_pack_suite,
    canonical_form,
    closed_value_domains_from_frame,
    evaluate_gates,
    verify_suite_integrity,
)
from slm_training.harnesses.train_data.frame_paraphrases import (
    FrameParaphraseFixtureProviderV1,
    generate_frame_paraphrases,
)
from slm_training.harnesses.train_data.semantic_counterfactuals import (
    root_family_for,
)

SCHEMA_VERSION = "slm368_cap1_experiment/v1"
EXPERIMENT_ID = "slm368-cap1-experiment"
CERT_CAP1 = "CERT_CAP1"
SUITE_SEED = 7

ArmId = Literal[
    "NL_NO_SCHEMA",
    "SCHEMA_UNFILTERED",
    "SCHEMA_FILTERED_SINGLE",
    "SCHEMA_FILTERED_MULTI",
]
ARMS: tuple[ArmId, ...] = (
    "NL_NO_SCHEMA",
    "SCHEMA_UNFILTERED",
    "SCHEMA_FILTERED_SINGLE",
    "SCHEMA_FILTERED_MULTI",
)
#: Arms whose prompts carry the schema side-channel (train and eval).
SCHEMA_ARMS: tuple[str, ...] = (
    "SCHEMA_UNFILTERED",
    "SCHEMA_FILTERED_SINGLE",
    "SCHEMA_FILTERED_MULTI",
)
#: Filtered schema-grounded candidate arms; the rest are controls.
FILTERED_ARMS: tuple[str, ...] = ("SCHEMA_FILTERED_SINGLE", "SCHEMA_FILTERED_MULTI")
CONTROL_ARMS: tuple[str, ...] = ("NL_NO_SCHEMA", "SCHEMA_UNFILTERED")

#: Declared data capability levers per arm; the ONLY thing that varies.
ARM_LEVERS: dict[str, dict[str, bool]] = {
    "NL_NO_SCHEMA": {
        "schema_side_channel": False,
        "ambiguity_filter": False,
        "multi_style_paraphrases": True,
    },
    "SCHEMA_UNFILTERED": {
        "schema_side_channel": True,
        "ambiguity_filter": False,
        "multi_style_paraphrases": True,
    },
    "SCHEMA_FILTERED_SINGLE": {
        "schema_side_channel": True,
        "ambiguity_filter": True,
        "multi_style_paraphrases": False,
    },
    "SCHEMA_FILTERED_MULTI": {
        "schema_side_channel": True,
        "ambiguity_filter": True,
        "multi_style_paraphrases": True,
    },
}

#: Held-out strata scored for the AST/equivalence paired comparison.
HELDOUT_STRATA: tuple[str, ...] = (
    STRATUM_ORIGINAL,
    STRATUM_PARAPHRASE,
    STRATUM_MARKER_PERMUTATION,
    STRATUM_COUNTERFACTUAL,
)

DEFAULT_SEEDS: tuple[int, ...] = (0, 1)
DEFAULT_STEPS = 8
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 3e-3
DEFAULT_RECORDS_PER_ARM = 16

#: Preregistered minimum effect for a filtered arm vs each control on
#: held-out AST/equivalence rows: strictly positive paired effect, more
#: candidate-only than control-only discordant pairs, and an exact paired
#: McNemar p-value at or below this level.
PREREGISTERED_MIN_EFFECT = 0.0
PREREGISTERED_MAX_P_VALUE = 0.10
#: Preregistered minimum pooled paired n (rows x preregistered seeds) on the
#: held-out AST/equivalence comparison. Production-scale certification needs
#: far more than the frozen fixture suite carries, so fixture runs honestly
#: reject with ``underpowered`` (the SLM-362 pattern).
PREREGISTERED_MIN_PAIRED_N = 64
#: Tolerance for invariant-strata flips (exact-zero: any flip is a failure).
INVARIANT_TOLERANCE = 0.0


class StopRule(str, Enum):
    IGNORES_SCHEMA = "ignores_schema"
    FAILS_HARD_CONTRASTS = "fails_hard_contrasts"
    CAP0_REGRESSION = "cap0_regression"
    UNDERPOWERED = "underpowered"
    CONTAMINATED = "contaminated"
    PREDICTION_IDENTICAL = "prediction_identical"


# --------------------------------------------------------------------------- #
# Train roots (deterministic, two-pack, split-disjoint from the frozen suite)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainCaseV1:
    """One training root: canonical target, prompt styles, declared markers."""

    pack_id: str
    case_id: str
    canonical_target: str
    prompts: tuple[tuple[str, str], ...]  # (style, prompt_text)
    ambiguous_prompt: str
    declared_markers: tuple[str, ...]


def _mini_prompt(source: str, *, style: str) -> str:
    """Deterministic mini-flow prompt renderer (concise / imperative styles)."""
    from slm_training.dsl.mini_pack import build_mini_pack

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
    return f"Flow: {prefix}{'; '.join(parts)}."


def _openui_train_cases() -> tuple[TrainCaseV1, ...]:
    from slm_training.dsl.semantic_frame import CAP1SchemaV1, derive_semantic_frame

    schema = CAP1SchemaV1.from_pack("openui")
    provider = FrameParaphraseFixtureProviderV1()
    sources = (
        (
            "openui-train-stack",
            'x = TextContent(":cap1tr.x", 12)\n'
            'y = Separator("horizontal", true)\n'
            'root = Stack([x, y], "column", 4, "start", "start", "nowrap")\n',
            "Lay out the items in a stack.",
        ),
        (
            "openui-train-form",
            'bt = Button(":cap1tr.go", null, "secondary", "button", "sm")\n'
            'in_ = Input("name", ":cap1tr.ph", "text")\n'
            'root = Form("g", [bt], [in_])',
            "Build the form with its controls.",
        ),
    )
    cases: list[TrainCaseV1] = []
    for case_id, source, ambiguous in sources:
        frame = derive_semantic_frame(source, schema)
        paraphrases = generate_frame_paraphrases(frame, provider=provider)
        prompts = tuple((row.style, row.prompt_text) for row in paraphrases.rows)
        if not prompts:
            raise ValueError(f"no accepted paraphrases for {case_id}")
        cases.append(
            TrainCaseV1(
                pack_id="openui",
                case_id=case_id,
                canonical_target=frame.canonical_source,
                prompts=prompts,
                ambiguous_prompt=ambiguous,
                declared_markers=tuple(sorted(extract_placeholders(source))),
            )
        )
    return tuple(cases)


def _mini_train_cases() -> tuple[TrainCaseV1, ...]:
    from slm_training.dsl.mini_pack import build_mini_pack

    bundle = build_mini_pack()
    sources = (
        (
            "mini-train-lint",
            'let tag = ":ct.a"\ntask lint { step ":ct.s" mode fast retry 1 }\n',
            "Run the task.",
        ),
        (
            "mini-train-pack",
            'let owner = ":ct.o"\ntask pack { step ":ct.p" mode safe retry 3 }\n',
            "Do the flow.",
        ),
    )
    cases: list[TrainCaseV1] = []
    for case_id, source, ambiguous in sources:
        canonical = bundle.canonicalize(source)
        prompts = (
            ("concise", _mini_prompt(canonical, style="concise")),
            ("imperative", _mini_prompt(canonical, style="imperative")),
        )
        cases.append(
            TrainCaseV1(
                pack_id=bundle.schema.dsl,
                case_id=case_id,
                canonical_target=canonical,
                prompts=prompts,
                ambiguous_prompt=ambiguous,
                declared_markers=tuple(sorted(extract_placeholders(canonical))),
            )
        )
    return tuple(cases)


def build_train_cases() -> tuple[TrainCaseV1, ...]:
    """Deterministic two-pack train roots (rebuilt identically every call)."""
    return _openui_train_cases() + _mini_train_cases()


def _schema_texts(cases: Sequence[TrainCaseV1]) -> dict[str, str]:
    """Per-pack schema contract projection over the train frames."""
    from slm_training.dsl.mini_pack import build_mini_pack
    from slm_training.dsl.semantic_frame import CAP1SchemaV1, derive_semantic_frame

    bundle = build_mini_pack()
    frames: dict[str, list[Any]] = {}
    schemas: dict[str, Any] = {
        "openui": CAP1SchemaV1.from_pack("openui"),
        bundle.schema.dsl: bundle.schema,
    }
    for case in cases:
        if case.pack_id == "openui":
            frame = derive_semantic_frame(case.canonical_target, schemas["openui"])
        else:
            frame = bundle.derive_frame(case.canonical_target)
        frames.setdefault(case.pack_id, []).append(frame)
    texts: dict[str, str] = {}
    for pack_id, schema in schemas.items():
        pack_frames = frames.get(pack_id, [])
        domains: dict[str, set[Any]] = {}
        for frame in pack_frames:
            for field, values in closed_value_domains_from_frame(frame).items():
                domains.setdefault(field, set()).update(values)
        projection = {
            "dsl": schema.dsl,
            "component_names": sorted(schema.component_names),
            "content_props": sorted(schema.content_props),
            "closed_value_domains": {
                name: sorted(values, key=repr) for name, values in sorted(domains.items())
            },
        }
        texts[pack_id] = json.dumps(projection, sort_keys=True, default=str)
    return texts


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Preregistration (emitted FIRST, before any results)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PreregistrationV1:
    """Locked experiment block: arms, seeds, exposure, min effect, stop rules."""

    arms: tuple[str, ...]
    arm_levers: dict[str, dict[str, bool]]
    seeds: tuple[int, ...]
    records_per_arm: int
    steps: int
    batch_size: int
    lr: float
    model_config: dict[str, Any]
    suite_seed: int
    heldout_strata: tuple[str, ...]
    min_effect: float
    max_p_value: float
    min_paired_n: int
    invariant_tolerance: float
    stop_rules: tuple[str, ...]
    suite_schema_version: str
    exposure_policy: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": EXPERIMENT_ID,
            "arms": list(self.arms),
            "arm_levers": {k: dict(v) for k, v in self.arm_levers.items()},
            "seeds": list(self.seeds),
            "records_per_arm": self.records_per_arm,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "model_config": dict(self.model_config),
            "suite_seed": self.suite_seed,
            "heldout_strata": list(self.heldout_strata),
            "min_effect": self.min_effect,
            "max_p_value": self.max_p_value,
            "min_paired_n": self.min_paired_n,
            "invariant_tolerance": self.invariant_tolerance,
            "stop_rules": list(self.stop_rules),
            "suite_schema_version": self.suite_schema_version,
            "exposure_policy": self.exposure_policy,
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict())


def build_preregistration(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    records_per_arm: int = DEFAULT_RECORDS_PER_ARM,
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    suite_seed: int = SUITE_SEED,
) -> PreregistrationV1:
    model_config = {
        "architecture": "tree_edit_diffusion",
        "context_backend": "scratch",
        # On-branch value labels are already mutation-count style
        # (1 - applied / (max_chain + 1)); no bounded-distance oracle is in
        # the timing path on this lineage.
        "max_chain": 3,
        "d_model": 64,
        "denoiser_layers": 2,
        "context_layers": 1,
        "beam_width": 2,
        "expand_per_state": 2,
        "max_search_steps": 2,
        "optimizer": "adam",
        "torch_num_threads": 1,
    }
    return PreregistrationV1(
        arms=ARMS,
        arm_levers={arm: dict(ARM_LEVERS[arm]) for arm in ARMS},
        seeds=tuple(int(seed) for seed in seeds),
        records_per_arm=records_per_arm,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        model_config=model_config,
        suite_seed=suite_seed,
        heldout_strata=HELDOUT_STRATA,
        min_effect=PREREGISTERED_MIN_EFFECT,
        max_p_value=PREREGISTERED_MAX_P_VALUE,
        min_paired_n=PREREGISTERED_MIN_PAIRED_N,
        invariant_tolerance=INVARIANT_TOLERANCE,
        stop_rules=tuple(rule.value for rule in StopRule),
        suite_schema_version="cap1_two_pack_freeze/v1",
        exposure_policy=(
            "equal_source_family_exposure: every arm derives from the same "
            "two-pack train roots; equal_target_compute_exposure: every arm "
            "is deterministically sized to records_per_arm and trained with "
            "identical steps/batch order/optimizer per seed"
        ),
    )


# --------------------------------------------------------------------------- #
# Arm corpora
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArmCorpusV1:
    arm_id: str
    records: tuple[ExampleRecord, ...]
    accounting: dict[str, Any]

    @property
    def sha256(self) -> str:
        return _sha256_json(
            [
                {
                    "id": record.id,
                    "prompt": record.prompt,
                    "openui": record.openui,
                    "meta": record.meta,
                }
                for record in self.records
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "sha256": self.sha256,
            "accounting": dict(self.accounting),
        }


def _record(record_id: str, prompt: str, openui: str, **meta: Any) -> ExampleRecord:
    return ExampleRecord(
        id=record_id,
        prompt=prompt,
        openui=openui,
        split="train",
        source="slm368_cap1",
        meta=meta,
    )


def wrap_prompt(arm: str, schema_text: str, prompt: str) -> str:
    """Arm-conditioned model input; the ONLY arm-dependent formatting."""
    if arm in SCHEMA_ARMS:
        return f"schema:\n{schema_text}\n\nrequest:\n{prompt}"
    return prompt


def _case_prompts(arm: ArmId, case: TrainCaseV1) -> tuple[tuple[str, str], ...]:
    """Accepted (style, prompt) pairs for one case under the arm's levers."""
    levers = ARM_LEVERS[arm]
    prompts = list(case.prompts)
    if levers["ambiguity_filter"]:
        prompts = [pair for pair in prompts if pair[0] != "ambiguous"]
        if not levers["multi_style_paraphrases"]:
            # Single accepted prompt: the imperative (determinate) style when
            # available, else the first accepted row — same choice every arm.
            imperative = [pair for pair in prompts if pair[0] == "imperative"]
            prompts = imperative[:1] or prompts[:1]
    else:
        prompts = [*prompts, ("ambiguous", case.ambiguous_prompt)]
    return tuple(prompts)


def build_arm_corpus(
    arm: ArmId,
    cases: Sequence[TrainCaseV1] | None = None,
    *,
    records_per_arm: int = DEFAULT_RECORDS_PER_ARM,
) -> ArmCorpusV1:
    """Build one arm's corpus; arms differ ONLY in the declared data levers.

    Every arm is deterministically sized to exactly ``records_per_arm``
    (family-stratified round-robin then replicate padding with a fixed seed
    shared across arms), so target-position and compute exposure are matched
    by construction.
    """
    cases = tuple(cases) if cases is not None else build_train_cases()
    schema_texts = _schema_texts(cases)
    levers = ARM_LEVERS[arm]
    pool: list[ExampleRecord] = []
    for case in cases:
        family = root_family_for(case.canonical_target, case.pack_id)
        for style, prompt in _case_prompts(arm, case):
            pool.append(
                _record(
                    f"{case.case_id}:{style}",
                    wrap_prompt(arm, schema_texts[case.pack_id], prompt),
                    case.canonical_target,
                    lever="schema_side_channel" if levers["schema_side_channel"] else "nl_only",
                    root_family_id=family,
                    pack_id=case.pack_id,
                    case_id=case.case_id,
                    style=style,
                    ambiguity_filtered=levers["ambiguity_filter"],
                )
            )
    if not pool:
        raise ValueError(f"arm {arm} produced no records")

    # Deterministic matched sizing with family-stratified round-robin, so
    # every arm covers every source family at exactly records_per_arm records.
    rng = random.Random(0)  # identical selection process across arms
    pool = sorted(pool, key=lambda record: record.id)
    by_family: dict[str, list[ExampleRecord]] = {}
    for record in pool:
        by_family.setdefault(str(record.meta.get("root_family_id")), []).append(record)
    for group in by_family.values():
        rng.shuffle(group)
    families = sorted(by_family)
    selected: list[ExampleRecord] = []
    exhausted: set[str] = set()
    while len(selected) < records_per_arm and len(exhausted) < len(families):
        for family in families:
            if len(selected) >= records_per_arm:
                break
            if family in exhausted:
                continue
            group = by_family[family]
            selected.append(group.pop())
            if not group:
                exhausted.add(family)
    selected = sorted(selected, key=lambda record: (record.id, record.openui))
    if len(selected) < records_per_arm:
        replicate = 0
        while len(selected) < records_per_arm:
            source = pool[replicate % len(pool)]
            replicate += 1
            selected.append(
                _record(
                    f"{source.id}#rep{replicate}",
                    source.prompt,
                    source.openui,
                    **{**source.meta, "replicate": replicate},
                )
            )
        selected = sorted(selected, key=lambda record: (record.id, record.openui))

    accounting = {
        "records_total": len(selected),
        "records_by_style": {
            style: sum(1 for record in selected if record.meta.get("style") == style)
            for style in sorted({str(record.meta.get("style")) for record in selected})
        },
        "source_families": sorted(
            {str(record.meta.get("root_family_id")) for record in selected}
        ),
        "target_chars_total": sum(len(record.openui) for record in selected),
        "unique_targets": len({record.openui for record in selected}),
        "levers": dict(levers),
    }
    return ArmCorpusV1(arm_id=arm, records=tuple(selected), accounting=accounting)


def exposure_accounting(corpora: Mapping[str, ArmCorpusV1]) -> dict[str, Any]:
    """Equal-exposure evidence: family coverage, record counts, compute."""
    arms = sorted(corpora)
    family_sets = {arm: set(corpora[arm].accounting["source_families"]) for arm in arms}
    all_families = set().union(*family_sets.values())
    per_arm = {
        arm: {
            "records_total": corpora[arm].accounting["records_total"],
            "target_chars_total": corpora[arm].accounting["target_chars_total"],
            "source_families": sorted(family_sets[arm]),
            "family_coverage": sorted(family_sets[arm] & all_families),
            "families_missing": sorted(all_families - family_sets[arm]),
        }
        for arm in arms
    }
    record_counts = {corpora[arm].accounting["records_total"] for arm in arms}
    return {
        "per_arm": per_arm,
        "equal_record_count": len(record_counts) == 1,
        "equal_source_family_exposure": all(
            not per_arm[arm]["families_missing"] for arm in arms
        ),
        "compute_exposure": "identical steps/batch_size/lr/batch-order per seed",
    }


def symbol_only_report(
    cases: Sequence[TrainCaseV1],
    corpora: Mapping[str, ArmCorpusV1],
    suite: Cap1TwoPackSuiteV1,
) -> dict[str, Any]:
    """Targets must carry only grammar symbols plus declared markers.

    Every train/eval target must canonicalize under its pack grammar, and
    every target placeholder must belong to the case's declared marker table
    (train) or appear in the frozen suite's own prompt/schema surface (eval).
    """
    declared_by_family: dict[str, set[str]] = {}
    for case in cases:
        family = root_family_for(case.canonical_target, case.pack_id)
        declared_by_family[family] = set(case.declared_markers)
    violations: list[str] = []
    checked = 0
    for corpus in corpora.values():
        for record in corpus.records:
            family = str(record.meta.get("root_family_id"))
            pack_id = str(record.meta.get("pack_id"))
            if str(record.meta.get("replicate", "")):
                continue  # replicates re-check the same target
            checked += 1
            try:
                canonical_form(pack_id, record.openui)
            except Exception:  # noqa: BLE001 - non-grammar target
                violations.append(f"{record.id}: target not in {pack_id} grammar")
                continue
            undeclared = sorted(
                set(extract_placeholders(record.openui))
                - declared_by_family.get(family, set())
            )
            if undeclared:
                violations.append(f"{record.id}: undeclared markers {undeclared}")
    eval_checked = 0
    for row in suite.rows:
        for target in row.targets:
            eval_checked += 1
            try:
                canonical_form(row.pack_id, target)
            except Exception:  # noqa: BLE001
                violations.append(f"suite row {row.row_id}: target not in grammar")
    return {
        "passed": not violations,
        "violations": tuple(violations),
        "train_targets_checked": checked,
        "eval_targets_checked": eval_checked,
        "contract": "targets only grammar symbols + declared markers",
    }


# --------------------------------------------------------------------------- #
# Training (identical across arms; only the corpus differs)
# --------------------------------------------------------------------------- #


def train_arm(
    records: Sequence[ExampleRecord],
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    model_config: Mapping[str, Any],
) -> Any:
    """Train one arm; identical init/optimizer/batch order across arms."""
    import torch

    from slm_training.models.tree_edit_diffusion import (
        TreeEditDiffusionConfig,
        TreeEditDiffusionModel,
    )

    torch.set_num_threads(int(model_config.get("torch_num_threads", 1)))
    torch.manual_seed(seed)
    config_kwargs = {
        key: int(model_config[key])
        for key in (
            "max_chain",
            "d_model",
            "denoiser_layers",
            "context_layers",
            "beam_width",
            "expand_per_state",
            "max_search_steps",
        )
        if key in model_config
    }
    config = TreeEditDiffusionConfig(
        seed=seed,
        context_backend=str(model_config.get("context_backend", "scratch")),
        **config_kwargs,
    )
    model = TreeEditDiffusionModel.from_records(list(records), config=config, device="cpu")
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    order = random.Random(0)  # identical batch order across arms
    model.train()
    for _ in range(steps):
        batch = [records[order.randrange(len(records))] for _ in range(batch_size)]
        loss = model.training_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Suite scoring (frozen two-pack adjudication + schema replay probe)
# --------------------------------------------------------------------------- #


def predict_suite(
    model: Any, suite: Cap1TwoPackSuiteV1, *, arm: str
) -> dict[str, str]:
    """Model decision per frozen row id (single batched decode pass)."""
    rows = [row for row in suite.rows if row.ambiguity_disposition != "excluded"]
    requests = [
        GenerationRequest(prompt=wrap_prompt(arm, row.schema_text, row.prompt))
        for row in rows
    ]
    outputs = model.generate_batch_requests(requests)
    return {row.row_id: output for row, output in zip(rows, outputs)}


@dataclass(frozen=True)
class SuiteScoreV1:
    """Arm metrics over the frozen suite plus per-row decision evidence."""

    canonical_accuracy: float
    facts_accuracy: float
    prompt_invariance_rate: float
    relevant_sensitivity: float
    invariant_rate: float
    schema_sensitivity_passed: bool
    determinacy_calibrated: bool
    cap0_retention_accuracy: float
    gate_passed: bool
    gate_failures: tuple[str, ...]
    decisions: Mapping[str, Mapping[str, Any]]  # row_id -> {stratum, canonical_match, decision_sha}

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_accuracy": self.canonical_accuracy,
            "facts_accuracy": self.facts_accuracy,
            "prompt_invariance_rate": self.prompt_invariance_rate,
            "relevant_sensitivity": self.relevant_sensitivity,
            "invariant_rate": self.invariant_rate,
            "schema_sensitivity_passed": self.schema_sensitivity_passed,
            "determinacy_calibrated": self.determinacy_calibrated,
            "cap0_retention_accuracy": self.cap0_retention_accuracy,
            "gate_passed": self.gate_passed,
            "gate_failures": list(self.gate_failures),
            "decisions": {k: dict(v) for k, v in self.decisions.items()},
        }


def score_predictions(
    suite: Cap1TwoPackSuiteV1,
    predictions: Mapping[str, str],
    *,
    tolerance: float = INVARIANT_TOLERANCE,
) -> tuple[SuiteScoreV1, GateReportV1]:
    """Adjudicate one arm's predictions over the frozen two-pack suite."""

    def decide_fn(row: SuiteRowV1) -> str:
        decision = str(predictions.get(row.row_id, "")).strip()
        return decision if decision else ABSTAIN

    adjudication = adjudicate_suite(suite, decide_fn, tolerance=tolerance)
    gate = evaluate_gates(suite, adjudication)
    sensitivity = adjudication.schema_sensitivity
    decisions = {
        decision.row_id: {
            "stratum": decision.stratum,
            "canonical_match": decision.canonical_match,
            "facts_satisfied": decision.facts_satisfied,
            "decision_sha": decision.decision_sha,
        }
        for decision in adjudication.decisions
    }
    score = SuiteScoreV1(
        canonical_accuracy=adjudication.canonical_accuracy,
        facts_accuracy=adjudication.facts_accuracy,
        prompt_invariance_rate=adjudication.prompt_invariance_rate,
        relevant_sensitivity=sensitivity.relevant_sensitivity,
        invariant_rate=sensitivity.invariant_rate,
        schema_sensitivity_passed=sensitivity.passed,
        determinacy_calibrated=bool(adjudication.determinacy["calibrated"]),
        cap0_retention_accuracy=float(adjudication.cap0_retention["accuracy"]),
        gate_passed=gate.passed,
        gate_failures=gate.failures,
        decisions=decisions,
    )
    return score, gate


def heldout_success_vector(
    suite: Cap1TwoPackSuiteV1,
    decisions: Mapping[str, Mapping[str, Any]],
) -> list[int]:
    """Per-row canonical-match vector on held-out AST/equivalence strata."""
    vector: list[int] = []
    for row in suite.rows:
        if row.stratum not in HELDOUT_STRATA:
            continue
        if row.ambiguity_disposition != "determinate":
            continue
        decision = decisions.get(row.row_id)
        vector.append(int(bool(decision and decision.get("canonical_match"))))
    return vector


def paired_evidence(
    control: Sequence[int], candidate: Sequence[int], *, label: str
) -> dict[str, Any]:
    """Wilson bounds per arm + exact paired McNemar (control vs candidate)."""
    if not control or len(control) != len(candidate):
        return {"label": label, "n": 0, "error": "no paired rows"}
    mcnemar = exact_paired_binary_test(control, candidate)
    return {
        "label": label,
        "n": len(control),
        "control_rate": sum(control) / len(control),
        "candidate_rate": sum(candidate) / len(candidate),
        "control_wilson": wilson_interval(sum(control), len(control)),
        "candidate_wilson": wilson_interval(sum(candidate), len(candidate)),
        "mcnemar": mcnemar,
        "meets_preregistered_effect": bool(
            mcnemar["effect"] > PREREGISTERED_MIN_EFFECT
            and mcnemar["candidate_only"] > mcnemar["control_only"]
            and mcnemar["p_value"] <= PREREGISTERED_MAX_P_VALUE
        ),
    }


# --------------------------------------------------------------------------- #
# Certificate decision
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CertificateDecisionV1:
    status: str  # "issued" | "rejected"
    certificate: str | None
    rejection_codes: tuple[str, ...]
    artifact: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "certificate": self.certificate,
            "rejection_codes": list(self.rejection_codes),
            "artifact": dict(self.artifact),
        }


def decide_certificate(
    *,
    filtered_scores: Mapping[str, SuiteScoreV1],
    control_predictions: Mapping[str, Mapping[str, str]],
    filtered_predictions: Mapping[str, Mapping[str, str]],
    paired: Sequence[Mapping[str, Any]],
    symbol_only: Mapping[str, Any],
    contamination_findings: Sequence[str],
    binding: Mapping[str, str],
    plan_id: str = EXPERIMENT_ID,
    plan_sha256: str,
    min_paired_n: int = PREREGISTERED_MIN_PAIRED_N,
    invariant_tolerance: float = INVARIANT_TOLERANCE,
) -> CertificateDecisionV1:
    """Pure CERT_CAP1 decision; every stop rule rejects with its reason.

    ``binding`` binds checkpoint/corpus/suite/gate/config/code/evidence
    digests into the certificate artifact (issued or rejected).
    """
    codes: list[StopRule] = []

    # Contamination: any suite/leakage/symbol finding invalidates everything.
    findings = list(contamination_findings)
    if not symbol_only.get("passed", False):
        findings.extend(str(v) for v in symbol_only.get("violations", ()))
    if findings:
        codes.append(StopRule.CONTAMINATED)

    # Underpowered: the preregistered paired n is unmet (fixture n is an
    # honest outcome, mirroring SLM-362).
    paired_n = max((int(row.get("n", 0)) for row in paired), default=0)
    if paired_n < min_paired_n:
        codes.append(StopRule.UNDERPOWERED)

    # Prediction-identical: a filtered arm emits a control's outputs verbatim.
    for filtered_id, filtered in sorted(filtered_predictions.items()):
        for control_id, control in sorted(control_predictions.items()):
            if control and all(
                filtered.get(key, "") == value for key, value in control.items()
            ):
                codes.append(StopRule.PREDICTION_IDENTICAL)

    # Causal consumption: relevant schema perturbations must change the
    # decision for EVERY filtered arm; anything less means the schema is
    # ignored (empty/contradictory replays do not move the model).
    for arm in FILTERED_ARMS:
        score = filtered_scores.get(arm)
        if score is None:
            continue
        if score.relevant_sensitivity < 1.0:
            codes.append(StopRule.IGNORES_SCHEMA)
        if score.invariant_rate < 1.0 - invariant_tolerance:
            codes.append(StopRule.FAILS_HARD_CONTRASTS)

    # Hard contrasts: at least one filtered arm must beat BOTH controls with
    # preregistered paired evidence on held-out AST/equivalence rows.
    beat_counts = {arm: 0 for arm in FILTERED_ARMS}
    for row in paired:
        if row.get("meets_preregistered_effect"):
            label = str(row.get("label", ""))
            arm = label.split(":")[0]
            if arm in beat_counts:
                beat_counts[arm] += 1
    if not any(count >= len(CONTROL_ARMS) for count in beat_counts.values()):
        codes.append(StopRule.FAILS_HARD_CONTRASTS)

    # CAP0 retention is a mandatory gate for every filtered arm.
    for arm in FILTERED_ARMS:
        score = filtered_scores.get(arm)
        if score is None:
            continue
        if score.cap0_retention_accuracy < 1.0 or "cap0_retention_perfect" in score.gate_failures:
            codes.append(StopRule.CAP0_REGRESSION)

    # Deduplicate, stable order.
    ordered = tuple(dict.fromkeys(codes))
    passed = not ordered

    process = ProcessProvenanceV1(
        process_id=plan_id,
        process_version=str(binding.get("suite_schema_version", "cap1_two_pack_freeze/v1")),
        config_sha256=_sha256_text(str(binding.get("config_sha256", plan_sha256))),
        code_sha256=_sha256_text(str(binding.get("code_commit", "UNKNOWN"))),
    )
    evidence_ids = tuple(
        _sha256_text(str(binding[key]))
        for key in ("suite_sha256", "evidence_json_uri")
        if binding.get(key)
    ) or (plan_sha256,)
    artifact = CapabilityCertificateV1(
        capability=Capability.CAP1_SEMANTICS,
        plan_id=plan_id,
        plan_sha256=plan_sha256,
        qa_pair_ids=(_sha256_text(str(binding.get("corpus_sha256", plan_id))),),
        validation_report_ids=evidence_ids,
        gate_process=process,
        passed=passed,
        rejection_codes=tuple(code.value for code in ordered),
    )
    artifact_dict = artifact.to_dict()
    artifact_dict["binding"] = dict(binding)
    artifact_dict["paired_n"] = paired_n
    artifact_dict["symbol_only"] = dict(symbol_only)
    artifact_dict["cap2_natural_language"] = "open" if passed else "CLOSED (stop rule)"
    return CertificateDecisionV1(
        status="issued" if passed else "rejected",
        certificate=CERT_CAP1 if passed else None,
        rejection_codes=tuple(code.value for code in ordered),
        artifact=artifact_dict,
    )


# --------------------------------------------------------------------------- #
# End-to-end run
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExperimentContextV1:
    """Shared, arm-independent experiment state (prereg, suite, corpora)."""

    prereg: PreregistrationV1
    suite: Cap1TwoPackSuiteV1
    suite_integrity_ok: bool
    contamination: tuple[str, ...]
    cases: tuple[TrainCaseV1, ...]
    corpora: dict[str, ArmCorpusV1]
    exposure: dict[str, Any]
    symbol_only: dict[str, Any]
    corpus_sha256: str


def prepare_context(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    records_per_arm: int = DEFAULT_RECORDS_PER_ARM,
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    suite_seed: int = SUITE_SEED,
) -> ExperimentContextV1:
    """Build the preregistration, frozen suite, and matched arm corpora."""
    prereg = build_preregistration(
        seeds=seeds,
        records_per_arm=records_per_arm,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        suite_seed=suite_seed,
    )

    # Frozen SLM-367 two-pack suite; fail-closed integrity.
    suite = build_cap1_two_pack_suite(seed=suite_seed)
    integrity = verify_suite_integrity(suite)

    # Contamination: train/eval families disjoint AND no shared targets.
    cases = build_train_cases()
    train_families = {
        root_family_for(case.canonical_target, case.pack_id) for case in cases
    }
    eval_families = {row.root_family_id for row in suite.rows}
    train_targets = {case.canonical_target.strip() for case in cases}
    eval_targets = {target.strip() for row in suite.rows for target in row.targets}
    contamination: list[str] = []
    if train_families & eval_families:
        contamination.append(
            f"train/eval family overlap: {sorted(train_families & eval_families)}"
        )
    if train_targets & eval_targets:
        contamination.append("train targets appear in the eval suite")
    if not integrity.ok:
        contamination.extend(f"suite_integrity:{v}" for v in integrity.violations)

    # Arm corpora with matched exposure + symbol-only target audit.
    corpora = {
        arm: build_arm_corpus(arm, cases, records_per_arm=records_per_arm)
        for arm in ARMS
    }
    exposure = exposure_accounting(corpora)
    symbol_only = symbol_only_report(cases, corpora, suite)
    corpus_sha = _sha256_json({arm: corpora[arm].sha256 for arm in ARMS})
    return ExperimentContextV1(
        prereg=prereg,
        suite=suite,
        suite_integrity_ok=integrity.ok,
        contamination=tuple(contamination),
        cases=cases,
        corpora=corpora,
        exposure=exposure,
        symbol_only=symbol_only,
        corpus_sha256=corpus_sha,
    )


def shard_path(output_dir: Path, arm: str) -> Path:
    return Path(output_dir) / f"shard_{arm}.json"


def run_arm_shard(
    context: ExperimentContextV1,
    arm: ArmId,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Train + evaluate one arm across all preregistered seeds; write a shard.

    Sharding keeps each CLI command inside the hard run cap: arms are
    computed independently and ``finalize_report`` aggregates the shards.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prereg = context.prereg
    seed_rows: list[dict[str, Any]] = []
    for seed in prereg.seeds:
        model = train_arm(
            context.corpora[arm].records,
            steps=prereg.steps,
            batch_size=prereg.batch_size,
            lr=prereg.lr,
            seed=seed,
            model_config=prereg.model_config,
        )
        checkpoint_path = output_dir / f"checkpoint_{arm}_seed{seed}.pt"
        model.save(checkpoint_path)
        predictions = predict_suite(model, context.suite, arm=arm)
        score, gate = score_predictions(context.suite, predictions)
        seed_rows.append(
            {
                "seed": seed,
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "score": score.to_dict(),
                "gate_passed": gate.passed,
                "gate_failures": list(gate.failures),
                "heldout_success": heldout_success_vector(
                    context.suite, score.decisions
                ),
                "predictions": dict(predictions),
            }
        )
    shard = {
        "arm_id": arm,
        "corpus": context.corpora[arm].to_dict(),
        "seeds": seed_rows,
        "preregistration_sha256": prereg.sha256,
        "suite_sha256": context.suite.suite_sha256,
    }
    shard_path(output_dir, arm).write_text(
        json.dumps(shard, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return shard


def load_shards(output_dir: Path, *, prereg_sha256: str) -> dict[str, Any]:
    """Load every arm shard; fail closed on missing or mismatched shards."""
    arm_results: dict[str, Any] = {}
    for arm in ARMS:
        path = shard_path(output_dir, arm)
        if not path.exists():
            raise FileNotFoundError(
                f"missing arm shard {path}; run --mode fixture --arms {arm} first"
            )
        shard = json.loads(path.read_text(encoding="utf-8"))
        if shard.get("preregistration_sha256") != prereg_sha256:
            raise ValueError(
                f"shard {arm} was produced under a different preregistration"
            )
        arm_results[arm] = {"corpus": shard["corpus"], "seeds": shard["seeds"]}
    return arm_results


def _score_from_dict(data: Mapping[str, Any]) -> SuiteScoreV1:
    return SuiteScoreV1(
        canonical_accuracy=data["canonical_accuracy"],
        facts_accuracy=data["facts_accuracy"],
        prompt_invariance_rate=data["prompt_invariance_rate"],
        relevant_sensitivity=data["relevant_sensitivity"],
        invariant_rate=data["invariant_rate"],
        schema_sensitivity_passed=data["schema_sensitivity_passed"],
        determinacy_calibrated=data["determinacy_calibrated"],
        cap0_retention_accuracy=data["cap0_retention_accuracy"],
        gate_passed=data["gate_passed"],
        gate_failures=tuple(data["gate_failures"]),
        decisions=data["decisions"],
    )


def finalize_report(
    context: ExperimentContextV1,
    arm_results: Mapping[str, Any],
    *,
    started: float | None = None,
) -> dict[str, Any]:
    """Aggregate arm shards into the full report + CERT_CAP1 decision."""
    suite = context.suite
    prereg = context.prereg

    # Paired evidence: each filtered arm vs each control, pooled across
    # preregistered seeds, on held-out AST/equivalence rows.
    def _pooled(arm: str) -> list[int]:
        values: list[int] = []
        for row in arm_results[arm]["seeds"]:
            values.extend(row["heldout_success"])
        return values

    paired: list[dict[str, Any]] = []
    for candidate in FILTERED_ARMS:
        for control in CONTROL_ARMS:
            paired.append(
                paired_evidence(
                    _pooled(control),
                    _pooled(candidate),
                    label=f"{candidate}:vs_{control}",
                )
            )

    # Certificate decision on the primary (first) preregistered seed.
    primary = 0
    filtered_scores = {
        arm: _score_from_dict(arm_results[arm]["seeds"][primary]["score"])
        for arm in FILTERED_ARMS
    }
    control_predictions = {
        arm: arm_results[arm]["seeds"][primary]["predictions"] for arm in CONTROL_ARMS
    }
    filtered_predictions = {
        arm: arm_results[arm]["seeds"][primary]["predictions"] for arm in FILTERED_ARMS
    }

    binding = {
        "checkpoint_sha256": arm_results["SCHEMA_FILTERED_MULTI"]["seeds"][primary][
            "checkpoint_sha256"
        ],
        "corpus_sha256": context.corpus_sha256,
        "suite_sha256": str(suite.suite_sha256),
        "suite_schema_version": str(suite.schema_version),
        "config_sha256": prereg.sha256,
        "code_commit": "UNKNOWN",  # replaced by the CLI with the stamp commit
        "evidence_json_uri": "docs/design/iter-slm368-cap1-experiment-20260725.json",
        "evidence_md_uri": "docs/design/iter-slm368-cap1-experiment-20260725.md",
    }
    decision = decide_certificate(
        filtered_scores=filtered_scores,
        control_predictions=control_predictions,
        filtered_predictions=filtered_predictions,
        paired=paired,
        symbol_only=context.symbol_only,
        contamination_findings=list(context.contamination),
        binding=binding,
        plan_sha256=prereg.sha256,
        min_paired_n=prereg.min_paired_n,
        invariant_tolerance=prereg.invariant_tolerance,
    )

    # Replay probe summary (causal consumption evidence, primary seed).
    replay_probe = {
        arm: {
            "relevant_sensitivity": filtered_scores[arm].relevant_sensitivity,
            "invariant_rate": filtered_scores[arm].invariant_rate,
            "cap0_retention_accuracy": filtered_scores[arm].cap0_retention_accuracy,
            "prompt_invariance_rate": filtered_scores[arm].prompt_invariance_rate,
        }
        for arm in FILTERED_ARMS
    }

    return {
        "schema": "Slm368Cap1ExperimentReportV1",
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "claim_class": "fixture",
        # Preregistered block FIRST — locked before any result below.
        "preregistration": prereg.to_dict(),
        "preregistration_sha256": prereg.sha256,
        "suite": {
            "seed": prereg.suite_seed,
            "suite_sha256": suite.suite_sha256,
            "rows_total": len(suite.rows),
            "integrity_ok": context.suite_integrity_ok,
            "contamination_findings": list(context.contamination),
        },
        "exposure": context.exposure,
        "symbol_only": context.symbol_only,
        "corpus_sha256": context.corpus_sha256,
        "arms": dict(arm_results),
        "paired_evidence": paired,
        "replay_probe": replay_probe,
        "certificate": decision.to_dict(),
        "elapsed_seconds": (
            round(time.perf_counter() - started, 3) if started is not None else None
        ),
    }


def run_experiment(
    *,
    output_dir: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    records_per_arm: int = DEFAULT_RECORDS_PER_ARM,
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    suite_seed: int = SUITE_SEED,
) -> dict[str, Any]:
    """Run the matched CAP1 grounding experiment at fixture scale."""
    started = time.perf_counter()
    context = prepare_context(
        seeds=seeds,
        records_per_arm=records_per_arm,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        suite_seed=suite_seed,
    )
    arm_results = {
        arm: {
            "corpus": shard["corpus"],
            "seeds": shard["seeds"],
        }
        for arm in ARMS
        for shard in [run_arm_shard(context, arm, output_dir=output_dir)]
    }
    return finalize_report(context, arm_results, started=started)


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def render_markdown(report: Mapping[str, Any], *, command: str) -> str:
    prereg = report["preregistration"]
    cert = report["certificate"]
    lines = [
        "# SLM-368 (DSH2-08): matched CAP1 grounding experiment",
        "",
        "**Claim class:** fixture (fixture n cannot issue CERT_CAP1)",
        "",
        f"**Certificate:** {cert['status']} "
        + (f"({', '.join(cert['rejection_codes'])})" if cert["rejection_codes"] else ""),
        "",
        "## Preregistration (locked before results)",
        "",
        f"- Arms: {', '.join(prereg['arms'])}",
        f"- Seeds: {prereg['seeds']}",
        f"- Records per arm (matched target/compute exposure): "
        f"{prereg['records_per_arm']}",
        f"- Steps x batch x lr: {prereg['steps']} x {prereg['batch_size']} x "
        f"{prereg['lr']}",
        f"- Held-out strata: {', '.join(prereg['heldout_strata'])}",
        f"- Preregistered minimum effect: paired effect > {prereg['min_effect']}, "
        f"McNemar p <= {prereg['max_p_value']}, paired n >= {prereg['min_paired_n']}",
        f"- Stop rules: {', '.join(prereg['stop_rules'])}",
        f"- Preregistration sha256: `{report['preregistration_sha256']}`",
        "",
        "## Exposure accounting",
        "",
        "| Arm | Records | Target chars | Families missing |",
        "| --- | --- | --- | --- |",
    ]
    for arm, row in sorted(report["exposure"]["per_arm"].items()):
        lines.append(
            f"| {arm} | {row['records_total']} | {row['target_chars_total']} "
            f"| {row['families_missing']} |"
        )
    lines += [
        "",
        f"- Equal record count: {report['exposure']['equal_record_count']}",
        f"- Equal source-family exposure: "
        f"{report['exposure']['equal_source_family_exposure']}",
        f"- Symbol-only targets: {report['symbol_only']['passed']} "
        f"(violations: {list(report['symbol_only']['violations'])})",
        "",
        "## Arm metrics (primary seed)",
        "",
        "| Arm | canon-acc | facts-acc | prompt-inv | rel-sens | inv-rate | "
        "cap0-ret | gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in prereg["arms"]:
        primary = report["arms"][arm]["seeds"][0]
        score = primary["score"]
        lines.append(
            f"| {arm} | {score['canonical_accuracy']:.3f} | "
            f"{score['facts_accuracy']:.3f} | {score['prompt_invariance_rate']:.3f} | "
            f"{score['relevant_sensitivity']:.3f} | {score['invariant_rate']:.3f} | "
            f"{score['cap0_retention_accuracy']:.3f} | "
            f"{'pass' if score['gate_passed'] else 'fail'} |"
        )
    lines += [
        "",
        "## Paired evidence (filtered arms vs controls, pooled seeds)",
        "",
        "| Comparison | n | control | candidate | effect | p | meets effect |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["paired_evidence"]:
        if row.get("error"):
            lines.append(f"| {row['label']} | 0 | - | - | - | - | {row['error']} |")
            continue
        lines.append(
            f"| {row['label']} | {row['n']} | {row['control_rate']:.3f} | "
            f"{row['candidate_rate']:.3f} | {row['mcnemar']['effect']:.3f} | "
            f"{row['mcnemar']['p_value']:.3f} | "
            f"{row['meets_preregistered_effect']} |"
        )
    lines += [
        "",
        "## Schema replay probe (causal consumption, primary seed)",
        "",
        "| Arm | relevant sensitivity | invariant rate | cap0 retention |",
        "| --- | --- | --- | --- |",
    ]
    for arm, row in sorted(report["replay_probe"].items()):
        lines.append(
            f"| {arm} | {row['relevant_sensitivity']:.3f} | "
            f"{row['invariant_rate']:.3f} | {row['cap0_retention_accuracy']:.3f} |"
        )
    lines += [
        "",
        "## Certificate decision",
        "",
        f"- Status: **{cert['status']}**; certificate: {cert['certificate']}",
        f"- Rejection codes: {cert['rejection_codes']}",
        f"- Contamination findings: {report['suite']['contamination_findings']}",
        "- Natural-language CAP2: "
        + ("open" if cert["status"] == "issued" else "CLOSED (stop rule)"),
        "",
        "## Fixture limitations (honest scope)",
        "",
        "- The on-branch tree-edit trainer fragment-validates targets through the "
        "OpenUI grammar, so mini-flow records contribute zero gradient; mini-pack "
        "suite rows measure an OpenUI-only-trained fixture model. This is recorded, "
        "not hidden, and the certificate decision above already rejects at fixture n.",
        "- Target-char totals differ across arms because the filtered arms carry "
        "fewer unique prompt styles before matched-count replicate padding; record "
        "count and source-family coverage are exactly matched.",
        "",
        "## Exact command",
        "",
        f"```bash\n{command}\n```",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "ARMS",
    "ARM_LEVERS",
    "CERT_CAP1",
    "CONTROL_ARMS",
    "DEFAULT_SEEDS",
    "EXPERIMENT_ID",
    "FILTERED_ARMS",
    "HELDOUT_STRATA",
    "INVARIANT_TOLERANCE",
    "PREREGISTERED_MAX_P_VALUE",
    "PREREGISTERED_MIN_EFFECT",
    "PREREGISTERED_MIN_PAIRED_N",
    "SCHEMA_ARMS",
    "SCHEMA_VERSION",
    "ArmCorpusV1",
    "CertificateDecisionV1",
    "ExperimentContextV1",
    "PreregistrationV1",
    "StopRule",
    "SuiteScoreV1",
    "TrainCaseV1",
    "build_arm_corpus",
    "build_preregistration",
    "build_train_cases",
    "decide_certificate",
    "exposure_accounting",
    "finalize_report",
    "heldout_success_vector",
    "load_shards",
    "paired_evidence",
    "predict_suite",
    "prepare_context",
    "render_markdown",
    "run_arm_shard",
    "run_experiment",
    "score_predictions",
    "shard_path",
    "symbol_only_report",
    "train_arm",
    "wrap_prompt",
]
