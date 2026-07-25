"""SLM-362 (DSH1-10): matched CAP0 curriculum experiment + CERT_CAP0 decision.

Tests whether staged grammar-complete data teaches symbolic structural
competence beyond current, atomic-only, and copy-heavy controls. Four matched
arms are built from the same DSH1 artifact roots with equal source-family
exposure and equal target/compute exposure; model architecture, decoder,
compiler, optimizer, and evaluation are held fixed — only the declared data
capability lever changes:

* ``BASE`` — current/canonical corpus records (whole canonical answers, no
  staging levers);
* ``ATOMIC_ONLY`` — atomic witnesses only (single verified statements; no
  composition, prefix completion, or equivalence preferences);
* ``STAGED_NO_PREFERENCE`` — atomic + composition + prefix completion;
* ``STAGED_FULL`` — all levers (adds DSH1-05 equivalence preferences).

Evaluation runs the DSH1-09 frozen suite
(:mod:`~slm_training.harnesses.train_data.cap0_eval_freeze`): all six
subsuites, the versioned :class:`Cap0GateV1`, and the constructed collapse
baselines, plus paired per-record confidence evidence (Wilson bounds + exact
paired McNemar) of ``STAGED_FULL`` against both controls on held-out
composition and accepted-set completion. AgentV is published as a diagnostic
(non-fatal on failure).

``CERT_CAP0`` is issued only when ``STAGED_FULL`` beats the copy/atomic
controls with paired confidence evidence, no strict metric regresses (parse/
static validity, marker fidelity, binding-aware, empty/trivial rate), and the
gate's preregistered ``n`` is met. Fixture ``n`` honestly reports
``insufficient_n`` and rejects. Stop rules — prediction-identical,
copy-dominated, structurally regressive, contaminated, underpowered — reject
the certificate and keep CAP1/CAP2 training closed.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from slm_training.dsl.canonicalize import canonical_fingerprint
from slm_training.dsl.grammar_capabilities import (
    GrammarCapabilityAdapterV1,
    production_id,
)
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
from slm_training.harnesses.train_data.answer_composition import (
    admit_answer,
    apply_combinator,
)
from slm_training.harnesses.train_data.cap0_eval_freeze import (
    CERT_CAP0,
    GATE_VERSION,
    Cap0EvalRowV1,
    Cap0EvalSuiteV1,
    Cap0GateReportV1,
    Cap0GateV1,
    Cap0MetricsV1,
    build_cap0_eval_suite,
    empty_trivial_baseline_metrics,
    evaluate_gate,
    identity_baseline_metrics,
    stop_rule_violations,
    suite_valid,
    unchanged_input_baseline_metrics,
    verify_suite_integrity,
)
from slm_training.harnesses.train_data.cap0_qa import generate_cap0_qa
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

SCHEMA_VERSION = "slm362_cap0_experiment/v1"
EXPERIMENT_ID = "slm362-cap0-experiment"
DSL = "openui"

ArmId = Literal["BASE", "ATOMIC_ONLY", "STAGED_NO_PREFERENCE", "STAGED_FULL"]
ARMS: tuple[ArmId, ...] = ("BASE", "ATOMIC_ONLY", "STAGED_NO_PREFERENCE", "STAGED_FULL")

#: Declared data capability levers per arm; the ONLY thing that varies.
ARM_LEVERS: dict[str, dict[str, bool]] = {
    "BASE": {
        "canonical_answers": True,
        "atomic_witnesses": False,
        "composition": False,
        "prefix_completion": False,
        "equivalence_preferences": False,
    },
    "ATOMIC_ONLY": {
        "canonical_answers": False,
        "atomic_witnesses": True,
        "composition": False,
        "prefix_completion": False,
        "equivalence_preferences": False,
    },
    "STAGED_NO_PREFERENCE": {
        "canonical_answers": False,
        "atomic_witnesses": True,
        "composition": True,
        "prefix_completion": True,
        "equivalence_preferences": False,
    },
    "STAGED_FULL": {
        "canonical_answers": False,
        "atomic_witnesses": True,
        "composition": True,
        "prefix_completion": True,
        "equivalence_preferences": True,
    },
}

DEFAULT_SEEDS: tuple[int, ...] = (0, 1)
DEFAULT_STEPS = 8
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 3e-3
DEFAULT_RECORDS_PER_ARM = 16

#: Preregistered minimum effect for STAGED_FULL vs each control on held-out
#: composition or accepted-set completion: strictly positive paired effect,
#: more candidate-only than control-only discordant pairs, and an exact
#: paired McNemar p-value at or below this level.
PREREGISTERED_MIN_EFFECT = 0.0
PREREGISTERED_MAX_P_VALUE = 0.10
#: Tolerance for strict-metric regressions vs BASE (exact-zero: any drop
#: beyond float noise is a regression).
REGRESSION_TOLERANCE = 1e-9


class StopRule(str, Enum):
    PREDICTION_IDENTICAL = "prediction_identical"
    COPY_DOMINATED = "copy_dominated"
    STRUCTURALLY_REGRESSIVE = "structurally_regressive"
    CONTAMINATED = "contaminated"
    UNDERPOWERED = "underpowered"


# --------------------------------------------------------------------------- #
# Fixture roots (deterministic, split-policy assigned)
# --------------------------------------------------------------------------- #

_POLICY = RootFamilySplitPolicyV1()

TRAIN_ANSWERS: tuple[tuple[str, str], ...] = tuple(
    zip(
        (
            'item = TextContent(":t.card.text")\nroot = Stack([item], "column")',
            'title = TextContent(":t.list.title")\n'
            'item = TextContent(":t.list.item")\n'
            'root = Stack([title, item], "column")',
            'cta = Button(":t.cta.label", "primary")\nroot = Stack([cta], "column")',
            'hero = TextContent(":t.hero.title")\n'
            'sub = TextContent(":t.hero.subtitle")\n'
            'cta = Button(":t.hero.cta", "secondary")\n'
            'root = Stack([hero, sub, cta], "column")',
            'nav = TextContent(":t.nav.logo")\nroot = Stack([nav], "row")',
            'avatar = Image(":t.user.avatar")\n'
            'name = TextContent(":t.user.name")\n'
            'root = Stack([avatar, name], "row")',
            'banner = TextContent(":t.banner.text")\n'
            'dismiss = Button(":t.banner.dismiss", "secondary")\n'
            'root = Stack([banner, dismiss], "row")',
            'heading = TextContent(":t.form.heading")\n'
            'input = TextContent(":t.form.input")\n'
            'root = Stack([heading, input], "column")',
        ),
        tuple(
            family
            for family in (f"cap0-train-{index}" for index in range(200))
            if _POLICY.assign(family) == "train"
        )[:8],
    )
)

EVAL_ANSWERS: tuple[tuple[str, str], ...] = tuple(
    zip(
        (
            'item = TextContent(":w.text")\nroot = Stack([item], "column")',
            'item = TextContent(":w.text")\n$s = item\nroot = Stack([item], "column")',
            'hero = TextContent(":smoke.hero.title")\nroot = Stack([hero], "column")',
            'root = Stack([hero, cta], "column")\n'
            'hero = TextContent(":smoke.hero.title")\n'
            'cta = Button(":cta.label", "primary")',
            "root = Separator()",
        ),
        tuple(
            family
            for family in (f"cap0-eval-{index}" for index in range(200))
            if _POLICY.assign(family) == "test"
        )[:5],
    )
)


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
    min_effect: float
    max_p_value: float
    regression_tolerance: float
    stop_rules: tuple[str, ...]
    gate_version: str
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
            "min_effect": self.min_effect,
            "max_p_value": self.max_p_value,
            "regression_tolerance": self.regression_tolerance,
            "stop_rules": list(self.stop_rules),
            "gate_version": self.gate_version,
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
    suite_seed: int = 7,
) -> PreregistrationV1:
    model_config = {
        "architecture": "tree_edit_diffusion",
        "context_backend": "scratch",
        # mutation_count value labels: the bounded-distance oracle (SLM-308)
        # dominates fixture wall-clock and is not the lever under test.
        "value_label_mode": "mutation_count",
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
        min_effect=PREREGISTERED_MIN_EFFECT,
        max_p_value=PREREGISTERED_MAX_P_VALUE,
        regression_tolerance=REGRESSION_TOLERANCE,
        stop_rules=tuple(rule.value for rule in StopRule),
        gate_version=GATE_VERSION,
        exposure_policy=(
            "equal_source_family_exposure: every arm derives from the same "
            "train root families; equal_target_compute_exposure: every arm is "
            "deterministically sized to records_per_arm and trained with "
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
        source="slm362_cap0",
        meta=meta,
    )


def _source_families(records: Iterable[ExampleRecord]) -> list[str]:
    families: set[str] = set()
    for record in records:
        # Composition records carry a composite "left+right" family id.
        families.update(str(record.meta.get("root_family_id")).split("+"))
    return sorted(families)


def _atomic_witness_records(train: tuple[tuple[str, str], ...]) -> list[ExampleRecord]:
    records: list[ExampleRecord] = []
    for source, family in train:
        for line in source.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(
                _record(
                    f"atomic:{family}:{len(records)}",
                    f"produce the atomic OpenUI statement witnessed by family {family}",
                    line,
                    lever="atomic_witnesses",
                    root_family_id=family,
                )
            )
    return records


def _composition_records(
    train: tuple[tuple[str, str], ...], *, dsl: str
) -> list[ExampleRecord]:
    records: list[ExampleRecord] = []
    for index in range(len(train) - 1):
        (left_source, left_family), (right_source, right_family) = (
            train[index],
            train[index + 1],
        )
        left = admit_answer(left_source, split="train", root_family_id=left_family, dsl=dsl)
        right = admit_answer(right_source, split="train", root_family_id=right_family, dsl=dsl)
        result = apply_combinator("COMPOSE_DOCUMENT", (left, right), dsl=dsl)
        if not result.ok or result.canonical_source is None:
            continue
        records.append(
            _record(
                f"compose:{left_family}:{right_family}",
                f"compose the OpenUI documents of families {left_family} and {right_family}",
                result.canonical_source,
                lever="composition",
                root_family_id=f"{left_family}+{right_family}",
                combinator=result.combinator,
                lineage_id=result.lineage_id,
            )
        )
    return records


def _qa_records(
    train: tuple[tuple[str, str], ...],
    *,
    adapter: GrammarCapabilityAdapterV1,
    seed: int,
    dsl: str,
) -> tuple[list[ExampleRecord], list[ExampleRecord]]:
    """Prefix-completion and equivalence-preference records from DSH1-08 QA."""
    dataset = generate_cap0_qa(
        (source for source, _ in train), adapter=adapter, identity_cap=0.1, seed=seed, dsl=dsl
    )
    family_by_source = {source: family for source, family in train}
    prefix_records: list[ExampleRecord] = []
    preference_records: list[ExampleRecord] = []
    for row in dataset.rows:
        answer = str(row.provenance.get("answer", ""))
        family = family_by_source.get(answer, "unknown")
        if row.kind == "complete_suffix":
            target = row.canonical_preference or (
                row.accepted_completions[0].source if row.accepted_completions else None
            )
            if target is None:
                continue
            prefix_records.append(
                _record(
                    f"prefix:{row.row_id}",
                    row.prompt,
                    target,
                    lever="prefix_completion",
                    root_family_id=family,
                    accepted=sorted(item.source for item in row.accepted_completions),
                )
            )
        elif row.kind == "canonicalize":
            assert row.completion is not None
            preference_records.append(
                _record(
                    f"preference:{row.row_id}",
                    row.prompt,
                    row.completion,
                    lever="equivalence_preferences",
                    root_family_id=family,
                    relations=list(row.relations),
                )
            )
    return prefix_records, preference_records


def build_arm_corpus(
    arm: ArmId,
    train: tuple[tuple[str, str], ...] = TRAIN_ANSWERS,
    *,
    adapter: GrammarCapabilityAdapterV1,
    records_per_arm: int = DEFAULT_RECORDS_PER_ARM,
    seed: int = 0,
    dsl: str = DSL,
) -> ArmCorpusV1:
    """Build one arm's corpus; arms differ ONLY in the declared data levers.

    Every arm is deterministically sized to exactly ``records_per_arm``
    (subsample or cycle-upsampling with a fixed seed shared across arms), so
    target-position and compute exposure are matched by construction.
    """
    levers = ARM_LEVERS[arm]
    pool: list[ExampleRecord] = []
    if levers["canonical_answers"]:
        for source, family in train:
            pool.append(
                _record(
                    f"canonical:{family}",
                    f"produce the canonical OpenUI document for family {family}",
                    source,
                    lever="canonical_answers",
                    root_family_id=family,
                )
            )
    if levers["atomic_witnesses"]:
        pool.extend(_atomic_witness_records(train))
    if levers["composition"]:
        pool.extend(_composition_records(train, dsl=dsl))
    if levers["prefix_completion"] or levers["equivalence_preferences"]:
        prefix_records, preference_records = _qa_records(
            train, adapter=adapter, seed=seed, dsl=dsl
        )
        if levers["prefix_completion"]:
            pool.extend(prefix_records)
        if levers["equivalence_preferences"]:
            pool.extend(preference_records)
    if not pool:
        raise ValueError(f"arm {arm} produced no records")

    # Deterministic matched sizing with family-stratified round-robin, so every
    # arm covers every source family at exactly ``records_per_arm`` records.
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
        # Pool smaller than the budget: deterministic replicate padding (the
        # same targets revisited; compute/target exposure stays matched).
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
        "records_by_lever": {
            lever: sum(1 for record in selected if record.meta.get("lever") == lever)
            for lever, enabled in sorted(levers.items())
            if enabled
        },
        "source_families": _source_families(selected),
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
    config = TreeEditDiffusionConfig(
        seed=seed,
        value_label_mode=str(model_config.get("value_label_mode", "bounded_distance")),
        context_backend=str(model_config.get("context_backend", "scratch")),
        max_chain=int(model_config.get("max_chain", 3)),
        d_model=int(model_config.get("d_model", 64)),
        denoiser_layers=int(model_config.get("denoiser_layers", 2)),
        context_layers=int(model_config.get("context_layers", 1)),
        beam_width=int(model_config.get("beam_width", 2)),
        expand_per_state=int(model_config.get("expand_per_state", 2)),
        max_search_steps=int(model_config.get("max_search_steps", 4)),
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
# Suite scoring
# --------------------------------------------------------------------------- #


def _row_prompt(row: Cap0EvalRowV1) -> str:
    """Deterministic eval prompt per frozen row (never the gold target)."""
    payload = row.payload
    for key in ("canonical", "expanded", "prefix", "candidate"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return f"cap0 {row.subsuite} {payload.get('operation', 'produce')}\n{value}"
    if row.subsuite == "compositional_held_out":
        return f"cap0 compose\n{'+'.join(sorted(payload.get('source_answer_ids') or ()))}"
    return f"cap0 {row.subsuite}"


def predict_suite(model: Any, suite: Cap0EvalSuiteV1) -> dict[str, str]:
    """Model prediction per frozen row id (single batched decode pass)."""
    from slm_training.data.contract import GenerationRequest

    rows = list(suite.rows)
    requests = [GenerationRequest(prompt=_row_prompt(row)) for row in rows]
    outputs = model.generate_batch_requests(requests)
    return {row.row_id: output for row, output in zip(rows, outputs)}


def _fingerprint_or_none(text: str, *, dsl: str) -> str | None:
    try:
        return canonical_fingerprint(text, dsl=dsl)
    except Exception:  # noqa: BLE001 - unparsable prediction: no match
        return None


def _production_ids_of(adapter: GrammarCapabilityAdapterV1, source: str) -> set[str]:
    starts = adapter.start_symbols
    primary = str(starts[0]) if starts else "start"
    trace = adapter.authority.production_trace
    if trace is None:
        return set()
    try:
        occurrences = trace(primary, source if source.endswith("\n") else source + "\n")
    except Exception:  # noqa: BLE001
        return set()
    return {production_id(item.production) for item in occurrences}


@dataclass(frozen=True)
class SuiteScoreV1:
    """Arm metrics over the frozen suite plus per-record success vectors."""

    metrics: Cap0MetricsV1
    parse_valid_rate: float
    marker_fidelity: float
    compositional_success: tuple[int, ...]
    prefix_success: tuple[int, ...]
    copy_hits: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "parse_valid_rate": self.parse_valid_rate,
            "marker_fidelity": self.marker_fidelity,
            "compositional_success": list(self.compositional_success),
            "prefix_success": list(self.prefix_success),
            "copy_hits": self.copy_hits,
        }


def score_predictions(
    suite: Cap0EvalSuiteV1,
    predictions: Mapping[str, str],
    *,
    adapter: GrammarCapabilityAdapterV1,
    dsl: str = DSL,
) -> SuiteScoreV1:
    """Score one arm's predictions into Cap0MetricsV1 + strict side metrics."""
    rows = list(suite.rows)
    n = len(rows) or 1
    pred = {row.row_id: predictions.get(row.row_id, "") for row in rows}
    fingerprints = {rid: _fingerprint_or_none(text, dsl=dsl) for rid, text in pred.items()}

    canonical_rows = suite.rows_for("atomic_coverage") + suite.rows_for(
        "expanded_canonicalization"
    )
    canonical_hits = sum(
        1
        for row in canonical_rows
        if fingerprints[row.row_id]
        and fingerprints[row.row_id] == row.target.get("canonical_fingerprint")
    )
    prefix_rows = suite.rows_for("prefix_completion")
    prefix_success = tuple(
        int(
            pred[row.row_id].strip()
            in {
                str(item.get("source")).strip()
                for item in row.target.get("accepted") or ()
                if isinstance(item, dict)
            }
        )
        for row in prefix_rows
    )
    comp_rows = suite.rows_for("compositional_held_out")
    comp_success = tuple(
        int(
            bool(fingerprints[row.row_id])
            and fingerprints[row.row_id] == row.target.get("canonical_fingerprint")
        )
        for row in comp_rows
    )
    near_miss = suite.rows_for("marker_permutation") + suite.rows_for("anti_identity")
    structure_hits = sum(
        1
        for row in near_miss
        if fingerprints[row.row_id] != row.target.get("negative_fingerprint")
    )
    binding_hits = sum(
        1
        for row in near_miss
        if fingerprints[row.row_id]
        and fingerprints[row.row_id] == row.target.get("source_fingerprint")
    )
    marker_rows = suite.rows_for("marker_permutation")
    marker_hits = sum(
        1
        for row in marker_rows
        if fingerprints[row.row_id] != row.target.get("negative_fingerprint")
    )

    # Production coverage: target productions realized in the prediction.
    covered = total = 0
    for row in suite.rows_for("atomic_coverage"):
        target_productions = set(row.target.get("productions") or ())
        if not target_productions:
            continue
        total += 1
        predicted = _production_ids_of(adapter, pred[row.row_id])
        covered += int(target_productions <= predicted)

    parse_valid = sum(
        1
        for row in rows
        if pred[row.row_id].strip() and fingerprints[row.row_id] is not None
    )
    empty = sum(1 for row in rows if not pred[row.row_id].strip())
    copy_hits = sum(
        1
        for row in rows
        if pred[row.row_id].strip()
        and pred[row.row_id].strip()
        in {
            str(row.payload.get(key)).strip()
            for key in ("canonical", "expanded", "prefix", "candidate")
            if isinstance(row.payload.get(key), str)
        }
    )
    distinct = len({text.strip() for text in pred.values()})

    metrics = Cap0MetricsV1(
        canonical_equivalence=canonical_hits / len(canonical_rows) if canonical_rows else 0.0,
        accepted_set_mass=sum(prefix_success) / len(prefix_rows) if prefix_rows else 0.0,
        production_coverage=covered / total if total else 0.0,
        structure=structure_hits / len(near_miss) if near_miss else 0.0,
        binding_aware=binding_hits / len(near_miss) if near_miss else 0.0,
        empty_trivial_rate=empty / n,
        diversity=distinct / n,
        identity_baseline_score=copy_hits / n,
        n_by_subsuite={name: len(suite.rows_for(name)) for name in (
            "atomic_coverage",
            "expanded_canonicalization",
            "compositional_held_out",
            "prefix_completion",
            "marker_permutation",
            "anti_identity",
        )},
    )
    return SuiteScoreV1(
        metrics=metrics,
        parse_valid_rate=parse_valid / n,
        marker_fidelity=marker_hits / len(marker_rows) if marker_rows else 0.0,
        compositional_success=comp_success,
        prefix_success=prefix_success,
        copy_hits=copy_hits,
    )


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
    staged_full: SuiteScoreV1,
    base: SuiteScoreV1,
    atomic: SuiteScoreV1,
    staged_gate: Cap0GateReportV1,
    staged_predictions: Mapping[str, str],
    control_predictions: Mapping[str, Mapping[str, str]],
    composition_evidence: Sequence[Mapping[str, Any]],
    prefix_evidence: Sequence[Mapping[str, Any]],
    contamination_findings: Sequence[str],
    binding: Mapping[str, str],
    plan_id: str = EXPERIMENT_ID,
    plan_sha256: str,
) -> CertificateDecisionV1:
    """Pure CERT_CAP0 decision; every stop rule rejects with its reason.

    ``binding`` binds checkpoint/corpus/suite/gate/config/code/evidence
    digests into the certificate artifact (issued or rejected).
    """
    codes: list[StopRule] = []

    # Contamination: any suite/leakage finding invalidates everything.
    if contamination_findings:
        codes.append(StopRule.CONTAMINATED)

    # Prediction-identical: STAGED_FULL emits the control's outputs verbatim.
    for control_id, control in sorted(control_predictions.items()):
        if control and all(
            staged_predictions.get(key, "") == value for key, value in control.items()
        ):
            codes.append(StopRule.PREDICTION_IDENTICAL)

    # Copy-dominated: the copy shortcut explains STAGED_FULL, or no paired win.
    gate = staged_gate.gate
    copy_dominated = (
        staged_full.metrics.identity_baseline_score > gate.max_identity_baseline_score
    )
    wins = [
        evidence
        for evidence in (*composition_evidence, *prefix_evidence)
        if evidence.get("meets_preregistered_effect")
    ]
    if copy_dominated or not wins:
        codes.append(StopRule.COPY_DOMINATED)

    # Structurally regressive vs BASE (exact-zero tolerance).
    regressions = {
        "parse_valid_rate": base.parse_valid_rate - staged_full.parse_valid_rate,
        "marker_fidelity": base.marker_fidelity - staged_full.marker_fidelity,
        "binding_aware": base.metrics.binding_aware - staged_full.metrics.binding_aware,
        "canonical_equivalence": (
            base.metrics.canonical_equivalence - staged_full.metrics.canonical_equivalence
        ),
        "empty_trivial_rate": (
            staged_full.metrics.empty_trivial_rate - base.metrics.empty_trivial_rate
        ),
    }
    if any(drop > REGRESSION_TOLERANCE for drop in regressions.values()):
        codes.append(StopRule.STRUCTURALLY_REGRESSIVE)

    # Underpowered: gate n unmet or paired confidence below preregistration.
    if staged_gate.status != "certified":
        codes.append(StopRule.UNDERPOWERED)

    # Deduplicate, stable order.
    ordered = tuple(dict.fromkeys(codes))
    passed = not ordered

    process = ProcessProvenanceV1(
        process_id=plan_id,
        process_version=str(binding.get("gate_version", GATE_VERSION)),
        config_sha256=_sha256_text(str(binding.get("config_sha256", plan_sha256))),
        code_sha256=_sha256_text(str(binding.get("code_commit", "UNKNOWN"))),
    )
    evidence_ids = tuple(
        _sha256_text(str(binding[key]))
        for key in ("suite_sha256", "evidence_json_uri")
        if binding.get(key)
    ) or (plan_sha256,)
    artifact = CapabilityCertificateV1(
        capability=Capability.CAP0_GRAMMAR,
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
    artifact_dict["regressions_vs_base"] = regressions
    return CertificateDecisionV1(
        status="issued" if passed else "rejected",
        certificate=CERT_CAP0 if passed else None,
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
    suite: Cap0EvalSuiteV1
    suite_valid_ok: bool
    contamination: tuple[str, ...]
    corpora: dict[str, ArmCorpusV1]
    exposure: dict[str, Any]
    corpus_sha256: str


def prepare_context(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    records_per_arm: int = DEFAULT_RECORDS_PER_ARM,
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    suite_seed: int = 7,
) -> tuple[ExperimentContextV1, Any]:
    """Build the preregistration, frozen suite, and matched arm corpora."""
    from slm_training.dsl.pack import get_pack

    adapter = GrammarCapabilityAdapterV1(get_pack(DSL))
    prereg = build_preregistration(
        seeds=seeds,
        records_per_arm=records_per_arm,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        suite_seed=suite_seed,
    )

    # Frozen DSH1-09 suite from eval-split roots only; zero-tolerance valid.
    suite = build_cap0_eval_suite(
        list(EVAL_ANSWERS), adapter=adapter, seed=suite_seed, dsl=DSL
    )
    verify_suite_integrity(suite)
    suite_ok = suite_valid(suite, adapter=adapter, dsl=DSL)
    stop_findings = stop_rule_violations(suite, adapter=adapter)

    # Contamination: train/eval families disjoint AND no shared fingerprints.
    train_families = {family for _, family in TRAIN_ANSWERS}
    eval_families = {row.root_family_id for row in suite.rows}
    train_fingerprints = {
        canonical_fingerprint(source, dsl=DSL) for source, _ in TRAIN_ANSWERS
    }
    eval_fingerprints = {
        row.target.get("canonical_fingerprint")
        for row in suite.rows
        if row.target.get("canonical_fingerprint")
    }
    contamination: list[str] = []
    if train_families & eval_families:
        contamination.append(
            f"train/eval family overlap: {sorted(train_families & eval_families)}"
        )
    if train_fingerprints & eval_fingerprints:
        contamination.append("train answer fingerprints appear in the eval suite")
    if not suite_ok:
        contamination.append("frozen suite failed zero-tolerance validation")
    contamination.extend(f"{f.code.value}:{f.row_id}" for f in stop_findings)

    # Arm corpora with matched exposure.
    corpora = {
        arm: build_arm_corpus(
            arm,
            adapter=adapter,
            records_per_arm=records_per_arm,
            seed=suite_seed,
            dsl=DSL,
        )
        for arm in ARMS
    }
    exposure = exposure_accounting(corpora)
    corpus_sha = _sha256_json({arm: corpora[arm].sha256 for arm in ARMS})
    context = ExperimentContextV1(
        prereg=prereg,
        suite=suite,
        suite_valid_ok=suite_ok,
        contamination=tuple(contamination),
        corpora=corpora,
        exposure=exposure,
        corpus_sha256=corpus_sha,
    )
    return context, adapter


def shard_path(output_dir: Path, arm: str) -> Path:
    return Path(output_dir) / f"shard_{arm}.json"


def run_arm_shard(
    context: ExperimentContextV1,
    adapter: Any,
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
        predictions = predict_suite(model, context.suite)
        score = score_predictions(context.suite, predictions, adapter=adapter, dsl=DSL)
        gate_report = evaluate_gate(score.metrics, gate=Cap0GateV1())
        seed_rows.append(
            {
                "seed": seed,
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "score": score.to_dict(),
                "gate": gate_report.to_dict(),
                "predictions": dict(predictions),
            }
        )
    shard = {
        "arm_id": arm,
        "corpus": context.corpora[arm].to_dict(),
        "seeds": seed_rows,
        "preregistration_sha256": prereg.sha256,
        "suite_rows_sha256": context.suite.manifest["rows_sha256"],
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


def _suite_score_from_dict(data: Mapping[str, Any]) -> SuiteScoreV1:
    return SuiteScoreV1(
        metrics=Cap0MetricsV1(**data["metrics"]),
        parse_valid_rate=data["parse_valid_rate"],
        marker_fidelity=data["marker_fidelity"],
        compositional_success=tuple(data["compositional_success"]),
        prefix_success=tuple(data["prefix_success"]),
        copy_hits=data["copy_hits"],
    )


def finalize_report(
    context: ExperimentContextV1,
    arm_results: Mapping[str, Any],
    *,
    started: float | None = None,
) -> dict[str, Any]:
    """Aggregate arm shards into the full report + CERT_CAP0 decision."""
    suite = context.suite
    prereg = context.prereg

    # Collapse baselines must fail the gate (control evidence).
    baselines = {
        "identity": evaluate_gate(
            identity_baseline_metrics(suite, dsl=DSL), gate=Cap0GateV1()
        ).to_dict(),
        "unchanged_input": evaluate_gate(
            unchanged_input_baseline_metrics(suite, dsl=DSL), gate=Cap0GateV1()
        ).to_dict(),
        "empty_trivial": evaluate_gate(
            empty_trivial_baseline_metrics(suite), gate=Cap0GateV1()
        ).to_dict(),
    }
    baselines_fail = all(
        row["status"] != "certified" and not row["passed"] for row in baselines.values()
    )

    # Paired evidence: STAGED_FULL vs controls, pooled across preregistered
    # seeds, on held-out composition and accepted-set completion.
    def _pooled(arm: str, key: str) -> list[int]:
        values: list[int] = []
        for row in arm_results[arm]["seeds"]:
            values.extend(row["score"][key])
        return values

    composition_evidence = [
        paired_evidence(_pooled(control, "compositional_success"),
                        _pooled("STAGED_FULL", "compositional_success"),
                        label=f"compositional_held_out:STAGED_FULL_vs_{control}")
        for control in ("BASE", "ATOMIC_ONLY")
    ]
    prefix_evidence = [
        paired_evidence(_pooled(control, "prefix_success"),
                        _pooled("STAGED_FULL", "prefix_success"),
                        label=f"prefix_completion:STAGED_FULL_vs_{control}")
        for control in ("BASE", "ATOMIC_ONLY")
    ]

    # Certificate decision: primary (first) preregistered seed for per-record
    # identity/regression checks plus that seed's gate status.
    primary = 0
    staged_primary = arm_results["STAGED_FULL"]["seeds"][primary]
    base_primary = arm_results["BASE"]["seeds"][primary]
    atomic_primary = arm_results["ATOMIC_ONLY"]["seeds"][primary]
    staged_score = _suite_score_from_dict(staged_primary["score"])
    base_score = _suite_score_from_dict(base_primary["score"])
    atomic_score = _suite_score_from_dict(atomic_primary["score"])
    staged_gate = evaluate_gate(staged_score.metrics, gate=Cap0GateV1())

    binding = {
        "checkpoint_sha256": staged_primary["checkpoint_sha256"],
        "corpus_sha256": context.corpus_sha256,
        "suite_sha256": str(suite.manifest["rows_sha256"]),
        "gate_version": GATE_VERSION,
        "config_sha256": prereg.sha256,
        "code_commit": "UNKNOWN",  # replaced by the CLI with the stamp commit
        "evidence_json_uri": "docs/design/iter-slm362-cap0-experiment-20260725.json",
        "evidence_md_uri": "docs/design/iter-slm362-cap0-experiment-20260725.md",
    }
    decision = decide_certificate(
        staged_full=staged_score,
        base=base_score,
        atomic=atomic_score,
        staged_gate=staged_gate,
        staged_predictions=staged_primary["predictions"],
        control_predictions={
            "BASE": base_primary["predictions"],
            "ATOMIC_ONLY": atomic_primary["predictions"],
        },
        composition_evidence=composition_evidence,
        prefix_evidence=prefix_evidence,
        contamination_findings=list(context.contamination),
        binding=binding,
        plan_sha256=prereg.sha256,
    )

    return {
        "schema": "Slm362Cap0ExperimentReportV1",
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "claim_class": "fixture",
        # Preregistered block FIRST — locked before any result below.
        "preregistration": prereg.to_dict(),
        "preregistration_sha256": prereg.sha256,
        "suite": {
            "seed": suite.seed,
            "manifest": dict(suite.manifest),
            "valid": context.suite_valid_ok,
            "contamination_findings": list(context.contamination),
        },
        "exposure": context.exposure,
        "corpus_sha256": context.corpus_sha256,
        "arms": dict(arm_results),
        "baselines": baselines,
        "baselines_fail_gate": baselines_fail,
        "paired_evidence": {
            "compositional_held_out": composition_evidence,
            "prefix_completion": prefix_evidence,
        },
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
    suite_seed: int = 7,
) -> dict[str, Any]:
    """Run the matched CAP0 curriculum experiment at fixture scale."""
    started = time.perf_counter()
    context, adapter = prepare_context(
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
        for shard in [run_arm_shard(context, adapter, arm, output_dir=output_dir)]
    }
    return finalize_report(context, arm_results, started=started)


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def render_markdown(report: Mapping[str, Any], *, command: str) -> str:
    prereg = report["preregistration"]
    cert = report["certificate"]
    lines = [
        "# SLM-362 (DSH1-10): matched CAP0 curriculum experiment",
        "",
        "**Claim class:** fixture (fixture n cannot issue CERT_CAP0)",
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
        f"- Preregistered minimum effect: paired effect > {prereg['min_effect']}, "
        f"McNemar p <= {prereg['max_p_value']}",
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
        "",
        "## Arm metrics (primary seed)",
        "",
        "| Arm | canon-equiv | accepted-mass | prod-cov | structure | binding | "
        "empty | diversity | identity-base | gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in prereg["arms"]:
        primary = report["arms"][arm]["seeds"][0]
        m = primary["score"]["metrics"]
        lines.append(
            f"| {arm} | {m['canonical_equivalence']:.3f} | "
            f"{m['accepted_set_mass']:.3f} | {m['production_coverage']:.3f} | "
            f"{m['structure']:.3f} | {m['binding_aware']:.3f} | "
            f"{m['empty_trivial_rate']:.3f} | {m['diversity']:.3f} | "
            f"{m['identity_baseline_score']:.3f} | {primary['gate']['status']} |"
        )
    lines += [
        "",
        "## Paired evidence (STAGED_FULL vs controls, pooled seeds)",
        "",
        "| Comparison | n | control | candidate | effect | p | meets effect |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for group in report["paired_evidence"].values():
        for row in group:
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
        "## Baselines (must fail the gate)",
        "",
        f"- All collapse baselines fail: {report['baselines_fail_gate']}",
        "",
        "## Certificate decision",
        "",
        f"- Status: **{cert['status']}**; certificate: {cert['certificate']}",
        f"- Rejection codes: {cert['rejection_codes']}",
        f"- Contamination findings: {report['suite']['contamination_findings']}",
        "- CAP1/CAP2 training: "
        + ("open" if cert["status"] == "issued" else "CLOSED (stop rule)"),
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
    "DEFAULT_SEEDS",
    "EVAL_ANSWERS",
    "EXPERIMENT_ID",
    "PREREGISTERED_MAX_P_VALUE",
    "PREREGISTERED_MIN_EFFECT",
    "REGRESSION_TOLERANCE",
    "SCHEMA_VERSION",
    "TRAIN_ANSWERS",
    "ArmCorpusV1",
    "CertificateDecisionV1",
    "ExperimentContextV1",
    "PreregistrationV1",
    "StopRule",
    "SuiteScoreV1",
    "build_arm_corpus",
    "build_preregistration",
    "decide_certificate",
    "exposure_accounting",
    "finalize_report",
    "load_shards",
    "paired_evidence",
    "predict_suite",
    "prepare_context",
    "render_markdown",
    "run_arm_shard",
    "run_experiment",
    "score_predictions",
    "shard_path",
    "train_arm",
]
