"""Build a versioned semantic-contrast corpus for OpenUI (SPV2-01)."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from slm_training.data.contract import (
    GenerationRequest,
    canonical_slot_contract,
    project_template_markers,
)
from slm_training.data.progspec.generate import GeneratorConfig, ProgramGenerator
from slm_training.data.progspec.schema import ProgramSpec, emit_record
from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.semantic_contrast.schema import (
    ContrastFamily,
    ContrastPair,
    ContrastRole,
    ContrastSeverity,
    CorpusSplit,
    FamilyMetrics,
    SemanticContrastRecord,
)
from slm_training.data.semantic_contrast.transforms import generate_transforms
from slm_training.data.semantic_plan.extract import OpenUISemanticPlanExtractor
from slm_training.data.semantic_plan.canonicalize import plan_factor_fingerprints
from slm_training.data.semantic_plan.seed import PlanSeedBuilder
from slm_training.data.store import DataStore, write_common_manifest
from slm_training.data.verify import (
    VerificationContext,
    run_preview_verifier_many,
    verify_record,
)
from slm_training.dsl.pack import get_pack
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.meaningful_program import binding_aware_meaningful_v2
from slm_training.harness_core.versioning import build_version_stamp


BUILDER_VERSION = "2.0.1"
PROGRAM_FAMILY = "semantic_contrast"


_HUMANIZE_RE = re.compile(r"(?<!^)(?=[A-Z])")


def _humanize_component_name(name: str) -> str:
    return _HUMANIZE_RE.sub(" ", name).lower()


def _prompt_for(spec: ProgramSpec) -> str:
    """Prompt that exposes the program's component and placeholder inventory."""
    raw_components = spec.facts.get("components") or []
    components = sorted(
        {
            _humanize_component_name(str(row))
            for row in raw_components
            if str(row) not in {"", "Stack"}
        }
    )
    placeholders = sorted(set(spec.facts.get("placeholders") or []))
    parts = ["Generate an OpenUI program"]
    if components:
        parts.append(f"with components: {', '.join(components)}")
    if placeholders:
        parts.append(f"using placeholders: {', '.join(placeholders)}")
    prompt = " ".join(parts) + "."
    # Inventory section so the evaluator can derive a deterministic contract.
    if placeholders:
        prompt += f"\nPlaceholders: {', '.join(placeholders)}"
    return prompt


def _request_for(record: ExampleRecord) -> GenerationRequest:
    return GenerationRequest(
        prompt=record.prompt,
        slot_contract=canonical_slot_contract(record.openui),
    )


def _score(record: ExampleRecord) -> dict[str, Any]:
    request = _request_for(record)
    report = binding_aware_meaningful_v2(record.openui, record=record, request=request)
    return report.to_dict()


def _verify(source: str) -> tuple[bool, str | None, Any]:
    record = ExampleRecord(
        id="verify-probe",
        prompt="verify semantic contrast candidate",
        openui=source,
        placeholders=[],
        split="train",
        source="semantic_contrast_probe",
    )
    report = verify_record(record, VerificationContext(source_kind="program"))
    return report.ok, None if report.tier is None else report.tier.value, report


def _make_negative_spec(
    source: ProgramSpec,
    corrupted_openui: str,
    transform_id: str,
    family: ContrastFamily,
    severity: Any,
    prompt: str,
) -> ProgramSpec:
    identity = json.dumps(
        [source.id, transform_id, corrupted_openui],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return ProgramSpec.from_openui(
        id=f"{source.id}_neg_{digest}",
        openui=corrupted_openui,
        facts={
            **source.facts,
            "semantic_contrast": True,
            "contrast_family": family.value,
            "contrast_transform": transform_id,
            "source_program_id": source.id,
        },
        program_family_id=f"{source.program_family_id}_contrast",
        lineage_id=source.lineage_id,
        split_group_id=source.split_group_id,
        split=source.split,
        provenance={
            **source.provenance,
            "builder": "semantic_contrast",
            "builder_version": BUILDER_VERSION,
            "transform_id": transform_id,
        },
    )


def _emit(spec: ProgramSpec, prompt: str, task: str) -> ExampleRecord:
    return emit_record(
        spec,
        prompt=prompt,
        task=task,
        source=PROGRAM_FAMILY,
        meta={
            "semantic_contrast": True,
            "contrast_family": spec.facts.get("contrast_family"),
            "contrast_transform": spec.facts.get("contrast_transform"),
            "source_program_id": spec.facts.get("source_program_id"),
        },
    )


class SemanticContrastBuilder:
    """Generate hard-valid, semantically wrong OpenUI contrast records."""

    def __init__(
        self,
        *,
        output_root: Path | str = Path("outputs/data"),
        dataset_id: str = "semantic_contrast_v1",
        seed: int = 0,
        source_count: int = 12,
        splits: tuple[CorpusSplit, ...] = ("train", "test"),
        split_weights: tuple[float, ...] = (0.8, 0.2),
        honesty_mode: str = "production",
        require_runtime: bool = False,
        min_pairs: int = 0,
        min_mutation_classes: int = 0,
        wide_sources: bool = False,
        strict_delta: bool = False,
    ) -> None:
        if len(splits) != len(split_weights):
            raise ValueError("splits and split_weights must have the same length")
        if not splits:
            raise ValueError("splits must be non-empty")
        self.dataset_id = dataset_id
        self.seed = seed
        self.source_count = source_count
        self.splits = splits
        self.split_weights = split_weights
        self.honesty_mode = honesty_mode
        self.require_runtime = require_runtime
        self.min_pairs = min_pairs
        self.min_mutation_classes = min_mutation_classes
        self.wide_sources = wide_sources
        self.strict_delta = strict_delta
        self.store = DataStore(local_root=output_root)
        self.output_dir = self.store.path("eval", dataset_id)
        self.pack = get_pack("openui")
        self.extractor = OpenUISemanticPlanExtractor()
        self.seed_builder = PlanSeedBuilder(self.pack)
        self._rng = random.Random(seed)

    def _assign_split(self, source_index: int) -> CorpusSplit:
        # Deterministic pseudo-random split assignment using a fixed multiplier.
        total = sum(self.split_weights)
        threshold = (source_index * 40503 + 101) % 1000
        position = threshold * total / 1000.0
        cursor = 0.0
        for split, weight in zip(self.splits, self.split_weights):
            cursor += weight
            if position < cursor:
                return split
        return self.splits[-1]

    def _build_sources(self) -> tuple[ProgramSpec, ...]:
        if self.wide_sources:
            groups = tuple(
                tuple(
                    "Button" if (mask >> bit) & 1 else "TextContent"
                    for bit in range(10)
                )
                for mask in range(1 << 10)
            )
            generator = ProgramGenerator(
                GeneratorConfig(
                    components=("TextContent", "Button"),
                    max_depth=2,
                    max_width=10,
                    selected_groups=groups,
                    split="train",
                ),
                seed=self.seed,
            )
            result = generator.generate(self.source_count)
            return tuple(
                self._with_opaque_markers(source) for source in result.programs
            )
        config = GeneratorConfig(
            max_depth=2,
            max_width=3,
            components=("TextContent", "Button"),
            split="train",
        )
        generator = ProgramGenerator(config, seed=self.seed)
        # Over-sample then filter for sources rich enough to carry a known
        # prompt-component contract.  Single-component roots leave the evaluator
        # in the UNKNOWN state, so they are dropped deterministically.
        result = generator.generate(self.source_count * 2)
        candidates = [self._with_opaque_markers(spec) for spec in result.programs]
        # Keep any source that has at least one non-Stack component and a
        # placeholder so the prompt contract is non-trivial.
        sources = [
            spec
            for spec in candidates
            if any(
                str(c) not in {"", "Stack"}
                for c in (spec.facts.get("components") or [])
            )
            and len(set(spec.facts.get("placeholders") or [])) >= 1
        ]
        if len(sources) < self.source_count:
            raise RuntimeError(
                f"only {len(sources)} of {self.source_count} requested sources "
                "passed the component/placeholder filter; increase source_count "
                "or widen the component set"
            )
        return tuple(sources[: self.source_count])

    @staticmethod
    def _with_opaque_markers(spec: ProgramSpec) -> ProgramSpec:
        markers = extract_placeholders(spec.canonical_openui)
        openui = project_template_markers(spec.canonical_openui, markers)
        assert openui is not None
        return ProgramSpec.from_openui(
            id=spec.id,
            openui=openui,
            facts={
                **spec.facts,
                "placeholders": extract_placeholders(openui),
            },
            program_family_id=spec.program_family_id,
            lineage_id=spec.lineage_id,
            split_group_id=spec.split_group_id,
            split=spec.split,
            derivative_refs=spec.derivative_refs,
            provenance=spec.provenance,
        )

    def _compile_candidate(self, plan: SemanticPlanV1) -> tuple[str | None, str | None]:
        seed_result = self.seed_builder.build(plan)
        if not seed_result.ok or seed_result.seed is None:
            return None, seed_result.reason
        markers = extract_placeholders(seed_result.seed)
        return project_template_markers(seed_result.seed, markers), None

    def _build_pair(
        self,
        source: ProgramSpec,
        plan: SemanticPlanV1,
        candidate: Any,
        split: CorpusSplit,
    ) -> ContrastPair | None:
        transform_id = candidate.transform_id
        family = candidate.family
        severity = candidate.severity
        description = candidate.description
        before_factors = plan_factor_fingerprints(plan)
        after_factors = plan_factor_fingerprints(candidate.plan)
        semantic_delta = {
            name: {"before": before_factors[name], "after": after_factors[name]}
            for name in before_factors
            if before_factors[name] != after_factors[name]
        }
        declared_delta_count = len(set(semantic_delta) - {"exact"})
        if (
            self.strict_delta
            and family is not ContrastFamily.POSITIVE
            and declared_delta_count != 1
        ):
            return None
        if family is ContrastFamily.POSITIVE:
            # Positive controls keep the original surface so the control pair
            # is bit-for-bit identical and is expected to pass meaningful eval.
            corrupted = source.canonical_openui
        else:
            corrupted, reason = self._compile_candidate(candidate.plan)
            if corrupted is None:
                return None

        verifier_ok, verifier_tier, _verifier_report = _verify(corrupted)
        if not verifier_ok:
            return None

        prompt = _prompt_for(source)
        positive_record = _emit(source, prompt, "generation")
        negative_spec = _make_negative_spec(
            source, corrupted, transform_id, family, severity, prompt
        )
        negative_spec = ProgramSpec.from_dict(
            {
                **negative_spec.to_dict(),
                "split": split,
            }
        )
        negative_record = _emit(negative_spec, prompt, "adversarial")

        positive_score = _score(positive_record)
        negative_score = _score(negative_record)

        positive = SemanticContrastRecord(
            record=positive_record,
            role=ContrastRole.POSITIVE,
            family=ContrastFamily.POSITIVE,
            transform_id="identity",
            transform_description="Original source program as positive control.",
            severity=ContrastSeverity.BENIGN,
            source_program_id=source.id,
            source_plan=plan.to_dict(),
            verifier_ok=True,
            verifier_tier=verifier_tier,
            meaningful_report=positive_score,
            meta={"split": split, "semantic_delta": semantic_delta},
        )
        negative = SemanticContrastRecord(
            record=negative_record,
            role=ContrastRole.NEGATIVE,
            family=family,
            transform_id=transform_id,
            transform_description=description,
            severity=severity,
            source_program_id=source.id,
            source_plan=candidate.plan.to_dict(),
            verifier_ok=verifier_ok,
            verifier_tier=verifier_tier,
            meaningful_report=negative_score,
            meta={"split": split, "semantic_delta": semantic_delta},
        )

        negative_has_expected_verdict = (
            bool(negative_score.get("verdict"))
            if family is ContrastFamily.POSITIVE
            else not bool(negative_score.get("verdict"))
        )
        admitted = bool(positive_score.get("verdict") and negative_has_expected_verdict)
        pair_id = f"{source.id}_{transform_id}_{split}"
        return ContrastPair(
            pair_id=pair_id,
            positive=positive,
            negative=negative,
            family=family,
            transform_id=transform_id,
            source_program_id=source.id,
            admitted=admitted,
            admission_reason=(
                None
                if admitted
                else (
                    "positive_failed"
                    if not positive_score.get("verdict")
                    else "negative_passed"
                )
            ),
        )

    def build(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sources = self._build_sources()
        pairs: list[ContrastPair] = []
        rejected: list[dict[str, Any]] = []
        records: list[SemanticContrastRecord] = []

        for index, source in enumerate(sources):
            split = self._assign_split(index)
            plan = self.extractor.extract(source, self.pack)
            candidates = generate_transforms(plan)
            for candidate in candidates:
                try:
                    pair = self._build_pair(source, plan, candidate, split)
                except Exception as exc:  # noqa: BLE001
                    rejected.append(
                        {
                            "source_id": source.id,
                            "transform_id": getattr(
                                candidate, "transform_id", "unknown"
                            ),
                            "reason": f"exception: {exc}",
                        }
                    )
                    continue
                if pair is None:
                    rejected.append(
                        {
                            "source_id": source.id,
                            "transform_id": getattr(
                                candidate, "transform_id", "unknown"
                            ),
                            "reason": "compilation or verifier rejection",
                        }
                    )
                    continue
                if not pair.admitted:
                    rejected.append(
                        {
                            "source_id": source.id,
                            "transform_id": pair.transform_id,
                            "reason": pair.admission_reason or "admission failed",
                        }
                    )
                    continue
                pairs.append(pair)
                records.append(pair.positive)
                records.append(pair.negative)

        if self.require_runtime:
            surfaces = tuple(dict.fromkeys(record.record.openui for record in records))
            evidence = dict(zip(surfaces, run_preview_verifier_many(surfaces)))
            admitted_pairs: list[ContrastPair] = []
            records = []
            for pair in pairs:
                sides = []
                for row in (pair.positive, pair.negative):
                    runtime = evidence[row.record.openui]
                    report = verify_record(
                        row.record,
                        VerificationContext(
                            source_kind="program",
                            runtime=runtime,
                            require_runtime=True,
                        ),
                    )
                    sides.append(
                        replace(
                            row,
                            verifier_ok=report.ok,
                            verifier_tier=report.tier.value,
                            meta={
                                **row.meta,
                                "runtime_evidence": runtime.to_dict(),
                                "verification": report.to_dict(),
                            },
                        )
                    )
                checked = replace(
                    pair,
                    positive=sides[0],
                    negative=sides[1],
                    admitted=pair.admitted
                    and sides[0].verifier_ok
                    and sides[1].verifier_ok,
                    admission_reason=(
                        pair.admission_reason
                        if pair.admitted
                        else pair.admission_reason
                    )
                    or (
                        None
                        if sides[0].verifier_ok and sides[1].verifier_ok
                        else "runtime_failed"
                    ),
                )
                if checked.admitted:
                    admitted_pairs.append(checked)
                    records.extend(sides)
                else:
                    rejected.append(
                        {
                            "source_id": checked.source_program_id,
                            "transform_id": checked.transform_id,
                            "reason": checked.admission_reason or "runtime_failed",
                        }
                    )
            pairs = admitted_pairs

        mutation_classes = sorted(
            {
                pair.transform_id
                for pair in pairs
                if pair.family is not ContrastFamily.POSITIVE
            }
        )
        if len(pairs) < self.min_pairs:
            raise RuntimeError(
                f"admitted pairs {len(pairs)} below required minimum {self.min_pairs}"
            )
        if len(mutation_classes) < self.min_mutation_classes:
            raise RuntimeError(
                f"mutation classes {len(mutation_classes)} below required minimum {self.min_mutation_classes}"
            )

        scoreboard = self._scoreboard(pairs)
        summary = {
            "builder_version": BUILDER_VERSION,
            "dataset_id": self.dataset_id,
            "seed": self.seed,
            "source_count": len(sources),
            "pairs": len(pairs),
            "records": len(records),
            "rejected": len(rejected),
            "mutation_classes": mutation_classes,
            "runtime_required": self.require_runtime,
            "scoreboard": [m.to_dict() for m in scoreboard],
            "version_stamp": build_version_stamp(
                "data.semantic_contrast",
                "evals.meaningful_program",
            ),
        }

        (self.output_dir / "pairs.jsonl").write_text(
            "".join(json.dumps(p.to_dict()) + "\n" for p in pairs), encoding="utf-8"
        )
        (self.output_dir / "records.jsonl").write_text(
            "".join(json.dumps(r.to_dict()) + "\n" for r in records), encoding="utf-8"
        )
        (self.output_dir / "rejected.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rejected), encoding="utf-8"
        )
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (self.output_dir / "source_manifest.json").write_text(
            json.dumps(
                {"seed": self.seed, "sources": [source.id for source in sources]},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "quality_report.json").write_text(
            json.dumps(
                {
                    "schema": "semantic_contrast_quality/v1",
                    "pairs": len(pairs),
                    "mutation_classes": mutation_classes,
                    "warnings": [],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "synthesis_feedback.json").write_text(
            json.dumps(
                {
                    "schema": "synthesis_feedback/v1",
                    "recommendations": [],
                    "experiment_candidates": [],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        manifest = {
            "schema_version": 2,
            "dataset_id": self.dataset_id,
            "kind": "eval",
            "builder": "semantic_contrast",
            "builder_version": BUILDER_VERSION,
            "trace_id": f"semantic-contrast-{self.dataset_id}",
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        write_common_manifest(
            self.output_dir,
            kind="eval",
            dataset_id=self.dataset_id,
            trace_id=manifest["trace_id"],
            immutable=False,
        )
        return summary

    def _scoreboard(self, pairs: list[ContrastPair]) -> list[FamilyMetrics]:
        # Group records by their family: one positive-control family and one
        # entry per corruption family for the negative sides.
        by_family: dict[str, list[SemanticContrastRecord]] = {}
        for pair in pairs:
            by_family.setdefault(ContrastFamily.POSITIVE.value, []).append(
                pair.positive
            )
            by_family.setdefault(pair.family.value, []).append(pair.negative)
        metrics: list[FamilyMetrics] = []
        for family, rows in sorted(by_family.items()):
            total = len(rows)
            verifier_pass = sum(r.verifier_ok for r in rows)
            meaningful_pass = sum(
                bool(r.meaningful_report.get("verdict")) for r in rows
            )
            negative_rows = [r for r in rows if r.role is ContrastRole.NEGATIVE]
            positive_rows = [r for r in rows if r.role is ContrastRole.POSITIVE]
            if family == ContrastFamily.POSITIVE.value:
                # Positive-control family: admission = positives that pass.
                admitted = sum(
                    bool(r.meaningful_report.get("verdict")) for r in positive_rows
                )
                false_negative_rate = 0.0
                reason_rows = positive_rows
            else:
                # A negative family is "admitted" when its positive control
                # passed and the negative itself failed meaningful-program eval.
                admitted = sum(
                    1
                    for pair in pairs
                    if pair.family.value == family
                    and pair.positive.meaningful_report.get("verdict") is True
                    and pair.negative.meaningful_report.get("verdict") is False
                )
                false_negatives = sum(
                    bool(r.meaningful_report.get("verdict")) for r in negative_rows
                )
                false_negative_rate = (
                    false_negatives / len(negative_rows) if negative_rows else 0.0
                )
                reason_rows = negative_rows
            mean_reasons = sum(
                len(r.meaningful_report.get("reason_codes", [])) for r in reason_rows
            ) / max(1, len(reason_rows))
            reasons = Counter(
                reason
                for r in reason_rows
                for reason in r.meaningful_report.get("reason_codes", [])
            )
            metrics.append(
                FamilyMetrics(
                    family=family,
                    n_total=total,
                    n_admitted=admitted,
                    verifier_pass_rate=verifier_pass / total if total else 0.0,
                    meaningful_pass_rate=meaningful_pass / total if total else 0.0,
                    false_negative_rate=false_negative_rate,
                    mean_reason_count=mean_reasons,
                    top_reasons=tuple(reason for reason, _ in reasons.most_common(5)),
                )
            )
        return metrics


__all__ = ["BUILDER_VERSION", "PROGRAM_FAMILY", "SemanticContrastBuilder"]
