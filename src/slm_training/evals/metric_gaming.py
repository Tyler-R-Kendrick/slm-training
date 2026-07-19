"""SDE0-02 metric-gaming stress suite.

Builds hard-valid semantic contrast cases across four slices:

* minimal_valid — valid programs that strip requested semantic content;
* rare_omission — programs that omit/substitute historically rare components;
* inventory_free — paired with and without the explicit prompt slot inventory;
* retry_sensitive — first-attempt / selected-attempt / oracle-best comparisons.

Every case is scored with the existing :func:`binding_aware_meaningful_v2` metric.
No new quality heuristic is introduced; the suite only exposes deterministic
metric-gaming channels in the current judge.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from slm_training.data.contract import GenerationRequest
from slm_training.dsl.parser import ParseError, validate
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.meaningful_program import (
    SemanticMeaningReportV2,
    binding_aware_meaningful_v2,
)

SCHEMA_VERSION = "metric_gaming/v1"

SLICE_MINIMAL_VALID = "minimal_valid"
SLICE_RARE_OMISSION = "rare_omission"
SLICE_INVENTORY_FREE = "inventory_free"
SLICE_RETRY_SENSITIVE = "retry_sensitive"

ALL_SLICES = (
    SLICE_MINIMAL_VALID,
    SLICE_RARE_OMISSION,
    SLICE_INVENTORY_FREE,
    SLICE_RETRY_SENSITIVE,
)

# Historically weak / low-recall components identified in the issue.
RARE_COMPONENTS = ("Form", "Tabs", "SwitchItem", "Slider")


@dataclass(frozen=True)
class MetricGamingCase:
    """One adversarial case with expected metric behavior."""

    id: str
    slice: str
    prompt: str
    pred_openui: str
    request: GenerationRequest
    expected_verdict: bool
    expected_reason_substrings: tuple[str, ...] = ()
    gold_openui: str | None = None
    transform: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["request"] = self.request.to_dict()
        return data


@dataclass(frozen=True)
class ScoredCase:
    """A case plus its v2 semantic-meaning report."""

    case: MetricGamingCase
    report: SemanticMeaningReportV2

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "report": self.report.to_dict(),
        }


@dataclass
class MetricGamingSliceReport:
    """Aggregate statistics for one slice."""

    slice: str
    n: int = 0
    expected_failures: int = 0
    observed_failures: int = 0
    false_positives: int = 0  # negative case passed
    false_negatives: int = 0  # positive case failed
    strict_rate: float = 0.0
    coverage_conditioned_rate: float = 0.0
    reason_prevalence: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice": self.slice,
            "n": self.n,
            "expected_failures": self.expected_failures,
            "observed_failures": self.observed_failures,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "strict_rate": self.strict_rate,
            "coverage_conditioned_rate": self.coverage_conditioned_rate,
            "reason_prevalence": self.reason_prevalence,
        }


@dataclass
class MetricGamingReport:
    """Top-level report emitted by the fixture runner."""

    schema_version: str
    metric_name: str
    metric_version: str
    n_cases: int
    strict_rate: float
    coverage_conditioned_rate: float
    false_positive_count: int
    false_negative_count: int
    slices: dict[str, MetricGamingSliceReport]
    cases: list[ScoredCase] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metric_name": self.metric_name,
            "metric_version": self.metric_version,
            "n_cases": self.n_cases,
            "strict_rate": self.strict_rate,
            "coverage_conditioned_rate": self.coverage_conditioned_rate,
            "false_positive_count": self.false_positive_count,
            "false_negative_count": self.false_negative_count,
            "slices": {
                name: rep.to_dict() for name, rep in self.slices.items()
            },
            "cases": [c.to_dict() for c in self.cases],
        }


def _record(
    case_id: str,
    prompt: str,
    openui: str,
) -> ExampleRecord:
    """Build a normalized adversarial ExampleRecord."""
    return ExampleRecord(
        id=case_id,
        prompt=prompt,
        openui=openui,
        split="adversarial",
        source="metric_gaming_fixture",
    )


def _request(
    prompt: str,
    slot_contract: tuple[str, ...] = (),
) -> GenerationRequest:
    return GenerationRequest(prompt=prompt, slot_contract=slot_contract)


def _ensure_valid(openui: str) -> bool:
    """Return True iff the OpenUI source parses and schema-validates."""
    try:
        validate(openui)
        return True
    except (ParseError, RuntimeError, ValueError):
        return False


def _validate_case(case: MetricGamingCase) -> None:
    """Fail closed if a pred is not hard-valid."""
    if not _ensure_valid(case.pred_openui):
        raise ValueError(
            f"Case {case.id!r} pred is not parser/schema-valid:\n{case.pred_openui}"
        )


def _score_case(case: MetricGamingCase) -> ScoredCase:
    record = _record(case.id, case.prompt, case.pred_openui)
    report = binding_aware_meaningful_v2(
        case.pred_openui,
        record=record,
        request=case.request,
    )
    return ScoredCase(case=case, report=report)


def score_cases(cases: Iterable[MetricGamingCase]) -> list[ScoredCase]:
    """Score every case and return scored rows."""
    return [_score_case(case) for case in cases]


def _archetypes() -> list[dict[str, Any]]:
    """Canonical positive archetypes used to derive adversarial negatives."""
    return [
        {
            "id": "card",
            "prompt": (
                "Build a card with title :card.title and body :card.body. "
                "Placeholders: :card.title :card.body"
            ),
            "positive": (
                'root = Card([header, body])\n'
                'header = CardHeader(":card.title")\n'
                'body = TextContent(":card.body")\n'
            ),
            "slot_contract": (":card.title", ":card.body"),
        },
        {
            "id": "slider",
            "prompt": (
                "Build a Slider for volume with caption :settings.caption. "
                "Placeholders: :settings.caption"
            ),
            "positive": (
                'root = Stack([s])\n'
                's = Slider(":settings.caption", "continuous", 0, 100)\n'
            ),
            "slot_contract": (":settings.caption",),
        },
        {
            "id": "switch",
            "prompt": (
                "Build a switch item for notifications with caption :settings.caption "
                "and description :settings.desc. "
                "Placeholders: :settings.caption :settings.desc"
            ),
            "positive": (
                'root = Stack([s])\n'
                's = SwitchItem(":settings.caption", ":settings.desc", "notifications")\n'
            ),
            "slot_contract": (":settings.caption", ":settings.desc"),
        },
        {
            "id": "tabs",
            "prompt": (
                "Build tabs with a tab item whose trigger is :tab.trigger "
                "and content is :tab.content. "
                "Placeholders: :tab.trigger :tab.content"
            ),
            "positive": (
                'root = Tabs([tab1])\n'
                'tab1 = TabItem("tab1", ":tab.trigger", [TextContent(":tab.content")])\n'
            ),
            "slot_contract": (":tab.trigger", ":tab.content"),
        },
        {
            "id": "button",
            "prompt": (
                "Build a Button with action :btn.action. "
                "Placeholders: :btn.action"
            ),
            "positive": 'root = Button(":btn.action")\n',
            "slot_contract": (":btn.action",),
        },
        {
            "id": "callout",
            "prompt": (
                "Build a Callout with title :callout.title and description "
                ":callout.desc. Placeholders: :callout.title :callout.desc"
            ),
            "positive": (
                'root = Callout("info", ":callout.title", ":callout.desc")\n'
            ),
            "slot_contract": (":callout.title", ":callout.desc"),
        },
        {
            "id": "image_block",
            "prompt": (
                "Build an image block with source :img.src and alt :img.alt. "
                "Placeholders: :img.src :img.alt"
            ),
            "positive": 'root = ImageBlock(":img.src", ":img.alt")\n',
            "slot_contract": (":img.src", ":img.alt"),
        },
    ]


def build_minimal_valid_trap_cases(seed: int = 0) -> list[MetricGamingCase]:
    """Valid programs that keep a common shell but delete requested content."""
    del seed  # deterministic fixture; seed reserved for future expansion
    cases: list[MetricGamingCase] = []
    for arch in _archetypes():
        base_id = arch["id"]
        prompt = arch["prompt"]
        positive = arch["positive"]
        slots = arch["slot_contract"]

        # 1. Replace the requested top-level component with a generic Stack.
        cases.append(
            MetricGamingCase(
                id=f"{base_id}_minimal_shell_only",
                slice=SLICE_MINIMAL_VALID,
                prompt=prompt,
                pred_openui='root = Stack([TextContent(":placeholder.text")])\n',
                request=_request(prompt, slots),
                expected_verdict=False,
                expected_reason_substrings=("prompt_component_missing",),
                gold_openui=positive,
                transform="replace_root_with_stack",
                notes="valid Stack with text, but requested component absent",
            )
        )

        # 2. Empty top-level container.
        empty_root = {
            "card": 'root = Card([])\n',
            "slider": 'root = Stack([])\n',
            "switch": 'root = Stack([])\n',
            "tabs": 'root = Tabs([])\n',
            "button": 'root = Stack([])\n',
            "callout": 'root = Stack([])\n',
            "image_block": 'root = Stack([])\n',
        }[base_id]
        cases.append(
            MetricGamingCase(
                id=f"{base_id}_minimal_empty",
                slice=SLICE_MINIMAL_VALID,
                prompt=prompt,
                pred_openui=empty_root,
                request=_request(prompt, slots),
                expected_verdict=False,
                expected_reason_substrings=("required_placeholder_missing",),
                gold_openui=positive,
                transform="empty_root",
                notes="empty but parser/schema-valid container",
            )
        )

        # 3. Common-root substitution that preserves one placeholder.
        substitution = {
            "card": 'root = Card([TextContent(":card.title")])\n',
            "slider": 'root = Stack([TextContent(":settings.caption")])\n',
            "switch": 'root = Stack([TextContent(":settings.caption")])\n',
            "tabs": 'root = Tabs([TabItem("tab1", ":tab.trigger", [])])\n',
            "button": 'root = Stack([TextContent(":btn.action")])\n',
            "callout": 'root = Callout("info", ":callout.title", ":callout.title")\n',
            "image_block": 'root = Stack([TextContent(":img.alt")])\n',
        }[base_id]
        partial_reasons = {
            "slider": ("required_component_missing",),
            "button": ("placeholder_semantic_role_mismatch",),
        }.get(base_id, ("required_placeholder_missing",))
        cases.append(
            MetricGamingCase(
                id=f"{base_id}_minimal_partial",
                slice=SLICE_MINIMAL_VALID,
                prompt=prompt,
                pred_openui=substitution,
                request=_request(prompt, slots),
                expected_verdict=False,
                expected_reason_substrings=partial_reasons,
                gold_openui=positive,
                transform="partial_content",
                notes="retains shell and one placeholder, omits requested remainder",
            )
        )

        # 4. Swap the requested component type for another valid component.
        type_swap = {
            "card": 'root = Stack([Button(":card.title")])\n',
            "slider": 'root = Card([TextContent(":settings.caption")])\n',
            "switch": 'root = Slider(":settings.caption", "continuous", 0, 100)\n',
            "tabs": 'root = Stack([Button(":tab.trigger")])\n',
            "button": 'root = TextContent(":btn.action")\n',
            "callout": 'root = Card([TextContent(":callout.title")])\n',
            "image_block": 'root = TextContent(":img.alt")\n',
        }[base_id]
        cases.append(
            MetricGamingCase(
                id=f"{base_id}_minimal_type_swap",
                slice=SLICE_MINIMAL_VALID,
                prompt=prompt,
                pred_openui=type_swap,
                request=_request(prompt, slots),
                expected_verdict=False,
                expected_reason_substrings=("prompt_component_missing",),
                gold_openui=positive,
                transform="type_swap",
                notes="parser/schema-valid component substitution, wrong requested type",
            )
        )

        # 5. Requested component present but nested and empty.
        nested_empty = {
            "card": 'root = Card([Stack([])])\n',
            "slider": 'root = Stack([Card([])])\n',
            "switch": 'root = Stack([Card([])])\n',
            "tabs": 'root = Tabs([TabItem("tab1", ":tab.trigger", [Stack([])])])\n',
            "button": 'root = Stack([Card([])])\n',
            "callout": 'root = Callout("info", ":callout.title", ":callout.title")\n',
            "image_block": 'root = Stack([ImageBlock(":img.src", ":img.src")])\n',
        }[base_id]
        cases.append(
            MetricGamingCase(
                id=f"{base_id}_minimal_nested_empty",
                slice=SLICE_MINIMAL_VALID,
                prompt=prompt,
                pred_openui=nested_empty,
                request=_request(prompt, slots),
                expected_verdict=False,
                expected_reason_substrings=("required_placeholder_missing",),
                gold_openui=positive,
                transform="nested_empty",
                notes="requested component present but content is empty or duplicated",
            )
        )
    return cases


def build_rare_component_omission_cases(seed: int = 0) -> list[MetricGamingCase]:
    """Programs that omit or substitute a rare component while staying valid."""
    del seed
    cases: list[MetricGamingCase] = []

    rare_positive_programs = {
        "Tabs": {
            "prompt": (
                "Build tabs with a tab item whose trigger is :tab.trigger "
                "and content is :tab.content. Placeholders: :tab.trigger :tab.content"
            ),
            "positive": (
                'root = Tabs([tab1])\n'
                'tab1 = TabItem("tab1", ":tab.trigger", [TextContent(":tab.content")])\n'
            ),
            "slots": (":tab.trigger", ":tab.content"),
        },
        "Slider": {
            "prompt": (
                "Build a Slider with caption :settings.caption. "
                "Placeholders: :settings.caption"
            ),
            "positive": (
                'root = Stack([s])\n'
                's = Slider(":settings.caption", "continuous", 0, 100)\n'
            ),
            "slots": (":settings.caption",),
        },
        "SwitchItem": {
            "prompt": (
                "Build a switch item with caption :settings.caption and description "
                ":settings.desc. Placeholders: :settings.caption :settings.desc"
            ),
            "positive": (
                'root = Stack([s])\n'
                's = SwitchItem(":settings.caption", ":settings.desc", "notifications")\n'
            ),
            "slots": (":settings.caption", ":settings.desc"),
        },
        "Form": {
            "prompt": (
                "Build a form with email input and submit button. "
                "Placeholders: :form.email :form.submit_text"
            ),
            "positive": (
                'root = Form("contact", buttons, [email])\n'
                'buttons = Buttons([submit])\n'
                'submit = Button(":form.submit_text")\n'
                'email = FormControl(":form.email_label", input)\n'
                'input = Input(":form.email")\n'
            ),
            "slots": (":form.email", ":form.submit_text", ":form.email_label"),
        },
    }

    substitutions = {
        "Tabs": [
            ('root = Stack([TextContent(":tab.trigger")])\n', "replace_with_stack"),
            ('root = Stack([Button(":tab.trigger")])\n', "replace_with_button"),
            ('root = Stack([])\n', "omit"),
        ],
        "Slider": [
            ('root = Stack([Button(":settings.caption")])\n', "replace_with_button"),
            ('root = Stack([TextContent(":settings.caption")])\n', "replace_with_text"),
            ('root = Stack([])\n', "omit"),
        ],
        "SwitchItem": [
            ('root = Stack([TextContent(":settings.caption")])\n', "replace_with_text"),
            ('root = Stack([Button(":settings.caption")])\n', "replace_with_button"),
            ('root = Stack([])\n', "omit"),
        ],
        "Form": [
            ('root = Stack([submit])\nsubmit = Button(":form.submit_text")\n', "replace_with_stack"),
            ('root = Stack([Button(":form.submit_text")])\n', "replace_with_button"),
            ('root = Stack([])\n', "omit"),
        ],
    }

    for rare, meta in rare_positive_programs.items():
        for neg_openui, transform in substitutions.get(rare, []):
            reasons = (
                ("no_nontrivial_content",)
                if transform == "omit"
                else ("prompt_component_missing",)
            )
            cases.append(
                MetricGamingCase(
                    id=f"rare_{rare.lower()}_{transform}",
                    slice=SLICE_RARE_OMISSION,
                    prompt=meta["prompt"],
                    pred_openui=neg_openui,
                    request=_request(meta["prompt"], meta["slots"]),
                    expected_verdict=False,
                    expected_reason_substrings=reasons,
                    gold_openui=meta["positive"],
                    transform=transform,
                    notes=f"{rare} requested but replaced/omitted with a valid alternative",
                )
            )
    return cases


def build_inventory_free_binding_cases(seed: int = 0) -> list[MetricGamingCase]:
    """Paired cases with and without the explicit prompt slot inventory."""
    del seed
    cases: list[MetricGamingCase] = []
    for arch in _archetypes():
        base_id = arch["id"]
        full_prompt = arch["prompt"]
        positive = arch["positive"]
        slots = arch["slot_contract"]

        # Positive: explicit Placeholders section in prompt + slot_contract.
        cases.append(
            MetricGamingCase(
                id=f"{base_id}_inventory_on",
                slice=SLICE_INVENTORY_FREE,
                prompt=full_prompt,
                pred_openui=positive,
                request=_request(full_prompt, slots),
                expected_verdict=True,
                expected_reason_substrings=(),
                gold_openui=positive,
                transform="with_inventory",
                notes="prompt inventory surfaced; binding is contract-supported",
            )
        )

        # Negative: same authored intent, no Placeholders section, no slot_contract.
        no_inventory_prompt = full_prompt.split("Placeholders:")[0].strip()
        cases.append(
            MetricGamingCase(
                id=f"{base_id}_inventory_off",
                slice=SLICE_INVENTORY_FREE,
                prompt=no_inventory_prompt,
                pred_openui=positive,
                request=_request(no_inventory_prompt, ()),
                expected_verdict=False,
                expected_reason_substrings=("required_inventory_unknown",),
                gold_openui=positive,
                transform="without_inventory",
                notes="authored text only; no explicit contract inventory",
            )
        )

        # Negative: prompt has a Placeholders section that omits one required slot.
        if len(slots) >= 2:
            partial_slots = slots[:-1]
            partial_prompt = (
                no_inventory_prompt
                + " Placeholders: "
                + " ".join(partial_slots)
            )
            cases.append(
                MetricGamingCase(
                    id=f"{base_id}_inventory_partial",
                    slice=SLICE_INVENTORY_FREE,
                    prompt=partial_prompt,
                    pred_openui=positive,
                    request=_request(partial_prompt, partial_slots),
                    expected_verdict=False,
                    expected_reason_substrings=("unexpected_placeholder_identity",),
                    gold_openui=positive,
                    transform="partial_inventory",
                    notes="explicit inventory present but incomplete",
                )
            )

        # Stronger negative: authored text names a slot, but no inventory.
        ambiguous_prompt = no_inventory_prompt + " Make sure to bind the requested slot."
        cases.append(
            MetricGamingCase(
                id=f"{base_id}_inventory_off_mentioned",
                slice=SLICE_INVENTORY_FREE,
                prompt=ambiguous_prompt,
                pred_openui=positive,
                request=_request(ambiguous_prompt, ()),
                expected_verdict=False,
                expected_reason_substrings=("required_inventory_unknown",),
                gold_openui=positive,
                transform="without_inventory_slot_mentioned",
                notes="slot mentioned in prose but not in contract inventory",
            )
        )
    return cases


def build_retry_sensitive_cases(seed: int = 0) -> list[MetricGamingCase]:
    """Cases where first-attempt and best-of-N metrics must differ."""
    del seed
    cases: list[MetricGamingCase] = []
    archetypes = _archetypes()
    partial_attempt = {
        "card": 'root = Card([TextContent(":card.title")])\n',
        "slider": 'root = Stack([TextContent(":settings.caption")])\n',
        "switch": 'root = Stack([TextContent(":settings.caption")])\n',
        "tabs": 'root = Tabs([TabItem("tab1", ":tab.trigger", [])])\n',
        "button": 'root = Stack([TextContent(":btn.action")])\n',
        "callout": 'root = Callout("info", ":callout.title", ":callout.title")\n',
        "image_block": 'root = Stack([TextContent(":img.alt")])\n',
    }
    empty_attempt = {
        "card": 'root = Card([])\n',
        "slider": 'root = Stack([])\n',
        "switch": 'root = Stack([])\n',
        "tabs": 'root = Tabs([])\n',
        "button": 'root = Stack([])\n',
        "callout": 'root = Stack([])\n',
        "image_block": 'root = Stack([])\n',
    }
    for attempt_count in (2, 3):
        for arch in archetypes:
            base_id = arch["id"]
            prompt = arch["prompt"]
            positive = arch["positive"]
            slots = arch["slot_contract"]
            # Build a list of attempts: first is minimal/empty, remainder improve.
            attempts: list[str] = []
            if attempt_count == 2:
                attempts = [
                    'root = Stack([])\n',
                    positive,
                ]
            else:
                attempts = [
                    'root = Stack([])\n',
                    partial_attempt[base_id],
                    positive,
                ]
            # Encode attempts as a JSON-serializable list in request runtime symbols
            # is awkward; instead store them in the gold_openui field as a list.
            cases.append(
                MetricGamingCase(
                    id=f"{base_id}_retry_{attempt_count}",
                    slice=SLICE_RETRY_SENSITIVE,
                    prompt=prompt,
                    pred_openui=positive,
                    request=_request(prompt, slots),
                    expected_verdict=True,
                    expected_reason_substrings=(),
                    gold_openui=json.dumps(attempts),
                    transform=f"retry_attempts_n={attempt_count}",
                    notes="attempt list stored in gold_openui as JSON array",
                )
            )

        # All-fail retry row: every attempt is invalid/minimal.
        for arch in archetypes:
            base_id = arch["id"]
            prompt = arch["prompt"]
            positive = arch["positive"]
            slots = arch["slot_contract"]
            all_fail_attempts = [
                'root = Stack([])\n',
                empty_attempt[base_id],
            ]
            cases.append(
                MetricGamingCase(
                    id=f"{base_id}_retry_all_fail",
                    slice=SLICE_RETRY_SENSITIVE,
                    prompt=prompt,
                    pred_openui=all_fail_attempts[-1],
                    request=_request(prompt, slots),
                    expected_verdict=False,
                    expected_reason_substrings=("required_placeholder_missing",),
                    gold_openui=json.dumps(all_fail_attempts),
                    transform="retry_all_fail",
                    notes="every attempt is hard-valid but semantically empty",
                )
            )
    return cases


def build_all_cases(seed: int = 0) -> list[MetricGamingCase]:
    """Return the full fixture-grade metric-gaming suite."""
    return (
        build_minimal_valid_trap_cases(seed)
        + build_rare_component_omission_cases(seed)
        + build_inventory_free_binding_cases(seed)
        + build_retry_sensitive_cases(seed)
    )


def _expected_failure(case: MetricGamingCase) -> bool:
    return not case.expected_verdict


def evaluate_metric_gaming(
    cases: Iterable[MetricGamingCase],
) -> MetricGamingReport:
    """Score a suite of metric-gaming cases and aggregate by slice."""
    case_list = list(cases)
    for case in case_list:
        _validate_case(case)

    scored = score_cases(case_list)
    slices: dict[str, MetricGamingSliceReport] = {
        name: MetricGamingSliceReport(slice=name) for name in ALL_SLICES
    }

    positives = 0
    covered_positives = 0
    covered = 0
    false_positives = 0
    false_negatives = 0
    all_reasons: Counter[str] = Counter()

    for sc in scored:
        report = sc.report
        case = sc.case
        slice_rep = slices[case.slice]
        slice_rep.n += 1
        if report.coverage_known:
            covered += 1
            if report.verdict:
                covered_positives += 1
        if report.verdict:
            positives += 1

        expected_failure = _expected_failure(case)
        if expected_failure:
            slice_rep.expected_failures += 1
            if not report.verdict:
                slice_rep.observed_failures += 1
            else:
                false_positives += 1
                slice_rep.false_positives += 1
        else:
            if not report.verdict:
                false_negatives += 1
                slice_rep.false_negatives += 1

        for reason in report.reason_codes:
            all_reasons[reason] += 1
            slice_rep.reason_prevalence[reason] = (
                slice_rep.reason_prevalence.get(reason, 0) + 1
            )

    n = len(case_list)
    for slice_rep in slices.values():
        if slice_rep.n:
            slice_rep.strict_rate = (
                slice_rep.n - slice_rep.false_positives - slice_rep.false_negatives
            ) / slice_rep.n
            covered_in_slice = sum(
                1
                for sc in scored
                if sc.case.slice == slice_rep.slice and sc.report.coverage_known
            )
            covered_pos_in_slice = sum(
                1
                for sc in scored
                if sc.case.slice == slice_rep.slice
                and sc.report.coverage_known
                and sc.report.verdict
            )
            slice_rep.coverage_conditioned_rate = (
                covered_pos_in_slice / covered_in_slice if covered_in_slice else 0.0
            )

    return MetricGamingReport(
        schema_version=SCHEMA_VERSION,
        metric_name=SemanticMeaningReportV2.metric_name,  # type: ignore[attr-defined]
        metric_version=SemanticMeaningReportV2.metric_version,  # type: ignore[attr-defined]
        n_cases=n,
        strict_rate=(n - false_positives - false_negatives) / n if n else 0.0,
        coverage_conditioned_rate=(covered_positives / covered if covered else 0.0),
        false_positive_count=false_positives,
        false_negative_count=false_negatives,
        slices=slices,
        cases=scored,
    )


def evaluate_retry_attempts(
    case: MetricGamingCase,
    attempts: list[str],
    selector: Callable[[list[ScoredCase]], int] | None = None,
) -> dict[str, Any]:
    """Score a retry-sensitive case across first, selected, and oracle attempts."""
    if case.slice != SLICE_RETRY_SENSITIVE:
        raise ValueError("evaluate_retry_attempts only accepts retry-sensitive cases")
    if not attempts:
        raise ValueError("attempts must be non-empty")

    scored_attempts: list[ScoredCase] = []
    for idx, attempt_openui in enumerate(attempts):
        attempt_case = MetricGamingCase(
            id=f"{case.id}_attempt_{idx}",
            slice=SLICE_RETRY_SENSITIVE,
            prompt=case.prompt,
            pred_openui=attempt_openui,
            request=case.request,
            expected_verdict=True,
            gold_openui=case.gold_openui,
            transform=f"attempt_{idx}",
        )
        scored_attempts.append(_score_case(attempt_case))

    if selector is None:
        # Default production-style selector: first passing attempt (greedy).
        selected_index = next(
            (i for i, sc in enumerate(scored_attempts) if sc.report.verdict),
            len(scored_attempts) - 1,
        )
    else:
        selected_index = selector(scored_attempts)

    oracle_pass = any(sc.report.verdict for sc in scored_attempts)
    first_pass = scored_attempts[0].report.verdict

    return {
        "case_id": case.id,
        "n_attempts": len(attempts),
        "first_attempt_pass": first_pass,
        "selected_attempt_index": selected_index,
        "selected_attempt_pass": scored_attempts[selected_index].report.verdict,
        "oracle_best_pass": oracle_pass,
        "attempt_reports": [sc.report.to_dict() for sc in scored_attempts],
    }


def evaluate_all_retry_cases(
    cases: Iterable[MetricGamingCase],
) -> list[dict[str, Any]]:
    """Evaluate every retry-sensitive case across attempts."""
    results: list[dict[str, Any]] = []
    for case in cases:
        if case.slice != SLICE_RETRY_SENSITIVE:
            continue
        if not case.gold_openui:
            continue
        attempts = json.loads(case.gold_openui)
        if not isinstance(attempts, list):
            continue
        results.append(evaluate_retry_attempts(case, attempts))
    return results


def write_manifest(
    report: MetricGamingReport,
    retry_results: list[dict[str, Any]],
    path: Path,
) -> None:
    """Write the machine-readable fixture manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metric_gaming_report": report.to_dict(),
        "retry_results": retry_results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "ALL_SLICES",
    "MetricGamingCase",
    "MetricGamingReport",
    "MetricGamingSliceReport",
    "ScoredCase",
    "SLICE_INVENTORY_FREE",
    "SLICE_MINIMAL_VALID",
    "SLICE_RARE_OMISSION",
    "SLICE_RETRY_SENSITIVE",
    "RARE_COMPONENTS",
    "SCHEMA_VERSION",
    "build_all_cases",
    "build_inventory_free_binding_cases",
    "build_minimal_valid_trap_cases",
    "build_rare_component_omission_cases",
    "build_retry_sensitive_cases",
    "evaluate_all_retry_cases",
    "evaluate_metric_gaming",
    "evaluate_retry_attempts",
    "score_cases",
    "write_manifest",
]
