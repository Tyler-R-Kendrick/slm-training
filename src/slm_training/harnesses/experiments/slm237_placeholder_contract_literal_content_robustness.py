"""SLM-237 (PCR0-01): placeholder-contract literal-content robustness probe.

``extract_placeholders`` in ``src/slm_training/dsl/placeholders.py`` scans an
OpenUI source string with a single regex
(``PLACEHOLDER_RE = re.compile(r":[A-Za-z_][A-Za-z0-9_]*(?:\\.[A-Za-z_]"
r"[A-Za-z0-9_]*)*")``) and returns every match, in source order, with no
awareness of *where* the match sits -- a real ``:placeholder`` slot reference
and a colon-prefixed identifier chain that merely happens to appear inside an
ordinary text-content string literal are indistinguishable to it. This
function is not diagnostic-only: it backs

* ``_placeholders_of`` / ``_contract_precision`` / ``_contract_recall`` /
  ``_placeholder_fidelity`` in
  ``src/slm_training/harnesses/model_build/eval_runner.py`` (the honest eval
  scoreboard's slot-contract metrics),
* ``score_openui`` in ``src/slm_training/integrations/openui_rl.py`` (the
  causal-LM RL reward contract's ``placeholder_fidelity`` term), and
* ``grammar_score`` in ``src/slm_training/harnesses/preference/__init__.py``
  (a *second, independent* call site: it returns ``0.0`` outright -- gating
  ``score_openui``'s ``parse`` term, not just fidelity -- whenever
  ``extract_placeholders(serialized)`` finds no placeholder-shaped token
  anywhere in the whole document, with the same raw-text-scan blindness).

This harness asks a narrow, falsifiable, CPU-only question with two parts:

1. **Precision contamination.** When a prediction correctly fills its one
   real contract placeholder but an unrelated leaf literal happens to contain
   a colon-prefixed dotted token that is *not* in the record's placeholder
   contract, does ``_contract_precision`` fall below 1.0 purely because of
   that incidental text, while ``_contract_recall`` / ``_placeholder_fidelity``
   (which only look at the gold set) stay at 1.0?
2. **False credit (the more severe direction).** When a prediction *omits*
   its one real contract placeholder entirely -- replacing it with hardcoded
   literal text, a genuine contract violation -- but an unrelated leaf
   literal happens to mention that exact placeholder token in prose, do
   ``_contract_precision`` / ``_contract_recall`` / ``_placeholder_fidelity``
   (and ``score_openui``'s ``placeholder_fidelity`` term) score the violated
   prediction as if the contract were perfectly satisfied (1.0 across the
   board), identically to a real, correctly-filled prediction, and strictly
   better than an otherwise-identical violation that omits the mention?

No new gate is implemented and no existing metric, default, or scoring
behavior is changed. This only exercises the real, unmodified
``extract_placeholders`` (dsl/placeholders.py), ``_placeholders_of`` /
``_contract_precision`` / ``_contract_recall`` / ``_placeholder_fidelity``
(eval_runner.py), and ``score_openui`` (openui_rl.py) functions against
synthetic, grammar-validated OpenUI document pairs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from slm_training.dsl.parser import validate
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.eval_runner import (
    _contract_precision,
    _contract_recall,
    _placeholder_fidelity,
    _placeholders_of,
)
from slm_training.integrations.openui_rl import score_openui
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "GOLD_PLACEHOLDER",
    "ContentVariant",
    "ContentRow",
    "RewardProbeRow",
    "PlaceholderContractRobustnessReport",
    "build_default_variants",
    "render_markdown",
    "run_robustness_fixture",
]

MATRIX_VERSION = "pcr0-01-v1"
MATRIX_SET = "slm237_placeholder_contract_literal_content_robustness"
EXPERIMENT_ID = "slm237-placeholder-contract-literal-content-robustness"

GOLD_PLACEHOLDER = ":hero.title"
NEUTRAL_CONTENT = "Contact our team today"
_TITLE_REPLACEMENT_TEXT = "Welcome to our site"
_DOWNSTREAM_REWARD_SHAPE = "single_leaf_card"

_HYPOTHESIS = (
    "extract_placeholders (dsl/placeholders.py) matches its placeholder "
    "regex against the raw source string with no awareness of string-"
    "literal boundaries, so a colon-prefixed dotted token inside an "
    "ordinary text-content literal is indistinguishable from a real "
    "``:placeholder`` slot reference. This has two consequences shared by "
    "eval_runner's _contract_precision / _contract_recall / "
    "_placeholder_fidelity and openui_rl's score_openui: (1) an unrelated "
    "placeholder-shaped mention in a free literal lowers _contract_"
    "precision even when the real contract is fully and correctly filled, "
    "while _contract_recall / _placeholder_fidelity stay at 1.0 because "
    "they only test the gold set; and (2) when a prediction *omits* its "
    "one real contract placeholder (a genuine violation) but a free "
    "literal happens to mention that exact placeholder token in prose, "
    "_contract_precision / _contract_recall / _placeholder_fidelity and "
    "score_openui's placeholder_fidelity term all score it as if the "
    "contract were perfectly satisfied -- identical to a correct "
    "prediction and strictly better than an otherwise-identical violation "
    "that does not happen to mention the token."
)

_FALSIFIER = (
    "Either: an unrelated placeholder-shaped literal mention leaves "
    "_contract_precision at 1.0 when the real contract is otherwise fully "
    "and correctly filled; or a prediction that omits its one real "
    "contract placeholder scores below 1.0 on _contract_precision / "
    "_contract_recall / _placeholder_fidelity / score_openui's "
    "placeholder_fidelity term even when a free literal mentions the "
    "exact placeholder token in prose (i.e. the omission is always "
    "correctly penalized regardless of incidental text)."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no checkpoint, GPU run, RL training "
    "step, or ship-gate claim is made or implied.",
    "All documents are synthetic but real, grammar-valid OpenUI (each is "
    "round-tripped through the real slm_training.dsl.parser.validate before "
    "being scored) -- this is not a reimplementation of "
    "extract_placeholders, _contract_precision, _contract_recall, "
    "_placeholder_fidelity, or score_openui, only real calls to them.",
    "The variant set (5 literal-content edits x 3 structural shapes) is "
    "small and hand-authored; it demonstrates the mechanism exists and is "
    "reproducible, not its exact prevalence across a real corpus of "
    "generated predictions.",
    "The downstream reward probe uses a single-placeholder slot_inventory. "
    "Composite-reward divergence between the mentioned/unmentioned "
    "contract-violation rows is NOT isolated to the placeholder_fidelity "
    "term alone: grammar_score (harnesses/preference/__init__.py) also "
    "calls extract_placeholders directly and returns 0.0 (failing the "
    "'parse' term) whenever the whole serialized document has no "
    "placeholder-shaped token anywhere, so the reward-probe rows show both "
    "the parse term and the placeholder_fidelity term flipping together -- "
    "a second, independent instance of the same raw-text-scan mechanism, "
    "not a clean single-term isolation. It does not characterize a full RL "
    "training run.",
    "This harness does not change extract_placeholders, _contract_"
    "precision, _contract_recall, _placeholder_fidelity, score_openui, or "
    "any eval/RL default. It documents a concrete scoring-mechanics gap as "
    "a candidate for a future, separately reviewed hardening change (e.g. "
    "scoring only parsed AST placeholder-slot nodes instead of raw text) "
    "-- never implemented here.",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: Any) -> str:
    return _sha256(_canonical_json(value))


def _shape_single_leaf_card(title_arg: str, body_text: str) -> str:
    body_literal = json.dumps(body_text)
    return (
        'root = Stack([hero], "column")\n'
        f"hero_title = TextContent({title_arg})\n"
        f"hero_body = TextContent({body_literal})\n"
        "hero = Card([hero_title, hero_body])"
    )


def _shape_nested_two_leaf_card(title_arg: str, body_text: str) -> str:
    body_literal = json.dumps(body_text)
    return (
        'root = Stack([card], "column")\n'
        f"note = TextContent({body_literal})\n"
        f"label = TextContent({title_arg})\n"
        "card = Card([label, note])"
    )


def _shape_button_row(title_arg: str, body_text: str) -> str:
    body_literal = json.dumps(body_text)
    return (
        'root = Stack([button, helper], "row")\n'
        f"button = Button({title_arg})\n"
        f"helper = TextContent({body_literal})\n"
    )


_SHAPES: dict[str, Callable[[str, str], str]] = {
    "single_leaf_card": _shape_single_leaf_card,
    "nested_two_leaf_card": _shape_nested_two_leaf_card,
    "button_row": _shape_button_row,
}

_SHAPE_DESCRIPTIONS: dict[str, str] = {
    "single_leaf_card": "Card([TextContent(title), TextContent(literal body)]) inside a Stack; title carries the real contract placeholder.",
    "nested_two_leaf_card": "Card([TextContent(title), TextContent(literal body)]) with the literal body leaf declared first.",
    "button_row": "Button(title) plus a sibling TextContent(literal helper) inside a row Stack; title carries the real contract placeholder.",
}


@dataclass(frozen=True)
class ContentVariant:
    """One (title, body-literal) edit applied to a shape's two free leaves."""

    name: str
    category: str
    # baseline | content_only | spurious_unrelated | contract_violation_mentioned | contract_violation_unmentioned
    title_is_real_placeholder: bool
    title_replacement_text: str
    body_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "title_is_real_placeholder": self.title_is_real_placeholder,
            "title_replacement_text": self.title_replacement_text,
            "body_text": self.body_text,
        }


def build_default_variants() -> list[ContentVariant]:
    """Five hand-authored (title, body) edits spanning five categories.

    ``baseline`` fills the real placeholder and uses neutral body content (a
    sanity control). ``content_only`` still fills the real placeholder but
    changes the body's wording with no placeholder-shaped substrings at all
    -- the expected-robust case. ``spurious_unrelated`` still fills the real
    placeholder but adds an unrelated placeholder-shaped mention in the body
    -- isolates the precision-contamination direction.
    ``contract_violation_mentioned`` *omits* the real placeholder (hardcodes
    the title) but the body happens to mention that exact placeholder token
    in prose -- the false-credit direction.
    ``contract_violation_unmentioned`` is the matched control: identical
    omission, neutral body with no mention -- expected to be correctly
    penalized.
    """
    return [
        ContentVariant("neutral", "baseline", True, "", NEUTRAL_CONTENT),
        ContentVariant(
            "plain_alt_wording", "content_only", True, "", "Reach out to support now"
        ),
        ContentVariant(
            "spurious_unrelated",
            "spurious_unrelated",
            True,
            "",
            "Email us at :support.email for help",
        ),
        ContentVariant(
            "violation_mentioned",
            "contract_violation_mentioned",
            False,
            _TITLE_REPLACEMENT_TEXT,
            f"See {GOLD_PLACEHOLDER} for details",
        ),
        ContentVariant(
            "violation_unmentioned",
            "contract_violation_unmentioned",
            False,
            _TITLE_REPLACEMENT_TEXT,
            "Reach out to support now",
        ),
    ]


@dataclass(frozen=True)
class ContentRow:
    """The real placeholder-contract-metric outcome for one (shape, variant) pair."""

    shape: str
    variant: str
    category: str
    contract_violated: bool
    body_text: str
    pred_placeholder_tokens: tuple[str, ...]
    contract_precision: float | None
    contract_recall: float | None
    placeholder_fidelity: float | None
    false_credit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "variant": self.variant,
            "category": self.category,
            "contract_violated": self.contract_violated,
            "body_text": self.body_text,
            "pred_placeholder_tokens": list(self.pred_placeholder_tokens),
            "contract_precision": self.contract_precision,
            "contract_recall": self.contract_recall,
            "placeholder_fidelity": self.placeholder_fidelity,
            "false_credit": self.false_credit,
        }


@dataclass(frozen=True)
class RewardProbeRow:
    """One real score_openui() outcome for the downstream RL-reward probe."""

    variant: str
    category: str
    contract_violated: bool
    composite: float
    placeholder_fidelity: float
    structural_similarity: float
    parse: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "category": self.category,
            "contract_violated": self.contract_violated,
            "composite": self.composite,
            "placeholder_fidelity": self.placeholder_fidelity,
            "structural_similarity": self.structural_similarity,
            "parse": self.parse,
        }


def _title_arg(variant: ContentVariant) -> str:
    text = GOLD_PLACEHOLDER if variant.title_is_real_placeholder else variant.title_replacement_text
    return json.dumps(text)


def _row_for(shape: str, variant: ContentVariant, gold_record: ExampleRecord) -> ContentRow:
    shape_fn = _SHAPES[shape]
    pred = shape_fn(_title_arg(variant), variant.body_text)
    # Fail closed: the prediction must be real, grammar-valid OpenUI.
    validate(pred)
    pred_placeholders = tuple(sorted(_placeholders_of(pred)))
    precision = _contract_precision(pred, gold_record)
    recall = _contract_recall(pred, gold_record)
    fidelity = _placeholder_fidelity(pred, gold_record)
    contract_violated = not variant.title_is_real_placeholder
    false_credit = bool(
        contract_violated
        and precision == 1.0
        and recall == 1.0
        and fidelity == 1.0
    )
    return ContentRow(
        shape=shape,
        variant=variant.name,
        category=variant.category,
        contract_violated=contract_violated,
        body_text=variant.body_text,
        pred_placeholder_tokens=pred_placeholders,
        contract_precision=precision,
        contract_recall=recall,
        placeholder_fidelity=fidelity,
        false_credit=false_credit,
    )


def _reward_probe_rows(variants: list[ContentVariant], gold_source: str) -> list[RewardProbeRow]:
    shape_fn = _SHAPES[_DOWNSTREAM_REWARD_SHAPE]
    rows: list[RewardProbeRow] = []
    for variant in variants:
        pred = shape_fn(_title_arg(variant), variant.body_text)
        validate(pred)
        reward = score_openui(pred, gold_openui=gold_source, slot_inventory=[GOLD_PLACEHOLDER])
        rows.append(
            RewardProbeRow(
                variant=variant.name,
                category=variant.category,
                contract_violated=not variant.title_is_real_placeholder,
                composite=reward.composite,
                placeholder_fidelity=reward.placeholder_fidelity,
                structural_similarity=reward.structural_similarity,
                parse=reward.parse,
            )
        )
    return rows


def _resolve_disposition(rows: list[ContentRow]) -> tuple[str, str]:
    baselines = [r for r in rows if r.category == "baseline"]
    controls_ok = bool(baselines) and all(
        (not r.contract_violated)
        and r.contract_precision == 1.0
        and r.contract_recall == 1.0
        and r.placeholder_fidelity == 1.0
        for r in baselines
    )
    if not controls_ok:
        return (
            "inconclusive",
            "At least one baseline (real placeholder filled, neutral body) "
            "row did not score exactly 1.0 on all three contract metrics; "
            "the fixture does not isolate the literal-content-robustness "
            "question cleanly.",
        )

    content_only = [r for r in rows if r.category == "content_only"]
    spurious = [r for r in rows if r.category == "spurious_unrelated"]
    mentioned = [r for r in rows if r.category == "contract_violation_mentioned"]
    unmentioned = [r for r in rows if r.category == "contract_violation_unmentioned"]

    content_only_robust = bool(content_only) and all(
        r.contract_precision == 1.0 and r.contract_recall == 1.0 and r.placeholder_fidelity == 1.0
        for r in content_only
    )
    precision_contaminated = bool(spurious) and all(
        r.contract_precision is not None
        and r.contract_precision < 1.0
        and r.contract_recall == 1.0
        and r.placeholder_fidelity == 1.0
        for r in spurious
    )
    false_credit_confirmed = bool(mentioned) and all(r.false_credit for r in mentioned)
    control_correctly_penalized = bool(unmentioned) and all(
        (not r.false_credit)
        and r.contract_precision == 0.0
        and r.contract_recall == 0.0
        and r.placeholder_fidelity == 0.0
        for r in unmentioned
    )

    if (
        content_only_robust
        and precision_contaminated
        and false_credit_confirmed
        and control_correctly_penalized
    ):
        n_shapes = len({r.shape for r in rows})
        return (
            "gap_confirmed",
            f"Across all {n_shapes} structural shapes: content-only body "
            "wording edits left all three contract metrics at 1.0 (robust); "
            "an unrelated placeholder-shaped body mention lowered "
            "_contract_precision below 1.0 while _contract_recall and "
            "_placeholder_fidelity stayed at 1.0 (asymmetric precision "
            "contamination); and a genuine contract violation (real "
            "placeholder omitted) scored a perfect 1.0 on all three metrics "
            "whenever the body happened to mention the placeholder token in "
            "prose, versus a correctly-penalized 0.0 on an otherwise-"
            "identical violation with no mention -- confirming "
            "extract_placeholders is not literal-content-invariant and can "
            "award full false credit for an unfilled contract slot.",
        )

    n_false_credit = sum(1 for r in mentioned if r.false_credit)
    return (
        "partial_confirmation",
        f"content_only_robust={content_only_robust}, "
        f"precision_contaminated={precision_contaminated}, "
        f"false_credit_confirmed={false_credit_confirmed} "
        f"({n_false_credit}/{len(mentioned)} mentioned-violation rows), "
        f"control_correctly_penalized={control_correctly_penalized} -- "
        "not every sub-claim held across all rows.",
    )


@dataclass(frozen=True)
class PlaceholderContractRobustnessReport:
    """Full fixture report for SLM-237."""

    schema: str = "PlaceholderContractRobustnessReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm237-placeholder-contract-literal-content-robustness"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    rows: tuple[ContentRow, ...] = field(default_factory=tuple)
    reward_probe_rows: tuple[RewardProbeRow, ...] = field(default_factory=tuple)
    reward_probe_shape: str = _DOWNSTREAM_REWARD_SHAPE
    gate_hash: str = ""
    disposition: str = "inconclusive"
    disposition_rationale: str = ""
    honest_caveats: tuple[str, ...] = _HONEST_CAVEATS
    version_stamp: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "rows": [r.to_dict() for r in self.rows],
            "reward_probe_rows": [r.to_dict() for r in self.reward_probe_rows],
            "reward_probe_shape": self.reward_probe_shape,
            "gate_hash": self.gate_hash,
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "honest_caveats": list(self.honest_caveats),
            "version_stamp": dict(self.version_stamp),
            "timestamp": self.timestamp,
        }

    def to_json(self, path) -> None:  # pragma: no cover - thin IO helper
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlaceholderContractRobustnessReport":
        rows = tuple(
            ContentRow(
                shape=str(r["shape"]),
                variant=str(r["variant"]),
                category=str(r["category"]),
                contract_violated=bool(r["contract_violated"]),
                body_text=str(r["body_text"]),
                pred_placeholder_tokens=tuple(str(t) for t in r.get("pred_placeholder_tokens", ())),
                contract_precision=(
                    None if r.get("contract_precision") is None else float(r["contract_precision"])
                ),
                contract_recall=(
                    None if r.get("contract_recall") is None else float(r["contract_recall"])
                ),
                placeholder_fidelity=(
                    None if r.get("placeholder_fidelity") is None else float(r["placeholder_fidelity"])
                ),
                false_credit=bool(r["false_credit"]),
            )
            for r in data.get("rows", ())
        )
        reward_rows = tuple(
            RewardProbeRow(
                variant=str(r["variant"]),
                category=str(r["category"]),
                contract_violated=bool(r["contract_violated"]),
                composite=float(r["composite"]),
                placeholder_fidelity=float(r["placeholder_fidelity"]),
                structural_similarity=float(r["structural_similarity"]),
                parse=float(r["parse"]),
            )
            for r in data.get("reward_probe_rows", ())
        )
        return cls(
            schema=str(data.get("schema", "PlaceholderContractRobustnessReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", EXPERIMENT_ID)),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            rows=rows,
            reward_probe_rows=reward_rows,
            reward_probe_shape=str(data.get("reward_probe_shape", _DOWNSTREAM_REWARD_SHAPE)),
            gate_hash=str(data.get("gate_hash", "")),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def run_robustness_fixture(
    *,
    variants: list[ContentVariant] | None = None,
    run_id: str | None = None,
) -> PlaceholderContractRobustnessReport:
    """Score every (shape, variant) pair through the real extract_placeholders
    / _contract_precision / _contract_recall / _placeholder_fidelity /
    score_openui functions and resolve a disposition."""
    variants = variants if variants is not None else build_default_variants()

    rows: list[ContentRow] = []
    for shape in _SHAPES:
        gold_source = _SHAPES[shape](json.dumps(GOLD_PLACEHOLDER), NEUTRAL_CONTENT)
        validate(gold_source)
        gold_record = ExampleRecord(
            id=f"{EXPERIMENT_ID}-{shape}-gold",
            prompt="placeholder-contract literal-content robustness fixture",
            openui=gold_source,
            placeholders=[GOLD_PLACEHOLDER],
        )
        for variant in variants:
            rows.append(_row_for(shape, variant, gold_record))

    reward_gold_source = _SHAPES[_DOWNSTREAM_REWARD_SHAPE](
        json.dumps(GOLD_PLACEHOLDER), NEUTRAL_CONTENT
    )
    validate(reward_gold_source)
    reward_rows = _reward_probe_rows(variants, reward_gold_source)

    disposition, rationale = _resolve_disposition(rows)

    payload = {
        "row_digests": sorted(_digest(r.to_dict()) for r in rows),
        "reward_probe_digests": sorted(_digest(r.to_dict()) for r in reward_rows),
    }
    gate_hash = _sha256(_canonical_json(payload))

    return PlaceholderContractRobustnessReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        rows=tuple(rows),
        reward_probe_rows=tuple(reward_rows),
        gate_hash=gate_hash,
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm237_placeholder_contract_literal_content_robustness",
        ),
    )


def render_markdown(report: PlaceholderContractRobustnessReport) -> str:
    lines = [
        f"# SLM-237 (PCR0-01): placeholder-contract literal-content robustness probe ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
        f"**Gate hash:** `{report.gate_hash[:16]}...`",
        f"**Disposition:** {report.disposition} — {report.disposition_rationale}",
        "",
        "## Hypothesis",
        "",
        report.hypothesis,
        "",
        "## Falsifier",
        "",
        report.falsifier,
        "",
        "## Honest caveats",
        "",
        *(f"- {c}" for c in report.honest_caveats),
        "",
        "## Per-row results (contract metrics)",
        "",
        "| shape | variant | category | violated | pred placeholders | precision | recall | fidelity | false credit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.rows:
        tokens = ", ".join(r.pred_placeholder_tokens) or "—"
        prec = "—" if r.contract_precision is None else f"{r.contract_precision:.4f}"
        rec = "—" if r.contract_recall is None else f"{r.contract_recall:.4f}"
        fid = "—" if r.placeholder_fidelity is None else f"{r.placeholder_fidelity:.4f}"
        lines.append(
            f"| {r.shape} | {r.variant} | {r.category} | {r.contract_violated} | {tokens} | "
            f"{prec} | {rec} | {fid} | {r.false_credit} |"
        )
    lines += [
        "",
        f"## Downstream RL reward probe (`score_openui`, shape=`{report.reward_probe_shape}`)",
        "",
        "| variant | category | violated | composite | placeholder_fidelity | structural_similarity | parse |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.reward_probe_rows:
        lines.append(
            f"| {r.variant} | {r.category} | {r.contract_violated} | {r.composite:.4f} | "
            f"{r.placeholder_fidelity:.4f} | {r.structural_similarity:.4f} | {r.parse:.4f} |"
        )
    lines += [
        "",
        "## No-go for promotion",
        "",
        "This report is wiring/fixture evidence only. It does not change "
        "`extract_placeholders`, `_contract_precision`, `_contract_recall`, "
        "`_placeholder_fidelity`, `score_openui`, or any eval/RL default, "
        "does not train a model, and makes no ship or gate claim. It "
        "documents a concrete scoring-mechanics gap in the literal-content "
        "robustness of the placeholder-contract metric family shared by the "
        "eval scoreboard and the RL reward contract, as a candidate for a "
        "future, separately reviewed hardening change (never implemented "
        "here).",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python -m scripts.run_slm237_placeholder_contract_literal_content_robustness --mode plan-only",
        "python -m scripts.run_slm237_placeholder_contract_literal_content_robustness --mode fixture",
        "```",
        "",
    ]
    return "\n".join(lines)
