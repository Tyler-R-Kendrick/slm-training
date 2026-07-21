"""SLM-236 (SSR0-01): structural-similarity literal-content robustness probe.

``structural_similarity`` in
``src/slm_training/harnesses/model_build/eval_runner.py`` is documented as a
"Jaccard-like similarity over component multisets + depth (style-agnostic)"
metric. It is not a diagnostic-only value: it is read directly into

* the honest eval scoreboard (``evaluate()`` in the same module, feeding
  ``struct_vals`` / ``tree_edit_vals`` / topology evidence),
* the causal-LM RL reward contract (``score_openui`` in
  ``src/slm_training/integrations/openui_rl.py``, a term of ``composite``
  reward alongside ``parse`` and ``placeholder_fidelity``), and
* preference counterfactual re-scoring
  (``harnesses/preference/counterfactuals.py``).

Its implementation has two raw-text proxies:

1. A component-occurrence multiset built with a regex,
   ``_COMPONENT_RE = re.compile(r"\\b([A-Z][A-Za-z0-9]*)\\s*\\(")``, applied to
   the *entire* source string (after only ``strip_style_literals``, which
   strips style-prop literals, not ordinary text-content literals).
2. A "depth" proxy computed as raw ``"["``/``"("`` character counts over the
   *entire* source string.

Neither proxy is literal-content-aware: both scan the full source text,
including the contents of ordinary string literals (e.g. a ``TextContent``
or ``Button`` label). This harness asks a narrow, falsifiable, CPU-only
question: **for two OpenUI documents with byte-identical DSL structure that
differ only in the literal text of one leaf string argument, does the real,
unmodified ``structural_similarity`` return 1.0 (as a "style-agnostic
structural" metric implies it must), or can ordinary literal content --
parentheses, brackets, or a capitalized-word-immediately-followed-by-"("
substring that coincidentally looks like a component call -- silently lower
the score despite the structure being identical?**

No new gate is implemented and no existing metric, default, or scoring
behavior is changed. This only exercises the real, unmodified
``structural_similarity`` / ``_component_multiset`` (eval_runner) and
``score_openui`` (openui_rl RL-reward contract) functions against synthetic,
grammar-validated OpenUI document pairs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from slm_training.dsl.parser import validate
from slm_training.harnesses.model_build.eval_runner import (
    _component_multiset,
    structural_similarity,
)
from slm_training.integrations.openui_rl import score_openui
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "ContentVariant",
    "ContentRow",
    "RewardProbeRow",
    "StructuralSimilarityRobustnessReport",
    "build_default_variants",
    "render_markdown",
    "run_robustness_fixture",
]

MATRIX_VERSION = "ssr0-01-v1"
MATRIX_SET = "slm236_structural_similarity_literal_content_robustness"
EXPERIMENT_ID = "slm236-structural-similarity-literal-content-robustness"

NEUTRAL_CONTENT = "Contact our team today"
_DOWNSTREAM_REWARD_SHAPE = "single_leaf_card"

_HYPOTHESIS = (
    "structural_similarity (eval_runner.py) computes its component-multiset "
    "and bracket/paren \"depth\" proxies over the raw source string, "
    "including the contents of ordinary text-content string literals (only "
    "strip_style_literals is applied first, which strips style-prop "
    "literals, not text-content literals). Because of this, two OpenUI "
    "documents with byte-identical DSL structure that differ only in the "
    "literal text of one leaf string argument can receive different "
    "structural_similarity scores: ordinary punctuation (parentheses, "
    "square brackets) in the literal text perturbs the depth proxy, and a "
    "literal substring shaped like \"Word(\" is misread by the component "
    "regex as an extra component occurrence, both independent of any real "
    "structural difference. This propagates into the RL reward contract "
    "(score_openui's composite term) without a corresponding change in "
    "parse validity or placeholder fidelity."
)

_FALSIFIER = (
    "For every same-shape, content-only-edited pair (identical DSL "
    "structure, one leaf literal changed), structural_similarity returns "
    "1.0 regardless of the literal text's punctuation or wording; or the "
    "component regex/depth proxies are shown not to read literal-string "
    "content in the real implementation."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no checkpoint, GPU run, RL training "
    "step, or ship-gate claim is made or implied.",
    "All documents are synthetic but real, grammar-valid OpenUI (each is "
    "round-tripped through the real slm_training.dsl.parser.validate before "
    "being scored) -- this is not a reimplementation of "
    "structural_similarity, _component_multiset, or score_openui, only real "
    "calls to them.",
    "The variant set (7 literal-content edits x 3 structural shapes) is "
    "small and hand-authored; it demonstrates the mechanism exists and is "
    "reproducible, not its exact prevalence across a real corpus of "
    "generated predictions.",
    "The downstream reward probe uses an empty slot_inventory so the "
    "placeholder_fidelity term of score_openui stays constant across "
    "variants -- this isolates the structural_similarity term's "
    "contribution to composite reward, it does not characterize a full "
    "RL training run.",
    "This harness does not change structural_similarity, "
    "_component_multiset, score_openui, or any eval/RL default. It "
    "documents a concrete scoring-mechanics gap as a candidate for a "
    "future, separately reviewed hardening change (e.g. scoring only "
    "parsed AST component/call nodes instead of raw text) -- never "
    "implemented here.",
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


def _shape_single_leaf_card(content: str) -> str:
    literal = json.dumps(content)
    return (
        'root = Stack([hero], "column")\n'
        'hero_title = TextContent(":hero.title")\n'
        f"hero_body = TextContent({literal})\n"
        "hero = Card([hero_title, hero_body])"
    )


def _shape_nested_two_leaf_card(content: str) -> str:
    literal = json.dumps(content)
    return (
        'root = Stack([card], "column")\n'
        f"note = TextContent({literal})\n"
        'label = TextContent(":card.label")\n'
        "card = Card([label, note])"
    )


def _shape_button_row(content: str) -> str:
    literal = json.dumps(content)
    return f'root = Stack([button], "row")\nbutton = Button({literal})\n'


_SHAPES: dict[str, Callable[[str], str]] = {
    "single_leaf_card": _shape_single_leaf_card,
    "nested_two_leaf_card": _shape_nested_two_leaf_card,
    "button_row": _shape_button_row,
}

_SHAPE_DESCRIPTIONS: dict[str, str] = {
    "single_leaf_card": "Card([TextContent(placeholder), TextContent(literal)]) inside a Stack.",
    "nested_two_leaf_card": "Card([TextContent(placeholder), TextContent(literal)]) with the literal leaf declared first.",
    "button_row": "A single Button(literal label) inside a row Stack (fewest structural parens/brackets of the three shapes).",
}


@dataclass(frozen=True)
class ContentVariant:
    """One literal-content edit applied to a shape's single free leaf."""

    name: str
    text: str
    category: str  # baseline | content_only | benign_punctuation | adversarial_regex | adversarial_mixed

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "text": self.text, "category": self.category}


def build_default_variants() -> list[ContentVariant]:
    """Seven hand-authored literal-content edits spanning four categories.

    ``baseline`` is the neutral content used to build each shape's "gold"
    document (pred == gold trivially; a sanity control). ``content_only``
    changes wording with no bracket/paren-shaped characters at all -- the
    expected-robust case. ``benign_punctuation`` adds ordinary parentheses or
    square brackets that do not resemble a component call. ``adversarial_regex``
    and ``adversarial_mixed`` add a capitalized-word-immediately-followed-by-"("
    substring that the component regex can misread as a real component call.
    """
    return [
        ContentVariant("neutral", NEUTRAL_CONTENT, "baseline"),
        ContentVariant("plain_alt_wording", "Reach out to support now", "content_only"),
        ContentVariant("parens_benign", "Available now (limited time only)", "benign_punctuation"),
        ContentVariant("brackets_benign", "See the [Support] section for help", "benign_punctuation"),
        ContentVariant("fake_component_single", "See Details(more info) for help", "adversarial_regex"),
        ContentVariant("fake_component_multi", "Ping Support(now) or Email(us) anytime", "adversarial_regex"),
        ContentVariant("mixed_adversarial", "Read Policy(v2) [ref 12] before (continuing)", "adversarial_mixed"),
    ]


@dataclass(frozen=True)
class ContentRow:
    """The real structural_similarity outcome for one (shape, variant) pair."""

    shape: str
    variant: str
    category: str
    variant_text: str
    is_baseline: bool
    structural_similarity: float
    spurious_component_keys: tuple[str, ...]
    divergent: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "variant": self.variant,
            "category": self.category,
            "variant_text": self.variant_text,
            "is_baseline": self.is_baseline,
            "structural_similarity": self.structural_similarity,
            "spurious_component_keys": list(self.spurious_component_keys),
            "divergent": self.divergent,
        }


@dataclass(frozen=True)
class RewardProbeRow:
    """One real score_openui() outcome for the downstream RL-reward probe."""

    variant: str
    category: str
    composite: float
    structural_similarity: float
    parse: float
    placeholder_fidelity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "category": self.category,
            "composite": self.composite,
            "structural_similarity": self.structural_similarity,
            "parse": self.parse,
            "placeholder_fidelity": self.placeholder_fidelity,
        }


def _row_for(shape: str, variant: ContentVariant, gold: str) -> ContentRow:
    pred = _SHAPES[shape](variant.text)
    # Fail closed: both sides must be real, grammar-valid OpenUI.
    validate(pred)
    score = structural_similarity(pred, gold)
    pred_c = _component_multiset(pred)
    gold_c = _component_multiset(gold)
    spurious = tuple(sorted(set(pred_c) - set(gold_c)))
    return ContentRow(
        shape=shape,
        variant=variant.name,
        category=variant.category,
        variant_text=variant.text,
        is_baseline=variant.category == "baseline",
        structural_similarity=score,
        spurious_component_keys=spurious,
        divergent=score < 1.0,
    )


def _reward_probe_rows(variants: list[ContentVariant]) -> list[RewardProbeRow]:
    shape_fn = _SHAPES[_DOWNSTREAM_REWARD_SHAPE]
    gold = shape_fn(NEUTRAL_CONTENT)
    validate(gold)
    rows: list[RewardProbeRow] = []
    for variant in variants:
        pred = shape_fn(variant.text)
        validate(pred)
        reward = score_openui(pred, gold_openui=gold, slot_inventory=[])
        rows.append(
            RewardProbeRow(
                variant=variant.name,
                category=variant.category,
                composite=reward.composite,
                structural_similarity=reward.structural_similarity,
                parse=reward.parse,
                placeholder_fidelity=reward.placeholder_fidelity,
            )
        )
    return rows


def _resolve_disposition(rows: list[ContentRow]) -> tuple[str, str]:
    baselines = [r for r in rows if r.is_baseline]
    controls_ok = bool(baselines) and all(
        (not r.divergent) and r.structural_similarity == 1.0 for r in baselines
    )
    if not controls_ok:
        return (
            "inconclusive",
            "At least one baseline (pred == gold, identical content) row did "
            "not score exactly 1.0; the fixture does not isolate the "
            "literal-content-robustness question cleanly.",
        )

    non_baseline = [r for r in rows if not r.is_baseline]
    content_only = [r for r in non_baseline if r.category == "content_only"]
    perturbing = [r for r in non_baseline if r.category != "content_only"]

    content_only_robust = bool(content_only) and all(not r.divergent for r in content_only)
    perturbing_all_divergent = bool(perturbing) and all(r.divergent for r in perturbing)
    perturbing_none_divergent = bool(perturbing) and all(not r.divergent for r in perturbing)

    if content_only_robust and perturbing_all_divergent:
        n_shapes = len({r.shape for r in rows})
        return (
            "gap_confirmed",
            f"Across all {n_shapes} structural shapes, every content-only "
            f"wording edit ({len(content_only)}/{len(content_only)} rows) "
            "left structural_similarity at exactly 1.0, while every literal "
            f"edit that added ordinary punctuation or a component-shaped "
            f"substring ({len(perturbing)}/{len(perturbing)} rows) lowered "
            "the score below 1.0 despite identical DSL structure -- "
            "confirming structural_similarity is not literal-content-"
            "invariant, and that this propagates into score_openui's "
            "composite RL reward term with parse validity and placeholder "
            "fidelity held constant.",
        )
    if content_only_robust and perturbing_none_divergent:
        return (
            "no_gap_found",
            "No literal-content edit -- benign punctuation or a "
            "component-shaped substring -- lowered structural_similarity "
            "below 1.0 for a content-only structural edit; the hypothesized "
            "mechanism gap does not hold as stated.",
        )
    n_divergent = sum(1 for r in perturbing if r.divergent)
    return (
        "partial_confirmation",
        f"Divergence was observed in {n_divergent}/{len(perturbing)} "
        "non-content-only rows and content-only wording edits were "
        f"{'robust' if content_only_robust else 'not robust'} -- "
        "inconsistent across categories rather than a clean universal gap.",
    )


@dataclass(frozen=True)
class StructuralSimilarityRobustnessReport:
    """Full fixture report for SLM-236."""

    schema: str = "StructuralSimilarityRobustnessReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm236-structural-similarity-literal-content-robustness"
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
    def from_dict(cls, data: dict[str, Any]) -> "StructuralSimilarityRobustnessReport":
        rows = tuple(
            ContentRow(
                shape=str(r["shape"]),
                variant=str(r["variant"]),
                category=str(r["category"]),
                variant_text=str(r["variant_text"]),
                is_baseline=bool(r["is_baseline"]),
                structural_similarity=float(r["structural_similarity"]),
                spurious_component_keys=tuple(str(k) for k in r.get("spurious_component_keys", ())),
                divergent=bool(r["divergent"]),
            )
            for r in data.get("rows", ())
        )
        reward_rows = tuple(
            RewardProbeRow(
                variant=str(r["variant"]),
                category=str(r["category"]),
                composite=float(r["composite"]),
                structural_similarity=float(r["structural_similarity"]),
                parse=float(r["parse"]),
                placeholder_fidelity=float(r["placeholder_fidelity"]),
            )
            for r in data.get("reward_probe_rows", ())
        )
        return cls(
            schema=str(data.get("schema", "StructuralSimilarityRobustnessReportV1")),
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
) -> StructuralSimilarityRobustnessReport:
    """Score every (shape, variant) pair through the real structural_similarity
    / _component_multiset / score_openui functions and resolve a disposition."""
    variants = variants if variants is not None else build_default_variants()

    rows: list[ContentRow] = []
    for shape in _SHAPES:
        gold = _SHAPES[shape](NEUTRAL_CONTENT)
        validate(gold)
        for variant in variants:
            rows.append(_row_for(shape, variant, gold))

    reward_rows = _reward_probe_rows(variants)

    disposition, rationale = _resolve_disposition(rows)

    payload = {
        "row_digests": sorted(_digest(r.to_dict()) for r in rows),
        "reward_probe_digests": sorted(_digest(r.to_dict()) for r in reward_rows),
    }
    gate_hash = _sha256(_canonical_json(payload))

    return StructuralSimilarityRobustnessReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        rows=tuple(rows),
        reward_probe_rows=tuple(reward_rows),
        gate_hash=gate_hash,
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm236_structural_similarity_literal_content_robustness",
        ),
    )


def render_markdown(report: StructuralSimilarityRobustnessReport) -> str:
    lines = [
        f"# SLM-236 (SSR0-01): structural-similarity literal-content robustness probe ({report.run_id})",
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
        "## Per-row results (structural_similarity)",
        "",
        "| shape | variant | category | score | divergent | spurious components |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.rows:
        spurious = ", ".join(r.spurious_component_keys) or "—"
        lines.append(
            f"| {r.shape} | {r.variant} | {r.category} | {r.structural_similarity:.4f} | "
            f"{r.divergent} | {spurious} |"
        )
    lines += [
        "",
        f"## Downstream RL reward probe (`score_openui`, shape=`{report.reward_probe_shape}`)",
        "",
        "| variant | category | composite | structural_similarity | parse | placeholder_fidelity |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.reward_probe_rows:
        lines.append(
            f"| {r.variant} | {r.category} | {r.composite:.4f} | {r.structural_similarity:.4f} | "
            f"{r.parse:.4f} | {r.placeholder_fidelity:.4f} |"
        )
    lines += [
        "",
        "## No-go for promotion",
        "",
        "This report is wiring/fixture evidence only. It does not change "
        "`structural_similarity`, `_component_multiset`, `score_openui`, or "
        "any eval/RL default, does not train a model, and makes no ship or "
        "gate claim. It documents a concrete scoring-mechanics gap in the "
        "literal-content robustness of a structural metric shared by the "
        "eval scoreboard, RL reward contract, and preference counterfactual "
        "re-scoring, as a candidate for a future, separately reviewed "
        "hardening change (never implemented here).",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python -m scripts.run_slm236_structural_similarity_literal_content_robustness --mode plan-only",
        "python -m scripts.run_slm236_structural_similarity_literal_content_robustness --mode fixture",
        "```",
        "",
    ]
    return "\n".join(lines)
