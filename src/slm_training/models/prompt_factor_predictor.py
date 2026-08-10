"""SIE-003 (SLM-477): prompt-to-partial-requirements prediction behind the
advisory-only seam (EXP-SR-2).

A learned, prompt-text-only predictor for one closed-vocabulary
``SemanticPlanV1`` factor (the ``style_layout`` factor family's archetype
choice), wired through the existing prompt-requirements/plan seam rather
than as a new legality path.

Honest scope note on "authorized factor families" (issue AC: "No factor
without an oracle ceiling is promoted into the model merely for
convenience"): EXP-SR-1 (SIE-002/SLM-476, "Execute held-out factor-wise
semantic oracle localization") has not run yet, so there is no real oracle-
ceiling evidence to gate on today. :data:`DEFAULT_AUTHORIZED_FACTOR_FAMILIES`
is therefore an explicit, documented *placeholder* -- callers pass their own
``authorized_factor_families`` once EXP-SR-1 produces real evidence;
:func:`prediction_to_requirement_fact` fails closed (returns ``None``) for
any factor family not in that set, so nothing is ever promoted "merely for
convenience" by omission.

Reuses every relevant owner rather than re-deriving anything:

* :class:`TinyBagOfWordsClassifier` is the "tiny trainable baseline" the
  issue scope asks for -- pure NumPy multinomial-Naive-Bayes, no external
  ML dependency, no gradient descent, trained in-process on
  ``role == "positive"`` (i.e. genuinely gold-matching, not deliberately
  corrupted contrast-negative) rows of the committed ``openui_hard_valid_v1``
  corpus. :func:`frequency_baseline_predict` is the matched majority-class
  control EXP-SR-2's own catalogue entry names
  (``matched_control_axis="Same prompts scored by the predictor and by a
  requirement-frequency-only baseline"``); :func:`deterministic_keyword_predict`
  is the deterministic-request-feature comparison arm.
* Calibration/abstention reuses
  ``dsl.grammar.fastpath.trust_train.gate_calibration_report`` (VCE-010's
  own risk-coverage/Brier/ECE machinery) rather than a second calibration
  mechanism -- its ``abstain_required``/``operating_point`` are the
  predictor's abstention threshold.
* :func:`prediction_to_requirement_fact` bridges a prediction into a real
  ``data.progspec.prompt_requirements.RequirementFact`` -- always
  ``provenance="predicted"`` (never ``"oracle_override"``/gold; that value
  does not even exist in ``RequirementProvenance``) and
  ``authority="advisory-learned"``, so the existing authority machinery
  (``requirements_project.requirements_may_hard_prune``, which requires
  compiler-authored-certified evidence) structurally can never let a
  prediction hard-prune. :func:`predict_factor`'s signature takes only
  prompt text -- there is no parameter through which a gold/oracle plan
  could reach production inference.

Registered as its own subsystem (same divergence precedent as
SGS-008/SRP-004/005/007/008/009): the ownership_map row for SIE-003
predicted ``new_owner_justified: false`` (extends ``semantic_plan``), but
these dataclasses are not literally defined in ``semantic_plan.py``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from slm_training.data.progspec.prompt_requirements import RequirementFact
from slm_training.dsl.grammar.fastpath.trust_train import gate_calibration_report
from slm_training.models.semantic_plan_factors import PROGRAM_FACTOR_FAMILIES

PREDICTOR_SCHEMA_VERSION = "v1"
PREDICTOR_VERSION = "prompt_factor_predictor/v1"

# Honest placeholder pending real EXP-SR-1 (SLM-476) evidence -- see module
# docstring. "style_layout" is the PROGRAM_FACTOR_FAMILIES entry the
# archetype choice this predictor targets falls under.
DEFAULT_AUTHORIZED_FACTOR_FAMILIES: tuple[str, ...] = ("style_layout",)
assert set(DEFAULT_AUTHORIZED_FACTOR_FAMILIES) <= set(PROGRAM_FACTOR_FAMILIES)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class PromptFactorExampleV1:
    """One (prompt, label) training/evaluation example, grouped by
    ``source_program_id`` so a train/held-out split can group before
    splitting (EXP-SR-2's own declared leakage control: "Prompt/requirement
    pairs are grouped by source template before any train/eval split")."""

    source_program_id: str
    prompt: str
    label: str


@dataclass(frozen=True)
class TinyBagOfWordsClassifier:
    """Multinomial-Naive-Bayes-style bag-of-words classifier -- the "tiny
    trainable baseline" this issue's scope asks for. Pure NumPy, no
    external ML dependency, no gradient descent."""

    vocab: tuple[str, ...]
    class_labels: tuple[str, ...]
    log_prior: tuple[float, ...]
    log_likelihood: tuple[tuple[float, ...], ...]  # [class_index][vocab_index]
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if not self.vocab:
            raise ValueError("vocab must be non-empty")
        if not self.class_labels:
            raise ValueError("class_labels must be non-empty")
        if len(self.log_prior) != len(self.class_labels):
            raise ValueError("log_prior must have one entry per class")
        if len(self.log_likelihood) != len(self.class_labels):
            raise ValueError("log_likelihood must have one row per class")
        if any(len(row) != len(self.vocab) for row in self.log_likelihood):
            raise ValueError("every log_likelihood row must have one entry per vocab word")

    @classmethod
    def fit(
        cls, examples: list[PromptFactorExampleV1], *, alpha: float = 1.0
    ) -> "TinyBagOfWordsClassifier":
        if not examples:
            raise ValueError("fit requires at least one example")
        class_labels = tuple(sorted({example.label for example in examples}))
        class_index = {label: index for index, label in enumerate(class_labels)}

        vocab_counter: Counter[str] = Counter()
        for example in examples:
            vocab_counter.update(_tokenize(example.prompt))
        vocab = tuple(sorted(vocab_counter))
        vocab_index = {word: index for index, word in enumerate(vocab)}

        word_counts = np.zeros((len(class_labels), len(vocab)), dtype=np.float64)
        class_counts = np.zeros(len(class_labels), dtype=np.float64)
        for example in examples:
            row = class_index[example.label]
            class_counts[row] += 1
            for token in _tokenize(example.prompt):
                word_counts[row, vocab_index[token]] += 1

        log_prior = np.log(class_counts / class_counts.sum())
        smoothed = word_counts + alpha
        log_likelihood = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        return cls(
            vocab=vocab,
            class_labels=class_labels,
            log_prior=tuple(float(x) for x in log_prior),
            log_likelihood=tuple(tuple(float(x) for x in row) for row in log_likelihood),
            alpha=alpha,
        )

    def predict_proba(self, text: str) -> dict[str, float]:
        vocab_index = {word: index for index, word in enumerate(self.vocab)}
        tokens = [token for token in _tokenize(text) if token in vocab_index]
        log_likelihood = np.array(self.log_likelihood, dtype=np.float64)
        scores = np.array(self.log_prior, dtype=np.float64).copy()
        for token in tokens:
            scores += log_likelihood[:, vocab_index[token]]
        scores -= scores.max()  # numerically stable softmax
        weights = np.exp(scores)
        probabilities = weights / weights.sum()
        return {label: float(p) for label, p in zip(self.class_labels, probabilities)}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TinyBagOfWordsClassifier":
        return cls(
            vocab=tuple(payload["vocab"]),
            class_labels=tuple(payload["class_labels"]),
            log_prior=tuple(payload["log_prior"]),
            log_likelihood=tuple(tuple(row) for row in payload["log_likelihood"]),
            alpha=payload["alpha"],
        )


def frequency_baseline_predict(train_examples: list[PromptFactorExampleV1]) -> dict[str, float]:
    """The matched majority-class control EXP-SR-2's catalogue entry names
    (a requirement-frequency-only baseline) -- ignores prompt text entirely."""
    counts = Counter(example.label for example in train_examples)
    total = sum(counts.values())
    return {label: count / total for label, count in counts.items()}


# A small, hand-authored, honest deterministic-keyword baseline -- the
# "deterministic request features" comparison arm. Not learned from data.
_ARCHETYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "stack_column": ("column", "vertical", "stack"),
    "stack_row": ("row", "horizontal", "toolbar"),
    "form": ("form", "input", "submit", "field"),
    "card": ("card", "panel", "tile"),
    "list": ("list", "items", "feed"),
    "grid": ("grid", "gallery", "columns"),
}


def deterministic_keyword_predict(text: str) -> str | None:
    """Returns the first archetype whose keyword set the prompt mentions, or
    ``None`` (the deterministic baseline's own honest abstention)."""
    tokens = set(_tokenize(text))
    for archetype, keywords in _ARCHETYPE_KEYWORDS.items():
        if tokens & set(keywords):
            return archetype
    return None


@dataclass(frozen=True)
class PromptFactorPredictionV1:
    """One typed, calibrated prediction. Structurally cannot represent gold
    provenance or non-advisory authority -- ``__post_init__`` fails closed
    on both."""

    schema_version: str
    predictor_version: str
    factor_family: str
    predicted_value: str | None
    confidence: float
    abstained: bool
    abstention_reason: str | None
    authority: str
    provenance: str

    def __post_init__(self) -> None:
        if self.schema_version != PREDICTOR_SCHEMA_VERSION:
            raise ValueError(f"unsupported predictor schema: {self.schema_version!r}")
        if self.authority != "advisory-learned":
            raise ValueError("prompt factor predictions must be authority='advisory-learned'")
        if self.provenance != "predicted":
            raise ValueError("prompt factor predictions must be provenance='predicted' -- never gold")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.abstained:
            if self.predicted_value is not None:
                raise ValueError("an abstained prediction must not carry a predicted_value")
            if not self.abstention_reason:
                raise ValueError("an abstained prediction must carry an abstention_reason")
        else:
            if self.predicted_value is None:
                raise ValueError("a non-abstained prediction must carry a predicted_value")
            if self.abstention_reason is not None:
                raise ValueError("a non-abstained prediction must not carry an abstention_reason")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibrate_abstention_threshold(
    classifier: TinyBagOfWordsClassifier,
    validation_examples: list[PromptFactorExampleV1],
    *,
    max_selective_risk: float = 0.05,
) -> dict[str, Any]:
    """Calibrate an abstention confidence threshold via VCE-010's own
    risk-coverage/Brier/ECE report -- never a second calibration mechanism."""
    probabilities: list[float] = []
    labels: list[float] = []
    for example in validation_examples:
        proba = classifier.predict_proba(example.prompt)
        top_label = max(proba, key=proba.get)
        probabilities.append(proba[top_label])
        labels.append(1.0 if top_label == example.label else 0.0)
    return gate_calibration_report(probabilities, labels, max_selective_risk=max_selective_risk)


def resolve_confidence_threshold(calibration_report: dict[str, Any]) -> float:
    """The safe abstention threshold implied by a calibration report.

    When calibration finds no threshold achieving the declared
    ``max_selective_risk`` (``operating_point`` is ``None``), the safe
    default is a threshold no real probability can ever clear -- every
    prediction abstains rather than emitting an uncalibrated one.
    """
    operating_point = calibration_report.get("operating_point")
    if operating_point is None:
        return 1.0 + 1e-9
    return float(operating_point["threshold"])


def predict_factor(
    classifier: TinyBagOfWordsClassifier,
    text: str,
    *,
    factor_family: str,
    confidence_threshold: float,
    predictor_version: str = PREDICTOR_VERSION,
) -> PromptFactorPredictionV1:
    """Predict ``factor_family`` from ``text`` alone -- no plan/oracle
    parameter exists on this signature, so gold/oracle data has no path
    into production inference through this function."""
    proba = classifier.predict_proba(text)
    top_label = max(proba, key=proba.get)
    top_confidence = proba[top_label]
    if top_confidence < confidence_threshold:
        return PromptFactorPredictionV1(
            schema_version=PREDICTOR_SCHEMA_VERSION,
            predictor_version=predictor_version,
            factor_family=factor_family,
            predicted_value=None,
            confidence=top_confidence,
            abstained=True,
            abstention_reason=(
                f"confidence {top_confidence:.4f} below calibrated threshold {confidence_threshold:.4f}"
            ),
            authority="advisory-learned",
            provenance="predicted",
        )
    return PromptFactorPredictionV1(
        schema_version=PREDICTOR_SCHEMA_VERSION,
        predictor_version=predictor_version,
        factor_family=factor_family,
        predicted_value=top_label,
        confidence=top_confidence,
        abstained=False,
        abstention_reason=None,
        authority="advisory-learned",
        provenance="predicted",
    )


def prediction_to_requirement_fact(
    prediction: PromptFactorPredictionV1,
    *,
    fact_id: str,
    authorized_factor_families: tuple[str, ...] = DEFAULT_AUTHORIZED_FACTOR_FAMILIES,
) -> RequirementFact | None:
    """Bridge a prediction into the real prompt-requirements contract, or
    ``None`` when the factor family isn't authorized or the predictor
    abstained -- never a fabricated, unauthorized, or gold-provenance fact."""
    if prediction.factor_family not in authorized_factor_families:
        return None
    if prediction.abstained or prediction.predicted_value is None:
        return None
    return RequirementFact(
        fact_id=fact_id,
        factor_family=prediction.factor_family,
        disposition="preferred",
        statement=(
            f"predicted {prediction.factor_family}={prediction.predicted_value!r} "
            f"(predictor {prediction.predictor_version}, confidence {prediction.confidence:.4f})"
        ),
        confidence=prediction.confidence,
        provenance="predicted",
        authority="advisory-learned",
        source_span=None,
    )


__all__ = [
    "DEFAULT_AUTHORIZED_FACTOR_FAMILIES",
    "PREDICTOR_SCHEMA_VERSION",
    "PREDICTOR_VERSION",
    "PromptFactorExampleV1",
    "PromptFactorPredictionV1",
    "TinyBagOfWordsClassifier",
    "calibrate_abstention_threshold",
    "deterministic_keyword_predict",
    "frequency_baseline_predict",
    "predict_factor",
    "prediction_to_requirement_fact",
    "resolve_confidence_threshold",
]
