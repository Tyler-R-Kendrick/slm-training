"""SLM-185: semantic resolution, test-retest reliability, and canonical-equivalence invariance.

Wiring/fixture-only harness.  No model is trained and no GPU is required.
All fixture judges are deterministic and pinned; no live LLM is invoked.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Sequence

from slm_training.dsl.canonicalize import canonical_equal, canonical_fingerprint
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.judge_independence import (
    calibration_error,
    cohen_kappa,
    fleiss_kappa,
    kendall_tau_b,
    spearman,
)
from slm_training.evals.meaningful_program import binding_aware_meaningful_v2
from slm_training.evals.power_protocol import intraclass_correlation
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "JudgeInvocationEnvelopeV1",
    "JudgeResolutionItemV1",
    "SemanticResolutionEndpointV1",
    "SemanticResolutionManifestV1",
    "abstention_aware_scores",
    "apply_resolution_manifest",
    "build_fixture_corpus",
    "build_fixture_envelopes",
    "build_resolution_manifest",
    "classify_delta",
    "equivalence_invariance_error_rate",
    "expected_calibration_error",
    "flip_rate",
    "krippendorff_alpha",
    "pairwise_ordering_consistency",
    "perturbation_detection_rate",
    "test_retest_reliability",
    "brier_score",
]

MATRIX_VERSION = "quality-v3"
MATRIX_SET = "slm185_judge_resolution"
EXPERIMENT_ID = "slm185-judge-resolution"

ExpectedClass = Literal["canonical_equivalent", "semantic_error", "historical_delta"]
JudgeVerdict = Literal["left", "right", "tie", "equivalent", "different", "refusal", "error"]


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True)
class JudgeInvocationEnvelopeV1:
    """Replayable provenance for one judge endpoint invocation policy."""

    provider: str
    model: str
    revision: str
    system_sha256: str
    rubric_sha256: str
    prompt_sha256: str
    temperature: float
    top_p: float
    seed: int | None
    retry_policy: dict[str, Any]
    response_digest: str
    parser_version: str
    prior_judge_families: tuple[str, ...]
    independent: bool
    participated_in_creation: bool
    participated_in_admission: bool
    participated_in_training: bool

    def __post_init__(self) -> None:
        for name in ("system_sha256", "rubric_sha256", "prompt_sha256"):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase sha256")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["prior_judge_families"] = list(self.prior_judge_families)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JudgeInvocationEnvelopeV1":
        return cls(
            provider=str(data["provider"]),
            model=str(data["model"]),
            revision=str(data["revision"]),
            system_sha256=str(data["system_sha256"]),
            rubric_sha256=str(data["rubric_sha256"]),
            prompt_sha256=str(data["prompt_sha256"]),
            temperature=float(data["temperature"]),
            top_p=float(data["top_p"]),
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            retry_policy=dict(data.get("retry_policy") or {}),
            response_digest=str(data["response_digest"]),
            parser_version=str(data["parser_version"]),
            prior_judge_families=tuple(str(f) for f in data.get("prior_judge_families", [])),
            independent=bool(data.get("independent", False)),
            participated_in_creation=bool(data.get("participated_in_creation", False)),
            participated_in_admission=bool(data.get("participated_in_admission", False)),
            participated_in_training=bool(data.get("participated_in_training", False)),
        )


@dataclass(frozen=True)
class JudgeResolutionItemV1:
    """One pair under repeated judgment for resolution analysis."""

    item_id: str
    pair_group: str
    source_a: str
    source_b: str
    transformation_provenance: str
    expected_class: ExpectedClass
    canonical_fingerprint_a: str
    canonical_fingerprint_b: str
    human_label: str | None
    repeated_scores_per_endpoint: dict[str, tuple[float | None, ...]]
    verdicts_per_endpoint: dict[str, tuple[str | None, ...]]
    abstentions_per_endpoint: dict[str, tuple[bool, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "pair_group": self.pair_group,
            "source_a": self.source_a,
            "source_b": self.source_b,
            "transformation_provenance": self.transformation_provenance,
            "expected_class": self.expected_class,
            "canonical_fingerprint_a": self.canonical_fingerprint_a,
            "canonical_fingerprint_b": self.canonical_fingerprint_b,
            "human_label": self.human_label,
            "repeated_scores_per_endpoint": {
                k: list(v) for k, v in self.repeated_scores_per_endpoint.items()
            },
            "verdicts_per_endpoint": {
                k: list(v) for k, v in self.verdicts_per_endpoint.items()
            },
            "abstentions_per_endpoint": {
                k: list(v) for k, v in self.abstentions_per_endpoint.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JudgeResolutionItemV1":
        return cls(
            item_id=str(data["item_id"]),
            pair_group=str(data["pair_group"]),
            source_a=str(data["source_a"]),
            source_b=str(data["source_b"]),
            transformation_provenance=str(data["transformation_provenance"]),
            expected_class=str(data["expected_class"]),  # type: ignore[arg-type]
            canonical_fingerprint_a=str(data["canonical_fingerprint_a"]),
            canonical_fingerprint_b=str(data["canonical_fingerprint_b"]),
            human_label=None if data.get("human_label") is None else str(data["human_label"]),
            repeated_scores_per_endpoint={
                k: tuple(v) for k, v in data.get("repeated_scores_per_endpoint", {}).items()
            },
            verdicts_per_endpoint={
                k: tuple(v) for k, v in data.get("verdicts_per_endpoint", {}).items()
            },
            abstentions_per_endpoint={
                k: tuple(v) for k, v in data.get("abstentions_per_endpoint", {}).items()
            },
        )


@dataclass(frozen=True)
class SemanticResolutionEndpointV1:
    """Per-endpoint resolution and reliability metrics."""

    endpoint_label: str
    provider: str
    model: str
    revision: str
    metric_family: str
    measured_flip_rate: float | None
    cohen_kappa: float | None
    fleiss_kappa: float | None
    krippendorff_alpha: float | None
    icc_1_1: dict[str, float]
    pairwise_ordering_consistency: dict[str, Any] | None
    equivalence_invariance_error_rate: float | None
    perturbation_detection_rate: float | None
    brier_score: float | None
    ece: float | None
    abstention_rate: float
    required_repeats: int
    majority_rule: bool
    minimum_resolvable_delta: float
    equivalence_margin: float
    claim_language_permit_set: tuple[str, ...]
    requires_independent_confirmation: bool

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["icc_1_1"] = dict(self.icc_1_1)
        data["pairwise_ordering_consistency"] = self.pairwise_ordering_consistency
        data["claim_language_permit_set"] = list(self.claim_language_permit_set)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticResolutionEndpointV1":
        return cls(
            endpoint_label=str(data["endpoint_label"]),
            provider=str(data["provider"]),
            model=str(data["model"]),
            revision=str(data["revision"]),
            metric_family=str(data["metric_family"]),
            measured_flip_rate=None if data.get("measured_flip_rate") is None else float(data["measured_flip_rate"]),
            cohen_kappa=None if data.get("cohen_kappa") is None else float(data["cohen_kappa"]),
            fleiss_kappa=None if data.get("fleiss_kappa") is None else float(data["fleiss_kappa"]),
            krippendorff_alpha=None if data.get("krippendorff_alpha") is None else float(data["krippendorff_alpha"]),
            icc_1_1=dict(data.get("icc_1_1", {})),
            pairwise_ordering_consistency=data.get("pairwise_ordering_consistency"),
            equivalence_invariance_error_rate=None if data.get("equivalence_invariance_error_rate") is None else float(data["equivalence_invariance_error_rate"]),
            perturbation_detection_rate=None if data.get("perturbation_detection_rate") is None else float(data["perturbation_detection_rate"]),
            brier_score=None if data.get("brier_score") is None else float(data["brier_score"]),
            ece=None if data.get("ece") is None else float(data["ece"]),
            abstention_rate=float(data.get("abstention_rate", 0.0)),
            required_repeats=int(data.get("required_repeats", 1)),
            majority_rule=bool(data.get("majority_rule", False)),
            minimum_resolvable_delta=float(data.get("minimum_resolvable_delta", 0.05)),
            equivalence_margin=float(data.get("equivalence_margin", 0.01)),
            claim_language_permit_set=tuple(str(x) for x in data.get("claim_language_permit_set", ("directional", "resolved", "below_noise_floor"))),
            requires_independent_confirmation=bool(data.get("requires_independent_confirmation", False)),
        )


@dataclass(frozen=True)
class SemanticResolutionManifestV1:
    """Full fixture manifest for SLM-185."""

    schema: str
    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    endpoints: tuple[SemanticResolutionEndpointV1, ...]
    global_floor: float
    version_stamp: dict[str, Any]
    generated_at: str
    provenance: dict[str, Any]

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
            "endpoints": [e.to_dict() for e in self.endpoints],
            "global_floor": self.global_floor,
            "version_stamp": self.version_stamp,
            "generated_at": self.generated_at,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticResolutionManifestV1":
        return cls(
            schema=str(data.get("schema", "SemanticResolutionManifestV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", "slm185_fixture")),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", "")),
            falsifier=str(data.get("falsifier", "")),
            endpoints=tuple(
                SemanticResolutionEndpointV1.from_dict(e)
                for e in data.get("endpoints", [])
            ),
            global_floor=float(data.get("global_floor", 0.0)),
            version_stamp=dict(data.get("version_stamp", {})),
            generated_at=str(data.get("generated_at", "")),
            provenance=dict(data.get("provenance", {})),
        )


def flip_rate(scores_or_verdicts: Sequence[Any], threshold: float | None = None) -> float | None:
    """Fraction of adjacent repeat pairs that disagree.

    If ``scores_or_verdicts`` is a sequence of sequences, computes the mean
    flip rate across items.  For scores, a ``threshold`` turns the comparison
    into a sign flip (``value > threshold``).
    """
    if not scores_or_verdicts:
        return None

    def _item_flips(values: Sequence[Any]) -> float | None:
        if len(values) < 2:
            return None
        pairs = list(zip(values, values[1:]))
        if not pairs:
            return None
        if threshold is not None:
            disagree = sum(
                1
                for a, b in pairs
                if a is None
                or b is None
                or ((a > threshold) != (b > threshold))
            )
        else:
            disagree = sum(1 for a, b in pairs if a != b)
        return disagree / len(pairs)

    if isinstance(scores_or_verdicts[0], (list, tuple)):
        rates = [_item_flips(tuple(item)) for item in scores_or_verdicts]
        rates = [r for r in rates if r is not None]
        return sum(rates) / len(rates) if rates else None
    return _item_flips(tuple(scores_or_verdicts))


def krippendorff_alpha(ratings: Sequence[Sequence[Any]], level: str = "nominal") -> float | None:
    """Simple nominal Krippendorff's alpha for complete rater matrices.

    Returns ``None`` for degenerate inputs (no units, single label, or
    mismatched rater counts).
    """
    if level != "nominal":
        raise NotImplementedError("only nominal alpha is implemented")
    if not ratings or not ratings[0]:
        return None
    n_raters = len(ratings[0])
    if any(len(row) != n_raters for row in ratings) or n_raters < 2:
        return None

    flat = [value for row in ratings for value in row]
    if len(set(flat)) < 2:
        return None

    counts = Counter(flat)
    total = len(flat)
    expected_disagreement = 1.0 - sum((count / total) ** 2 for count in counts.values())

    observed_disagreement = 0.0
    pair_count = 0
    for row in ratings:
        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                observed_disagreement += 0.0 if row[i] == row[j] else 1.0
                pair_count += 1
    observed_disagreement /= pair_count

    if expected_disagreement <= 0.0:
        return None
    return 1.0 - observed_disagreement / expected_disagreement


def pairwise_ordering_consistency(
    scores_a: Sequence[float], scores_b: Sequence[float], repeats: int
) -> dict[str, Any] | None:
    """Rank stability of paired repeated scores (Spearman/Kendall)."""
    if len(scores_a) != len(scores_b) or len(scores_a) < 2:
        return None
    return {
        "spearman": spearman(scores_a, scores_b),
        "kendall_tau_b": kendall_tau_b(scores_a, scores_b),
        "repeats": repeats,
    }


def brier_score(probabilities: Sequence[float], outcomes: Sequence[bool]) -> float | None:
    """Mean squared error of probabilistic predictions against binary outcomes."""
    if len(probabilities) != len(outcomes) or not probabilities:
        return None
    return sum((p - (1.0 if o else 0.0)) ** 2 for p, o in zip(probabilities, outcomes)) / len(probabilities)


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> float | None:
    """Expected calibration error; delegates to the judge_independence helper."""
    return calibration_error(confidences, correct, bins=bins)


def test_retest_reliability(
    repeated_scores: Sequence[float], cluster_ids: Sequence[Any] | None = None
) -> dict[str, float]:
    """ICC(1,1) or paired bootstrap CI for repeated measurements."""
    if cluster_ids is None:
        cluster_ids = list(range(len(repeated_scores)))
    return intraclass_correlation(repeated_scores, cluster_ids)


# This is a public API function, not a pytest test; prevent pytest collection.
test_retest_reliability.__test__ = False  # type: ignore[attr-defined]


def equivalence_invariance_error_rate(
    items: Sequence[JudgeResolutionItemV1], endpoint_label: str
) -> float | None:
    """Among canonical-equivalent items, rate of non-equivalent endpoint verdicts."""
    equiv = [item for item in items if item.expected_class == "canonical_equivalent"]
    if not equiv:
        return None
    errors = 0
    for item in equiv:
        verdicts = item.verdicts_per_endpoint.get(endpoint_label, ())
        if any(v not in (None, "equivalent", "tie") for v in verdicts):
            errors += 1
    return errors / len(equiv)


def perturbation_detection_rate(
    items: Sequence[JudgeResolutionItemV1], endpoint_label: str
) -> float | None:
    """Among semantic-error items, rate of detected difference."""
    error_items = [item for item in items if item.expected_class == "semantic_error"]
    if not error_items:
        return None
    detected = 0
    for item in error_items:
        verdicts = item.verdicts_per_endpoint.get(endpoint_label, ())
        if any(v in ("different", "left", "right") for v in verdicts):
            detected += 1
    return detected / len(error_items)


def abstention_aware_scores(
    verdicts: Sequence[str | None], abstentions: Sequence[bool] | None = None
) -> dict[str, Any] | None:
    """Pass rate computed over non-abstained verdicts."""
    if not verdicts:
        return None
    abst = list(abstentions or [False] * len(verdicts))
    n = len(verdicts)
    abstained = sum(bool(a) for a in abst)
    scored = [v for v, a in zip(verdicts, abst) if not a]
    pass_rate = (
        sum(v in ("left", "equivalent", "pass") for v in scored) / len(scored)
        if scored
        else None
    )
    return {
        "n": n,
        "abstention_rate": abstained / n,
        "scored_n": len(scored),
        "pass_rate": pass_rate,
    }


def _endpoint_verdict_matrix(
    items: Sequence[JudgeResolutionItemV1], endpoint_label: str
) -> list[list[str | None]]:
    return [list(item.verdicts_per_endpoint.get(endpoint_label, ())) for item in items]


def _endpoint_score_matrix(
    items: Sequence[JudgeResolutionItemV1], endpoint_label: str
) -> list[list[float | None]]:
    return [list(item.repeated_scores_per_endpoint.get(endpoint_label, ())) for item in items]


def build_resolution_manifest(
    items: Sequence[JudgeResolutionItemV1],
    endpoints: Sequence[SemanticResolutionEndpointV1],
    corpus_path: str | None = None,
    envelopes: Sequence[JudgeInvocationEnvelopeV1] | None = None,
    *,
    run_id: str = "slm185-judge-resolution",
) -> SemanticResolutionManifestV1:
    """Assemble a full resolution manifest from judged items and endpoint metadata."""
    return SemanticResolutionManifestV1(
        schema="SemanticResolutionManifestV1",
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        status="fixture",
        claim_class="wiring",
        hypothesis=(
            "Deterministic fixture judges can expose test-retest reliability, "
            "canonical-equivalence invariance, and semantic perturbation detection "
            "at a per-endpoint resolution floor."
        ),
        falsifier=(
            "The reliability metrics collapse (NaN/undefined) or the equivalence "
            "invariance error rate is non-zero for canonical-equivalent pairs."
        ),
        endpoints=tuple(endpoints),
        global_floor=min(
            (e.minimum_resolvable_delta for e in endpoints),
            default=0.0,
        ),
        version_stamp=build_version_stamp(
            "evals.judge_resolution",
            "harness.experiments",
            "matrix.quality",
        ),
        generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        provenance={
            "corpus_path": corpus_path,
            "judge_envelopes": [e.to_dict() for e in (envelopes or ())],
            "item_n": len(items),
        },
    )


def classify_delta(
    delta: float, endpoint_label: str, manifest: SemanticResolutionManifestV1
) -> str:
    """Classify an endpoint effect size against its resolution floor."""
    endpoint = next(
        (e for e in manifest.endpoints if e.endpoint_label == endpoint_label), None
    )
    if endpoint is None:
        return "below_noise_floor"
    abs_delta = abs(float(delta))
    if abs_delta <= endpoint.equivalence_margin:
        return "below_noise_floor"
    if abs_delta < endpoint.minimum_resolvable_delta:
        return "resolved"
    return "directional"


def apply_resolution_manifest(
    scoreboard: dict[str, Any], manifest: SemanticResolutionManifestV1
) -> dict[str, Any]:
    """Annotate a scoreboard with ``semantic_resolution`` labels per endpoint.

    Raw metrics are never altered.  The returned dict contains a new top-level
    key ``semantic_resolution`` mapping endpoint labels to classified deltas.
    """
    out = dict(scoreboard)
    resolution: dict[str, Any] = {}
    suites = out.get("suites") or {}
    for endpoint in manifest.endpoints:
        label = endpoint.endpoint_label
        deltas: dict[str, str] = {}
        for suite_name, suite_metrics in suites.items():
            raw = suite_metrics.get(label)
            if raw is None:
                # Try common aggregate keys.
                for key in ("rate", "mean", "score", "value", "delta"):
                    if key in suite_metrics:
                        raw = suite_metrics[key]
                        break
            if raw is None:
                continue
            try:
                delta = float(raw)
            except (TypeError, ValueError):
                continue
            deltas[str(suite_name)] = classify_delta(delta, label, manifest)
        resolution[label] = {
            "minimum_resolvable_delta": endpoint.minimum_resolvable_delta,
            "equivalence_margin": endpoint.equivalence_margin,
            "claim_language_permit_set": list(endpoint.claim_language_permit_set),
            "requires_independent_confirmation": endpoint.requires_independent_confirmation,
            "deltas": deltas,
        }
    out["semantic_resolution"] = resolution
    return out


# ---------------------------------------------------------------------------
# Fixture corpus and deterministic judges
# ---------------------------------------------------------------------------

_FIXTURE_ENVELOPE = JudgeInvocationEnvelopeV1(
    provider="fixture",
    model="binding_aware_meaningful_v2",
    revision="2.1.0",
    system_sha256="0" * 64,
    rubric_sha256="0" * 64,
    prompt_sha256="0" * 64,
    temperature=0.0,
    top_p=1.0,
    seed=0,
    retry_policy={"max_attempts": 1, "backoff": "none"},
    response_digest="fixture_deterministic",
    parser_version="openui_canonical_v2",
    prior_judge_families=(),
    independent=True,
    participated_in_creation=False,
    participated_in_admission=False,
    participated_in_training=False,
)

_NOISY_ENVELOPE = JudgeInvocationEnvelopeV1(
    provider="fixture",
    model="seeded_hash_scorer",
    revision="1.0.0",
    system_sha256="1" * 64,
    rubric_sha256="1" * 64,
    prompt_sha256="1" * 64,
    temperature=0.0,
    top_p=1.0,
    seed=185,
    retry_policy={"max_attempts": 1, "backoff": "none"},
    response_digest="fixture_seeded_hash",
    parser_version="none",
    prior_judge_families=("binding_aware_meaningful_v2",),
    independent=True,
    participated_in_creation=False,
    participated_in_admission=False,
    participated_in_training=False,
)


def build_fixture_envelopes() -> tuple[JudgeInvocationEnvelopeV1, JudgeInvocationEnvelopeV1]:
    """Return the two deterministic fixture judge envelopes."""
    return (_FIXTURE_ENVELOPE, _NOISY_ENVELOPE)


_CANONICAL_EQUIVALENT_PAIRS: list[tuple[str, str, str]] = [
    (
        "alpha_rename",
        'root = Stack([a])\na = Button(":cta.label")',
        'root = Stack([b])\nb = Button(":cta.label")',
    ),
    (
        "statement_reorder",
        'a = Button(":cta.label")\nroot = Stack([a])',
        'root = Stack([a])\na = Button(":cta.label")',
    ),
    (
        "serializer_parenthesization",
        'root = Stack([a])\na = Button(":cta.label")',
        'root=Stack([a])\na=Button(":cta.label")',
    ),
    (
        "style_literal_strip",
        'root = Button(":cta.label", "primary")',
        'root = Button(":cta.label")',
    ),
    (
        "slot_permutation_identical_children",
        'root = Stack([a, b])\na = Button(":cta.label")\nb = Button(":cta.label")',
        'root = Stack([b, a])\na = Button(":cta.label")\nb = Button(":cta.label")',
    ),
    (
        "ast_identical_surface_variant",
        'root = Stack([x])\nx = Button(":cta.label")',
        'root = Stack([y])\ny = Button(":cta.label")',
    ),
]

_SEMANTIC_ERROR_PAIRS: list[tuple[str, str, str, str]] = [
    (
        "wrong_component_family",
        "Build a Button. Placeholders: :cta.label",
        'root = Button(":cta.label")',
        'root = TextContent(":cta.label")',
    ),
    (
        "omitted_required_role",
        "Build a Card with title. Placeholders: :hero.title",
        'root = Card([title])\ntitle = TextContent(":hero.title")',
        'root = Card([])',
    ),
    (
        "wrong_slot_binding",
        "Build a Button. Placeholders: :cta.label",
        'root = Button(":cta.label")',
        'root = Button(":hero.title")',
    ),
    (
        "wrong_reference_target",
        "Build a Button. Placeholders: :cta.label",
        '$label = ":cta.label"\nroot = Button($label)',
        '$label = ":hero.title"\nroot = Button($label)',
    ),
    (
        "topology_edge_moved",
        "Build a Stack with a Button. Placeholders: :cta.label",
        'root = Stack([a])\na = Button(":cta.label")',
        'root = Stack([a])\na = TextContent(":cta.label")',
    ),
    (
        "empty_minimal_valid_collapse",
        "Build a Button and a TextContent. Placeholders: :cta.label :hero.title",
        'root = Stack([a, b])\na = Button(":cta.label")\nb = TextContent(":hero.title")',
        'root = Button(":cta.label")',
    ),
]

_HISTORICAL_DELTA_STUBS: list[tuple[str, str, str | None, str | None]] = [
    ("E561_vs_E562", "ricoeval_delta", None, None),
    ("X22_vs_X9", "grammar_matrix_delta", None, None),
    ("placeholder_delta", "historical_placeholder", None, None),
]


def _record_for_pair(prompt: str, source: str, item_id: str) -> ExampleRecord:
    return ExampleRecord(
        id=item_id,
        prompt=prompt,
        openui=source,
        split="smoke",
        source="fixture",
    )


def _binding_aware_pair_verdict(source_a: str, source_b: str, prompt: str, item_id: str) -> tuple[str, float]:
    """Return (verdict, score) for the binding-aware fixture judge on a pair."""
    rec_a = _record_for_pair(prompt, source_a, f"{item_id}_a")
    rec_b = _record_for_pair(prompt, source_b, f"{item_id}_b")
    pass_a = binding_aware_meaningful_v2(source_a, record=rec_a).verdict
    pass_b = binding_aware_meaningful_v2(source_b, record=rec_b).verdict
    if pass_a and not pass_b:
        return "left", 1.0
    if pass_b and not pass_a:
        return "right", -1.0
    return "tie", 0.0


def _canonical_equal_pair_verdict(source_a: str, source_b: str) -> tuple[str, float]:
    """Return (verdict, score) for the canonical-equality fixture judge."""
    if canonical_equal(source_a, source_b):
        return "equivalent", 1.0
    return "different", 0.0


def _seeded_hash_pair_score(source_a: str, source_b: str, seed: int) -> float:
    """Deterministic seeded synthetic score in [-1, 1]."""
    combined = f"{source_a}\n{source_b}\n{seed}".encode("utf-8")
    digest = hashlib.sha256(combined).hexdigest()
    # Convert first 8 hex digits to a signed float in [-1, 1].
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return (value - 0.5) * 2.0


def build_fixture_corpus(
    *,
    repeats: int = 5,
) -> tuple[list[JudgeResolutionItemV1], list[JudgeInvocationEnvelopeV1]]:
    """Build the deterministic SLM-185 fixture corpus.

    Returns judged items and the envelopes used.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    envelopes = list(build_fixture_envelopes())
    items: list[JudgeResolutionItemV1] = []

    # Canonical-equivalent pairs: both sources should pass the semantic judge
    # and canonical_equal should return True.
    for index, (provenance, source_a, source_b) in enumerate(_CANONICAL_EQUIVALENT_PAIRS, start=1):
        item_id = f"ce_{index:02d}_{provenance}"
        prompt = "Build a Button. Placeholders: :cta.label"
        scores: dict[str, tuple[float | None, ...]] = {}
        verdicts: dict[str, tuple[str | None, ...]] = {}
        abstentions: dict[str, tuple[bool, ...]] = {}

        binding_verdict, binding_score = _binding_aware_pair_verdict(
            source_a, source_b, prompt, item_id
        )
        canonical_verdict, canonical_score = _canonical_equal_pair_verdict(source_a, source_b)

        scores["fixture_binding_aware_v2"] = tuple(binding_score for _ in range(repeats))
        verdicts["fixture_binding_aware_v2"] = tuple(binding_verdict for _ in range(repeats))
        abstentions["fixture_binding_aware_v2"] = tuple(False for _ in range(repeats))

        scores["fixture_canonical_equal"] = tuple(canonical_score for _ in range(repeats))
        verdicts["fixture_canonical_equal"] = tuple(canonical_verdict for _ in range(repeats))
        abstentions["fixture_canonical_equal"] = tuple(False for _ in range(repeats))

        hash_scores = tuple(
            _seeded_hash_pair_score(source_a, source_b, seed)
            for seed in range(repeats)
        )
        hash_verdicts = tuple(
            "left" if s > 0.2 else "right" if s < -0.2 else "tie"
            for s in hash_scores
        )
        scores["fixture_seeded_hash_scorer"] = hash_scores
        verdicts["fixture_seeded_hash_scorer"] = hash_verdicts
        abstentions["fixture_seeded_hash_scorer"] = tuple(False for _ in range(repeats))

        items.append(
            JudgeResolutionItemV1(
                item_id=item_id,
                pair_group="canonical_equivalent",
                source_a=source_a,
                source_b=source_b,
                transformation_provenance=provenance,
                expected_class="canonical_equivalent",
                canonical_fingerprint_a=canonical_fingerprint(source_a),
                canonical_fingerprint_b=canonical_fingerprint(source_b),
                human_label=None,
                repeated_scores_per_endpoint=scores,
                verdicts_per_endpoint=verdicts,
                abstentions_per_endpoint=abstentions,
            )
        )

    # Semantic-error pairs: source A is correct, source B is a semantic error.
    for index, (provenance, prompt, source_a, source_b) in enumerate(
        _SEMANTIC_ERROR_PAIRS, start=1
    ):
        item_id = f"se_{index:02d}_{provenance}"
        scores: dict[str, tuple[float | None, ...]] = {}
        verdicts: dict[str, tuple[str | None, ...]] = {}
        abstentions: dict[str, tuple[bool, ...]] = {}

        binding_verdict, binding_score = _binding_aware_pair_verdict(
            source_a, source_b, prompt, item_id
        )
        canonical_verdict, canonical_score = _canonical_equal_pair_verdict(source_a, source_b)

        scores["fixture_binding_aware_v2"] = tuple(binding_score for _ in range(repeats))
        verdicts["fixture_binding_aware_v2"] = tuple(binding_verdict for _ in range(repeats))
        abstentions["fixture_binding_aware_v2"] = tuple(False for _ in range(repeats))

        scores["fixture_canonical_equal"] = tuple(canonical_score for _ in range(repeats))
        verdicts["fixture_canonical_equal"] = tuple(canonical_verdict for _ in range(repeats))
        abstentions["fixture_canonical_equal"] = tuple(False for _ in range(repeats))

        hash_scores = tuple(
            _seeded_hash_pair_score(source_a, source_b, seed)
            for seed in range(repeats)
        )
        hash_verdicts = tuple(
            "left" if s > 0.2 else "right" if s < -0.2 else "tie"
            for s in hash_scores
        )
        scores["fixture_seeded_hash_scorer"] = hash_scores
        verdicts["fixture_seeded_hash_scorer"] = hash_verdicts
        abstentions["fixture_seeded_hash_scorer"] = tuple(False for _ in range(repeats))

        items.append(
            JudgeResolutionItemV1(
                item_id=item_id,
                pair_group="semantic_error",
                source_a=source_a,
                source_b=source_b,
                transformation_provenance=provenance,
                expected_class="semantic_error",
                canonical_fingerprint_a=canonical_fingerprint(source_a),
                canonical_fingerprint_b=canonical_fingerprint(source_b),
                human_label=None,
                repeated_scores_per_endpoint=scores,
                verdicts_per_endpoint=verdicts,
                abstentions_per_endpoint=abstentions,
            )
        )

    # Historical-delta stubs carry no sources; they exist to exercise classification.
    for index, (item_id_stub, provenance, _a, _b) in enumerate(
        _HISTORICAL_DELTA_STUBS, start=1
    ):
        item_id = f"hd_{index:02d}_{item_id_stub}"
        items.append(
            JudgeResolutionItemV1(
                item_id=item_id,
                pair_group="historical_delta",
                source_a="",
                source_b="",
                transformation_provenance=provenance,
                expected_class="historical_delta",
                canonical_fingerprint_a="" * 64,
                canonical_fingerprint_b="" * 64,
                human_label=None,
                repeated_scores_per_endpoint={},
                verdicts_per_endpoint={},
                abstentions_per_endpoint={},
            )
        )

    return items, envelopes


def _compute_endpoint(
    items: Sequence[JudgeResolutionItemV1],
    endpoint_label: str,
    envelope: JudgeInvocationEnvelopeV1,
    *,
    required_repeats: int = 5,
    minimum_resolvable_delta: float = 0.05,
    equivalence_margin: float = 0.01,
) -> SemanticResolutionEndpointV1:
    """Compute resolution metrics for one endpoint from judged items."""
    matrix = _endpoint_verdict_matrix(items, endpoint_label)
    score_matrix = _endpoint_score_matrix(items, endpoint_label)

    flat_verdicts: list[str | None] = []
    for row in matrix:
        flat_verdicts.extend(row)
    flat_scores: list[float | None] = []
    for row in score_matrix:
        flat_scores.extend(row)

    # Per-item first-repeat scores for ordering consistency.
    first_scores_a = [
        item.repeated_scores_per_endpoint.get(endpoint_label, ())[0]
        for item in items
        if item.repeated_scores_per_endpoint.get(endpoint_label)
    ]
    last_scores_b = [
        item.repeated_scores_per_endpoint.get(endpoint_label, ())[-1]
        for item in items
        if item.repeated_scores_per_endpoint.get(endpoint_label)
    ]

    ordering = None
    if len(first_scores_a) == len(last_scores_b) and len(first_scores_a) >= 2:
        ordering = pairwise_ordering_consistency(
            [float(x) for x in first_scores_a if x is not None],
            [float(x) for x in last_scores_b if x is not None],
            required_repeats,
        )

    # Flip rate across items.
    flip = flip_rate(matrix)

    # Inter-rater agreement using first repeat across items.
    first_repeat = [row[0] if row else None for row in matrix]
    second_repeat = [row[1] if len(row) > 1 else None for row in matrix]
    clean_first = [v for v in first_repeat if v is not None]
    clean_second = [v for v in second_repeat if v is not None]
    kappa = (
        cohen_kappa(clean_first, clean_second)
        if len(clean_first) == len(clean_second) and clean_first
        else None
    )
    fleiss = fleiss_kappa(matrix) if matrix else None
    alpha = krippendorff_alpha(matrix) if matrix else None

    numeric_scores = [float(s) for s in flat_scores if s is not None]
    icc = test_retest_reliability(
        numeric_scores,
        cluster_ids=[i // required_repeats for i in range(len(numeric_scores))],
    )

    equiv_rate = equivalence_invariance_error_rate(items, endpoint_label)
    pert_rate = perturbation_detection_rate(items, endpoint_label)

    # Brier/ECE: treat positive scores as confidence that A is better.
    clean_scores_float = [float(s) for s in flat_scores if s is not None]
    clean_outcomes = [
        item.expected_class == "semantic_error"
        for item in items
        for _ in item.repeated_scores_per_endpoint.get(endpoint_label, ())
        if item.repeated_scores_per_endpoint.get(endpoint_label)
    ]
    brier = None
    ece = None
    if len(clean_scores_float) == len(clean_outcomes) and clean_scores_float:
        # Normalize scores to [0, 1] probability-like values.
        min_s = min(clean_scores_float)
        max_s = max(clean_scores_float)
        if max_s > min_s:
            probs = [(s - min_s) / (max_s - min_s) for s in clean_scores_float]
        else:
            probs = [0.5 for _ in clean_scores_float]
        brier = brier_score(probs, clean_outcomes)
        ece = expected_calibration_error(probs, clean_outcomes, bins=5)

    abst_count = sum(
        sum(item.abstentions_per_endpoint.get(endpoint_label, ()))
        for item in items
    )
    total_abst = sum(
        len(item.abstentions_per_endpoint.get(endpoint_label, ()))
        for item in items
    )
    abst_rate = abst_count / total_abst if total_abst else 0.0

    return SemanticResolutionEndpointV1(
        endpoint_label=endpoint_label,
        provider=envelope.provider,
        model=envelope.model,
        revision=envelope.revision,
        metric_family=envelope.model,
        measured_flip_rate=flip,
        cohen_kappa=kappa,
        fleiss_kappa=fleiss,
        krippendorff_alpha=alpha,
        icc_1_1=icc,
        pairwise_ordering_consistency=ordering,
        equivalence_invariance_error_rate=equiv_rate,
        perturbation_detection_rate=pert_rate,
        brier_score=brier,
        ece=ece,
        abstention_rate=abst_rate,
        required_repeats=required_repeats,
        majority_rule=False,
        minimum_resolvable_delta=minimum_resolvable_delta,
        equivalence_margin=equivalence_margin,
        claim_language_permit_set=("directional", "resolved", "below_noise_floor"),
        requires_independent_confirmation=envelope.independent,
    )


def run_resolution_fixture(
    *,
    repeats: int = 5,
    run_id: str = "slm185-judge-resolution",
    corpus_path: str | None = None,
) -> tuple[SemanticResolutionManifestV1, list[JudgeResolutionItemV1]]:
    """Run the full deterministic fixture and return the manifest plus items."""
    items, envelopes = build_fixture_corpus(repeats=repeats)
    endpoint_specs = [
        ("fixture_binding_aware_v2", envelopes[0], 0.05, 0.01),
        ("fixture_canonical_equal", envelopes[0], 0.02, 0.0),
        ("fixture_seeded_hash_scorer", envelopes[1], 0.3, 0.1),
    ]
    endpoints = [
        _compute_endpoint(
            items,
            label,
            envelope,
            required_repeats=repeats,
            minimum_resolvable_delta=delta,
            equivalence_margin=margin,
        )
        for label, envelope, delta, margin in endpoint_specs
    ]
    manifest = build_resolution_manifest(
        items,
        endpoints,
        corpus_path=corpus_path,
        envelopes=envelopes,
        run_id=run_id,
    )
    return manifest, items
