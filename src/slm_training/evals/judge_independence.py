"""Versioned evidence and agreement statistics for independent judge audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from collections import Counter
import hashlib
from math import sqrt
import random
from time import monotonic
from typing import Any, Callable, Literal, Sequence
import uuid

JudgeProvenance = Literal["deterministic", "external_model", "human"]
JudgeVerdict = Literal["left", "right", "tie", "refusal", "error"]


def text_sha256(value: str) -> str:
    """Return the canonical UTF-8 digest used by the audit envelope."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True)
class JudgeEvidenceV1:
    """One replayable decision with fail-closed independence provenance."""

    evidence_id: str
    audit_id: str
    record_id: str
    generation_id: str
    pair_id: str
    judge_id: str
    provenance: JudgeProvenance
    prompt_sha256: str
    left_output_sha256: str
    right_output_sha256: str
    left_checkpoint_sha256: str
    right_checkpoint_sha256: str
    candidate_model_families: tuple[str, str]
    prior_judge_providers: tuple[str, ...]
    prior_judge_model_families: tuple[str, ...]
    provider: str
    provider_version: str
    model_family: str
    model_version: str
    rubric_id: str
    rubric_version: str
    rubric_sha256: str
    participated_in_creation: bool
    participated_in_admission: bool
    participated_in_training: bool
    participated_in_preference: bool
    participated_in_evaluation: bool
    rubric_used_for_training_admission: bool
    verdict: JudgeVerdict
    left_acceptable: bool | None
    right_acceptable: bool | None
    score: float | None
    reason_codes: tuple[str, ...]
    confidence: float | None
    created_at: str
    blinded: bool
    order_seed: int
    order_sha256: str
    saw_candidate_identity: bool
    saw_automatic_judgments: bool
    retry_count: int = 0
    refused: bool = False
    error: str | None = None
    latency_ms: int | None = None
    cost_usd: float | None = None
    annotator_role: Literal["rater", "adjudicator"] | None = None

    def __post_init__(self) -> None:
        if self.provenance not in {"deterministic", "external_model", "human"}:
            raise ValueError(f"invalid judge provenance: {self.provenance!r}")
        if self.verdict not in {"left", "right", "tie", "refusal", "error"}:
            raise ValueError(f"invalid judge verdict: {self.verdict!r}")
        for name in (
            "prompt_sha256",
            "left_output_sha256",
            "right_output_sha256",
            "left_checkpoint_sha256",
            "right_checkpoint_sha256",
            "rubric_sha256",
            "order_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase sha256")
        if len(self.candidate_model_families) != 2 or not all(
            self.candidate_model_families
        ):
            raise ValueError("candidate_model_families must identify both candidates")
        if self.provenance == "external_model" and (
            not self.prior_judge_providers or not self.prior_judge_model_families
        ):
            raise ValueError("external evidence requires prior-judge provider and family sets")
        for name in (
            "provider",
            "provider_version",
            "model_family",
            "model_version",
            "rubric_id",
            "rubric_version",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.score is not None and not 0 <= self.score <= 1:
            raise ValueError("score must be in [0, 1]")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if self.retry_count < 0:
            raise ValueError("retry_count cannot be negative")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if self.cost_usd is not None and self.cost_usd < 0:
            raise ValueError("cost_usd cannot be negative")
        try:
            timestamp = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("created_at must be ISO-8601") from exc
        if timestamp.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        if self.refused != (self.verdict == "refusal"):
            raise ValueError("refused must agree with the refusal verdict")
        if (self.verdict == "error") != (self.error is not None):
            raise ValueError("error must be present only for the error verdict")
        if self.verdict not in {"error", "refusal"} and (
            not isinstance(self.left_acceptable, bool)
            or not isinstance(self.right_acceptable, bool)
        ):
            raise ValueError("successful evidence requires both acceptability labels")
        if self.provenance == "human":
            if self.annotator_role not in {"rater", "adjudicator"}:
                raise ValueError("human evidence requires an annotator role")
            if not self.blinded or self.saw_candidate_identity or self.saw_automatic_judgments:
                raise ValueError("human evidence must remain blinded")
        elif self.annotator_role is not None:
            raise ValueError("annotator_role is valid only for human evidence")

    @property
    def independent(self) -> bool:
        """True only when every declared external-judge independence gate passes."""
        participation = (
            self.participated_in_creation,
            self.participated_in_admission,
            self.participated_in_training,
            self.participated_in_preference,
            self.participated_in_evaluation,
            self.rubric_used_for_training_admission,
        )
        return (
            self.provenance == "external_model"
            and self.blinded
            and not self.saw_candidate_identity
            and not self.saw_automatic_judgments
            and not any(participation)
            and self.model_family not in self.candidate_model_families
            and self.provider not in self.prior_judge_providers
            and self.model_family not in self.prior_judge_model_families
        )

    def require_independent(self) -> None:
        if not self.independent:
            raise ValueError("judge evidence does not satisfy independence gates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "JudgeEvidenceV1",
            **asdict(self),
            "candidate_model_families": list(self.candidate_model_families),
            "prior_judge_providers": list(self.prior_judge_providers),
            "prior_judge_model_families": list(self.prior_judge_model_families),
            "reason_codes": list(self.reason_codes),
            "independent": self.independent,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> JudgeEvidenceV1:
        if value.get("schema") != "JudgeEvidenceV1":
            raise ValueError("unsupported judge evidence schema")
        fields = dict(value)
        fields.pop("schema")
        claimed_independent = fields.pop("independent", None)
        fields["candidate_model_families"] = tuple(fields["candidate_model_families"])
        fields["prior_judge_providers"] = tuple(fields["prior_judge_providers"])
        fields["prior_judge_model_families"] = tuple(
            fields["prior_judge_model_families"]
        )
        fields["reason_codes"] = tuple(str(item) for item in fields.get("reason_codes") or ())
        evidence = cls(**fields)
        if claimed_independent is not None and claimed_independent is not evidence.independent:
            raise ValueError("serialized independence claim does not match provenance")
        return evidence


@dataclass(frozen=True)
class ExternalJudgeConfig:
    """Provider-neutral, fully pinned invocation policy."""

    provider: str
    provider_version: str
    model_family: str
    model_version: str
    rubric_id: str
    rubric_version: str
    rubric: str
    temperature: float
    seed: int | None
    max_attempts: int
    max_tokens: int
    max_cost_usd: float

    def __post_init__(self) -> None:
        for name in (
            "provider",
            "provider_version",
            "model_family",
            "model_version",
            "rubric_id",
            "rubric_version",
            "rubric",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.max_attempts < 1 or self.max_tokens < 1 or self.max_cost_usd < 0:
            raise ValueError("invalid retry, token, or cost limit")


class ExternalJudgeAdapter:
    """Invoke an injected transport without exposing provenance to the provider."""

    def __init__(
        self,
        config: ExternalJudgeConfig,
        transport: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.config = config
        self.transport = transport

    def judge(
        self,
        *,
        audit_id: str,
        record_id: str,
        generation_id: str,
        pair_id: str,
        prompt: str,
        left_openui: str,
        right_openui: str,
        left_checkpoint_sha256: str,
        right_checkpoint_sha256: str,
        candidate_model_families: tuple[str, str],
        prior_judge_providers: tuple[str, ...],
        prior_judge_model_families: tuple[str, ...],
        order_seed: int,
    ) -> JudgeEvidenceV1:
        request = {
            "model": self.config.model_version,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_tokens": self.config.max_tokens,
            "response_schema": "JudgePairResponseV1",
            "rubric": self.config.rubric,
            "pair": {
                "prompt": prompt,
                "left_openui": left_openui,
                "right_openui": right_openui,
            },
        }
        started = monotonic()
        last_error: Exception | None = None
        cost = 0.0
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = self.transport(request)
                response_cost = float(response.get("cost_usd", 0.0))
                cost += response_cost
                if cost > self.config.max_cost_usd:
                    last_error = ValueError("external judge exceeded the pinned cost limit")
                    break
                verdict = str(response["verdict"])
                refused = verdict == "refusal"
                return self._evidence(
                    audit_id=audit_id,
                    record_id=record_id,
                    generation_id=generation_id,
                    pair_id=pair_id,
                    prompt=prompt,
                    left_openui=left_openui,
                    right_openui=right_openui,
                    left_checkpoint_sha256=left_checkpoint_sha256,
                    right_checkpoint_sha256=right_checkpoint_sha256,
                    candidate_model_families=candidate_model_families,
                    prior_judge_providers=prior_judge_providers,
                    prior_judge_model_families=prior_judge_model_families,
                    order_seed=order_seed,
                    verdict=verdict,
                    left_acceptable=(
                        None if refused else _required_bool(response, "left_acceptable")
                    ),
                    right_acceptable=(
                        None if refused else _required_bool(response, "right_acceptable")
                    ),
                    score=_optional_float(response.get("score")),
                    reason_codes=tuple(str(item) for item in response.get("reason_codes") or ()),
                    confidence=_optional_float(response.get("confidence")),
                    retry_count=attempt - 1,
                    refused=refused,
                    error=None,
                    latency_ms=round((monotonic() - started) * 1000),
                    cost_usd=cost,
                )
            except Exception as exc:
                last_error = exc
        return self._evidence(
            audit_id=audit_id,
            record_id=record_id,
            generation_id=generation_id,
            pair_id=pair_id,
            prompt=prompt,
            left_openui=left_openui,
            right_openui=right_openui,
            left_checkpoint_sha256=left_checkpoint_sha256,
            right_checkpoint_sha256=right_checkpoint_sha256,
            candidate_model_families=candidate_model_families,
            prior_judge_providers=prior_judge_providers,
            prior_judge_model_families=prior_judge_model_families,
            order_seed=order_seed,
            verdict="error",
            left_acceptable=None,
            right_acceptable=None,
            score=None,
            reason_codes=("transport_or_schema_error",),
            confidence=None,
            retry_count=self.config.max_attempts - 1,
            refused=False,
            error=type(last_error).__name__ if last_error else "unknown_error",
            latency_ms=round((monotonic() - started) * 1000),
            cost_usd=cost,
        )

    def _evidence(self, **values: Any) -> JudgeEvidenceV1:
        order_sha = text_sha256(
            f"{values['order_seed']}:{values['left_checkpoint_sha256']}:{values['right_checkpoint_sha256']}"
        )
        return JudgeEvidenceV1(
            evidence_id=f"ev_{uuid.uuid4().hex}",
            judge_id=f"{self.config.provider}:{self.config.model_version}",
            provenance="external_model",
            prompt_sha256=text_sha256(values.pop("prompt")),
            left_output_sha256=text_sha256(values.pop("left_openui")),
            right_output_sha256=text_sha256(values.pop("right_openui")),
            provider=self.config.provider,
            provider_version=self.config.provider_version,
            model_family=self.config.model_family,
            model_version=self.config.model_version,
            rubric_id=self.config.rubric_id,
            rubric_version=self.config.rubric_version,
            rubric_sha256=text_sha256(self.config.rubric),
            participated_in_creation=False,
            participated_in_admission=False,
            participated_in_training=False,
            participated_in_preference=False,
            participated_in_evaluation=False,
            rubric_used_for_training_admission=False,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            blinded=True,
            order_sha256=order_sha,
            saw_candidate_identity=False,
            saw_automatic_judgments=False,
            **values,
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _required_bool(value: dict[str, Any], key: str) -> bool:
    result = value[key]
    if not isinstance(result, bool):
        raise ValueError(f"{key} must be boolean")
    return result


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_ranks(left), _ranks(right))


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    concordant = discordant = left_ties = right_ties = 0
    for first in range(len(left) - 1):
        for second in range(first + 1, len(left)):
            left_delta = left[first] - left[second]
            right_delta = right[first] - right[second]
            if left_delta == 0 and right_delta == 0:
                continue
            if left_delta == 0:
                left_ties += 1
            elif right_delta == 0:
                right_ties += 1
            elif left_delta * right_delta > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = sqrt(
        (concordant + discordant + left_ties)
        * (concordant + discordant + right_ties)
    )
    return (concordant - discordant) / denominator if denominator else None


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    labels = set(left) | set(right)
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    expected = sum(
        (left.count(label) / len(left)) * (right.count(label) / len(right))
        for label in labels
    )
    return (observed - expected) / (1 - expected) if expected < 1 else None


def fleiss_kappa(ratings: Sequence[Sequence[str]]) -> float | None:
    """Fleiss kappa for complete rows with the same number of raters."""
    if not ratings or len(ratings[0]) < 2 or any(len(row) != len(ratings[0]) for row in ratings):
        return None
    labels = sorted({label for row in ratings for label in row})
    n_raters = len(ratings[0])
    agreement = []
    totals = {label: 0 for label in labels}
    for row in ratings:
        counts = {label: row.count(label) for label in labels}
        agreement.append(sum(count * count for count in counts.values()) - n_raters)
        for label, count in counts.items():
            totals[label] += count
    observed = sum(agreement) / (len(ratings) * n_raters * (n_raters - 1))
    denominator = len(ratings) * n_raters
    expected = sum((count / denominator) ** 2 for count in totals.values())
    return (observed - expected) / (1 - expected) if expected < 1 else None


def set_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def binary_metrics(predicted: Sequence[bool], actual: Sequence[bool]) -> dict[str, float | int | None]:
    if len(predicted) != len(actual) or not predicted:
        raise ValueError("binary metric inputs must have equal non-zero length")
    tp = sum(p and a for p, a in zip(predicted, actual))
    fp = sum(p and not a for p, a in zip(predicted, actual))
    fn = sum(not p and a for p, a in zip(predicted, actual))
    return {
        "n": len(actual),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def bootstrap_pairwise_ci(
    left: Sequence[float],
    right: Sequence[float],
    metric: Callable[[Sequence[float], Sequence[float]], float | None],
    *,
    seed: int = 0,
    resamples: int = 1_000,
) -> dict[str, float | int | None]:
    """Deterministic paired bootstrap interval for a rank statistic."""
    if len(left) != len(right) or len(left) < 2 or resamples < 1:
        raise ValueError("bootstrap inputs require equal pairs and positive resamples")
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        indices = [rng.randrange(len(left)) for _ in left]
        estimate = metric(
            [left[index] for index in indices],
            [right[index] for index in indices],
        )
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        return {"estimate": metric(left, right), "low": None, "high": None, "resamples": 0}
    estimates.sort()
    low = estimates[int(0.025 * (len(estimates) - 1))]
    high = estimates[int(0.975 * (len(estimates) - 1))]
    return {
        "estimate": metric(left, right),
        "low": low,
        "high": high,
        "resamples": len(estimates),
    }


def judge_evidence_agentv_cases(
    evidence: Sequence[JudgeEvidenceV1],
) -> list[dict[str, Any]]:
    """Lower provenance checks to AgentV without calling them semantic scores."""
    cases = []
    for row in evidence:
        failures = []
        if row.provenance == "external_model" and not row.independent:
            failures.append("external evidence failed an independence provenance gate")
        if row.verdict in {"error", "refusal"}:
            failures.append(f"judge outcome was {row.verdict}")
        cases.append(
            {
                "id": row.evidence_id,
                "criteria": "Evidence is schema-valid, attributable, and independent when declared external.",
                "pass": not failures,
                "failures": failures,
                "result": row.to_dict(),
                "metadata": {
                    "audit_id": row.audit_id,
                    "pair_id": row.pair_id,
                    "provenance": row.provenance,
                    "semantic_judge": False,
                },
            }
        )
    return cases


def publish_judge_evidence_agentv(
    run_dir: str,
    evidence: Sequence[JudgeEvidenceV1],
) -> dict[str, Any]:
    """Publish the evidence-integrity envelope with the pinned AgentV SDK."""
    from slm_training.evals.agentv import publish_agentv_evaluation

    return publish_agentv_evaluation(
        run_dir,
        name=f"judge-evidence-{evidence[0].audit_id}" if evidence else "judge-evidence",
        claim="judge_evidence_integrity_not_semantic_quality",
        cases=judge_evidence_agentv_cases(evidence),
    )


def calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], *, bins: int = 10
) -> float | None:
    if len(confidences) != len(correct) or not confidences:
        return None
    if bins < 1 or any(not 0 <= value <= 1 for value in confidences):
        raise ValueError("invalid calibration inputs")
    total = len(confidences)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, value in enumerate(confidences)
            if lower <= value <= upper and (bin_index == bins - 1 or value < upper)
        ]
        if members:
            accuracy = sum(correct[index] for index in members) / len(members)
            confidence = sum(confidences[index] for index in members) / len(members)
            error += len(members) / total * abs(accuracy - confidence)
    return error


def wilson_interval(successes: int, n: int, *, z: float = 1.959963984540054) -> dict[str, float | int]:
    if n < 1 or not 0 <= successes <= n:
        raise ValueError("Wilson interval requires 0 <= successes <= n")
    rate = successes / n
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    margin = z * sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return {"n": n, "estimate": rate, "low": center - margin, "high": center + margin}


def analyze_triple_judges(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build the publishable agreement report from aligned per-pair decisions."""
    if not rows:
        raise ValueError("triple-judge analysis requires at least one pair")
    pair_ids = [str(row.get("pair_id") or "") for row in rows]
    if "" in pair_ids or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("analysis rows require unique pair_id values")
    required = {
        f"{judge}_{field}"
        for judge in ("deterministic", "external", "human")
        for field in ("score", "verdict", "pass")
    }
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"analysis row {row['pair_id']} is missing {sorted(missing)}")
    full = _agreement_slice(rows)
    unambiguous_rows = [row for row in rows if row.get("human_ambiguous") is not True]
    disagreements = [
        row
        for row in rows
        if len(
            {
                str(row[f"{judge}_verdict"])
                for judge in ("deterministic", "external", "human")
            }
        )
        > 1
    ]
    costs = [float(row.get("external_cost_usd") or 0.0) for row in rows]
    latencies = [
        float(row["external_latency_ms"])
        for row in rows
        if row.get("external_latency_ms") is not None
    ]
    return {
        "schema": "TripleJudgeAnalysisV1",
        "pair_n": len(rows),
        "full": full,
        "excluding_ambiguous_human_pairs": (
            _agreement_slice(unambiguous_rows) if unambiguous_rows else None
        ),
        "ambiguous_pair_n": len(rows) - len(unambiguous_rows),
        "disagreement": {
            "pair_n": len(disagreements),
            "reason_clusters": dict(
                sorted(
                    Counter(
                        str(reason)
                        for row in disagreements
                        for reason in row.get("reason_codes") or []
                    ).items()
                )
            ),
            "checkpoint_family": dict(
                sorted(Counter(str(row.get("checkpoint_family") or "unknown") for row in disagreements).items())
            ),
            "suite": dict(
                sorted(Counter(str(row.get("suite") or "unknown") for row in disagreements).items())
            ),
        },
        "external_operations": {
            "total_cost_usd": sum(costs),
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "refusal_n": sum(row.get("external_verdict") == "refusal" for row in rows),
            "error_n": sum(row.get("external_verdict") == "error" for row in rows),
        },
    }


def _agreement_slice(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    judges = ("deterministic", "external", "human")
    pairwise: dict[str, Any] = {}
    for left_judge, right_judge in (
        ("deterministic", "external"),
        ("deterministic", "human"),
        ("external", "human"),
    ):
        score_rows = [
            row
            for row in rows
            if row[f"{left_judge}_score"] is not None
            and row[f"{right_judge}_score"] is not None
        ]
        left_scores = [float(row[f"{left_judge}_score"]) for row in score_rows]
        right_scores = [float(row[f"{right_judge}_score"]) for row in score_rows]
        left_verdicts = [str(row[f"{left_judge}_verdict"]) for row in rows]
        right_verdicts = [str(row[f"{right_judge}_verdict"]) for row in rows]
        left_pass = {
            str(row["pair_id"]) for row in rows if row[f"{left_judge}_pass"] is True
        }
        right_pass = {
            str(row["pair_id"]) for row in rows if row[f"{right_judge}_pass"] is True
        }
        agreement_n = sum(a == b for a, b in zip(left_verdicts, right_verdicts))
        key = f"{left_judge}_vs_{right_judge}"
        pairwise[key] = {
            "score_pair_n": len(score_rows),
            "spearman": spearman(left_scores, right_scores),
            "kendall_tau_b": kendall_tau_b(left_scores, right_scores),
            "kendall_tau_b_ci": (
                bootstrap_pairwise_ci(
                    left_scores, right_scores, kendall_tau_b, seed=0, resamples=1_000
                )
                if len(score_rows) >= 2
                else None
            ),
            "cohen_kappa": cohen_kappa(left_verdicts, right_verdicts),
            "pass_set_jaccard": set_jaccard(left_pass, right_pass),
            "admission_divergence_rate": len(left_pass ^ right_pass) / len(rows),
            "verdict_agreement_ci": wilson_interval(agreement_n, len(rows)),
        }
    valid_external = [
        row for row in rows if row["external_verdict"] not in {"error", "refusal"}
    ]
    human_pass = [bool(row["human_pass"]) for row in valid_external]
    external_pass = [bool(row["external_pass"]) for row in valid_external]
    calibrated = [
        row for row in valid_external if row.get("external_confidence") is not None
    ]
    return {
        "pair_n": len(rows),
        "pairwise": pairwise,
        "fleiss_kappa": fleiss_kappa(
            [[str(row[f"{judge}_verdict"]) for judge in judges] for row in rows]
        ),
        "external_vs_human": {
            **(
                binary_metrics(external_pass, human_pass)
                if valid_external
                else {"n": 0, "precision": None, "recall": None}
            ),
            "calibration_error": calibration_error(
                [float(row["external_confidence"]) for row in calibrated],
                [
                    bool(row["external_pass"]) == bool(row["human_pass"])
                    for row in calibrated
                ],
            ),
        },
    }
