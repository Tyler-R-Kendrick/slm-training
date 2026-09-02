"""Power/decidability preflight plugin: block undecidable screening designs.

Implements the ``§4-R1`` successor approach to RC1 of
``docs/design/harness-evolution-architecture-review-20260809.md``: a
candidate experiment is inadmissible unless (a) its decision statistic
can reject at alpha at all at the declared ``n_seeds``, and (b) its
minimum detectable effect at that budget is <= its preregistered
``minimum_effect``.  Computable pre-run; fails closed (malformed or
missing candidate fields block with an explanatory reason — this check
never raises).  Per the preflight package's fail-soft law, an internal
check *bug* yields ``"warn"``, never a silent block: only a genuine
decidability determination (or unusable candidate data) blocks.

The check evaluates the **paired** variant (``paired=True``) because the
climb loop's screening decision statistic is the paired sign test over
signed per-document effects (``credit_engine._two_sided_p_from_signed_effects``);
see ``slm_training.autoresearch.power`` for the paired vs unpaired
reconciliation.

Seeds-policy reconciliation (post-launch fix): the continuous loop spends
one seed per screening cycle and accumulates evidence for an arm across
many cycles (``resources/experiments/autotrain_climb/evidence_ledger.v1.json``
``n_complete``). ``scripts/run_autotrain_continuous.py::_preflight_screening_slug``
therefore passes the arm's *cumulative* seed count (ledger ``n_complete + 1``,
projecting the seed this cycle would add), not the marginal contribution of
a single cycle — a literal ``n_seeds=1`` is undecidable by construction
(``min_attainable_p`` at n=1 is 1.0) and would block every screening cycle
forever, which would prevent the very accumulation the fix requires. Given
a cumulative n below ``required_n_for_effect``, the verdict is ``"warn"``
(still accumulating, not yet decided) rather than ``"block"``, UNLESS
``required_n_for_effect`` exceeds ``MAX_REASONABLE_N`` — i.e. the design
cannot become decidable through any realistic amount of further
accumulation, which is a genuine RC1-style structural block.

Observed metric SD: resolved per endpoint metric through
``screening_sample_size.lookup_paired_sd_for_metric`` — the
``metric_expectations.screening.v1.json`` ``observed_paired_sd_by_metric``
slot, then the committed evidence ledger (``resources/experiments/
autotrain_climb/evidence_ledger.v1.json``: per-arm Welford ``m2_delta`` /
``n_delta`` of candidate-minus-control deltas, exactly the SD of paired
differences the paired MDE needs) when its stats are keyed to the metric
(the legacy ledger is untagged and counts for the v1 screening primary
``smoke.structural_similarity`` only) with enough degrees of freedom, then
the module's tagged measured constant.  Another metric's SD is never
borrowed.  When no same-metric SD exists the SD is *unmeasured*: the check
still reports the MDE under the documented conservative default
``sqrt(0.25 / 3) ~= 0.2887`` (worst-case SD of a mean over
``screening_smoke_n = 3`` binary-quantized documents) as an advisory
number, but an unmeasured SD can only ``warn`` (``sd_source="unmeasured"``)
— a ``block`` is a measured determination.  RC2: the previous rule blocked
every confirmatory ``smoke.eval_nll`` arm from the conservative default
(``required_n=262 > 64``) although nothing about that metric was measured.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from slm_training.autoresearch.power import is_decidable
from slm_training.autoresearch.screening_sample_size import (
    DEFAULT_EVIDENCE_LEDGER_PATH,
    MIN_LEDGER_DOF,
    lookup_paired_sd_for_metric,
    metric_leaf,
    pooled_ledger_paired_sd,
)

try:  # Contract owner: preflight/__init__.py (may not exist yet).
    from . import PreflightVerdict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - exercised when __init__ is absent

    from pydantic import BaseModel, ConfigDict, Field

    class _PreflightVerdict(BaseModel):
        """Structural stand-in for the preflight package's verdict model."""

        model_config = ConfigDict(extra="allow")

        check_id: str
        verdict: str
        reasons: list[str] = Field(default_factory=list)
        data: dict[str, Any] = Field(default_factory=dict)

    PreflightVerdict = _PreflightVerdict  # type: ignore[misc]

CHECK_ID = "power_decidability"
ALPHA = 0.05
#: The loop's screening decision statistic is a paired sign test.
PAIRED = True
#: Fallback when the candidate omits minimum_effect: the climb policy's
#: screening_primary.minimum_effect (policy.v1.json).
DEFAULT_MINIMUM_EFFECT = 0.01
#: Worst-case SD of a mean over screening_smoke_n=3 binary-quantized
#: documents: sqrt(0.25 / 3).  Advisory only: reported when the endpoint
#: metric's paired SD is unmeasured, never grounds for a block.
CONSERVATIVE_DEFAULT_SD = math.sqrt(0.25 / 3.0)
#: ``sd_source`` when no same-metric paired SD exists.
SD_SOURCE_UNMEASURED = "unmeasured"
#: Re-exported: minimum pooled degrees of freedom for a usable ledger SD.
MIN_LEDGER_DOF = MIN_LEDGER_DOF
#: Ceiling on ``required_n_for_effect`` beyond which a design is treated as
#: structurally undecidable (block) rather than merely still-accumulating
#: (warn). Generous relative to ``ADEQUATE_POWER_MIN_SEEDS`` (8, see
#: ``preflight/prior_attempts.py`` and the conclusion-policy default of 8)
#: so ordinary accumulation is never blocked, only designs whose minimum
#: effect is unreachable at any realistic n.
MAX_REASONABLE_N = 64
#: The legacy ledger's deltas are measurements of the v1 screening primary
#: (kept for callers; the metric-keyed rule lives in screening_sample_size).
LEDGER_METRIC_LEAF = "structural_similarity"
#: Process / heal / wiring arms execute a local successor (new snapshot,
#: rebuild_data resume). They are not confirmatory climb designs — a
#: first-cycle n=1 is expected and must not be blocked by RC1 power.
PROCESS_CLAIM_CLASSES = frozenset({"process", "heal", "wiring"})

DEFAULT_LEDGER_PATH = DEFAULT_EVIDENCE_LEDGER_PATH


def _pooled_ledger_sd(ledger_path: Path) -> tuple[float, int] | None:
    """Pooled SD of per-run deltas across all arms: sqrt(sum m2 / sum (n-1)).

    Returns ``(sd, dof)`` or None when the ledger is missing, unreadable,
    or carries fewer than ``MIN_LEDGER_DOF`` degrees of freedom. Metric-agnostic
    (legacy pooled estimate); the verdict path uses the metric-keyed lookup.
    """
    return pooled_ledger_paired_sd(ledger_path)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value) if float(value) > 0 else None
    return None


def _is_process_candidate(candidate: Mapping[str, Any]) -> bool:
    """True when the candidate is a process/heal arm, not a confirmatory screen."""
    if candidate.get("process_arm") or candidate.get("heal_resume"):
        return True
    claim = candidate.get("claim_class")
    return isinstance(claim, str) and claim in PROCESS_CLAIM_CLASSES


def _make_verdict(
    verdict: str, reasons: list[str], data: dict[str, Any]
) -> PreflightVerdict:  # type: ignore[valid-type]
    """Construct a verdict, tolerating a contract model without ``data``."""
    try:
        return PreflightVerdict(
            check_id=CHECK_ID, verdict=verdict, reasons=reasons, data=data
        )
    except Exception:  # noqa: BLE001 - structural tolerance, never raise
        return PreflightVerdict(check_id=CHECK_ID, verdict=verdict, reasons=reasons)


class PowerDecidabilityCheck:
    """Preflight check blocking statistically undecidable screening designs."""

    check_id = CHECK_ID

    def __init__(
        self,
        ledger_path: Path = DEFAULT_LEDGER_PATH,
        expectations_path: Path | None = None,
    ) -> None:
        self.ledger_path = ledger_path
        # None -> the committed metric_expectations.screening.v1.json.
        self.expectations_path = expectations_path

    def _observed_sd(self, endpoint_metric: str) -> tuple[float, str, bool]:
        """``(sd, sd_source, measured)`` for the endpoint metric, never borrowed."""
        lookup = lookup_paired_sd_for_metric(
            endpoint_metric,
            expectations_path=self.expectations_path,
            ledger_path=self.ledger_path,
        )
        if lookup.measured and lookup.observed_sd is not None:
            sd = float(Fraction(lookup.observed_sd))
            detail = ", ".join(f"{k}={v}" for k, v in sorted(lookup.detail.items()))
            return sd, f"{lookup.source} ({detail})", True
        return CONSERVATIVE_DEFAULT_SD, SD_SOURCE_UNMEASURED, False

    def run(self, candidate: dict) -> PreflightVerdict:  # type: ignore[valid-type]
        try:
            return self._run(candidate)
        except Exception as exc:  # noqa: BLE001 - preflight must never raise
            # Package fail-soft law: a check bug must never silently block.
            return _make_verdict(
                "warn",
                [
                    "power_decidability check errored; a bug is not evidence: "
                    f"{type(exc).__name__}: {exc}"
                ],
                {},
            )

    def _run(self, candidate: dict) -> PreflightVerdict:  # type: ignore[valid-type]
        if not isinstance(candidate, Mapping):
            return _make_verdict(
                "block",
                [
                    "candidate must be a mapping with n_seeds/minimum_effect/"
                    f"endpoint_metric; got {type(candidate).__name__}"
                ],
                {},
            )
        reasons: list[str] = []
        # ``n_seeds`` is the arm's *cumulative* seed count (ledger n_complete
        # projected forward by this cycle), not this cycle's marginal
        # contribution — see the module docstring's seeds-policy note.
        n_seeds = _positive_int(candidate.get("n_seeds"))
        if n_seeds is None:
            return _make_verdict(
                "block",
                [
                    "candidate has no usable n_seeds "
                    f"(got {candidate.get('n_seeds')!r}); cannot establish "
                    "decidability, failing closed"
                ],
                {"candidate_n_seeds": repr(candidate.get("n_seeds"))},
            )
        minimum_effect = _positive_float(candidate.get("minimum_effect"))
        if minimum_effect is None:
            minimum_effect = DEFAULT_MINIMUM_EFFECT
            reasons.append(
                "candidate has no usable minimum_effect; assuming the climb "
                f"policy screening default {DEFAULT_MINIMUM_EFFECT}"
            )
        endpoint_metric = str(candidate.get("endpoint_metric") or "").strip()
        if not endpoint_metric:
            return _make_verdict(
                "block",
                [
                    "candidate has no usable endpoint_metric "
                    f"(got {candidate.get('endpoint_metric')!r}); the paired "
                    "SD is metric-keyed, failing closed"
                ],
                {"candidate_endpoint_metric": repr(candidate.get("endpoint_metric"))},
            )
        sd, sd_source, sd_measured = self._observed_sd(endpoint_metric)

        report = is_decidable(
            n=n_seeds,
            sd=sd,
            minimum_effect=minimum_effect,
            alpha=ALPHA,
            paired=PAIRED,
        )
        reasons.extend(report.reasons)
        if sd_measured:
            reasons.append(f"sd={sd:.6g} from {sd_source}")
        else:
            reasons.append(
                f"paired SD for {endpoint_metric} is unmeasured; advisory "
                f"conservative default sd={sd:.6g} (sqrt(0.25/3)) reported "
                "only — record observed_paired_sd_by_metric["
                f"{metric_leaf(endpoint_metric)}] in "
                "metric_expectations.screening.v1.json to power this design"
            )
        details = {
            "n_seeds": n_seeds,
            "steps": candidate.get("steps"),
            "endpoint_metric": endpoint_metric,
            "config_fingerprint": candidate.get("config_fingerprint"),
            "alpha": ALPHA,
            "paired": PAIRED,
            "minimum_effect": minimum_effect,
            "sd": sd,
            "sd_source": sd_source,
            "sd_measured": sd_measured,
            "min_attainable_p": report.min_attainable_p,
            "min_detectable_effect": report.min_detectable_effect,
            "required_n_for_effect": report.required_n_for_effect,
            "max_reasonable_n": MAX_REASONABLE_N,
        }
        if report.decidable:
            verdict = "pass"
        elif not sd_measured:
            # An unmeasured SD is not evidence of undecidability: warn only.
            verdict = "warn"
            reasons.append(
                "power cannot block on an unmeasured SD; verdict is advisory "
                f"(required_n_for_effect={report.required_n_for_effect} under "
                "the conservative default is not a measured determination)"
            )
        elif report.required_n_for_effect > MAX_REASONABLE_N:
            verdict = "block"
            reasons.append(
                f"required_n_for_effect={report.required_n_for_effect} exceeds "
                f"the reasonable screening ceiling ({MAX_REASONABLE_N}); this "
                "design cannot become decidable through realistic accumulation"
            )
        else:
            verdict = "warn"
            reasons.append(
                f"not yet decidable at cumulative n_seeds={n_seeds}; still "
                f"accumulating toward required_n_for_effect="
                f"{report.required_n_for_effect}"
            )
        if verdict == "block" and _is_process_candidate(candidate):
            reasons.append(
                "process/heal arm is not a confirmatory design; power "
                "decidability warns instead of blocking first execution"
            )
            details["process_arm"] = True
            verdict = "warn"
        return _make_verdict(verdict, reasons, details)


#: Module-level plugin object discovered by the preflight registry.
CHECK = PowerDecidabilityCheck()
