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

Observed metric SD: pooled from the committed darkfactory phase 1
evidence ledger (``resources/experiments/autotrain_climb/
evidence_ledger.v1.json``) — per-arm Welford ``m2_delta`` / ``n_delta``
of the loop primary metric's candidate-minus-control deltas, which is
exactly the SD of paired differences the paired MDE needs.  The ledger
does not record a metric name; it is the loop's screening primary
(``smoke.structural_similarity`` per ``policy.v1.json``), so the pooled
SD is used only when the candidate's ``endpoint_metric`` is that metric
and the ledger carries enough degrees of freedom.  Otherwise a
documented conservative default applies: ``sqrt(0.25 / 3) ~= 0.2887``,
the worst-case SD of a mean over ``screening_smoke_n = 3`` documents
whose per-document contribution is binary (RC1: the screening metrics
move in quanta of ~1/3 per flipped document).  Conservative means large:
a larger SD inflates the MDE, so unknown-variance designs block rather
than slip through.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from slm_training.autoresearch.power import is_decidable

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
#: documents: sqrt(0.25 / 3).  Used when the ledger SD is not applicable.
CONSERVATIVE_DEFAULT_SD = math.sqrt(0.25 / 3.0)
#: Minimum pooled degrees of freedom before the ledger SD counts as usable.
MIN_LEDGER_DOF = 20
#: Ceiling on ``required_n_for_effect`` beyond which a design is treated as
#: structurally undecidable (block) rather than merely still-accumulating
#: (warn). Generous relative to ``ADEQUATE_POWER_MIN_SEEDS`` (8, see
#: ``preflight/prior_attempts.py`` and the conclusion-policy default of 8)
#: so ordinary accumulation is never blocked, only designs whose minimum
#: effect is unreachable at any realistic n.
MAX_REASONABLE_N = 64
#: The ledger's deltas are measurements of the loop screening primary.
LEDGER_METRIC_LEAF = "structural_similarity"
#: Process / heal / wiring arms execute a local successor (new snapshot,
#: rebuild_data resume). They are not confirmatory climb designs — a
#: first-cycle n=1 is expected and must not be blocked by RC1 power.
PROCESS_CLAIM_CLASSES = frozenset({"process", "heal", "wiring"})

DEFAULT_LEDGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "experiments"
    / "autotrain_climb"
    / "evidence_ledger.v1.json"
)


def _pooled_ledger_sd(ledger_path: Path) -> tuple[float, int] | None:
    """Pooled SD of per-run deltas across arms: sqrt(sum m2 / sum (n-1)).

    Returns ``(sd, dof)`` or None when the ledger is missing, unreadable,
    or carries fewer than ``MIN_LEDGER_DOF`` degrees of freedom.
    """
    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    arms = payload.get("arms")
    if not isinstance(arms, Mapping):
        return None
    m2_total = 0.0
    dof = 0
    for arm in arms.values():
        if not isinstance(arm, Mapping):
            continue
        m2 = arm.get("m2_delta")
        n_delta = arm.get("n_delta")
        if not isinstance(m2, (int, float)) or not isinstance(n_delta, int):
            continue
        if not math.isfinite(float(m2)) or float(m2) < 0 or n_delta < 2:
            continue
        m2_total += float(m2)
        dof += n_delta - 1
    if dof < MIN_LEDGER_DOF:
        return None
    sd = math.sqrt(m2_total / dof)
    if not math.isfinite(sd) or sd <= 0:
        return None
    return sd, dof


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

    def __init__(self, ledger_path: Path = DEFAULT_LEDGER_PATH) -> None:
        self.ledger_path = ledger_path

    def _observed_sd(self, endpoint_metric: str) -> tuple[float, str]:
        if LEDGER_METRIC_LEAF in endpoint_metric:
            pooled = _pooled_ledger_sd(self.ledger_path)
            if pooled is not None:
                sd, dof = pooled
                return sd, f"evidence_ledger.v1.json pooled delta sd (dof={dof})"
        return (
            CONSERVATIVE_DEFAULT_SD,
            "conservative default sqrt(0.25/3) (no applicable ledger variance)",
        )

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
        endpoint_metric = str(candidate.get("endpoint_metric") or "")
        sd, sd_source = self._observed_sd(endpoint_metric)

        report = is_decidable(
            n=n_seeds,
            sd=sd,
            minimum_effect=minimum_effect,
            alpha=ALPHA,
            paired=PAIRED,
        )
        reasons.extend(report.reasons)
        reasons.append(f"sd={sd:.6g} from {sd_source}")
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
            "min_attainable_p": report.min_attainable_p,
            "min_detectable_effect": report.min_detectable_effect,
            "required_n_for_effect": report.required_n_for_effect,
            "max_reasonable_n": MAX_REASONABLE_N,
        }
        if report.decidable:
            verdict = "pass"
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
