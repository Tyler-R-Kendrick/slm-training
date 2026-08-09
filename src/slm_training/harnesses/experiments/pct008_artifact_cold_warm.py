"""PCT-008 (SLM-464): fixture-scale cold/warm artifact-parity evidence campaign.

Extends the shipped harnesses this issue is explicitly a follow-on to,
rather than reimplementing them: the PCT-003 subprocess cold/warm bench
(:mod:`slm_training.harnesses.model_build.cold_warm_bench`) supplies real
fresh-process trials, and the PCT-005/PCT-006 completion-artifact seam
(:mod:`slm_training.dsl.grammar.fastpath.completion_artifact`,
:mod:`slm_training.dsl.grammar.fastpath.static_control_domain`) supplies the
four arm code paths measured here.

Scope, deliberately narrower than the issue's full "Measurements" list:
this campaign measures the *artifact seam* only -- process launch,
tokenizer/pack construction, the arm-specific artifact step, and the first
completion-domain query. Model init/first-forward and full
first-verified-program latency are out of scope for this fixture-scale
pass (no model is loaded here at all) -- reported explicitly as such in the
results payload, never silently omitted. ``claim_class="fixture"``: no
model-quality or production-speedup claim is made.

Arms (matching the issue's list):

* ``no_artifact`` (control) -- exact live construction via the Lark
  reference oracle, no static artifact touched at all.
* ``certified_direct`` -- ``load_checked_completion_artifact`` called
  directly, bypassing the provider abstraction.
* ``provider_mediated`` -- the real production entry point:
  ``require_certified_static_projection`` (which resolves the provider via
  ``provider_for_pack``) followed by the packed completion-domain query.
* ``deliberate_miss`` -- a provider pointed at a nonexistent artifact path;
  ``try_load`` must fail closed (return ``None``, never raise) and the
  caller must fall back to exact live construction.
* ``warm_repeated`` -- an in-process steady-state control: N repeated
  completion-domain queries against the same request-local session
  (mirroring the E9 ``request_local_memo_reuse`` convention in
  ``docs/design/certified-completion-artifact-and-tps-target.md`` --
  never "AOT cold"). Measured, not assumed: this arm does *not* by itself
  demonstrate a memo-reuse speedup, because ``production_domain_snapshot``'s
  default ``require_artifact=True`` re-verifies the full static artifact
  (sha256 over the artifact bytes, grammar file, and control tensors) on
  every call, dominating per-call cost regardless of session reuse (AC:
  "Artifact load/check overhead is visible" -- confirmed visible precisely
  because it is not amortized away here).

Parity (AC: "All semantic/legal outputs are parity-clean before
performance interpretation") is checked once, cheaply, in-process, before
any subprocess is spawned: every arm's completion-domain candidate set at a
fixed corpus prefix must exactly match the live Lark oracle. A parity
failure aborts the campaign before any latency trial runs.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
)
from slm_training.autoresearch.schemas import CampaignBudget, CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.model_build.cold_warm_bench import (
    BenchTrialV1,
    run_cold_trial,
    run_warm_trial,
    summarize_trials,
)
from slm_training.levers import MAX_RUN_MINUTES

__all__ = [
    "COLD_ARMS",
    "FIXED_PREFIX_SURFACE",
    "WARM_ARM",
    "Pct008CampaignV1",
    "check_domain_parity",
    "run_campaign",
]

# The first program in the canonical E1 corpus (test_static_control_domain.py,
# static_control_domain.STATIC_LALR_CORPUS) -- already proven, by that
# suite, to produce exact production/oracle parity, so this campaign's own
# parity check is expected to confirm a known-good fact rather than probe
# unknown territory.
FIXED_PREFIX_SURFACE = 'root = TextContent(":slot_0")\n'

COLD_ARMS: tuple[str, ...] = (
    "no_artifact",
    "certified_direct",
    "provider_mediated",
    "deliberate_miss",
)
WARM_ARM = "warm_repeated"

_NOISE_FLOOR_MS = 5.0


def _prefix_ids_snippet() -> str:
    return f"[tok.bos_id, *tok.encode({FIXED_PREFIX_SURFACE!r}, add_special=False)]"


def _script_for_arm(arm: str) -> str:
    """The fresh-subprocess script for one cold arm (real code, not a mock)."""

    if arm not in COLD_ARMS:
        raise ValueError(f"unknown cold arm: {arm!r}")
    prefix = _prefix_ids_snippet()
    header = (
        "from slm_training.harnesses.model_build.cold_warm_bench import phase_marker\n"
        "print(phase_marker('process_launch'))\n"
        "from slm_training.models.dsl_tokenizer import DSLNativeTokenizer\n"
        "tok = DSLNativeTokenizer.build()\n"
        "print(phase_marker('tokenizer_pack_construction'))\n"
    )
    if arm == "no_artifact":
        artifact_step = ""
        query = (
            "from slm_training.dsl.grammar.fastpath.static_control_domain import "
            "oracle_lark_domain_snapshot\n"
            f"oracle_lark_domain_snapshot({prefix}, tok, budget=32)\n"
        )
    elif arm == "certified_direct":
        artifact_step = (
            "from slm_training.dsl.grammar.fastpath.completion_artifact import "
            "load_checked_completion_artifact\n"
            "from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine\n"
            "eng = OpenUIIncrementalEngine()\n"
            "load_checked_completion_artifact(tok, eng._parser, grammar_path=eng.grammar_path)\n"
        )
        query = (
            "from slm_training.dsl.grammar.fastpath.static_control_domain import "
            "production_domain_snapshot\n"
            f"production_domain_snapshot({prefix}, tok, budget=32)\n"
        )
    elif arm == "provider_mediated":
        artifact_step = (
            "from slm_training.dsl.grammar.fastpath.static_control_domain import "
            "require_certified_static_projection\n"
            "require_certified_static_projection(tok)\n"
        )
        query = (
            "from slm_training.dsl.grammar.fastpath.static_control_domain import "
            "production_domain_snapshot\n"
            f"production_domain_snapshot({prefix}, tok, budget=32)\n"
        )
    else:  # deliberate_miss
        artifact_step = (
            "from pathlib import Path\n"
            "from slm_training.dsl.grammar.fastpath.completion_artifact import "
            "StaticCompletionArtifactProvider\n"
            "from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine\n"
            "eng = OpenUIIncrementalEngine()\n"
            "missing = StaticCompletionArtifactProvider(\n"
            "    pack_id='openui',\n"
            "    artifact_path=Path('/nonexistent/openui_completion_v1.safetensors'),\n"
            "    manifest_path=Path('/nonexistent/openui_completion_v1.manifest.json'),\n"
            ")\n"
            "artifact, reason = missing.try_load(tok, eng._parser, grammar_path=eng.grammar_path)\n"
            "assert artifact is None and reason, 'deliberate miss must fail closed, never raise'\n"
        )
        query = (
            "from slm_training.dsl.grammar.fastpath.static_control_domain import "
            "oracle_lark_domain_snapshot\n"
            f"oracle_lark_domain_snapshot({prefix}, tok, budget=32)  # safe fallback\n"
        )
    body = (
        header
        + artifact_step
        + "print(phase_marker('artifact_step'))\n"
        + query
        + "print(phase_marker('first_domain_query'))\n"
    )
    return body


def check_domain_parity(tokenizer: Any) -> dict[str, Any]:
    """The one-shot, in-process correctness gate run before any timing trial.

    Every arm's candidate output at :data:`FIXED_PREFIX_SURFACE` must match
    the live Lark oracle exactly (AC: parity-clean before performance
    interpretation). ``deliberate_miss``'s fallback path is oracle-equal by
    construction (it *is* the oracle path); ``certified_direct`` and
    ``provider_mediated`` build candidates through the same packed
    completion-domain pipeline, so both are checked against production
    parity once.
    """
    from slm_training.dsl.grammar.fastpath.static_control_domain import (
        compare_domain_parity,
        oracle_lark_domain_snapshot,
        production_domain_snapshot,
    )

    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode(FIXED_PREFIX_SURFACE, add_special=False),
    ]
    oracle = oracle_lark_domain_snapshot(prefix, tokenizer, budget=32)
    production = production_domain_snapshot(prefix, tokenizer, budget=32)
    report = compare_domain_parity(production, oracle)
    return {
        "equal": bool(report.equal),
        "reasons": list(report.reasons),
        "oracle_candidate_count": len(oracle.candidates),
        "production_candidate_count": len(production.candidates),
    }


@dataclass(frozen=True)
class Pct008CampaignV1:
    campaign_id: str = "pct008-artifact-cold-warm-parity"
    cold_trials_per_arm: int = 2
    warm_trials: int = 5
    seed: int = 0
    source_commit: str = "0" * 40
    claim_class: str = "fixture"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)

    def __post_init__(self) -> None:
        if self.cold_trials_per_arm < 1:
            raise ValueError("cold_trials_per_arm must be >= 1")
        if self.warm_trials < 1:
            raise ValueError("warm_trials must be >= 1")
        if self.claim_class != "fixture":
            raise ValueError("PCT-008 only exposes the fixture evidence campaign")
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")

    @property
    def fingerprint(self) -> str:
        return content_sha(asdict(self))

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        arms = tuple(
            CampaignArmV1(
                arm_id=arm,
                role="control" if arm == "no_artifact" else "candidate",
                config_sha256=content_sha({"campaign": fingerprint, "arm": arm}),
            )
            for arm in (*COLD_ARMS, WARM_ARM)
        )
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=self.campaign_id,
            hypothesis=(
                "The provider-mediated static-artifact seam adds visible, "
                "measurable load/check overhead versus exact live construction "
                "without changing any legal/semantic output, and a deliberate "
                "artifact miss falls back to exact live construction rather "
                "than failing or silently widening legality."
            ),
            decision=(
                "Report cold/warm artifact-seam overhead honestly; no "
                "production-speedup or model-quality claim is made here."
            ),
            endpoints=(
                CampaignEndpointV1(
                    endpoint_id="artifact_step_p50_ms",
                    metric="artifact_step_phase_p50_ms",
                    role="primary",
                    direction="decrease",
                    minimum_effect=_NOISE_FLOOR_MS,
                ),
            ),
            arms=arms,
            seeds=(self.seed,),
            budget=CampaignBudget(
                max_experiments=self.cold_trials_per_arm * len(COLD_ARMS)
                + self.warm_trials,
                max_wall_minutes=self.max_wall_minutes,
            ),
            stopping_rules=(
                "Abort before any timing trial if the domain-parity check fails.",
                "Reject any timed-out or non-zero-exit cold trial; only complete trials enter a summary.",
            ),
            controls=(
                CampaignControlV1(
                    control_id="matched_prefix_control",
                    description=(
                        "Every arm queries the identical fixed corpus prefix "
                        "at the same budget, so arm timing differences cannot "
                        "be attributed to differing request shape."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="deliberate_miss_fallback_control",
                    description=(
                        "A provider pointed at a nonexistent artifact must "
                        "fail closed (try_load returns None, never raises) "
                        "and fall back to exact live construction."
                    ),
                    kind="negative",
                ),
            ),
            negative_controls=("deliberate_miss_fallback_control",),
            multiplicity_families=(
                MultiplicityFamilyV1(
                    family_id="artifact_seam_overhead",
                    hypothesis_ids=("artifact_step_p50_ms",),
                    alpha=0.05,
                ),
            ),
            promotion_gates=(
                # Fixture evidence is never a promotion claim; this gate is
                # deliberately permissive (mirrors slm287_power_protocol.py's
                # "diagnostic_only" pattern) rather than a real promotion bar.
                CampaignGateV1(
                    gate_id="fixture_only",
                    endpoint_id="artifact_step_p50_ms",
                    operator="le",
                    threshold=1.0e9,
                ),
            ),
            rollback_gates=(
                # A genuine, non-vacuous anomaly guard, not a placeholder:
                # any real artifact-step measurement is far below this: a
                # value above it means a trial hung or a stage measured the
                # wrong boundary, not a real "artifact seam is slow" result.
                CampaignGateV1(
                    gate_id="artifact_step_anomalously_slow",
                    endpoint_id="artifact_step_p50_ms",
                    operator="gt",
                    threshold=100_000.0,
                ),
            ),
            artifact_requirements=(
                ArtifactRequirementV1(kind="version_stamp"),
                ArtifactRequirementV1(kind="endpoint_result"),
            ),
            claim_class=self.claim_class,
            source_commit=self.source_commit,
            source_dirty=False,
            author="slm-training",
            created_at="1970-01-01T00:00:00Z",
        )


def _warm_repeated_trial(
    tokenizer: Any, *, repeats: int
) -> Callable[[], tuple[dict[str, Any], ...]]:
    """Build the ``warm_repeated`` in-process callable for ``run_warm_trial``.

    Reuses one ``GrammarDecodeState`` across every repeated query -- a real
    warm-serving loop reuses its row state rather than discarding it -- so
    any session-level memoization the pack legitimately offers is exercised
    here, not hidden behind a fresh-session-per-call artifact.
    """

    from slm_training.dsl.grammar.fastpath.engine import engine_for_dsl
    from slm_training.dsl.grammar.fastpath.static_control_domain import (
        production_domain_snapshot,
    )
    from slm_training.models.grammar import GrammarDecodeState

    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode(FIXED_PREFIX_SURFACE, add_special=False),
    ]

    def _fn() -> tuple[dict[str, Any], ...]:
        # run_warm_trial() times the call to this callable itself (its own
        # total_ms); the single returned event is supplementary phase
        # detail, not the timing source of record. A lone event has no
        # predecessor marker, so cold_warm_bench reports its duration as
        # None rather than fabricating a delta -- see PhaseTiming.
        state = GrammarDecodeState(engine=engine_for_dsl("openui"))
        state.remaining_tokens = 32
        for _ in range(repeats):
            production_domain_snapshot(prefix, tokenizer, budget=32, state=state)
        return ({"phase": "repeated_domain_queries", "t": time.perf_counter()},)

    return _fn


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _phase_breakdown_by_arm(
    trials: list[BenchTrialV1], arm_by_config_hash: dict[str, str]
) -> dict[str, dict[str, float]]:
    """Median per-phase duration for each arm (AC: artifact overhead visible).

    Complete trials only -- a partial/timeout trial's truncated phase list
    would otherwise understate the real cost of the phases it never reached.
    """
    by_arm_phase: dict[str, dict[str, list[float]]] = {}
    for trial in trials:
        if trial.completeness != "complete":
            continue
        arm = arm_by_config_hash.get(trial.config_hash, "")
        for phase in trial.phases:
            if phase.duration_ms is None:
                continue
            by_arm_phase.setdefault(arm, {}).setdefault(phase.phase, []).append(
                phase.duration_ms
            )
    return {
        arm: {phase: round(_median(values), 3) for phase, values in phases.items()}
        for arm, phases in by_arm_phase.items()
    }


def run_campaign(
    campaign: Pct008CampaignV1,
    *,
    root: Path,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Run the fixture campaign; raises on a parity failure before any trial.

    Locks the manifest, runs the domain-parity gate, then runs
    ``cold_trials_per_arm`` fresh-subprocess cold trials per arm in
    :data:`COLD_ARMS` plus ``warm_trials`` in-process warm trials for
    :data:`WARM_ARM`, summarizes, and persists a ``CampaignResultV1`` under
    campaign governance.
    """

    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="Fixture-scale cold/warm artifact-parity evidence (PCT-008)",
            primary_metric="artifact_step_p50_ms",
            budget=CampaignBudget(
                max_experiments=campaign.cold_trials_per_arm * len(COLD_ARMS)
                + campaign.warm_trials,
                max_wall_minutes=campaign.max_wall_minutes,
            ),
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())

    tokenizer = DSLNativeTokenizer.build()
    parity = check_domain_parity(tokenizer)
    if not parity["equal"]:
        raise RuntimeError(
            "PCT-008 aborted before any timing trial: domain parity failed "
            f"({parity['reasons']})"
        )

    trials: list[BenchTrialV1] = []
    arm_by_config_hash: dict[str, str] = {}
    for arm in COLD_ARMS:
        script = _script_for_arm(arm)
        for _ in range(campaign.cold_trials_per_arm):
            trial = run_cold_trial(
                [sys.executable, "-c", script],
                config={"campaign": campaign.fingerprint, "arm": arm},
                timeout_s=timeout_s,
            )
            arm_by_config_hash[trial.config_hash] = arm
            trials.append(trial)
    warm_fn = _warm_repeated_trial(tokenizer, repeats=campaign.warm_trials)
    warm_trial = run_warm_trial(
        warm_fn, config={"campaign": campaign.fingerprint, "arm": WARM_ARM}
    )
    arm_by_config_hash[warm_trial.config_hash] = WARM_ARM
    trials.append(warm_trial)

    summary = summarize_trials(trials)
    for arm_summary in summary["arms"].values():
        arm_summary["arm"] = arm_by_config_hash.get(arm_summary["config_hash"], "")
    result = {
        "schema": "pct008_cold_warm_artifact_parity_result/v1",
        "campaign_id": campaign.campaign_id,
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": campaign.claim_class,
        "domain_parity": parity,
        "trials": [trial.to_dict() for trial in trials],
        "summary": summary,
        "phase_breakdown_by_arm": _phase_breakdown_by_arm(trials, arm_by_config_hash),
        "noise_floor_ms": _NOISE_FLOOR_MS,
        "scope_disclaimer": (
            "Artifact-seam phases only (process launch, tokenizer/pack "
            "construction, artifact step, first completion-domain query). "
            "No model is loaded; model init/first-forward and full "
            "first-verified-program latency are out of scope for this "
            "fixture pass. No production-speedup or model-quality claim is "
            "made."
        ),
    }
    artifact = store.write_artifact("pct008_cold_warm_result", result)
    store.append_event(
        "pct008_cold_warm_completed",
        experiment_id=campaign.campaign_id,
        status="fixture",
        artifact_sha256=artifact.stem,
        detail={"manifest_sha256": lock.manifest_sha256},
    )
    return result
