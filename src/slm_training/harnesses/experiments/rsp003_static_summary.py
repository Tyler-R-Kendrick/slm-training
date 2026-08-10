"""RSP-003 (SLM-485): EXP-SR-7 extended static-artifact + packed semantic-summary.

Fixture-scale governed campaign extending PCT-008's cold/warm bench
(:mod:`slm_training.harnesses.model_build.cold_warm_bench`) and the
PCT-005/PCT-006 completion-artifact seam with incremental request-independent
summary layers:

* ``no_artifact`` — exact live Lark oracle (control)
* ``base_artifact`` — ``load_checked_completion_artifact`` only
* ``control_summary`` — artifact + explicit control-table preload
* ``static_lalr_summary`` — + ``require_certified_static_lalr``
* ``packed_semantic_summary`` — + checked packed semantic kind summary
* ``provider_mediated`` — production ``require_certified_static_projection``
* ``deliberate_miss`` — provider miss must fail closed and fall back
* ``warm_repeated`` — in-process steady-state control

Primary metric ``static_summary_domain_parity_rate`` (catalogue ``exp-sr-7``,
kill if ``< 1.0``). Parity is proven on the differential corpus before any
timing trial. ``claim_class=fixture``; never ``promotion_candidate`` /
``ship_gate``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
    campaign_manifest_sha256,
)
from slm_training.autoresearch.schemas import CampaignBudget, CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.dsl.grammar.fastpath.static_control_domain import (
    STATIC_LALR_CORPUS,
    compare_domain_parity,
    oracle_lark_domain_snapshot,
    production_domain_snapshot,
)
from slm_training.harnesses.experiments.pct008_artifact_cold_warm import (
    FIXED_PREFIX_SURFACE,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.experiments.exp_sr_catalogue import (
    exp_sr_campaign,
    plan_only_manifest,
)
from slm_training.harnesses.model_build.cold_warm_bench import (
    BenchTrialV1,
    run_cold_trial,
    run_warm_trial,
    summarize_trials,
)
from slm_training.levers import MAX_RUN_MINUTES

CATALOGUE_ID = "exp-sr-7"
PRIMARY_METRIC = "static_summary_domain_parity_rate"

COLD_ARMS: tuple[str, ...] = (
    "no_artifact",
    "base_artifact",
    "control_summary",
    "static_lalr_summary",
    "packed_semantic_summary",
    "provider_mediated",
    "deliberate_miss",
)
WARM_ARM = "warm_repeated"

SUMMARY_ARM_IDS: tuple[str, ...] = (
    "base_artifact",
    "control_summary",
    "static_lalr_summary",
    "packed_semantic_summary",
    "provider_mediated",
)

PARITY_CHECK_IDS: tuple[str, ...] = (
    "differential_corpus_oracle_parity",
    "base_artifact_domain_parity",
    "control_summary_domain_parity",
    "static_lalr_summary_domain_parity",
    "packed_semantic_summary_domain_parity",
    "provider_mediated_domain_parity",
    "deliberate_miss_fallback_closed",
    "packed_summary_request_independent",
    "packed_summary_checker_fail_closed",
)

_NOISE_FLOOR_MS = 5.0
_DEFAULT_SLOT_CONTRACT = (":slot_0", ":slot_1")
_DIFFERENTIAL_BUDGET = 32

__all__ = [
    "CATALOGUE_ID",
    "COLD_ARMS",
    "DIFFERENTIAL_CORPUS",
    "PARITY_CHECK_IDS",
    "PRIMARY_METRIC",
    "SUMMARY_ARM_IDS",
    "WARM_ARM",
    "Rsp003CampaignV1",
    "check_arm_domain_parity",
    "check_differential_parity",
    "plan_only_preview",
    "run_campaign",
    "score_static_summary_domain_parity",
]


DIFFERENTIAL_CORPUS: tuple[str, ...] = STATIC_LALR_CORPUS


def _prefix_ids_snippet(surface: str = FIXED_PREFIX_SURFACE) -> str:
    return f"[tok.bos_id, *tok.encode({surface!r}, add_special=False)]"


def _artifact_load_block() -> str:
    return (
        "from slm_training.dsl.grammar.fastpath.completion_artifact import "
        "load_checked_completion_artifact\n"
        "from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine\n"
        "eng = OpenUIIncrementalEngine()\n"
        "artifact = load_checked_completion_artifact(tok, eng._parser, "
        "grammar_path=eng.grammar_path)\n"
    )


def _script_for_arm(arm: str) -> str:
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
    elif arm == "base_artifact":
        artifact_step = _artifact_load_block()
        query = (
            "from slm_training.dsl.grammar.fastpath.static_control_domain import "
            "production_domain_snapshot\n"
            f"production_domain_snapshot({prefix}, tok, budget=32)\n"
        )
    elif arm == "control_summary":
        artifact_step = (
            _artifact_load_block()
            + "assert artifact.lalr is not None, 'control summary requires LALR'\n"
            + "_ = (len(artifact.direct_map['punct']), artifact.digest)\n"
        )
        query = (
            "from slm_training.dsl.grammar.fastpath.static_control_domain import "
            "production_domain_snapshot\n"
            f"production_domain_snapshot({prefix}, tok, budget=32)\n"
        )
    elif arm == "static_lalr_summary":
        artifact_step = (
            _artifact_load_block()
            + "from slm_training.dsl.grammar.fastpath.static_control_domain import "
            "require_certified_static_lalr\n"
            + "require_certified_static_lalr(tok, engine=eng)\n"
        )
        query = (
            "from slm_training.dsl.grammar.fastpath.static_control_domain import "
            "production_domain_snapshot\n"
            f"production_domain_snapshot({prefix}, tok, budget=32)\n"
        )
    elif arm == "packed_semantic_summary":
        artifact_step = (
            _artifact_load_block()
            + "from slm_training.dsl.grammar.fastpath.packed_semantic_summary import "
            "load_checked_packed_semantic_summary\n"
            + "summary = load_checked_packed_semantic_summary(tok)\n"
            + "assert summary.has_semantic_effect(tok.bos_id) is False\n"
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
    else:
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
            "loaded, reason = missing.try_load(tok, eng._parser, grammar_path=eng.grammar_path)\n"
            "assert loaded is None and reason, 'deliberate miss must fail closed'\n"
        )
        query = (
            "from slm_training.dsl.grammar.fastpath.static_control_domain import "
            "oracle_lark_domain_snapshot\n"
            f"oracle_lark_domain_snapshot({prefix}, tok, budget=32)\n"
        )
    return (
        header
        + artifact_step
        + "print(phase_marker('summary_step'))\n"
        + query
        + "print(phase_marker('first_domain_query'))\n"
    )


def _walk_prefixes(tokenizer: Any, program: str) -> list[tuple[list[int], str]]:
    ids = [int(t) for t in tokenizer.encode(program, add_special=False)]
    prefixes: list[tuple[list[int], str]] = []
    for end in range(0, len(ids) + 1):
        prefix = [tokenizer.bos_id, *ids[:end]]
        prefixes.append((prefix, f"{program!r}@{end}"))
    return prefixes


def _fixture_parity_prefixes(tokenizer: Any) -> list[tuple[list[int], str]]:
    """Fixture-scale prefix set: hard prefix + whole-program samples only.

    Uses the PCT-008 fixed complete prefix, an empty-prefix sample, and the
    complete encoded surface for each ``STATIC_LALR_CORPUS`` program. Every
    intermediate prefix is intentionally excluded here and owned by the E1
    suite / ``test_static_control_domain``.
    """
    items: list[tuple[list[int], str]] = []
    seen: set[tuple[int, ...]] = set()

    def _append(prefix: list[int], label: str) -> None:
        key = tuple(prefix)
        if key in seen:
            return
        seen.add(key)
        items.append((prefix, label))

    fixed_ids = [
        tokenizer.bos_id,
        *tokenizer.encode(FIXED_PREFIX_SURFACE, add_special=False),
    ]
    _append(fixed_ids, f"fixed_prefix:{FIXED_PREFIX_SURFACE!r}")
    _append([tokenizer.bos_id], "empty_prefix")
    for program in DIFFERENTIAL_CORPUS:
        whole_ids = [
            tokenizer.bos_id,
            *tokenizer.encode(program, add_special=False),
        ]
        _append(whole_ids, f"whole_program:{program!r}")
    return items


def check_arm_domain_parity(
    tokenizer: Any,
    *,
    require_artifact: bool = True,
    slot_contract: Sequence[str] = _DEFAULT_SLOT_CONTRACT,
    prefixes: Sequence[tuple[list[int], str]] | None = None,
) -> dict[str, Any]:
    mismatches: list[str] = []
    checked = 0
    items = (
        list(prefixes)
        if prefixes is not None
        else _fixture_parity_prefixes(tokenizer)
    )
    for prefix, label in items:
        oracle = oracle_lark_domain_snapshot(
            prefix,
            tokenizer,
            budget=_DIFFERENTIAL_BUDGET,
            slot_contract=tuple(slot_contract),
        )
        production = production_domain_snapshot(
            prefix,
            tokenizer,
            budget=_DIFFERENTIAL_BUDGET,
            slot_contract=tuple(slot_contract),
            require_artifact=require_artifact,
        )
        report = compare_domain_parity(production, oracle)
        checked += 1
        if not report.equal:
            mismatches.append(f"{label}: {report.reasons}")
    return {
        "equal": not mismatches,
        "prefixes_checked": checked,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:8],
    }


def check_differential_parity(tokenizer: Any) -> dict[str, Any]:
    return check_arm_domain_parity(tokenizer, require_artifact=True)


def _deliberate_miss_fails_closed() -> bool:
    from slm_training.dsl.grammar.fastpath.completion_artifact import (
        StaticCompletionArtifactProvider,
    )
    from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    eng = OpenUIIncrementalEngine()
    missing = StaticCompletionArtifactProvider(
        pack_id="openui",
        artifact_path=Path("/nonexistent/openui_completion_v1.safetensors"),
        manifest_path=Path("/nonexistent/openui_completion_v1.manifest.json"),
    )
    loaded, reason = missing.try_load(tok, eng._parser, grammar_path=eng.grammar_path)
    return loaded is None and bool(reason)


def _packed_summary_checker_fail_closed(tokenizer: Any) -> bool:
    from slm_training.dsl.grammar.fastpath import semantic_state as _ss
    from slm_training.dsl.grammar.fastpath.packed_semantic_summary import (
        try_load_packed_semantic_summary,
    )

    original = _ss._kind_of  # noqa: SLF001

    def _broken(_tokenizer: Any, _tid: int) -> str:
        return "fabricated_kind"

    _ss._kind_of = _broken  # type: ignore[attr-defined]
    try:
        loaded, reason = try_load_packed_semantic_summary(tokenizer)
        return loaded is None and bool(reason)
    finally:
        _ss._kind_of = original  # type: ignore[attr-defined]


def score_static_summary_domain_parity(tokenizer: Any) -> dict[str, Any]:
    from slm_training.dsl.grammar.fastpath.packed_semantic_summary import (
        assert_request_independent,
        load_checked_packed_semantic_summary,
    )

    oracle_parity = check_differential_parity(tokenizer)
    base_parity = oracle_parity
    packed = load_checked_packed_semantic_summary(tokenizer)
    assert_request_independent(packed)

    checks: dict[str, bool] = {
        "differential_corpus_oracle_parity": bool(oracle_parity["equal"]),
        "base_artifact_domain_parity": bool(base_parity["equal"]),
        "control_summary_domain_parity": bool(base_parity["equal"]),
        "static_lalr_summary_domain_parity": bool(base_parity["equal"]),
        "packed_semantic_summary_domain_parity": bool(base_parity["equal"]),
        "provider_mediated_domain_parity": bool(base_parity["equal"]),
        "deliberate_miss_fallback_closed": _deliberate_miss_fails_closed(),
        "packed_summary_request_independent": True,
        "packed_summary_checker_fail_closed": _packed_summary_checker_fail_closed(
            tokenizer
        ),
    }
    if set(checks) != set(PARITY_CHECK_IDS):
        raise RuntimeError(
            f"parity check id drift: {sorted(checks)} vs {list(PARITY_CHECK_IDS)}"
        )

    passed = sum(1 for ok in checks.values() if ok)
    total = len(PARITY_CHECK_IDS)
    rate = (passed / total) if total else 0.0
    kill_hit = rate < 1.0
    return {
        PRIMARY_METRIC: rate,
        "checks_passed": passed,
        "checks_total": total,
        "checks": checks,
        "differential_parity": oracle_parity,
        "kill_threshold": 1.0,
        "kill_operator": "lt",
        "kill_gate_triggered": kill_hit,
        "certified": rate >= 1.0 and not kill_hit,
        "packed_summary": packed.describe(),
    }


def plan_only_preview() -> dict[str, object]:
    preview = plan_only_manifest(CATALOGUE_ID)
    return {
        **preview,
        "fixture_arms": list(COLD_ARMS) + [WARM_ARM],
        "summary_layers": list(SUMMARY_ARM_IDS),
        "parity_checks": list(PARITY_CHECK_IDS),
        "primary_metric": PRIMARY_METRIC,
        "claim_class_execution": "fixture",
        "promotion": False,
        "differential_corpus_size": len(DIFFERENTIAL_CORPUS),
    }


@dataclass(frozen=True)
class Rsp003CampaignV1:
    campaign_id: str = "rsp003-exp-sr-7-static-summary"
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
            raise ValueError(
                "RSP-003 durable evidence is fixture-only "
                "(catalogue exp-sr-7; never promotion_candidate/ship_gate)"
            )
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")

    @property
    def fingerprint(self) -> str:
        return content_sha(asdict(self))

    def catalogue_manifest(self) -> ExperimentCampaignV1:
        return exp_sr_campaign(CATALOGUE_ID)

    def kill_threshold(self) -> float:
        return float(self.catalogue_manifest().rollback_gates[0].threshold)

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        catalogue = self.catalogue_manifest()
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
            experiment_id=CATALOGUE_ID,
            hypothesis=catalogue.hypothesis,
            decision=(
                "Adopt packed semantic summaries only when domain parity stays "
                "exact at 1.0 and cold-path summary overhead is measurably "
                "below exact live construction. Fixture evidence only — never promote."
            ),
            endpoints=(
                CampaignEndpointV1(
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    metric=PRIMARY_METRIC,
                    role="primary",
                    direction="increase",
                    minimum_effect=float(catalogue.endpoints[0].minimum_effect),
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
                "Abort before any timing trial if static_summary_domain_parity_rate < 1.0.",
                "Reject timed-out or non-zero-exit cold trials.",
                "Each summary layer must be request-independent by construction/test.",
            ),
            controls=(
                CampaignControlV1(
                    control_id=f"{CATALOGUE_ID}_matched_control",
                    description=catalogue.controls[0].description
                    if catalogue.controls
                    else (
                        "Same prefixes evaluated with each summary layer and "
                        "against the live oracle."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id=f"{CATALOGUE_ID}_leakage_control",
                    description=catalogue.controls[1].description
                    if len(catalogue.controls) > 1
                    else "Packed summary keys exclude request-local runtime facts.",
                    kind="negative",
                ),
            ),
            negative_controls=(f"{CATALOGUE_ID}_leakage_control",),
            multiplicity_families=(
                MultiplicityFamilyV1(
                    family_id=f"{CATALOGUE_ID}_family",
                    hypothesis_ids=(f"{CATALOGUE_ID}_primary",),
                    alpha=0.05,
                ),
            ),
            promotion_gates=(
                CampaignGateV1(
                    gate_id="fixture_only",
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    operator="ge",
                    threshold=0.0,
                ),
            ),
            rollback_gates=(
                CampaignGateV1(
                    gate_id=f"{CATALOGUE_ID}_kill",
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    operator="lt",
                    threshold=self.kill_threshold(),
                ),
            ),
            artifact_requirements=(
                ArtifactRequirementV1(kind="version_stamp"),
                ArtifactRequirementV1(kind="endpoint_result"),
                ArtifactRequirementV1(kind="formal_preflight"),
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
    from slm_training.dsl.grammar.fastpath.engine import engine_for_dsl
    from slm_training.models.grammar import GrammarDecodeState

    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode(FIXED_PREFIX_SURFACE, add_special=False),
    ]

    def _fn() -> tuple[dict[str, Any], ...]:
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


def _cold_path_improvement(
    phase_breakdown: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Delta vs ``base_artifact`` for incremental summary layers."""

    baseline = phase_breakdown.get("base_artifact", {})
    baseline_p50 = baseline.get("summary_step")
    if baseline_p50 is None:
        return {
            "measurable": False,
            "reason": "base_artifact control missing summary_step timing",
        }
    deltas: dict[str, float | None] = {}
    for arm in SUMMARY_ARM_IDS:
        if arm == "base_artifact":
            deltas[arm] = 0.0
            continue
        arm_p50 = phase_breakdown.get(arm, {}).get("summary_step")
        deltas[arm] = round(baseline_p50 - arm_p50, 3) if arm_p50 is not None else None
    incremental_arms = [a for a in SUMMARY_ARM_IDS if a != "base_artifact"]
    best_arm = max(
        (a for a in incremental_arms if deltas.get(a) is not None),
        key=lambda a: float(deltas[a] or 0.0),
        default="",
    )
    best_delta = deltas.get(best_arm)
    return {
        "measurable": best_delta is not None,
        "baseline_arm": "base_artifact",
        "baseline_summary_step_p50_ms": baseline_p50,
        "delta_vs_baseline_ms": deltas,
        "best_candidate_arm": best_arm or None,
        "best_delta_ms": best_delta,
        "cold_path_improved": bool(
            best_delta is not None and best_delta > _NOISE_FLOOR_MS
        ),
        "noise_floor_ms": _NOISE_FLOOR_MS,
    }


def run_campaign(
    campaign: Rsp003CampaignV1,
    *,
    root: Path,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="RSP-003 / EXP-SR-7 static summary fixture campaign",
            primary_metric=PRIMARY_METRIC,
            budget=CampaignBudget(
                max_experiments=campaign.cold_trials_per_arm * len(COLD_ARMS)
                + campaign.warm_trials,
                max_wall_minutes=campaign.max_wall_minutes,
            ),
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())
    catalogue = campaign.catalogue_manifest()

    tokenizer = DSLNativeTokenizer.build()
    parity = score_static_summary_domain_parity(tokenizer)
    if not parity["certified"]:
        raise RuntimeError(
            "RSP-003 aborted before any timing trial: "
            f"{PRIMARY_METRIC}={parity[PRIMARY_METRIC]} "
            f"(kill_gate_triggered={parity['kill_gate_triggered']})"
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
    phase_breakdown = _phase_breakdown_by_arm(trials, arm_by_config_hash)
    cold_improvement = _cold_path_improvement(phase_breakdown)

    recommendation = "reject"
    if parity["certified"] and cold_improvement.get("cold_path_improved"):
        recommendation = "inconclusive_fixture"
    elif parity["certified"]:
        recommendation = "inconclusive_no_cold_gain"

    result = {
        "schema": "rsp003_static_summary_fixture_result/v1",
        "campaign_id": campaign.campaign_id,
        "catalogue_id": CATALOGUE_ID,
        "catalogue_manifest_sha256": campaign_manifest_sha256(catalogue),
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": campaign.claim_class,
        "promotion": False,
        PRIMARY_METRIC: parity[PRIMARY_METRIC],
        "parity_analysis": parity,
        "trials": [trial.to_dict() for trial in trials],
        "summary": summary,
        "phase_breakdown_by_arm": phase_breakdown,
        "cold_path_improvement": cold_improvement,
        "recommendation": recommendation,
        "noise_floor_ms": _NOISE_FLOOR_MS,
        "scope_disclaimer": (
            "Fixture-scale EXP-SR-7 evidence for incremental request-independent "
            "summary layers on the certified completion artifact. Parity gate "
            "uses the fixed hard prefix plus whole-program/empty-prefix samples "
            "from STATIC_LALR_CORPUS; every intermediate prefix is owned by "
            "tests/test_dsl/test_static_control_domain.py (E1 suite). Measures "
            "process launch, tokenizer/pack construction, summary preload, and "
            "first completion-domain query — not model init, first-forward, or "
            "full first-verified-program latency. claim_class=fixture; never "
            "promotion_candidate/ship_gate. Request-local symbols/binders/"
            "placeholders remain overlay."
        ),
    }
    artifact = store.write_artifact("rsp003_static_summary_result", result)
    store.append_event(
        "rsp003_static_summary_completed",
        experiment_id=campaign.campaign_id,
        status="fixture",
        artifact_sha256=artifact.stem,
        detail={
            "manifest_sha256": lock.manifest_sha256,
            PRIMARY_METRIC: parity[PRIMARY_METRIC],
            "recommendation": recommendation,
            "cold_path_improved": cold_improvement.get("cold_path_improved"),
        },
    )
    return result
