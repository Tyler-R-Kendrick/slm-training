"""RSP-005 (SLM-480): EXP-SR-9 prospective certified macro/library learning.

Fixture-scale governed campaign comparing three matched-budget arms:

* ``no_macros`` — control (no library)
* ``frequency_macros`` — frequency-ranked spans via :func:`induce_macros`
* ``learned_mdl`` — greedy MDL library via :func:`induce_macros`

The macro library is mined only from training-split solved ASTs
(``train_seeds.jsonl``). Prospective evidence scores
``macro_library_size_reduction_rate`` on frozen held-out/OOD programs
(``test_seeds.jsonl`` splits ``held_out`` + ``ood``) with expansion-equivalence
checks on every admitted macro. ``claim_class=fixture``; never
``promotion_candidate`` / ``ship_gate``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

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
from slm_training.data.macro_induction import MacroInductionConfig, induce_macros
from slm_training.dsl.canonicalize import canonicalize
from slm_training.dsl.language_contract import output_contract_violations
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.experiments.exp_sr_catalogue import (
    exp_sr_campaign,
    plan_only_manifest,
)
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer, SymbolTable

CATALOGUE_ID = "exp-sr-9"
PRIMARY_METRIC = "macro_library_size_reduction_rate"
ARM_IDS: tuple[str, ...] = ("no_macros", "frequency_macros", "learned_mdl")
TRAIN_SEEDS_PATH = Path("src/slm_training/resources/train_seeds.jsonl")
PROSPECTIVE_SEEDS_PATH = Path("src/slm_training/resources/test_seeds.jsonl")
PROSPECTIVE_SPLITS: frozenset[str] = frozenset({"held_out", "ood"})

__all__ = [
    "ARM_IDS",
    "CATALOGUE_ID",
    "PRIMARY_METRIC",
    "PROSPECTIVE_SPLITS",
    "Rsp005CampaignV1",
    "arm_induction_config",
    "build_macro_library_pack",
    "load_library_corpus",
    "load_prospective_corpus",
    "macro_library_pack_sha256",
    "plan_only_preview",
    "run_campaign",
    "score_arm_on_prospective",
    "verify_expansion_equivalence",
]


def _load_jsonl_sources(path: Path, *, split_filter: frozenset[str] | None) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing corpus: {path}")
    sources: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if split_filter is not None and row.get("split") not in split_filter:
            continue
        source = row["openui"]
        if output_contract_violations(source):
            continue
        sources.append(source)
    return sources


def load_library_corpus(*, train_path: Path = TRAIN_SEEDS_PATH) -> tuple[str, ...]:
    return tuple(_load_jsonl_sources(train_path, split_filter=frozenset({"train"})))


def load_prospective_corpus(
    *, eval_path: Path = PROSPECTIVE_SEEDS_PATH
) -> tuple[str, ...]:
    return tuple(_load_jsonl_sources(eval_path, split_filter=PROSPECTIVE_SPLITS))


def arm_induction_config(arm: str, *, max_macros: int) -> MacroInductionConfig | None:
    if arm == "no_macros":
        return None
    if arm == "frequency_macros":
        return MacroInductionConfig(
            max_macros=max_macros,
            selection_objective="frequency",
            min_gain_tokens=0,
        )
    if arm == "learned_mdl":
        return MacroInductionConfig(max_macros=max_macros, selection_objective="mdl")
    raise ValueError(f"unknown arm: {arm!r}")


def macro_library_pack_sha256(expansions: tuple[tuple[str, ...], ...]) -> str:
    return content_sha(
        {
            "schema": "macro_library_pack/v1",
            "expansions": [list(exp) for exp in expansions],
        }
    )


@dataclass(frozen=True)
class MacroLibraryPackV1:
    arm: str
    expansions: tuple[tuple[str, ...], ...]
    pack_sha256: str
    induction_stats: dict[str, Any]
    tokenizer_version: int

    @property
    def n_macros(self) -> int:
        return len(self.expansions)


def build_macro_library_pack(
    arm: str,
    library_sources: tuple[str, ...],
    *,
    max_macros: int,
    tokenizer: DSLNativeTokenizer | None = None,
) -> MacroLibraryPackV1:
    tok = tokenizer or DSLNativeTokenizer.build()
    cfg = arm_induction_config(arm, max_macros=max_macros)
    if cfg is None:
        return MacroLibraryPackV1(
            arm=arm,
            expansions=(),
            pack_sha256=macro_library_pack_sha256(()),
            induction_stats={"n_macros": 0, "arm": arm},
            tokenizer_version=tok.version,
        )
    if tok.macro_expansions:
        tok = DSLNativeTokenizer.build()
    result = induce_macros(library_sources, tok, cfg)
    expansions = result.expansions
    return MacroLibraryPackV1(
        arm=arm,
        expansions=expansions,
        pack_sha256=macro_library_pack_sha256(expansions),
        induction_stats=result.stats,
        tokenizer_version=tok.version,
    )


def verify_expansion_equivalence(
    sources: tuple[str, ...],
    pack: MacroLibraryPackV1,
) -> dict[str, Any]:
    """Macro substitution must expand to the plain token sequence, then decode canonically."""

    tok = DSLNativeTokenizer.build()
    if pack.expansions:
        tok.set_macro_expansions(pack.expansions)
    plain = DSLNativeTokenizer.build()
    failures: list[dict[str, str]] = []
    tokens_before = 0
    tokens_after = 0
    table_tokens = sum(len(exp) for exp in pack.expansions)
    for idx, source in enumerate(sources):
        canon = canonicalize(source)
        plain_table = SymbolTable()
        macro_table = SymbolTable()
        plain_ids = plain.encode(canon, add_special=False, table=plain_table)
        macro_ids = tok.encode(canon, add_special=False, table=macro_table)
        expanded = tok.expand_macros(list(macro_ids))
        plain_tokens = [plain.id_to_token[tid] for tid in plain_ids]
        expanded_tokens = [tok.id_to_token[tid] for tid in expanded]
        tokens_before += len(plain_ids)
        tokens_after += len(macro_ids)
        if plain_tokens != expanded_tokens:
            failures.append({"index": str(idx), "reason": "expansion_token_mismatch"})
    tokens_after_with_table = tokens_after + table_tokens
    size_reduction_rate = 0.0
    if tokens_before > 0:
        size_reduction_rate = max(
            0.0, (tokens_before - tokens_after_with_table) / tokens_before
        )
    return {
        "n_sources": len(sources),
        "tokens_before": tokens_before,
        "tokens_after_with_table": tokens_after_with_table,
        "table_tokens": table_tokens,
        PRIMARY_METRIC: round(size_reduction_rate, 6),
        "semantics_preserved": len(failures) == 0,
        "semantics_preserved_rate": 1.0 if not failures else 0.0,
        "failures": failures,
        "pack_sha256": pack.pack_sha256,
        "n_macros": pack.n_macros,
    }


def score_arm_on_prospective(
    arm: str,
    library_sources: tuple[str, ...],
    prospective_sources: tuple[str, ...],
    *,
    max_macros: int,
) -> dict[str, Any]:
    pack = build_macro_library_pack(arm, library_sources, max_macros=max_macros)
    prospective = verify_expansion_equivalence(prospective_sources, pack)
    return {
        "arm": arm,
        "pack": {
            "arm": pack.arm,
            "pack_sha256": pack.pack_sha256,
            "n_macros": pack.n_macros,
            "tokenizer_version": pack.tokenizer_version,
            "induction_stats": pack.induction_stats,
        },
        "library_sources": len(library_sources),
        "prospective_sources": len(prospective_sources),
        PRIMARY_METRIC: prospective[PRIMARY_METRIC],
        "semantics_preserved": prospective["semantics_preserved"],
        "semantics_preserved_rate": prospective["semantics_preserved_rate"],
        "prospective": prospective,
    }


Recommendation = Literal["reject", "inconclusive", "inconclusive_fixture"]


def _recommendation(
    *,
    semantics_ok: bool,
    mdl_rate: float,
    frequency_rate: float,
    minimum_effect: float,
) -> Recommendation:
    if not semantics_ok:
        return "reject"
    if mdl_rate >= minimum_effect and mdl_rate > frequency_rate:
        return "inconclusive_fixture"
    return "inconclusive"


def plan_only_preview() -> dict[str, object]:
    preview = plan_only_manifest(CATALOGUE_ID)
    return {
        **preview,
        "fixture_arms": list(ARM_IDS),
        "primary_metric": PRIMARY_METRIC,
        "claim_class_execution": "fixture",
        "promotion": False,
        "library_corpus": str(TRAIN_SEEDS_PATH),
        "prospective_corpus": str(PROSPECTIVE_SEEDS_PATH),
        "prospective_splits": sorted(PROSPECTIVE_SPLITS),
        "leakage_control": (
            "Macro library is learned only from training-side corpus, never "
            "from held-out evaluation programs."
        ),
    }


@dataclass(frozen=True)
class Rsp005CampaignV1:
    campaign_id: str = "rsp005-exp-sr-9-macro-library-fixture"
    max_macros: int = 8
    source_commit: str = "0" * 40
    claim_class: str = "fixture"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)
    train_path: Path = TRAIN_SEEDS_PATH
    prospective_path: Path = PROSPECTIVE_SEEDS_PATH

    def __post_init__(self) -> None:
        if self.max_macros < 1:
            raise ValueError("max_macros must be >= 1")
        if self.claim_class != "fixture":
            raise ValueError(
                "RSP-005 durable evidence is fixture-only "
                "(catalogue exp-sr-9; never promotion_candidate/ship_gate)"
            )
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["train_path"] = str(self.train_path)
        payload["prospective_path"] = str(self.prospective_path)
        return content_sha(payload)

    def catalogue_manifest(self) -> ExperimentCampaignV1:
        return exp_sr_campaign(CATALOGUE_ID)

    def minimum_effect(self) -> float:
        return float(self.catalogue_manifest().endpoints[0].minimum_effect)

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        catalogue = self.catalogue_manifest()
        arms = tuple(
            CampaignArmV1(
                arm_id=arm,
                role="control" if arm == "no_macros" else "candidate",
                config_sha256=content_sha({"campaign": fingerprint, "arm": arm}),
            )
            for arm in ARM_IDS
        )
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=CATALOGUE_ID,
            hypothesis=catalogue.hypothesis,
            decision=(
                "Adopt learned macros only where prospective size reduction holds "
                "with zero certified-semantics regressions. Experimental until "
                "prospective evidence at frontier scale."
            ),
            endpoints=(
                CampaignEndpointV1(
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    metric=PRIMARY_METRIC,
                    role="primary",
                    direction="increase",
                    minimum_effect=self.minimum_effect(),
                ),
            ),
            arms=arms,
            seeds=(0,),
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS),
                max_wall_minutes=self.max_wall_minutes,
            ),
            stopping_rules=(
                "Abort if any admitted macro fails expansion-equivalence on "
                "prospective programs.",
                "Library induction uses training-split ASTs only.",
                "Matched macro budget across frequency and MDL arms.",
            ),
            controls=(
                CampaignControlV1(
                    control_id=f"{CATALOGUE_ID}_matched_control",
                    description=catalogue.controls[0].description
                    if catalogue.controls
                    else (
                        "Same corpus expressed with and without learned macro "
                        "substitution."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id=f"{CATALOGUE_ID}_leakage_control",
                    description=catalogue.controls[1].description
                    if len(catalogue.controls) > 1
                    else (
                        "Macro library is learned only from the training-side "
                        "corpus, never from held-out evaluation programs."
                    ),
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
                    operator="le",
                    threshold=0.0,
                ),
            ),
            artifact_requirements=(
                ArtifactRequirementV1(kind="version_stamp"),
                ArtifactRequirementV1(kind="endpoint_result"),
                ArtifactRequirementV1(kind="paired_examples"),
            ),
            claim_class=self.claim_class,
            source_commit=self.source_commit,
            source_dirty=False,
            author="slm-training",
            created_at="1970-01-01T00:00:00Z",
        )


def run_campaign(campaign: Rsp005CampaignV1, *, root: Path) -> dict[str, Any]:
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="RSP-005 / EXP-SR-9 macro library fixture campaign",
            primary_metric=PRIMARY_METRIC,
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS),
                max_wall_minutes=campaign.max_wall_minutes,
            ),
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())
    catalogue = campaign.catalogue_manifest()

    library_sources = load_library_corpus(train_path=campaign.train_path)
    prospective_sources = load_prospective_corpus(
        eval_path=campaign.prospective_path
    )
    if not library_sources:
        raise RuntimeError("RSP-005: empty library corpus")
    if not prospective_sources:
        raise RuntimeError("RSP-005: empty prospective corpus")

    arm_results: dict[str, dict[str, Any]] = {}
    for arm in ARM_IDS:
        arm_results[arm] = score_arm_on_prospective(
            arm,
            library_sources,
            prospective_sources,
            max_macros=campaign.max_macros,
        )

    semantics_ok = all(r["semantics_preserved"] for r in arm_results.values())
    if not semantics_ok:
        failed = [a for a, r in arm_results.items() if not r["semantics_preserved"]]
        raise RuntimeError(
            "RSP-005 aborted: macro substitution changed certified semantics "
            f"on prospective programs (arms={failed})"
        )

    mdl_rate = float(arm_results["learned_mdl"][PRIMARY_METRIC])
    frequency_rate = float(arm_results["frequency_macros"][PRIMARY_METRIC])
    control_rate = float(arm_results["no_macros"][PRIMARY_METRIC])
    minimum_effect = campaign.minimum_effect()
    recommendation = _recommendation(
        semantics_ok=semantics_ok,
        mdl_rate=mdl_rate,
        frequency_rate=frequency_rate,
        minimum_effect=minimum_effect,
    )

    result = {
        "schema": "rsp005_macro_library_fixture_result/v1",
        "campaign_id": campaign.campaign_id,
        "catalogue_id": CATALOGUE_ID,
        "catalogue_manifest_sha256": campaign_manifest_sha256(catalogue),
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": campaign.claim_class,
        "promotion": False,
        PRIMARY_METRIC: mdl_rate,
        "minimum_effect": minimum_effect,
        "arm_results": arm_results,
        "matched_budget": {
            "max_macros": campaign.max_macros,
            "library_sources": len(library_sources),
            "prospective_sources": len(prospective_sources),
        },
        "control_rate": control_rate,
        "frequency_rate": frequency_rate,
        "mdl_rate": mdl_rate,
        "semantics_preserved": semantics_ok,
        "recommendation": recommendation,
        "scope_disclaimer": (
            "Fixture-scale EXP-SR-9 evidence for certified macro/library learning. "
            "Library mined from train_seeds.jsonl only; prospective scoring on "
            "held_out+ood rows from test_seeds.jsonl. Retrospective compression "
            "on the training corpus is reported in induction_stats but is not "
            "the success criterion. claim_class=fixture; experimental until "
            "frontier-scale prospective evidence."
        ),
    }
    artifact = store.write_artifact("rsp005_macro_library_result", result)
    store.append_event(
        "rsp005_macro_library_completed",
        experiment_id=campaign.campaign_id,
        status="fixture",
        artifact_sha256=artifact.stem,
        detail={
            "manifest_sha256": lock.manifest_sha256,
            PRIMARY_METRIC: mdl_rate,
            "recommendation": recommendation,
            "semantics_preserved": semantics_ok,
        },
    )
    return result
