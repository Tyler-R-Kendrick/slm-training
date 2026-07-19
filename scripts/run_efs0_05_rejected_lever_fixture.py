"""EFS0-05 fixture: rejected-lever registry and five-seed paired re-adjudication.

This script exercises ``slm_training.harnesses.experiments.rejected_lever_registry``
with a synthetic registry of five historically rejected levers.  It does not run
real training or evaluation; it proves the schema, pairing, seed completeness,
statistical classification, and autoresearch evidence integration are wired
correctly.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from slm_training.autoresearch.evidence import collect_evidence
from slm_training.harnesses.experiments.rejected_lever_registry import (
    PairedSeedObservation,
    ReAdjudicationRowV1,
    RejectedLeverRegistryV1,
    RejectedLeverV1,
    build_preregistered_campaign,
    check_duplicate_experiment_ids,
    check_duplicate_run_ids,
    closed_lever_signatures,
    paired_seed_result,
    save_registry,
    to_evidence_items,
)
from slm_training.versioning import build_version_stamp


OUTPUT_ROOT = Path("outputs/runs/efs0-05-rejected-lever/iter-efs0-05-20260719")
DOCS_JSON = Path("docs/design/iter-efs0-05-rejected-lever-readjudication-20260719.json")


def _example_registry() -> RejectedLeverRegistryV1:
    """Return a synthetic registry covering the required candidate levers."""
    base_commit = "a" * 40
    return RejectedLeverRegistryV1(
        registry_id="efs0-05-fixture",
        entries=(
            RejectedLeverV1(
                entry_id="E175_retrieval",
                experiment_ids=("E175",),
                run_ids=("E175_run_20260710",),
                hypothesis="typed retrieval improves semantic coverage by reusing representation-matched candidates",
                original_matrix="quality",
                original_source_commit=base_commit,
                original_train_commit=base_commit,
                original_eval_commit=base_commit,
                decoder_path="checkpoint_declared",
                decoder_version="v1",
                corpus_size=128,
                suite_size=64,
                seeds=(0, 1),
                original_primary_metric="binding_aware_meaningful_v2",
                original_primary_value=0.42,
                judge_provenance="meaningful_program_v1",
                observed_effect=-0.01,
                cost_metric="wall_seconds",
                observed_cost=1.2,
                confounds=("representation_mismatch", "underexposure"),
                status="reopen_candidate",
                evidence_needed="rerun with corrected decoder and five seeds at 64+ suite size",
                notes="8-step underexposure suspected",
            ),
            RejectedLeverV1(
                entry_id="E255_E256_ar_diffusion",
                experiment_ids=("E255", "E256"),
                run_ids=("E255_run_20260711", "E256_run_20260711"),
                hypothesis="AR-to-diffusion adapter preserves semantics at small diffusion budget",
                original_matrix="quality",
                original_source_commit=base_commit,
                original_train_commit=base_commit,
                original_eval_commit=base_commit,
                decoder_path="checkpoint_declared",
                decoder_version="v1",
                corpus_size=256,
                suite_size=128,
                seeds=(0, 1, 2),
                original_primary_metric="binding_aware_meaningful_v2",
                original_primary_value=0.45,
                judge_provenance="meaningful_program_v1",
                observed_effect=0.005,
                cost_metric="wall_seconds",
                observed_cost=2.1,
                confounds=("tiny_n", "underexposure"),
                status="reopen_candidate",
                evidence_needed="matched five-seed run at equal wall/verifier budget",
                notes="small budget, compare with B4 full runs",
            ),
            RejectedLeverV1(
                entry_id="X9_X14_typed_topology",
                experiment_ids=("X9", "X14"),
                run_ids=("X9_run_20260712", "X14_run_20260712"),
                hypothesis="typed topology constraints reduce invalid plan fragments",
                original_matrix="grammar",
                original_source_commit=base_commit,
                original_train_commit=base_commit,
                original_eval_commit=base_commit,
                decoder_path="current_native",
                decoder_version="v1",
                corpus_size=96,
                suite_size=48,
                seeds=(0, 1, 2),
                original_primary_metric="binding_aware_meaningful_v2",
                original_primary_value=0.38,
                judge_provenance="meaningful_program_v1",
                observed_effect=-0.02,
                cost_metric="wall_seconds",
                observed_cost=1.5,
                confounds=("tiny_n", "seed_instability"),
                status="reopen_candidate",
                evidence_needed="five seeds vs X22 family with corrected decoder",
                notes="compare with X22 tree-edit family",
            ),
            RejectedLeverV1(
                entry_id="E263_set_preference",
                experiment_ids=("E263", "E265"),
                run_ids=("E263_run_20260713", "E265_run_20260713"),
                hypothesis="set-valued preference objective improves local decision calibration",
                original_matrix="quality",
                original_source_commit=base_commit,
                original_train_commit=base_commit,
                original_eval_commit=base_commit,
                decoder_path="checkpoint_declared",
                decoder_version="v1",
                corpus_size=192,
                suite_size=96,
                seeds=(0, 1),
                original_primary_metric="binding_aware_meaningful_v2",
                original_primary_value=0.40,
                judge_provenance="meaningful_program_v1",
                observed_effect=-0.015,
                cost_metric="wall_seconds",
                observed_cost=1.8,
                confounds=("harness_interference", "tiny_n"),
                status="reopen_candidate",
                evidence_needed="non-interfering preference harness with five seeds",
                notes="prior harness may have coupled gradient updates",
            ),
            RejectedLeverV1(
                entry_id="E244_ptrm_sentinel",
                experiment_ids=("E244",),
                run_ids=("E244_run_20260709",),
                hypothesis="always-on PTRM improves meaningful-program rate",
                original_matrix="quality",
                original_source_commit=base_commit,
                original_train_commit=base_commit,
                original_eval_commit=base_commit,
                decoder_path="checkpoint_declared",
                decoder_version="v1",
                corpus_size=512,
                suite_size=256,
                seeds=(0,),
                original_primary_metric="binding_aware_meaningful_v2",
                original_primary_value=0.30,
                judge_provenance="meaningful_program_v1",
                observed_effect=-0.08,
                cost_metric="wall_seconds",
                observed_cost=4.0,
                confounds=("strong_negative_control", "decoder_bug"),
                status="closed",
                evidence_needed="equal-wall-time and equal-forward-pass replication only",
                notes="sentinel strong negative; used to verify the decision contract does not reopen spuriously",
            ),
        ),
    )


def _synthetic_observations(row: ReAdjudicationRowV1) -> list[PairedSeedObservation]:
    """Generate deterministic paired observations for one campaign row.

    The synthetic values are hand-tuned so that the fixture demonstrates every
    preregistered verdict class across the five required levers.
    """
    rng = random.Random(hash(row.row_id) & 0xFFFFFFFF)
    base_control = {
        "E175_retrieval": 0.43,
        "E255_E256_ar_diffusion": 0.46,
        "X9_X14_typed_topology": 0.39,
        "E263_set_preference": 0.41,
        "E244_ptrm_sentinel": 0.32,
    }[row.lever_id]

    # Treatment deltas tuned to verdict classes.
    mean_delta = {
        "E175_retrieval": 0.065,  # reopened_positive
        "E255_E256_ar_diffusion": 0.005,  # equivalent
        "X9_X14_typed_topology": -0.015,  # confirmed_negative
        "E263_set_preference": 0.03,  # inconclusive (inside equivalence-ish band around min_effect)
        "E244_ptrm_sentinel": -0.075,  # confirmed_negative sentinel
    }[row.lever_id]

    observations: list[PairedSeedObservation] = []
    for seed in row.seeds:
        control = base_control + rng.gauss(0, 0.01)
        treatment = control + mean_delta + rng.gauss(0, 0.008)
        control_cost = 1.0 + rng.gauss(0, 0.05)
        treatment_cost = control_cost + rng.gauss(0, 0.05)
        observations.append(
            PairedSeedObservation(
                seed=seed,
                control_value=round(control, 5),
                treatment_value=round(treatment, 5),
                control_cost=round(control_cost, 5),
                treatment_cost=round(treatment_cost, 5),
            )
        )
    return observations


def _run_fixture() -> dict[str, object]:
    registry = _example_registry()

    dup_runs = check_duplicate_run_ids(registry)
    dup_exps = check_duplicate_experiment_ids(registry)
    if dup_runs or dup_exps:
        raise ValueError(f"duplicate ids in fixture registry: runs={dup_runs}, exps={dup_exps}")

    campaign = build_preregistered_campaign(registry, required_levers=5, seed_count=5)
    rows = [row.model_dump(mode="json") for row in campaign]

    results: list[dict[str, object]] = []
    for row in campaign:
        observations = _synthetic_observations(row)
        result = paired_seed_result(row, observations)
        results.append(result.model_dump(mode="json"))

    evidence_items = [item.model_dump(mode="json") for item in to_evidence_items(registry)]
    closed = sorted(closed_lever_signatures(registry))

    summary = {
        "schema_version": "efs0-05-fixture-summary/v1",
        "registry_id": registry.registry_id,
        "campaign_rows": rows,
        "paired_results": results,
        "evidence_items": evidence_items,
        "closed_signatures": closed,
        "duplicate_run_ids": dup_runs,
        "duplicate_experiment_ids": dup_exps,
        "version_stamp": build_version_stamp("harness.experiments"),
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    registry_path = OUTPUT_ROOT / "rejected_lever_registry.json"
    summary_path = OUTPUT_ROOT / "summary.json"
    save_registry(registry, registry_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Also write the durable docs/design JSON used by autoresearch evidence intake.
    DOCS_JSON.parent.mkdir(parents=True, exist_ok=True)
    docs_payload = {
        "schema_version": "efs0-05-fixture-summary/v1",
        "registry_id": registry.registry_id,
        "campaign_rows": rows,
        "paired_results": results,
        "evidence_items": evidence_items,
        "closed_signatures": closed,
        "version_stamp": summary["version_stamp"],
    }
    DOCS_JSON.write_text(json.dumps(docs_payload, indent=2), encoding="utf-8")

    # Verify that evidence intake picks up the registry file.
    snapshot = collect_evidence([OUTPUT_ROOT], repo_root=Path("."))
    kinds = {item.kind for item in snapshot.items}
    if "rejected_lever" not in kinds:
        raise RuntimeError(f"evidence intake did not classify rejected_lever: {kinds}")

    return summary


if __name__ == "__main__":
    summary = _run_fixture()
    print(f"EFS0-05 fixture wrote {OUTPUT_ROOT}/rejected_lever_registry.json")
    print(f"EFS0-05 fixture wrote {DOCS_JSON}")
    print(
        "verdicts:",
        [r["verdict"] for r in summary["paired_results"]],
    )
