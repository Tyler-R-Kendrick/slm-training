"""Pre-execution identity for the driver's canonical full-smoke loss probe.

Only selection/configuration is resolved here: no model or scorer is executed.
The producer differential test binds this adapter to ``autotrain_nll.compute_nll``.
Decode selection remains separate; its limit is not a loss-probe limit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from slm_training.data.store import DataStore
from slm_training.evals import denoising_nll
from slm_training.evals.loss_suites import load_suite_spec
from slm_training.evals.measurement_identity import content_digest, selected_identity
from slm_training.harnesses.model_build.data import load_suite_records
from slm_training.harnesses.model_build.ship_gates import DEFAULT_SHIP_GATES


def bind_endpoint(arms, endpoint, search_slug):
    selections = [evaluation_selection(arm["commands"]) for arm in arms]
    if selections[0] != selections[1]:
        raise ValueError("treatment_design:unmatched_evaluation_selection")
    endpoint = dict(endpoint, selection=selections[0])
    if search_slug is not None:
        if not isinstance(search_slug, str) or not search_slug.strip():
            raise ValueError("treatment_design:explicit_search_slug_required")
        losses = [screening_loss_binding(arm["commands"]) for arm in arms]
        if losses[0] != losses[1]:
            raise ValueError("treatment_design:unmatched_loss_selection")
        endpoint["loss_measurement"] = dict(
            losses[0], endpoint_id=endpoint["primary"]["endpoint_id"]
        )
    return endpoint


def evaluation_inputs(commands):
    evaluations = [cmd for cmd in commands if "scripts.evaluate_model" in cmd]
    if len(evaluations) != 1:
        raise ValueError("treatment_design:one_evaluation_command_required")
    command = evaluations[0]
    values = {
        command[i]: command[i + 1]
        for i in range(3, len(command) - 1)
        if command[i].startswith("--") and not command[i + 1].startswith("--")
    }
    root = DataStore().resolve_path("eval", Path(values["--test-dir"]))
    return root, values


def evaluation_selection(commands):
    root, values = evaluation_inputs(commands)
    suites = values.get("--suites", ",".join(DEFAULT_SHIP_GATES)).split(",")
    limit = int(values["--eval-limit"]) if "--eval-limit" in values else None
    if limit is not None and limit <= 0:
        raise ValueError("treatment_design:invalid_decode_limit")
    selection = {}
    for suite in suites:
        records = load_suite_records(root, suite.strip())
        identity = selected_identity(records)
        rows = [
            {"case_id": case, "root_id": family, "record_sha256": sha}
            for case, family, sha in zip(
                identity["selected_record_ids"],
                identity["selected_root_ids"],
                identity["input_sha256s"],
                strict=True,
            )
        ]
        selection[suite.strip()] = {"nll_cases": rows, "decode_cases": rows[:limit]}
    return selection


def screening_loss_binding(commands):
    """Mirror the existing driver's fixed specification, never training seeds."""
    root, _ = evaluation_inputs(commands)
    selection = selected_identity(load_suite_records(root, "smoke"))
    spec = load_suite_spec(denoising_nll.LOSS_SUITE_VERSION)
    cfg = denoising_nll.DenoisingNLLConfig(
        suite_version=denoising_nll.LOSS_SUITE_VERSION,
        mask_rates=tuple(float(rate) for rate in spec["mask_rates"]),
        mask_seed=spec["mask_seed"],
        compute_legal_support=False,
    )
    if type(cfg.mask_seed) is not int or cfg.mask_seed < 0:
        raise ValueError("treatment_design:invalid_loss_spec_seed")
    scorer = hashlib.sha256(Path(denoising_nll.__file__).read_bytes()).hexdigest()
    return {
        "version": cfg.suite_version,
        "units": "nats_per_masked_token",
        "estimator_id": "conditional_masked_token_ce/"
        + content_digest(
            {
                **cfg.key(),
                "evaluator_sha256": scorer,
                "position_filter": "all_eligible",
            }
        ),
        "evaluator_sha256": scorer,
        "selection": selection,
        "mask_seed": cfg.mask_seed,
        "config": cfg.key(),
        "suite_manifest_sha256": content_digest(
            {
                "suite": "smoke",
                "selection": selection,
                "exposure": "public_regression_not_sealed",
            }
        ),
    }
