"""Diagnostic NLL attachment for the canonical continuous driver.

Uses the existing loss-suite evaluator; this is not a second evaluation engine.
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any

from scripts.autotrain_io import read_json as _read_json
from scripts.autotrain_metrics import paired_nll_selection
from scripts.autotrain_metrics import EVAL_NLL_RECORDS_NAME as _EVAL_NLL_RECORDS_NAME
from scripts.autotrain_metrics import (
    EVAL_NLL_RECORDS_SCHEMA as _EVAL_NLL_RECORDS_SCHEMA,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.evals.loss_suites import per_record_nll_map
from slm_training.evals.measurement_identity import content_digest


def compute_nll(test_dir, checkpoint, model, nll_config):
    from slm_training.evals.denoising_nll import DenoisingNLLConfig
    from slm_training.evals.loss_suites import (
        LOSS_SUITE_VERSION,
        evaluate_loss_suites,
        load_suite_spec,
    )

    if model is None:
        if checkpoint is None:
            raise ValueError("eval_nll requires model, checkpoint, or eval_nll")
        from slm_training.models.twotower import TwoTowerModel

        model = TwoTowerModel.from_checkpoint(checkpoint, device="cpu")
    if test_dir is None:
        raise ValueError("eval_nll compute path requires test_dir")
    spec = load_suite_spec(LOSS_SUITE_VERSION)
    cfg = nll_config or DenoisingNLLConfig(
        suite_version=LOSS_SUITE_VERSION,
        mask_rates=tuple(
            float(r) for r in (spec.get("mask_rates") or [0.15, 0.30, 0.50, 0.70, 0.85])
        ),
        mask_seed=int(spec.get("mask_seed", 0) or 0),
        compute_legal_support=False,
    )
    return evaluate_loss_suites(
        model,
        Path(test_dir),
        nll_config=cfg,
        base_suite="smoke",
        ood_suite="smoke",
    )


def _scoreboard(scoreboard_path):
    scoreboard = _read_json(scoreboard_path)
    suites = scoreboard.get("suites")
    if not isinstance(suites, dict):
        suites = {}
        scoreboard["suites"] = suites
    smoke = suites.get("smoke")
    if not isinstance(smoke, dict):
        smoke = {}
        suites["smoke"] = smoke
    return scoreboard, smoke


def run_arm_eval_nll(run_dir: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    records = inputs.get("records")
    selection = inputs.get("selection")
    row_evidence = inputs.get("row_evidence")
    estimator_id = inputs.get("estimator_id")
    attempt_id = inputs.get("attempt_id")
    from slm_training.autoresearch.climb_policy import screening_nll_definition_hash

    digest = inputs.get("definition_hash") or screening_nll_definition_hash()
    value = inputs.get("eval_nll")
    report: dict[str, Any] | None = None
    per_record: dict[str, float] | None = dict(records) if records is not None else None
    if value is None:
        report = compute_nll(
            inputs.get("test_dir"),
            inputs.get("checkpoint"),
            inputs.get("model"),
            inputs.get("nll_config"),
        )
        broad = (report.get("categories") or {}).get("broad") or {}
        mean = (broad.get("aggregate") or {}).get("mean_nll")
        if mean is None:
            mean = (report.get("aggregate") or {}).get("weighted_nll")
        if not isinstance(mean, (int, float)):
            raise ValueError("evaluate_loss_suites did not yield a finite smoke NLL")
        value = float(mean)
        per_record = per_record_nll_map(report)
        selection = report.get("selection")
        row_evidence = report.get("per_record")
        estimator_id = report.get("estimator_id")
        digest = content_digest(
            {
                "screening_definition": digest,
                "loss_definition": report.get("definition"),
            }
        )
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid_evidence: nonfinite diagnostic NLL")
    if per_record is not None and (
        not per_record
        or any(
            not isinstance(key, str)
            or not key
            or type(item) not in (int, float)
            or not math.isfinite(item)
            for key, item in per_record.items()
        )
    ):
        raise ValueError("invalid_evidence: malformed or empty per-record NLL")
    scoreboard_path = Path(run_dir) / "scoreboard.json"
    scoreboard, smoke = _scoreboard(scoreboard_path)
    smoke["eval_nll"] = float(value)
    smoke["eval_nll_definition_hash"] = digest
    smoke["eval_nll_claim_class"] = "diagnostic"
    paired_nll_selection({"selection": selection}, {"selection": selection})
    smoke["diagnostic_complete"] = bool(
        per_record
        and selection is not None
        and set(selection.get("selected_record_ids") or []) == set(per_record)
    )
    smoke.setdefault("decoded_probe_complete", False)
    scoreboard.setdefault("measurement_complete", False)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    records_path: Path | None = None
    smoke["eval_nll_records_sha256"] = None
    if per_record is not None:
        smoke["eval_nll_n_records"] = len(per_record)
        records_path = Path(run_dir) / _EVAL_NLL_RECORDS_NAME
        CampaignStore._replace_durable(
            records_path,
            json.dumps(
                {
                    "schema": _EVAL_NLL_RECORDS_SCHEMA,
                    "definition_hash": digest,
                    "claim_class": "diagnostic",
                    "suite": "smoke",
                    "eval_version": inputs.get("eval_version"),
                    "n_records": len(per_record),
                    "mean_nll": float(value),
                    "records": {k: per_record[k] for k in sorted(per_record)},
                    "selection": dict(selection) if selection is not None else None,
                    "row_evidence": row_evidence,
                    "attempt_id": attempt_id,
                    "estimator_id": estimator_id,
                    "units": "nats_per_masked_token",
                    "selection_locked": selection is not None,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
        )
        smoke["eval_nll_records_sha256"] = hashlib.sha256(
            records_path.read_bytes()
        ).hexdigest()
    CampaignStore._replace_durable(
        scoreboard_path, json.dumps(scoreboard, indent=2, allow_nan=False) + "\n"
    )
    return {
        "eval_nll": float(value),
        "definition_hash": digest,
        "scoreboard": str(scoreboard_path),
        "report": report,
        "records": per_record,
        "records_path": str(records_path) if records_path else None,
    }
