"""SLM-197 matched direct-policy controls over the SLM-196 bridge corpus."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from slm_training.data.flow.bridge_corpus import (
    LegalEditBridgeRowV1,
    RequestEditContractV1,
    load_corpus,
)
from slm_training.flow.termination import FixedKPolicy
from slm_training.models.legal_edit_batch import LegalEditBatch
from slm_training.models.legal_edit_scorer import (
    DirectLegalEditPolicy,
    LegalEditScorer,
    LegalEditScorerConfig,
    multi_positive_set_loss,
)

DEFAULT_CORPUS = Path(
    "src/slm_training/resources/data/train/slm196_legal_edit_bridge_fixture"
)
DEFAULT_RECORDS = Path("tests/fixtures/slm196_legal_edit_bridge/records.jsonl")
MATRIX_ARMS = {
    "D0": "X22 control",
    "D1": "plain local corruption",
    "D2": "plain full bridge",
    "D3-linear": "time-conditioned full bridge; linear schedule",
    "D3-fourier": "time-conditioned full bridge; Fourier schedule",
    "D4": "one-hot planner negative control",
    "D5": "multi-positive direct control reserved for flow comparison",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan_hint(batch: LegalEditBatch, rows: Sequence[LegalEditBridgeRowV1]) -> torch.Tensor:
    values = torch.zeros(len(batch.candidate_ids))
    for row_index, row in enumerate(rows):
        start, end = int(batch.row_offsets[row_index]), int(batch.row_offsets[row_index + 1])
        for index in range(start, end):
            if batch.candidate_ids[index] == row.planner_selected_candidate_id:
                values[index] = 1.0
    return values


def _schedule(rows: Sequence[LegalEditBridgeRowV1], fixed_budget: int = 4) -> torch.Tensor:
    """Inference-visible schedule; never use gold remaining distance/bridge length."""
    return torch.tensor(
        [min(1.0, row.step_index / max(1, fixed_budget)) for row in rows],
        dtype=torch.float32,
    )


def _evaluate(
    policy: DirectLegalEditPolicy,
    rows: Sequence[LegalEditBridgeRowV1],
    candidate_sets: dict[str, Any],
    *,
    arm: str,
) -> dict[str, Any]:
    batch = LegalEditBatch.pack(rows, candidate_sets)
    schedule = _schedule(rows)
    plan = _plan_hint(batch, rows) if arm == "D4" else None
    with torch.no_grad():
        logits = policy.scorer(
            batch, schedule_progress=schedule, plan_hint=plan
        )
        loss, masses = multi_positive_set_loss(logits, batch)
    top_positive = 0
    for row_index in range(len(rows)):
        start, end = int(batch.row_offsets[row_index]), int(batch.row_offsets[row_index + 1])
        selected = start + int(logits[start:end].argmax().item())
        top_positive += int(bool(batch.positive_mask[selected]))
    return {
        "teacher_forced": {
            "rows": len(rows),
            "set_mass_loss": float(loss),
            "positive_mass": masses["positive_mass"],
            "unknown_mass": masses["unknown_mass"],
            "top1_positive_rate": top_positive / max(1, len(rows)),
        },
        "candidate_membership": {
            "exact": all(
                tuple(sorted(row.complete_candidate_ids))
                == batch.candidate_ids[
                    int(batch.row_offsets[index]) : int(batch.row_offsets[index + 1])
                ]
                for index, row in enumerate(rows)
            ),
            "candidate_count": len(batch.candidate_ids),
            "candidate_set_digests": list(batch.candidate_set_digests),
            "unknown_as_explicit_negative": bool(
                (batch.unknown_mask & batch.unsupported_mask).any()
            ),
        },
    }


def _train_arm(
    arm: str,
    rows: Sequence[LegalEditBridgeRowV1],
    candidate_sets: dict[str, Any],
    *,
    seed: int,
    steps: int,
    learning_rate: float,
) -> tuple[DirectLegalEditPolicy, dict[str, Any]]:
    encoding = {
        "D3-linear": "linear",
        "D3-fourier": "fourier",
    }.get(arm, "no_time")
    scorer = LegalEditScorer(
        LegalEditScorerConfig(
            time_encoding=encoding, plan_enabled=arm == "D4", seed=seed
        )
    )
    policy = DirectLegalEditPolicy(scorer)
    batch = LegalEditBatch.pack(rows, candidate_sets)
    schedule = _schedule(rows)
    plan = _plan_hint(batch, rows) if arm == "D4" else None
    optimizer = torch.optim.Adam(scorer.parameters(), lr=learning_rate)
    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = scorer(batch, schedule_progress=schedule, plan_hint=plan)
        loss, _ = multi_positive_set_loss(logits, batch)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return policy, {
        "initial_loss": history[0],
        "final_loss": history[-1],
        "steps": steps,
        "exposures": steps * len(rows),
    }


def _records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _free_running(
    policy: DirectLegalEditPolicy,
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    traces = []
    exact = 0
    legal = 0
    for index, record in enumerate(records):
        trace = policy.decode_exact(
            record["source_program"],
            RequestEditContractV1.from_dict(record["request_contract"]),
            termination=FixedKPolicy(k=2, max_steps=2),
            max_steps=2,
            seed=seed + index,
        )
        target = record["target_program"]
        from slm_training.dsl.canonicalize import canonical_fingerprint

        exact += int(trace.final_fingerprint == canonical_fingerprint(target))
        legal += int(
            all(
                item["selected_candidate_id"] in item["candidate_ids"]
                for item in trace.decisions
            )
        )
        traces.append(
            {
                "record_id": record["id"],
                "target_exact": trace.final_fingerprint
                == canonical_fingerprint(target),
                "stop_reason": trace.stop_reason,
                "model_calls": trace.model_calls,
                "decisions": list(trace.decisions),
            }
        )
    return {
        "records": len(records),
        "target_exact_rate": exact / max(1, len(records)),
        "all_actions_live_rate": legal / max(1, len(records)),
        "traces": traces,
    }


def run_matrix(
    *,
    corpus_dir: Path = DEFAULT_CORPUS,
    records_path: Path = DEFAULT_RECORDS,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    steps: int = 8,
    learning_rate: float = 0.03,
    max_wall_minutes: float = 2.8,
) -> dict[str, Any]:
    if not 0 < max_wall_minutes <= 3:
        raise ValueError("max_wall_minutes must be in (0, 3]")
    started = time.monotonic()
    rows, candidate_sets, manifest = load_corpus(corpus_dir)
    train_rows = [row for row in rows if row.split == "train"]
    dev_rows = [row for row in rows if row.split == "dev"]
    records = _records(records_path)
    arms: dict[str, Any] = {
        "D0": {
            "status": "unavailable",
            "reason": manifest["diagnostics"]["coverage_comparison"][
                "x22_local_corruption"
            ]["reason"],
        },
        "D1": {
            "status": "unavailable",
            "reason": "no hash-pinned local-corruption corpus supplied",
        },
    }
    parameter_counts: set[int] = set()
    for arm in ("D2", "D3-linear", "D3-fourier", "D4", "D5"):
        runs = []
        for seed in seeds:
            if time.monotonic() - started > max_wall_minutes * 60:
                raise TimeoutError("SLM-197 cumulative wall budget exhausted")
            policy, training = _train_arm(
                arm,
                train_rows,
                candidate_sets,
                seed=seed,
                steps=steps,
                learning_rate=learning_rate,
            )
            parameter_counts.add(policy.scorer.artifact_identity()["param_count"])
            runs.append(
                {
                    "seed": seed,
                    "training": training,
                    "evaluation": _evaluate(
                        policy, dev_rows, candidate_sets, arm=arm
                    ),
                    "free_running": _free_running(policy, records, seed=seed),
                    "artifact_identity": policy.scorer.artifact_identity(),
                }
            )
        arms[arm] = {"status": "measured_fixture", "runs": runs}
    return {
        "schema": "DirectBridgePolicyMatrixV1",
        "issue": "SLM-197",
        "run_class": "fixture_wiring",
        "claim_class": "wiring",
        "status": "upstream_blocked",
        "honest_verdict": "inconclusive_fixture_only",
        "matrix": MATRIX_ARMS,
        "arms": arms,
        "recipe": {
            "device": "cpu",
            "backend": "exact legal-edit candidates",
            "steps": steps,
            "learning_rate": learning_rate,
            "seeds": list(seeds),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "independent_targets": manifest["diagnostics"]["independent_targets"],
            "max_wall_minutes": max_wall_minutes,
            "plan_default": "off",
            "termination": FixedKPolicy(k=2, max_steps=2).to_dict(),
        },
        "matched_controls": {
            "parameter_count_equal": len(parameter_counts) == 1,
            "parameter_counts": sorted(parameter_counts),
            "candidate_corpus_shared": True,
            "decoder_shared": True,
            "termination_shared": True,
        },
        "inputs": {
            "corpus": str(corpus_dir),
            "corpus_manifest_sha256": _sha(corpus_dir / "manifest.json"),
            "corpus_content_fingerprint": manifest["content_fingerprint"],
            "corpus_publishable": manifest["publishable"],
            "records": str(records_path),
            "records_sha256": _sha(records_path),
        },
        "confirmation": {
            "status": "blocked",
            "reasons": [
                "SLM-196 corpus manifest is non-publishable fixture evidence",
                "D0 X22 and D1 local-corruption controls are not hash-pinned",
                "only two independent targets and four bridge rows are available",
            ],
        },
        "checkpoint": {
            "written": False,
            "reason": "bounded matrix trained in memory; no reusable checkpoint designated",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
