#!/usr/bin/env python3
"""SLM-301 (LAR1-04): prompt-observability experiment (as-is vs +inventory arms).

Phase 0 (model-free): classify v2 contract observability for the 16
``test_seeds.jsonl`` rows and the 35 SLM-294 frozen rico rows. Corpus reality:
0/16 test_seeds and 0/35 rico prompts carry a visible inventory section, so
placeholder coverage is request-only or UNKNOWN everywhere pre-inventory.

Phase 1 (decode, 16 test_seeds only): two baselines × two matched arms —
  - AR tiny baseline (``outputs/runs/slm294_tiny_baseline/checkpoints/last.pt``,
    unconstrained decode + post-validation; grammar-constrained decode is
    infeasible per SLM-294 measurements),
  - deterministic X22 (SLM-155 minimal seed vs inventory-derived
    certified-minimal skeleton + simulated conflict-slice repair, equal edit
    budget),
each on arm A (prompt as-is) vs arm B (prompt + visible slot inventory via
``ensure_prompt_inventory``), same seed (0) and budget. RICO-scale decode with
this checkpoint is infeasible (>280s / 16 tokens, SLM-294), so the 35 rico rows
are declared ``not_run`` with that evidence. Per-record decodes exceeding
--decode-timeout-s are marked ``decode_timeout`` — honest, never evidence.

Scoring: ``binding_aware_meaningful_v2`` with the explicit arm request (so
provenance distinguishes prompt vs request coverage), syntax validity,
placeholder fidelity, structural similarity vs gold; Wilson 95% intervals;
paired per-row A-vs-B tables with the predeclared 0.10 minimum useful paired
delta on OBSERVABLE_REQUEST_ONLY/UNKNOWN rows and a no-regression check on
already-observable rows.

Artifacts: ``docs/design/iter-slm301-prompt-observability-20260724.{json,md}``
plus AgentEvals JSONL (diagnostic claim; publication failure is non-fatal).

Example:
  python -m scripts.run_slm301_prompt_observability
  python -m scripts.run_slm301_prompt_observability --shard 0 --num-shards 2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.parser import validate
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.meaningful_program import binding_aware_meaningful_v2
from slm_training.evals.power_protocol import wilson_interval
from slm_training.harnesses.experiments.slm155_factorization_comparison import (
    X22_SIMULATED_REPAIR_NOTE,
    _ast_to_topology,
    _build_conflict_slice,
    _build_minimal_seed_source,
)
from slm_training.harnesses.experiments.conflict_slice_repair import (
    apply_repair_policy,
)
from slm_training.harnesses.experiments.slm301_prompt_observability import (
    EXPERIMENT_ID,
    MIN_USEFUL_PAIRED_DELTA,
    OBSERVABLE_PROMPT,
    OBSERVABLE_REQUEST_ONLY,
    UNKNOWN,
    build_matched_requests,
    observability_report,
    paired_delta,
    paired_outcome_table,
)
from slm_training.models.template_fill import (
    build_slot_contract_template,
    inventory_from_prompt,
)
from slm_training.versioning import build_version_stamp

TEST_SEEDS = Path("src/slm_training/resources/test_seeds.jsonl")
RICO_CORPUS = Path("src/slm_training/resources/evals/slm294_frozen_requests_v1.jsonl")
AR_CHECKPOINT = Path("outputs/runs/slm294_tiny_baseline/checkpoints/last.pt")
TRAIN_DIR = Path("outputs/data/train/slm230_symbol_only_v1")
TEST_DIR = Path("outputs/data/eval/v1")

DEFAULT_JSON_OUT = Path("docs/design/iter-slm301-prompt-observability-20260724.json")
DEFAULT_MD_OUT = Path("docs/design/iter-slm301-prompt-observability-20260724.md")

COMPONENT = "harness.experiments.slm301_prompt_observability"

RICO_NOT_RUN_EVIDENCE = (
    "SLM-294 measured >280s for 16 tokens with this checkpoint at rico prompt "
    "scale; 35 rico rows declared not_run (classification only, phase 0)."
)

X22_EDIT_BUDGET = 64  # equal forwards budget across arms (SLM-155 fixture value)


def _load_records(path: Path, *, default_split: str) -> list[ExampleRecord]:
    records: list[ExampleRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append(
                ExampleRecord(
                    id=row["id"],
                    prompt=row["prompt"],
                    openui=row["openui"],
                    placeholders=list(row.get("placeholders", [])),
                    split=row.get("split", default_split),
                    source=row.get("source", default_split),
                )
            )
    return records


def _predeclaration() -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": (
            "Adding the production visible slot inventory to the prompt "
            "(arm B) closes the OBSERVABLE_REQUEST_ONLY coverage gap and does "
            "not regress v2-strict meaning on already-observable rows."
        ),
        "falsifier": (
            "Arm B v2-strict paired delta on OBSERVABLE_REQUEST_ONLY/UNKNOWN "
            "rows is below the predeclared minimum useful delta, or arm B "
            "regresses on OBSERVABLE_PROMPT rows."
        ),
        "n_test_seeds": 16,
        "seeds": [0],
        "arms": {
            "A": "prompt as-is (GenerationRequest.from_record)",
            "B": "prompt + visible Placeholders inventory via ensure_prompt_inventory over the arm-A slot contract",
        },
        "baselines": {
            "ar_tiny": {
                "checkpoint": str(AR_CHECKPOINT),
                "decode": "unconstrained + post-validation (grammar-constrained infeasible per SLM-294)",
                "device": "cpu",
                "run_class": "scratch_matrix",
            },
            "x22_deterministic": {
                "path": "slm155 minimal seed (arm A) vs inventory-derived certified-minimal skeleton (arm B) + simulated conflict-slice repair",
                "equal_edit_budget_forwards": X22_EDIT_BUDGET,
                "note": X22_SIMULATED_REPAIR_NOTE,
            },
        },
        "primary_metric": "binding_aware_meaningful_v2 strict verdict (explicit arm request)",
        "min_useful_paired_delta": MIN_USEFUL_PAIRED_DELTA,
        "primary_population": "OBSERVABLE_REQUEST_ONLY + UNKNOWN rows (paired B - A)",
        "no_regression_population": "OBSERVABLE_PROMPT rows (paired B - A must not be negative beyond noise)",
        "rico": {"status": "not_run", "evidence": RICO_NOT_RUN_EVIDENCE},
        "claim_class": "diagnostic",
    }


def _score_prediction(pred: str, record: ExampleRecord, request: Any) -> dict[str, Any]:
    from slm_training.harnesses.model_build.eval_runner import (
        _placeholder_fidelity,
        structural_similarity,
    )

    try:
        validate(pred)
        syntax_valid = True
    except Exception:  # noqa: BLE001 - parse fact, never raised
        syntax_valid = False
    report = binding_aware_meaningful_v2(pred, record=record, request=request)
    return {
        "syntax_valid": syntax_valid,
        "v2_strict": bool(report.verdict),
        "v2_reason_codes": list(report.reason_codes),
        "v2_provenance": [dict(p) for p in report.prompt_contract.provenance],
        "placeholder_fidelity": _placeholder_fidelity(pred, record),
        "structural_similarity": structural_similarity(pred, record.openui),
    }


def _build_ar_plugin() -> Any:
    from slm_training.harnesses.model_build.config import ModelBuildConfig
    from slm_training.harnesses.model_build.data import load_train_records
    from slm_training.harnesses.model_build.factory import build_model

    config = ModelBuildConfig(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        run_id="slm301_prompt_observability_ar",
        device="cpu",
        run_class="scratch_matrix",
        seed=0,
    )
    records = load_train_records(TRAIN_DIR)
    plugin = build_model(config, records, checkpoint=AR_CHECKPOINT)
    if not callable(getattr(plugin, "generate_batch_requests", None)):
        raise RuntimeError("AR plugin lacks generate_batch_requests")
    return plugin


def _arm_scoring_record(record: ExampleRecord, arm: str, req: Any) -> ExampleRecord:
    """Record the v2 evaluator sees for this arm.

    ``_prompt_contract`` reads the visible inventory from ``record.prompt``;
    for arm B the record must carry the inventoried prompt so provenance
    shows prompt-visible coverage instead of request-only coverage.
    """
    if arm == "B" and req.prompt != record.prompt.strip():
        return dataclasses.replace(record, prompt=req.prompt)
    return record


def _decode_worker(record: ExampleRecord, arm: str, max_len: int, queue: Any) -> None:
    """Worker: build the plugin fresh (cheap, ~4s) and decode one request."""
    req_a, req_b = build_matched_requests(record)
    req = req_a if arm == "A" else req_b
    try:
        plugin = _build_ar_plugin()
        t0 = time.perf_counter()
        pred = plugin.generate_batch_requests(
            [req], max_len=max_len, grammar_constrained=False
        )[0]
        queue.put(
            {
                "status": "ok",
                "elapsed_s": round(time.perf_counter() - t0, 2),
                "prediction": pred,
            }
        )
    except Exception as exc:  # noqa: BLE001 - decode failure is data
        queue.put({"status": "decode_error", "error": str(exc)[:400]})


def _decode_one(
    record: ExampleRecord, arm: str, *, max_len: int, decode_timeout_s: float
) -> dict[str, Any]:
    """Decode one (record, arm) with a HARD wall-clock kill.

    The trained slm294 checkpoint decodes at ~17s+/token (per-token lark
    lexer rebuild, SLM-294), so a soft post-hoc check can burn ~20 min on a
    single 64-token decode. The worker is terminated at the budget and the
    row is marked ``decode_timeout`` — honest, never evidence.
    """
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    queue: Any = ctx.Queue()
    proc = ctx.Process(target=_decode_worker, args=(record, arm, max_len, queue))
    t0 = time.perf_counter()
    proc.start()
    proc.join(decode_timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(30)
        return {
            "status": "decode_timeout",
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "prediction": None,
            "scores": None,
        }
    result = queue.get() if not queue.empty() else {
        "status": "decode_error",
        "error": f"worker exited without result (code {proc.exitcode})",
    }
    if result["status"] != "ok":
        return {
            "status": result["status"],
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "error": result.get("error"),
            "prediction": None,
            "scores": None,
        }
    req_a, req_b = build_matched_requests(record)
    req = req_a if arm == "A" else req_b
    return {
        "status": "ok",
        "elapsed_s": result["elapsed_s"],
        "prediction": result["prediction"],
        "scores": _score_prediction(
            result["prediction"], _arm_scoring_record(record, arm, req), req
        ),
    }


def _run_ar_arm(
    plugin: Any,
    records: list[ExampleRecord],
    *,
    max_len: int,
    decode_timeout_s: float,
) -> dict[str, dict[str, Any]]:
    """Per-record AR decode for both arms; keyed {record_id: {arm: outcome}}."""
    del plugin  # decodes run in hard-killed worker processes (see _decode_one)
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        row: dict[str, Any] = {}
        for arm in ("A", "B"):
            row[arm] = _decode_one(
                record, arm, max_len=max_len, decode_timeout_s=decode_timeout_s
            )
        out[record.id] = row
    return out


def _x22_final_program(request: Any, record_id: str, *, seed: int) -> tuple[str, str]:
    """Deterministic X22 path: prompt-visible inventory drives the seed.

    Arm A (no visible inventory) starts from the SLM-155 minimal seed; arm B
    (visible inventory) starts from the certified-minimal slot skeleton built
    from the prompt-visible inventory. Equal simulated conflict-slice repair
    budget on both; the seed source is retained as the final program.
    """
    visible = inventory_from_prompt(request.prompt, heuristic=False)
    if visible:
        source = build_slot_contract_template(visible)
        note = f"{X22_SIMULATED_REPAIR_NOTE}; seed from prompt-visible inventory"
    else:
        source = _build_minimal_seed_source()
        note = X22_SIMULATED_REPAIR_NOTE
    slice_ = _build_conflict_slice(source, "minimal", record_id, seed)
    if slice_ is not None:
        apply_repair_policy(
            _ast_to_topology(validate(source).root),
            slice_,
            "conflict_slice",
            seed=seed,
            budget_forwards=X22_EDIT_BUDGET,
            budget_verifier_calls=16,
        )
    return source, note


def _run_x22_arm(records: list[ExampleRecord], *, seed: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        req_a, req_b = build_matched_requests(record)
        row: dict[str, Any] = {}
        for arm, req in (("A", req_a), ("B", req_b)):
            source, note = _x22_final_program(req, record.id, seed=seed)
            row[arm] = {
                "status": "ok",
                "elapsed_s": 0.0,
                "prediction": source,
                "program_note": note,
                "scores": _score_prediction(
                    source, _arm_scoring_record(record, arm, req), req
                ),
            }
        out[record.id] = row
    return out


def _summarize_model(
    outcomes: dict[str, dict[str, Any]],
    phase0_by_id: dict[str, str],
) -> dict[str, Any]:
    """Overall + coverage-conditioned rates, Wilson 95%, paired A-vs-B table."""
    summary: dict[str, Any] = {}
    for arm in ("A", "B"):
        for cls in (None, OBSERVABLE_PROMPT, OBSERVABLE_REQUEST_ONLY, UNKNOWN):
            ids = [
                rid
                for rid in outcomes
                if cls is None or phase0_by_id.get(rid) == cls
            ]
            measured = [
                outcomes[rid][arm]
                for rid in ids
                if outcomes[rid].get(arm, {}).get("status") == "ok"
            ]
            passes = sum(1 for o in measured if o["scores"]["v2_strict"])
            key = f"arm_{arm}" + ("" if cls is None else f".{cls.lower()}")
            syntax = [o for o in measured if o["scores"]["syntax_valid"]]
            fidelities = [
                o["scores"]["placeholder_fidelity"]
                for o in measured
                if o["scores"]["placeholder_fidelity"] is not None
            ]
            summary[key] = {
                "n": len(ids),
                "n_measured": len(measured),
                "n_timeout_or_unrun": len(ids) - len(measured),
                "v2_strict_rate": wilson_interval(passes, len(measured)),
                "syntax_valid_rate": wilson_interval(len(syntax), len(measured)),
                "mean_placeholder_fidelity": (
                    round(sum(fidelities) / len(fidelities), 4) if fidelities else None
                ),
                "mean_structural_similarity": (
                    round(
                        sum(o["scores"]["structural_similarity"] for o in measured)
                        / len(measured),
                        4,
                    )
                    if measured
                    else None
                ),
                "reason_prevalence": dict(
                    sorted(
                        Counter(
                            reason
                            for o in measured
                            for reason in o["scores"]["v2_reason_codes"]
                        ).items()
                    )
                ),
            }
    # Paired tables: primary population vs no-regression population.
    for label, classes in (
        ("target", {OBSERVABLE_REQUEST_ONLY, UNKNOWN}),
        ("no_regression", {OBSERVABLE_PROMPT}),
        ("all", None),
    ):
        pairs = []
        for rid, row in outcomes.items():
            if classes is not None and phase0_by_id.get(rid) not in classes:
                continue
            a = row.get("A", {})
            b = row.get("B", {})
            pairs.append(
                (
                    rid,
                    a["scores"]["v2_strict"] if a.get("status") == "ok" else None,
                    b["scores"]["v2_strict"] if b.get("status") == "ok" else None,
                )
            )
        summary[f"paired.{label}"] = {
            **paired_delta(pairs),
            "table": paired_outcome_table(pairs),
            "rows": [
                {"id": rid, "arm_a": a, "arm_b": b} for rid, a, b in pairs
            ],
        }
    return summary


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SLM-301 (LAR1-04): prompt observability — as-is vs +inventory arms",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- claim_class: `{payload['predeclaration']['claim_class']}` (diagnostic; fixture-scale, not ship evidence)",
        f"- predeclared: n={payload['predeclaration']['n_test_seeds']} test_seeds, "
        f"seeds={payload['predeclaration']['seeds']}, "
        f"min useful paired delta={payload['predeclaration']['min_useful_paired_delta']} v2-strict "
        f"on OBSERVABLE_REQUEST_ONLY/UNKNOWN rows",
        f"- rico: **not_run** — {payload['predeclaration']['rico']['evidence']}",
        "",
        "## Phase 0: observability coverage classes (model-free)",
        "",
        "| corpus | n | OBSERVABLE_PROMPT | OBSERVABLE_REQUEST_ONLY | UNKNOWN |",
        "| --- | --- | --- | --- | --- |",
    ]
    for corpus, report in payload["phase0"].items():
        hist = report["class_histogram"]
        lines.append(
            f"| {corpus} | {report['n']} | {hist.get(OBSERVABLE_PROMPT, 0)} | "
            f"{hist.get(OBSERVABLE_REQUEST_ONLY, 0)} | {hist.get(UNKNOWN, 0)} |"
        )
    lines += ["", "### Per-row classes", ""]
    for corpus, report in payload["phase0"].items():
        lines.append(f"- **{corpus}**:")
        for row in report["rows"]:
            lines.append(
                f"  - `{row['id']}` {row['coverage_class']} "
                f"(inventory_in_prompt={row['prompt_has_inventory']}, "
                f"slots={row['n_slot_contract']}, decode={row['decode']})"
            )
    lines += ["", "## Phase 1: decode outcomes (16 test_seeds, seed 0)", ""]
    for model, summary in payload["phase1"].items():
        lines += [f"### {model}", ""]
        if summary.get("status") == "not_run":
            lines.append(f"- not_run: {summary['reason']}")
            continue
        if summary.get("disposition"):
            lines += [
                f"- **disposition: `{summary['disposition']}`** — {summary['disposition_evidence']}",
                "",
            ]
        lines += [
            "| slice | n_measured | v2-strict (Wilson 95%) | syntax |",
            "| --- | --- | --- | --- |",
        ]
        for key, row in summary.items():
            if not key.startswith("arm_"):
                continue
            rate = row["v2_strict_rate"]
            est = rate["estimate"]
            txt = (
                f"{est:.3f} [{rate['low']:.3f}, {rate['high']:.3f}]"
                if est is not None
                else "unmeasured"
            )
            syn = row["syntax_valid_rate"]["estimate"]
            lines.append(
                f"| {key} | {row['n_measured']}/{row['n']} | {txt} | "
                f"{syn if syn is not None else '—'} |"
            )
        lines.append("")
        for label in ("target", "no_regression", "all"):
            paired = summary[f"paired.{label}"]
            lines.append(
                f"- paired **{label}**: delta={paired['delta']} "
                f"(min useful {paired['min_useful_delta']}, "
                f"meets={paired['meets_min_useful_delta']}, "
                f"n_measured={paired['n_measured']}) table={paired['table']}"
            )
        lines.append("")
    if payload.get("agentv"):
        lines += ["## AgentEvals", "", f"```json\n{json.dumps(payload['agentv'], indent=2, sort_keys=True, default=str)}\n```", ""]
    return "\n".join(lines)


def _tag_ar_disposition(summary: dict[str, Any]) -> None:
    """If every AR row timed out, declare the arm decode-infeasible honestly."""
    per_record = summary.get("per_record") or {}
    statuses = [
        row[arm]["status"]
        for row in per_record.values()
        for arm in ("A", "B")
        if arm in row
    ]
    if statuses and not any(s == "ok" for s in statuses):
        summary["disposition"] = "not_run_decode_infeasible"
        summary["disposition_evidence"] = (
            f"all {len(statuses)} decodes exceeded the 60s hard budget at "
            "max_len=128 and were killed (decode_timeout); a max_len=64 probe "
            "on smoke_hero_01 also exceeded 60s in both arms. Root cause per "
            "SLM-294: per-token lark lexer rebuild in build_completion_forest "
            "(>280s for 16 tokens at rico scale with this checkpoint). No AR "
            "row is evidence; X22 deterministic arm + phase-0 classification "
            "carry the experiment."
        )


SHARD_DIR = Path("outputs/runs/slm301_prompt_observability/shards")


def _shard_path(index: int) -> Path:
    return SHARD_DIR / f"shard_{index}.json"


def _finalize_payload(payload: dict[str, Any]) -> None:
    """Version stamp + AgentEvals + durable docs for a complete payload."""
    stamp = build_version_stamp(COMPONENT)
    payload["version_stamp"] = stamp

    # AgentEvals (diagnostic claim; publication failure is non-fatal).
    agentv: Any = None
    try:
        from slm_training.evals.agentv import publish_agentv_evaluation

        cases = []
        for model, summary in payload["phase1"].items():
            if summary.get("status") == "not_run":
                continue
            target = summary["paired.target"]
            noregress = summary["paired.no_regression"]
            cases.append(
                {
                    "id": f"{model}_paired_target",
                    "criteria": (
                        "paired arm-B delta on OBSERVABLE_REQUEST_ONLY/UNKNOWN "
                        "rows is measured against the predeclared 0.10 minimum"
                    ),
                    "pass": target["n_measured"] > 0,
                    "failures": [] if target["n_measured"] > 0 else ["unmeasured"],
                    "result": target,
                    "metadata": {"model": model},
                }
            )
            if noregress["n_measured"]:
                ok = noregress["delta"] is not None and noregress["delta"] >= 0
                cases.append(
                    {
                        "id": f"{model}_no_regression",
                        "criteria": "arm B does not regress OBSERVABLE_PROMPT rows",
                        "pass": ok,
                        "failures": [] if ok else ["observable_row_regression"],
                        "result": noregress,
                        "metadata": {"model": model},
                    }
                )
        agentv = publish_agentv_evaluation(
            Path("outputs/runs/slm301_prompt_observability"),
            name="slm301-prompt-observability",
            claim="diagnostic",
            cases=cases,
            version_stamp=stamp,
        )
    except Exception as exc:  # noqa: BLE001 - AgentV must not sink the report
        agentv = {"status": "agentv_publication_failed", "error": str(exc)}
    if agentv is not None:
        # Repo policy: durable docs must not embed machine-absolute paths.
        root = str(Path.cwd()) + "/"
        agentv = json.loads(json.dumps(agentv, default=str).replace(root, ""))
    payload["agentv"] = agentv


def _write_docs(payload: dict[str, Any], json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    md_out.write_text(render_markdown(payload), encoding="utf-8")


def _print_summary(payload: dict[str, Any], json_out: Path, md_out: Path) -> None:
    phase0 = payload["phase0"]
    print(f"phase0: {json.dumps({k: v['class_histogram'] for k, v in phase0.items()})}")
    for model, summary in payload["phase1"].items():
        if summary.get("status") == "not_run":
            print(f"{model}: not_run ({summary['reason']})")
        else:
            print(
                f"{model}: paired target delta={summary['paired.target']['delta']} "
                f"table={summary['paired.target']['table']}"
            )
    print(f"wrote {json_out} and {md_out}")


def _merge_shards(
    num_shards: int,
    *,
    json_out: Path,
    md_out: Path,
) -> int:
    """Combine per-shard raw outcomes into the final report + docs."""
    phase0: dict[str, Any] | None = None
    merged: dict[str, dict[str, Any]] = {}
    for index in range(num_shards):
        path = _shard_path(index)
        if not path.exists():
            print(f"missing shard file: {path}")
            return 1
        raw = json.loads(path.read_text(encoding="utf-8"))
        phase0 = phase0 or raw["phase0"]
        for model, outcomes in raw["outcomes"].items():
            if outcomes is None:
                merged.setdefault(model, {"status": "not_run", "reason": raw.get("ar_not_run_reason", "shard not_run")})
                continue
            bucket = merged.setdefault(model, {})
            if "status" in bucket:
                continue
            bucket.update(outcomes)
    assert phase0 is not None
    phase0_by_id = {
        row["id"]: row["coverage_class"] for row in phase0["test_seeds"]["rows"]
    }
    payload: dict[str, Any] = {
        "schema": "slm301_prompt_observability/v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "predeclaration": _predeclaration(),
        "phase0": phase0,
        "shard": {"index": "merged", "num_shards": num_shards},
        "phase1": {},
    }
    for model, outcomes in merged.items():
        if "status" in outcomes:
            payload["phase1"][model] = outcomes
            continue
        summary = _summarize_model(outcomes, phase0_by_id)
        summary["per_record"] = outcomes
        if model == "ar_tiny":
            _tag_ar_disposition(summary)
        payload["phase1"][model] = summary
    _finalize_payload(payload)
    _write_docs(payload, json_out, md_out)
    _print_summary(payload, json_out, md_out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-len", type=int, default=128, help="AR decode token cap")
    parser.add_argument(
        "--decode-timeout-s",
        type=float,
        default=60.0,
        help="per-record decode budget; overruns are marked decode_timeout",
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--merge-shards",
        type=int,
        default=None,
        metavar="N",
        help="merge N shard files from previous --shard runs into the final report",
    )
    parser.add_argument("--skip-ar", action="store_true", help="phase 0 + X22 only")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    if args.merge_shards is not None:
        return _merge_shards(args.merge_shards, json_out=args.json_out, md_out=args.md_out)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    test_seeds = _load_records(TEST_SEEDS, default_split="smoke")
    rico = _load_records(RICO_CORPUS, default_split="held_out")

    # Phase 0 (model-free).
    phase0 = {
        "test_seeds": observability_report(test_seeds, decode="phase1"),
        "slm294_rico": observability_report(rico, decode="not_run"),
    }
    phase0_by_id = {
        row["id"]: row["coverage_class"] for row in phase0["test_seeds"]["rows"]
    }

    shard_records = [
        record
        for index, record in enumerate(test_seeds)
        if index % args.num_shards == args.shard
    ]

    # X22 deterministic baseline (cheap, both arms).
    outcomes: dict[str, Any] = {"x22_deterministic": _run_x22_arm(shard_records, seed=0)}

    # AR tiny baseline.
    ar_not_run_reason: str | None = None
    if args.skip_ar or not AR_CHECKPOINT.exists():
        ar_not_run_reason = (
            "--skip-ar" if args.skip_ar else f"checkpoint missing: {AR_CHECKPOINT}"
        )
        outcomes["ar_tiny"] = None
    else:
        outcomes["ar_tiny"] = _run_ar_arm(
            None,
            shard_records,
            max_len=args.max_len,
            decode_timeout_s=args.decode_timeout_s,
        )

    if args.num_shards > 1:
        # Shard run: persist raw per-record outcomes; --merge-shards builds
        # the durable report (never let a partial shard overwrite the docs).
        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        raw = {
            "phase0": phase0,
            "shard": {"index": args.shard, "num_shards": args.num_shards},
            "ar_not_run_reason": ar_not_run_reason,
            "outcomes": outcomes,
        }
        path = _shard_path(args.shard)
        path.write_text(json.dumps(raw, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"shard {args.shard}/{args.num_shards}: wrote {path}")
        return 0

    # Payload skeleton: predeclaration is written BEFORE any results.
    payload: dict[str, Any] = {
        "schema": "slm301_prompt_observability/v1",
        "generated_at": generated_at,
        "predeclaration": _predeclaration(),
        "phase0": phase0,
        "shard": {"index": args.shard, "num_shards": args.num_shards},
        "phase1": {},
    }
    for model, model_outcomes in outcomes.items():
        if model_outcomes is None:
            payload["phase1"][model] = {
                "status": "not_run",
                "reason": ar_not_run_reason,
            }
            continue
        summary = _summarize_model(model_outcomes, phase0_by_id)
        summary["per_record"] = model_outcomes
        if model == "ar_tiny":
            _tag_ar_disposition(summary)
        payload["phase1"][model] = summary

    _finalize_payload(payload)
    _write_docs(payload, args.json_out, args.md_out)
    _print_summary(payload, args.json_out, args.md_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
