"""G5 (SLM-37): meta-trace corpus — decode traces + harness verdicts, typed.

Schema + retention over artifacts the stack already writes (per the issue:
not new infrastructure). One typed record joins what a future DSL-generating
meta-model needs per example: the request, the decode configuration, the
emitted program, and the honest harness verdicts — with provenance back to
the source artifacts. Retention follows the campaign-store conventions
(local JSON tree is authoritative; `campaign.json` makes the tree
`sync_campaign`-mirrorable, dry by default).

Replay contract (documented boundary): records carry enough identity to
re-decode (`prompt`, `slot_contract`, `checkpoint_sha`, `model_kind`,
`decode_config`). Exact-output replay is guaranteed only for deterministic
decoders (`tree_edit_diffusion` — value-guided search, no sampling); MaskGIT
paths are replay-from-spec, not bit-exact, and are marked as such.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import Field

from slm_training.autoresearch.schemas import StrictModel

META_TRACE_SCHEMA_VERSION = 1


class MetaTraceRecord(StrictModel):
    """One meta-model training example harvested from run artifacts."""

    schema_version: int = META_TRACE_SCHEMA_VERSION
    # Identity / provenance
    run_id: str
    record_id: str
    dsl_id: str = "openui"
    trace_id: str | None = None
    traceparent: str | None = None
    source_artifacts: tuple[str, ...] = ()
    # Request
    prompt: str
    slot_contract: tuple[str, ...] = ()
    # Decode configuration (replay-from-spec identity)
    model_kind: str = "twotower"
    checkpoint_sha: str | None = None
    decode_config: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None
    deterministic_decode: bool = False
    # Outcome
    prediction: str = ""
    gold: str | None = None
    verdicts: dict[str, Any] = Field(default_factory=dict)
    # Optional per-step trajectory (from distill TraceStore rows, when present)
    trajectory: dict[str, Any] | None = None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - absent/corrupt artifacts are skipped
        return None


def harvest_run_dir(run_dir: Path, *, dsl_id: str = "openui") -> list[MetaTraceRecord]:
    """Join a run directory's existing artifacts into typed records.

    Sources: `trace.json` (W3C ids, optional), `matrix_result.json` /
    `scoreboard.json` (gate verdicts, optional), and every
    `eval_<suite>.json` `details[]` row (per-example prediction + metrics).
    Missing artifacts degrade gracefully — provenance records what was used.
    """
    run_dir = Path(run_dir)
    records: list[MetaTraceRecord] = []
    trace = _load_json(run_dir / "trace.json") or {}
    gates: dict[str, Any] = {}
    for name in ("matrix_result.json", "scoreboard.json"):
        payload = _load_json(run_dir / name)
        if payload and ("pass" in payload or "gates" in payload):
            block = payload.get("gates") if "gates" in payload else payload
            gates = {
                "pass": bool(block.get("pass")),
                "failures": list(block.get("failures") or []),
                "source": name,
            }
            break
    summary = _load_json(run_dir / "train_summary.json") or {}
    recipe = dict(summary.get("recipe") or {})
    checkpoint_sha = None
    checkpoint = summary.get("checkpoint")
    if checkpoint and Path(checkpoint).is_file():
        checkpoint_sha = _sha256_file(Path(checkpoint))

    for eval_path in sorted(run_dir.glob("eval_*.json")):
        payload = _load_json(eval_path)
        if not payload:
            continue
        suite = eval_path.stem.removeprefix("eval_")
        for row in payload.get("details") or []:
            if not isinstance(row, dict) or "id" not in row:
                continue
            verdicts = {
                key: row[key]
                for key in (
                    "parse_ok",
                    "syntax_parse_valid",
                    "placeholder_fidelity",
                    "structural_similarity",
                    "component_type_recall",
                    "reward_score",
                    "exact_match",
                    "target_score",
                )
                if key in row
            }
            verdicts["suite"] = suite
            if gates:
                verdicts["run_gates"] = gates
            records.append(
                MetaTraceRecord(
                    run_id=str(summary.get("run_id") or run_dir.name),
                    record_id=str(row["id"]),
                    dsl_id=dsl_id,
                    trace_id=trace.get("trace_id"),
                    traceparent=trace.get("traceparent"),
                    source_artifacts=(str(eval_path),),
                    prompt=str(row.get("prompt") or ""),
                    model_kind=str(summary.get("model") or "twotower"),
                    checkpoint_sha=checkpoint_sha,
                    decode_config={
                        key: recipe[key]
                        for key in ("learning_rate", "seed", "batch_size")
                        if key in recipe
                    },
                    seed=recipe.get("seed"),
                    prediction=str(row.get("prediction") or ""),
                    verdicts=verdicts,
                )
            )
    return records


def write_corpus(
    records: list[MetaTraceRecord], output_root: Path, campaign_id: str
) -> dict[str, Any]:
    """Persist the corpus: campaign.json + append-only traces.jsonl +
    manifest with per-line sha256 (campaign-store conventions; the local
    tree is authoritative and `sync_campaign`-compatible)."""
    root = Path(output_root) / campaign_id
    root.mkdir(parents=True, exist_ok=True)
    campaign_path = root / "campaign.json"
    if not campaign_path.exists():
        campaign_path.write_text(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "kind": "meta_trace_corpus",
                    "schema_version": META_TRACE_SCHEMA_VERSION,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    lines: list[str] = []
    shas: list[str] = []
    for record in records:
        line = record.model_dump_json()
        lines.append(line)
        shas.append(hashlib.sha256(line.encode("utf-8")).hexdigest())
    with (root / "traces.jsonl").open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path) or {
        "campaign_id": campaign_id,
        "schema_version": META_TRACE_SCHEMA_VERSION,
        "n_records": 0,
        "line_sha256": [],
    }
    manifest["n_records"] = int(manifest["n_records"]) + len(records)
    manifest["line_sha256"] = list(manifest["line_sha256"]) + shas
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_corpus(output_root: Path, campaign_id: str) -> list[MetaTraceRecord]:
    root = Path(output_root) / campaign_id
    records: list[MetaTraceRecord] = []
    path = root / "traces.jsonl"
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(MetaTraceRecord.model_validate_json(line))
    return records


def record_live_decode(
    *,
    run_id: str,
    record_id: str,
    dsl_id: str,
    prompt: str,
    slot_contract: tuple[str, ...],
    model_kind: str,
    checkpoint_path: Path,
    prediction: str,
    verdicts: dict[str, Any] | None = None,
    decode_config: dict[str, Any] | None = None,
    seed: int | None = None,
    deterministic_decode: bool = False,
) -> MetaTraceRecord:
    """Build a record straight from a live decode (fixture/replay path)."""
    return MetaTraceRecord(
        run_id=run_id,
        record_id=record_id,
        dsl_id=dsl_id,
        prompt=prompt,
        slot_contract=tuple(slot_contract),
        model_kind=model_kind,
        checkpoint_sha=_sha256_file(Path(checkpoint_path)),
        decode_config=dict(decode_config or {}),
        seed=seed,
        deterministic_decode=deterministic_decode,
        prediction=prediction,
        verdicts=dict(verdicts or {}),
        source_artifacts=(str(checkpoint_path),),
    )


def replay_trace(record: MetaTraceRecord, checkpoint_path: Path) -> str:
    """Re-decode a stored trace from its spec and return the output.

    Bit-exact reproduction is asserted only for records marked
    `deterministic_decode` (tree_edit_diffusion's value-guided search).
    The checkpoint is verified against the stored sha first (fail closed).
    """
    checkpoint_path = Path(checkpoint_path)
    actual_sha = _sha256_file(checkpoint_path)
    if record.checkpoint_sha and actual_sha != record.checkpoint_sha:
        raise ValueError(
            "checkpoint sha mismatch: trace was recorded against "
            f"{record.checkpoint_sha[:12]}…, got {actual_sha[:12]}…"
        )
    if record.model_kind != "tree_edit_diffusion":
        raise ValueError(
            f"replay supports deterministic decoders only; {record.model_kind!r} "
            "is replay-from-spec (not bit-exact) and must be re-run via its harness"
        )
    from slm_training.harnesses.model_build.plugin import GenerationRequest
    from slm_training.models.tree_edit_diffusion import TreeEditDiffusionModel

    model = TreeEditDiffusionModel.from_checkpoint(checkpoint_path, device="cpu")
    return model.generate_batch_requests(
        [
            GenerationRequest(
                prompt=record.prompt,
                slot_contract=tuple(record.slot_contract),
            )
        ]
    )[0]
