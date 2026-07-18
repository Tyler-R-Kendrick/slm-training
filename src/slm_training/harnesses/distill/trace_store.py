"""Append-only decode trajectory store + MaskGIT trace recorder.

Random masking is not the same distribution as progressive MaskGIT decoding;
closing that gap (trajectory-aligned SFT / RL) requires the *actual*
intermediate canvases, commit decisions, remask events, and their
probabilities under the rollout policy. This module persists them.

Every trace carries the identity needed to make it reusable later: policy
checkpoint SHA, decode-config hash, tokenizer/grammar versions, seed, reward
vector (not just a scalar), and accept/reject labels. Rollouts from different
checkpoints must never be mixed unlabeled.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Iterator

TRACE_VERSION = 3

# Config keys that change decode behavior (used for the decode-config hash).
_DECODE_KEYS = (
    "gen_steps",
    "grammar_constrained",
    "grammar_dsl",
    "grammar_top_k",
    "structural_bias",
    "grammar_ltr_repair",
    "grammar_ltr_max_tokens",
    "grammar_ltr_stages",
    "grammar_ltr_primary",
    "grammar_finalize_validate",
    "grammar_prefer_structural",
    "grammar_trust_model",
    "grammar_sample_decode",
    "grammar_sample_temperature",
    "grammar_block_decode",
    "grammar_block_size",
    "grammar_fastpath",
    "grammar_fastpath_mode",
    "parallel_unmask",
    "remask_ratio",
    "remask_policy",
    "remask_span",
    "remask_to_mask",
    "remask_use_gate",
    "remask_use_entropy",
    "core_perturb_frac",
    "suffix_rollback_window",
    "slot_contract_constrained_decode",
    "template_fill_decode",
    "honest_slot_contract",
    "best_of_n",
    "output_tokenizer",
    "use_symbol_table",
)


def decode_config_hash(config: Any) -> str:
    """Stable hash over decode-relevant configuration."""
    if is_dataclass(config) and not isinstance(config, type):
        raw = asdict(config)
    elif isinstance(config, dict):
        raw = dict(config)
    else:
        raw = {k: getattr(config, k) for k in _DECODE_KEYS if hasattr(config, k)}
    payload = {k: raw.get(k) for k in _DECODE_KEYS if k in raw}
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def checkpoint_sha(path: Path | str) -> str | None:
    """Content hash of a policy checkpoint file."""
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DecodeTraceRecorder:
    """Collects per-step decode events from TwoTower MaskGIT generation.

    Attach via ``model.trace_recorder = recorder`` before ``generate``; the
    decode path emits events with zero cost when no recorder is attached.
    Nested decode passes (e.g. the unconstrained retry) are recorded into the
    same trajectory with phase markers rather than resetting it.
    """

    def __init__(
        self, *, record_canvases: bool = True, record_support: bool = False
    ) -> None:
        self.record_canvases = bool(record_canvases)
        self.record_support = bool(record_support)
        self.meta: dict[str, Any] = {}
        self.steps: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.final: dict[str, Any] | None = None
        self.nfe = 0
        self.commit_count = 0
        self.repair_commit_count = 0
        self.remask_count = 0
        self._depth = 0
        # VSS1-04: mode-serialized solver certificates + aggregate counters the
        # replay validator needs (None until record_solver is called).
        self.solver: dict[str, Any] | None = None

    # ── model-side hooks ─────────────────────────────────────────────────

    def begin(self, **meta: Any) -> None:
        self._depth += 1
        if self._depth > 1:
            self.events.append({"kind": "nested_begin", "meta": meta})
            return
        self.meta.update(meta)

    def forward(self) -> None:
        """One denoiser forward evaluation (NFE accounting)."""
        self.nfe += 1

    def step(
        self,
        step: int,
        *,
        canvas: list[int] | None = None,
        unknown: list[bool] | None = None,
        commits: list[dict[str, Any]] | None = None,
        remasks: list[dict[str, Any]] | None = None,
    ) -> None:
        row: dict[str, Any] = {"step": step, "depth": self._depth}
        if self.record_canvases and canvas is not None:
            row["canvas"] = list(canvas)
            if unknown is not None:
                row["unknown_positions"] = [
                    i for i, u in enumerate(unknown) if u
                ]
        if commits:
            row["commits"] = commits
            self.commit_count += sum(
                1 for c in commits if c.get("phase") != "ltr_repair"
            )
            self.repair_commit_count += sum(
                1 for c in commits if c.get("phase") == "ltr_repair"
            )
        if remasks:
            row["remasks"] = remasks
            self.remask_count += sum(len(r.get("positions") or []) for r in remasks)
        self.steps.append(row)

    def event(self, kind: str, **payload: Any) -> None:
        self.events.append({"kind": kind, "depth": self._depth, **payload})

    def record_solver(self, solver_block: dict[str, Any]) -> None:
        """Attach the VSS1-04 solver certificate/counter block for replay.

        The typed solver transition events themselves are appended via
        ``event("solver_state"|...)``; this stores the mode-serialized (bounded)
        certificates and aggregate counters the replay validator cross-checks.
        Overwrites any prior block (one solver run per decode trace).
        """
        self.solver = dict(solver_block)

    def end(self, *, canvas: list[int] | None = None, text: str | None = None) -> None:
        self._depth = max(0, self._depth - 1)
        if self._depth > 0:
            self.events.append({"kind": "nested_end"})
            return
        self.final = {
            "canvas": list(canvas) if canvas is not None else None,
            "text": text,
        }

    # ── consumer-side assembly ───────────────────────────────────────────

    def finalize(
        self,
        *,
        final_text: str | None = None,
        reward: dict[str, Any] | None = None,
        labels: dict[str, Any] | None = None,
        **meta: Any,
    ) -> dict[str, Any]:
        final = dict(self.final or {})
        if final_text is not None:
            final["text"] = final_text
        trace = {
            "version": TRACE_VERSION,
            "meta": {**self.meta, **meta},
            "steps": self.steps,
            "events": self.events,
            "final": final,
            "counters": {
                "nfe": self.nfe,
                "commits": self.commit_count,
                "repair_commits": self.repair_commit_count,
                "remasked_positions": self.remask_count,
                "decode_steps": len(self.steps),
            },
            "reward": reward or {},
            "labels": labels or {},
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        if self.solver is not None:
            # VSS1-04 solver certificate/counter sidecar; absent on non-solver
            # traces so historical v1/v2 readers are unaffected.
            trace["solver"] = self.solver
        return trace


class TraceStore:
    """Append-only JSONL trajectory store with a manifest.

    Existing traces are never rewritten or deleted through this API; new
    rollouts only append. Consumers filter by the identity fields in each
    trace (checkpoint SHA, decode-config hash, labels).
    """

    def __init__(
        self,
        root: Path | str,
        *,
        run_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        from slm_training.runtime.telemetry import current_trace

        active = current_trace()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.traces_path = self.root / "traces.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.run_id = run_id or (active.run_id if active else None)
        self.trace_id = trace_id or (active.trace_id if active else None)
        self.span_id = span_id or (active.span_id if active else None)

    def __len__(self) -> int:
        if not self.traces_path.exists():
            return 0
        with self.traces_path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def append(self, trace: dict[str, Any]) -> str:
        index = len(self)
        digest = hashlib.sha256(
            json.dumps(trace, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        trajectory_id = f"{index:08d}-{digest[:12]}"
        row = {
            "trajectory_id": trajectory_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "run_id": self.run_id,
            **trace,
        }
        with self.traces_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._update_manifest(count=index + 1)
        return trajectory_id

    def iter_traces(self) -> Iterator[dict[str, Any]]:
        if not self.traces_path.exists():
            return
        with self.traces_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("version") == 1 and "trajectory_id" not in row:
                        row["trajectory_id"] = row.pop("trace_id", None)
                    yield row

    def iter_kind(self, kind: str) -> Iterator[dict[str, Any]]:
        """Filter the store by meta-model trace kind (decode rows = 'decode')."""
        for row in self.iter_traces():
            if row.get("kind", "decode") == kind:
                yield row

    def _update_manifest(self, *, count: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        manifest = {
            "kind": "trace_store",
            "version": TRACE_VERSION,
            "append_only": True,
            "count": count,
            "traces": str(self.traces_path.as_posix()),
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "updated_at": now,
        }
        if self.manifest_path.exists():
            try:
                prev = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                manifest["created_at"] = prev.get("created_at", now)
            except Exception:  # noqa: BLE001
                manifest["created_at"] = now
        else:
            manifest["created_at"] = now
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


# ── G5 (SLM-37): meta-model trace capture ────────────────────────────────
#
# The eventual DSL-generating meta-model trains on three trace kinds sharing
# one identity envelope (checkpoint SHA, decode-config hash, run/trace ids):
#   decode           — per-step canvases/commits/remasks (DecodeTraceRecorder)
#   harness_decision — a bounded decision a harness made (lever chosen,
#                      candidate accepted/rejected, gate verdict)
#   matrix_outcome   — one experiment row's scoreboard + pass/fail
# Schema documented in docs/design/meta-model-traces.md.


def record_harness_decision(
    store: TraceStore,
    *,
    harness: str,
    decision: str,
    inputs: dict[str, Any] | None = None,
    outcome: dict[str, Any] | None = None,
    **meta: Any,
) -> str:
    return store.append(
        {
            "version": TRACE_VERSION,
            "kind": "harness_decision",
            "harness": harness,
            "decision": decision,
            "inputs": inputs or {},
            "outcome": outcome or {},
            "meta": meta,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def record_matrix_outcome(
    store: TraceStore,
    matrix_result: dict[str, Any],
    *,
    matrix_set: str,
    **meta: Any,
) -> str:
    """Persist one quality/grammar-matrix experiment row as a trace."""
    if "id" not in matrix_result:
        raise ValueError("matrix result must carry its experiment id")
    return store.append(
        {
            "version": TRACE_VERSION,
            "kind": "matrix_outcome",
            "matrix_set": matrix_set,
            "experiment_id": matrix_result.get("id"),
            "run_id_source": matrix_result.get("run_id"),
            "passed": matrix_result.get("pass"),
            "failures": list(matrix_result.get("failures") or []),
            "suites": matrix_result.get("suites") or {},
            "meta": meta,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def record_grammar_decisions(
    store: TraceStore,
    records: Sequence[dict[str, Any]],
    **meta: Any,
) -> list[str]:
    """Persist grammar-state decision traces as a batch.

    Each record must already carry the CAP1-02 schema fields. The store appends
    them with the shared identity envelope and version.
    """
    ids: list[str] = []
    for record in records:
        ids.append(
            store.append(
                {
                    "version": TRACE_VERSION,
                    "kind": "grammar_decision",
                    **record,
                    "meta": meta,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        )
    return ids


def replay_violations(trace: dict[str, Any]) -> list[str]:
    """Check a decode trace's replay invariant; empty list = replayable.

    Each recorded step canvas must reflect that step's commits except where
    the same step's remasks (or a later EOS pad) removed them, and remasked
    positions must actually hold the mask id in the *next* recorded canvas.
    This certifies the (canvas, commits, remasks) stream is self-consistent,
    which is what meta-model training replays.
    """
    violations: list[str] = []
    steps = [s for s in trace.get("steps", []) if s.get("canvas") is not None]
    for row in steps:
        canvas = row["canvas"]
        remasked = {
            int(t)
            for remask in row.get("remasks", [])
            for t in (remask.get("positions") or [])
        }
        for commit in row.get("commits", []) or []:
            t, tid = int(commit["t"]), int(commit["id"])
            if t in remasked:
                continue
            # canvas[t] == 0 (pad) is legal: EOS truncation pads the tail
            # after commits within the same step.
            if t < len(canvas) and canvas[t] != tid and canvas[t] != 0:
                violations.append(
                    f"step {row.get('step')}: canvas[{t}]={canvas[t]} != commit {tid}"
                )
    final = (trace.get("final") or {}).get("canvas")
    if steps and final is None:
        violations.append("trace has steps but no final canvas")

    # VSS1-04: validate any solver-transition events via the solver replay
    # checker. Decode-only traces carry no such events, so this is a no-op there
    # and historical v1/v2 traces are unaffected.
    from slm_training.dsl.solver.replay import (
        SOLVER_EVENT_KINDS,
        solver_replay_violations,
    )

    solver_events = [
        e for e in trace.get("events", []) if e.get("kind") in SOLVER_EVENT_KINDS
    ]
    if solver_events:
        solver = trace.get("solver") or {}
        violations.extend(
            solver_replay_violations(
                solver_events,
                certificates=solver.get("certificates"),
                certificate_mode=solver.get("certificate_mode", "summary"),
                counters=solver.get("counters"),
            )
        )
    return violations


def trace_bucket_uri(run_id: str) -> str:
    """Meta-model traces live beside checkpoints in the existing bucket."""
    return f"hf://buckets/TKendrick/OpenUI/traces/{run_id}"


def sync_traces(
    root: Path | str,
    run_id: str,
    *,
    push: bool = False,
) -> dict[str, Any]:
    """Mirror a trace store into the bucket (same shape as sync_campaign)."""
    local = Path(root)
    if not (local / "manifest.json").is_file():
        raise FileNotFoundError(f"trace store not found: {local}")
    command = [
        "hf",
        "buckets",
        "sync",
        str(local),
        trace_bucket_uri(run_id),
        "--no-delete",
    ]
    if not push:
        return {
            "push": False,
            "command": command,
            "remote_uri": trace_bucket_uri(run_id),
        }
    import subprocess

    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return {
        "push": True,
        "command": command,
        "remote_uri": trace_bucket_uri(run_id),
        "stdout": completed.stdout.strip(),
    }
