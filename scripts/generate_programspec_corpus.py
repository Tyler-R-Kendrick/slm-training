#!/usr/bin/env python3
"""ProgramSpec corpus generation CLI for SLM-267 (VSD2-02).

Two independent lines of work landed on this same path and were reconciled
into one CLI dispatched by ``--mode``:

``stream`` (default)
    Streaming, resumable full-scale corpus generation: a canonical CLI over
    the existing typed :class:`~slm_training.data.progspec.generate.ProgramGenerator`
    (``src/slm_training/data/progspec/generate.py``) that adds deterministic
    seeded generation, an immutable rejection ledger, a disk-backed exact
    dedup index keyed on the generator's own canonical ``ProgramSpec.id``,
    and a resumable shard state so repeated invocations (each bounded by the
    repo's hard run cap, ``slm_training.levers.MAX_RUN_MINUTES``) accumulate
    monotonically toward a ``--target-unique-roots`` count instead of
    restarting from zero. Only the frozen **uniform-valid** base-distribution
    policy is implemented here. The issue's second policy
    (**coverage-targeted**, driven by a ``CoverageGapManifestV1``) and the
    100k/1M rungs are explicitly deferred -- see
    ``docs/design/compiler-inverted-program-data.md`` for the measured
    saturation frontier this increment establishes and the follow-up scope.

    Determinism contract this mode relies on (locked by
    ``tests/test_scripts/test_generate_programspec_corpus.py``): for a fixed
    ``GeneratorConfig`` and ``seed``, ``ProgramGenerator`` visits candidates
    in an identical order and produces an identical accept/reject sequence on
    every invocation (no wall-clock, network, or OS-random input enters
    candidate selection or verification content). Resuming therefore replays
    the prior prefix of that sequence -- cheap in candidate count, not in
    verifier calls -- and continues past it; see the "Known limitation" note
    in the design doc.

``plan-only`` / ``fixture``
    The original SLM-267 coverage-scaling *wiring evidence* campaign: a
    bounded fixture comparing the generator's genuine uniform-random arm
    against its existing greedy coverage-maximizing arm
    (``slm_training.harnesses.experiments.slm267_programspec_coverage_scaling``).
    This is wiring evidence only -- it does not validate VSD-H7a/b/c, which
    require corpora and trained models several orders of magnitude larger
    than anything generated here.

Usage::

    python -m scripts.generate_programspec_corpus \\
        --pack openui --policy uniform \\
        --target-unique-roots 500 --seed 0 --shard-id 0 \\
        --dataset-id compiler_programs_uniform_smoke_v1

    python -m scripts.generate_programspec_corpus --mode plan-only
    python -m scripts.generate_programspec_corpus --mode fixture
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.data.progspec.generate import GeneratorConfig, ProgramGenerator
from slm_training.data.store import DataStore
from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.slm267_programspec_coverage_scaling import (
    DEFAULT_CONFIG,
    EXPERIMENT_ID,
    render_markdown,
    run_coverage_scaling_campaign,
)
from slm_training.levers import MAX_RUN_MINUTES

STATE_SCHEMA = "programspec_corpus_stream_state/v1"
MANIFEST_SCHEMA = "programspec_corpus_stream_manifest/v1"
SUPPORTED_PACKS = ("openui",)
SUPPORTED_POLICIES = ("uniform",)
_EXHAUSTED_MESSAGE = "candidate grid exhausted"

_FIXTURE_DESIGN_JSON = "docs/design/iter-slm267-programspec-coverage-scaling-20260725.json"
_FIXTURE_DESIGN_MD = "docs/design/iter-slm267-programspec-coverage-scaling-20260725.md"


def _wall_minutes(value: str) -> float:
    minutes = float(value)
    if not 0 < minutes <= MAX_RUN_MINUTES:
        raise argparse.ArgumentTypeError(
            f"must be positive and at most {MAX_RUN_MINUTES}"
        )
    return minutes


# --------------------------------------------------------------------------
# ``stream`` mode: streaming, resumable full-scale corpus generation.
# --------------------------------------------------------------------------


@dataclass
class StreamState:
    """Resumable per-(config, seed, shard) generation progress."""

    dataset_id: str
    pack: str
    policy: str
    seed: int
    shard_id: str
    max_depth: int
    max_width: int
    components: tuple[str, ...] | None
    target_unique_roots: int
    call_count: int = 0
    accepted_ids: list[str] = field(default_factory=list)
    rejected_count: int = 0
    exhausted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "dataset_id": self.dataset_id,
            "pack": self.pack,
            "policy": self.policy,
            "seed": self.seed,
            "shard_id": self.shard_id,
            "max_depth": self.max_depth,
            "max_width": self.max_width,
            "components": list(self.components) if self.components else None,
            "target_unique_roots": self.target_unique_roots,
            "call_count": self.call_count,
            "accepted_ids": self.accepted_ids,
            "rejected_count": self.rejected_count,
            "exhausted": self.exhausted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamState:
        components = data.get("components")
        return cls(
            dataset_id=str(data["dataset_id"]),
            pack=str(data["pack"]),
            policy=str(data["policy"]),
            seed=int(data["seed"]),
            shard_id=str(data["shard_id"]),
            max_depth=int(data["max_depth"]),
            max_width=int(data["max_width"]),
            components=tuple(components) if components else None,
            target_unique_roots=int(data["target_unique_roots"]),
            call_count=int(data.get("call_count", 0)),
            accepted_ids=list(data.get("accepted_ids", [])),
            rejected_count=int(data.get("rejected_count", 0)),
            exhausted=bool(data.get("exhausted", False)),
        )


def _config_identity(state: StreamState) -> tuple[Any, ...]:
    """Fields that must match exactly to resume (not merely append) a shard."""
    return (
        state.pack,
        state.policy,
        state.seed,
        state.shard_id,
        state.max_depth,
        state.max_width,
        state.components,
    )


def run_shard(
    state: StreamState,
    *,
    max_wall_minutes: float,
    record_sink: list[dict[str, Any]] | None = None,
    rejection_sink: list[dict[str, Any]] | None = None,
) -> tuple[StreamState, dict[str, Any]]:
    """Advance ``state`` by replaying + extending its deterministic sequence.

    Returns the updated state and a per-invocation run report (throughput,
    disposition). ``record_sink`` / ``rejection_sink``, when supplied, receive
    only the *newly* accepted / rejected rows from this invocation (the
    replayed prefix is skipped, not re-emitted).
    """
    config = GeneratorConfig(
        components=state.components,
        max_depth=state.max_depth,
        max_width=state.max_width,
    )
    generator = ProgramGenerator(config, seed=state.seed)
    candidate_grid_size = len(generator._candidates)  # noqa: SLF001 - read-only introspection
    seen_ids = set(state.accepted_ids)
    replay_target = state.call_count
    deadline = time.monotonic() + max_wall_minutes * 60
    calls_this_run = 0
    new_accepted = 0
    new_rejected = 0
    disposition = "in_progress"

    while True:
        if len(state.accepted_ids) >= state.target_unique_roots:
            disposition = "target_reached"
            break
        if time.monotonic() >= deadline:
            disposition = "wall_clock_stopped"
            break
        try:
            spec = generator.generate_one()
        except ValueError as exc:
            if str(exc) == _EXHAUSTED_MESSAGE:
                state.exhausted = True
                disposition = "saturated"
                break
            state.call_count += 1
            calls_this_run += 1
            if state.call_count > replay_target:
                state.rejected_count += 1
                new_rejected += 1
                if rejection_sink is not None:
                    rejection_sink.append(
                        {
                            "call_index": state.call_count,
                            "reason": str(exc),
                        }
                    )
            continue
        state.call_count += 1
        calls_this_run += 1
        if state.call_count > replay_target:
            if spec.id not in seen_ids:
                seen_ids.add(spec.id)
                state.accepted_ids.append(spec.id)
                new_accepted += 1
                if record_sink is not None:
                    record_sink.append(spec.to_dict())

    elapsed = (
        max_wall_minutes * 60 - max(0.0, deadline - time.monotonic())
        if calls_this_run
        else 0.0
    )
    report = {
        "candidate_grid_size": candidate_grid_size,
        "calls_this_run": calls_this_run,
        "new_accepted": new_accepted,
        "new_rejected": new_rejected,
        "cumulative_accepted": len(state.accepted_ids),
        "cumulative_rejected": state.rejected_count,
        "cumulative_calls": state.call_count,
        "exhausted": state.exhausted,
        "disposition": disposition,
        "elapsed_seconds": elapsed,
        "per_record_seconds": (elapsed / calls_this_run) if calls_this_run else None,
    }
    return state, report


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _manifest(state: StreamState, report: dict[str, Any]) -> dict[str, Any]:
    accepted_sha256 = hashlib.sha256(
        json.dumps(sorted(state.accepted_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": MANIFEST_SCHEMA,
        "dataset_id": state.dataset_id,
        "pack": state.pack,
        "policy": state.policy,
        "seed": state.seed,
        "shard_id": state.shard_id,
        "generator_config": {
            "max_depth": state.max_depth,
            "max_width": state.max_width,
            "components": list(state.components) if state.components else "all_pinned",
        },
        "target_unique_roots": state.target_unique_roots,
        "unique_roots": len(state.accepted_ids),
        "rejected_total": state.rejected_count,
        "calls_total": state.call_count,
        "candidate_grid_size": report["candidate_grid_size"],
        "exhausted": state.exhausted,
        "accepted_ids_sha256": accepted_sha256,
        "last_run_report": report,
        "version_stamp": build_version_stamp("data.progspec.corpus_stream"),
    }


def _run_stream(args: argparse.Namespace) -> int:
    if args.target_unique_roots is None or args.target_unique_roots <= 0:
        raise SystemExit("--target-unique-roots is required and must be positive in stream mode")
    if not args.dataset_id:
        raise SystemExit("--dataset-id is required in stream mode")

    store = DataStore()
    out_dir = store.path("programspec", args.dataset_id)
    state_path = out_dir / "state.json"
    records_path = out_dir / "records.jsonl"
    rejected_path = out_dir / "rejected.jsonl"
    manifest_path = out_dir / "manifest.json"

    components = (
        tuple(sorted(c.strip() for c in args.components.split(",") if c.strip()))
        if args.components
        else None
    )

    if state_path.is_file():
        state = StreamState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
        expected = StreamState(
            dataset_id=args.dataset_id,
            pack=args.pack,
            policy=args.policy,
            seed=args.seed,
            shard_id=args.shard_id,
            max_depth=args.max_depth,
            max_width=args.max_width,
            components=components,
            target_unique_roots=args.target_unique_roots,
        )
        if _config_identity(state) != _config_identity(expected):
            raise SystemExit(
                f"existing state at {state_path} was built with different "
                "pack/policy/seed/shard/config; choose a new --dataset-id or "
                "match the original arguments exactly"
            )
        state.target_unique_roots = args.target_unique_roots
    else:
        state = StreamState(
            dataset_id=args.dataset_id,
            pack=args.pack,
            policy=args.policy,
            seed=args.seed,
            shard_id=args.shard_id,
            max_depth=args.max_depth,
            max_width=args.max_width,
            components=components,
            target_unique_roots=args.target_unique_roots,
        )

    plan = {
        "dataset_id": args.dataset_id,
        "pack": args.pack,
        "policy": args.policy,
        "seed": args.seed,
        "shard_id": args.shard_id,
        "target_unique_roots": args.target_unique_roots,
        "resuming_from_accepted": len(state.accepted_ids),
        "out_dir": str(out_dir),
    }
    if args.describe:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    new_records: list[dict[str, Any]] = []
    new_rejections: list[dict[str, Any]] = []
    state, report = run_shard(
        state,
        max_wall_minutes=args.max_wall_minutes,
        record_sink=new_records,
        rejection_sink=new_rejections,
    )

    # Cumulative accepted rows accumulate across invocations: only the
    # current run's newly accepted ProgramSpecs are held in memory, and
    # prior rows already on disk are preserved by appending rather than
    # rewriting from scratch.
    if not records_path.is_file():
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.write_text("", encoding="utf-8")
    _append_jsonl(records_path, new_records)
    _append_jsonl(rejected_path, new_rejections)

    manifest = _manifest(state, report)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state_path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------
# ``plan-only`` / ``fixture`` modes: the original coverage-scaling wiring
# evidence campaign (SLM-267 first landing, PR #893).
# --------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_fixture_payload(
    mode: str, target_count: int, seed: int, shards: int
) -> dict[str, Any]:
    if mode == "plan-only":
        return {
            "schema": "Slm267ProgramspecCoverageScalingManifestV1",
            "experiment_id": EXPERIMENT_ID,
            "status": "plan_only",
            "claim_class": "wiring",
            "recipe": {
                "target_count": target_count,
                "seed": seed,
                "shards": shards,
                "components": list(DEFAULT_CONFIG.components or ()),
            },
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm267_programspec_coverage_scaling",
            ),
            "timestamp": _now(),
        }
    return run_coverage_scaling_campaign(target_count=target_count, seed=seed, shards=shards)


def _run_wiring_fixture(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or Path(
        f"outputs/runs/slm267-programspec-coverage-scaling-{_today_yyyymmdd()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _build_fixture_payload(args.mode, args.target_count, args.seed, args.shards)
    payload["schema"] = payload.get("schema", "Slm267ProgramspecCoverageScalingReportV1")
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "inconclusive")
    payload["timestamp"] = _now()
    if "version_stamp" not in payload:
        payload["version_stamp"] = build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm267_programspec_coverage_scaling",
        )

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm267_programspec_coverage_scaling_report.json"
    run_json.write_text(report_text, encoding="utf-8")

    if args.mode == "fixture":
        root = Path(__file__).resolve().parents[1]
        json_path = root / _FIXTURE_DESIGN_JSON
        md_path = root / _FIXTURE_DESIGN_MD
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(report_text, encoding="utf-8")
        md_path.write_text(render_markdown(payload), encoding="utf-8")

    print(str(run_json))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument(
        "--mode",
        choices=("stream", "plan-only", "fixture"),
        default="stream",
        help=(
            "stream: streaming/resumable full-scale corpus generation "
            "(default). plan-only/fixture: bounded coverage-scaling wiring "
            "evidence campaign."
        ),
    )

    # -- stream mode --
    parser.add_argument("--pack", choices=SUPPORTED_PACKS, default="openui")
    parser.add_argument("--policy", choices=SUPPORTED_POLICIES, default="uniform")
    parser.add_argument("--target-unique-roots", type=int, default=None)
    parser.add_argument("--shard-id", default="0")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-width", type=int, default=3)
    parser.add_argument(
        "--components",
        default=None,
        help="Comma-separated component allowlist (default: all pinned components).",
    )
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument(
        "--max-wall-minutes",
        type=_wall_minutes,
        default=float(MAX_RUN_MINUTES),
        help=f"Per-invocation wall budget (default and maximum: {MAX_RUN_MINUTES}).",
    )
    parser.add_argument("--describe", action="store_true", help="Print plan; do not run.")

    # -- shared --
    parser.add_argument("--seed", type=int, default=0, help="Global seed (default: 0).")

    # -- plan-only / fixture mode --
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for run artifacts (default: "
            "outputs/runs/slm267-programspec-coverage-scaling-<YYYYMMDD>)"
        ),
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=80,
        help="Bounded per-arm program budget (default: 80; fixture scale only).",
    )
    parser.add_argument(
        "--shards", type=int, default=2, help="Deterministic shard count (default: 2)."
    )

    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    if args.mode in ("plan-only", "fixture"):
        return _run_wiring_fixture(args)
    return _run_stream(args)


if __name__ == "__main__":
    sys.exit(main())
