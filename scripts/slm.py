#!/usr/bin/env python3
"""Unified training-pipeline CLI: a thin dispatcher over the scripts/ mains.

``slm <phase> <action> [args...]`` forwards the remaining arguments verbatim
to the underlying ``scripts.<module>.main``; nothing is reimplemented here.
Passthrough phases (``preference``, ``cycle``, ``autoresearch``) own their own
subcommands, so their first forwarded token is that script's subcommand.

Meta commands: ``slm list`` (every command), ``slm guide <slug>`` (print the
matching train-skill reference; requires a repository checkout), ``slm --help``.

Forwarded ``--help``/usage errors keep the target script's prog name (for
example ``train_model`` rather than ``slm sft train``) — cosmetic only; the
behavior is identical to invoking ``python -m scripts.<module>`` directly.

Usage::

    slm list
    slm guide sft
    slm data build-train --source fixture --version v0 --synthesizer quality
    slm sft train --train-dir outputs/data/train/v1 --run-id twotower_v1
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCES = REPO_ROOT / ".agents" / "skills" / "autotrain" / "references"


@dataclass(frozen=True)
class Command:
    """One dispatchable phase action: target module, summary, guide slug."""

    module: str
    summary: str
    guide: str


GROUP_SUMMARIES: dict[str, str] = {
    "data": "Build, publish, and resolve versioned train/test data",
    "sft": "Phase A supervised model-build training (local, pod, HF Jobs)",
    "eval": "Honest multi-suite evaluation, ship gates, diagnostics",
    "distill": "Distillation and the P1-P3 self-distill climb",
    "preference": "Phase B preference learning (surrogate-DPO; passthrough)",
    "rl": "Phase C RL (GRPO-lite, fail-closed readiness) + external backends",
    "experiments": "Experiment matrices, scaling, mixtures, recipe evolution",
    "checkpoints": "Checkpoint bucket sync and format migration",
    "cycle": "Immutable model lifecycle (snapshot/branch/train/promote/...; passthrough)",
    "annotations": "Export human annotations for downstream training",
    "bench": "Benchmarks and generation profiling",
    "autoresearch": "Bounded self-improvement campaigns + RL gate (passthrough)",
}

COMMANDS: dict[tuple[str, ...], Command] = {
    ("data", "build-train"): Command(
        "scripts.build_train_data", "Build a versioned training corpus", "train-data"
    ),
    ("data", "build-test"): Command(
        "scripts.build_test_data",
        "Build disjoint held-out/adversarial/OOD suites",
        "test-data",
    ),
    ("data", "publish-train"): Command(
        "scripts.publish_train_data",
        "Publish an immutable train snapshot to Git",
        "train-data",
    ),
    ("data", "store"): Command(
        "scripts.data_store",
        "List/resolve/verify/publish canonical data roots",
        "train-data",
    ),
    ("sft", "train"): Command(
        "scripts.train_model", "Supervised train from a versioned snapshot", "sft"
    ),
    ("sft", "remote"): Command(
        "scripts.remote_train", "Run the train on a remote pod", "sft"
    ),
    ("sft", "hf-jobs"): Command(
        "scripts.hf_jobs_train", "Run the train as a managed HF Jobs GPU job", "sft"
    ),
    ("eval", "model"): Command(
        "scripts.evaluate_model", "Multi-suite scoreboard with --ship-gates", "eval"
    ),
    ("eval", "diagnose"): Command(
        "scripts.diagnose_eval", "Decode diagnostics for a checkpoint", "eval"
    ),
    ("eval", "loss-suites"): Command(
        "scripts.evaluate_loss_suites", "Loss suites (base + OOD)", "eval"
    ),
    ("eval", "tasks"): Command(
        "scripts.evaluate_tasks", "AgentEvals task cases to a scoreboard", "eval"
    ),
    ("distill", "collect"): Command(
        "scripts.collect_trajectories", "Collect decision trajectories", "distill"
    ),
    ("distill", "self"): Command(
        "scripts.self_distill", "Select traces / SFT from selected traces", "distill"
    ),
    ("distill", "resume-climb"): Command(
        "scripts.resume_climb", "P3 rollouts + gated trajectory RL", "distill"
    ),
    ("preference",): Command(
        "scripts.train_preference",
        "Pairs + surrogate-DPO (build-pairs/train/train-events/train-local)",
        "preference",
    ),
    ("rl", "train"): Command(
        "scripts.train_rl", "GRPO-lite RL (approved readiness report required)", "rl"
    ),
    ("rl", "nemo"): Command(
        "scripts.run_nemo_rl", "External NVIDIA NeMo-RL wrapper (env-driven)", "rl"
    ),
    ("rl", "molt"): Command(
        "scripts.run_molt_rl", "External MOLT RL wrapper (env-driven)", "rl"
    ),
    ("experiments", "quality-matrix"): Command(
        "scripts.run_quality_matrix", "Quality matrix (E*)", "experiments"
    ),
    ("experiments", "grammar-matrix"): Command(
        "scripts.run_grammar_matrix", "Grammar matrix (X*)", "experiments"
    ),
    ("experiments", "perf-matrix"): Command(
        "scripts.run_perf_matrix", "Perf matrix (P/Q/R)", "experiments"
    ),
    ("experiments", "phase-pipeline"): Command(
        "scripts.run_phase_pipeline", "Phases A->B->C to completion", "experiments"
    ),
    ("experiments", "scaling-ladder"): Command(
        "scripts.run_scaling_ladder", "Scaling ladder arms", "experiments"
    ),
    ("experiments", "mixture-search"): Command(
        "scripts.run_mixture_search", "Data mixture search", "experiments"
    ),
    ("experiments", "recipe-evolution"): Command(
        "scripts.run_recipe_evolution", "Recipe evolution under frozen gates", "experiments"
    ),
    ("experiments", "reproduce-baseline"): Command(
        "scripts.reproduce_baseline", "Reproduce the documented baseline", "experiments"
    ),
    ("checkpoints", "sync"): Command(
        "scripts.sync_checkpoints", "Sync run checkpoints to the OpenUI bucket", "checkpoints"
    ),
    ("checkpoints", "migrate"): Command(
        "scripts.migrate_checkpoint", "Migrate a checkpoint format", "checkpoints"
    ),
    ("cycle",): Command(
        "scripts.model_cycle",
        "Lifecycle subcommands (snapshot-data/init/branch/train/evaluate/promote/...)",
        "checkpoints",
    ),
    ("annotations", "export"): Command(
        "scripts.export_annotations", "Export annotations to pairs/records", "annotations"
    ),
    ("bench", "telemetry"): Command(
        "scripts.bench_telemetry", "Telemetry decode bench", "bench"
    ),
    ("bench", "accel"): Command(
        "scripts.bench_accel", "Acceleration comparison / microbench", "bench"
    ),
    ("bench", "cactus"): Command(
        "scripts.bench_cactus", "Cactus export bench", "bench"
    ),
    ("bench", "profile"): Command(
        "scripts.profile_generate", "Generation hot-path profile", "bench"
    ),
    ("autoresearch",): Command(
        "scripts.autoresearch",
        "Campaign subcommands (init/research/hypothesize/run/diagnose/validate-rl/...)",
        "autoresearch",
    ),
}


def _print_usage() -> None:
    print(__doc__.strip())
    print()
    print("phases:")
    for group, summary in GROUP_SUMMARIES.items():
        print(f"  {group:<13} {summary}")
    print()
    print("meta: slm list | slm guide <slug> | slm <phase> [-h]")


def _cmd_list() -> int:
    width = max(len(" ".join(key)) for key in COMMANDS)
    for group, summary in GROUP_SUMMARIES.items():
        print(f"{group}: {summary}")
        for key, command in COMMANDS.items():
            if key[0] != group:
                continue
            name = " ".join(key)
            print(f"  slm {name:<{width}}  {command.summary}  ({command.module})")
    return 0


def _guide_slugs() -> list[str]:
    return sorted(path.stem for path in REFERENCES.glob("*.md"))


def _cmd_guide(args: list[str]) -> int:
    if not REFERENCES.is_dir():
        print(
            "slm guide requires a repository checkout "
            "(missing .agents/skills/autotrain/references)",
            file=sys.stderr,
        )
        return 2
    slugs = _guide_slugs()
    if not args:
        print("usage: slm guide <slug>")
        print("available: " + " ".join(slugs))
        return 0
    slug = args[0]
    path = REFERENCES / f"{slug}.md"
    if not path.is_file():
        print(f"unknown guide '{slug}'; available: " + " ".join(slugs), file=sys.stderr)
        return 2
    print(path.read_text(encoding="utf-8"), end="")
    return 0


def _print_group(group: str) -> None:
    print(f"{group}: {GROUP_SUMMARIES[group]}")
    for key, command in COMMANDS.items():
        if key[0] != group:
            continue
        suffix = " <args...>" if len(key) == 1 else ""
        print(f"  slm {' '.join(key)}{suffix}  {command.summary}  ({command.module})")


def _dispatch(command: Command, rest: list[str]) -> int:
    module = importlib.import_module(command.module)
    entry = module.main
    try:
        if inspect.signature(entry).parameters:
            result = entry(list(rest))
        else:
            # A few wrappers (run_nemo_rl, run_molt_rl, run_recipe_evolution)
            # take no arguments and read sys.argv themselves.
            saved = sys.argv
            sys.argv = [command.module.rsplit(".", 1)[-1], *rest]
            try:
                result = entry()
            finally:
                sys.argv = saved
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 1
    return 0 if result is None else int(result)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if not args or args[0] in {"-h", "--help", "help"}:
            _print_usage()
            return 0
        if args[0] == "list":
            return _cmd_list()
        if args[0] == "guide":
            return _cmd_guide(args[1:])
    except BrokenPipeError:
        # `slm list | head` closes stdout early; exit quietly like a good pipe
        # citizen (dispatched scripts keep their own behavior).
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    if len(args) >= 2 and tuple(args[:2]) in COMMANDS:
        return _dispatch(COMMANDS[tuple(args[:2])], args[2:])
    if tuple(args[:1]) in COMMANDS:
        return _dispatch(COMMANDS[tuple(args[:1])], args[1:])
    if args[0] in GROUP_SUMMARIES:
        _print_group(args[0])
        if len(args) >= 2 and args[1] in {"-h", "--help"}:
            return 0
        print(
            f"unknown action for '{args[0]}'"
            + (f": '{args[1]}'" if len(args) >= 2 else " (action required)"),
            file=sys.stderr,
        )
        return 2
    print(f"unknown command: '{args[0]}' — try 'slm list'", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
