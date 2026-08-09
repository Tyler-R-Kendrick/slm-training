#!/usr/bin/env python3
"""Seed-fanout screening on Hugging Face Jobs — strictly opt-in spend.

RC1 of ``docs/design/harness-evolution-architecture-review-20260809.md``:
local screening at n=3 smoke documents is statistically undecidable (the
minimum attainable two-sided p is 0.25 — no result can ever reject at
alpha=0.05). This launcher fans one candidate config out across ``--seeds N``
(default 8) independent HF Jobs, one seed per job, so screening deltas become
decidable before any promotion decision is attempted.

SPEND SAFETY (hard requirements, no interactive prompt):

- ``--dry-run`` is the DEFAULT. It prints the exact per-seed job specs,
  hardware flavor, job count, and estimated wall-clock — and never submits.
- Submitting requires ``--no-dry-run`` AND the literal flag
  ``--i-understand-this-costs-money`` AND an exported ``HF_TOKEN``.
- No other file in this repository may invoke this script — automation never
  calls it (enforced by ``tests/test_scripts/test_hf_jobs_screen_policy.py``).

EVIDENCE LAW: a timed out, interrupted, or killed run is never evidence.
Each job carries a hard ``--timeout {max_minutes}m`` in its spec; a job the
platform kills never publishes its per-seed result file, and the collector
marks that seed ``incomplete`` and EXCLUDES it from the mean, the sum/sumsq
accumulators, and the permutation p-value.

Submission machinery is shared with ``scripts.hf_jobs_train`` (whose CLI is
unchanged). See docs/design/hf-jobs-train.md, "Seed-fanout screening".
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shlex
from itertools import combinations
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from scripts.hf_jobs_train import (
    BUCKET_MOUNT,
    DEFAULT_BUCKET,
    DEFAULT_FLAVOR,
    DEFAULT_IMAGE,
    DEFAULT_REPO,
    build_entrypoint_script,
    build_jobs_run_command,
    hf_token_present,
    submit_jobs_command,
)
from slm_training.versioning import build_version_stamp

SCHEMA = "hf_jobs_screening/v1"
SEED_SCHEMA = "hf_jobs_screening_seed/v1"
#: Proposed versions.json component id (registration proposed in WP-7).
SCREENING_COMPONENT = "experiments.hf_screening"
MONEY_FLAG = "--i-understand-this-costs-money"
#: Exact enumeration cap; above it the fallback test samples deterministically.
EXACT_PERMUTATION_LIMIT = 200_000
MONTE_CARLO_PERMUTATIONS = 20_000
INCOMPLETE_NOTE = (
    "a timed out, interrupted, or killed run is never evidence; "
    "incomplete seeds are excluded from mean/sum/sumsq and the p-value"
)


class ScreenConfig(BaseModel):
    """Candidate screening config (JSON file merged with CLI overrides)."""

    screen_id: str = "hf_screen"
    seeds: int = Field(default=8, ge=2)
    steps: int = Field(default=200, ge=1)
    flavor: str = DEFAULT_FLAVOR
    image: str = DEFAULT_IMAGE
    repo_url: str = DEFAULT_REPO
    branch: str = "main"
    context_backend: str = "hf"
    checkpoint_bucket: str = DEFAULT_BUCKET
    extra_train_args: str = ""
    metric: str = "meaningful_program_rate"
    suite: str = "smoke"
    max_minutes: int = Field(default=30, ge=1)
    trackio_project: str = "openui-hf-screening"

    def run_id(self, seed: int) -> str:
        return f"{self.screen_id}_s{int(seed)}"

    @property
    def bucket_prefix(self) -> str:
        """Bucket-relative directory the per-seed result files land in."""
        return f"screening/{self.screen_id}"


# ---------------------------------------------------------------------------
# Job construction (reuses scripts.hf_jobs_train helpers)
# ---------------------------------------------------------------------------


def build_metric_publish_block(config: ScreenConfig, seed: int) -> str:
    """Bash appended to the train entrypoint: extract + publish the metric.

    Writes ``{BUCKET_MOUNT}/screening/<screen_id>/seed_<k>.json`` (durable via
    the bucket volume mount) and mirrors the metric to Trackio following the
    ``slm_training.autoresearch.telemetry.TrackioSink`` pattern (init + log;
    best-effort — local/bucket JSON stays authoritative).
    """
    run_id = config.run_id(seed)
    seed_path = f"{BUCKET_MOUNT}/{config.bucket_prefix}/seed_{int(seed)}.json"
    return f"""
echo "[hf-screen] publish {config.metric} ({config.suite}) seed={int(seed)}"
python - <<'PYEOF'
import json
from pathlib import Path

scoreboard = json.loads(
    Path("outputs/runs/{run_id}/scoreboard.json").read_text(encoding="utf-8")
)
value = scoreboard.get("suites", {{}}).get({config.suite!r}, {{}}).get(
    {config.metric!r}
)
result = {{
    "schema": {SEED_SCHEMA!r},
    "screen_id": {config.screen_id!r},
    "seed": {int(seed)},
    "run_id": {run_id!r},
    "metric": {config.metric!r},
    "suite": {config.suite!r},
    "steps": {int(config.steps)},
    "value": value,
    # Only a job that reached this line publishes at all: a killed or
    # platform-timed-out job never writes this file, and the collector marks
    # the seed incomplete — a killed run is never evidence.
    "status": "complete" if value is not None else "incomplete",
    "version_stamp": scoreboard.get("version_stamp"),
}}
out = Path({seed_path!r})
try:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\\n", encoding="utf-8")
except OSError as exc:  # bucket mount absent — stdout record still lands
    print("[hf-screen] bucket write failed:", exc)
print(json.dumps(result))
try:  # Trackio mirror (telemetry.TrackioSink pattern); never fails the job
    import trackio

    trackio.init(project={config.trackio_project!r}, name={run_id!r})
    trackio.log(
        {{{config.metric!r}: value, "seed": {int(seed)}, "steps": {int(config.steps)}}}
    )
except Exception as exc:
    print("[hf-screen] trackio mirror skipped:", exc)
PYEOF
""".strip()


def build_seed_job(config: ScreenConfig, seed: int) -> dict[str, Any]:
    """One fully-specified HF Jobs submission for a single seed."""
    extra = f"{config.extra_train_args} --seed {int(seed)}".strip()
    entrypoint = (
        build_entrypoint_script(
            repo_url=config.repo_url,
            branch=config.branch,
            run_id=config.run_id(seed),
            steps=config.steps,
            context_backend=config.context_backend,
            checkpoint_bucket=config.checkpoint_bucket,
            sync_checkpoints=True,
            skip_eval=False,
            extra_train_args=extra,
            max_wall_minutes=config.max_minutes,
        )
        + "\n"
        + build_metric_publish_block(config, seed)
    )
    timeout = f"{int(config.max_minutes)}m"
    command = build_jobs_run_command(
        flavor=config.flavor,
        timeout=timeout,
        image=config.image,
        entrypoint=entrypoint,
        checkpoint_bucket=config.checkpoint_bucket,
        mount_bucket=True,
        env={
            "SCREEN_ID": config.screen_id,
            "SCREEN_SEED": str(int(seed)),
            "TRACKIO_PROJECT": config.trackio_project,
        },
    )
    return {
        "seed": int(seed),
        "run_id": config.run_id(seed),
        "flavor": config.flavor,
        "timeout": timeout,
        "command": command,
        "command_pretty": " ".join(shlex.quote(c) for c in command[:20]) + " …",
        "entrypoint": entrypoint,
    }


def build_screen_plan(config: ScreenConfig) -> dict[str, Any]:
    jobs = [build_seed_job(config, seed) for seed in range(config.seeds)]
    return {
        "schema": "hf_jobs_screening_plan/v1",
        "screen_id": config.screen_id,
        "flavor": config.flavor,
        "job_count": len(jobs),
        "seeds": [job["seed"] for job in jobs],
        "per_job_timeout": f"{int(config.max_minutes)}m",
        # Jobs are independent and run concurrently: wall-clock is bounded by
        # one job's cap; the spend budget scales with the fanout.
        "estimated_wall_clock_minutes": int(config.max_minutes),
        "estimated_gpu_minutes_total": int(config.max_minutes) * len(jobs),
        "metric": config.metric,
        "suite": config.suite,
        "steps": config.steps,
        "collect_from": f"{config.checkpoint_bucket}/{config.bucket_prefix}",
        "incomplete_policy": INCOMPLETE_NOTE,
        "jobs": jobs,
    }


def submit_screen(plan: dict[str, Any]) -> dict[str, Any]:
    """Submit every planned job; returns the submission manifest."""
    # Spend transparency: flavor + job count are printed BEFORE any submit.
    print(
        json.dumps(
            {
                "submitting": True,
                "flavor": plan["flavor"],
                "job_count": plan["job_count"],
                "per_job_timeout": plan["per_job_timeout"],
                "estimated_gpu_minutes_total": plan["estimated_gpu_minutes_total"],
            }
        )
    )
    submissions = []
    for job in plan["jobs"]:
        returncode = submit_jobs_command(list(job["command"]))
        submissions.append(
            {
                "seed": job["seed"],
                "run_id": job["run_id"],
                "returncode": returncode,
            }
        )
    return {
        "schema": "hf_jobs_screening_submission/v1",
        "screen_id": plan["screen_id"],
        "flavor": plan["flavor"],
        "job_count": plan["job_count"],
        "submissions": submissions,
    }


# ---------------------------------------------------------------------------
# Statistics — guarded slm_training.autoresearch.power, inline exact fallback
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def inline_permutation_p_value(
    candidate: list[float], baseline: list[float]
) -> dict[str, Any]:
    """Two-sided two-sample permutation test on the difference of means.

    Exact enumeration when C(n1+n2, n1) <= EXACT_PERMUTATION_LIMIT, else a
    deterministic Monte Carlo approximation (seeded; add-one smoothed).
    """
    n1, n2 = len(candidate), len(baseline)
    pooled = list(candidate) + list(baseline)
    observed = abs(_mean(candidate) - _mean(baseline))
    tol = 1e-12
    total_exact = math.comb(n1 + n2, n1)
    if total_exact <= EXACT_PERMUTATION_LIMIT:
        extreme = 0
        pooled_sum = sum(pooled)
        for picks in combinations(range(n1 + n2), n1):
            sum1 = sum(pooled[i] for i in picks)
            diff = abs(sum1 / n1 - (pooled_sum - sum1) / n2)
            if diff >= observed - tol:
                extreme += 1
        return {
            "p_value": extreme / total_exact,
            "method": "exact_permutation",
            "n_permutations": total_exact,
        }
    rng = random.Random(0)
    extreme = 0
    pooled_sum = sum(pooled)
    for _ in range(MONTE_CARLO_PERMUTATIONS):
        rng.shuffle(pooled)
        sum1 = sum(pooled[:n1])
        diff = abs(sum1 / n1 - (pooled_sum - sum1) / n2)
        if diff >= observed - tol:
            extreme += 1
    return {
        "p_value": (extreme + 1) / (MONTE_CARLO_PERMUTATIONS + 1),
        "method": "monte_carlo_permutation",
        "n_permutations": MONTE_CARLO_PERMUTATIONS,
    }


def inline_min_attainable_p(n1: int, n2: int) -> float:
    """Smallest two-sided p an exact two-sample permutation test can return.

    The observed labelling and its mirror both attain the maximal |mean
    difference|, so the floor is 2 / C(n1+n2, n1). At n1=n2=3 this is 0.1
    (> 0.05: undecidable — RC1); at n1=n2=8 it is ~1.6e-4 (decidable).
    """
    return min(1.0, 2.0 / math.comb(n1 + n2, n1))


def permutation_report(
    candidate: list[float], baseline: list[float], *, alpha: float = 0.05
) -> dict[str, Any]:
    """p-value + decidability; prefers slm_training.autoresearch.power."""
    report: dict[str, Any] = {"alpha": alpha, "source": "inline_fallback"}
    power_mod = None
    try:  # concurrent WP delivers this module; fall back until it lands
        from slm_training.autoresearch import power as power_mod  # type: ignore
    except ImportError:
        power_mod = None

    n1, n2 = len(candidate), len(baseline)
    min_p: float | None = None
    decidable: bool | None = None
    p_value_block: dict[str, Any] | None = None
    if power_mod is not None:
        try:
            min_p = float(power_mod.min_attainable_p(n1, n2))
            decidable = bool(power_mod.is_decidable(n1, n2, alpha=alpha))
            report["source"] = "slm_training.autoresearch.power"
        except Exception:  # unknown/older signature — inline math instead
            min_p = None
            decidable = None
        for name in ("permutation_p_value", "two_sample_permutation_p_value"):
            fn = getattr(power_mod, name, None)
            if fn is None:
                continue
            try:
                p_value_block = {
                    "p_value": float(fn(candidate, baseline)),
                    "method": f"power.{name}",
                    "n_permutations": None,
                }
                break
            except Exception:
                p_value_block = None
    if min_p is None:
        min_p = inline_min_attainable_p(n1, n2)
    if decidable is None:
        decidable = min_p <= alpha
    if p_value_block is None:
        p_value_block = inline_permutation_p_value(candidate, baseline)
    report.update(p_value_block)
    report["min_attainable_p"] = min_p
    report["decidable"] = decidable
    return report


# ---------------------------------------------------------------------------
# Collection — evidence JSON matching the loop's conventions
# ---------------------------------------------------------------------------


def _screening_version_stamp() -> dict[str, Any]:
    """version_stamp via the repo helper (quality-matrix convention).

    Degrades to a component-less stamp while the proposed
    ``experiments.hf_screening`` registry entry is pending (WP-7 does not edit
    ``versions.json``); provenance (commit, dirty, stamped_at) is still bound.
    """
    try:
        return build_version_stamp(SCREENING_COMPONENT)
    except KeyError:
        stamp = build_version_stamp()
        stamp["components"] = {SCREENING_COMPONENT: "UNREGISTERED"}
        return stamp


def load_baseline(path: Path) -> dict[str, Any]:
    """Baseline seed values: a prior screening result, {"values": []}, or []."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict) and "values" in payload:
        values = payload["values"]
    else:
        raise SystemExit(
            f"baseline {path} must be a JSON list of values or carry a "
            '"values" array (e.g. a prior screening result)'
        )
    return {"source": str(path), "values": [float(v) for v in values]}


def collect_results(
    config: ScreenConfig,
    *,
    collect_dir: Path,
    baseline: dict[str, Any] | None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fold per-seed result files into one screening evidence payload.

    Shape mapping (documented in docs/design/hf-jobs-train.md):

    - ``n_obs`` / ``n_complete`` / ``sum`` / ``sumsq`` mirror the per-arm
      accumulators of ``autotrain_climb/evidence_ledger.v1.json`` (its
      ``by_eval_key`` entries), so the loop can fold this file in directly;
    - the top-level ``version_stamp`` envelope follows
      ``docs/design/quality-matrix-results.json``.
    """
    per_seed: list[dict[str, Any]] = []
    for seed in range(config.seeds):
        seed_file = collect_dir / f"seed_{seed}.json"
        if not seed_file.exists():
            per_seed.append(
                {
                    "seed": seed,
                    "run_id": config.run_id(seed),
                    "status": "incomplete",
                    "value": None,
                    "reason": (
                        "no result file — job killed, interrupted, or hit the "
                        f"{config.max_minutes}m cap ({INCOMPLETE_NOTE})"
                    ),
                }
            )
            continue
        payload = json.loads(seed_file.read_text(encoding="utf-8"))
        status = payload.get("status")
        value = payload.get("value")
        if status != "complete" or value is None:
            per_seed.append(
                {
                    "seed": seed,
                    "run_id": payload.get("run_id", config.run_id(seed)),
                    "status": "incomplete",
                    "value": None,
                    "reason": f"job reported status={status!r} ({INCOMPLETE_NOTE})",
                }
            )
            continue
        per_seed.append(
            {
                "seed": seed,
                "run_id": payload.get("run_id", config.run_id(seed)),
                "status": "complete",
                "value": float(value),
            }
        )

    values = [row["value"] for row in per_seed if row["status"] == "complete"]
    n_complete = len(values)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "screen_id": config.screen_id,
        "arm": config.screen_id,
        "metric": config.metric,
        "suite": config.suite,
        "steps": config.steps,
        "flavor": config.flavor,
        "max_minutes": config.max_minutes,
        "n_seeds": config.seeds,
        "per_seed": per_seed,
        # Evidence-ledger accumulator convention (arms.*.by_eval_key):
        "n_obs": config.seeds,
        "n_complete": n_complete,
        "n_incomplete_excluded": config.seeds - n_complete,
        "values": values,
        "sum": sum(values),
        "sumsq": sum(v * v for v in values),
        "mean": _mean(values) if values else None,
        "incomplete_policy": INCOMPLETE_NOTE,
        "trackio_project": config.trackio_project,
        "version_stamp": _screening_version_stamp(),
    }
    if baseline is not None and values and baseline["values"]:
        result["baseline"] = {
            "source": baseline["source"],
            "values": baseline["values"],
            "n": len(baseline["values"]),
            "mean": _mean(baseline["values"]),
        }
        result["permutation"] = permutation_report(
            values, baseline["values"], alpha=alpha
        )
    else:
        result["baseline"] = baseline
        result["permutation"] = {
            "p_value": None,
            "reason": (
                "needs a baseline (--baseline) and at least one complete "
                "seed on each side"
            ),
        }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON file of ScreenConfig fields; CLI flags override it.",
    )
    # Overridable fields default to None so a config file's values survive
    # unless the flag is explicitly passed; ScreenConfig supplies defaults.
    parser.add_argument("--screen-id", default=None, help="Screening run id.")
    parser.add_argument("--seeds", type=int, default=None, help="Fanout width (default 8).")
    parser.add_argument("--steps", type=int, default=None, help="Train steps per seed (default 200).")
    parser.add_argument("--flavor", default=None, help="Jobs hardware flavor.")
    parser.add_argument("--image", default=None)
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--branch", default=None)
    parser.add_argument("--context-backend", choices=("scratch", "hf"), default=None)
    parser.add_argument("--checkpoint-bucket", default=None)
    parser.add_argument("--extra-train-args", default=None)
    parser.add_argument("--metric", default=None, help="Endpoint metric (default meaningful_program_rate).")
    parser.add_argument("--suite", default=None, help="Scoreboard suite (default smoke).")
    parser.add_argument(
        "--max-minutes",
        type=int,
        default=None,
        help="Per-job wall-clock cap in the job spec (default 30). Jobs that "
        "hit it are incomplete and excluded — killed runs are never evidence.",
    )
    parser.add_argument("--trackio-project", default=None)
    # Spend safety — dry-run is the DEFAULT; both extra flags are required to
    # spend, headlessly, with no interactive prompt.
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print exact job specs without submitting (DEFAULT). Submitting "
        f"requires --no-dry-run AND {MONEY_FLAG} AND HF_TOKEN.",
    )
    parser.add_argument(
        MONEY_FLAG,
        dest="i_understand_this_costs_money",
        action="store_true",
        help="Literal acknowledgement that submission spends paid HF Jobs "
        "credits. Required (with --no-dry-run and HF_TOKEN) to submit.",
    )
    parser.add_argument(
        "--collect",
        type=Path,
        default=None,
        help="Collect mode (no spend): directory holding per-seed "
        "seed_<k>.json files (the bucket's screening/<screen-id>/ dir, "
        "local mirror or mount).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline per-seed values for the permutation test: a prior "
        'screening result JSON, {"values": [...]}, or a bare JSON list.',
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Screening-result JSON path (default outputs/screening/"
        "<screen-id>/screening_result.json).",
    )
    return parser


_OVERRIDE_FIELDS = (
    "screen_id",
    "seeds",
    "steps",
    "flavor",
    "image",
    "repo_url",
    "branch",
    "context_backend",
    "checkpoint_bucket",
    "extra_train_args",
    "metric",
    "suite",
    "max_minutes",
    "trackio_project",
)


def resolve_config(args: argparse.Namespace) -> ScreenConfig:
    payload: dict[str, Any] = {}
    if args.config is not None:
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise SystemExit(f"--config {args.config} must be a JSON object")
        payload.update(loaded)
    for field in _OVERRIDE_FIELDS:
        value = getattr(args, field)
        if value is not None:
            payload[field] = value
    return ScreenConfig(**payload)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = resolve_config(args)

    if args.collect is not None:
        baseline = load_baseline(args.baseline) if args.baseline else None
        result = collect_results(
            config,
            collect_dir=args.collect,
            baseline=baseline,
            alpha=args.alpha,
        )
        out = args.out or Path(
            f"outputs/screening/{config.screen_id}/screening_result.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["output"] = str(out)
        print(json.dumps(result, indent=2))
        return 0

    plan = build_screen_plan(config)
    if args.dry_run:
        plan["dry_run"] = True
        plan["submit_requires"] = [
            "--no-dry-run",
            MONEY_FLAG,
            "HF_TOKEN exported",
        ]
        print(json.dumps(plan, indent=2))
        return 0

    # Headless spend guards — hard refusals, never an interactive prompt.
    if not args.i_understand_this_costs_money:
        print(
            json.dumps(
                {
                    "error": "refusing to submit paid HF Jobs",
                    "missing": MONEY_FLAG,
                    "flavor": plan["flavor"],
                    "job_count": plan["job_count"],
                }
            )
        )
        return 2
    if not hf_token_present():
        print(
            json.dumps(
                {
                    "error": "refusing to submit paid HF Jobs",
                    "missing": "HF_TOKEN (export a write token)",
                    "flavor": plan["flavor"],
                    "job_count": plan["job_count"],
                }
            )
        )
        return 2

    manifest = submit_screen(plan)
    out = args.out or Path(
        f"outputs/screening/{config.screen_id}/submission.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    failed = [s for s in manifest["submissions"] if s["returncode"] != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
