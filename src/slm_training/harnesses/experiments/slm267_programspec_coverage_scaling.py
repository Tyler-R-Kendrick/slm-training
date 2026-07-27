"""Bounded ProgramSpec-first coverage-scaling wiring evidence for SLM-267 (VSD2-02).

VSD2-02 asks for 10k/100k/1M-record corpora comparing uniform-valid against
coverage-targeted sampling. The repository's existing coverage-guided
generator (SLM-5, ``slm_training.data.progspec.generate``) samples from a
fixed, exhaustible candidate grid rather than an open-ended program space, so
those rungs are out of reach without a materially different generator. This
module measures what the *existing* generator actually supports: a genuine
uniform-random control arm (``ProgramGenerator.generate_uniform``) against its
existing greedy coverage-maximizing arm (``ProgramGenerator.generate_one``),
at fixture scale, plus the deterministic ``(global_seed, shard_id,
worker_id)`` sharding primitive VSD2-02 requires. This is wiring evidence
only: it does not validate VSD-H7a/b/c, which require corpora and trained
models several orders of magnitude larger than anything generated here.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from slm_training.data.progspec.generate import (
    PROGRAM_FAMILY,
    GeneratorConfig,
    ProgramGenerator,
)
from slm_training.data.progspec.schema import ProgramSpec, emit_record
from slm_training.harnesses.train_data.diversity import fingerprint_record

SCHEMA = "programspec_coverage_scaling_report/v1"
EXPERIMENT_ID = "slm267-programspec-coverage-scaling"
POLICIES: tuple[str, ...] = ("uniform", "coverage_targeted")

DEFAULT_CONFIG = GeneratorConfig(
    components=("TextContent", "Button", "Separator"),
    max_depth=3,
    max_width=3,
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def shard_seed(global_seed: int, shard_id: int, worker_id: int) -> int:
    """Deterministically derive a per-shard/per-worker seed.

    The same ``(global_seed, shard_id, worker_id)`` triple always derives the
    same seed; varying any one input changes it. This is the sharding
    primitive VSD2-02 requires ("deterministic sharding from (global_seed,
    shard_id, worker_id)") — resumable and reproducible without a stateful
    RNG carried across shards.
    """
    if shard_id < 0 or worker_id < 0:
        raise ValueError("shard_id and worker_id must be non-negative")
    payload = json.dumps([global_seed, shard_id, worker_id], separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _canonical_root_id(spec: ProgramSpec) -> str:
    record = emit_record(
        spec,
        prompt="Generate this typed OpenUI program.",
        task="generation",
        source=PROGRAM_FAMILY,
    )
    return fingerprint_record(record).canonical_root_id


@dataclass(frozen=True)
class ArmResultV1:
    """One policy arm's yield/coverage/dedup counters over a bounded budget."""

    policy: str
    seed: int
    requested_count: int
    proposed: int
    accepted: int
    duplicate_roots: int
    unique_canonical_roots: int
    exhausted_at: int | None
    cells_total: int
    cells_covered: int
    programs_to_full_coverage: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "requested_count": self.requested_count,
            "proposed": self.proposed,
            "accepted": self.accepted,
            "duplicate_roots": self.duplicate_roots,
            "unique_canonical_roots": self.unique_canonical_roots,
            "exhausted_at": self.exhausted_at,
            "cells_total": self.cells_total,
            "cells_covered": self.cells_covered,
            "programs_to_full_coverage": self.programs_to_full_coverage,
        }


def run_policy_arm(
    policy: str, *, config: GeneratorConfig, seed: int, target_count: int
) -> ArmResultV1:
    """Run one sampling policy up to ``target_count`` programs or exhaustion.

    ``"uniform"`` draws uniformly at random from the unused candidate grid
    (no coverage bias). ``"coverage_targeted"`` uses the generator's existing
    greedy coverage-maximizing selection, which always prioritizes cells not
    yet covered — the closest honest analogue of VSD2-02's coverage-targeted
    principal policy available from the current generator without accepting
    an unverified external gap-manifest mapping.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    if target_count < 0:
        raise ValueError("target_count must be non-negative")
    generator = ProgramGenerator(config, seed=seed)
    rng = random.Random(seed)
    proposed = 0
    accepted = 0
    exhausted_at: int | None = None
    programs_to_full_coverage: int | None = None
    seen_roots: set[str] = set()
    duplicate_roots = 0
    for _ in range(target_count):
        proposed += 1
        try:
            spec = (
                generator.generate_uniform(rng)
                if policy == "uniform"
                else generator.generate_one()
            )
        except ValueError as exc:
            if str(exc) == "candidate grid exhausted":
                exhausted_at = accepted
                break
            raise
        accepted += 1
        root_id = _canonical_root_id(spec)
        if root_id in seen_roots:
            duplicate_roots += 1
        else:
            seen_roots.add(root_id)
        if programs_to_full_coverage is None and generator.tracker.complete:
            programs_to_full_coverage = accepted
    report = generator.tracker.report()
    return ArmResultV1(
        policy=policy,
        seed=seed,
        requested_count=target_count,
        proposed=proposed,
        accepted=accepted,
        duplicate_roots=duplicate_roots,
        unique_canonical_roots=len(seen_roots),
        exhausted_at=exhausted_at,
        cells_total=report["targets"],
        cells_covered=report["covered"],
        programs_to_full_coverage=programs_to_full_coverage,
    )


def _first_full_coverage(shard_results: list[dict[str, Any]]) -> int | None:
    values = [
        result["programs_to_full_coverage"]
        for result in shard_results
        if result["programs_to_full_coverage"] is not None
    ]
    return min(values) if values else None


def run_coverage_scaling_campaign(
    *,
    target_count: int = 60,
    seed: int = 0,
    shards: int = 1,
    config: GeneratorConfig | None = None,
) -> dict[str, Any]:
    """Compare uniform vs. coverage-targeted sampling at fixture scale.

    Returns a versioned, self-hashed payload; never a ship or scale claim.
    """
    if shards < 1:
        raise ValueError("shards must be positive")
    cfg = config or DEFAULT_CONFIG
    arms: dict[str, list[dict[str, Any]]] = {}
    for policy in POLICIES:
        arms[policy] = [
            run_policy_arm(
                policy,
                config=cfg,
                seed=shard_seed(seed, shard_id, worker_id=0),
                target_count=target_count,
            ).to_dict()
            for shard_id in range(shards)
        ]
    uniform_first = _first_full_coverage(arms["uniform"])
    targeted_first = _first_full_coverage(arms["coverage_targeted"])
    targeted_no_slower = (
        uniform_first is not None
        and targeted_first is not None
        and targeted_first <= uniform_first
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "claim_class": "wiring",
        "recipe": {
            "target_count": target_count,
            "seed": seed,
            "shards": shards,
            "components": list(cfg.components or ()),
            "max_depth": cfg.max_depth,
            "max_width": cfg.max_width,
        },
        "arms": arms,
        "comparison": {
            "uniform_programs_to_full_coverage": uniform_first,
            "coverage_targeted_programs_to_full_coverage": targeted_first,
            "coverage_targeted_reaches_full_coverage_no_slower": targeted_no_slower,
        },
        "status": "inconclusive",
        "limitations": [
            "Fixture-scale only: target_count and the candidate grid are far "
            "below the 10k/100k/1M rungs VSD-H7a-c require; no claim is made "
            "about those hypotheses at this scale.",
            "The 'coverage_targeted' arm reuses the generator's existing "
            "greedy CoverageTracker.score() bias toward uncovered cells; it "
            "does not yet consume an external CoverageGapManifestV1 "
            "(SLM-265) gap manifest — that mapping is explicitly future work.",
            "Canonical-root dedup here is in-memory only; a disk-backed "
            "exact index for 100k/1M-scale generation is not implemented.",
            "No training/model comparison is run and no corpus is published "
            "to the DataStore; this measures the generator mechanism only.",
        ],
    }
    payload["manifest_hash"] = _sha(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )
    return payload


def render_markdown(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    lines = [
        "# SLM-267 (VSD2-02): ProgramSpec coverage-scaling wiring evidence",
        "",
        "**Claim class:** wiring / fixture only",
        "",
        f"**Status:** `{report['status']}`",
        "",
        "## Recipe",
        "",
        f"```json\n{json.dumps(report['recipe'], indent=2)}\n```",
        "",
        "## Arms",
        "",
        "| policy | shard | seed | proposed | accepted | duplicate_roots | "
        "unique_roots | exhausted_at | cells_covered/total | "
        "programs_to_full_coverage |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for policy, shard_results in report["arms"].items():
        for shard_id, result in enumerate(shard_results):
            lines.append(
                f"| {policy} | {shard_id} | {result['seed']} | "
                f"{result['proposed']} | {result['accepted']} | "
                f"{result['duplicate_roots']} | {result['unique_canonical_roots']} | "
                f"{result['exhausted_at']} | "
                f"{result['cells_covered']}/{result['cells_total']} | "
                f"{result['programs_to_full_coverage']} |"
            )
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- uniform programs to full coverage: "
            f"{comparison['uniform_programs_to_full_coverage']}",
            f"- coverage-targeted programs to full coverage: "
            f"{comparison['coverage_targeted_programs_to_full_coverage']}",
            f"- coverage-targeted reaches full coverage no slower than uniform: "
            f"{comparison['coverage_targeted_reaches_full_coverage_no_slower']}",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_CONFIG",
    "EXPERIMENT_ID",
    "POLICIES",
    "SCHEMA",
    "ArmResultV1",
    "render_markdown",
    "run_coverage_scaling_campaign",
    "run_policy_arm",
    "shard_seed",
]
