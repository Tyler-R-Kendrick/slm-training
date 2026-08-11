"""Run the governed proof-carrying goal-support fixture campaign.

    python -m scripts.run_goal_support_domain_adequacy --mode fixture

Fixture mode executes two independent governed runs and requires identical
canonical semantic result digests before writing durable JSON and Markdown.
No model, checkpoint, remote service, or ship gate is involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_training.harness_core.versioning import build_version_stamp
from slm_training.autoresearch.experiment_campaign import campaign_manifest_sha256
from slm_training.harnesses.experiments.goal_support_domain_adequacy import (
    ARM_IDS,
    METRIC_IDS,
    GoalSupportDomainAdequacyCampaignV1,
    run_campaign,
)


def _markdown(payload: dict[str, Any]) -> str:
    recipe = payload["recipe"]
    lines = [
        "# Proof-carrying goal support fixture results",
        "",
        "Claim class: **fixture/diagnostic only**. This is bounded wiring evidence, not a model-quality, ship-readiness, global-unrealizability, or production-performance claim.",
        "",
        f"- Source commit: `{recipe['source_commit']}` (`dirty={str(recipe['source_dirty']).lower()}`)",
        f"- Suite: `n={recipe['suite_n']}`, seed `{recipe['seed']}`, local CPU/in-process, no model",
        f"- Solver bounds: `{json.dumps(recipe['solver_bounds'], sort_keys=True)}`",
        f"- Exact action cap: `{recipe['exact_action_cap']}`; goal-query cap per arm: `{recipe['goal_support_query_cap']}`",
        f"- Manifest: `{payload['campaign_governance']['manifest_sha256']}`",
        f"- Canonical result digest: `{payload['determinism']['canonical_result_digest_a']}`",
        f"- Deterministic rerun: `{'PASS' if payload['determinism']['match'] else 'FAIL'}`",
        "",
        "## Arm results",
        "",
        "| arm | coverage | supported | unsupported | unknown | unobserved | selection regret | inadequate under bounds | false hard prune | verifier calls | expanded nodes | wall time (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm_id in ARM_IDS:
        metrics = payload["arm_results"][arm_id]["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    arm_id,
                    str(metrics["exact_action_coverage_rate"]),
                    str(metrics["supported_action_rate"]),
                    str(metrics["unsupported_action_rate"]),
                    str(metrics["unknown_action_rate"]),
                    str(metrics["unobserved_action_rate"]),
                    str(metrics["selection_regret_rate"]),
                    str(metrics["domain_inadequate_under_bounds_rate"]),
                    str(metrics["false_hard_prune_count"]),
                    str(metrics["verifier_calls"]),
                    str(metrics["expanded_nodes"]),
                    str(metrics["wall_time"]),
                ]
            )
            + " |"
        )
    certified = payload["arm_results"]["goal_support_certified_fixture"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The production-exact diagnostic arm distinguishes the satisfying and violating alternatives, emits a replayable bounded obstruction for the fully observed inadequate case, preserves incomplete/unknown candidates, and reports the candidate-cap case as coverage uncertainty. The evaluation-oracle arm requires `G11`; its mandatory skip remains UNKNOWN and it is never supplied to certified closure.",
            "",
            f"Certified fixture pruning removed `{certified['goal_support_pruned']}` replay-valid compiler-hard UNSUPPORTED actions, produced `{certified['certified_empty_domains']}` bounded empty domain, and recorded `{certified['metrics']['false_hard_prune_count']}` false hard prunes. Obstruction-core replay rate was `{certified['metrics']['obstruction_core_replay_rate']}` with mean core size `{certified['metrics']['mean_obstruction_core_size']}`.",
            "",
            "No checkpoint was created, so `docs/MODEL_CARD.md` and its README summary were intentionally unchanged. No AgentV model-eval bundle was required because this command performs no model evaluation.",
            "",
            "Remaining limitation: these are small exact fixtures over terminal completions. Successor work should measure the default-off diagnostic/certified path on real complete OpenUI completion forests without changing the pinned verifier or using fixture evidence for promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan-only", "fixture"), default="plan-only")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/autoresearch/proof-carrying-goal-support-fixture"),
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path("docs/design/proof-carrying-goal-support-fixture-results.json"),
    )
    parser.add_argument(
        "--docs-md-out",
        type=Path,
        default=Path("docs/design/proof-carrying-goal-support-fixture-results.md"),
    )
    parser.add_argument("--exact-action-cap", type=int, default=2)
    parser.add_argument("--goal-support-query-cap", type=int, default=32)
    args = parser.parse_args(argv)

    stamp = build_version_stamp("harness.experiments")
    source_commit = str(stamp["code_commit"])
    if len(source_commit) != 40 or any(
        char not in "0123456789abcdef" for char in source_commit
    ):
        raise RuntimeError("fixture campaign requires an exact Git source commit")
    campaign = GoalSupportDomainAdequacyCampaignV1(
        exact_action_cap=args.exact_action_cap,
        goal_support_query_cap=args.goal_support_query_cap,
        source_commit=source_commit,
        source_dirty=bool(stamp["code_dirty"]),
    )
    manifest = campaign.manifest()
    if args.mode == "plan-only":
        print(
            json.dumps(
                {
                    "status": "plan_only",
                    "claim_class": manifest.claim_class,
                    "campaign_id": manifest.campaign_id,
                    "arms": [arm.arm_id for arm in manifest.arms],
                    "metrics": list(METRIC_IDS),
                    "manifest_sha256": campaign_manifest_sha256(manifest),
                },
                indent=2,
            )
        )
        return 0

    first = run_campaign(campaign, root=args.out_dir / "run_a")
    second = run_campaign(campaign, root=args.out_dir / "run_b")
    digest_a = first["canonical_result_digest"]
    digest_b = second["canonical_result_digest"]
    if digest_a != digest_b:
        raise RuntimeError(
            f"goal-support fixture is nondeterministic: {digest_a} != {digest_b}"
        )
    payload = {
        "kind": "proof_carrying_goal_support_fixture/v1",
        "claim_class": "fixture",
        "status": "fixture_diagnostic_complete",
        "recipe": {
            "command": [
                "python",
                "-m",
                "scripts.run_goal_support_domain_adequacy",
                "--mode",
                "fixture",
            ],
            "source_commit": source_commit,
            "source_dirty": bool(stamp["code_dirty"]),
            "device": "cpu",
            "backend": "in_process_vss",
            "suite_n": first["suite_n"],
            "seed": first["seed"],
            "solver_bounds": first["bounds"]["solver"],
            "exact_action_cap": args.exact_action_cap,
            "goal_support_query_cap": args.goal_support_query_cap,
            "honesty_mode": "fixture_diagnostic_no_model_no_ship_claim",
        },
        "campaign_governance": {
            "campaign_id": first["campaign_id"],
            "manifest_sha256": first["manifest_sha256"],
            "event_chain_verified": first["event_chain_verified"]
            and second["event_chain_verified"],
            "claim_class": first["claim_class"],
            "promotion": False,
        },
        "problem_digests": first["problem_digests"],
        "request_digests": first["request_digests"],
        "constraint_set_digests": first["constraint_set_digests"],
        "profile_digests": first["profile_digests"],
        "pack_identity_digests": first["pack_identity_digests"],
        "bounds": first["bounds"],
        "arm_results": first["arm_results"],
        "metric_definitions": first["metric_definitions"],
        "determinism": {
            "canonical_digest_scope": "all identities, partitions, proofs, classifications, and counters; excludes wall_time and event timestamps",
            "canonical_result_digest_a": digest_a,
            "canonical_result_digest_b": digest_b,
            "match": True,
        },
        "scope_disclaimer": first["scope_disclaimer"],
        "checkpoint_created": False,
        "model_card_updated": False,
        "version_stamp": stamp,
    }
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.docs_md_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_md_out.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "docs_out": str(args.docs_out),
                "docs_md_out": str(args.docs_md_out),
                "canonical_result_digest": digest_a,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
