#!/usr/bin/env python3
"""Run the LDI3-04 immutable remine -> intervene -> regenerate campaign (SLM-132).

This is an autoresearch campaign integration; it persists through the existing
``CampaignStore`` (content-addressed artifacts + hash-chained events) and adds no new
scheduler. The default backend is the torch-free ``FixtureBackend``, so ``--smoke`` is a
bounded one-iteration **wiring-only** run that publishes every expected artifact and is
resumable at every stage. A real generation/training backend (frontier mode) is injected
by the follow-on quality-bearing run, not here.

    # resolve config, stage DAG, arms, and identities without running anything
    python -m scripts.run_remine_campaign --describe

    # bounded one-iteration CPU wiring-only smoke under a campaign root
    python -m scripts.run_remine_campaign --smoke --root outputs/autoresearch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.preference.remine_campaign import (
    RemineCampaignConfig,
    describe_campaign,
    run_campaign,
)

_SMOKE_CONFIG = {
    "campaign_id": "ldi-remine-smoke",
    "created_at": "2026-07-18T00:00:00Z",
    "base_checkpoint_sha": "fixture-parent",
    "tokenizer_sha": "fixture-tokenizer",
    "prompt_group_ids": ["group_a", "group_b"],
    "suite_mix": ["grammar", "schema", "dataflow"],
    "decode_config_hash": "fixture-decode-v1",
    "seeds": [0, 1],
    "verifier_bundle_hash": "fixture-verifier-v1",
    "adapter_spec": {"method": "twotower_adapter", "rank": 4},
    "max_iterations": 2,
    "min_new_evidence": 1,
    "notes": "wiring-only fixture smoke",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="Resolve config/stages without running.")
    parser.add_argument("--smoke", action="store_true", help="Run the bounded fixture wiring-only smoke.")
    parser.add_argument("--config", type=Path, default=None, help="JSON config (fail-closed on unknown fields).")
    parser.add_argument("--root", type=Path, default=Path("outputs/autoresearch"))
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    data = json.loads(args.config.read_text(encoding="utf-8")) if args.config else dict(_SMOKE_CONFIG)
    config = RemineCampaignConfig.from_mapping(data)

    if args.describe or not args.smoke:
        print(json.dumps(describe_campaign(config), indent=args.indent, sort_keys=True))
        return 0

    result = run_campaign(config, root=args.root)
    print(json.dumps(result.to_dict(), indent=args.indent, sort_keys=True))
    print(
        f"campaign={result.campaign_id} status={result.status} "
        f"iterations={len(result.iterations)} stages_run={result.stages_run} "
        f"stages_reused={result.stages_reused}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
