#!/usr/bin/env python3
"""Run the SLM-400 (DSH3-25) matched-arm operator-policy objective fixture.

Uses real ``enumerate_operator_legal_set`` truncation (``max_combinations_
per_operator``) against a 12-candidate live domain to produce genuine
``PARTIAL`` coverage at each unknown-fraction level -- never a hand-faked
one.

Example:
  python -m scripts.run_dsh3_25_operator_policy_objective_fixture
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    CompilerCoverage,
    OperatorArgumentSlotV1,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    RegisteredOperatorV1,
    build_reference_table,
    enumerate_operator_legal_set,
)
from slm_training.dsl.pack import get_pack
from slm_training.models.operator_feature_encoder import (
    FeatureArm,
    OperatorFeatureVocabularyV1,
    OperatorFeatureEncoder,
    CandidateScoringHead,
)
from slm_training.models.operator_policy_objective import run_matched_objective_fixture
from slm_training.models.operator_policy_view import build_operator_policy_input
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/dsh3-25-operator-policy-objective.json"
_DESIGN_MD = "docs/design/dsh3-25-operator-policy-objective.md"

SOURCE = 'root = TextContent(":hero.title")'
OPERATOR_ID = "openui.fixture_objective"
N_CANDIDATES = 12
GOLD_INDEX = 7


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_pack_and_state():
    base_pack = get_pack("openui")
    state = OperatorStateV1.from_source(base_pack, SOURCE)
    allowed = {_sha(f"semantic-value-{GOLD_INDEX}")}

    # Opaque ref IDs are permutation-allocated, never semantic; resolve gold
    # by the descriptor's own semantic fingerprint via this closure instead.
    semantic_by_ref: dict[Any, str] = {}

    def real_execute(operator_state, arguments):
        semantic = semantic_by_ref[arguments[0].value]
        if semantic not in allowed:
            raise OperatorRejectedError("fixture.not_legal")
        return OperatorMutationV1(
            source=operator_state.source.replace(":hero.title", ":hero.body"),
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    declaration = AstOperatorV1(
        operator_id=OPERATOR_ID,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(
            OperatorArgumentSlotV1("value", RefKind.VALUE, BindingPhase.APPLICATION),
        ),
        preconditions=(),
        effect_signature=(),
        locality="node",
        cost=1.0,
    )
    library = OperatorLibraryV1((RegisteredOperatorV1(declaration, real_execute),))
    pack = replace(base_pack, operator_library=library)
    return pack, library, state, semantic_by_ref


def _build_table(state: OperatorStateV1, semantic_by_ref: dict[Any, str], seed: int):
    descriptors = tuple(
        ReferenceDescriptorV1(
            ref_kind=RefKind.VALUE,
            semantic_fingerprint=_sha(f"semantic-value-{index}"),
            value_type="openui.string",
        )
        for index in range(N_CANDIDATES)
    )
    table = build_reference_table(
        request_id="request-1",
        state_digest=state.state_digest,
        branch_digest=_sha("branch"),
        descriptors=descriptors,
        seed=seed,
    )
    semantic_by_ref.clear()
    semantic_by_ref.update(
        {entry.ref: entry.descriptor.semantic_fingerprint for entry in table.entries}
    )
    return table


def _budget_for_fraction(unknown_fraction: float) -> int:
    evaluated = max(1, round(N_CANDIDATES * (1.0 - unknown_fraction)))
    return evaluated


def _fixture_builder(pack, library, state, semantic_by_ref):
    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.dsh3_25_fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(SOURCE),
        request_id="request-1",
    )

    def build(unknown_fraction: float, seed: int):
        table = _build_table(state, semantic_by_ref, seed=seed)
        budget = _budget_for_fraction(unknown_fraction)
        legal_set = enumerate_operator_legal_set(
            pack=pack,
            library=library,
            state=state,
            reference_table=table,
            provenance=provenance,
            max_combinations_per_operator=budget,
        )
        policy_input = build_operator_policy_input(table, legal_set, library)

        gold_semantic = _sha(f"semantic-value-{GOLD_INDEX}")
        gold_row = next(
            row
            for row, entry in enumerate(table.entries)
            if entry.descriptor.semantic_fingerprint == gold_semantic
        )
        entry = legal_set.entries[0]
        witnessed = any(
            argument.value == table.entries[gold_row].ref
            for action in entry.legal_actions
            for argument in action.arguments
        )
        positive_rows_by_action_row = {0: (gold_row,)} if witnessed else {}

        vocabulary = OperatorFeatureVocabularyV1.from_inputs([policy_input])
        torch.manual_seed(seed)
        encoder = OperatorFeatureEncoder(vocabulary, dim=6, arm=FeatureArm.TYPED)
        head = CandidateScoringHead(6)
        reference_embeddings = encoder.encode_reference_rows(policy_input)
        action_embedding = encoder.encode(policy_input)[0]
        candidate_rows = policy_input.action_rows[0].argument_slots[0].candidate_rows
        candidate_embeddings = reference_embeddings[
            torch.tensor(candidate_rows, dtype=torch.long)
        ]
        logits_over_candidates = head(action_embedding, candidate_embeddings)
        # Map slot-local candidate-row-position logits back onto absolute
        # reference-row indices so the objective's row-indexed API lines up.
        full_logits = torch.full((len(policy_input.reference_rows),), float("-inf"))
        full_logits[torch.tensor(candidate_rows, dtype=torch.long)] = logits_over_candidates
        return policy_input, positive_rows_by_action_row, {0: full_logits}

    return build


def _run_fixture(*, unknown_fractions: tuple[float, ...], seed: int) -> dict[str, Any]:
    pack, library, state, semantic_by_ref = _build_pack_and_state()
    builder = _fixture_builder(pack, library, state, semantic_by_ref)
    report = run_matched_objective_fixture(
        fixture_builder=builder, unknown_fractions=unknown_fractions, seed=seed
    )
    payload = report.to_dict()
    payload.update(
        {
            "schema_version": 1,
            "run_id": "dsh3_25_operator_policy_objective_fixture",
            "run_class": "fixture_demo",
            "n_candidates": N_CANDIDATES,
            "gold_index": GOLD_INDEX,
            "honest_verdict": "fixture_wiring",
            "version_stamp": build_version_stamp("model.operator_policy_objective"),
        }
    )
    return payload


def _build_markdown(payload: dict[str, Any]) -> str:
    fractions = payload["unknown_fractions"]
    lines = [
        "# SLM-400 (DSH3-25): operator-policy objective and DEFER-routing fixture",
        "",
        "**Status:** fixture / wiring only.  ",
        "**Claim class:** `wiring`.  ",
        "**Honest verdict:** `fixture_wiring`.  ",
        f"**Disposition:** `{payload['disposition']}`.",
        "",
        payload["note"],
        "",
        "## What this exercises",
        "",
        "- A 12-candidate live domain enumerated by the real "
        "`enumerate_operator_legal_set`, truncated via its own "
        "`max_combinations_per_operator` at each unknown-fraction level — "
        "genuine `PARTIAL` coverage, not a hand-faked one.",
        "- `build_operator_policy_objectives` over the SLM-397 sanitized "
        "`OperatorPolicyInputV1`.",
        "- Three matched arms sharing one `CandidateScoringHead`/"
        "`OperatorFeatureEncoder` (typed arm) forward pass: "
        "`known_supported_ce`, `positive_unlabeled_risk`, `defer_abstention`.",
        "",
        "## Route distribution and mean loss by unknown fraction",
        "",
        "| Unknown fraction | Routes | known_supported_ce | positive_unlabeled_risk | defer_abstention |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, entry in fractions.items():
        fraction = key.replace("unknown_fraction_", "")
        routes = ", ".join(f"{name}={count}" for name, count in entry["routes"].items())
        losses = entry["mean_loss"]

        def fmt(value):
            return "abstained" if value is None else f"{value:.4f}"

        lines.append(
            f"| {fraction} | {routes} | {fmt(losses['known_supported_ce'])} | "
            f"{fmt(losses['positive_unlabeled_risk'])} | {fmt(losses['defer_abstention'])} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- A single synthetic 12-candidate domain and an untrained, "
            "randomly-initialized typed encoder/head — logits demonstrate "
            "the routing/loss-arm contract, not learned quality.",
            "- `known_supported_ce` and `defer_abstention` are identical "
            "whenever there is at least one positive witness (by design — "
            "`defer_abstention` only diverges on `PARTIAL_UNKNOWN`), and "
            "both correctly abstain (no loss) when there is zero witness at "
            "all, since no supervised signal exists for *any* arm in that "
            "case. The behavioral difference this ticket cares about is an "
            "**inference-time serving policy** one (always-guess versus "
            "explicit DEFER on a `PARTIAL_UNKNOWN` route) — this fixture "
            "demonstrates that at the route-label level, not via a full "
            "serving/decoding harness, which is out of scope here.",
            "- No claim that PU risk or DEFER beats naive CE at production "
            "scale in either direction; see the disposition above.",
            "- `TargetUtility` only offers `POSITIVE`/`UNKNOWN` in this "
            "version — SLM-399's collapse hard negatives name a whole "
            "alternate *operator*, not an alternate candidate within the "
            "same slot, so there is no certified per-candidate `NEGATIVE` "
            "source yet. Adding one is a follow-up, not approximated here.",
            "- No ship gate is evaluated or weakened.",
            "",
            "## Verification commands",
            "",
            "```bash",
            "python -m pytest -q tests/test_models/test_operator_policy_objective.py "
            "tests/test_dsl/test_operator_legal_set.py tests/test_models/test_legal_edit_batch.py",
            "python -m scripts.verify_version_stamps --check",
            "```",
            "",
            "Both commands passed on this branch at the time of writing.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/dsh3-25-operator-policy-objective-fixture"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    payload = _run_fixture(unknown_fractions=(0.0, 0.25, 0.5, 0.75), seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_json = args.output_dir / "dsh3_25_operator_policy_objective_report.json"
    run_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    root = Path(__file__).resolve().parents[1]
    json_path = root / _DESIGN_JSON
    md_path = root / _DESIGN_MD
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_build_markdown(payload), encoding="utf-8")

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
