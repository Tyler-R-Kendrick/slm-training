"""LDI1-01 fixture: exact causal decision-state trace and replay evidence.

This script exercises the torch-free core of ``slm_training.models.causal_trace``
with synthetic logits and a hand-defined grammar legal-set seam.  It produces a
fixture-grade trace showing:

* exact prefix token IDs as state authority;
* raw argmax, legal set, and constrained selection per step;
* constraint-shadow detection (raw winner illegal, constrained selection legal);
* forced-action replay from the stored exact prefix;
* TraceStore persistence, manifest, and fail-closed identity loading.

No model, tokenizer, or torch is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slm_training.harnesses.distill.trace_store import TraceStore
from slm_training.models.causal_trace import (
    CausalTraceIdentity,
    CausalTraceWriter,
    GeneratedOutcome,
    RawStepObservation,
    TracePolicy,
    TraceSelection,
    capture_raw_steps,
    emit_causal_decision,
    load_causal_decision_states,
)
from slm_training.versioning import build_version_stamp


OUTPUT_ROOT = Path("outputs/runs/ldi1-01-causal-trace/iter-ldi1-01-20260719")
DOCS_JSON = Path("docs/design/iter-ldi1-01-causal-trace-20260719.json")

# Tiny synthetic vocabulary: 0=EOS, 1=root, 2=child_A, 3=child_B, 4=invalid_shadow, 5=filler
VOCAB_SIZE = 6
PROMPT = (1,)

# Grammar legal set by suffix length after the prompt.
LEGAL_BY_SUFFIX: dict[int, tuple[int, ...]] = {
    0: (2, 3, 4),  # first decision: A, B, or an illegal-looking token (4 is illegal but we keep it legal here for shadow demo)
    1: (5,),       # forced continuation
    2: (0, 5),     # can end or continue
}

# Logits that cause token 4 (illegal in second decision) to be raw argmax but legal set excludes it.
_LOGITS_BY_PREFIX: dict[tuple[int, ...], list[float]] = {
    PROMPT: [0.0, 0.1, 2.0, 1.5, 5.0, 0.5],  # raw argmax = 4
    (1, 2): [10.0, 0.0, 0.0, 0.0, 0.0, 2.0],  # EOS forced/legal
    (1, 3): [10.0, 0.0, 0.0, 0.0, 0.0, 2.0],
    (1, 2, 0): [0.0] * VOCAB_SIZE,
    (1, 3, 0): [0.0] * VOCAB_SIZE,
}


def _forward_logits(prefix: tuple[int, ...]) -> list[float]:
    """Synthetic logits that reproduce exactly for the same prefix."""
    return list(_LOGITS_BY_PREFIX.get(prefix, [0.0] * VOCAB_SIZE))


def _allowed_ids(prefix: tuple[int, ...]) -> tuple[int, ...]:
    """Hand grammar: legal set depends only on suffix length after prompt."""
    suffix_len = len(prefix) - len(PROMPT)
    # Token 4 becomes illegal after the first decision to demonstrate a constraint shadow.
    if suffix_len == 0:
        return (2, 3)
    return LEGAL_BY_SUFFIX.get(suffix_len, (0,))


def _role_of(prefix: tuple[int, ...]) -> str | None:
    suffix = prefix[len(PROMPT) :]
    if len(suffix) == 0:
        return "first_decision"
    if suffix[-1] == 2:
        return "branch_A"
    if suffix[-1] == 3:
        return "branch_B"
    return "continuation"


def _replay_forced_action(
    obs: RawStepObservation,
    forced_action_id: int,
) -> GeneratedOutcome:
    """Simulate a forced-action replay from the exact stored prefix.

    In the real plug-in this calls ``model(context_ids).logits`` with the forced
    token appended; here we continue with the same synthetic seam to prove the
    replay contract.
    """
    base_prefix = (*obs.prefix_token_ids, int(forced_action_id))
    result = capture_raw_steps(
        forward_logits=_forward_logits,
        allowed_ids=_allowed_ids,
        eos_id=0,
        max_new_tokens=4,
        initial_prefix=base_prefix,
    )
    generated = (*base_prefix[len(PROMPT) :], *result.generated_token_ids)
    raw_text = " ".join(str(token) for token in generated)
    return GeneratedOutcome(
        action_id=int(forced_action_id),
        continuation_seed=7,
        finish_reason=result.stop_reason,
        raw_program=raw_text,
        canonical_program=raw_text,
    )


def _run_fixture() -> dict[str, Any]:
    identity = CausalTraceIdentity(
        group_id="ldi1-01-fixture",
        context_text="root=Stack([",
        policy_checkpoint_sha="policy_" + "a" * 56,
        tokenizer_sha="tokenizer_" + "b" * 52,
        decode_config_hash="decode_" + "c" * 53,
        base_model_revision="rev-fixture",
        adapter_identity="adapter_none",
    )

    policy = TracePolicy(selection=TraceSelection.EVERY, top_k=3)
    result = capture_raw_steps(
        forward_logits=_forward_logits,
        allowed_ids=_allowed_ids,
        eos_id=0,
        max_new_tokens=5,
        initial_prefix=PROMPT,
        policy=policy,
        role_of=_role_of,
    )

    # Verify exact replay: re-run forward_logits on each stored prefix, compare.
    replay_errors: list[str] = []
    for obs in result.observations:
        replay_logits = _forward_logits(obs.prefix_token_ids)
        replay_argmax = max(range(len(replay_logits)), key=lambda i: (replay_logits[i], -i))
        if replay_argmax != obs.raw_argmax_id:
            replay_errors.append(
                f"prefix={obs.prefix_token_ids} replay_argmax={replay_argmax} "
                f"observed_argmax={obs.raw_argmax_id}"
            )

    # Build DecisionEventV2 rows and demonstrate forced-action replay for one state.
    events: list[dict[str, Any]] = []
    forced_replays: list[dict[str, Any]] = []
    shadow_count = 0
    for obs in result.observations:
        state, outcomes, view = emit_causal_decision(obs, identity)
        events.append(
            {
                "state": state.to_dict(),
                "outcomes": [o.to_dict() for o in outcomes],
                "view": view.to_dict() if view is not None else None,
            }
        )
        if obs.constraint_shadow:
            shadow_count += 1
        if not obs.forced and len(state.legal_action_ids) > 1:
            # Pick the first legal alternative that is not the selected token.
            alternative = next(
                (token for token in state.legal_action_ids if token != obs.selected_token_id),
                state.legal_action_ids[0],
            )
            replay = _replay_forced_action(obs, alternative)
            forced_replays.append(
                {
                    "state_id": state.state_id,
                    "forced_action_id": alternative,
                    "replay": replay.to_dict(),
                }
            )

    # Persist to TraceStore and write manifest.
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    store = TraceStore(OUTPUT_ROOT / "traces", run_id="ldi1-01-fixture")
    writer = CausalTraceWriter(store, identity)
    writer.record_all(result)
    writer.write_manifest(OUTPUT_ROOT / "manifest.json")

    # Verify fail-closed loading.
    loaded_states = load_causal_decision_states(
        store,
        expected_checkpoint_sha=identity.policy_checkpoint_sha,
        expected_tokenizer_sha=identity.tokenizer_sha,
    )

    summary = {
        "schema_version": "ldi1-01-fixture-summary/v1",
        "group_id": identity.group_id,
        "observations": [obs.to_dict() for obs in result.observations],
        "generated_token_ids": list(result.generated_token_ids),
        "stop_reason": result.stop_reason,
        "constraint_shadow_count": shadow_count,
        "decision_event_v2_rows": events,
        "forced_action_replays": forced_replays,
        "replay_errors": replay_errors,
        "loaded_state_count": len(loaded_states),
        "manifest": writer.manifest(),
        "version_stamp": build_version_stamp(),
    }

    DOCS_JSON.parent.mkdir(parents=True, exist_ok=True)
    DOCS_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


if __name__ == "__main__":
    summary = _run_fixture()
    print(f"LDI1-01 fixture wrote {OUTPUT_ROOT}/summary.json")
    print(f"LDI1-01 fixture wrote {DOCS_JSON}")
    print(
        "observations:",
        len(summary["observations"]),
        "shadows:",
        summary["constraint_shadow_count"],
        "replays:",
        len(summary["forced_action_replays"]),
    )
