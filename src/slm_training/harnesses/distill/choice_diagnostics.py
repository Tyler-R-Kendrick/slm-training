"""Diagnostics over supplied compiler domains, never a legality authority."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal


def record_forced_canvas(model, ids, unknown_mask, exact_commit, step):
    """Capture a proved single hole without any extra grammar/model query."""
    recorder = getattr(model, "grammar_trace_recorder", None)
    if recorder is None:
        return
    from slm_training.evals.measurement_identity import content_digest

    canvas, unknown = ids[0].tolist(), unknown_mask[0].tolist()
    position, token_id = exact_commit
    token = model.tokenizer.id_to_token[int(token_id)]
    recorder.record(
        state_fingerprint=content_digest(
            {"canvas": canvas, "unknown": unknown, "position": position}
        ),
        state_signature_version="diffusion_canvas/v1",
        legal_action_ids=[token],
        compiler_coverage="complete",
        selected_action_id=token,
        completion_support_size_exact=1,
        diffusion_timestep=step,
    )


def normalize_legal_probs(
    values: Sequence[float], *, convention: Literal["logit", "energy"] = "logit"
) -> tuple[float, ...]:
    """Stable softmax over the supplied legal actions; lower energies are better."""
    if convention not in {"logit", "energy"}:
        raise ValueError("convention must be logit or energy")
    if not values:
        return ()
    if any(not math.isfinite(value) for value in values):
        raise ValueError("legal scores must be finite")
    scores = [-value for value in values] if convention == "energy" else values
    maximum = max(scores)
    exps = [math.exp(value - maximum) for value in scores]
    total = sum(exps)
    return tuple(value / total for value in exps)


def legal_logits(tokenizer, legal_ids, logits):
    """Project already-computed logits, preserving exactly the supplied order."""
    if logits is None or len(legal_ids) <= 1:
        return None
    token_ids = {str(value): key for key, value in tokenizer.id_to_token.items()}
    return [float(logits[token_ids[action]].item()) for action in legal_ids]


def decision_diagnostics(row: dict) -> dict:
    legal = row.get("legal_action_ids") or []
    selected = row.get("selected_action_id")
    if len(set(legal)) != len(legal) or (
        selected is not None and selected not in legal
    ):
        raise ValueError("trace differs from its supplied legal domain")
    scores = row.get("logits_or_energies")
    if scores is not None and (
        len(scores) != len(legal) or any(not math.isfinite(value) for value in scores)
    ):
        raise ValueError("scores must be finite and aligned with legal actions")
    complete = row.get("compiler_coverage") == "complete"
    forced = complete and len(legal) == 1
    targets = set(row.get("target_action_ids") or ())
    reachable = bool(targets.intersection(legal)) if targets else None
    selected_rank = None
    if scores is not None and selected is not None and not forced:
        selected_score = scores[legal.index(selected)]
        selected_rank = 1 + sum(value > selected_score for value in scores)
    cause = "unlabelled"
    if targets:
        if not reachable:
            cause = "representation" if complete else "unknown_domain_coverage"
        elif selected not in targets:
            cause = "ranking"
        elif row.get("verification_outcome") == "semantic_failure":
            cause = "rollout"
        else:
            cause = "correct_local_choice"
    return {
        "forced": forced,
        "ambiguous": len(legal) > 1,
        "correct_action_reachable": reachable,
        "selected_action_rank": selected_rank,
        "failure_attribution": cause,
        "domain_complete": complete,
        "semantic_label_available": bool(targets),
    }


def compare_legal_controls(legal_ids, *, coverage, scorers, oracle_targets=()):
    """Caller-owned rankers share one legal domain; expert is diagnostic only.

    A proven singleton invokes none of the scorers, even for instrumentation.
    Context ablations must use the existing model path, never fabricated scores.
    """
    legal = tuple(legal_ids)
    if not legal or len(set(legal)) != len(legal):
        raise ValueError("nonempty unique supplied legal domain required")
    if coverage == "complete" and len(legal) == 1:
        return {"forced": True, "selected": legal[0], "scorer_calls": 0, "controls": {}}
    controls = {}
    for name, scorer in scorers.items():
        values = tuple(scorer(legal))
        if len(values) != len(legal) or any(not math.isfinite(v) for v in values):
            raise ValueError("control returned invalid legal-domain scores")
        controls[name] = legal[max(range(len(legal)), key=lambda i: values[i])]
    targets = set(oracle_targets)
    expert = next((action for action in legal if action in targets), None)
    return {
        "forced": False,
        "scorer_calls": len(controls),
        "controls": controls,
        "expert_upper_bound": expert,
        "expert_deployable": False,
        "information_sensitive": len(set(controls.values())) > 1,
    }


def choice_summary(records):
    rows = [decision_diagnostics(row) for row in records]
    return {
        "n": len(rows),
        "forced_n": sum(row["forced"] for row in rows),
        "ambiguous_n": sum(row["ambiguous"] for row in rows),
        "labelled_n": sum(row["semantic_label_available"] for row in rows),
        "attribution_counts": {
            cause: sum(row["failure_attribution"] == cause for row in rows)
            for cause in sorted({row["failure_attribution"] for row in rows})
        },
        "claim": "diagnostic_not_capability_certificate",
    }


def _certified_hole_domain(model, canvas, position, placeholders):
    """Exhaustive fixed-canvas certification, separate from all learned scoring."""
    from slm_training.dsl.grammar.fastpath.engine import engine_for_dsl
    from slm_training.dsl.lang_core import ParseError, validate
    from slm_training.dsl.language_contract import assert_symbol_only_output
    from slm_training.models.grammar import active_dsl

    tokenizer = model.tokenizer
    if [i for i, token in enumerate(canvas) if token == tokenizer.mask_id] != [
        position
    ]:
        raise ValueError("diagnostic requires exactly one declared hole")
    legal, programs = [], {}
    engine = engine_for_dsl(active_dsl())
    if engine is None:
        raise ValueError("configured grammar authority unavailable")
    for token_id in tokenizer.id_to_token:
        candidate = list(canvas)
        candidate[position] = token_id
        text = model._decode_openui(candidate, placeholders=placeholders)
        if not engine.set_prefix(text) or "$END" not in engine.next_terminals():
            continue
        try:
            validate(text)
            assert_symbol_only_output(text)
        except (ParseError, ValueError):
            continue
        legal.append(token_id)
        programs[token_id] = text
    if not legal:
        raise ValueError("certified single-hole domain is empty")
    return legal, programs


def rank_single_hole(
    model, prompt, canvas, position, *, context_controls=False, placeholders=None
):
    """Rank an exhaustive single-hole domain; singleton programs need no model.

    Certification uses both the configured grammar and official full-program
    validator plus symbol-only law. This is not a multi-hole completeness claim.
    The full token domain is retained, including certified equivalent aliases.
    """
    import torch

    from slm_training.dsl.lang_core import validate
    from slm_training.models.decode_stats import collect_decode_stats

    legal, programs = _certified_hole_domain(model, canvas, position, placeholders)
    controls = {}
    # Byte/native aliases can encode the same certified completed program.
    # Preserve the full token domain, but a singleton program needs no ranking.
    program_count = len(set(programs.values()))
    with collect_decode_stats() as stats:
        if program_count == 1:
            selected = legal[0]
            margin = None
        else:
            with torch.inference_mode():
                ctx, pad = model._encode_context([prompt])
                ids = torch.tensor([canvas], dtype=torch.long, device=model.device_name)
                logits = model._denoiser_forward(ids, ctx, pad)[0, position]
                scores = [float(logits[token].item()) for token in legal]
                if context_controls:
                    for name, changed, mask in (
                        ("ablated", torch.zeros_like(ctx), pad),
                        ("shuffled", ctx.flip(1), pad.flip(1)),
                    ):
                        alternate = model._denoiser_forward(ids, changed, mask)[
                            0, position
                        ]
                        values = [float(alternate[token].item()) for token in legal]
                        if any(not math.isfinite(value) for value in values):
                            raise ValueError("nonfinite context-control score")
                        controls[name] = legal[
                            max(range(len(legal)), key=lambda i: values[i])
                        ]
            if any(not math.isfinite(score) for score in scores):
                raise ValueError("nonfinite learned legal score")
            selected = legal[max(range(len(legal)), key=lambda i: scores[i])]
            ordered = sorted(scores, reverse=True)
            margin = ordered[0] - ordered[1]
    validate(programs[selected])
    return programs[selected], {
        "legal_token_ids": legal,
        "selected_token_id": selected,
        "forwards": stats.forwards_count,
        "margin": margin,
        "coverage": "complete",
        "distinct_certified_programs": program_count,
        "equivalence_certificate": "identical_decoded_program/v1",
        "context_controls": controls,
        "authority": "exhaustive_frozen_vocabulary_full_program_certification",
        "claim": "single_hole_diagnostic_not_production_decoder",
    }
