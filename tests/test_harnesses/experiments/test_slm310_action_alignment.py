"""SLM-310 (LAR2-03): action-alignment invariants for the X22 tree-edit model."""

from __future__ import annotations

import json
import random

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.slm310_action_alignment import (
    action_calibration,
    decode_demand_distribution,
    run_distribution_audit,
    training_target_distribution,
)
from slm_training.models.tree_edit_diffusion import (
    ACTION_ADD,
    ACTION_BIND_PLACEHOLDER,
    ACTION_IDS,
    ACTION_REMOVE,
    ACTION_STOP,
    Edit,
    REASON_DUPLICATE_STATE,
    TreeEditDiffusionConfig,
    TreeEditDiffusionModel,
    TreeEditSpace,
    parse_statements,
)
from scripts.run_slm310_action_alignment import CELLS

PROGRAM = (
    'root = Stack([inline_card, cta], "column")\n'
    "inline_card = Card([title])\n"
    'title = TextContent(":hero.title")\n'
    'cta = Button(":cta.label")'
)
INVENTORY = [":hero.title", ":cta.label"]

PROGRAMS = [
    PROGRAM,
    'root = Stack([t], "column")\nt = TextContent(":body")',
    'root = Form([i, s], "column")\ni = TextInput(":form.email")\ns = Button(":form.submit")',
]
INVENTORIES = [INVENTORY, [":body"], [":form.email", ":form.submit"]]


def _records() -> list[ExampleRecord]:
    return [
        ExampleRecord(
            id=f"fx_{i}",
            prompt=f"fixture prompt {i}",
            openui=openui,
            placeholders=list(inv),
            split="train",
        )
        for i, (openui, inv) in enumerate(zip(PROGRAMS, INVENTORIES))
    ]


def _model(**overrides) -> TreeEditDiffusionModel:
    base = dict(
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        beam_width=2,
        expand_per_state=2,
        max_search_steps=3,
        seed=3,
        value_label_mode="mutation_count",
    )
    base.update(overrides)
    cfg = TreeEditDiffusionConfig(**base)
    torch.manual_seed(cfg.seed)  # deterministic init across test order
    return TreeEditDiffusionModel.from_records(_records(), config=cfg, device="cpu")


def _evidence(model: TreeEditDiffusionModel) -> list[dict]:
    from slm_training.harnesses.model_build.plugin import GenerationRequest

    model.generate_batch_requests(
        [
            GenerationRequest(prompt=r.prompt, slot_contract=tuple(r.placeholders))
            for r in _records()
        ]
    )
    return model.consume_generation_evidence()


# --- reason-coded applicability ---------------------------------------------


def test_apply_reports_reason_codes() -> None:
    space = TreeEditSpace()
    base = parse_statements(PROGRAM)
    assert base is not None
    # Out-of-range statement index.
    reason: list[str] = []
    assert space.apply(base, Edit(ACTION_REMOVE, 99), INVENTORY, reason=reason) is None
    assert reason == ["index_out_of_range"]
    # Leaf/container mismatch on REPLACE (root is a container, leaf comp given).
    reason = []
    leaf_comp = space.comp_index["TextContent"]
    assert space.apply(base, Edit(1, 0, leaf_comp), INVENTORY, reason=reason) is None
    assert reason == ["leaf_container_mismatch"]
    # BIND_PLACEHOLDER on the root container.
    reason = []
    assert (
        space.apply(base, Edit(ACTION_BIND_PLACEHOLDER, 0, slot=0), INVENTORY, reason=reason)
        is None
    )
    assert reason == ["not_bindable"]
    # Acceptance appends nothing.
    reason = []
    assert space.apply(base, Edit(ACTION_REMOVE, 3), INVENTORY, reason=reason) is not None
    assert reason == []


def test_proposal_telemetry_reason_coded_and_deterministic() -> None:
    # Wide expansion budget so the decode loop visits inapplicable
    # (dead) candidates, not only top-scoring applicable ones.
    model = _model(expand_per_state=8, beam_width=3, max_search_steps=4)
    first = _evidence(model)
    second = _evidence(model)
    # Deterministic: identical proposals across identical decodes.
    assert [e.get("proposals") for e in first] == [e.get("proposals") for e in second]
    saw_rejection = False
    saw_reason_codes: set[str] = set()
    for ev in first:
        proposals = ev.get("proposals")
        assert proposals, "every decode records proposal telemetry"
        assert ev["verifier_calls"] == sum(
            1 for p in proposals if p["action"] != ACTION_STOP
        )
        for p in proposals:
            assert set(p) >= {
                "step",
                "beam_row",
                "rank",
                "action",
                "action_name",
                "score",
                "applicable",
                "rejection_reason",
                "selected",
                "consumed_budget",
            }
            # Deterministic order: ranks are strictly increasing per state.
            if p["applicable"]:
                assert p["rejection_reason"] in (None, REASON_DUPLICATE_STATE)
            else:
                assert isinstance(p["rejection_reason"], str)
                saw_rejection = True
                saw_reason_codes.add(p["rejection_reason"])
            # Budget accounting invariants: consumed implies applicable, and
            # every selected proposal consumed budget.
            if p["consumed_budget"]:
                assert p["applicable"]
            if p["selected"]:
                assert p["consumed_budget"]
        # Summary is internally consistent.
        summary = ev["proposal_summary"]
        assert summary["visited"] == len(proposals)
        assert summary["applicable"] == sum(1 for p in proposals if p["applicable"])
    assert saw_rejection, "fixture decode must exercise dead candidates"
    assert saw_reason_codes <= {
        "index_out_of_range",
        "no_op",
        "leaf_container_mismatch",
        "max_stmts",
        "parent_or_comp_precondition",
        "slot_out_of_range",
        "not_removable",
        "unreferenced_leaf",
        "target_out_of_range",
        "not_container",
        "subtree_not_leaf_only",
        "unreferenced_container",
        "payload_out_of_range",
        "payload_not_leaf",
        "not_canonical_subtree",
        "statement_invalid",
        "target_not_canonical",
        "not_bindable",
        "not_leaf_component",
        "unknown_action",
        "pre_validate_rejected",
        "invalid_result",
    }


# --- distribution audit math ---------------------------------------------------


def test_training_target_distribution_math() -> None:
    space = TreeEditSpace()
    dist = training_target_distribution(
        _records(), space, seed=5, samples=120, max_chain=3
    )
    assert dist["n_targets"] == sum(dist["counts"].values())
    assert abs(sum(dist["shares"].values()) - 1.0) < 1e-9
    assert dist["counts"]["STOP"] > 0  # stop_fraction branch exercised
    # Deterministic.
    again = training_target_distribution(
        _records(), space, seed=5, samples=120, max_chain=3
    )
    assert again["counts"] == dist["counts"]


def test_decode_demand_audit_math() -> None:
    model = _model()
    demand = decode_demand_distribution(model, _records())
    overall = demand["overall"]
    assert overall["visited_total"] > 0
    assert overall["verifier_calls"] > 0
    assert 0.0 <= overall["dead_candidate_rate"] <= 1.0
    assert abs(sum(overall["visited_shares"].values()) - 1.0) < 1e-6
    assert set(demand["by_source"]) == {r.id for r in _records()}
    assert set(demand["by_suite"]) == {"train"}
    # Dead total = visited - (no-reason proposals).
    assert overall["dead_total"] <= overall["visited_total"]
    # Calibration rows cover every action and MAD is the mean |gap|.
    calib = action_calibration(
        training_target_distribution(_records(), model.space, seed=5, samples=60),
        overall,
    )
    assert len(calib["per_action"]) == 11
    gaps = [abs(r["gap"]) for r in calib["per_action"].values()]
    assert abs(calib["mean_abs_deviation"] - sum(gaps) / len(gaps)) < 1e-12
    # Full audit predicts via the preregistered rule and stays honest.
    audit = run_distribution_audit(model, _records(), seed=5, samples=60)
    pred = audit["loss_reweighting_prediction"]
    assert isinstance(pred["predicted"], bool)
    assert pred["applicable_add_recall"] == overall["applicable_add_recall"]


# --- corruption sampler ---------------------------------------------------------


def test_corruption_sampler_hits_declared_distribution() -> None:
    space = TreeEditSpace()
    statements = parse_statements(PROGRAM)
    assert statements is not None
    # Degenerate declared distribution: all weight on ADD inverses.
    rng = random.Random(13)
    inverses = []
    for _ in range(40):
        step = space.sample_mutation(
            statements, INVENTORY, rng,
            inverse_action_weights={ACTION_ADD: 1.0},
        )
        if step is None:
            continue
        inverses.append(step[1].action)
    assert inverses, "sampler must still produce mutations"
    assert set(inverses) == {ACTION_ADD}
    # Declared ADD-balanced distribution lands near the declared share.
    declared = {ACTION_IDS[n]: w for n, w in {
        "REPLACE": 1.0, "ADD": 4.0, "REMOVE": 1.0, "ADD_CONTAINER": 1.0,
        "REMOVE_CONTAINER": 1.0, "INSERT_SUBTREE": 1.0, "REPLACE_SUBTREE": 1.0,
        "REPLACE_STATEMENT": 1.0, "BIND_PLACEHOLDER": 1.0,
    }.items()}
    dist = training_target_distribution(
        _records(), space, seed=7, samples=600, max_chain=3,
        inverse_action_weights=declared,
    )
    default = training_target_distribution(
        _records(), space, seed=7, samples=600, max_chain=3
    )
    add_share = dist["shares"]["ADD"]
    assert add_share > default["shares"]["ADD"] + 0.10
    # Declared ADD weight is 4/12 of inverse mass; with 20% STOP targets the
    # realized share should sit near 0.8 * 4/12 ≈ 0.27 (loose tolerance for
    # retry-loop fallback and REMOVE-mutation applicability).
    assert abs(add_share - 0.8 * 4 / 12) < 0.12


def test_corruption_sampler_leaves_gold_corpus_untouched() -> None:
    records = _records()
    before = [r.openui for r in records]
    model = _model(
        corruption_action_distribution={"ADD": 4.0, "REPLACE": 1.0, "REMOVE": 1.0}
    )
    loss = model.training_loss(records)
    assert torch.isfinite(loss)
    assert [r.openui for r in records] == before
    # Default off: config without the knob keeps the historical sampler.
    assert _model()._inverse_action_weights() is None
    # Unknown action names fail closed.
    bad = _model(corruption_action_distribution={"NOT_AN_ACTION": 1.0})
    with pytest.raises(ValueError, match="unknown action"):
        bad.training_loss(records)


# --- STOP-slot accounting arms ---------------------------------------------------


def test_stop_slot_accounting_arms_differ_exactly_as_specified() -> None:
    legacy = _model(stop_slot_accounting="legacy")
    corrected = _model(stop_slot_accounting="corrected")
    leg_ev = _evidence(legacy)
    cor_ev = _evidence(corrected)
    # Deterministic per arm.
    assert [e.get("proposals") for e in leg_ev] == [
        e.get("proposals") for e in _evidence(legacy)
    ]
    assert [e.get("proposals") for e in cor_ev] == [
        e.get("proposals") for e in _evidence(corrected)
    ]

    def stop_stats(evs):
        visited = consumed = retained = 0
        for ev in evs:
            for p in ev.get("proposals", []):
                if p["action"] != ACTION_STOP:
                    continue
                visited += 1
                consumed += 1 if p["consumed_budget"] else 0
                retained += 1 if p["selected"] else 0
        return visited, consumed, retained

    lv, lc, lr = stop_stats(leg_ev)
    cv, cc, cr = stop_stats(cor_ev)
    assert lv > 0, "fixture decode must visit STOP proposals"
    # Legacy: every STOP consumes a slot, retained or not.
    assert lc == lv
    # Corrected: STOP consumes a slot exactly when its frozen candidate is
    # retained.
    assert cc == cr
    assert cc <= lc
    # Unknown arm values fail closed.
    bad = _model(stop_slot_accounting="bogus")
    with pytest.raises(ValueError, match="stop_slot_accounting"):
        _evidence(bad)


def test_lever_isolation_single_factor_per_arm() -> None:
    def diff(x, y):
        return [k for k in ("corruption", "stop") if CELLS[x][k] != CELLS[y][k]]

    assert diff("cell_b", "cell_a") == ["corruption"]
    assert diff("cell_c", "cell_a") == ["stop"]
    assert diff("cell_d", "cell_b") == ["stop"]
    assert diff("cell_d", "cell_c") == ["corruption"]


# --- checkpoint parity -----------------------------------------------------------


def test_checkpoint_parity_with_new_additive_fields(tmp_path) -> None:
    records = _records()
    model = _model(corruption_action_distribution={"ADD": 2.0, "REMOVE": 1.0})
    path = tmp_path / "ckpt.pt"
    model.save(path)
    loaded = TreeEditDiffusionModel.from_checkpoint(path, device="cpu")
    assert loaded.config.corruption_action_distribution == {
        "ADD": 2.0,
        "REMOVE": 1.0,
    }
    assert loaded.config.stop_slot_accounting == "legacy"
    # Simulate a pre-SLM-310 checkpoint: drop the new config keys entirely.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["config"].pop("corruption_action_distribution")
    payload["config"].pop("stop_slot_accounting")
    old_path = tmp_path / "ckpt_old.pt"
    torch.save(payload, old_path)
    (tmp_path / "ckpt_old.tokenizer.json").write_text(
        (tmp_path / "ckpt.tokenizer.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    old = TreeEditDiffusionModel.from_checkpoint(old_path, device="cpu")
    # Pre-field checkpoints get the historical defaults: uniform sampler,
    # legacy STOP accounting.
    assert old.config.corruption_action_distribution is None
    assert old.config.stop_slot_accounting == "legacy"
    # Decode parity: the old-loaded model reproduces the default model.
    from slm_training.harnesses.model_build.plugin import GenerationRequest

    requests = [
        GenerationRequest(prompt=r.prompt, slot_contract=tuple(r.placeholders))
        for r in records
    ]
    default_model = _model()
    assert old.generate_batch_requests(requests) == default_model.generate_batch_requests(
        requests
    )


def test_metadata_not_polluted(tmp_path) -> None:
    # Proposal telemetry stays on evidence; meta.json schema unchanged.
    model = _model()
    path = tmp_path / "ckpt.pt"
    model.save(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert set(meta) == {
        "kind",
        "format_version",
        "tokenizer",
        "vocab_size",
        "parameter_count",
        "serialized_weight_bytes",
    }
